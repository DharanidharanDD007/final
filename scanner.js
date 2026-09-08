/**
 * Real-Time Deepfake Detection - Sandboxed Multi-Participant Inference Engine (scanner.js)
 * =========================================================================================
 * Runs inside the sandboxed iframe (scanner.html).
 * 
 * Key Responsibilities:
 * 1. Initializes MediaPipe FaceMesh and TensorFlow.js CNN Model with WebGL backend.
 * 2. Receives raw video frames and audio spectral metrics per participantId.
 * 3. Extracts face bounding box dynamically from MediaPipe landmarks with a 15% margin.
 * 4. Crops and renders the face ROI to an offscreen 64x64 canvas.
 * 5. Preprocesses to [1, 64, 64, 1] grayscale normalized tensor (div 255.0).
 * 6. Executes CNN inference with strict WebGL memory leak prevention (tf.tidy + dispose).
 * 7. Maintains independent 15-frame rolling average buffers keyed by participantId.
 * 8. Fuses visual trust and Web Audio spectral consistency into a unified Multimodal Trust Score:
 *       Final Trust Score = (w_v * Visual Trust) + (w_a * Audio Consistency Score)
 *       (with silence fallback to pure visual when speech is inactive).
 */

const canvas = document.getElementById('offscreenCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// Auxiliary 64x64 offscreen canvas for GPU-accelerated face cropping and scaling
const faceCanvas = document.createElement('canvas');
faceCanvas.width = 64;
faceCanvas.height = 64;
const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });

let faceMesh;
let cnnModel;
let isModelReady = false;

// ---------------------------------------------------------------------------
// MULTI-PARTICIPANT STATE & TEMPORAL BUFFER CONFIGURATION
// ---------------------------------------------------------------------------
const BUFFER_SIZE = 15;
const participantBuffers = new Map();

// Context variables for the frame currently undergoing inference
let currentParticipantId = 'default';
let currentDisplayName = 'Remote Caller';
let currentAudioMetrics = { isSilent: true, audioScore: 100 };

// Auto-prune inactive participants after 15 seconds of inactivity
setInterval(() => {
    const now = Date.now();
    for (const [id, state] of participantBuffers.entries()) {
        if (now - state.lastSeen > 15000) {
            participantBuffers.delete(id);
        }
    }
}, 10000);

/**
 * Posts diagnostic logs back to the parent window for devtools inspection.
 */
function logToMain(msg, isError = false) {
    window.parent.postMessage({ type: 'DEBUG_LOG', message: msg, isError: isError }, '*');
}

/**
 * Loads the TensorFlow.js CNN Model exported from train_cnn.py.
 */
async function loadModel() {
    try {
        logToMain("Loading CNN Model from './models/model.json'...");
        cnnModel = await tf.loadLayersModel('./models/model.json');
        
        // Warm up the WebGL engine with a dummy tensor [1, 64, 64, 1]
        tf.tidy(() => {
            const dummy = tf.zeros([1, 64, 64, 1]);
            cnnModel.predict(dummy);
        });
        logToMain("CNN Model Loaded & WebGL backend warmed up successfully!");
    } catch (err) {
        logToMain("CRITICAL ERROR loading CNN Model: " + err.message, true);
        throw err;
    }
}

/**
 * Initializes MediaPipe FaceMesh.
 */
async function initFaceMesh() {
    try {
        await loadModel();

        logToMain("Initializing MediaPipe FaceMesh engine...");
        faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceMeshResults);

        logToMain("Warming up FaceMesh pipeline...");
        await faceMesh.initialize();

        isModelReady = true;
        logToMain("FaceMesh & CNN Multimodal Pipeline Ready!");
        window.parent.postMessage({ type: 'SCANNER_READY' }, '*');
    } catch (err) {
        logToMain("ERROR initializing FaceMesh: " + err.message, true);
    }
}

/**
 * Computes bounding box from normalized landmarks and adds a 15% boundary margin.
 * Captures edge blending artifacts where face swaps are composited.
 */
function computeFaceBoundingBox(landmarks, imgWidth, imgHeight) {
    let minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0;

    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        if (pt.x < minX) minX = pt.x;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.y > maxY) maxY = pt.y;
    }

    const rawWidth = (maxX - minX) * imgWidth;
    const rawHeight = (maxY - minY) * imgHeight;

    // 15% safety margin around facial perimeter
    const marginRatio = 0.15;
    const padX = rawWidth * marginRatio;
    const padY = rawHeight * marginRatio;

    const cropX = Math.max(0, minX * imgWidth - padX);
    const cropY = Math.max(0, minY * imgHeight - padY);
    const cropW = Math.min(imgWidth - cropX, rawWidth + 2 * padX);
    const cropH = Math.min(imgHeight - cropY, rawHeight + 2 * padY);

    return { cropX, cropY, cropW, cropH };
}

/**
 * Callback invoked every time MediaPipe finishes processing a video frame.
 */
function onFaceMeshResults(results) {
    let state = participantBuffers.get(currentParticipantId);
    if (!state) {
        state = {
            visualQueue: [],
            audioQueue: [],
            combinedQueue: [],
            consecutiveNoFace: 0,
            lastSeen: Date.now(),
            displayName: currentDisplayName
        };
        participantBuffers.set(currentParticipantId, state);
    }
    state.lastSeen = Date.now();
    state.displayName = currentDisplayName;

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        state.consecutiveNoFace = 0;

        try {
            const landmarks = results.multiFaceLandmarks[0];
            const { cropX, cropY, cropW, cropH } = computeFaceBoundingBox(landmarks, canvas.width, canvas.height);

            if (cropW < 10 || cropH < 10) {
                return;
            }

            // 1. Crop face ROI from main canvas and scale into 64x64 auxiliary canvas
            faceCtx.clearRect(0, 0, 64, 64);
            faceCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, 64, 64);

            // 2. Build normalized 1-channel grayscale tensor [1, 64, 64, 1] inside tf.tidy
            const tensorInput = tf.tidy(() => {
                return tf.browser.fromPixels(faceCanvas, 1)
                    .toFloat()
                    .div(255.0)
                    .expandDims(0); // Shape: [1, 64, 64, 1]
            });

            // 3. Run CNN Inference
            const prediction = cnnModel.predict(tensorInput);
            const fakeProb = prediction.dataSync()[0]; // Sigmoid output: 0.0 = Real, 1.0 = Fake

            // Dispose explicit tensors outside tf.tidy to prevent WebGL GPU memory leaks
            tensorInput.dispose();
            prediction.dispose();

            // 4. Calculate Frame Visual Trust Score (100% = Authentic, 0% = Manipulated)
            const frameVisualTrust = Math.max(0, Math.min(100, (1.0 - fakeProb) * 100));
            state.visualQueue.push(frameVisualTrust);
            if (state.visualQueue.length > BUFFER_SIZE) {
                state.visualQueue.shift();
            }

            // 5. Audio Consistency Score from payload
            const audioConsistency = currentAudioMetrics.audioScore !== undefined ? currentAudioMetrics.audioScore : 100;
            state.audioQueue.push(audioConsistency);
            if (state.audioQueue.length > BUFFER_SIZE) {
                state.audioQueue.shift();
            }

            // 6. Multimodal Fusion:
            // Final Trust = (w_v * Visual) + (w_a * Audio)
            // If caller is muted/silent, adapt weights dynamically to pure visual (w_v=1.0, w_a=0.0)
            let frameCombinedTrust;
            if (currentAudioMetrics.isSilent) {
                frameCombinedTrust = frameVisualTrust;
            } else {
                const w_v = 0.75;
                const w_a = 0.25;
                frameCombinedTrust = (w_v * frameVisualTrust) + (w_a * audioConsistency);
            }

            state.combinedQueue.push(frameCombinedTrust);
            if (state.combinedQueue.length > BUFFER_SIZE) {
                state.combinedQueue.shift();
            }

            // 7. Compute temporal moving averages to eliminate flickering
            const avgCombined = Math.round(state.combinedQueue.reduce((a, b) => a + b, 0) / state.combinedQueue.length);
            const avgVisual = Math.round(state.visualQueue.reduce((a, b) => a + b, 0) / state.visualQueue.length);
            const avgAudio = Math.round(state.audioQueue.reduce((a, b) => a + b, 0) / state.audioQueue.length);

            // 8. Transmit multimodal metrics to content script UI
            window.parent.postMessage({
                type: 'SCORE_UPDATE',
                participantId: currentParticipantId,
                displayName: currentDisplayName,
                score: avgCombined,
                visualScore: avgVisual,
                audioScore: avgAudio,
                isSilent: currentAudioMetrics.isSilent,
                hasFace: true
            }, '*');

        } catch (err) {
            logToMain("Inference Error in scanner.js: " + err.message, true);
        }
    } else {
        // No face detected in this frame
        state.consecutiveNoFace++;
        
        if (state.consecutiveNoFace > BUFFER_SIZE) {
            state.visualQueue.length = 0;
            state.audioQueue.length = 0;
            state.combinedQueue.length = 0;
            window.parent.postMessage({
                type: 'SCORE_UPDATE',
                participantId: currentParticipantId,
                displayName: currentDisplayName,
                score: 100,
                visualScore: 100,
                audioScore: 100,
                isSilent: true,
                hasFace: false
            }, '*');
        }
    }
}

/**
 * Handle incoming frames transferred from content.js.
 */
window.addEventListener('message', async (e) => {
    if (e.data.type === 'PROCESS_FRAME' && isModelReady) {
        const { bitmap, participantId, displayName, audioMetrics } = e.data;
        currentParticipantId = participantId || 'default';
        currentDisplayName = displayName || 'Remote Caller';
        currentAudioMetrics = audioMetrics || { isSilent: true, audioScore: 100 };

        try {
            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            await faceMesh.send({ image: canvas });
        } catch (err) {
            logToMain("Frame processing error: " + err.message, true);
        } finally {
            bitmap.close();
            // ALWAYS post FRAME_PROCESSED so content.js isProcessing lock is released
            window.parent.postMessage({
                type: 'FRAME_PROCESSED',
                participantId: currentParticipantId
            }, '*');
        }
    }
});

/**
 * Startup hook.
 */
window.addEventListener('load', () => {
    if (typeof tf === 'undefined') {
        logToMain("CRITICAL: TensorFlow (tf) is missing! Ensure tf.min.js is present.", true);
        return;
    }
    initFaceMesh();
});
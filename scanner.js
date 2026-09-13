/**
 * Real-Time Deepfake Detection - Sandboxed Multi-Participant Inference Engine (scanner.js)
 * =========================================================================================
 * Runs inside the sandboxed iframe (scanner.html).
 * 
 * Key Responsibilities:
 * 1. Synchronous Multi-Participant FIFO Queue (processQueue()).
 * 2. MediaPipe FaceMesh & TensorFlow.js WebGL model execution.
 * 3. 15% Boundary margin face ROI extraction to capture boundary blending artifacts.
 * 4. Normalizes face ROI to [1, 64, 64, 1] grayscale tensor.
 * 5. Memory leak prevention via tf.tidy() and .dispose().
 * 6. Independent 15-frame temporal buffers using participantStates = new Map().
 * 7. Multimodal Fusion:
 *       Final Trust Score = (w_v * Visual Trust) + (w_a * Audio Consistency Score)
 *       (with silence fallback to pure visual when speech is inactive).
 */

const canvas = document.getElementById('offscreenCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// Auxiliary 64x64 offscreen canvas for face ROI cropping and scaling
const faceCanvas = document.createElement('canvas');
faceCanvas.width = 64;
faceCanvas.height = 64;
const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true });

let faceMesh;
let cnnModel;
let isModelReady = false;

// ---------------------------------------------------------------------------
// 1. SYNCHRONOUS MULTI-PARTICIPANT QUEUE & INDEPENDENT TEMPORAL BUFFERS
// ---------------------------------------------------------------------------
const BUFFER_SIZE = 15;
const participantStates = new Map(); // id -> { visualBuffer: [], audioBuffer: [], fusedBuffer: [], consecutiveNoFace: 0, lastSeen, displayName }

const frameQueue = [];
let isProcessingQueue = false;

let currentParticipantId = 'default';
let currentDisplayName = 'Remote Caller';
let currentAudioMetrics = { isSilent: true, audioScore: 100 };

// Auto-prune inactive participants after 15 seconds
setInterval(() => {
    const now = Date.now();
    for (const [id, state] of participantStates.entries()) {
        if (now - state.lastSeen > 15000) {
            participantStates.delete(id);
        }
    }
}, 8000);

/**
 * Diagnostic logger to parent window
 */
function logToMain(msg, isError = false) {
    window.parent.postMessage({ type: 'DEBUG_LOG', message: msg, isError: isError }, '*');
}

/**
 * Loads the TensorFlow.js CNN Model
 */
async function loadModel() {
    try {
        logToMain("Loading CNN Model from './models/model.json'...");
        cnnModel = await tf.loadLayersModel('./models/model.json');

        // Warm up WebGL backend
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
 * Initializes MediaPipe FaceMesh engine
 */
async function initFaceMesh() {
    try {
        await loadModel();

        logToMain("Initializing MediaPipe FaceMesh engine...");
        faceMesh = new FaceMesh({
            locateFile: (file) => {
                // Try local mediapipe/ directory first, fallback to CDN if not local
                return `./mediapipe/${file}`;
            }
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
        logToMain("ERROR initializing FaceMesh with local files, retrying with CDN: " + err.message, false);
        try {
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
            await faceMesh.initialize();
            isModelReady = true;
            logToMain("FaceMesh & CNN Multimodal Pipeline Ready (via CDN)!");
            window.parent.postMessage({ type: 'SCANNER_READY' }, '*');
        } catch (cdnErr) {
            logToMain("CRITICAL ERROR initializing FaceMesh: " + cdnErr.message, true);
        }
    }
}

/**
 * Computes bounding box from normalized landmarks and adds 15% safety boundary margin
 * Enforces explicit minimum dimension gate (cropW >= 60 && cropH >= 60)
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

    // Explicit minimum dimension quality gate
    const isValidQuality = cropW >= 60 && cropH >= 60;

    return { cropX, cropY, cropW, cropH, isValidQuality };
}

/**
 * Callback invoked every time MediaPipe finishes processing a video frame
 */
function onFaceMeshResults(results) {
    let state = participantStates.get(currentParticipantId);
    if (!state) {
        state = {
            visualBuffer: [],
            audioBuffer: [],
            fusedBuffer: [],
            consecutiveNoFace: 0,
            lastSeen: Date.now(),
            displayName: currentDisplayName
        };
        participantStates.set(currentParticipantId, state);
    }
    state.lastSeen = Date.now();
    state.displayName = currentDisplayName;

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        try {
            const landmarks = results.multiFaceLandmarks[0];
            const { cropX, cropY, cropW, cropH, isValidQuality } = computeFaceBoundingBox(landmarks, canvas.width, canvas.height);

            // Step 2: Strict Face Quality Gates (cropW >= 60 && cropH >= 60)
            if (!isValidQuality || cropW < 60 || cropH < 60) {
                state.consecutiveNoFace++;
                state.visualBuffer.push(-1);
                state.audioBuffer.push(-1);
                state.fusedBuffer.push(-1);
                if (state.visualBuffer.length > BUFFER_SIZE) state.visualBuffer.shift();
                if (state.audioBuffer.length > BUFFER_SIZE) state.audioBuffer.shift();
                if (state.fusedBuffer.length > BUFFER_SIZE) state.fusedBuffer.shift();

                // If no valid face for > 5 cycles, reset buffer to neutral
                if (state.consecutiveNoFace > 5) {
                    state.visualBuffer.length = 0;
                    state.audioBuffer.length = 0;
                    state.fusedBuffer.length = 0;
                }

                window.parent.postMessage({
                    type: 'SCORE_UPDATE',
                    participantId: currentParticipantId,
                    id: currentParticipantId,
                    displayName: currentDisplayName,
                    score: -1,
                    visualScore: -1,
                    audioScore: -1,
                    isSilent: true,
                    hasFace: false
                }, '*');
                return;
            }

            // Face passed quality gate
            state.consecutiveNoFace = 0;

            // 1. Crop face ROI and scale into 64x64 auxiliary canvas
            faceCtx.clearRect(0, 0, 64, 64);
            faceCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, 64, 64);

            // 2. Build normalized 1-channel grayscale tensor [1, 64, 64, 1] inside tf.tidy
            let fakeProb = 0.5;
            let tensorInput = null;
            let prediction = null;
            try {
                tensorInput = tf.tidy(() => {
                    return tf.browser.fromPixels(faceCanvas, 1)
                        .toFloat()
                        .div(255.0)
                        .expandDims(0);
                });

                // 3. CNN WebGL Inference
                prediction = cnnModel.predict(tensorInput);
                fakeProb = prediction.dataSync()[0]; // 0.0 = Real, 1.0 = Fake
            } finally {
                // Ensure tensorInput and prediction are properly cleaned up without WebGL texture memory leaks
                if (tensorInput) tensorInput.dispose();
                if (prediction) prediction.dispose();
            }

            // Step 3: Calibrated Sigmoid Curve (ensures raw probabilities below 0.55 map smoothly to 80%–98% Authentic)
            const calibrated = 1 / (1 + Math.exp(-10 * (fakeProb - 0.68)));
            const frameVisualTrust = Math.max(0, Math.min(100, Math.round((1.0 - calibrated) * 100)));

            // Only push valid score to state.visualBuffer
            state.visualBuffer.push(frameVisualTrust);
            if (state.visualBuffer.length > BUFFER_SIZE) {
                state.visualBuffer.shift();
            }

            // Audio Consistency Score (Dynamic Silence Fallback preserved)
            const audioConsistency = currentAudioMetrics.audioScore !== undefined ? currentAudioMetrics.audioScore : 100;
            state.audioBuffer.push(audioConsistency);
            if (state.audioBuffer.length > BUFFER_SIZE) {
                state.audioBuffer.shift();
            }

            // Multimodal Fusion Equation:
            // Silence Fallback: When audio is inactive/muted (audioLevel < 2), w_v = 1.0, w_a = 0.0
            let frameFusedTrust;
            if (currentAudioMetrics.isSilent) {
                frameFusedTrust = frameVisualTrust;
            } else {
                const w_v = 0.75;
                const w_a = 0.25;
                frameFusedTrust = Math.round((w_v * frameVisualTrust) + (w_a * audioConsistency));
            }

            state.fusedBuffer.push(frameFusedTrust);
            if (state.fusedBuffer.length > BUFFER_SIZE) {
                state.fusedBuffer.shift();
            }

            // Frame Threshold: Only send valid trust score if buffer has at least 5 valid frames
            const validScores = state.fusedBuffer.filter(v => v >= 0);
            if (validScores.length < 5) {
                window.parent.postMessage({
                    type: 'SCORE_UPDATE',
                    participantId: currentParticipantId,
                    id: currentParticipantId,
                    displayName: currentDisplayName,
                    score: -1,
                    visualScore: -1,
                    audioScore: -1,
                    isSilent: true,
                    hasFace: true
                }, '*');
                return;
            }

            // 7. Compute temporal moving averages across valid buffer frames
            const avgFused = Math.round(validScores.reduce((a, b) => a + b, 0) / validScores.length);
            const validVisual = state.visualBuffer.filter(v => v >= 0);
            const avgVisual = validVisual.length > 0 ? Math.round(validVisual.reduce((a, b) => a + b, 0) / validVisual.length) : avgFused;
            const validAudio = state.audioBuffer.filter(v => v >= 0);
            const avgAudio = validAudio.length > 0 ? Math.round(validAudio.reduce((a, b) => a + b, 0) / validAudio.length) : 100;

            // 8. Transmit fused result to content script
            window.parent.postMessage({
                type: 'SCORE_UPDATE',
                participantId: currentParticipantId,
                id: currentParticipantId,
                displayName: currentDisplayName,
                score: avgFused,
                visualScore: avgVisual,
                audioScore: avgAudio,
                isSilent: currentAudioMetrics.isSilent,
                hasFace: true
            }, '*');

        } catch (err) {
            logToMain("Inference Error in scanner.js: " + err.message, true);
        }
    } else {
        // No face detected in frame
        state.consecutiveNoFace++;
        state.visualBuffer.push(-1);
        state.audioBuffer.push(-1);
        state.fusedBuffer.push(-1);
        if (state.visualBuffer.length > BUFFER_SIZE) state.visualBuffer.shift();
        if (state.audioBuffer.length > BUFFER_SIZE) state.audioBuffer.shift();
        if (state.fusedBuffer.length > BUFFER_SIZE) state.fusedBuffer.shift();

        // Insufficient Data: Resets to -1 if no face is detected for > 5 consecutive cycles
        if (state.consecutiveNoFace > 5) {
            state.visualBuffer.length = 0;
            state.audioBuffer.length = 0;
            state.fusedBuffer.length = 0;
            window.parent.postMessage({
                type: 'SCORE_UPDATE',
                participantId: currentParticipantId,
                id: currentParticipantId,
                displayName: currentDisplayName,
                score: -1,
                visualScore: -1,
                audioScore: -1,
                isSilent: true,
                hasFace: false
            }, '*');
        }
    }
}

/**
 * Multi-Participant Synchronous Queue Processor (processQueue())
 */
async function processQueue() {
    if (isProcessingQueue || frameQueue.length === 0 || !isModelReady) {
        return;
    }

    isProcessingQueue = true;
    const task = frameQueue.shift();
    const { bitmap, participantId, displayName, audioMetrics, audioLevel } = task;

    currentParticipantId = participantId || 'default';
    currentDisplayName = displayName || 'Remote Caller';
    currentAudioMetrics = audioMetrics || { isSilent: true, audioScore: 100 };

    // Silence Fallback: When audio is inactive/muted (audioLevel < 2), w_v = 1.0 and w_a = 0.0
    const effectiveAudioLevel = audioLevel !== undefined ? audioLevel : (audioMetrics && audioMetrics.audioLevel !== undefined ? audioMetrics.audioLevel : 0);
    if (effectiveAudioLevel < 2 || (audioMetrics && audioMetrics.isSilent)) {
        currentAudioMetrics.isSilent = true;
    }

    try {
        ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        await faceMesh.send({ image: canvas });
    } catch (err) {
        logToMain("Frame processing error: " + err.message, true);
    } finally {
        if (bitmap) {
            try { bitmap.close(); } catch (e) {} // Mandatory cleanup
        }
        window.parent.postMessage({
            type: 'FRAME_PROCESSED',
            participantId: currentParticipantId
        }, '*');

        isProcessingQueue = false;

        // Process next item in queue synchronously
        if (frameQueue.length > 0) {
            processQueue();
        }
    }
}

/**
 * Handle incoming frames transferred from content.js
 */
window.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'PROCESS_FRAME') {
        if (!isModelReady) {
            if (e.data.bitmap) {
                try { e.data.bitmap.close(); } catch (err) {}
            }
            window.parent.postMessage({
                type: 'FRAME_PROCESSED',
                participantId: e.data.participantId
            }, '*');
            return;
        }

        // Prevent queue backup and ImageBitmap memory leak
        if (frameQueue.length >= 3) {
            const stale = frameQueue.shift();
            if (stale && stale.bitmap) {
                try { stale.bitmap.close(); } catch (err) {}
            }
            window.parent.postMessage({
                type: 'FRAME_PROCESSED',
                participantId: stale ? stale.participantId : 'dropped'
            }, '*');
        }

        frameQueue.push(e.data);
        processQueue();
    }
});

/**
 * Startup hook
 */
window.addEventListener('load', () => {
    if (typeof tf === 'undefined') {
        logToMain("CRITICAL: TensorFlow (tf) is missing! Ensure tf.min.js is loaded.", true);
        return;
    }
    initFaceMesh();
});
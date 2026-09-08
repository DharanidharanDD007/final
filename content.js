// ONLY run in the main window, ignore hidden video conferencing sub-iframes
if (window.self === window.top) {
    let isProcessing = false;
    let scannerIframe;
    let scannerReady = false;
    let roundRobinIndex = 0;

    // Multi-participant tracking states
    const participants = new Map(); // id -> { displayName, score, visualScore, audioScore, isSilent, hasFace, lastSeen }

    // ---------------------------------------------------------------------------
    // 1. WEB AUDIO API SUBSYSTEM FOR ACOUSTIC ANALYSIS
    // ---------------------------------------------------------------------------
    let audioCtx = null;
    const videoAudioMap = new WeakMap();

    function getAudioContext() {
        if (!audioCtx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                audioCtx = new AudioCtx();
            }
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume().catch(() => {});
        }
        return audioCtx;
    }

    function getAudioMetricsForElement(videoEl) {
        try {
            const ctx = getAudioContext();
            if (!ctx) return { isSilent: true, audioScore: 100 };

            let nodeObj = videoAudioMap.get(videoEl);

            // Attempt to bind to video element's MediaStream if available
            if (!nodeObj && videoEl.srcObject && typeof videoEl.srcObject.getAudioTracks === 'function') {
                const tracks = videoEl.srcObject.getAudioTracks();
                if (tracks.length > 0) {
                    const source = ctx.createMediaStreamSource(videoEl.srcObject);
                    const analyser = ctx.createAnalyser();
                    analyser.fftSize = 512;
                    analyser.smoothingTimeConstant = 0.8;
                    source.connect(analyser); // Connect only to analyser, leaving native call audio untouched
                    nodeObj = {
                        analyser,
                        freqData: new Uint8Array(analyser.frequencyBinCount),
                        timeData: new Uint8Array(analyser.fftSize)
                    };
                    videoAudioMap.set(videoEl, nodeObj);
                }
            }

            // Fallback: check standalone <audio> tags in meeting DOM (common in Google Meet & Teams)
            if (!nodeObj) {
                const audios = document.querySelectorAll('audio');
                for (const aud of audios) {
                    if (aud.srcObject && typeof aud.srcObject.getAudioTracks === 'function' && !aud.paused) {
                        if (!videoAudioMap.has(aud)) {
                            const source = ctx.createMediaStreamSource(aud.srcObject);
                            const analyser = ctx.createAnalyser();
                            analyser.fftSize = 512;
                            analyser.smoothingTimeConstant = 0.8;
                            source.connect(analyser);
                            nodeObj = {
                                analyser,
                                freqData: new Uint8Array(analyser.frequencyBinCount),
                                timeData: new Uint8Array(analyser.fftSize)
                            };
                            videoAudioMap.set(aud, nodeObj);
                        } else {
                            nodeObj = videoAudioMap.get(aud);
                        }
                        break;
                    }
                }
            }

            if (!nodeObj) {
                return { isSilent: true, audioScore: 100, rms: 0, spectralCentroid: 0 };
            }

            nodeObj.analyser.getByteTimeDomainData(nodeObj.timeData);
            nodeObj.analyser.getByteFrequencyData(nodeObj.freqData);

            // 1. Compute RMS energy to detect speech presence
            let sumSq = 0;
            for (let i = 0; i < nodeObj.timeData.length; i++) {
                const norm = (nodeObj.timeData[i] - 128) / 128.0;
                sumSq += norm * norm;
            }
            const rms = Math.sqrt(sumSq / nodeObj.timeData.length);

            // Voice Activity Detection (VAD) silence threshold
            if (rms < 0.015) {
                return { isSilent: true, audioScore: 100, rms, spectralCentroid: 0 };
            }

            // 2. Compute Spectral Centroid and High-Frequency Energy Ratio
            let num = 0, den = 0;
            const nyquist = ctx.sampleRate / 2;
            const binWidth = nyquist / nodeObj.freqData.length;
            let highFreqEnergy = 0, totalEnergy = 0;

            for (let i = 0; i < nodeObj.freqData.length; i++) {
                const magnitude = nodeObj.freqData[i];
                const freq = i * binWidth;
                num += freq * magnitude;
                den += magnitude;
                totalEnergy += magnitude;
                if (freq > 4000) {
                    highFreqEnergy += magnitude;
                }
            }

            const spectralCentroid = den > 0 ? (num / den) : 0;
            const highFreqRatio = totalEnergy > 0 ? (highFreqEnergy / totalEnergy) : 0;

            // 3. Acoustic Consistency Heuristic
            // Natural human speech typically centers between 800Hz and 3200Hz
            let penalty = 0;
            if (spectralCentroid > 3800 || spectralCentroid < 400) {
                penalty += 25; // Out of typical human vocal formant envelope
            }
            if (highFreqRatio > 0.45) {
                penalty += 35; // Unnatural synthetic vocoder high-frequency artifact
            }

            const audioScore = Math.max(20, Math.min(100, 100 - penalty));
            return {
                isSilent: false,
                audioScore,
                rms: Math.round(rms * 1000) / 1000,
                spectralCentroid: Math.round(spectralCentroid),
                highFreqRatio: Math.round(highFreqRatio * 100) / 100
            };
        } catch (err) {
            return { isSilent: true, audioScore: 100 };
        }
    }

    // ---------------------------------------------------------------------------
    // 2. IN-CALL HUD DASHBOARD INJECTION
    // ---------------------------------------------------------------------------
    function injectUI() {
        if (document.getElementById('df-dashboard')) return;

        const dashboard = document.createElement('div');
        dashboard.id = 'df-dashboard';
        dashboard.innerHTML = `
            <div class="df-header-row">
                <span class="df-title">Biometric Scanner</span>
                <span class="df-status-pill" id="df-global-pill">Active</span>
            </div>
            <div class="df-participant-list" id="df-participant-list">
                <div class="df-empty-notice" id="df-empty-notice">Scanning video feeds...</div>
            </div>
        `;
        document.body.appendChild(dashboard);
    }

    function renderHUD() {
        const listEl = document.getElementById('df-participant-list');
        const globalPill = document.getElementById('df-global-pill');
        const dashboard = document.getElementById('df-dashboard');
        if (!listEl) return;

        const now = Date.now();
        // Prune participants not seen in last 7 seconds
        for (const [id, data] of participants.entries()) {
            if (now - data.lastSeen > 7000) {
                participants.delete(id);
            }
        }

        if (participants.size === 0) {
            listEl.innerHTML = `<div class="df-empty-notice">Waiting for active participant feed...</div>`;
            globalPill.className = 'df-status-pill';
            globalPill.innerText = 'Scanning';
            dashboard.style.borderColor = '#00ff88';
            return;
        }

        let anyAlert = false;
        let html = '';

        for (const [id, data] of participants.entries()) {
            const isAlert = data.score < 50;
            if (isAlert) anyAlert = true;

            const cardClass = isAlert ? 'df-card alert' : 'df-card';
            const fillClass = isAlert ? 'df-progress-fill alert' : 'df-progress-fill';
            const statusText = isAlert ? '⚠️ ALERT' : 'Authentic';
            const audioText = data.isSilent ? '🎙️ Muted' : `🎙️ Aud: ${data.audioScore}%`;

            html += `
                <div class="${cardClass}" id="card-${id}">
                    <div class="df-card-header">
                        <span class="df-card-name" title="${data.displayName}">${data.displayName}</span>
                        <span class="df-card-score">${data.score}%</span>
                    </div>
                    <div class="df-progress-bar">
                        <div class="${fillClass}" style="width: ${data.score}%"></div>
                    </div>
                    <div class="df-chips-row">
                        <span class="df-chip">👁️ Vis: ${data.visualScore}%</span>
                        <span class="df-chip">${audioText}</span>
                        <span class="df-chip" style="margin-left: auto; font-weight: 600;">${statusText}</span>
                    </div>
                </div>
            `;
        }

        listEl.innerHTML = html;

        if (anyAlert) {
            dashboard.style.borderColor = '#ef4444';
            globalPill.className = 'df-status-pill alert';
            globalPill.innerText = 'Deepfake Alert';
        } else {
            dashboard.style.borderColor = '#00ff88';
            globalPill.className = 'df-status-pill';
            globalPill.innerText = 'Secure';
        }
    }

    // ---------------------------------------------------------------------------
    // 3. SANDBOX SCANNER IFRAME BRIDGE
    // ---------------------------------------------------------------------------
    function injectScanner() {
        scannerIframe = document.createElement('iframe');
        scannerIframe.src = chrome.runtime.getURL('scanner.html');
        scannerIframe.style.display = 'none';
        document.body.appendChild(scannerIframe);

        window.addEventListener('message', (e) => {
            if (e.data.type === 'SCANNER_READY') {
                scannerReady = true;
                const notice = document.getElementById('df-empty-notice');
                if (notice) notice.innerText = 'Scanner ready. Monitoring calls...';
            } else if (e.data.type === 'SCORE_UPDATE') {
                const { participantId, displayName, score, visualScore, audioScore, isSilent, hasFace } = e.data;
                participants.set(participantId, {
                    displayName,
                    score,
                    visualScore,
                    audioScore,
                    isSilent,
                    hasFace,
                    lastSeen: Date.now()
                });
                renderHUD();
            } else if (e.data.type === 'FRAME_PROCESSED') {
                isProcessing = false;
            } else if (e.data.type === 'DEBUG_LOG') {
                if (e.data.isError) {
                    console.error("🚨 [Scanner Engine Error]:", e.data.message);
                } else {
                    console.log("✅ [Scanner Engine]:", e.data.message);
                }
            }
        });
    }

    // ---------------------------------------------------------------------------
    // 4. MULTI-PARTICIPANT DOM VIDEO EXTRACTION
    // ---------------------------------------------------------------------------
    function extractDisplayName(vid, fallbackIndex) {
        // Look up parent container text for Google Meet, Teams, or Zoom participant names
        try {
            let el = vid.parentElement;
            let depth = 0;
            while (el && depth < 6) {
                // Google Meet name selector
                const meetName = el.querySelector('.notranslate, [data-self-name]');
                if (meetName && meetName.textContent.trim().length > 1) {
                    return meetName.textContent.trim();
                }
                // Zoom participant name selector
                const zoomName = el.querySelector('.video-avatar__avatar-name, .video-box__name');
                if (zoomName && zoomName.textContent.trim().length > 1) {
                    return zoomName.textContent.trim();
                }
                // Microsoft Teams participant name selector
                const teamsName = el.querySelector('[data-tid="participant-name"], span[id*="name"]');
                if (teamsName && teamsName.textContent.trim().length > 1) {
                    return teamsName.textContent.trim();
                }
                el = el.parentElement;
                depth++;
            }
        } catch (e) {}

        const aria = vid.getAttribute('aria-label');
        if (aria && aria.trim().length > 1) return aria.trim();

        return `Participant ${fallbackIndex}`;
    }

    function getRemoteVideos() {
        const videos = Array.from(document.querySelectorAll('video'));
        const validVideos = [];
        let index = 1;

        for (const vid of videos) {
            // Must be actively playing with valid dimensions
            if (vid.readyState >= 2 && vid.videoWidth > 0 && !vid.paused) {
                const rect = vid.getBoundingClientRect();
                const area = rect.width * rect.height;
                const isMirrored = vid.style.transform && vid.style.transform.includes('scaleX(-1)');

                // Exclude mirrored self-view and small preview tiles
                if (!isMirrored && area > 4000) {
                    if (!vid.dataset.dfParticipantId) {
                        vid.dataset.dfParticipantId = `p-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
                    }
                    if (!vid.dataset.dfDisplayName) {
                        vid.dataset.dfDisplayName = extractDisplayName(vid, index);
                    }
                    validVideos.push(vid);
                    index++;
                }
            }
        }
        return validVideos;
    }

    // ---------------------------------------------------------------------------
    // 5. CYCLICAL ROUND-ROBIN FRAME CAPTURE & STREAMING
    // ---------------------------------------------------------------------------
    function captureAndStreamFrames() {
        const remoteVideos = getRemoteVideos();

        if (remoteVideos.length > 0 && !isProcessing && scannerReady) {
            const targetVideo = remoteVideos[roundRobinIndex % remoteVideos.length];
            roundRobinIndex++;

            const participantId = targetVideo.dataset.dfParticipantId;
            const displayName = targetVideo.dataset.dfDisplayName;
            const audioMetrics = getAudioMetricsForElement(targetVideo);

            isProcessing = true;
            createImageBitmap(targetVideo).then((bitmap) => {
                scannerIframe.contentWindow.postMessage(
                    {
                        type: 'PROCESS_FRAME',
                        bitmap: bitmap,
                        participantId: participantId,
                        displayName: displayName,
                        audioMetrics: audioMetrics
                    },
                    '*',
                    [bitmap]
                );
            }).catch(() => {
                isProcessing = false;
            });
        }

        requestAnimationFrame(captureAndStreamFrames);
    }

    // Initialize subsystems
    injectUI();
    injectScanner();
    captureAndStreamFrames();
}
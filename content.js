/**
 * Real-Time Deepfake Detection - Content Script Orchestrator
 * =========================================================
 * Injected into Google Meet, Microsoft Teams, and Zoom Web.
 * 
 * Features:
 * 1. URL Gate: Validated inside IIFE to eliminate SyntaxError. Exits on home/landing pages.
 * 2. SPA Navigation Watcher: Automatically initializes when navigating from /home to a call.
 * 3. Privacy-Safe Audio Extraction: Uses vid.captureStream() with Web Audio API (AnalyserNode).
 * 4. Accurate DOM Scraping: Extracts participant display names from platform wrappers.
 * 5. Collapsible Multi-Speaker HUD (#df-wrapper, #df-toggle-btn, and #df-card-${safeId}).
 * 6. High-performance asynchronous frame streaming via ImageBitmap transfers.
 */

(function () {
    'use strict';

    // ---------------------------------------------------------------------------
    // 1. URL GATE & SPA NAVIGATION WATCHER
    // ---------------------------------------------------------------------------
    function isMeetingPage() {
        const path = window.location.pathname;
        return path !== '/' && path !== '/home' && path.length > 1;
    }

    // If on homepage or landing page, do not inject UI.
    // Watch for client-side single-page navigation to an active meeting room.
    if (!isMeetingPage()) {
        let lastPath = window.location.pathname;
        const spaWatcher = setInterval(() => {
            if (window.location.pathname !== lastPath) {
                lastPath = window.location.pathname;
                if (isMeetingPage()) {
                    clearInterval(spaWatcher);
                    initExtension();
                }
            }
        }, 1000);
        return;
    }

    // Only execute in the top-level window context
    if (window.self !== window.top) {
        return;
    }

    initExtension();

    function initExtension() {
        console.log("🛡️ [Deepfake Detector]: Initializing extension in meeting session:", window.location.href);

        let isProcessing = false;
        let scannerIframe = null;
        let scannerReady = false;
        let roundRobinIndex = 0;

        // Track active participants and DOM elements
        const activeParticipants = new Map(); // safeId -> { id, displayName, score, visualScore, audioScore, isSilent, hasFace, lastSeen }

        // ---------------------------------------------------------------------------
        // 2. PRIVACY-PRESERVING AUDIO EXTRACTION (NO getUserMedia)
        // ---------------------------------------------------------------------------
        let audioCtx = null;
        const videoAudioMap = new WeakMap();

        function getAudioContext() {
            if (!audioCtx) {
                const AudioContextClass = window.AudioContext || window.webkitAudioContext;
                if (AudioContextClass) {
                    audioCtx = new AudioContextClass();
                }
            }
            if (audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume().catch(() => {});
            }
            return audioCtx;
        }

        function extractAudioMetrics(videoEl) {
            try {
                const ctx = getAudioContext();
                if (!ctx) return { isSilent: true, audioScore: 100, rms: 0, spectralCentroid: 0 };

                let audioNode = videoAudioMap.get(videoEl);

                if (!audioNode) {
                    // Use vid.captureStream() on remote video - NEVER getUserMedia
                    let stream = null;
                    if (typeof videoEl.captureStream === 'function') {
                        try { stream = videoEl.captureStream(); } catch (e) {}
                    } else if (typeof videoEl.mozCaptureStream === 'function') {
                        try { stream = videoEl.mozCaptureStream(); } catch (e) {}
                    } else if (videoEl.srcObject instanceof MediaStream) {
                        stream = videoEl.srcObject;
                    }

                    if (stream && typeof stream.getAudioTracks === 'function' && stream.getAudioTracks().length > 0) {
                        const source = ctx.createMediaStreamSource(stream);
                        const analyser = ctx.createAnalyser();
                        analyser.fftSize = 512;
                        analyser.smoothingTimeConstant = 0.8;
                        source.connect(analyser); // Connect only to analyser, leaving speakers untouched

                        audioNode = {
                            analyser,
                            freqData: new Uint8Array(analyser.frequencyBinCount),
                            timeData: new Uint8Array(analyser.fftSize)
                        };
                        videoAudioMap.set(videoEl, audioNode);
                    }
                }

                if (!audioNode) {
                    return { isSilent: true, audioScore: 100, rms: 0, spectralCentroid: 0 };
                }

                audioNode.analyser.getByteTimeDomainData(audioNode.timeData);
                audioNode.analyser.getByteFrequencyData(audioNode.freqData);

                // 1. RMS Energy Calculation (Voice Activity Detection)
                let sumSq = 0;
                for (let i = 0; i < audioNode.timeData.length; i++) {
                    const norm = (audioNode.timeData[i] - 128) / 128.0;
                    sumSq += norm * norm;
                }
                const rms = Math.sqrt(sumSq / audioNode.timeData.length);

                // If audio is below silence threshold, mark silent
                if (rms < 0.015) {
                    return { isSilent: true, audioScore: 100, rms: Math.round(rms * 1000) / 1000, spectralCentroid: 0 };
                }

                // 2. Spectral Centroid and High-Frequency Energy
                let num = 0, den = 0;
                const nyquist = ctx.sampleRate / 2;
                const binWidth = nyquist / audioNode.freqData.length;
                let highFreqEnergy = 0, totalEnergy = 0;

                for (let i = 0; i < audioNode.freqData.length; i++) {
                    const magnitude = audioNode.freqData[i];
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
                let penalty = 0;
                if (spectralCentroid > 3800 || spectralCentroid < 400) {
                    penalty += 25; // Outside natural human vocal envelope
                }
                if (highFreqRatio > 0.45) {
                    penalty += 35; // Artificial high-frequency vocoder distortion
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
                return { isSilent: true, audioScore: 100, rms: 0, spectralCentroid: 0 };
            }
        }

        // ---------------------------------------------------------------------------
        // 3. DOM SCRAPING: GET REAL PARTICIPANT NAMES
        // ---------------------------------------------------------------------------
        function getParticipantName(videoEl, fallbackIndex) {
            try {
                let el = videoEl.parentElement;
                let depth = 0;

                while (el && depth < 7) {
                    // 1. Google Meet DOM selectors
                    const meetEl = el.querySelector('.notranslate, [data-self-name], div[jsname][aria-label]');
                    if (meetEl) {
                        const txt = meetEl.getAttribute('data-self-name') || meetEl.textContent;
                        if (txt && txt.trim().length > 1) return txt.trim();
                    }

                    // 2. Zoom Web DOM selectors
                    const zoomEl = el.querySelector('.video-avatar__avatar-name, .video-box__name, .participants-item__name');
                    if (zoomEl && zoomEl.textContent && zoomEl.textContent.trim().length > 1) {
                        return zoomEl.textContent.trim();
                    }

                    // 3. Microsoft Teams DOM selectors
                    const teamsEl = el.querySelector('[data-tid="participant-name"], span[id*="name"], [data-cid="roster-participant"]');
                    if (teamsEl && teamsEl.textContent && teamsEl.textContent.trim().length > 1) {
                        return teamsEl.textContent.trim();
                    }

                    el = el.parentElement;
                    depth++;
                }
            } catch (e) {}

            const aria = videoEl.getAttribute('aria-label');
            if (aria && aria.trim().length > 1) return aria.trim();

            return `Participant ${fallbackIndex}`;
        }

        function toSafeId(rawId) {
            return String(rawId).replace(/[^a-zA-Z0-9_-]/g, '_');
        }

        // ---------------------------------------------------------------------------
        // 4. NEW COLLAPSIBLE HUD: #df-wrapper, #df-toggle-btn, and #df-card-${safeId}
        // ---------------------------------------------------------------------------
        function injectUI() {
            if (document.getElementById('df-wrapper')) return;

            // Main Wrapper Container
            const wrapper = document.createElement('div');
            wrapper.id = 'df-wrapper';
            wrapper.className = 'df-expanded';

            wrapper.innerHTML = `
                <div id="df-toggle-btn" title="Toggle Deepfake Detection HUD">
                    <span class="df-shield-icon">🛡️</span>
                    <span class="df-toggle-text">Deepfake Shield</span>
                </div>
                <div id="df-panel">
                    <div class="df-panel-header">
                        <span class="df-panel-title">Real-Time Biometric HUD</span>
                        <span class="df-badge" id="df-global-badge">Monitoring</span>
                    </div>
                    <div id="df-cards-container">
                        <div class="df-empty-notice" id="df-empty-state">Waiting for active video feeds...</div>
                    </div>
                </div>
            `;

            if (document.body) {
                document.body.appendChild(wrapper);
            } else {
                document.addEventListener('DOMContentLoaded', () => {
                    document.body.appendChild(wrapper);
                });
            }

            // Setup Collapse / Expand interaction
            const toggleBtn = wrapper.querySelector('#df-toggle-btn');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', () => {
                    wrapper.classList.toggle('df-collapsed');
                    wrapper.classList.toggle('df-expanded');
                });
            }
        }

        function updateParticipantCard(data) {
            const safeId = toSafeId(data.participantId);
            const container = document.getElementById('df-cards-container');
            const emptyNotice = document.getElementById('df-empty-state');
            if (!container) return;

            if (emptyNotice) {
                emptyNotice.style.display = 'none';
            }

            let card = document.getElementById(`df-card-${safeId}`);
            const isAlert = data.score < 50;
            const statusText = isAlert ? '⚠️ ALERT' : 'Authentic';
            const cardAlertClass = isAlert ? 'df-participant-card alert' : 'df-participant-card';
            const fillAlertClass = isAlert ? 'df-bar-fill alert' : 'df-bar-fill';
            const audioText = data.isSilent ? '🎙️ Muted' : `🎙️ Aud: ${data.audioScore}%`;

            if (!card) {
                card = document.createElement('div');
                card.id = `df-card-${safeId}`;
                card.className = cardAlertClass;
                container.appendChild(card);
            } else {
                card.className = cardAlertClass;
            }

            card.innerHTML = `
                <div class="df-card-top">
                    <span class="df-name" title="${data.displayName}">${data.displayName}</span>
                    <span class="df-score-val">${data.score}%</span>
                </div>
                <div class="df-bar-track">
                    <div class="${fillAlertClass}" style="width: ${data.score}%;"></div>
                </div>
                <div class="df-metrics-row">
                    <span class="df-metric-chip">👁️ Vis: ${data.visualScore}%</span>
                    <span class="df-metric-chip">${audioText}</span>
                    <span class="df-status-tag" style="margin-left: auto;">${statusText}</span>
                </div>
            `;

            updateGlobalBadge();
        }

        function updateGlobalBadge() {
            const globalBadge = document.getElementById('df-global-badge');
            const wrapper = document.getElementById('df-wrapper');
            if (!globalBadge || !wrapper) return;

            let hasAlert = false;
            for (const [_, data] of activeParticipants.entries()) {
                if (data.score < 50) {
                    hasAlert = true;
                    break;
                }
            }

            if (hasAlert) {
                globalBadge.innerText = 'Deepfake Alert';
                globalBadge.className = 'df-badge alert';
                wrapper.classList.add('has-alert');
            } else {
                globalBadge.innerText = 'Secure';
                globalBadge.className = 'df-badge';
                wrapper.classList.remove('has-alert');
            }
        }

        function pruneInactiveParticipants() {
            const now = Date.now();
            const container = document.getElementById('df-cards-container');
            const emptyNotice = document.getElementById('df-empty-state');

            for (const [safeId, data] of activeParticipants.entries()) {
                if (now - data.lastSeen > 8000) {
                    activeParticipants.delete(safeId);
                    const card = document.getElementById(`df-card-${safeId}`);
                    if (card && card.parentNode) {
                        card.parentNode.removeChild(card);
                    }
                }
            }

            if (activeParticipants.size === 0 && emptyNotice) {
                emptyNotice.style.display = 'block';
                emptyNotice.innerText = 'Waiting for active video feeds...';
            }
            updateGlobalBadge();
        }

        setInterval(pruneInactiveParticipants, 4000);

        // ---------------------------------------------------------------------------
        // 5. SANDBOXED SCANNER COMMUNICATION
        // ---------------------------------------------------------------------------
        function injectScanner() {
            scannerIframe = document.createElement('iframe');
            scannerIframe.src = chrome.runtime.getURL('scanner.html');
            scannerIframe.style.display = 'none';
            document.body.appendChild(scannerIframe);

            window.addEventListener('message', (e) => {
                if (e.data.type === 'SCANNER_READY') {
                    scannerReady = true;
                    const notice = document.getElementById('df-empty-state');
                    if (notice) notice.innerText = 'Scanner Engine Ready. Monitoring video streams...';
                } else if (e.data.type === 'SCORE_UPDATE') {
                    const { participantId, displayName, score, visualScore, audioScore, isSilent, hasFace } = e.data;
                    const safeId = toSafeId(participantId);

                    const record = {
                        participantId,
                        displayName,
                        score,
                        visualScore,
                        audioScore,
                        isSilent,
                        hasFace,
                        lastSeen: Date.now()
                    };
                    activeParticipants.set(safeId, record);
                    updateParticipantCard(record);
                } else if (e.data.type === 'FRAME_PROCESSED') {
                    isProcessing = false;
                } else if (e.data.type === 'DEBUG_LOG') {
                    if (e.data.isError) {
                        console.error("🚨 [Scanner Error]:", e.data.message);
                    } else {
                        console.log("✅ [Scanner]:", e.data.message);
                    }
                }
            });
        }

        // ---------------------------------------------------------------------------
        // 6. DOM VIDEO EXTRACTION & ROUND-ROBIN FRAME SCHEDULING
        // ---------------------------------------------------------------------------
        function getRemoteVideos() {
            const videos = Array.from(document.querySelectorAll('video'));
            const remoteVideos = [];
            let index = 1;

            for (const vid of videos) {
                if (vid.readyState >= 1 && vid.videoWidth > 0 && !vid.paused) {
                    const rect = vid.getBoundingClientRect();
                    const area = rect.width * rect.height;
                    const isMirrored = vid.style.transform && vid.style.transform.includes('scaleX(-1)');

                    // Exclude local mirrored self-view and tiny thumbnail icons
                    if (!isMirrored && area > 3000) {
                        if (!vid.dataset.dfParticipantId) {
                            vid.dataset.dfParticipantId = `part-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
                        }
                        if (!vid.dataset.dfDisplayName) {
                            vid.dataset.dfDisplayName = getParticipantName(vid, index);
                        }
                        remoteVideos.push(vid);
                        index++;
                    }
                }
            }
            return remoteVideos;
        }

        function captureAndStream() {
            const videos = getRemoteVideos();

            if (videos.length > 0 && !isProcessing && scannerReady) {
                const targetVideo = videos[roundRobinIndex % videos.length];
                roundRobinIndex++;

                const participantId = targetVideo.dataset.dfParticipantId;
                const displayName = targetVideo.dataset.dfDisplayName;
                const audioMetrics = extractAudioMetrics(targetVideo);

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

            requestAnimationFrame(captureAndStream);
        }

        // Initialize extension components
        injectUI();
        injectScanner();
        captureAndStream();
    }
})();
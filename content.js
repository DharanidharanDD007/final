/**
 * Real-Time Deepfake Detection - Content Script Orchestrator
 * =========================================================
 * Injected into Google Meet, Microsoft Teams, and Zoom Web.
 * 
 * Features:
 * 1. URL Gate: Immediately exits on home/landing pages without UI injection.
 * 2. WebRTC Integrity: Zero captureStream() calls, eliminating the remote video black screen bug.
 * 3. Dynamic Silence Fallback: Dispatches audioLevel: 0 to trigger 100% visual trust in scanner.js.
 * 4. Accurate DOM Scraping: Extracts real participant names from platform wrappers.
 * 5. Collapsible Glassmorphism HUD: #df-wrapper, #df-toggle-btn, and #df-card-${safeId}.
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

    // Never inject on Google Meet / Zoom / Teams home or landing pages
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
        console.log("🛡️ [Deepfake Detector]: Extension initialized for session:", window.location.href);

        let isProcessing = false;
        let scannerIframe = null;
        let scannerReady = false;
        let roundRobinIndex = 0;

        // Track active participants and DOM elements
        const activeParticipants = new Map(); // safeId -> { id, displayName, score, visualScore, audioScore, isSilent, hasFace, lastSeen }

        // ---------------------------------------------------------------------------
        // 2. DOM SCRAPING: GET REAL PARTICIPANT NAMES
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
        // 3. COLLAPSIBLE HUD INJECTION (#df-wrapper, #df-toggle-btn, and #df-card-${safeId})
        // ---------------------------------------------------------------------------
        function injectUI() {
            if (document.getElementById('df-wrapper')) return;

            // Main Glassmorphism Wrapper Container
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

        function updateDashboard(data) {
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
                    <span class="df-metric-chip">🎙️ Aud: Muted</span>
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

        // 15-second participant auto-pruning loop
        function pruneInactiveParticipants() {
            const now = Date.now();
            const container = document.getElementById('df-cards-container');
            const emptyNotice = document.getElementById('df-empty-state');

            for (const [safeId, data] of activeParticipants.entries()) {
                if (now - data.lastSeen > 15000) {
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

        setInterval(pruneInactiveParticipants, 5000);

        // ---------------------------------------------------------------------------
        // 4. SANDBOXED SCANNER COMMUNICATION
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
                        isSilent: true,
                        hasFace,
                        lastSeen: Date.now()
                    };
                    activeParticipants.set(safeId, record);
                    updateDashboard(record);
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
        // 5. DOM VIDEO EXTRACTION & ROUND-ROBIN FRAME SCHEDULING (captureGrid)
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

        function captureGrid() {
            const videos = getRemoteVideos();

            if (videos.length > 0 && !isProcessing && scannerReady) {
                const targetVideo = videos[roundRobinIndex % videos.length];
                roundRobinIndex++;

                const participantId = targetVideo.dataset.dfParticipantId;
                const displayName = targetVideo.dataset.dfDisplayName;

                // WebRTC-Safe: Pure ImageBitmap snapshot, zero stream hijacking
                isProcessing = true;
                createImageBitmap(targetVideo).then((bitmap) => {
                    scannerIframe.contentWindow.postMessage(
                        {
                            type: 'PROCESS_FRAME',
                            bitmap: bitmap,
                            participantId: participantId,
                            displayName: displayName,
                            audioLevel: 0,
                            audioMetrics: {
                                isSilent: true,
                                audioScore: 100,
                                audioLevel: 0,
                                rms: 0,
                                spectralCentroid: 0
                            }
                        },
                        '*',
                        [bitmap]
                    );
                }).catch(() => {
                    isProcessing = false;
                });
            }

            requestAnimationFrame(captureGrid);
        }

        // Initialize extension components
        injectUI();
        injectScanner();
        captureGrid();
    }
})();
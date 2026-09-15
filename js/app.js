/**
 * AI Face Scanner — Real-Time Face Detection & Analysis
 * Developed by Sadman Sakib
 * © 2026 Sadman Sakib — AI Face Scanner
 *
 * Architecture:
 * - Client-side only browser AI inference using face-api.js & TensorFlow.js
 * - Models: Tiny Face Detector, Face Landmark 68, Age & Gender Net, Face Expression Net
 * - Temporal Smoothing & Multi-Person Tracking: Eliminates rapid age jitter and DOM buffering
 * - Zero cloud upload, 100% on-device privacy
 */

(function () {
  'use strict';

  /* ==========================================================================
     CONFIGURATION & CONSTANTS
     ========================================================================== */
  const CONFIG = {
    // Primary local models path with CDN fallback
    LOCAL_MODEL_PATH: './models',
    CDN_MODEL_PATH: 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights',
    
    // Default analysis interval in milliseconds (500ms balanced)
    DEFAULT_INTERVAL: 500,
    
    // Default sensitivity threshold
    DEFAULT_SCORE_THRESHOLD: 0.45,
    
    // TinyFaceDetector input resolution
    INPUT_SIZE: 320,
    
    // Temporal Smoothing Parameters
    AGE_SMOOTH_ALPHA: 0.08,        // Low-pass filter for rock-solid age stabilization
    AGE_HYSTERESIS_THRESHOLD: 0.75, // Only flip displayed integer if smoothed age changes by >= 0.75
    GENDER_SMOOTH_ALPHA: 0.15,     // Smoothed gender confidence
    EXPRESSION_SMOOTH_ALPHA: 0.25, // Expression stabilization
    BOX_SMOOTH_ALPHA: 0.35,        // Bounding box jitter reducer
    
    // Tracking tolerance: frames a face can be missed before card exits (avoids flickering on momentary blinks)
    MAX_MISSED_FRAMES: 3,

    // Life stages according to age
    AGE_STAGES: [
      { max: 12, label: 'Child' },
      { max: 19, label: 'Teen' },
      { max: 35, label: 'Young Adult' },
      { max: 59, label: 'Adult' },
      { max: 120, label: 'Senior' }
    ],

    // Emotion definitions with emojis and theme colors
    EMOTIONS: {
      neutral:   { emoji: '😐', label: 'Neutral', color: '#64748b' },
      happy:     { emoji: '😄', label: 'Happy', color: '#10b981' },
      sad:       { emoji: '😢', label: 'Sad', color: '#3b82f6' },
      angry:     { emoji: '😠', label: 'Angry', color: '#ef4444' },
      fearful:   { emoji: '😨', label: 'Fearful', color: '#8b5cf6' },
      disgusted: { emoji: '🤢', label: 'Disgusted', color: '#84cc16' },
      surprised: { emoji: '😲', label: 'Surprised', color: '#f59e0b' }
    }
  };

  /* ==========================================================================
     APPLICATION STATE & TRACKING REGISTRY
     ========================================================================== */
  const state = {
    modelsLoaded: false,
    cameraStream: null,
    isScanning: false,
    isAnalyzing: false,
    currentMode: 'idle', // 'idle' | 'webcam' | 'sample'
    
    // Settings
    analysisInterval: CONFIG.DEFAULT_INTERVAL,
    scoreThreshold: CONFIG.DEFAULT_SCORE_THRESHOLD,
    showLandmarks: true,
    showScanline: true,
    showEmotionBars: true,
    mirrorWebcam: true,
    
    // Performance Tracking
    lastAnalysisTimestamp: 0,
    fpsHistory: [],
    currentFps: 0,
    currentLatencyMs: 0,
    
    // Active detections
    currentDetections: [],
    
    // Multi-camera
    availableVideoDevices: [],
    currentDeviceIndex: 0
  };

  // Persistent tracked persons registry: ID -> TrackedPerson object
  const trackedPersonsMap = new Map();
  let nextPersonId = 1;

  /* ==========================================================================
     DOM ELEMENTS
     ========================================================================== */
  const DOM = {
    // Video & Canvas
    video: document.getElementById('video'),
    sampleImage: document.getElementById('sampleImage'),
    canvas: document.getElementById('overlay'),
    viewportContainer: document.getElementById('viewportContainer'),
    standbyOverlay: document.getElementById('standbyOverlay'),
    scannerLine: document.getElementById('scannerLine'),
    
    // Badges & Status Indicators
    systemStatusPill: document.getElementById('systemStatusPill'),
    systemStatusText: document.getElementById('systemStatusText'),
    systemStatusDot: document.getElementById('systemStatusDot'),
    cameraStatusPill: document.getElementById('cameraStatusPill'),
    cameraStatusText: document.getElementById('cameraStatusText'),
    viewportFaceCount: document.getElementById('viewportFaceCount'),
    viewportFaceCountNum: document.getElementById('viewportFaceCountNum'),
    liveBadge: document.getElementById('liveBadge'),
    liveBadgeText: document.getElementById('liveBadgeText'),
    perfFps: document.getElementById('perfFps'),
    perfLatency: document.getElementById('perfLatency'),
    
    // Controls
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    snapshotBtn: document.getElementById('snapshotBtn'),
    switchCamBtn: document.getElementById('switchCamBtn'),
    standbyStartBtn: document.getElementById('standbyStartBtn'),
    standbySampleBtn: document.getElementById('standbySampleBtn'),
    imageUploadInput: document.getElementById('imageUploadInput'),
    
    // Settings elements
    toggleSettingsBtn: document.getElementById('toggleSettingsBtn'),
    settingsToggleText: document.getElementById('settingsToggleText'),
    settingsPanel: document.getElementById('settingsPanel'),
    toggleLandmarks: document.getElementById('toggleLandmarks'),
    toggleScanline: document.getElementById('toggleScanline'),
    toggleEmotionBars: document.getElementById('toggleEmotionBars'),
    toggleMirror: document.getElementById('toggleMirror'),
    speedSelect: document.getElementById('speedSelect'),
    speedValue: document.getElementById('speedValue'),
    thresholdInput: document.getElementById('thresholdInput'),
    thresholdValue: document.getElementById('thresholdValue'),
    
    // Statistics & Results
    counterBadge: document.getElementById('counterBadge'),
    peopleCount: document.getElementById('peopleCount'),
    aggregateSummaryBar: document.getElementById('aggregateSummaryBar'),
    summaryAvgAge: document.getElementById('summaryAvgAge'),
    summaryTopEmotion: document.getElementById('summaryTopEmotion'),
    summaryGenderSplit: document.getElementById('summaryGenderSplit'),
    results: document.getElementById('results'),
    cardsContainer: document.getElementById('cardsContainer'),
    emptyState: document.getElementById('emptyState'),
    emptyStateTitle: document.getElementById('emptyStateTitle'),
    emptyStateDesc: document.getElementById('emptyStateDesc'),
    
    // Modals
    aboutModal: document.getElementById('aboutModal'),
    techModal: document.getElementById('techModal'),
    privacyModal: document.getElementById('privacyModal'),
    snapshotModal: document.getElementById('snapshotModal'),
    snapshotImage: document.getElementById('snapshotImage'),
    downloadSnapshotBtn: document.getElementById('downloadSnapshotBtn'),
    
    // Modal buttons
    openAboutBtn: document.getElementById('openAboutBtn'),
    openTechBtn: document.getElementById('openTechBtn'),
    openPrivacyBtn: document.getElementById('openPrivacyBtn'),
    footerAboutLink: document.getElementById('footerAboutLink'),
    footerTechLink: document.getElementById('footerTechLink'),
    footerPrivacyLink: document.getElementById('footerPrivacyLink'),
    
    // Toast Container
    toastContainer: document.getElementById('toastContainer')
  };

  /* ==========================================================================
     CORE FUNCTION 1: loadModels()
     Loads all four AI models (TinyFaceDetector, Landmark68, AgeGender, Expressions)
     with graceful fallback from local /models to jsdelivr CDN
     ========================================================================== */
  async function loadModels() {
    setSystemStatus('loading', 'Loading Face Detector...');
    setCameraStatus('Loading AI Face Detector...');

    let loadPath = CONFIG.LOCAL_MODEL_PATH;

    try {
      // Step 1: Tiny Face Detector
      await faceapi.nets.tinyFaceDetector.loadFromUri(loadPath);
      
      // Step 2: Face Landmark 68
      setSystemStatus('loading', 'Loading 68 Landmarks...');
      setCameraStatus('Loading Face Landmarks...');
      await faceapi.nets.faceLandmark68Net.loadFromUri(loadPath);
      
      // Step 3: Age & Gender Model
      setSystemStatus('loading', 'Loading Age & Gender Net...');
      setCameraStatus('Loading Age/Gender Model...');
      await faceapi.nets.ageGenderNet.loadFromUri(loadPath);
      
      // Step 4: Face Expression Model
      setSystemStatus('loading', 'Loading Expression Net...');
      setCameraStatus('Loading Expression Model...');
      await faceapi.nets.faceExpressionNet.loadFromUri(loadPath);

      onModelsReady();

    } catch (localError) {
      console.warn('Could not load models from local path, attempting CDN fallback...', localError);
      
      try {
        loadPath = CONFIG.CDN_MODEL_PATH;
        setSystemStatus('loading', 'Connecting to AI CDN...');
        
        await faceapi.nets.tinyFaceDetector.loadFromUri(loadPath);
        await faceapi.nets.faceLandmark68Net.loadFromUri(loadPath);
        await faceapi.nets.ageGenderNet.loadFromUri(loadPath);
        await faceapi.nets.faceExpressionNet.loadFromUri(loadPath);

        onModelsReady();

      } catch (cdnError) {
        console.error('Fatal: Failed to load AI models from both local and CDN sources:', cdnError);
        setSystemStatus('error', 'Model Loading Failed');
        setCameraStatus('AI Models Failed to Load');
        showToast('Failed to load AI neural models. Please check your internet connection.', 'error');
      }
    }
  }

  function onModelsReady() {
    state.modelsLoaded = true;
    
    // Update System Status Pill to Ready
    setSystemStatus('ready', 'AI Models Ready');
    setCameraStatus('Ready to Scan');

    // Enable Start Buttons
    DOM.startBtn.disabled = false;
    DOM.standbyStartBtn.disabled = false;

    showToast('AI Neural Models Ready for Inference', 'success');

    // Check for multiple camera devices
    enumerateCameraDevices();
  }

  function setSystemStatus(status, message) {
    DOM.systemStatusPill.className = `system-status-pill ${status}`;
    DOM.systemStatusText.textContent = message;
  }

  function setCameraStatus(message, isLive = false) {
    DOM.cameraStatusText.textContent = message;
    if (isLive) {
      DOM.cameraStatusPill.classList.add('live');
    } else {
      DOM.cameraStatusPill.classList.remove('live');
    }
  }

  /* ==========================================================================
     CORE FUNCTION 2: startCamera()
     Requests webcam permissions, starts MediaStream, aligns viewport
     ========================================================================== */
  async function startCamera() {
    if (!state.modelsLoaded) {
      showToast('Please wait for AI models to finish loading.', 'warning');
      return;
    }

    if (state.cameraStream) {
      console.warn('Camera stream is already active.');
      return;
    }

    // Stop any sample viewing mode & reset tracking
    stopSampleMode();
    clearAllTrackedPersons();

    setCameraStatus('Requesting Camera Access...');
    DOM.startBtn.disabled = true;

    try {
      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      };

      if (state.availableVideoDevices.length > 0 && state.availableVideoDevices[state.currentDeviceIndex]) {
        constraints.video.deviceId = { exact: state.availableVideoDevices[state.currentDeviceIndex].deviceId };
      }

      state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
      DOM.video.srcObject = state.cameraStream;
      state.currentMode = 'webcam';
      state.isScanning = true;

      // Update UI Controls
      DOM.startBtn.disabled = true;
      DOM.stopBtn.disabled = false;
      DOM.snapshotBtn.disabled = false;
      DOM.standbyOverlay.classList.add('hidden');
      DOM.liveBadge.className = 'live-badge active';
      DOM.liveBadgeText.textContent = 'LIVE SCANNER';
      
      if (state.showScanline) {
        DOM.scannerLine.classList.add('active');
      }

      setCameraStatus('Scanning Active', true);
      showToast('Webcam connected successfully.', 'info');

      DOM.video.onloadedmetadata = () => {
        DOM.video.play();
        syncCanvasDimensions();
        requestAnimationFrame(scannerLoop);
      };

    } catch (error) {
      handleCameraError(error);
    }
  }

  /* ==========================================================================
     CORE FUNCTION 3: stopCamera()
     Stops all MediaStream tracks, resets canvas and statistics
     ========================================================================== */
  function stopCamera() {
    state.isScanning = false;
    state.isAnalyzing = false;
    state.currentMode = 'idle';

    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(track => track.stop());
      state.cameraStream = null;
    }

    DOM.video.srcObject = null;

    // Reset Controls
    DOM.startBtn.disabled = false;
    DOM.stopBtn.disabled = true;
    DOM.snapshotBtn.disabled = true;
    DOM.standbyOverlay.classList.remove('hidden');
    DOM.scannerLine.classList.remove('active');

    // Status
    DOM.liveBadge.className = 'live-badge inactive';
    DOM.liveBadgeText.textContent = 'OFFLINE';
    setCameraStatus('Camera Stopped');

    // Clear Overlays & Tracked Persons
    clearCanvas();
    clearAllTrackedPersons();

    DOM.perfFps.textContent = '0 FPS';
    DOM.perfLatency.textContent = '0 ms';
    DOM.viewportFaceCount.style.display = 'none';

    showToast('Camera feed stopped.', 'info');
  }

  /* ==========================================================================
     CORE FUNCTION 4: detectFaces() & scannerLoop()
     Executes neural inference at controlled intervals with performance logging
     ========================================================================== */
  function scannerLoop(timestamp) {
    if (!state.isScanning || state.currentMode !== 'webcam') {
      return;
    }

    if (timestamp - state.lastAnalysisTimestamp >= state.analysisInterval) {
      state.lastAnalysisTimestamp = timestamp;
      detectFaces(DOM.video);
    }

    requestAnimationFrame(scannerLoop);
  }

  async function detectFaces(sourceElement) {
    if (!state.modelsLoaded || state.isAnalyzing) {
      return;
    }

    if (sourceElement instanceof HTMLVideoElement && (sourceElement.readyState < 2 || sourceElement.paused)) {
      return;
    }

    state.isAnalyzing = true;
    const startTime = performance.now();

    try {
      const options = new faceapi.TinyFaceDetectorOptions({
        inputSize: CONFIG.INPUT_SIZE,
        scoreThreshold: state.scoreThreshold
      });

      // Multi-model inference
      const rawDetections = await faceapi
        .detectAllFaces(sourceElement, options)
        .withFaceLandmarks()
        .withAgeAndGender()
        .withFaceExpressions();

      const latency = Math.round(performance.now() - startTime);
      state.currentLatencyMs = latency;
      state.currentDetections = rawDetections;

      updatePerformanceMetrics(latency);

      // Match raw detections with tracked persons & apply temporal smoothing
      const activeTrackedPersons = matchDetectionsToTracked(rawDetections, sourceElement);

      // Draw bounding boxes, HUD brackets, and labels using smoothed data
      drawFaceOverlays(activeTrackedPersons, sourceElement);

      // Update aggregate numbers and ensure DOM cards are updated smoothly in-place
      updateStatistics(activeTrackedPersons);

    } catch (error) {
      console.error('Inference error during face detection:', error);
    } finally {
      state.isAnalyzing = false;
    }
  }

  function updatePerformanceMetrics(latency) {
    const now = performance.now();
    state.fpsHistory.push(now);
    
    state.fpsHistory = state.fpsHistory.filter(t => now - t <= 1000);
    state.currentFps = state.fpsHistory.length;

    DOM.perfFps.textContent = `${state.currentFps} FPS`;
    DOM.perfLatency.textContent = `${latency} ms`;
  }

  /* ==========================================================================
     MULTI-PERSON TEMPORAL TRACKING & STABILIZATION ENGINE
     Solves the rapid age jumping and board buffering/flickering
     ========================================================================== */
  function matchDetectionsToTracked(rawDetections, sourceElement) {
    // If no detections in sample mode, clear immediately
    if (state.currentMode === 'sample' && rawDetections.length === 0) {
      clearAllTrackedPersons();
      return [];
    }

    // Convert raw detections to normalized geometry
    const incoming = rawDetections.map(det => {
      const box = det.detection.box;
      return {
        detection: det,
        box: {
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
          centerX: box.x + box.width / 2,
          centerY: box.y + box.height / 2
        }
      };
    });

    const matchedIncomingIndices = new Set();

    // 1. Match each existing tracked person with the closest incoming detection
    for (const [id, tracked] of trackedPersonsMap.entries()) {
      let bestDist = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < incoming.length; i++) {
        if (matchedIncomingIndices.has(i)) continue;

        const inc = incoming[i];
        const dx = tracked.box.centerX - inc.box.centerX;
        const dy = tracked.box.centerY - inc.box.centerY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Maximum spatial distance threshold based on face size
        const maxDist = Math.max(tracked.box.width, tracked.box.height) * 1.35;

        if (dist < maxDist && dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }

      if (bestIdx !== -1) {
        matchedIncomingIndices.add(bestIdx);
        updateExistingTrackedPerson(tracked, incoming[bestIdx].detection, incoming[bestIdx].box);
      } else {
        // Face not detected in this frame -> increment missed counter
        tracked.missedFrames++;
      }
    }

    // 2. Any incoming detection not matched to an existing face is a new person
    for (let i = 0; i < incoming.length; i++) {
      if (!matchedIncomingIndices.has(i)) {
        createNewTrackedPerson(incoming[i].detection, incoming[i].box);
      }
    }

    // 3. Handle disappearing faces: allow grace period of MAX_MISSED_FRAMES
    for (const [id, tracked] of trackedPersonsMap.entries()) {
      if (tracked.missedFrames >= CONFIG.MAX_MISSED_FRAMES) {
        removeTrackedPerson(id);
      }
    }

    // 4. Return currently active faces, sorted from left to right (natural order)
    const activeList = Array.from(trackedPersonsMap.values()).filter(t => t.missedFrames < CONFIG.MAX_MISSED_FRAMES);
    activeList.sort((a, b) => a.box.x - b.box.x);

    // Update Person numbering (Person 1, Person 2...) smoothly
    activeList.forEach((person, index) => {
      const num = index + 1;
      if (person.personNumber !== num) {
        person.personNumber = num;
        if (person.domRefs) {
          if (person.domRefs.personBadge) person.domRefs.personBadge.textContent = num;
          if (person.domRefs.personTitle) person.domRefs.personTitle.textContent = `Person ${num}`;
        }
      }
    });

    return activeList;
  }

  function createNewTrackedPerson(det, box) {
    const newId = nextPersonId++;
    const rawAge = det.age;
    const initialDisplayAge = Math.round(rawAge);

    // Initial smoothed expressions
    const smoothedExp = { ...det.expressions };
    let topExp = 'neutral';
    let maxExpProb = 0;
    for (const [exp, prob] of Object.entries(smoothedExp)) {
      if (prob > maxExpProb) {
        topExp = exp;
        maxExpProb = prob;
      }
    }

    const tracked = {
      id: newId,
      personNumber: trackedPersonsMap.size + 1,
      box: { ...box },
      detection: det,
      
      // Age Stabilization
      rawAge: rawAge,
      smoothedAge: rawAge,
      displayAge: initialDisplayAge,
      ageHistory: [rawAge],
      lifeStage: getLifeStage(initialDisplayAge),
      
      // Gender Stabilization
      gender: det.gender === 'male' ? 'Male' : 'Female',
      smoothedGenderProb: det.genderProbability,
      genderProb: Math.round(det.genderProbability * 100),
      
      // Expression Stabilization
      smoothedExpressions: smoothedExp,
      topExpression: topExp,
      topExpressionProb: Math.round(maxExpProb * 100),
      
      // Detection score
      detectionScore: Math.round(det.detection.score * 100),
      
      // Tracking state
      lastSeen: Date.now(),
      missedFrames: 0,
      
      // DOM Elements & cached references
      cardElement: null,
      domRefs: null
    };

    // Create persistent DOM card (renders with entrance animation ONCE)
    const { card, refs } = createPersonCard(tracked);
    tracked.cardElement = card;
    tracked.domRefs = refs;

    // Append to container
    DOM.cardsContainer.appendChild(card);
    trackedPersonsMap.set(newId, tracked);

    return tracked;
  }

  function updateExistingTrackedPerson(tracked, det, newBox) {
    tracked.missedFrames = 0;
    tracked.lastSeen = Date.now();
    tracked.detection = det;

    // 1. Smooth Bounding Box Coordinates (Reduces visual canvas shaking)
    const bAlpha = CONFIG.BOX_SMOOTH_ALPHA;
    tracked.box.x = tracked.box.x * (1 - bAlpha) + newBox.x * bAlpha;
    tracked.box.y = tracked.box.y * (1 - bAlpha) + newBox.y * bAlpha;
    tracked.box.width = tracked.box.width * (1 - bAlpha) + newBox.width * bAlpha;
    tracked.box.height = tracked.box.height * (1 - bAlpha) + newBox.height * bAlpha;
    tracked.box.centerX = tracked.box.x + tracked.box.width / 2;
    tracked.box.centerY = tracked.box.y + tracked.box.height / 2;

    // 2. Detection score
    tracked.detectionScore = Math.round(det.detection.score * 100);

    // 3. ROCK-SOLID AGE STABILIZATION
    // Exponential Moving Average + Integer Hysteresis
    const rawAge = det.age;
    tracked.ageHistory.push(rawAge);
    if (tracked.ageHistory.length > 10) tracked.ageHistory.shift();

    const aAlpha = CONFIG.AGE_SMOOTH_ALPHA;
    tracked.smoothedAge = (tracked.smoothedAge * (1 - aAlpha)) + (rawAge * aAlpha);

    // Only update displayed digit when smoothed age drifts by >= 0.75
    // Completely stops annoying flickering between numbers like 22 <-> 23
    if (Math.abs(tracked.smoothedAge - tracked.displayAge) >= CONFIG.AGE_HYSTERESIS_THRESHOLD) {
      tracked.displayAge = Math.round(tracked.smoothedAge);
      tracked.lifeStage = getLifeStage(tracked.displayAge);
    }

    // 4. Gender Stabilization
    const gAlpha = CONFIG.GENDER_SMOOTH_ALPHA;
    tracked.smoothedGenderProb = (tracked.smoothedGenderProb * (1 - gAlpha)) + (det.genderProbability * gAlpha);
    tracked.gender = det.gender === 'male' ? 'Male' : 'Female';
    tracked.genderProb = Math.round(tracked.smoothedGenderProb * 100);

    // 5. Expression Stabilization
    const eAlpha = CONFIG.EXPRESSION_SMOOTH_ALPHA;
    for (const [exp, prob] of Object.entries(det.expressions)) {
      tracked.smoothedExpressions[exp] = (tracked.smoothedExpressions[exp] || 0) * (1 - eAlpha) + (prob * eAlpha);
    }

    let topExp = 'neutral';
    let maxExpProb = 0;
    for (const [exp, prob] of Object.entries(tracked.smoothedExpressions)) {
      if (prob > maxExpProb) {
        topExp = exp;
        maxExpProb = prob;
      }
    }
    tracked.topExpression = topExp;
    tracked.topExpressionProb = Math.round(maxExpProb * 100);

    // 6. In-place DOM update without re-creating or buffering cards
    updatePersonCardDOM(tracked);
  }

  function removeTrackedPerson(id) {
    const tracked = trackedPersonsMap.get(id);
    if (!tracked) return;

    if (tracked.cardElement) {
      tracked.cardElement.classList.add('exiting');
      setTimeout(() => {
        if (tracked.cardElement && tracked.cardElement.parentNode) {
          tracked.cardElement.remove();
        }
      }, 250);
    }

    trackedPersonsMap.delete(id);
  }

  function clearAllTrackedPersons() {
    for (const [id, tracked] of trackedPersonsMap.entries()) {
      if (tracked.cardElement) {
        tracked.cardElement.remove();
      }
    }
    trackedPersonsMap.clear();
    nextPersonId = 1;

    DOM.cardsContainer.innerHTML = '';
    DOM.emptyState.style.display = 'flex';
    DOM.peopleCount.textContent = '0';
    DOM.counterBadge.classList.remove('has-faces');
    DOM.aggregateSummaryBar.style.display = 'none';
  }

  /* ==========================================================================
     CORE FUNCTION 5: drawFaceOverlays()
     Renders modern cyan/blue bounding boxes, HUD brackets, landmarks & tags
     using stabilized tracking coordinates and calm age values
     ========================================================================== */
  function drawFaceOverlays(activeTrackedPersons, sourceElement) {
    syncCanvasDimensions();

    const displaySize = {
      width: DOM.canvas.width,
      height: DOM.canvas.height
    };

    clearCanvas();

    const ctx = DOM.canvas.getContext('2d');
    const isMirrored = state.currentMode === 'webcam' && state.mirrorWebcam;

    // Scale calculation between video resolution and display canvas
    let scaleX = 1;
    let scaleY = 1;
    if (sourceElement instanceof HTMLVideoElement && sourceElement.videoWidth > 0) {
      scaleX = displaySize.width / sourceElement.videoWidth;
      scaleY = displaySize.height / sourceElement.videoHeight;
    } else if (sourceElement instanceof HTMLImageElement && sourceElement.naturalWidth > 0) {
      scaleX = displaySize.width / sourceElement.naturalWidth;
      scaleY = displaySize.height / sourceElement.naturalHeight;
    }

    activeTrackedPersons.forEach(person => {
      let boxX = person.box.x * scaleX;
      let boxY = person.box.y * scaleY;
      let boxW = person.box.width * scaleX;
      let boxH = person.box.height * scaleY;

      if (isMirrored) {
        boxX = displaySize.width - (boxX + boxW);
      }

      // 1. Draw Subtle Translucent Glow Fill
      ctx.fillStyle = 'rgba(14, 165, 233, 0.08)';
      roundRect(ctx, boxX, boxY, boxW, boxH, 8, true, false);

      // 2. Draw Cyan Bounding Box
      ctx.strokeStyle = '#0ea5e9';
      ctx.lineWidth = 2;
      ctx.shadowColor = 'rgba(14, 165, 233, 0.6)';
      ctx.shadowBlur = 8;
      roundRect(ctx, boxX, boxY, boxW, boxH, 8, false, true);
      ctx.shadowBlur = 0;

      // 3. Draw High-Tech Corner Brackets
      drawCornerBrackets(ctx, boxX, boxY, boxW, boxH, 14, '#38bdf8');

      // 4. Draw Floating Header Pill with Stabilized Data
      drawFaceHeaderPill(ctx, boxX, boxY, boxW, person);

      // 5. Draw 68 Face Landmarks (if enabled)
      if (state.showLandmarks && person.detection && person.detection.landmarks) {
        drawCustomLandmarks(ctx, person.detection.landmarks, isMirrored, displaySize.width, scaleX, scaleY);
      }
    });

    // Floating Face Count Badge inside Viewport
    if (activeTrackedPersons.length > 0) {
      DOM.viewportFaceCount.style.display = 'block';
      DOM.viewportFaceCountNum.textContent = activeTrackedPersons.length;
    } else {
      DOM.viewportFaceCount.style.display = 'none';
    }
  }

  function drawCornerBrackets(ctx, x, y, w, h, len, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + len);
    ctx.lineTo(x, y);
    ctx.lineTo(x + len, y);
    ctx.stroke();

    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + len);
    ctx.stroke();

    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + h - len);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + len, y + h);
    ctx.stroke();

    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - len);
    ctx.stroke();
  }

  function drawFaceHeaderPill(ctx, x, y, w, person) {
    const expMeta = CONFIG.EMOTIONS[person.topExpression] || { emoji: '🙂', label: person.topExpression };
    const labelText = `Person ${person.personNumber}  •  ${person.displayAge}y  •  ${person.gender} (${person.genderProb}%)  •  ${expMeta.emoji} ${expMeta.label}`;

    ctx.font = '600 12px Inter, system-ui, sans-serif';
    const textMetrics = ctx.measureText(labelText);
    const pillWidth = Math.max(textMetrics.width + 18, 140);
    const pillHeight = 24;
    
    let pillX = x + (w - pillWidth) / 2;
    if (pillX < 6) pillX = 6;
    if (pillX + pillWidth > DOM.canvas.width - 6) pillX = DOM.canvas.width - pillWidth - 6;
    
    let pillY = y - pillHeight - 6;
    if (pillY < 6) {
      pillY = y + 6;
    }

    // Pill Background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    roundRect(ctx, pillX, pillY, pillWidth, pillHeight, 6, true, false);

    // Pill Border
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.5)';
    ctx.lineWidth = 1;
    roundRect(ctx, pillX, pillY, pillWidth, pillHeight, 6, false, true);

    // Pill Text
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, pillX + 9, pillY + pillHeight / 2);
  }

  function drawCustomLandmarks(ctx, landmarks, isMirrored, canvasWidth, scaleX = 1, scaleY = 1) {
    const positions = landmarks.positions;
    ctx.fillStyle = 'rgba(56, 189, 248, 0.85)';

    for (let i = 0; i < positions.length; i++) {
      let ptX = positions[i].x * scaleX;
      let ptY = positions[i].y * scaleY;

      if (isMirrored) {
        ptX = canvasWidth - ptX;
      }

      ctx.beginPath();
      ctx.arc(ptX, ptY, 1.8, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    if (typeof radius === 'number') {
      radius = { tl: radius, tr: radius, br: radius, bl: radius };
    }
    ctx.beginPath();
    ctx.moveTo(x + radius.tl, y);
    ctx.lineTo(x + width - radius.tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
    ctx.lineTo(x + width, y + height - radius.br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
    ctx.lineTo(x + radius.bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
    ctx.lineTo(x, y + radius.tl);
    ctx.quadraticCurveTo(x, y, x + radius.tl, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  function syncCanvasDimensions() {
    const containerWidth = DOM.viewportContainer.clientWidth;
    const containerHeight = DOM.viewportContainer.clientHeight;

    if (DOM.canvas.width !== containerWidth || DOM.canvas.height !== containerHeight) {
      DOM.canvas.width = containerWidth;
      DOM.canvas.height = containerHeight;
    }
  }

  function clearCanvas() {
    const ctx = DOM.canvas.getContext('2d');
    ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
  }

  /* ==========================================================================
     CORE FUNCTION 6: updateStatistics()
     Updates real-time person count, aggregate summary without DOM recreation
     ========================================================================== */
  function updateStatistics(activeTrackedPersons) {
    const count = activeTrackedPersons.length;
    DOM.peopleCount.textContent = count;

    if (count > 0) {
      DOM.counterBadge.classList.add('has-faces');
      DOM.aggregateSummaryBar.style.display = 'grid';
      DOM.emptyState.style.display = 'none';
      updateAggregateMetrics(activeTrackedPersons);
    } else {
      DOM.counterBadge.classList.remove('has-faces');
      DOM.aggregateSummaryBar.style.display = 'none';
      DOM.emptyState.style.display = 'flex';
      
      DOM.emptyStateTitle.textContent = state.currentMode === 'idle' ? 'Scanner Ready' : 'Looking for Faces...';
      DOM.emptyStateDesc.textContent = state.currentMode === 'idle' 
        ? 'Start the camera or select a sample image to begin real-time multi-person face analysis.'
        : 'No faces currently detected in view. Ensure proper lighting and look towards the camera.';
    }
  }

  function updateAggregateMetrics(activeTrackedPersons) {
    let totalAge = 0;
    let maleCount = 0;
    let femaleCount = 0;
    const emotionTotals = {};

    activeTrackedPersons.forEach(person => {
      totalAge += person.displayAge;
      if (person.gender === 'Male') maleCount++;
      else femaleCount++;

      for (const [exp, prob] of Object.entries(person.smoothedExpressions)) {
        emotionTotals[exp] = (emotionTotals[exp] || 0) + prob;
      }
    });

    // 1. Stabilized Avg Age
    const avgAge = Math.round(totalAge / activeTrackedPersons.length);
    DOM.summaryAvgAge.textContent = `${avgAge} yrs`;

    // 2. Top Group Emotion
    let topExp = 'neutral';
    let maxExpProb = 0;
    for (const [exp, prob] of Object.entries(emotionTotals)) {
      if (prob > maxExpProb) {
        topExp = exp;
        maxExpProb = prob;
      }
    }
    const expMeta = CONFIG.EMOTIONS[topExp] || { emoji: '😐', label: topExp };
    DOM.summaryTopEmotion.textContent = `${expMeta.emoji} ${expMeta.label}`;

    // 3. Gender Split
    DOM.summaryGenderSplit.textContent = `${maleCount}M • ${femaleCount}F`;
  }

  /* ==========================================================================
     CORE FUNCTION 7: createPersonCard() & updatePersonCardDOM()
     Builds cards ONCE and updates DOM nodes in-place to prevent buffering
     ========================================================================== */
  function createPersonCard(person) {
    const expMeta = CONFIG.EMOTIONS[person.topExpression] || { emoji: '😐', label: person.topExpression };

    const card = document.createElement('div');
    card.className = 'person-card entering';
    card.setAttribute('data-person-id', person.id);

    // Remove entrance animation class after initial entrance completes
    setTimeout(() => {
      card.classList.remove('entering');
    }, 400);

    card.innerHTML = `
      <div class="person-header">
        <div class="person-identity">
          <div class="person-number-badge ref-badge">${person.personNumber}</div>
          <span class="person-title-text ref-title">Person ${person.personNumber}</span>
        </div>
        <div class="person-status-badge">
          <span class="status-mini-dot"></span>
          Tracked
        </div>
      </div>

      <!-- Metric 1: Estimated Age (Calm, Stabilized) -->
      <div class="person-metric-row">
        <div class="metric-meta">
          <span class="metric-label">🎂 Estimated Age</span>
          <span class="metric-sub">Stabilized AI Regression</span>
        </div>
        <div class="metric-value-row">
          <span class="metric-value-primary">
            <span class="ref-age-val">${person.displayAge}</span>
            <small style="font-size:0.875rem; font-weight:500;"> years</small>
          </span>
          <span class="age-tag ref-age-tag">${person.lifeStage}</span>
        </div>
      </div>

      <!-- Metric 2: Gender Model -->
      <div class="person-metric-row">
        <div class="metric-meta">
          <span class="metric-label">⚧️ Gender Model</span>
          <span class="metric-sub ref-gender-conf">${person.genderProb}% Confidence</span>
        </div>
        <div class="metric-value-row">
          <span class="metric-value-primary ref-gender-val">${person.gender}</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-fill ref-gender-bar" style="width: ${person.genderProb}%;"></div>
        </div>
      </div>

      <!-- Metric 3: Facial Expression -->
      <div class="person-metric-row">
        <div class="metric-meta">
          <span class="metric-label">🙂 Expression State</span>
          <span class="metric-sub ref-exp-conf">${person.topExpressionProb}% Match</span>
        </div>
        <div class="metric-value-row">
          <span class="metric-value-primary ref-exp-val">${expMeta.emoji} ${expMeta.label}</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-fill expression-fill ref-exp-bar" style="width: ${person.topExpressionProb}%;"></div>
        </div>
        <div class="ref-emotions-container"></div>
      </div>

      <!-- Metric 4: Face Detection Confidence -->
      <div class="person-metric-row">
        <div class="metric-meta">
          <span class="metric-label">🎯 Detection Confidence</span>
          <span class="metric-sub ref-det-conf">${person.detectionScore}% Score</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-fill ref-det-bar" style="width: ${person.detectionScore}%;"></div>
        </div>
      </div>
    `;

    // Cache direct DOM element references for high-speed in-place updates
    const refs = {
      personBadge: card.querySelector('.ref-badge'),
      personTitle: card.querySelector('.ref-title'),
      ageVal: card.querySelector('.ref-age-val'),
      ageTag: card.querySelector('.ref-age-tag'),
      genderConf: card.querySelector('.ref-gender-conf'),
      genderVal: card.querySelector('.ref-gender-val'),
      genderBar: card.querySelector('.ref-gender-bar'),
      expConf: card.querySelector('.ref-exp-conf'),
      expVal: card.querySelector('.ref-exp-val'),
      expBar: card.querySelector('.ref-exp-bar'),
      emotionsContainer: card.querySelector('.ref-emotions-container'),
      detConf: card.querySelector('.ref-det-conf'),
      detBar: card.querySelector('.ref-det-bar')
    };

    updateEmotionBreakdownDOM(refs.emotionsContainer, person.smoothedExpressions);

    return { card, refs };
  }

  function updatePersonCardDOM(person) {
    const refs = person.domRefs;
    if (!refs) return;

    // 1. Age (Only updates text when integer actually transitions!)
    if (refs.ageVal.textContent !== String(person.displayAge)) {
      refs.ageVal.textContent = person.displayAge;
    }
    if (refs.ageTag.textContent !== person.lifeStage) {
      refs.ageTag.textContent = person.lifeStage;
    }

    // 2. Gender
    refs.genderConf.textContent = `${person.genderProb}% Confidence`;
    refs.genderVal.textContent = person.gender;
    refs.genderBar.style.width = `${person.genderProb}%`;

    // 3. Expression
    const expMeta = CONFIG.EMOTIONS[person.topExpression] || { emoji: '😐', label: person.topExpression };
    refs.expConf.textContent = `${person.topExpressionProb}% Match`;
    refs.expVal.innerHTML = `${expMeta.emoji} ${expMeta.label}`;
    refs.expBar.style.width = `${person.topExpressionProb}%`;

    if (state.showEmotionBars) {
      updateEmotionBreakdownDOM(refs.emotionsContainer, person.smoothedExpressions);
    } else {
      refs.emotionsContainer.innerHTML = '';
    }

    // 4. Detection Score
    refs.detConf.textContent = `${person.detectionScore}% Score`;
    refs.detBar.style.width = `${person.detectionScore}%`;
  }

  function updateEmotionBreakdownDOM(container, expressions) {
    if (!state.showEmotionBars) {
      container.innerHTML = '';
      return;
    }

    // Top 4 expressions by smoothed probability
    const sorted = Object.entries(expressions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4);

    let box = container.querySelector('.emotion-breakdown-box');
    if (!box) {
      box = document.createElement('div');
      box.className = 'emotion-breakdown-box';
      sorted.forEach(([expName, prob]) => {
        const meta = CONFIG.EMOTIONS[expName] || { emoji: '•', label: expName };
        const pct = Math.round(prob * 100);
        const item = document.createElement('div');
        item.className = 'emotion-item';
        item.innerHTML = `
          <span class="emotion-name">${meta.emoji} ${meta.label}</span>
          <div class="emotion-bar-track">
            <div class="emotion-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="emotion-pct">${pct}%</span>
        `;
        box.appendChild(item);
      });
      container.appendChild(box);
    } else {
      const items = box.querySelectorAll('.emotion-item');
      sorted.forEach(([expName, prob], idx) => {
        if (items[idx]) {
          const meta = CONFIG.EMOTIONS[expName] || { emoji: '•', label: expName };
          const pct = Math.round(prob * 100);
          const nameEl = items[idx].querySelector('.emotion-name');
          const fillEl = items[idx].querySelector('.emotion-bar-fill');
          const pctEl = items[idx].querySelector('.emotion-pct');

          if (nameEl) nameEl.innerHTML = `${meta.emoji} ${meta.label}`;
          if (fillEl) fillEl.style.width = `${pct}%`;
          if (pctEl) pctEl.textContent = `${pct}%`;
        }
      });
    }
  }

  function getLifeStage(age) {
    for (const stage of CONFIG.AGE_STAGES) {
      if (age <= stage.max) return stage.label;
    }
    return 'Adult';
  }

  /* ==========================================================================
     CORE FUNCTION 8: handleCameraError()
     User-friendly error handling for permissions, missing hardware, or streams
     ========================================================================== */
  function handleCameraError(error) {
    console.error('Camera initialization error:', error);
    
    stopCamera();

    let friendlyMessage = 'Unable to access your camera.';
    
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      friendlyMessage = 'Camera access was denied. Please allow camera permission in your browser address bar settings.';
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      friendlyMessage = 'No camera device detected. You can test the AI scanner using the sample photos below.';
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      friendlyMessage = 'Camera is already in use by another application. Please close other camera apps and retry.';
    } else if (error.name === 'OverconstrainedError') {
      friendlyMessage = 'Requested camera resolution is not supported by your hardware.';
    } else if (!window.isSecureContext && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
      friendlyMessage = 'Webcam access requires a secure HTTPS connection.';
    }

    setCameraStatus('Camera Error');
    showToast(friendlyMessage, 'error', 6000);
    alert(`Camera Notice:\n\n${friendlyMessage}\n\nYou can also click any of the "Try Sample" buttons to test the AI on photos immediately!`);
  }

  /* ==========================================================================
     SAMPLE PHOTOS & CUSTOM IMAGE TESTING
     Enables immediate AI face detection testing without webcam hardware
     ========================================================================== */
  const SAMPLE_IMAGES = {
    single: 'samples/sample-single.jpg',
    duo:    'samples/sample-duo.jpg',
    group:  'samples/sample-group.jpg'
  };

  async function loadSampleImage(key) {
    if (!state.modelsLoaded) {
      showToast('Please wait for AI models to finish loading.', 'warning');
      return;
    }

    const src = SAMPLE_IMAGES[key];
    if (!src) return;

    if (state.cameraStream) {
      stopCamera();
    }

    clearAllTrackedPersons();

    state.currentMode = 'sample';
    setCameraStatus(`Analyzing Sample: ${key.toUpperCase()}`);

    DOM.standbyOverlay.classList.add('hidden');
    DOM.video.style.display = 'none';
    DOM.sampleImage.style.display = 'block';
    DOM.sampleImage.src = src;

    DOM.sampleImage.onload = async () => {
      syncCanvasDimensions();
      DOM.liveBadge.className = 'live-badge active';
      DOM.liveBadgeText.textContent = 'SAMPLE MODE';
      DOM.snapshotBtn.disabled = false;

      await detectFaces(DOM.sampleImage);
      showToast(`Analyzed ${key} sample photo.`, 'info');
    };
  }

  function handleImageUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      showToast('Please select a valid image file (JPEG, PNG, WebP).', 'warning');
      return;
    }

    if (!state.modelsLoaded) {
      showToast('Please wait for AI models to load first.', 'warning');
      return;
    }

    if (state.cameraStream) {
      stopCamera();
    }

    clearAllTrackedPersons();

    const reader = new FileReader();
    reader.onload = function (e) {
      state.currentMode = 'sample';
      setCameraStatus('Analyzing Uploaded Image');
      
      DOM.standbyOverlay.classList.add('hidden');
      DOM.video.style.display = 'none';
      DOM.sampleImage.style.display = 'block';
      DOM.sampleImage.src = e.target.result;

      DOM.sampleImage.onload = async () => {
        syncCanvasDimensions();
        DOM.liveBadge.className = 'live-badge active';
        DOM.liveBadgeText.textContent = 'IMAGE MODE';
        DOM.snapshotBtn.disabled = false;

        await detectFaces(DOM.sampleImage);
        showToast('Image analyzed successfully.', 'success');
      };
    };
    reader.readAsDataURL(file);
  }

  function stopSampleMode() {
    DOM.sampleImage.style.display = 'none';
    DOM.video.style.display = 'block';
  }

  /* ==========================================================================
     SNAPSHOT FEATURE
     Composites video/image feed + canvas overlay into downloadable snapshot
     ========================================================================== */
  function captureSnapshot() {
    let source = null;

    if (state.currentMode === 'webcam' && DOM.video.srcObject) {
      source = DOM.video;
    } else if (state.currentMode === 'sample' && DOM.sampleImage.src) {
      source = DOM.sampleImage;
    }

    if (!source) {
      showToast('No active video or image to capture.', 'warning');
      return;
    }

    const offscreen = document.createElement('canvas');
    offscreen.width = DOM.canvas.width;
    offscreen.height = DOM.canvas.height;
    const ctx = offscreen.getContext('2d');

    // 1. Draw background source
    if (state.currentMode === 'webcam' && state.mirrorWebcam) {
      ctx.save();
      ctx.translate(offscreen.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(source, 0, 0, offscreen.width, offscreen.height);
      ctx.restore();
    } else {
      ctx.drawImage(source, 0, 0, offscreen.width, offscreen.height);
    }

    // 2. Composite detection canvas overlay on top
    ctx.drawImage(DOM.canvas, 0, 0);

    // 3. Attribution watermark
    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.fillRect(offscreen.width - 240, offscreen.height - 30, 234, 24);
    ctx.fillStyle = '#38bdf8';
    ctx.fillText('AI Face Scanner • Sadman Sakib', offscreen.width - 230, offscreen.height - 14);

    const dataUrl = offscreen.toDataURL('image/png');
    DOM.snapshotImage.src = dataUrl;
    DOM.downloadSnapshotBtn.href = dataUrl;
    DOM.downloadSnapshotBtn.download = `AI-Face-Scanner-${Date.now()}.png`;

    openModal(DOM.snapshotModal);
    showToast('Snapshot captured!', 'success');
  }

  /* ==========================================================================
     MULTI-CAMERA DEVICE ENUMERATION
     ========================================================================== */
  async function enumerateCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      state.availableVideoDevices = devices.filter(d => d.kind === 'videoinput');

      if (state.availableVideoDevices.length > 1) {
        DOM.switchCamBtn.style.display = 'inline-flex';
      }
    } catch (e) {
      console.warn('Could not enumerate video devices:', e);
    }
  }

  async function switchCamera() {
    if (state.availableVideoDevices.length <= 1) return;
    
    state.currentDeviceIndex = (state.currentDeviceIndex + 1) % state.availableVideoDevices.length;
    showToast(`Switching camera...`, 'info');
    
    stopCamera();
    await startCamera();
  }

  /* ==========================================================================
     SETTINGS & EVENT LISTENERS
     ========================================================================== */
  function setupEventListeners() {
    // Primary Controls
    DOM.startBtn.addEventListener('click', startCamera);
    DOM.standbyStartBtn.addEventListener('click', startCamera);
    DOM.stopBtn.addEventListener('click', stopCamera);
    DOM.snapshotBtn.addEventListener('click', captureSnapshot);
    DOM.switchCamBtn.addEventListener('click', switchCamera);

    // Standby sample trigger
    DOM.standbySampleBtn.addEventListener('click', () => loadSampleImage('single'));

    // Sample Photo Buttons
    document.querySelectorAll('[data-sample]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-sample]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        loadSampleImage(btn.getAttribute('data-sample'));
      });
    });

    // Custom Image Upload
    DOM.imageUploadInput.addEventListener('change', handleImageUpload);

    // Settings Accordion Toggle
    DOM.toggleSettingsBtn.addEventListener('click', () => {
      const isExpanded = DOM.settingsPanel.style.display === 'block';
      DOM.settingsPanel.style.display = isExpanded ? 'none' : 'block';
      DOM.toggleSettingsBtn.classList.toggle('expanded', !isExpanded);
      DOM.toggleSettingsBtn.setAttribute('aria-expanded', !isExpanded);
      DOM.settingsToggleText.textContent = isExpanded ? 'Customize' : 'Close';
    });

    // Toggle Landmarks
    DOM.toggleLandmarks.addEventListener('change', e => {
      state.showLandmarks = e.target.checked;
      const activePersons = Array.from(trackedPersonsMap.values());
      if (activePersons.length > 0) {
        const source = state.currentMode === 'webcam' ? DOM.video : DOM.sampleImage;
        drawFaceOverlays(activePersons, source);
      }
    });

    // Toggle Scanline
    DOM.toggleScanline.addEventListener('change', e => {
      state.showScanline = e.target.checked;
      if (state.isScanning && state.showScanline) {
        DOM.scannerLine.classList.add('active');
      } else {
        DOM.scannerLine.classList.remove('active');
      }
    });

    // Toggle Emotion Bars
    DOM.toggleEmotionBars.addEventListener('change', e => {
      state.showEmotionBars = e.target.checked;
      for (const tracked of trackedPersonsMap.values()) {
        updatePersonCardDOM(tracked);
      }
    });

    // Toggle Mirror
    DOM.toggleMirror.addEventListener('change', e => {
      state.mirrorWebcam = e.target.checked;
      DOM.video.classList.toggle('unmirrored', !state.mirrorWebcam);
      const activePersons = Array.from(trackedPersonsMap.values());
      if (activePersons.length > 0) {
        const source = state.currentMode === 'webcam' ? DOM.video : DOM.sampleImage;
        drawFaceOverlays(activePersons, source);
      }
    });

    // Speed / Interval Select
    DOM.speedSelect.addEventListener('change', e => {
      const val = parseInt(e.target.value, 10);
      state.analysisInterval = val;
      DOM.speedValue.textContent = `${val} ms`;
    });

    // Sensitivity Threshold Range
    DOM.thresholdInput.addEventListener('input', e => {
      const val = parseFloat(e.target.value);
      state.scoreThreshold = val;
      DOM.thresholdValue.textContent = val.toFixed(2);
    });

    // Modal Triggers
    DOM.openAboutBtn.addEventListener('click', () => openModal(DOM.aboutModal));
    DOM.footerAboutLink.addEventListener('click', () => openModal(DOM.aboutModal));
    
    DOM.openTechBtn.addEventListener('click', () => openModal(DOM.techModal));
    DOM.footerTechLink.addEventListener('click', () => openModal(DOM.techModal));

    DOM.openPrivacyBtn.addEventListener('click', () => openModal(DOM.privacyModal));
    DOM.footerPrivacyLink.addEventListener('click', () => openModal(DOM.privacyModal));

    // Modal Close buttons (data-close-modal)
    document.querySelectorAll('[data-close-modal]').forEach(btn => {
      btn.addEventListener('click', closeAllModals);
    });

    // Close on backdrop click
    document.querySelectorAll('.modal-backdrop').forEach(modal => {
      modal.addEventListener('click', e => {
        if (e.target === modal) closeAllModals();
      });
    });

    // Close on Esc key
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeAllModals();
    });

    // Window Resize handler to keep canvas aligned
    window.addEventListener('resize', debounce(() => {
      syncCanvasDimensions();
      const activePersons = Array.from(trackedPersonsMap.values());
      if (activePersons.length > 0) {
        const source = state.currentMode === 'webcam' ? DOM.video : DOM.sampleImage;
        drawFaceOverlays(activePersons, source);
      }
    }, 100));
  }

  /* ==========================================================================
     MODAL & TOAST HELPERS
     ========================================================================== */
  function openModal(modalElement) {
    closeAllModals();
    if (modalElement) {
      modalElement.style.display = 'flex';
      document.body.style.overflow = 'hidden';
    }
  }

  function closeAllModals() {
    document.querySelectorAll('.modal-backdrop').forEach(modal => {
      modal.style.display = 'none';
    });
    document.body.style.overflow = '';
  }

  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
      success: '✓',
      info: 'ℹ',
      warning: '⚠',
      error: '✕'
    };

    toast.innerHTML = `
      <span style="font-weight:700;">${icons[type] || '•'}</span>
      <span>${escapeHtml(message)}</span>
    `;

    DOM.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(12px)';
      toast.style.transition = 'all 200ms ease';
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  /* ==========================================================================
     INITIALIZATION ON DOM READY
     ========================================================================== */
  document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    syncCanvasDimensions();
    loadModels();
  });

})();

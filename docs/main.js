/* ============================================================
   TritoNapse - CXR ML Models (JS)
   Frontend Comments in part via LLM analysis of code structure and intent.
   ============================================================
   Table of Contents:
     1. Configuration 
     2. Application State
     3. DOM References
     4. Connection Status
     5. File Upload Handling
     6. Series List Management
     7. Canvas Drawing
     8. Inference Engine (Hugging Face Space)
     9. Findings Renderer
     10. Spinner Helpers
     11. Tab Switching
     12. Toolbar Interactions
     13. Utilities
     14. Initialization
   ============================================================ */

/* -----------------------------------------------------------
   1. CONFIGURATION
   Hardcoded Hugging Face Space URL.
   ----------------------------------------------------------- */

const CONFIG = {
  // Hugging Face Space 
  hfSpaceUrl: 'https://k-phantastic-tritonapse.hf.space',
};

/* -----------------------------------------------------------
   2. APPLICATION STATE
   ----------------------------------------------------------- */

let currentImageFile = null;   // File object of the active image
let currentImageObj  = null;   // HTMLImageElement of the active image 
let uploadedImages   = [];     // Array of { file, img, id } for the series list 

/* -----------------------------------------------------------
   3. DOM REFERENCES
   ----------------------------------------------------------- */

const $ = (id) => document.getElementById(id); // Simple helper for getElementById, $('id') instead of document.getElementById('id')

const viewport          = $('viewport');
const mainCanvas        = $('mainCanvas');
const uploadOverlay     = $('uploadOverlay');
const inferenceSpinner  = $('inferenceSpinner');
const spinnerLabel      = $('spinnerLabel');
const findingsContainer = $('findingsContainer');
const seriesList        = $('seriesList');
const fileInput         = $('fileInput');
const hfBadge           = $('hfBadge');

/* -----------------------------------------------------------
   4. CONNECTION STATUS
   Updates the header badge and info panel backend field.
   ----------------------------------------------------------- */

async function updateConnectionStatus() {
  hfBadge.textContent = "CHECKING...";
  hfBadge.className = "header-badge running";

  try {
    await fetch(CONFIG.hfSpaceUrl + "/", { mode: "no-cors", cache: "no-store" });
    hfBadge.textContent = "HF SPACE ONLINE";
    hfBadge.className = "header-badge online";
  } catch {
    hfBadge.textContent = "HF SPACE OFFLINE";
    hfBadge.className = "header-badge offline";
  }
}


/* -----------------------------------------------------------
   5. FILE UPLOAD HANDLING
   Click the overlay, click the toolbar button, or drag & drop.
   ----------------------------------------------------------- */

uploadOverlay.addEventListener('click', () => fileInput.click()); // Click main area
$('uploadToolBtn').addEventListener('click', () => fileInput.click()); // Toolbar button

// Native file input 
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
  fileInput.value = '';
});

/**
 * Main handler: loads image, updates metadata, draws on canvas,
 * adds to series list, triggers inference.
 */
function handleFile(file) {
  currentImageFile = file; // Store the active file

  // Update Image Details section of File Info panel
  $('infoFilename').textContent = file.name;
  $('infoFileSize').textContent = formatBytes(file.size);
  $('infoFileType').textContent = file.type || 'unknown';
  $('infoFileDate').textContent = new Date(file.lastModified).toISOString().slice(0, 16).replace('T', ' ');
  
  const reader = new FileReader(); // similar to with open(file) as f: in Python
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      currentImageObj = img;

      // Dimensional metadata
      $('infoDimensions').textContent  = `${img.naturalWidth} X ${img.naturalHeight}`;
      $('overlayFilename').textContent = file.name;
      $('overlayMeta').textContent     = `${img.naturalWidth} X ${img.naturalHeight}, ${formatBytes(file.size)}`;
      $('overlayDate').textContent     = new Date().toISOString().slice(0, 16).replace('T', ' ');

      // Show overlays
      ['overlayTL', 'overlayTR']
        .forEach((id) => $(id).style.display = '');

      addToSeriesList(file, img);
      drawImageOnCanvas(img);
      uploadOverlay.classList.add('has-image');
      runInference(file);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/* -----------------------------------------------------------
   6. SERIES LIST MANAGEMENT
   ----------------------------------------------------------- */

function addToSeriesList(file, img) {
  uploadedImages.push({ file, img, id: uploadedImages.length + 1 });

  seriesList.innerHTML = '';

  uploadedImages.forEach((s, idx) => {
    const isActive = idx === uploadedImages.length - 1;
    const card = document.createElement('div');
    card.className = `series-card${isActive ? ' active' : ''}`;

    const thumbCanvas = createThumbnail(s.img, 52, 52);

    card.innerHTML = `
      <div class="series-thumb"></div>
      <div class="series-meta">
        <div class="series-label">${s.file.name}</div>
        <div class="series-detail">${s.img.naturalWidth} X${s.img.naturalHeight}, ${formatBytes(s.file.size)}</div>
      </div>`;
    card.querySelector('.series-thumb').appendChild(thumbCanvas);

    card.addEventListener('click', () => {
      document.querySelectorAll('.series-card').forEach((c) => c.classList.remove('active'));
      card.classList.add('active');
      currentImageObj  = s.img;
      currentImageFile = s.file;
      drawImageOnCanvas(s.img);
      runInference(s.file);
    });

    seriesList.appendChild(card);
  });

  $('seriesCount').textContent = uploadedImages.length;
}

/** Draws a fitted thumbnail on a small canvas. */
function createThumbnail(img, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  const aspect = img.naturalWidth / img.naturalHeight;
  let dw = w, dh = h;
  if (aspect > 1) dh = w / aspect;
  else             dw = h * aspect;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  return canvas;
}

/* -----------------------------------------------------------
   7. CANVAS DRAWING
   ----------------------------------------------------------- */

function drawImageOnCanvas(img) {
  const rect = mainCanvas.parentElement.getBoundingClientRect();
  mainCanvas.width  = rect.width;
  mainCanvas.height = rect.height;
  const ctx = mainCanvas.getContext('2d');
  const w = mainCanvas.width;
  const h = mainCanvas.height;

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);

  const scale = Math.min(w / img.naturalWidth, h / img.naturalHeight) * 0.92;
  const dw = img.naturalWidth  * scale;
  const dh = img.naturalHeight * scale;

  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);

  // Radiological orientation markers
  ctx.font      = '600 14px "IBM Plex Mono"';
  ctx.fillStyle = 'rgba(196, 210, 224, 0.35)';
  ctx.textAlign = 'center';
  ctx.fillText('R', 24, h / 2);
  ctx.fillText('L', w - 24, h / 2);
}

function drawPlaceholderCanvas() {
  const rect = mainCanvas.parentElement.getBoundingClientRect();
  mainCanvas.width  = rect.width;
  mainCanvas.height = rect.height;
  const ctx = mainCanvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
}

/* -----------------------------------------------------------
   8. INFERENCE ENGINE
   ----------------------------------------------------------- */

async function runInference(file) {
  showSpinner('Running inference...');
  $('infoStatus').textContent = 'Running...';
  hfBadge.textContent         = 'ANALYZING...';
  hfBadge.className           = 'header-badge running';

  // Auto-switch to findings tab
  switchToTab('findings');

  const startTime = performance.now();

  try {
    const predictions = await callHuggingFaceAPI(file);

const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    $('infoLatency').textContent = `${elapsed}s`;
    $('infoStatus').textContent  = 'Complete';

    renderFindings(predictions);
    updateConnectionStatus();

  } catch (err) {
    console.error('Inference error:', err);
  } finally {
    hideSpinner();
  }
}

/**
 * Calls the Hugging Face Gradio Space API (v4+ / v5 format).
 *
 * Modern Gradio uses a three-step flow:
 *   1. POST file to /gradio_api/upload → get temp file ref
 *   2. POST to /call/predict with file ref → get event_id
 *   3. GET /call/predict/{event_id} → stream/read result
 */
async function callHuggingFaceAPI(file) {
  const base = CONFIG.hfSpaceUrl;

  // Step 1: Upload the file to Gradio's file server
  const formData = new FormData();
  formData.append('files', file);

  const uploadRes = await fetch(`${base}/gradio_api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!uploadRes.ok) {
    throw new Error(`Upload failed (${uploadRes.status}). Is the Space running?`);
  }

  const uploadData = await uploadRes.json();
  // uploadData is an array of file paths, e.g. ["/tmp/gradio/abc123/image.png"]
  const filePath = Array.isArray(uploadData) ? uploadData[0] : uploadData;

  // Step 2: Call the prediction endpoint - returns an event_id
  const callRes = await fetch(`${base}/gradio_api/call/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [{ path: filePath, meta: { _type: 'gradio.FileData' } }],
    }),
  });
  if (!callRes.ok) {
    throw new Error(`Predict call failed (${callRes.status}). Check Space logs.`);
  }

  const { event_id } = await callRes.json();
  if (!event_id) throw new Error('No event_id returned from /call/predict');

  // Step 3: Stream the result via SSE (Server-Sent Events)
  const resultRes = await fetch(`${base}/gradio_api/call/predict/${event_id}`);
  if (!resultRes.ok) {
    throw new Error(`Result fetch failed (${resultRes.status}).`);
  }

  const text = await resultRes.text();

  // Parse the SSE stream - look for the "complete" event's data line
  const lines = text.split('\n');
  let resultData = null;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('event: complete')) {
      // The data line follows immediately
      const dataLine = lines[i + 1];
      if (dataLine && dataLine.startsWith('data: ')) {
        resultData = JSON.parse(dataLine.slice(6));
        break;
      }
    }
  }

  // Check for errors in the stream
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('event: error')) {
      const dataLine = lines[i + 1];
      if (dataLine && dataLine.startsWith('data: ')) {
        const errMsg = JSON.parse(dataLine.slice(6));
        throw new Error(`Space error: ${errMsg}`);
      }
    }
  }

  if (!resultData || !resultData[0]) {
    throw new Error('Unexpected response format - no data in stream');
  }

  const labelData = resultData[0];

  // Gradio Label returns { label, confidences: [{label, confidence}] }
  if (labelData.confidences) {
    return labelData.confidences.map((c) => ({
      label: c.label,
      score: c.confidence,
    }));
  } else if (typeof labelData === 'object') {
    return Object.entries(labelData)
      .map(([label, score]) => ({ label, score }))
      .sort((a, b) => b.score - a.score);
  }

  throw new Error('Could not parse model predictions');
}

/* -----------------------------------------------------------
   9. FINDINGS RENDERER
   ----------------------------------------------------------- */

function renderFindings(predictions) {
  findingsContainer.innerHTML = '';

  predictions.forEach((f, i) => {
    const severity  = f.score >= 0.7 ? 'critical' : f.score >= 0.4 ? 'warning' : 'normal';
    const confClass = f.score >= 0.7 ? 'high'     : f.score >= 0.4 ? 'medium'  : 'low';
    const barColor  = severity === 'critical' ? 'var(--accent-red)'
                    : severity === 'warning'  ? 'var(--accent-amber)'
                    :                           'var(--accent-green)';

    const card = document.createElement('div');
    card.className = 'finding-card';
    card.style.animationDelay = `${i * 0.05}s`;

    card.innerHTML = `
      <div class="finding-header">
        <span class="finding-label">
          <span class="finding-dot ${severity}"></span>${f.label}
        </span>
        <span class="finding-confidence ${confClass}">
          ${(f.score * 100).toFixed(1)}%
        </span>
      </div>
      <div class="finding-bar-bg">
        <div class="finding-bar" style="width:${f.score * 100}%; background:${barColor};"></div>
      </div>`;

    findingsContainer.appendChild(card);
  });
}

/* -----------------------------------------------------------
   10. SPINNER HELPERS
   ----------------------------------------------------------- */

function showSpinner(text) {
  spinnerLabel.textContent = text || 'Running inference...';
  inferenceSpinner.classList.add('active');
}

function hideSpinner() {
  inferenceSpinner.classList.remove('active');
}

/* -----------------------------------------------------------
   11. TAB SWITCHING
   ----------------------------------------------------------- */

document.querySelectorAll('.info-tab').forEach((tab) => {
  tab.addEventListener('click', () => switchToTab(tab.dataset.tab));
});

function switchToTab(tabName) {
  document.querySelectorAll('.info-tab').forEach((t) => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
  $(`tab-${tabName}`).classList.add('active');
}

/* -----------------------------------------------------------
   12. TOOLBAR INTERACTIONS
   ----------------------------------------------------------- */

// Reset View
document.querySelector('[data-tool="reset"]').addEventListener('click', () => {
  if (currentImageObj) drawImageOnCanvas(currentImageObj);
});

/* -----------------------------------------------------------
   13. UTILITIES
   ----------------------------------------------------------- */

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k     = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i     = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/* -----------------------------------------------------------
   14. INITIALIZATION
   ----------------------------------------------------------- */

window.addEventListener('resize', () => {
  if (currentImageObj) drawImageOnCanvas(currentImageObj);
  else                 drawPlaceholderCanvas();
});

updateConnectionStatus();
drawPlaceholderCanvas();

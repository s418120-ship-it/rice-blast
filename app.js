// ====== 可依需求調整的設定 ======
const MODEL_URL = "models/best.onnx";
const LABELS_URL = "labels.json";
const MODEL_INPUT_SIZE = 640; // 必須與 YOLOv8 匯出 ONNX 時的 imgsz 一致
const IOU_THRESHOLD = 0.45;
let CONF_THRESHOLD = 0.25;

// ====== DOM ======
const modelStatus = document.getElementById("modelStatus");
const imageInput = document.getElementById("imageInput");
const cameraButton = document.getElementById("cameraButton");
const stopCameraButton = document.getElementById("stopCameraButton");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const video = document.getElementById("video");
const sourceImage = document.getElementById("sourceImage");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const summary = document.getElementById("summary");
const counts = document.getElementById("counts");
const detectionsBox = document.getElementById("detections");

let session = null;
let labels = [];
let stream = null;
let animationId = null;
let isRunningCamera = false;
let lastInferTime = 0;
const INFER_INTERVAL_MS = 120; // 數值越小越即時，但手機負擔越重

confSlider.addEventListener("input", () => {
  CONF_THRESHOLD = Number(confSlider.value);
  confValue.textContent = CONF_THRESHOLD.toFixed(2);
});

async function init() {
  try {
    modelStatus.textContent = "模型載入中...";

    labels = await fetch(LABELS_URL).then((r) => {
      if (!r.ok) throw new Error("找不到 labels.json");
      return r.json();
    });

    // wasm 適合 GitHub Pages；若瀏覽器支援，onnxruntime-web 會自動使用可用能力。
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    modelStatus.textContent = "模型已載入";
    modelStatus.className = "status ready";
  } catch (err) {
    console.error(err);
    modelStatus.textContent = "模型載入失敗，請確認 best.onnx 路徑";
    modelStatus.className = "status error";
    summary.textContent = err.message;
  }
}

imageInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  stopCamera();

  const url = URL.createObjectURL(file);
  sourceImage.onload = async () => {
    URL.revokeObjectURL(url);
    video.style.display = "none";
    sourceImage.style.display = "none";
    await detectAndDraw(sourceImage);
  };
  sourceImage.src = url;
});

cameraButton.addEventListener("click", startCamera);
stopCameraButton.addEventListener("click", stopCamera);

async function startCamera() {
  if (!session) {
    alert("模型尚未載入完成");
    return;
  }

  stopCamera();

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;
    video.style.display = "none";
    sourceImage.style.display = "none";
    await video.play();

    isRunningCamera = true;
    cameraButton.disabled = true;
    stopCameraButton.disabled = false;
    runCameraLoop();
  } catch (err) {
    console.error(err);
    alert("無法開啟鏡頭。請確認網頁為 HTTPS，並允許瀏覽器使用相機。" + err.message);
  }
}

function stopCamera() {
  isRunningCamera = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  cameraButton.disabled = false;
  stopCameraButton.disabled = true;
}

async function runCameraLoop(time = 0) {
  if (!isRunningCamera) return;

  if (time - lastInferTime >= INFER_INTERVAL_MS) {
    lastInferTime = time;
    await detectAndDraw(video);
  }

  animationId = requestAnimationFrame(runCameraLoop);
}

async function detectAndDraw(source) {
  if (!session) return;

  const originalWidth = source.videoWidth || source.naturalWidth || source.width;
  const originalHeight = source.videoHeight || source.naturalHeight || source.height;
  if (!originalWidth || !originalHeight) return;

  canvas.width = originalWidth;
  canvas.height = originalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

  const { tensor, scale, padX, padY } = preprocess(source, originalWidth, originalHeight);
  const feeds = { [session.inputNames[0]]: tensor };
  const outputMap = await session.run(feeds);
  const output = outputMap[session.outputNames[0]];

  const detections = postprocess(output, originalWidth, originalHeight, scale, padX, padY);
  drawDetections(detections);
  renderResults(detections);
}

function preprocess(source, originalWidth, originalHeight) {
  const inputCanvas = document.createElement("canvas");
  inputCanvas.width = MODEL_INPUT_SIZE;
  inputCanvas.height = MODEL_INPUT_SIZE;
  const inputCtx = inputCanvas.getContext("2d");

  inputCtx.fillStyle = "rgb(114,114,114)";
  inputCtx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

  const scale = Math.min(MODEL_INPUT_SIZE / originalWidth, MODEL_INPUT_SIZE / originalHeight);
  const newWidth = Math.round(originalWidth * scale);
  const newHeight = Math.round(originalHeight * scale);
  const padX = Math.floor((MODEL_INPUT_SIZE - newWidth) / 2);
  const padY = Math.floor((MODEL_INPUT_SIZE - newHeight) / 2);

  inputCtx.drawImage(source, padX, padY, newWidth, newHeight);
  const imageData = inputCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;

  const input = new Float32Array(1 * 3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  for (let i = 0; i < pixels; i++) {
    input[i] = imageData[i * 4] / 255; // R
    input[i + pixels] = imageData[i * 4 + 1] / 255; // G
    input[i + pixels * 2] = imageData[i * 4 + 2] / 255; // B
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    scale,
    padX,
    padY,
  };
}

function postprocess(output, originalWidth, originalHeight, scale, padX, padY) {
  const data = output.data;
  const dims = output.dims;

  // YOLOv8 ONNX 常見輸出：[1, 4 + classCount, 8400]
  // 有些模型可能是：[1, 8400, 4 + classCount]
  let rows, cols, transposed;
  if (dims.length === 3 && dims[1] < dims[2]) {
    cols = dims[1];
    rows = dims[2];
    transposed = true;
  } else if (dims.length === 3) {
    rows = dims[1];
    cols = dims[2];
    transposed = false;
  } else {
    throw new Error("不支援的模型輸出維度：" + dims.join("x"));
  }

  const classCount = cols - 4;
  const boxes = [];

  for (let i = 0; i < rows; i++) {
    let cx, cy, w, h;
    if (transposed) {
      cx = data[0 * rows + i];
      cy = data[1 * rows + i];
      w = data[2 * rows + i];
      h = data[3 * rows + i];
    } else {
      const offset = i * cols;
      cx = data[offset];
      cy = data[offset + 1];
      w = data[offset + 2];
      h = data[offset + 3];
    }

    let bestScore = -Infinity;
    let classId = 0;

    for (let c = 0; c < classCount; c++) {
      const score = transposed ? data[(4 + c) * rows + i] : data[i * cols + 4 + c];
      if (score > bestScore) {
        bestScore = score;
        classId = c;
      }
    }

    if (bestScore < CONF_THRESHOLD) continue;

    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    // 還原 letterbox 前的原圖座標
    x1 = (x1 - padX) / scale;
    y1 = (y1 - padY) / scale;
    x2 = (x2 - padX) / scale;
    y2 = (y2 - padY) / scale;

    x1 = clamp(x1, 0, originalWidth);
    y1 = clamp(y1, 0, originalHeight);
    x2 = clamp(x2, 0, originalWidth);
    y2 = clamp(y2, 0, originalHeight);

    boxes.push({
      x1,
      y1,
      x2,
      y2,
      width: x2 - x1,
      height: y2 - y1,
      score: bestScore,
      classId,
      label: labels[classId] || `class_${classId}`,
    });
  }

  return nonMaxSuppression(boxes, IOU_THRESHOLD);
}

function nonMaxSuppression(boxes, iouThreshold) {
  const sorted = boxes.sort((a, b) => b.score - a.score);
  const selected = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
    selected.push(current);

    for (let i = sorted.length - 1; i >= 0; i--) {
      const sameClass = sorted[i].classId === current.classId;
      if (sameClass && iou(current, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.width) * Math.max(0, a.height);
  const areaB = Math.max(0, b.width) * Math.max(0, b.height);
  const union = areaA + areaB - intersection;

  return union <= 0 ? 0 : intersection / union;
}

function drawDetections(detections) {
  const lineWidth = Math.max(2, Math.round(canvas.width / 300));
  ctx.lineWidth = lineWidth;
  ctx.font = `${Math.max(16, Math.round(canvas.width / 45))}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  detections.forEach((det) => {
    const text = `${det.label} ${(det.score * 100).toFixed(1)}%`;
    const textMetrics = ctx.measureText(text);
    const textHeight = Math.max(22, Math.round(canvas.width / 32));

    ctx.strokeStyle = "#00ff88";
    ctx.fillStyle = "#00ff88";
    ctx.strokeRect(det.x1, det.y1, det.width, det.height);

    const labelY = det.y1 - textHeight < 0 ? det.y1 : det.y1 - textHeight;
    ctx.fillRect(det.x1, labelY, textMetrics.width + 12, textHeight);
    ctx.fillStyle = "#001b10";
    ctx.fillText(text, det.x1 + 6, labelY + 3);
  });

  drawCountOverlay(detections);
}

function drawCountOverlay(detections) {
  const countMap = countByClass(detections);
  const lines = [`總數：${detections.length}`];
  Object.entries(countMap).forEach(([label, count]) => lines.push(`${label}：${count}`));

  ctx.font = `${Math.max(18, Math.round(canvas.width / 42))}px system-ui, sans-serif`;
  const padding = 12;
  const lineHeight = Math.max(24, Math.round(canvas.width / 32));
  const boxWidth = Math.max(...lines.map((line) => ctx.measureText(line).width)) + padding * 2;
  const boxHeight = lines.length * lineHeight + padding;

  ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
  ctx.fillRect(10, 10, boxWidth, boxHeight);
  ctx.fillStyle = "#ffffff";

  lines.forEach((line, index) => {
    ctx.fillText(line, 10 + padding, 10 + padding / 2 + index * lineHeight);
  });
}

function renderResults(detections) {
  const countMap = countByClass(detections);
  summary.textContent = `共偵測到 ${detections.length} 個物件`;

  counts.innerHTML = Object.keys(countMap).length
    ? Object.entries(countMap)
        .map(([label, count]) => `<div class="countItem"><strong>${escapeHtml(label)}</strong>：${count}</div>`)
        .join("")
    : `<div class="countItem">沒有超過門檻的物件</div>`;

  detectionsBox.innerHTML = detections.length
    ? detections
        .map(
          (det, idx) => `
          <div class="detItem">
            <strong>${idx + 1}. ${escapeHtml(det.label)}</strong><br />
            <small>信心值：${(det.score * 100).toFixed(1)}%；座標：(${Math.round(det.x1)}, ${Math.round(det.y1)})</small>
          </div>`
        )
        .join("")
    : "";
}

function countByClass(detections) {
  return detections.reduce((acc, det) => {
    acc[det.label] = (acc[det.label] || 0) + 1;
    return acc;
  }, {});
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

init();

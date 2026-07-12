(() => {
  "use strict";

  const LIMITS = {
    fileBytes: 20 * 1024 * 1024,
    batchFiles: 20,
    batchBytes: 200 * 1024 * 1024,
  };
  const HISTORY_KEY = "agile-agent-session-history-v1";
  const RESULT_CACHE_LIMIT = 10;
  const SENSOR_LABELS = { ir: "红外", sar: "SAR" };
  const SCENE_LABELS = { air: "空域", forest: "林地", sea: "海域", urban: "城市场景" };
  const CLASS_LABELS = {
    soldier: "人员",
    small_aircraft: "小型飞行器",
    warship: "舰船",
    tank: "坦克",
  };
  const VALID_TYPES = new Set([
    "image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp", "image/tiff", "image/x-tiff",
  ]);
  const VALID_EXTENSIONS = new Set(["png", "jpg", "jpeg", "bmp", "tif", "tiff"]);

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const state = {
    singleFile: null,
    singlePreviewUrl: null,
    singleResult: null,
    batchFiles: [],
    batchResult: null,
    resultCache: new Map(),
    history: readHistory(),
    recognitionMode: "standard",
    capabilities: null,
  };

  function icon(name) {
    return `<svg aria-hidden="true"><use href="/assets/icons.svg#${name}"></use></svg>`;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function classLabel(value) {
    return CLASS_LABELS[value] || value || "未知";
  }

  function sensorLabel(value) {
    return SENSOR_LABELS[value] || value || "未知";
  }

  function sceneLabel(value) {
    return SCENE_LABELS[value] || value || "未知";
  }

  function validateFile(file) {
    const extension = (file.name.split(".").pop() || "").toLowerCase();
    if (!VALID_TYPES.has(file.type) && !VALID_EXTENSIONS.has(extension)) {
      throw new Error(`不支持的图像格式：${file.name}`);
    }
    if (!file.size) throw new Error(`图像为空：${file.name}`);
    if (file.size > LIMITS.fileBytes) throw new Error(`单张图像不能超过20MB：${file.name}`);
  }

  function imageDimensions(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => reject(new Error("浏览器无法读取这张图像。"));
      image.src = url;
    });
  }

  function showToast(message, type = "success") {
    const toast = document.createElement("div");
    toast.className = `toast${type === "error" ? " is-error" : ""}`;
    toast.innerHTML = icon(type === "error" ? "alert" : "check");
    const text = document.createElement("span");
    text.textContent = message;
    toast.appendChild(text);
    $("#toastRegion").appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function setLoading(visible, title = "正在分析图像", message = "正在准备识别") {
    $("#loadingTitle").textContent = title;
    $("#loadingMessage").textContent = message;
    $("#loadingOverlay").classList.toggle("is-hidden", !visible);
    document.body.classList.toggle("is-locked", visible);
  }

  async function responseError(response) {
    try {
      const payload = await response.json();
      return payload.error || `请求失败（${response.status}）`;
    } catch (_error) {
      return `请求失败（${response.status}）`;
    }
  }

  function switchView(viewName) {
    $$(".view").forEach((view) => view.classList.toggle("is-active", view.id === `view-${viewName}`));
    $$("[data-view]").forEach((button) => {
      const active = button.dataset.view === viewName;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-current", active ? "page" : "false");
    });
    $(".main-nav").classList.remove("is-open");
    $("#mobileMenu").setAttribute("aria-expanded", "false");
    $("#mobileMenu").setAttribute("aria-label", "打开导航");
    if (viewName === "history") renderHistory();
    window.location.hash = viewName;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function refreshHealth() {
    const node = $("#serviceStatus");
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error("health check failed");
      const payload = await response.json();
      const busy = Boolean(payload.queue && (payload.queue.active || payload.queue.waiting));
      node.className = `service-status ${busy ? "is-busy" : "is-ready"}`;
      node.querySelector("b").textContent = busy ? "正在处理" : "服务就绪";
    } catch (_error) {
      node.className = "service-status is-error";
      node.querySelector("b").textContent = "暂未连接";
    }
  }

  async function loadCapabilities() {
    const incrementalButton = $("[data-mode='incremental']");
    const select = $("#incrementalProtocol");
    try {
      const response = await fetch("/api/capabilities", { cache: "no-store" });
      if (!response.ok) throw new Error("capabilities unavailable");
      const payload = await response.json();
      state.capabilities = payload;
      select.replaceChildren();
      payload.protocols.forEach((protocol) => {
        const option = document.createElement("option");
        option.value = protocol.id;
        option.disabled = !protocol.available;
        option.textContent = protocol.available
          ? `${classLabel(protocol.new_class)}增量模型`
          : `${classLabel(protocol.new_class)}（优化中）`;
        select.appendChild(option);
      });
      const firstAvailable = payload.protocols.find((protocol) => protocol.available);
      incrementalButton.disabled = !payload.incremental_enabled || !firstAvailable;
      if (firstAvailable) select.value = firstAvailable.id;
    } catch (_error) {
      incrementalButton.disabled = true;
    }
  }

  function setRecognitionMode(mode) {
    const incrementalButton = $("[data-mode='incremental']");
    if (mode === "incremental" && incrementalButton.disabled) return;
    state.recognitionMode = mode;
    $$("[data-mode]").forEach((button) => button.classList.toggle("is-active", button.dataset.mode === mode));
    $("#incrementalProtocol").classList.toggle("is-hidden", mode !== "incremental");
  }

  function setWorkflow(step) {
    $$("[data-step]").forEach((node) => {
      const value = Number(node.dataset.step);
      node.classList.toggle("is-active", value === step);
      node.classList.toggle("is-complete", value < step);
    });
  }

  async function selectSingle(file) {
    try {
      validateFile(file);
      clearSingle(false);
      state.singleFile = file;
      state.singlePreviewUrl = URL.createObjectURL(file);
      $("#previewImage").src = state.singlePreviewUrl;
      const dimensions = await imageDimensions(state.singlePreviewUrl).catch(() => ({ width: null, height: null }));
      if (state.singleFile !== file) return;
      $("#singleFilename").textContent = file.name;
      $("#singleMeta").textContent = dimensions.width
        ? `${dimensions.width} × ${dimensions.height} · ${formatBytes(file.size)}`
        : `服务端读取尺寸 · ${formatBytes(file.size)}`;
      $("#inputPreview").classList.toggle("is-unavailable", !dimensions.width);
      $("#singleDropzone").classList.add("is-hidden");
      $("#inputPreview").classList.remove("is-hidden");
      $("#detectButton").disabled = false;
      $("#clearSingle").disabled = false;
      setWorkflow(2);
    } catch (error) {
      clearSingle(false);
      showToast(error.message || "无法读取图像。", "error");
    }
  }

  function clearSingle(notify = true) {
    if (state.singlePreviewUrl) URL.revokeObjectURL(state.singlePreviewUrl);
    state.singleFile = null;
    state.singlePreviewUrl = null;
    state.singleResult = null;
    $("#singleFile").value = "";
    $("#previewImage").removeAttribute("src");
    $("#resultImage").removeAttribute("src");
    $("#singleDropzone").classList.remove("is-hidden", "is-dragging");
    $("#inputPreview").classList.add("is-hidden");
    $("#inputPreview").classList.remove("is-unavailable");
    $("#detectButton").disabled = true;
    $("#clearSingle").disabled = true;
    $("#resultEmpty").classList.remove("is-hidden");
    $("#resultCanvas").classList.add("is-hidden");
    $("#resultDetails").classList.add("is-hidden");
    $("#resultState").textContent = "等待输入";
    $("#resultState").className = "result-state";
    setWorkflow(1);
    if (notify) showToast("当前检测任务已清除。", "success");
  }

  function renderSingleResult(result) {
    const context = result.context || {};
    const detections = Array.isArray(result.detections) ? result.detections : [];
    const counts = result.class_counts || {};
    $("#resultImage").src = `data:image/png;base64,${result.annotated_base64}`;
    $("#overlaySensor").textContent = sensorLabel(context.sensor);
    $("#overlayScene").textContent = sceneLabel(context.scene);
    $("#overlayCount").textContent = `${result.detection_count || 0} 个目标`;
    $("#resultEmpty").classList.add("is-hidden");
    $("#resultCanvas").classList.remove("is-hidden");
    $("#resultDetails").classList.remove("is-hidden");
    $("#resultState").textContent = "分析完成";
    $("#resultState").className = "result-state is-complete";
    renderCollaboration(result);

    const summary = [
      ["传感器", sensorLabel(context.sensor), `${Math.round((context.sensor_confidence || 0) * 100)}% 置信度`],
      ["场景", sceneLabel(context.scene), `${Math.round((context.scene_confidence || 0) * 100)}% 置信度`],
      ["检测目标", String(result.detection_count || 0), "已完成定位"],
      ["纯推理时间", `${Number(result.inference_ms || 0).toFixed(1)} ms`, "模型前向计算"],
      ["系统总用时", `${Number(result.system_total_ms || 0).toFixed(1)} ms`, "完整处理链路"],
    ];
    const summaryNode = $("#resultSummary");
    summaryNode.replaceChildren();
    summary.forEach(([label, value, hint], index) => {
      const item = document.createElement("div");
      item.className = "summary-item";
      const labelNode = document.createElement("span");
      labelNode.textContent = label;
      const valueNode = document.createElement("strong");
      valueNode.textContent = value;
      const hintNode = document.createElement("small");
      hintNode.textContent = hint;
      item.append(labelNode, valueNode, hintNode);
      if (index === 2 && Object.keys(counts).length) {
        const classes = document.createElement("div");
        classes.className = "class-summary";
        Object.entries(counts).forEach(([name, count]) => {
          const chip = document.createElement("b");
          chip.textContent = `${classLabel(name)} ${count}`;
          classes.appendChild(chip);
        });
        item.appendChild(classes);
      }
      summaryNode.appendChild(item);
    });

    renderDetectionRows($("#detectionRows"), detections);
    $("#detectionTotal").textContent = `${detections.length} 条结果`;
    setWorkflow(3);
  }

  function renderDetectionRows(rows, detections) {
    rows.replaceChildren();
    if (!detections.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 3;
      cell.className = "table-empty";
      cell.textContent = "当前置信度下未检测到目标";
      row.appendChild(cell);
      rows.appendChild(row);
    } else {
      detections.forEach((detection) => {
        const row = document.createElement("tr");
        const classCell = document.createElement("td");
        classCell.textContent = classLabel(detection.class_name);
        const confidenceCell = document.createElement("td");
        const bar = document.createElement("span");
        bar.className = "confidence-bar";
        const fill = document.createElement("i");
        fill.style.width = `${Math.max(0, Math.min(100, Number(detection.confidence || 0) * 100))}%`;
        bar.appendChild(fill);
        confidenceCell.append(bar, document.createTextNode(`${(Number(detection.confidence || 0) * 100).toFixed(1)}%`));
        const boxCell = document.createElement("td");
        boxCell.textContent = (detection.xyxy || []).map((value) => Number(value).toFixed(1)).join(", ");
        row.append(classCell, confidenceCell, boxCell);
        rows.appendChild(row);
      });
    }
  }

  function renderCollaboration(result) {
    const context = result.context || {};
    const agent = result.agent || {};
    const protocol = agent.protocol || null;
    $("#collaborationMode").textContent = protocol ? "增量协同" : "标准识别";
    const flow = $("#collaborationFlow");
    flow.replaceChildren();
    const steps = protocol
      ? [
          ["01 场景认知", `${sensorLabel(context.sensor)} · ${sceneLabel(context.scene)}`, "理解输入模态与任务场景"],
          ["02 历史能力", "冻结旧类检测器", "保持原有类别识别能力"],
          ["03 快速学习", `${classLabel(protocol.new_class)}增量模型`, `New-mAP50 ${Number(protocol.new_map50).toFixed(3)} · KRR ${Number(protocol.krr).toFixed(3)}`],
        ]
      : [
          ["01 场景认知", `${sensorLabel(context.sensor)} · ${sceneLabel(context.scene)}`, "理解输入模态与任务场景"],
          ["02 策略选择", "统一多模态识别", "根据场景选择当前识别策略"],
          ["03 目标检测", `${result.detection_count || 0} 个目标`, "完成类别判断与目标定位"],
        ];
    steps.forEach(([stage, title, detail]) => {
      const node = document.createElement("div");
      node.className = "collaboration-step";
      const stageNode = document.createElement("span");
      stageNode.textContent = stage;
      const titleNode = document.createElement("strong");
      titleNode.textContent = title;
      const detailNode = document.createElement("small");
      detailNode.textContent = detail;
      node.append(stageNode, titleNode, detailNode);
      flow.appendChild(node);
    });
  }

  async function detectSingle() {
    if (!state.singleFile) return;
    $("#detectButton").disabled = true;
    $("#resultState").textContent = "正在分析";
    $("#resultState").className = "result-state is-busy";
    setLoading(true, "正在分析图像", "正在完成场景与目标分析");
    try {
      const form = new FormData();
      form.append("file", state.singleFile, state.singleFile.name);
      form.append("confidence", $("#confidence").value);
      if (state.recognitionMode === "incremental") {
        form.append("incremental_protocol", $("#incrementalProtocol").value);
      }
      const response = await fetch("/api/detect", { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json();
      state.singleResult = result;
      renderSingleResult(result);
      const resultKey = rememberResult("single", result);
      addHistory({
        type: "single",
        filename: result.filename || state.singleFile.name,
        sensor: result.context && result.context.sensor,
        scene: result.context && result.context.scene,
        detectionCount: Number(result.detection_count || 0),
        inferenceMs: Number(result.inference_ms || 0),
        systemTotalMs: Number(result.system_total_ms || 0),
        mode: result.agent && result.agent.mode,
        resultKey,
        createdAt: new Date().toISOString(),
      });
      showToast(`检测完成，共发现${result.detection_count || 0}个目标。`);
    } catch (error) {
      $("#resultState").textContent = "分析失败";
      $("#resultState").className = "result-state is-error";
      showToast(error.message || "检测失败，请稍后重试。", "error");
    } finally {
      setLoading(false);
      $("#detectButton").disabled = !state.singleFile;
      refreshHealth();
    }
  }

  function base64Blob(base64, type) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return new Blob([bytes], { type });
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function downloadSingleImage() {
    if (!state.singleResult) return;
    downloadBlob(base64Blob(state.singleResult.annotated_base64, "image/png"), `agile-agent-${Date.now()}.png`);
  }

  function downloadSingleJson() {
    if (!state.singleResult) return;
    const payload = { ...state.singleResult };
    delete payload.annotated_base64;
    downloadBlob(new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json;charset=utf-8" }), `agile-agent-${Date.now()}.json`);
  }

  async function selectBatch(fileList) {
    try {
      const files = Array.from(fileList);
      if (!files.length) return;
      if (files.length > LIMITS.batchFiles) throw new Error("单批最多选择20张图像。");
      const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
      if (totalBytes > LIMITS.batchBytes) throw new Error("单批图像总大小不能超过200MB。");
      files.forEach(validateFile);
      clearBatch(false);
      state.batchFiles = files;
      renderBatchQueue();
    } catch (error) {
      clearBatch(false);
      showToast(error.message || "无法读取批量图像。", "error");
    }
  }

  function renderBatchQueue() {
    const queue = $("#batchQueue");
    queue.replaceChildren();
    state.batchFiles.forEach((file) => {
      const row = document.createElement("div");
      row.className = "queue-row";
      row.innerHTML = icon("image");
      const name = document.createElement("div");
      const strong = document.createElement("strong");
      strong.textContent = file.name;
      const small = document.createElement("small");
      small.textContent = formatBytes(file.size);
      name.append(strong, small);
      row.append(name);
      queue.appendChild(row);
    });
    const totalBytes = state.batchFiles.reduce((sum, file) => sum + file.size, 0);
    $("#batchCount").textContent = `${state.batchFiles.length}张`;
    $("#batchSize").textContent = formatBytes(totalBytes);
    $("#batchToolbar").classList.toggle("is-hidden", !state.batchFiles.length);
    $("#batchButton").disabled = !state.batchFiles.length;
    $("#batchResult").classList.add("is-hidden");
    $("#batchGallery").classList.add("is-hidden");
  }

  function clearBatch(notify = true) {
    state.batchFiles = [];
    state.batchResult = null;
    $("#batchFiles").value = "";
    $("#batchQueue").replaceChildren();
    $("#batchToolbar").classList.add("is-hidden");
    $("#batchResult").classList.add("is-hidden");
    $("#batchGallery").classList.add("is-hidden");
    $("#batchButton").disabled = true;
    if (notify) showToast("批量任务列表已清空。", "success");
  }

  async function detectBatch() {
    if (!state.batchFiles.length) return;
    $("#batchButton").disabled = true;
    setLoading(true, "正在执行批量检测", `正在依次处理${state.batchFiles.length}张图像`);
    try {
      const form = new FormData();
      state.batchFiles.forEach((file) => form.append("files", file, file.name));
      form.append("confidence", $("#batchConfidence").value);
      const response = await fetch("/api/batch", { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const payload = await response.json();
      state.batchResult = payload;
      const imageCount = Number(payload.image_count || state.batchFiles.length);
      const detectionCount = Number(payload.detection_count || 0);
      const inferenceMs = Number(payload.inference_ms || 0);
      const systemTotalMs = Number(payload.system_total_ms || 0);
      $("#batchResultText").textContent = `${imageCount}张图像 · ${detectionCount}个目标 · 纯推理 ${inferenceMs.toFixed(1)} ms · 总用时 ${systemTotalMs.toFixed(1)} ms`;
      $("#batchResult").classList.remove("is-hidden");
      renderBatchResults(payload);
      const resultKey = rememberResult("batch", payload);
      addHistory({
        type: "batch",
        filename: `批量任务（${imageCount}张）`,
        sensor: "mixed",
        scene: "mixed",
        detectionCount,
        inferenceMs,
        systemTotalMs,
        resultKey,
        createdAt: new Date().toISOString(),
      });
      showToast(`批量检测完成，共发现${detectionCount}个目标。`);
    } catch (error) {
      showToast(error.message || "批量检测失败，请稍后重试。", "error");
    } finally {
      setLoading(false);
      $("#batchButton").disabled = !state.batchFiles.length;
      refreshHealth();
    }
  }

  function renderBatchResults(payload) {
    const results = Array.isArray(payload.results) ? payload.results : [];
    const list = $("#batchPreviewList");
    list.replaceChildren();
    results.forEach((result, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "batch-preview-item";
      button.dataset.index = String(index);
      const image = document.createElement("img");
      image.src = result.preview_url;
      image.alt = "";
      image.loading = "lazy";
      const label = document.createElement("span");
      const name = document.createElement("strong");
      name.textContent = result.filename || `结果 ${index + 1}`;
      const detail = document.createElement("small");
      detail.textContent = `${result.detection_count || 0} 个目标`;
      label.append(name, detail);
      button.append(image, label);
      button.addEventListener("click", () => selectBatchResult(index));
      list.appendChild(button);
    });
    $("#batchGallery").classList.toggle("is-hidden", !results.length);
    if (results.length) selectBatchResult(0);
  }

  function selectBatchResult(index) {
    const results = state.batchResult && Array.isArray(state.batchResult.results) ? state.batchResult.results : [];
    const result = results[index];
    if (!result) return;
    $$(".batch-preview-item").forEach((button) => button.classList.toggle("is-active", Number(button.dataset.index) === index));
    $("#batchPreviewImage").src = result.preview_url;
    $("#batchPreviewName").textContent = result.filename || `结果 ${index + 1}`;
    $("#batchPreviewPosition").textContent = `${index + 1} / ${results.length}`;
    const context = result.context || {};
    const metrics = [
      ["传感器", sensorLabel(context.sensor), `${Math.round((context.sensor_confidence || 0) * 100)}% 置信度`],
      ["场景", sceneLabel(context.scene), `${Math.round((context.scene_confidence || 0) * 100)}% 置信度`],
      ["检测目标", String(result.detection_count || 0), "已完成定位"],
      ["纯推理时间", `${Number(result.inference_ms || 0).toFixed(1)} ms`, "模型前向计算"],
    ];
    const summary = $("#batchPreviewSummary");
    summary.replaceChildren();
    metrics.forEach(([label, value, hint]) => {
      const item = document.createElement("div");
      item.className = "batch-detail-item";
      const labelNode = document.createElement("span");
      labelNode.textContent = label;
      const valueNode = document.createElement("strong");
      valueNode.textContent = value;
      const hintNode = document.createElement("small");
      hintNode.textContent = hint;
      item.append(labelNode, valueNode, hintNode);
      summary.appendChild(item);
    });
    const classes = $("#batchPreviewClasses");
    classes.replaceChildren();
    Object.entries(result.class_counts || {}).forEach(([name, count]) => {
      const chip = document.createElement("b");
      chip.textContent = `${classLabel(name)} ${count}`;
      classes.appendChild(chip);
    });
    if (!classes.children.length) {
      const empty = document.createElement("span");
      empty.textContent = "未检测到目标";
      classes.appendChild(empty);
    }
    const detections = Array.isArray(result.detections) ? result.detections : [];
    renderDetectionRows($("#batchDetectionRows"), detections);
    $("#batchDetectionTotal").textContent = `${detections.length} 条结果`;
  }

  function readHistory() {
    try {
      const payload = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(payload) ? payload.slice(0, 20) : [];
    } catch (_error) {
      return [];
    }
  }

  function rememberResult(type, payload) {
    const identity = type === "batch" ? payload.batch_id : payload.filename;
    const key = `${type}:${identity || "result"}:${Date.now()}`;
    state.resultCache.set(key, { type, payload });
    while (state.resultCache.size > RESULT_CACHE_LIMIT) {
      state.resultCache.delete(state.resultCache.keys().next().value);
    }
    return key;
  }

  function openHistoryItem(item) {
    const cached = state.resultCache.get(item.resultKey);
    if (!cached) {
      showToast("该记录的完整详情已过期，请重新执行检测。", "error");
      return;
    }
    if (cached.type === "single") {
      state.singleResult = cached.payload;
      switchView("detect");
      renderSingleResult(cached.payload);
      window.requestAnimationFrame(() => $("#resultDetails").scrollIntoView({ behavior: "smooth", block: "start" }));
      return;
    }
    state.batchResult = cached.payload;
    const payload = cached.payload;
    $("#batchResultText").textContent = `${Number(payload.image_count || 0)}张图像 · ${Number(payload.detection_count || 0)}个目标 · 纯推理 ${Number(payload.inference_ms || 0).toFixed(1)} ms · 总用时 ${Number(payload.system_total_ms || 0).toFixed(1)} ms`;
    $("#batchResult").classList.remove("is-hidden");
    renderBatchResults(payload);
    switchView("batch");
    window.requestAnimationFrame(() => $("#batchGallery").scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  function addHistory(item) {
    state.history = [item, ...state.history].slice(0, 20);
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
    renderHistory();
  }

  function clearHistory() {
    state.history = [];
    state.resultCache.clear();
    sessionStorage.removeItem(HISTORY_KEY);
    renderHistory();
    showToast("当前会话记录已清空。", "success");
  }

  function renderHistory() {
    const count = state.history.length;
    const images = state.history.reduce((sum, item) => sum + (item.type === "batch" ? Number((item.filename.match(/\d+/) || [1])[0]) : 1), 0);
    const detections = state.history.reduce((sum, item) => sum + Number(item.detectionCount || 0), 0);
    const stats = $("#historyStats");
    stats.replaceChildren();
    [["会话任务", count], ["处理图像", images], ["累计目标", detections]].forEach(([label, value]) => {
      const node = document.createElement("div");
      node.className = "history-stat";
      const labelNode = document.createElement("span");
      labelNode.textContent = label;
      const valueNode = document.createElement("strong");
      valueNode.textContent = String(value);
      node.append(labelNode, valueNode);
      stats.appendChild(node);
    });

    const list = $("#historyList");
    list.replaceChildren();
    if (!count) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.innerHTML = `${icon("history")}<div><strong>当前会话还没有任务</strong><small>完成检测后，轻量摘要会显示在这里</small></div>`;
      list.appendChild(empty);
      return;
    }
    state.history.forEach((item) => {
      const available = state.resultCache.has(item.resultKey);
      const row = document.createElement("button");
      row.type = "button";
      row.className = "history-row";
      row.classList.toggle("is-available", available);
      row.classList.toggle("is-expired", !available);
      row.setAttribute("aria-label", `${item.filename || "未命名任务"}，${available ? "查看详情" : "详情已过期"}`);
      row.innerHTML = icon(item.type === "batch" ? "layers" : "image");
      const identity = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = item.filename || "未命名任务";
      const detail = document.createElement("small");
      detail.textContent = `${new Date(item.createdAt).toLocaleString("zh-CN", { hour12: false })} · ${available ? "查看详情" : "详情已过期"}`;
      identity.append(name, detail);
      const context = document.createElement("span");
      context.className = "history-cell";
      context.textContent = item.sensor === "mixed" ? "混合输入" : `${sensorLabel(item.sensor)} / ${sceneLabel(item.scene)}`;
      const targets = document.createElement("span");
      targets.className = "history-cell";
      targets.textContent = `${item.detectionCount || 0} 个目标`;
      const elapsed = document.createElement("span");
      elapsed.className = "history-cell";
      elapsed.textContent = item.inferenceMs == null
        ? "耗时数据暂无"
        : `推理 ${Number(item.inferenceMs).toFixed(1)} / 总用时 ${Number(item.systemTotalMs || 0).toFixed(1)} ms`;
      row.append(identity, context, targets, elapsed);
      row.addEventListener("click", () => openHistoryItem(item));
      list.appendChild(row);
    });
  }

  function bindDropzone(node, handler) {
    ["dragenter", "dragover"].forEach((eventName) => node.addEventListener(eventName, (event) => {
      event.preventDefault();
      node.classList.add("is-dragging");
    }));
    ["dragleave", "drop"].forEach((eventName) => node.addEventListener(eventName, (event) => {
      event.preventDefault();
      node.classList.remove("is-dragging");
    }));
    node.addEventListener("drop", (event) => handler(event.dataTransfer.files));
  }

  function initialize() {
    $(".brand").addEventListener("click", (event) => {
      event.preventDefault();
      switchView("detect");
    });
    $$("[data-view]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
    $("#mobileMenu").addEventListener("click", () => {
      const open = $(".main-nav").classList.toggle("is-open");
      $("#mobileMenu").setAttribute("aria-expanded", String(open));
      $("#mobileMenu").setAttribute("aria-label", open ? "关闭导航" : "打开导航");
    });
    $("#singleDropzone").addEventListener("click", () => $("#singleFile").click());
    $("#singleFile").addEventListener("change", (event) => event.target.files[0] && selectSingle(event.target.files[0]));
    bindDropzone($("#singleDropzone"), (files) => files[0] && selectSingle(files[0]));
    $("#clearSingle").addEventListener("click", () => clearSingle());
    $("#detectButton").addEventListener("click", detectSingle);
    $("#confidence").addEventListener("input", (event) => {
      $("#confidenceLabel").textContent = `置信度 ${Number(event.target.value).toFixed(2)}`;
    });
    $("#downloadImage").addEventListener("click", downloadSingleImage);
    $("#downloadJson").addEventListener("click", downloadSingleJson);
    $$("[data-mode]").forEach((button) => button.addEventListener("click", () => setRecognitionMode(button.dataset.mode)));

    $("#batchDropzone").addEventListener("click", () => $("#batchFiles").click());
    $("#batchFiles").addEventListener("change", (event) => selectBatch(event.target.files));
    bindDropzone($("#batchDropzone"), selectBatch);
    $("#clearBatch").addEventListener("click", () => clearBatch());
    $("#batchButton").addEventListener("click", detectBatch);
    $("#batchConfidence").addEventListener("input", (event) => {
      $("#batchConfidenceLabel").textContent = Number(event.target.value).toFixed(2);
    });
    $("#downloadBatch").addEventListener("click", () => {
      if (state.batchResult && state.batchResult.download_url) window.location.assign(state.batchResult.download_url);
    });
    $("#clearHistory").addEventListener("click", clearHistory);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        $(".main-nav").classList.remove("is-open");
        $("#mobileMenu").setAttribute("aria-expanded", "false");
        $("#mobileMenu").setAttribute("aria-label", "打开导航");
      }
    });

    const initialView = ["detect", "batch", "history"].includes(window.location.hash.slice(1))
      ? window.location.hash.slice(1)
      : "detect";
    switchView(initialView);
    window.addEventListener("hashchange", () => {
      const nextView = window.location.hash.slice(1);
      if (["detect", "batch", "history"].includes(nextView)) switchView(nextView);
    });
    renderHistory();
    loadCapabilities();
    refreshHealth();
    window.setInterval(refreshHealth, 15000);
  }

  initialize();
})();

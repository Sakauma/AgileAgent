(() => {
  "use strict";

  const LIMITS = { fileBytes: 0, batchFiles: 0, batchBytes: 0, incrementalBytes: 0 };
  const UI = { historyLimit: 0, resultCacheLimit: 0, healthPollMs: 0, toastDurationMs: 0 };
  const INCREMENTAL = { previewLimit: 0, logTailLines: 0, pollIntervalMs: 0 };
  const HISTORY_KEY = "agile-agent-session-history-v1";
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
    history: [],
    incrementalFile: null,
    incrementalBatches: [],
    incrementalBatch: null,
    trainingJob: null,
    trainingPoll: null,
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
    if (file.size > LIMITS.fileBytes) throw new Error(`单张图像不能超过${formatBytes(LIMITS.fileBytes)}：${file.name}`);
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
    window.setTimeout(() => toast.remove(), UI.toastDurationMs);
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
    if (viewName === "incremental") loadIncrementalBatches();
    window.location.hash = viewName;
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const BATCH_STATUS = {
    AUDITED: "审计通过", REJECTED: "审计未通过", INJECTED: "已准备",
    TRAINING: "训练中", TRAINED_CANDIDATE: "候选已生成", FAILED: "训练失败",
    CANCELLED: "已停止", QUEUED: "排队中", RUNNING: "训练中", COMPLETED: "训练完成",
    CANCELLING: "正在停止",
  };

  function batchStatus(value) {
    return BATCH_STATUS[value] || value || "未知";
  }

  function selectIncrementalFile(file) {
    if (!file || !file.name.toLowerCase().endsWith(".zip")) {
      showToast("请选择ZIP格式的增量数据包。", "error");
      return;
    }
    if (!file.size || file.size > LIMITS.incrementalBytes) {
      showToast(`数据包不能超过${formatBytes(LIMITS.incrementalBytes)}。`, "error");
      return;
    }
    state.incrementalFile = file;
    $("#selectedArchiveName").textContent = file.name;
    $("#selectedArchiveSize").textContent = formatBytes(file.size);
    $("#selectedArchive").classList.remove("is-hidden");
    $("#uploadIncremental").disabled = false;
  }

  function clearIncrementalFile() {
    state.incrementalFile = null;
    $("#incrementalFile").value = "";
    $("#selectedArchive").classList.add("is-hidden");
    $("#uploadIncremental").disabled = true;
  }

  async function uploadIncrementalBatch() {
    if (!state.incrementalFile) return;
    const form = new FormData();
    form.append("file", state.incrementalFile);
    form.append("name", $("#incrementalName").value.trim());
    form.append("class_names", $("#incrementalClasses").value.trim());
    setLoading(true, "正在审计增量数据", "检查图像、标签、类别和数据边界");
    try {
      const response = await fetch("/api/incremental/batches", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "数据审计未通过。");
      clearIncrementalFile();
      $("#incrementalName").value = "";
      $("#incrementalClasses").value = "";
      await loadIncrementalBatches(payload.batch_id);
      showToast("增量数据已保存并通过审计。");
    } catch (error) {
      showToast(error.message || "增量数据上传失败。", "error");
      await loadIncrementalBatches();
    } finally {
      setLoading(false);
    }
  }

  async function loadIncrementalBatches(openBatchId = null) {
    try {
      const response = await fetch("/api/incremental/batches", { cache: "no-store" });
      if (!response.ok) throw new Error(await responseError(response));
      const payload = await response.json();
      state.incrementalBatches = payload.batches || [];
      renderIncrementalBatchList();
      if (openBatchId) await openIncrementalBatch(openBatchId);
    } catch (error) {
      showToast(error.message || "无法读取增量批次。", "error");
    }
  }

  function renderIncrementalBatchList() {
    const list = $("#incrementalBatchList");
    list.innerHTML = "";
    $("#incrementalBatchCount").textContent = `${state.incrementalBatches.length}批`;
    if (!state.incrementalBatches.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.innerHTML = `${icon("archive")}<div><strong>还没有增量批次</strong><small>上传后可在这里浏览和操作</small></div>`;
      list.appendChild(empty);
      return;
    }
    state.incrementalBatches.forEach((batch) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `incremental-batch-row${state.incrementalBatch && state.incrementalBatch.batch_id === batch.batch_id ? " is-active" : ""}`;
      const glyph = document.createElement("span");
      glyph.innerHTML = icon(batch.status === "REJECTED" || batch.status === "FAILED" ? "alert" : "archive");
      const content = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = batch.name;
      const meta = document.createElement("small");
      meta.textContent = `${Number((batch.audit || {}).image_count || 0)}张图像 · ${new Date(batch.created_at).toLocaleString()}`;
      content.append(title, meta);
      const status = document.createElement("span");
      status.className = "batch-status";
      status.textContent = batchStatus(batch.status);
      row.append(glyph, content, status);
      row.addEventListener("click", () => openIncrementalBatch(batch.batch_id));
      list.appendChild(row);
    });
  }

  function metricNode(label, value) {
    const node = document.createElement("div");
    node.className = "incremental-metric";
    const title = document.createElement("span");
    title.textContent = label;
    const content = document.createElement("b");
    content.textContent = String(value);
    node.append(title, content);
    return node;
  }

  function renderIncrementalClassEditor(batch, audit) {
    const bindings = Array.isArray(audit.class_bindings) ? audit.class_bindings : [];
    const list = $("#incrementalClassList");
    list.replaceChildren();
    $("#incrementalClassCount").textContent = `${bindings.length}类`;
    const editable = ["AUDITED", "INJECTED", "FAILED", "TRAINED_CANDIDATE"].includes(batch.status);
    $("#saveIncrementalClasses").disabled = !editable || !bindings.length;
    if (!bindings.length) {
      const empty = document.createElement("div");
      empty.className = "incremental-class-empty";
      empty.textContent = "没有可用的类别绑定";
      list.appendChild(empty);
      return;
    }
    bindings.forEach((binding) => {
      const row = document.createElement("div");
      row.className = "incremental-class-row";
      const identity = document.createElement("div");
      identity.className = "incremental-class-identity";
      const title = document.createElement("strong");
      title.textContent = binding.is_existing_class ? "已有类别" : "新增类别";
      const ids = document.createElement("small");
      ids.textContent = `源ID ${binding.source_class_id} · 训练ID ${binding.training_class_id} · 全局ID ${binding.global_class_id}`;
      identity.append(title, ids);
      const field = document.createElement("label");
      field.textContent = "类别名称";
      const input = document.createElement("input");
      input.type = "text";
      input.maxLength = 80;
      input.value = binding.display_name || "";
      input.dataset.sourceClassId = String(binding.source_class_id);
      input.disabled = !editable;
      field.appendChild(input);
      const source = document.createElement("span");
      source.className = `semantic-status${binding.semantic_status === "provisional" ? " is-provisional" : ""}`;
      source.textContent = binding.semantic_status === "provisional" ? "自动命名" : "名称已确认";
      row.append(identity, field, source);
      list.appendChild(row);
    });
  }

  async function openIncrementalBatch(batchId) {
    try {
      const response = await fetch(`/api/incremental/batches/${encodeURIComponent(batchId)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(await responseError(response));
      state.incrementalBatch = await response.json();
      renderIncrementalDetail();
      renderIncrementalBatchList();
      await loadIncrementalEvents();
      if (state.incrementalBatch.training_job_id && state.incrementalBatch.status === "TRAINING") {
        startTrainingPoll(state.incrementalBatch.training_job_id);
      }
    } catch (error) {
      showToast(error.message || "无法打开增量批次。", "error");
    }
  }

  function renderIncrementalDetail() {
    const batch = state.incrementalBatch;
    if (!batch) return;
    const audit = batch.audit || {};
    $("#incrementalDetail").classList.remove("is-hidden");
    $("#incrementalDetailName").textContent = batch.name;
    $("#incrementalDetailMeta").textContent = `${batch.source.filename} · ${formatBytes(batch.source.size_bytes)} · ${audit.incremental_mode === "class_incremental" ? "类别增量" : "目标增量"}`;
    $("#incrementalDetailStatus").textContent = batchStatus(batch.status);
    const metrics = $("#incrementalMetrics");
    metrics.innerHTML = "";
    const classNames = Object.values(audit.class_map || {}).join("、") || "待确认";
    const labelFormat = audit.label_format === "bbox_only" ? "四列·类别待确认" : "五列YOLO";
    [["图像", audit.image_count || 0], ["目标", audit.object_count || 0], ["类别", classNames], ["标签格式", labelFormat], ["旧样本读取", audit.old_raw_image_count || 0], ["合规审计", audit.compliance === "passed" ? "通过" : "未通过"]]
      .forEach(([label, value]) => metrics.appendChild(metricNode(label, value)));
    renderIncrementalClassEditor(batch, audit);
    $("#injectIncremental").disabled = batch.status !== "AUDITED";
    $("#trainIncremental").disabled = !["INJECTED", "FAILED"].includes(batch.status);
    const gallery = $("#incrementalGallery");
    gallery.innerHTML = "";
    (batch.files || []).slice(0, INCREMENTAL.previewLimit).forEach((item, index) => {
      const node = document.createElement("div");
      node.className = "incremental-thumb";
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = `/api/incremental/batches/${encodeURIComponent(batch.batch_id)}/images/${index}`;
      image.alt = `增量样本 ${index + 1}`;
      const count = document.createElement("span");
      count.textContent = `${item.object_count}个目标`;
      node.append(image, count);
      gallery.appendChild(node);
    });
  }

  async function saveIncrementalClasses() {
    if (!state.incrementalBatch) return;
    const inputs = $$("#incrementalClassList input[data-source-class-id]");
    const names = {};
    for (const input of inputs) {
      const value = input.value.trim();
      if (!value) {
        showToast("类别名称不能为空。", "error");
        input.focus();
        return;
      }
      names[input.dataset.sourceClassId] = value;
    }
    setLoading(true, "正在更新类别", "保存类别绑定和训练配置");
    try {
      const response = await fetch(
        `/api/incremental/batches/${encodeURIComponent(state.incrementalBatch.batch_id)}/classes`,
        { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ names }) },
      );
      if (!response.ok) throw new Error(await responseError(response));
      state.incrementalBatch = await response.json();
      renderIncrementalDetail();
      await loadIncrementalBatches();
      await loadIncrementalEvents();
      showToast("类别名称已保存。");
    } catch (error) {
      showToast(error.message || "类别名称保存失败。", "error");
    } finally {
      setLoading(false);
    }
  }

  async function injectIncrementalBatch() {
    if (!state.incrementalBatch) return;
    setLoading(true, "正在准备训练数据", "生成独立训练视图和内部批次配置");
    try {
      const response = await fetch(`/api/incremental/batches/${encodeURIComponent(state.incrementalBatch.batch_id)}/inject`, { method: "POST" });
      if (!response.ok) throw new Error(await responseError(response));
      state.incrementalBatch = await response.json();
      renderIncrementalDetail();
      await loadIncrementalBatches();
      await loadIncrementalEvents();
      showToast("训练视图已准备完成。");
    } catch (error) {
      showToast(error.message || "数据注入失败。", "error");
    } finally {
      setLoading(false);
    }
  }

  async function trainIncrementalBatch() {
    if (!state.incrementalBatch) return;
    try {
      const response = await fetch(`/api/incremental/batches/${encodeURIComponent(state.incrementalBatch.batch_id)}/train`, { method: "POST" });
      if (!response.ok) throw new Error(await responseError(response));
      state.trainingJob = await response.json();
      $("#trainingMonitor").classList.remove("is-hidden");
      $("#cancelTraining").classList.remove("is-hidden");
      startTrainingPoll(state.trainingJob.job_id);
      showToast("训练任务已提交到GPU队列。");
    } catch (error) {
      showToast(error.message || "无法启动训练。", "error");
    }
  }

  function startTrainingPoll(jobId) {
    if (state.trainingPoll) window.clearInterval(state.trainingPoll);
    pollTraining(jobId);
    state.trainingPoll = window.setInterval(() => pollTraining(jobId), INCREMENTAL.pollIntervalMs);
  }

  async function pollTraining(jobId) {
    if (!state.incrementalBatch) return;
    const batchId = encodeURIComponent(state.incrementalBatch.batch_id);
    try {
      const [jobResponse, logResponse] = await Promise.all([
        fetch(`/api/incremental/jobs/${encodeURIComponent(jobId)}?batch_id=${batchId}`, { cache: "no-store" }),
        fetch(`/api/incremental/jobs/${encodeURIComponent(jobId)}/logs?batch_id=${batchId}&tail=${INCREMENTAL.logTailLines}`, { cache: "no-store" }),
      ]);
      if (!jobResponse.ok) throw new Error(await responseError(jobResponse));
      state.trainingJob = await jobResponse.json();
      $("#trainingMonitor").classList.remove("is-hidden");
      $("#trainingStatus").textContent = batchStatus(state.trainingJob.status);
      $("#trainingLog").textContent = logResponse.ok ? (await logResponse.text() || "训练进程正在初始化...") : "暂时无法读取训练日志。";
      $("#trainingLog").scrollTop = $("#trainingLog").scrollHeight;
      const terminal = ["COMPLETED", "FAILED", "CANCELLED"].includes(state.trainingJob.status);
      $("#cancelTraining").classList.toggle("is-hidden", terminal);
      $("#trainingProgressBar").style.animationPlayState = terminal ? "paused" : "running";
      if (terminal) {
        window.clearInterval(state.trainingPoll);
        state.trainingPoll = null;
        await openIncrementalBatch(state.incrementalBatch.batch_id);
      }
    } catch (error) {
      showToast(error.message || "训练状态更新失败。", "error");
    }
  }

  async function cancelTraining() {
    if (!state.incrementalBatch || !state.trainingJob) return;
    const response = await fetch(`/api/incremental/jobs/${encodeURIComponent(state.trainingJob.job_id)}/cancel?batch_id=${encodeURIComponent(state.incrementalBatch.batch_id)}`, { method: "POST" });
    if (!response.ok) showToast(await responseError(response), "error");
    else showToast("已请求停止训练。")
  }

  async function loadIncrementalEvents() {
    if (!state.incrementalBatch) return;
    const response = await fetch(`/api/logs?batch_id=${encodeURIComponent(state.incrementalBatch.batch_id)}&limit=100`, { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    const root = $("#incrementalEventLog");
    root.innerHTML = "";
    (payload.events || []).forEach((event) => {
      const node = document.createElement("div");
      node.className = "operation-event";
      const time = document.createElement("time");
      time.textContent = new Date(event.timestamp).toLocaleString();
      const name = document.createElement("b");
      name.textContent = event.event;
      const duration = document.createElement("span");
      duration.textContent = event.duration_ms == null ? event.level : `${Number(event.duration_ms).toFixed(1)} ms`;
      node.append(time, name, duration);
      root.appendChild(node);
    });
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
    const canvas = $("#resultImage");
    canvas.width = 0;
    canvas.height = 0;
    $("#singleDropzone").classList.remove("is-hidden", "is-dragging");
    $("#inputPreview").classList.add("is-hidden");
    $("#inputPreview").classList.remove("is-unavailable");
    $("#detectButton").disabled = true;
    $("#clearSingle").disabled = true;
    $("#resultEmpty").classList.remove("is-hidden");
    $("#resultCanvas").classList.add("is-hidden");
    $("#collaborationPanel").classList.add("is-hidden");
    $("#resultDetails").classList.add("is-hidden");
    $("#resultState").textContent = "等待输入";
    $("#resultState").className = "result-state";
    setWorkflow(1);
    if (notify) showToast("当前检测任务已清除。", "success");
  }

  async function drawDetectionCanvas(result, imageUrl) {
    if (!imageUrl) throw new Error("原始图像预览已过期。");
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("浏览器无法绘制结果图。"));
      image.src = imageUrl;
    });
    const canvas = $("#resultImage");
    canvas.width = Number(result.image_width || image.naturalWidth);
    canvas.height = Number(result.image_height || image.naturalHeight);
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const colors = { 0: "#159a91", 1: "#3978d4", 2: "#d17a35", 3: "#8b65c8" };
    context.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 320));
    (result.detections || []).forEach((detection) => {
      const [x1, y1, x2, y2] = detection.xyxy.map(Number);
      context.strokeStyle = colors[detection.class_id] || "#159a91";
      context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
  }

  async function renderSingleResult(result) {
    const context = result.context || {};
    const detections = Array.isArray(result.detections) ? result.detections : [];
    const counts = result.class_counts || {};
    await drawDetectionCanvas(result, result._preview_url || state.singlePreviewUrl);
    $("#overlaySensor").textContent = sensorLabel(context.sensor);
    $("#overlayScene").textContent = sceneLabel(context.scene);
    $("#overlayCount").textContent = `${result.detection_count || 0} 个目标`;
    $("#resultEmpty").classList.add("is-hidden");
    $("#resultCanvas").classList.remove("is-hidden");
    $("#resultDetails").classList.remove("is-hidden");
    $("#collaborationPanel").classList.remove("is-hidden");
    $("#resultState").textContent = "分析完成";
    $("#resultState").className = "result-state is-complete";
    renderCollaboration(result);

    const summary = [
      ["传感器", sensorLabel(context.sensor), `${Math.round((context.sensor_confidence || 0) * 100)}% 置信度`],
      ["场景", sceneLabel(context.scene), `${Math.round((context.scene_confidence || 0) * 100)}% 置信度`],
      ["检测目标", String(result.detection_count || 0), `本次阈值 ${Number(result.confidence_threshold || 0).toFixed(2)}`],
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
    const decision = agent.decision || {};
    const protocols = Array.isArray(agent.protocols) ? agent.protocols : [];
    const activated = Array.isArray(decision.activated_classes) ? decision.activated_classes : [];
    const eligible = Array.isArray(decision.eligible_protocols) ? decision.eligible_protocols : [];
    const skipped = Array.isArray(decision.skipped_protocols) ? decision.skipped_protocols : [];
    const fusion = decision.fusion_summary || {};
    $("#collaborationMode").textContent = "Agent 自动决策";
    const flow = $("#collaborationFlow");
    flow.replaceChildren();
    const evaluations = protocols.length
      ? protocols.map((item) => `${classLabel(item.class_name || item.new_class)} ${item.activated ? "已激活" : "未激活"}`).join(" · ")
      : skipped.length ? "当前输入无需调用专项模型" : "保持统一检测流程";
    const steps = [
      ["01 场景认知", `${sensorLabel(context.sensor)} · ${sceneLabel(context.scene)}`, "自动理解输入模态与任务场景"],
      ["02 统一检测", `${Number(decision.base_detection_count || 0)} 个基础候选`, "建立当前图像的统一检测结果"],
      ["03 智能评估", `${eligible.length} 项候选 · ${Number(decision.evaluated_specialists || 0)} 项执行`, evaluations],
      activated.length
        ? ["04 自动融合", activated.map(classLabel).join("、"), `保留 ${Number(fusion.output_count || result.detection_count || 0)} 个最终目标`]
        : ["04 结果确认", `${result.detection_count || 0} 个目标`, "保持统一检测结果，无需人工选择模型"],
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
      const response = await fetch("/api/detect", { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      const result = await response.json();
      state.singleResult = result;
      result._preview_url = state.singlePreviewUrl;
      await renderSingleResult(result);
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
    $("#resultImage").toBlob((blob) => {
      if (blob) downloadBlob(blob, `lingdong-agent-${Date.now()}.png`);
    }, "image/png");
  }

  function downloadSingleJson() {
    if (!state.singleResult) return;
    const payload = { ...state.singleResult };
    delete payload._preview_url;
    downloadBlob(new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json;charset=utf-8" }), `agile-agent-${Date.now()}.json`);
  }

  async function selectBatch(fileList) {
    try {
      const files = Array.from(fileList);
      if (!files.length) return;
      if (files.length > LIMITS.batchFiles) throw new Error(`单批最多选择${LIMITS.batchFiles}张图像。`);
      const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
      if (totalBytes > LIMITS.batchBytes) throw new Error(`单批图像总大小不能超过${formatBytes(LIMITS.batchBytes)}。`);
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
      setLoading(true, "正在执行批量检测", `正在批量处理${state.batchFiles.length}张图像`);
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
      ["检测目标", String(result.detection_count || 0), `本次阈值 ${Number(result.confidence_threshold || 0).toFixed(2)}`],
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
      return Array.isArray(payload) ? payload.slice(0, UI.historyLimit) : [];
    } catch (_error) {
      return [];
    }
  }

  function rememberResult(type, payload) {
    const identity = type === "batch" ? payload.batch_id : payload.filename;
    const key = `${type}:${identity || "result"}:${Date.now()}`;
    state.resultCache.set(key, { type, payload });
    while (state.resultCache.size > UI.resultCacheLimit) {
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
      renderSingleResult(cached.payload).catch((error) => showToast(error.message, "error"));
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
    state.history = [item, ...state.history].slice(0, UI.historyLimit);
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

  async function loadPublicConfig() {
    const response = await fetch("/api/config/public", { cache: "no-store" });
    if (!response.ok) throw new Error(await responseError(response));
    const config = await response.json();
    LIMITS.fileBytes = Number(config.limits.max_file_bytes);
    LIMITS.batchFiles = Number(config.limits.max_batch_files);
    LIMITS.batchBytes = Number(config.limits.max_batch_bytes);
    LIMITS.incrementalBytes = Number(config.incremental.max_archive_bytes);
    INCREMENTAL.previewLimit = Number(config.incremental.preview_limit);
    INCREMENTAL.logTailLines = Number(config.incremental.job_log_tail_lines);
    INCREMENTAL.pollIntervalMs = Number(config.incremental.poll_interval_ms);
    UI.historyLimit = Number(config.ui.history_limit);
    UI.resultCacheLimit = Number(config.ui.result_cache_limit);
    UI.healthPollMs = Number(config.ui.health_poll_ms);
    UI.toastDurationMs = Number(config.ui.toast_duration_ms);
    state.history = readHistory();
    const confidence = config.confidence;
    [$("#confidence"), $("#batchConfidence")].forEach((input) => {
      input.min = String(confidence.min);
      input.max = String(confidence.max);
      input.value = String(confidence.default);
    });
    $("#confidenceLabel").textContent = `置信度 ${Number(confidence.default).toFixed(2)}`;
    $("#batchConfidenceLabel").textContent = Number(confidence.default).toFixed(2);
    $("#singleLimitText").textContent = `或点击选择文件 · 最大${formatBytes(LIMITS.fileBytes)}`;
    $("#batchLimitText").textContent = `单批最多${LIMITS.batchFiles}张，总计不超过${formatBytes(LIMITS.batchBytes)}`;
    $("#batchIntro").textContent = `一次处理最多${LIMITS.batchFiles}张图像，完成后可逐张查看标注图和结果清单。`;
    $("#incrementalLimitText").textContent = `包含图像、标签及可选data.yaml · 最大${formatBytes(LIMITS.incrementalBytes)}`;
  }

  async function initialize() {
    try {
      await loadPublicConfig();
    } catch (error) {
      showToast(error.message || "无法读取服务配置。", "error");
      return;
    }
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
    $("#incrementalDropzone").addEventListener("click", () => $("#incrementalFile").click());
    $("#incrementalFile").addEventListener("change", (event) => event.target.files[0] && selectIncrementalFile(event.target.files[0]));
    bindDropzone($("#incrementalDropzone"), (files) => files[0] && selectIncrementalFile(files[0]));
    $("#clearIncrementalFile").addEventListener("click", clearIncrementalFile);
    $("#uploadIncremental").addEventListener("click", uploadIncrementalBatch);
    $("#refreshIncremental").addEventListener("click", () => loadIncrementalBatches(state.incrementalBatch && state.incrementalBatch.batch_id));
    $("#injectIncremental").addEventListener("click", injectIncrementalBatch);
    $("#saveIncrementalClasses").addEventListener("click", saveIncrementalClasses);
    $("#trainIncremental").addEventListener("click", trainIncrementalBatch);
    $("#cancelTraining").addEventListener("click", cancelTraining);
    $("#clearHistory").addEventListener("click", clearHistory);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        $(".main-nav").classList.remove("is-open");
        $("#mobileMenu").setAttribute("aria-expanded", "false");
        $("#mobileMenu").setAttribute("aria-label", "打开导航");
      }
    });

    const initialView = ["detect", "batch", "incremental", "history"].includes(window.location.hash.slice(1))
      ? window.location.hash.slice(1)
      : "detect";
    switchView(initialView);
    window.addEventListener("hashchange", () => {
      const nextView = window.location.hash.slice(1);
      if (["detect", "batch", "incremental", "history"].includes(nextView)) switchView(nextView);
    });
    renderHistory();
    refreshHealth();
    window.setInterval(refreshHealth, UI.healthPollMs);
  }

  initialize();
})();

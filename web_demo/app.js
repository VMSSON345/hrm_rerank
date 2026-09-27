(() => {
  "use strict";

  const STORAGE_KEY = "finhrm-demo-sessions-v1";
  const MAX_SESSIONS = 8;
  const DOCUMENTS_KEY = "finhrm-demo-documents-v1";
  const MAX_FILE_BYTES = 512 * 1024 * 1024;
  const OPERATION_LABELS = {
    SELECT: "Chọn giá trị",
    ADD: "Cộng",
    SUBTRACT: "Trừ",
    MULTIPLY: "Nhân",
    DIVIDE: "Chia",
    RATIO_PERCENT: "Tỷ lệ phần trăm",
    PERCENT_CHANGE: "Mức thay đổi",
    AVERAGE: "Trung bình",
    SUM3: "Tổng ba giá trị",
    SUM4: "Tổng bốn giá trị",
  };

  const state = {
    health: null,
    document: null,
    filename: null,
    sessionId: null,
    messages: [],
    ready: false,
    ingesting: false,
    busy: false,
    jobId: null,
    thinkingTimer: null,
    toastTimer: null,
    userIsNearBottom: true,
  };

  const byId = (id) => document.getElementById(id);
  const elements = {
    body: document.body,
    welcome: byId("welcome-state"),
    uploadPanel: byId("upload-panel"),
    chooseFile: byId("choose-file"),
    fileInput: byId("file-input"),
    processingCard: byId("processing-card"),
    processingTitle: byId("processing-title"),
    processingDescription: byId("processing-description"),
    processingProgress: byId("processing-progress"),
    processingPercent: byId("processing-percent"),
    conversation: byId("conversation"),
    conversationScroll: byId("conversation-scroll"),
    composerWrap: byId("composer-wrap"),
    questionForm: byId("question-form"),
    questionInput: byId("question-input"),
    sendQuestion: byId("send-question"),
    exportSession: byId("export-session"),
    documentIndicator: byId("document-indicator"),
    documentIndicatorName: byId("document-indicator-name"),
    recentList: byId("recent-list"),
    newSession: byId("new-session"),
    sidebar: byId("sidebar"),
    openSidebar: byId("open-sidebar"),
    closeSidebar: byId("close-sidebar"),
    sidebarScrim: byId("sidebar-scrim"),
    openSystemDialog: byId("open-system-dialog"),
    topModelButton: byId("top-model-button"),
    systemDialog: byId("system-dialog"),
    toast: byId("toast"),
    jumpToLatest: byId("jump-to-latest"),
  };

  function createElement(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function requestJSON(path, options = {}) {
    const response = await fetch(path, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      payload = { error: `Dịch vụ trả về HTTP ${response.status}.` };
    }
    if (!response.ok) {
      const error = new Error(payload.error || `Yêu cầu thất bại (${response.status}).`);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function sleep(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  function setStatusDot(dot, mode) {
    dot.classList.remove("is-online", "is-busy");
    if (mode) dot.classList.add(mode);
  }

  async function refreshHealth() {
    try {
      const health = await requestJSON("/health");
      state.health = health;
      const model = health.model || {};
      const checkpoint = model.checkpoint || "Document HRM";
      byId("sidebar-model-name").textContent = "FinHRM Local";
      byId("top-model-name").textContent = "HRM · Local";
      byId("dialog-checkpoint").textContent = checkpoint;
      byId("dialog-device").textContent = String(model.device || "cpu").toUpperCase();
      byId("dialog-model-state").textContent = health.model_loaded ? "Đã nạp" : "Chưa nạp";
      byId("dialog-mineru-state").textContent = describeMinerU(health.mineru);
      byId("dialog-inference-state").textContent = health.inference_busy ? "Đang suy luận" : "Sẵn sàng";
      byId("dialog-architecture").textContent = model.architecture || "Document HRM";

      const mineruHealthy = health.mineru && health.mineru.status === "healthy";
      const busy = Boolean(health.inference_busy);
      const statusCopy = busy
        ? "Mô hình đang suy luận"
        : mineruHealthy
          ? "Mô hình và OCR sẵn sàng"
          : "Mô hình sẵn sàng · OCR hạn chế";
      byId("sidebar-system-status").textContent = health.model_loaded ? statusCopy : "Mô hình chưa sẵn sàng";
      byId("topbar-status-text").textContent = busy ? "Đang suy luận" : health.model_loaded ? "Sẵn sàng" : "Chưa nạp mô hình";
      setStatusDot(byId("sidebar-status-dot"), busy ? "is-busy" : health.model_loaded ? "is-online" : null);
      setStatusDot(byId("topbar-status-dot"), busy ? "is-busy" : health.model_loaded ? "is-online" : null);
    } catch {
      byId("sidebar-system-status").textContent = "Không kết nối được dịch vụ";
      byId("topbar-status-text").textContent = "Mất kết nối";
      byId("dialog-model-state").textContent = "Không kết nối";
      setStatusDot(byId("sidebar-status-dot"), null);
      setStatusDot(byId("topbar-status-dot"), null);
    }
  }

  function describeMinerU(mineru) {
    if (!mineru) return "Không rõ";
    if (mineru.status === "healthy") return "Sẵn sàng";
    if (mineru.status === "disabled") return "Đã tắt";
    return "Không kết nối";
  }

  function showToast(message) {
    window.clearTimeout(state.toastTimer);
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    state.toastTimer = window.setTimeout(() => {
      elements.toast.hidden = true;
    }, 3600);
  }

  function setProcessing({ title, description, step }) {
    elements.welcome.hidden = true;
    elements.processingCard.hidden = false;
    elements.processingTitle.textContent = title;
    elements.processingDescription.textContent = description;
    if (step === "ready") {
      elements.processingProgress.value = 100;
      elements.processingPercent.textContent = "Hoàn tất";
    } else {
      elements.processingProgress.removeAttribute("value");
      elements.processingPercent.textContent = "Đang xử lý";
    }
    const steps = ["upload", "extract", "index", "ready"];
    const current = Math.max(0, steps.indexOf(step));
    steps.forEach((name, index) => {
      const node = byId(`step-${name}`);
      node.classList.toggle("is-active", index === current);
      node.classList.toggle("is-done", index < current);
    });
  }

  function resetSession() {
    if (isLocked()) return;
    const selected = state.ready ? { document: state.document, filename: state.filename } : null;
    stopThinking();
    state.document = null;
    state.filename = null;
    state.sessionId = null;
    state.messages = [];
    state.ready = false;
    state.busy = false;
    state.jobId = null;
    state.userIsNearBottom = true;
    elements.fileInput.value = "";
    elements.processingCard.hidden = true;
    elements.conversation.textContent = "";
    elements.composerWrap.hidden = true;
    elements.documentIndicator.hidden = true;
    elements.jumpToLatest.hidden = true;
    elements.welcome.hidden = false;
    elements.questionInput.value = "";
    resizeTextarea();
    closeSidebar();
    if (selected) startDocumentChat(selected);
    syncWorkspace();
  }

  function validateFile(file) {
    const extension = file.name.toLowerCase().split(".").pop();
    if (!["pdf", "txt"].includes(extension)) {
      throw new Error("Chỉ hỗ trợ tài liệu PDF hoặc TXT.");
    }
    if (!file.size) throw new Error("Tệp đang rỗng.");
    if (file.size > MAX_FILE_BYTES) {
      throw new Error("Tệp vượt quá giới hạn 512 MiB.");
    }
  }

  async function handleFile(file) {
    if (isLocked()) return;
    try { validateFile(file); } catch (error) { showToast(error.message); return; }
    state.ingesting = true;
    state.ready = false;
    state.document = null;
    elements.conversation.textContent = "";
    elements.processingCard.querySelectorAll(".error-box").forEach(node => node.remove());
    closeDocuments();
    syncWorkspace();
    try {
      state.filename = file.name;
      state.sessionId = `session-${Date.now()}`;
      state.messages = [];
      setProcessing({
        title: `Đang tải ${file.name}`,
        description: `${formatBytes(file.size)} · lưu vào vùng demo local`,
        progress: 8,
        step: "upload",
      });

      const upload = await requestJSON(
        `/v1/upload?filename=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          headers: { "Content-Type": file.type || "application/octet-stream" },
          body: file,
        },
      );
      state.document = upload.document;
      setProcessing({
        title: "Đang đọc cấu trúc báo cáo",
        description: file.name.toLowerCase().endsWith(".pdf") ? "MinerU đang chuẩn bị nội dung và bảng biểu trong PDF." : "Đang chuẩn bị nội dung tài liệu văn bản.",
        progress: 28,
        step: "extract",
      });

      const job = await requestJSON("/v1/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document: state.document }),
      });
      state.jobId = job.job_id;
      await pollIngestion(job.job_id);
    } catch (error) {
      showProcessingError(error.message || "Không thể chuẩn bị tài liệu.");
    } finally {
      state.ingesting = false;
      state.jobId = null;
      syncWorkspace();
    }
  }

  async function pollIngestion(jobId) {
    while (state.jobId === jobId) {
      const job = await requestJSON(`/v1/jobs/${encodeURIComponent(jobId)}`);
      if (job.status === "failed") {
        throw new Error(job.error || "Quá trình đọc tài liệu thất bại.");
      }
      if (job.status === "ready") {
        markDocumentReady(job);
        return;
      }
      const knownTotal = Number(job.documents_total || 0);
      const completed = Number(job.documents_completed || 0);
      setProcessing({
        title: job.status === "queued" ? "Tài liệu đang chờ xử lý" : "Đang lập chỉ mục theo từng trang",
        description: knownTotal
          ? `Đã xử lý ${completed}/${knownTotal} tài liệu trong gói tải lên.`
          : "Đang phân tích lớp chữ, bảng biểu và chất lượng từng trang.",
        step: job.status === "queued" ? "extract" : "index",
      });
      await sleep(900);
    }
  }

  function markDocumentReady(job) {
    setProcessing({
      title: "Tài liệu đã sẵn sàng",
      description: summarizeIndexResults(job.index_results),
      progress: 100,
      step: "ready",
    });
    state.ready = true;
    elements.documentIndicator.hidden = false;
    elements.documentIndicatorName.textContent = state.filename || "Tài liệu";
    {
      elements.processingCard.hidden = true;
      elements.composerWrap.hidden = false;
      elements.conversation.textContent = "";
      appendIntroMessage(job);
      elements.questionInput.focus();
    }
    rememberDocument();
    saveSession();
    syncWorkspace();
  }

  function summarizeIndexResults(results) {
    const stats = (results || [])
      .map((item) => item.stats || {})
      .reduce(
        (sum, item) => ({
          pages: sum.pages + Number(item.total_pages || 0),
          native: sum.native + Number(item.native_pages || 0),
          ocr: sum.ocr + Number(item.ocr_pages || 0),
        }),
        { pages: 0, native: 0, ocr: 0 },
      );
    if (!stats.pages) return "Tài liệu văn bản đã được chuẩn bị để truy vấn.";
    return `${stats.pages} trang đã được chuẩn bị.`;
  }

  function showProcessingError(message) {
    setProcessing({
      title: "Chưa thể chuẩn bị tài liệu",
      description: message,
      progress: 0,
      step: "upload",
    });
    elements.processingProgress.value = 0;
    elements.processingPercent.textContent = "Lỗi";
    elements.processingCard.appendChild(makeErrorBox(
      "Hãy kiểm tra dịch vụ local và thử lại. PDF cần MinerU để đọc nội dung.",
    ));
    elements.welcome.hidden = false;
  }

  function appendIntroMessage(job) {
    const message = createElement("article", "message assistant-message");
    const label = makeMessageLabel("H", "FinHRM");
    const body = createElement("div", "message-body");
    const name = createElement("strong", null, state.filename || "Tài liệu");
    const copy = createElement(
      "p",
      null,
      " đã được đọc và lập chỉ mục. Bạn có thể hỏi trực tiếp một số liệu hoặc yêu cầu mô hình tính chênh lệch, tỷ lệ, tăng trưởng và tổng hợp nhiều khoản mục.",
    );
    const paragraph = document.createElement("p");
    paragraph.append(name, copy.firstChild);
    body.append(paragraph);
    const note = createElement("div", "warning-box");
    const indexCopy = summarizeIndexResults(job.index_results);
    note.append(
      createElement("strong", null, "Tài liệu sẵn sàng"),
      document.createTextNode(indexCopy),
    );
    body.append(note);
    message.append(label, body);
    elements.conversation.append(message);
    scrollToBottom({ force: true, behavior: "auto" });
  }

  function makeMessageLabel(initial, labelText) {
    const label = createElement("div", "message-label");
    label.append(createElement("span", "avatar", initial), document.createTextNode(labelText));
    return label;
  }

  function appendUserMessage(question, { scroll = true } = {}) {
    const message = createElement("article", "message user-message");
    const body = createElement("div", "message-body");
    body.append(createElement("p", null, question));
    message.append(makeMessageLabel("B", "Bạn"), body);
    elements.conversation.append(message);
    if (scroll) scrollToBottom({ force: true });
  }

  function appendThinking() {
    const message = createElement("article", "message assistant-message");
    message.id = "thinking-message";
    const body = createElement("div", "message-body");
    const row = createElement("div", "thinking-row");
    const dots = createElement("span", "thinking-dots");
    dots.append(createElement("span"), createElement("span"), createElement("span"));
    const text = createElement("span", null, "Đang tìm các trang liên quan…");
    text.id = "thinking-copy";
    row.append(dots, text);
    body.append(row);
    message.append(makeMessageLabel("H", "FinHRM"), body);
    elements.conversation.append(message);
    const stages = [
      "Đang tìm các trang liên quan…",
      "Đang trích xuất các ứng viên số…",
      "HRM đang lập chương trình suy luận…",
      "Đang kiểm tra phép tính và bằng chứng…",
    ];
    let index = 0;
    state.thinkingTimer = window.setInterval(() => {
      index = Math.min(index + 1, stages.length - 1);
      const copy = byId("thinking-copy");
      if (copy) copy.textContent = stages[index];
    }, 2600);
    scrollToBottom({ force: true });
  }

  function stopThinking() {
    window.clearInterval(state.thinkingTimer);
    state.thinkingTimer = null;
    const message = byId("thinking-message");
    if (message) message.remove();
  }

  async function submitQuestion(question) {
    if (!state.ready || state.busy || !question.trim()) return;
    state.busy = true;
    syncWorkspace();
    elements.sendQuestion.disabled = true;
    elements.questionInput.disabled = true;
    appendUserMessage(question);
    state.messages.push({ role: "user", text: question, createdAt: Date.now() });
    elements.questionInput.value = "";
    resizeTextarea();
    appendThinking();
    try {
      const answer = await requestJSON("/v1/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document: state.document,
          question,
          top_k_chunks: 8,
          neighbor_chunks: 1,
          page_top_k: 6,
          page_neighbor_pages: 1,
        }),
      });
      const shouldFollowAnswer = isNearConversationBottom();
      stopThinking();
      appendAnswer(answer, { scroll: shouldFollowAnswer });
      state.messages.push({ role: "assistant", answer, createdAt: Date.now() });
      saveSession();
    } catch (error) {
      const shouldFollowError = isNearConversationBottom();
      stopThinking();
      appendAnswerError(error, { scroll: shouldFollowError });
    } finally {
      state.busy = false;
      syncWorkspace();
      elements.sendQuestion.disabled = !elements.questionInput.value.trim();
      elements.questionInput.disabled = false;
      elements.questionInput.focus({ preventScroll: true });
      refreshHealth();
    }
  }

  function appendAnswer(answer, { scroll = true } = {}) {
    const message = createElement("article", "message assistant-message");
    const body = createElement("div", "message-body");
    const abstained = answer.status === "ABSTAIN" || answer.status === "REJECT";
    const requestOutputs = Array.isArray(answer.outputs);
    body.append(
      requestOutputs ? makeMultiOutputSummary(answer) : makeAnswerSummary(answer),
    );
    if (!requestOutputs && !abstained) {
      const sources = Array.isArray(answer.citations) && answer.citations.length ? answer.citations : (answer.evidence || []);
      if (sources.length) {
        const details = createElement("details", "source-details");
        details.append(createElement("summary", null, `Nguồn tham chiếu · ${sources.length}`));
        if (answer.program && answer.program.operation !== "SELECT") details.append(makeProgramCard(answer));
        details.append(makeEvidenceSection(sources));
        body.append(details);
      }
    }
    const copy = createElement("button", "text-button message-copy", "Sao chép phản hồi");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(answer.rendered || body.querySelector(".answer-summary, .multi-output-summary")?.textContent || ""); showToast("Đã sao chép phản hồi."); }
      catch { showToast("Trình duyệt chưa cho phép sao chép. Bạn có thể chọn văn bản để sao chép."); }
    });
    body.append(copy, makeTechnicalDetails(answer));
    message.append(makeMessageLabel("H", "FinHRM"), body);
    elements.conversation.append(message);
    if (scroll) {
      scrollToBottom({ force: true });
    } else {
      updateJumpToLatest();
    }
  }

  function makeMultiOutputSummary(answer) {
    const wrapper = createElement("section", "multi-output-summary");
    const outputs = answer.outputs || [];
    const overview = answer.split_strategy === "COMPARISON_OVERVIEW";
    const verified = answer.verified_output_ids || [];
    const statusText = overview && answer.status !== "ACCEPTED"
      ? (verified.length
        ? "So s\u00e1nh: \u0111\u00e3 x\u00e1c minh m\u1ed9t ph\u1ea7n"
        : "Ch\u01b0a x\u00e1c minh \u0111\u01b0\u1ee3c th\u00f4ng tin so s\u00e1nh")
      : answer.status === "ACCEPTED"
      ? "Y\u00eau c\u1ea7u nhi\u1ec1u ph\u1ea7n \u0111\u00e3 \u0111\u01b0\u1ee3c x\u00e1c minh"
      : answer.status === "REJECT"
        ? "Y\u00eau c\u1ea7u c\u00f3 b\u1eb1ng ch\u1ee9ng m\u00e2u thu\u1eabn"
        : "Y\u00eau c\u1ea7u ch\u01b0a \u0111\u1ee7 b\u1eb1ng ch\u1ee9ng";
    wrapper.append(
      createElement("span", "answer-status", statusText),
      createElement(
        "h2",
        "answer-value",
        `${verified.length}/${outputs.length} m\u1ee5c \u0111\u00e3 x\u00e1c minh \u0111\u1ed9c l\u1eadp`,
      ),
    );
    if (!outputs.length) {
      wrapper.append(
        createElement(
          "p",
          "multi-output-proof",
          answer.rendered || "Kh\u00f4ng th\u1ec3 t\u00e1ch c\u00e2u h\u1ecfi m\u1ed9t c\u00e1ch an to\u00e0n.",
        ),
      );
      return wrapper;
    }
    const list = createElement("div", "multi-output-list");
    outputs.forEach((output) => {
      const card = createElement("article", "multi-output-card");
      const heading = createElement("div", "multi-output-heading");
      heading.append(
        createElement(
          "h3",
          null,
          `M\u1ee5c ${output.ordinal}: ${overview ? output.question : output.source_text}`,
        ),
        createElement(
          "span",
          "operation-chip",
          overview ? ({ VERIFIED: "\u0110\u00e3 x\u00e1c minh", MISSING: "Thi\u1ebfu d\u1eef li\u1ec7u", UNKNOWN: "Ch\u01b0a \u0111\u1ee7 c\u0103n c\u1ee9", AMBIGUOUS: "Ch\u01b0a r\u00f5", CONFLICT: "Ch\u01b0a qua ki\u1ec3m ch\u1ee9ng" }[output.classification] || "Ch\u01b0a k\u1ebft lu\u1eadn")
            : output.classification || output.status || "\u2014",
        ),
      );
      const unit = output.unit && output.unit !== "unknown" ? ` ${output.unit}` : "";
      card.append(
        heading,
        createElement(
          "strong",
          "multi-output-value",
          output.classification === "VERIFIED"
            ? `${formatNumeric(output.answer)}${unit}`
            : "Kh\u00f4ng k\u1ebft lu\u1eadn",
        ),
        createElement("div", "multi-output-proof", output.rendered || ""),
      );
      list.append(card);
    });
    wrapper.append(list);
    return wrapper;
  }

  function makeAnswerSummary(answer) {
    const summary = createElement("div", "answer-summary");
    const abstained = answer.status === "ABSTAIN" || answer.status === "REJECT";
    const valueBlock = createElement("div");
    valueBlock.append(
      createElement(
        "span",
        "answer-status",
        answer.status === "ACCEPTED" ? "Câu trả lời được chấp nhận" : "Mô hình từ chối kết luận",
      ),
    );
    const response = answer.rendered || (abstained
      ? "Không đủ bằng chứng để kết luận"
      : `${formatNumeric(answer.answer)}${answer.unit ? ` ${answer.unit}` : ""}`);
    valueBlock.append(createElement("p", "verified-response", response));
    summary.append(valueBlock);
    return summary;
  }

  function makeProgramCard(answer) {
    const program = answer.program || {};
    const card = createElement("section", "program-card");
    const head = createElement("div", "program-head");
    head.append(
      createElement("span", "operation-chip", program.operation || "—"),
      createElement("span", "program-title", OPERATION_LABELS[program.operation] || "Chương trình suy luận"),
    );
    card.append(head, createElement("div", "calculation", formatCalculation(answer)));
    return card;
  }

  function formatCalculation(answer) {
    const operation = answer.program && answer.program.operation;
    const values = (answer.evidence || []).map((item) => formatNumeric(item.value));
    const result = formatNumeric(answer.answer);
    const symbols = {
      ADD: " + ",
      SUBTRACT: " − ",
      MULTIPLY: " × ",
      DIVIDE: " ÷ ",
      AVERAGE: " + ",
      SUM3: " + ",
      SUM4: " + ",
    };
    if (operation === "SELECT") return values.length ? `${values[0]} → ${result}` : result;
    if (operation === "RATIO_PERCENT" && values.length >= 2) {
      return `(${values[0]} ÷ ${values[1]}) × 100 = ${result}`;
    }
    if (operation === "PERCENT_CHANGE" && values.length >= 2) {
      return `((${values[0]} − ${values[1]}) ÷ ${values[1]}) × 100 = ${result}`;
    }
    if (operation === "AVERAGE" && values.length) {
      return `(${values.join(" + ")}) ÷ ${values.length} = ${result}`;
    }
    const symbol = symbols[operation];
    return symbol && values.length ? `${values.join(symbol)} = ${result}` : result;
  }

  function makeEvidenceSection(evidence) {
    const section = createElement("section", "evidence-section");
    const heading = createElement("div", "evidence-heading");
    heading.append(
      createElement("h3", null, "Bằng chứng nguồn"),
      createElement("span", null, `${evidence.length} nguồn`),
    );
    const list = createElement("div", "evidence-list");
    evidence.forEach((item, index) => {
      const card = createElement("article", "evidence-card");
      const meta = createElement("div", "evidence-meta");
      meta.append(createElement("span", "evidence-number", String(index + 1)));
      meta.append(createElement("span", "evidence-value", formatNumeric(item.value)));
      if (item.unit) meta.append(document.createTextNode(item.unit));
      if (item.page !== null && item.page !== undefined) {
        meta.append(document.createTextNode(`· Trang ${item.page}`));
      }
      const sourceCopy = [item.row_label, item.column_label, item.source_name].filter(Boolean).join(" · ");
      card.append(meta, createElement("p", "evidence-context", sourceCopy || "Xem thông tin nguồn trong phản hồi."));
      list.append(card);
    });
    section.append(heading, list);
    return section;
  }

  function makeTechnicalDetails(answer) {
    const details = createElement("details", "technical-details");
    details.append(createElement("summary", null, "Chi tiết kỹ thuật"));
    const grid = createElement("div", "technical-grid");
    const fields = [
      ["Thời gian", `${Number(answer.service_answer_seconds || 0).toFixed(2)} giây`],
      ["Ứng viên", String(answer.candidate_count ?? "—")],
      ["Tài liệu", String(answer.document_count ?? "—")],
      ["Trạng thái", String(answer.status || "—")],
    ];
    Object.entries(answer.stage_timings || {}).forEach(([name, seconds]) => {
      fields.push([name, `${Number(seconds).toFixed(2)}s`]);
    });
    fields.forEach(([label, value]) => {
      const row = createElement("div");
      row.append(createElement("span", null, label), createElement("code", null, value));
      grid.append(row);
    });
    details.append(grid);
    if (Array.isArray(answer.warnings) && answer.warnings.length) {
      details.append(createElement("p", "technical-warning", answer.warnings.join(" · ")));
    }
    if (answer.status === "ABSTAIN" && answer.program) {
      const rejected = {
        ...answer,
        answer: answer.candidate_answer,
      };
      details.append(
        createElement("h4", null, "Chương trình ứng viên đã bị loại"),
        createElement("code", "technical-rejected-program", formatCalculation(rejected)),
      );
    }
    return details;
  }

  function appendAnswerError(error, { scroll = true } = {}) {
    const message = createElement("article", "message assistant-message");
    const body = createElement("div", "message-body");
    body.append(makeErrorBox(error.message || "Không thể tạo câu trả lời."));
    message.append(makeMessageLabel("H", "FinHRM"), body);
    elements.conversation.append(message);
    if (scroll) {
      scrollToBottom({ force: true });
    } else {
      updateJumpToLatest();
    }
  }

  function makeErrorBox(message) {
    const box = createElement("div", "error-box");
    box.append(
      createElement("strong", null, "Yêu cầu chưa hoàn tất"),
      document.createTextNode(message),
    );
    return box;
  }

  function formatNumeric(value) {
    if (value === null || value === undefined || value === "") return "—";
    const raw = String(value);
    const numeric = Number(raw);
    if (!Number.isFinite(numeric)) return raw;
    return new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 6 }).format(numeric);
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
  }

  function resizeTextarea() {
    byId("character-count").textContent = `${elements.questionInput.value.length}/1000`;
    elements.sendQuestion.disabled = !state.ready || state.busy || state.ingesting || !elements.questionInput.value.trim();
    elements.questionInput.style.height = "auto";
    elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 160)}px`;
  }

  function isNearConversationBottom() {
    const viewport = elements.conversationScroll;
    const remaining = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    return remaining <= 96;
  }

  function updateJumpToLatest() {
    state.userIsNearBottom = isNearConversationBottom();
    elements.jumpToLatest.hidden = state.userIsNearBottom || !state.ready;
  }

  function scrollToBottom({ force = false, behavior = "auto" } = {}) {
    if (!force && !isNearConversationBottom()) {
      updateJumpToLatest();
      return;
    }
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        elements.conversationScroll.scrollTo({
          top: elements.conversationScroll.scrollHeight,
          behavior,
        });
        state.userIsNearBottom = true;
        elements.jumpToLatest.hidden = true;
      });
    });
  }

  function getSessions() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function saveSession() {
    if (!state.sessionId || !state.document || !state.filename) return;
    const session = {
      id: state.sessionId,
      document: state.document,
      filename: state.filename,
      updatedAt: Date.now(),
      messages: state.messages.slice(-12),
    };
    const sessions = getSessions().filter((item) => item.id !== session.id);
    sessions.unshift(session);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
    } catch {
      showToast("Bộ nhớ lịch sử local đã đầy; phiên hiện tại vẫn hoạt động.");
    }
    renderRecentSessions();
    renderDocuments();
  }

  function renderRecentSessions() {
    const all = getSessions();
    byId("history-count").textContent = String(all.length);
    const query = byId("history-search").value.toLocaleLowerCase("vi");
    const sessions = all.filter(item => `${item.filename} ${(item.messages || []).filter(m => m.role === "user").map(m => m.text).join(" ")}`.toLocaleLowerCase("vi").includes(query));
    elements.recentList.textContent = "";
    if (!sessions.length) {
      elements.recentList.append(createElement("p", "empty-history", query ? "Không tìm thấy hội thoại." : "Chưa có hội thoại."));
      return;
    }
    sessions.forEach((session) => {
      const button = createElement("button", "recent-item");
      button.type = "button";
      button.dataset.sessionId = session.id;
      button.classList.toggle("is-active", session.id === state.sessionId);
      button.setAttribute("aria-current", session.id === state.sessionId ? "true" : "false");
      const lastQuestion = [...(session.messages || [])].reverse().find((item) => item.role === "user");
      button.append(
        createElement("strong", null, session.filename || "Tài liệu"),
        createElement("small", null, lastQuestion ? lastQuestion.text : "Sẵn sàng đặt câu hỏi"),
      );
      button.append(createElement("span", "recent-date", new Date(session.updatedAt).toLocaleDateString("vi-VN")));
      button.disabled = state.busy || state.ingesting;
      button.addEventListener("click", () => restoreSession(session));
      elements.recentList.append(button);
    });
  }

  function restoreSession(session) {
    if (isLocked()) return;
    stopThinking();
    state.document = session.document;
    state.filename = session.filename;
    state.sessionId = session.id;
    state.messages = Array.isArray(session.messages) ? session.messages : [];
    state.ready = true;
    state.busy = false;
    elements.welcome.hidden = true;
    elements.processingCard.hidden = true;
    elements.composerWrap.hidden = false;
    elements.documentIndicator.hidden = false;
    elements.documentIndicatorName.textContent = state.filename;
    elements.conversation.textContent = "";
    const restoredIntro = {
      index_results: [],
    };
    appendIntroMessage(restoredIntro);
    state.messages.forEach((item) => {
      if (item.role === "user") appendUserMessage(item.text, { scroll: false });
      if (item.role === "assistant" && item.answer) {
        appendAnswer(item.answer, { scroll: false });
      }
    });
    scrollToBottom({ force: true, behavior: "auto" });
    closeSidebar();
    elements.questionInput.value = "";
    syncWorkspace();
    elements.questionInput.focus();
  }

  function exportSession() {
    if (!state.ready || state.busy || state.ingesting) {
      showToast("Chưa có phiên sẵn sàng để xuất.");
      return;
    }
    const payload = {
      exported_at: new Date().toISOString(),
      model: (state.health && state.health.model) || null,
      document: {
        filename: state.filename,
        local_path: state.document,
      },
      messages: state.messages,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `finhrm-${(state.filename || "session").replace(/\.[^.]+$/, "")}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast("Đã xuất phiên phân tích.");
  }

  function openSidebar() {
    closeDocuments();
    elements.body.classList.add("sidebar-open");
  }

  function closeSidebar() {
    elements.body.classList.remove("sidebar-open");
  }

  function openSystemDialog() {
    if (typeof elements.systemDialog.showModal === "function") {
      elements.systemDialog.showModal();
    }
  }


  function isLocked() {
    if (!state.busy && !state.ingesting) return false;
    showToast("Đợi thao tác hiện tại hoàn tất trước khi đổi hội thoại hoặc tài liệu.");
    return true;
  }

  function getDocuments() {
    let stored = [];
    try { const data = JSON.parse(localStorage.getItem(DOCUMENTS_KEY) || "[]"); if (Array.isArray(data)) stored = data; } catch { /* Empty local library. */ }
    const byPath = new Map();
    [...stored, ...getSessions()].forEach(item => {
      if (item && typeof item.document === "string" && typeof item.filename === "string") {
        const previous = byPath.get(item.document);
        if (!previous || Number(item.updatedAt || 0) >= Number(previous.updatedAt || 0)) {
          byPath.set(item.document, { document: item.document, filename: item.filename, updatedAt: item.updatedAt });
        }
      }
    });
    return [...byPath.values()].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  }

  function rememberDocument() {
    const documents = getDocuments().filter(item => item.document !== state.document);
    documents.unshift({ document: state.document, filename: state.filename, updatedAt: Date.now() });
    try { localStorage.setItem(DOCUMENTS_KEY, JSON.stringify(documents.slice(0, 40))); }
    catch { showToast("Không lưu được thư viện tài liệu trong trình duyệt."); }
  }

  function renderDocuments() {
    const all = getDocuments();
    byId("library-count").textContent = String(all.length);
    byId("selection-count").textContent = state.ready ? "1/1" : "0/1";
    const query = byId("document-search").value.toLocaleLowerCase("vi");
    const documents = all.filter(item => item.filename.toLocaleLowerCase("vi").includes(query));
    const container = byId("document-library"); container.textContent = "";
    if (!documents.length) container.append(createElement("p", "empty-library", query ? "Không tìm thấy tài liệu." : "Tải một báo cáo để bắt đầu. Tài liệu đã chuẩn bị sẽ được lưu ở đây."));
    documents.forEach(item => {
      const active = state.ready && item.document === state.document;
      const label = createElement("label", `source-document${active ? " is-active" : ""}`);
      const input = createElement("input"); input.type = "radio"; input.name = "source-document"; input.checked = active;
      input.disabled = state.busy || state.ingesting;
      input.setAttribute("aria-label", `Chọn ${item.filename}`);
      input.addEventListener("change", () => { if (!isLocked()) startDocumentChat(item); });
      const copy = createElement("span", "source-copy");
      copy.append(createElement("strong", null, item.filename), createElement("small", null, active ? "Đang sử dụng" : "Đã chuẩn bị · Chọn để hỏi"));
      label.append(input, copy); container.append(label);
    });
  }

  function startDocumentChat(item) {
    state.document = item.document; state.filename = item.filename;
    state.sessionId = `session-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    state.messages = []; state.ready = true; state.jobId = null;
    elements.conversation.textContent = ""; elements.welcome.hidden = true; elements.processingCard.hidden = true;
    elements.questionInput.value = "";
    appendIntroMessage({ index_results: [] });
    saveSession(); syncWorkspace(); closeDocuments(); closeSidebar(); elements.questionInput.focus();
  }

  function syncWorkspace() {
    const locked = state.busy || state.ingesting;
    elements.composerWrap.hidden = false;
    elements.questionInput.disabled = !state.ready || locked;
    elements.questionInput.placeholder = state.ready ? "Hỏi về số liệu trong tài liệu đang chọn…" : "Chọn tài liệu để bắt đầu đặt câu hỏi…";
    elements.newSession.disabled = locked;
    elements.chooseFile.disabled = locked;
    byId("welcome-upload").disabled = locked;
    elements.uploadPanel.setAttribute("aria-disabled", String(locked));
    elements.uploadPanel.tabIndex = locked ? -1 : 0;
    elements.exportSession.disabled = !state.ready || locked;
    byId("header-export").disabled = !state.ready || locked;
    byId("nav-export").disabled = !state.ready || locked;
    elements.documentIndicator.hidden = !state.ready;
    elements.documentIndicatorName.textContent = state.filename || "";
    resizeTextarea(); renderRecentSessions(); renderDocuments();
  }

  function openDocuments() {
    closeSidebar(); elements.body.classList.add("context-open");
    byId("open-documents").setAttribute("aria-expanded", "true");
    byId("document-search").focus();
  }
  function closeDocuments() {
    elements.body.classList.remove("context-open");
    byId("open-documents").setAttribute("aria-expanded", "false");
  }

  function wireEvents() {
    elements.chooseFile.addEventListener("click", (event) => {
      event.stopPropagation();
      elements.fileInput.click();
    });
    elements.uploadPanel.addEventListener("click", () => elements.fileInput.click());
    elements.uploadPanel.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        elements.fileInput.click();
      }
    });
    elements.fileInput.addEventListener("change", () => {
      const file = elements.fileInput.files && elements.fileInput.files[0];
      if (file) handleFile(file);
    });
    ["dragenter", "dragover"].forEach((eventName) => {
      elements.uploadPanel.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.uploadPanel.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      elements.uploadPanel.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.uploadPanel.classList.remove("is-dragging");
      });
    });
    elements.uploadPanel.addEventListener("drop", (event) => {
      const file = event.dataTransfer && event.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    elements.questionForm.addEventListener("submit", (event) => {
      event.preventDefault();
      submitQuestion(elements.questionInput.value.trim());
    });
    elements.questionInput.addEventListener("input", resizeTextarea);
    elements.questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        elements.questionForm.requestSubmit();
      }
    });
    elements.conversationScroll.addEventListener(
      "scroll",
      () => {
        updateJumpToLatest();
      },
      { passive: true },
    );
    elements.jumpToLatest.addEventListener("click", () => {
      scrollToBottom({ force: true, behavior: "smooth" });
      elements.questionInput.focus({ preventScroll: true });
    });

    document.querySelectorAll(".example-question").forEach((button) => {
      button.addEventListener("click", () => {
        const question = button.dataset.question || "";
        if (!state.ready) {
          showToast("Chọn một tài liệu đã sẵn sàng để đặt câu hỏi.");
          openDocuments();
          closeSidebar();
          return;
        }
        elements.questionInput.value = question;
        resizeTextarea();
        closeSidebar();
        elements.questionInput.focus();
      });
    });

    elements.newSession.addEventListener("click", resetSession);
    elements.exportSession.addEventListener("click", exportSession);
    elements.openSidebar.addEventListener("click", openSidebar);
    elements.closeSidebar.addEventListener("click", closeSidebar);
    elements.sidebarScrim.addEventListener("click", closeSidebar);
    elements.openSystemDialog.addEventListener("click", openSystemDialog);
    elements.topModelButton.addEventListener("click", openSystemDialog);
    elements.systemDialog.addEventListener("click", (event) => {
      if (event.target === elements.systemDialog) elements.systemDialog.close();
    });
  }

  wireEvents();
  byId("history-search").addEventListener("input", renderRecentSessions);
  byId("document-search").addEventListener("input", renderDocuments);
  byId("open-documents").addEventListener("click", openDocuments);
  byId("nav-documents").addEventListener("click", openDocuments);
  byId("close-documents").addEventListener("click", closeDocuments);
  byId("context-scrim").addEventListener("click", closeDocuments);
  byId("welcome-upload").addEventListener("click", () => { if (!isLocked()) elements.fileInput.click(); });
  byId("header-export").addEventListener("click", exportSession);
  byId("nav-export").addEventListener("click", exportSession);
  byId("nav-system").addEventListener("click", openSystemDialog);
  byId("focus-assistant").addEventListener("click", () => { closeSidebar(); closeDocuments(); if (state.ready) elements.questionInput.focus(); });
  document.addEventListener("keydown", event => { if (event.key === "Escape") { closeSidebar(); closeDocuments(); } });
  syncWorkspace();
  refreshHealth();
  window.setInterval(refreshHealth, 5000);
})();

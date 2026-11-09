"use strict";

const get = (id) => document.getElementById(id);
const setup = get("setup");
const host = window.__adobe_cep__;
const themeToggle = get("theme-toggle");
const themeKey = "3dscener.theme";
let source = null;
let scene = null;
let busy = false;
let basePython;
let readiness = null;
const logPanel = get("python-log");
const logToggle = get("open-log");
const logOutput = get("log-output");
const logLines = [];

function renderLog() {
  if (!logLines.length) return;
  const follow = logOutput.scrollHeight - logOutput.scrollTop - logOutput.clientHeight < 20;
  logOutput.textContent = logLines.join("\n");
  if (follow) logOutput.scrollTop = logOutput.scrollHeight;
}

function logPython(message) {
  const time = new Date().toLocaleTimeString();
  logLines.push("[" + time + "] " + message.slice(0, 1600));
  if (logLines.length > 300) logLines.shift();
  if (!logPanel.hidden) renderLog();
}

logToggle.addEventListener("click", () => {
  logPanel.hidden = !logPanel.hidden;
  logToggle.setAttribute("aria-expanded", String(!logPanel.hidden));
  logToggle.title = logPanel.hidden ? "Show Python log" : "Hide Python log";
  if (!logPanel.hidden) {
    renderLog();
    logOutput.scrollTop = logOutput.scrollHeight;
    logOutput.focus();
  }
});

function applyTheme(theme) {
  const light = theme === "light";
  document.documentElement.dataset.theme = light ? "light" : "dark";
  themeToggle.setAttribute("aria-pressed", String(light));
  themeToggle.title = light ? "Switch to dark theme" : "Switch to light theme";
}

let savedTheme;
try {
  savedTheme = localStorage.getItem(themeKey);
} catch (_) {
  // Theme switching still works when storage is unavailable.
}
applyTheme(savedTheme);
themeToggle.addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  applyTheme(theme);
  try {
    localStorage.setItem(themeKey, theme);
  } catch (_) {
    // Keep the selected theme for this session.
  }
});

function status(message, error = false) {
  get("status").textContent = message;
  get("indicator").classList.toggle("error", error);
}

function showSetup() {
  if (!setup.open) setup.showModal();
}

function setupMessage(message) {
  get("setup-status").textContent = message;
}

let checkSetup;
get("menu").addEventListener("click", () => {
  showSetup();
  if (checkSetup) checkSetup();
});
get("close").addEventListener("click", () => setup.close());

try {
  if (!host) throw new Error("Open 3DScener inside After Effects.");
  const node = window.cep_node && window.cep_node.require ? window.cep_node : window;
  if (typeof node.require !== "function") throw new Error("The CEP Node runtime is unavailable.");
  checkSetup = startPanel(createLocalRuntime(node, logPython));
} catch (error) {
  const message = "3DScener could not start. " + error.message;
  for (const button of setup.querySelectorAll("button:not(#close)")) button.disabled = true;
  logPython(message);
  setupMessage(message);
  status(message, true);
}

function startPanel(local) {
  const { workerCall, runtime, models } = local;
  let previewURL;
  let previewRequest = 0;
  const sceneModelPickers = {};
  const modelSections = {};
  const picker = get("image-picker");
  const drop = get("drop");
  const imagePreview = get("image-preview");
  const imageContent = get("image-content");
  const thumbnail = get("thumbnail");
  const dropPrompt = get("drop-prompt");
  const selectionPanel = get("object-selection");
  const selectionActions = get("selection-actions");
  const stopProcessing = get("stop-processing");
  const selectionMask = get("selection-mask");
  const selectionImage = selectionMask.querySelector("image");
  const selectionTrace = get("selection-trace");
  const selectionOutline = selectionTrace.querySelector("polygon");
  const selectionStatus = get("selection-status");
  const selectionModes = { "select-object": "select", "add-selection": "add", "remove-selection": "remove" };
  let selecting = false;
  let processing = null;
  let selectionMode = null;
  let selectionModel = null;
  const promptPreferenceKey = "3dscener.selection-model";
  let promptPreference = localStorage.getItem(promptPreferenceKey);
  let draftSelection = null;
  let savedObjects = [];
  let editingObject = null;
  let stroke = null;
  let pan = null;
  let zoom = 1;
  let zoomCenter = [.5, .5];

  function showModelMeta(button, key) {
    const model = models[key];
    const meta = button.querySelector(".model-meta");
    const verified = Boolean(readiness && readiness.models[key]);
    meta.textContent = verified ? "✓" : model.downloadSize;
    meta.classList.toggle("verified", verified);
    if (verified) button.setAttribute("aria-label", model.label + ", installed");
    else button.removeAttribute("aria-label");
  }

  function modelReady(key) {
    return Boolean(readiness && readiness.models[key] && (key !== "mat" || readiness.matRuntime));
  }

  function modelKeys(role) {
    return Object.keys(models).filter((key) => models[key].role === role ||
      (role === "prompt" && models[key].selection));
  }

  function promptModel() {
    return selectionModel || sceneModelPickers.prompt.value || "hq-sam-2";
  }

  function usesYoloSelection() {
    return models[promptModel()].architecture === "yolo26-seg";
  }

  function refreshUI() {
    for (const id of ["drop", "replace", "recheck", "python-action"]) get(id).disabled = busy;
    if (!draftSelection && !savedObjects.length) selectionModel = null;
    for (const [role, picker] of Object.entries(sceneModelPickers)) {
      if (role === "prompt") {
        const preference = picker.keys.includes(promptPreference) ? promptPreference : "hq-sam-2";
        const available = modelReady(preference) ? preference : picker.keys.find(modelReady);
        picker.refresh(selectionModel || available || "", modelReady, Boolean(selectionModel));
        continue;
      }
      const selected = modelReady(picker.value) ? picker.value : picker.keys.find(modelReady) || "";
      if (selected !== picker.value) scene = null;
      picker.refresh(selected, modelReady);
    }
    get("selection-model-hint").textContent = selectionModel ?
      "Clear the current masks to switch models." : "Used for Select Object, Add, and Remove.";
    get("process").disabled = busy || !source || (selecting && (!savedObjects.length || Boolean(draftSelection)));
    get("process-label").textContent = selecting ? "Process Selected" : "Process Image";
    get("create").disabled = busy || !scene;
    selectionPanel.hidden = !selecting;
    selectionActions.hidden = !selecting;
    const stopping = Boolean(processing && processing.signal.aborted);
    stopProcessing.disabled = !selecting || stopping || (busy && !processing);
    stopProcessing.textContent = stopping ? "Stopping…" : "Stop Processing";
    const idlePreview = selecting && !selectionMode;
    drop.classList.toggle("selecting", selecting && Boolean(selectionMode));
    drop.setAttribute("aria-disabled", String(idlePreview));
    drop.tabIndex = idlePreview ? -1 : 0;
    let imageAction = "Click or draw around an object to select it";
    if (idlePreview) imageAction = "Image preview";
    else if (selectionMode === "remove") imageAction = "Draw around an object to remove it";
    else if (selectionMode === "select" && !usesYoloSelection()) imageAction = "Click an object in the photo";
    drop.setAttribute("aria-label", selecting ? imageAction : source ? "Replace image" : "Browse for an image");
    for (const [id, mode] of Object.entries(selectionModes)) {
      get(id).disabled = busy || (mode !== "select" && !draftSelection);
      get(id).setAttribute("aria-pressed", String(mode === selectionMode));
    }
    get("save-object").textContent = editingObject ? "Update in List" : "Add to List";
    get("save-object").disabled = busy || !draftSelection || (!editingObject && savedObjects.length >= 12);
    get("discard-selection").textContent = editingObject ? "Cancel Edit" : "Discard Mask";
    get("discard-selection").hidden = !draftSelection;
    get("discard-selection").disabled = busy;
    get("auto-objects").disabled = busy;
    for (const button of get("selected-objects").querySelectorAll("button")) button.disabled = busy;
    get("python-ready").hidden = !basePython;
    get("python-action").hidden = Boolean(basePython);
    get("python-state").textContent = basePython ? "Ready · " + basePython.version :
      basePython === undefined ? "Waiting for Python check…" : "64-bit Python 3.11–3.13 not found";
    const installed = readiness && readiness.runtime;
    get("runtime-state").textContent = installed ? "DirectML GPU check passed" : "GPU runtime required";
    get("runtime-ready").hidden = !installed;
    get("runtime-action").hidden = Boolean(installed);
    get("runtime-action").disabled = busy || !basePython || installed;
    for (const section of Object.values(modelSections)) section.refresh();
  }

  async function inspect(recheckPython = false, signal) {
    if (recheckPython || !basePython) {
      basePython = await local.findPython(null, localStorage.getItem("3dscener.python"), signal);
    }
    if (local.hasRuntime()) {
      try {
        readiness = await workerCall([runtime], ["check"], undefined, undefined, signal);
        return;
      } catch (error) {
        if (!basePython || (signal && signal.aborted)) throw error;
      }
    }
    if (basePython) {
      readiness = await workerCall(basePython.command, ["check"], undefined, undefined, signal);
      // Require the isolated runtime even if another installation has the same modules.
      readiness.runtime = false;
      readiness.matRuntime = false;
    } else {
      readiness = null;
    }
  }

  async function task(action, onError) {
    if (busy) return;
    busy = true;
    refreshUI();
    try {
      await action();
    } catch (error) {
      onError(error.message);
    } finally {
      busy = false;
      refreshUI();
    }
  }

  function closeSelection() {
    selecting = false;
    selectionModel = null;
    savedObjects = [];
    clearDraft();
    renderObjectList();
  }

  function processingTask(action, onError) {
    if (busy) return;
    const controller = new AbortController();
    processing = controller;
    return task(async () => {
      try {
        await action(controller.signal);
      } finally {
        processing = null;
        if (controller.signal.aborted) {
          closeSelection();
          status("Processing stopped.");
        }
      }
    }, (error) => {
      if (!controller.signal.aborted) onError(error);
    });
  }

  stopProcessing.addEventListener("click", () => {
    if (!selecting || stopProcessing.disabled) return;
    if (processing) {
      processing.abort();
      status("Stopping processing…");
    } else {
      closeSelection();
      status("Selection closed.");
    }
    refreshUI();
  });

  function setupCheck() {
    showSetup();
    return task(async () => {
      setupMessage("Checking Python, the GPU runtime and model files…");
      await inspect(true);
      if (!basePython && !readiness) {
        setupMessage("Install Python from python.org, then use Locate Python.");
      } else {
        setupMessage(local.hasRuntime() && readiness && readiness.runtimeError ? readiness.runtimeError : "");
      }
    }, setupMessage);
  }

  function chooseFile(title, types) {
    const fs = window.cep && window.cep.fs;
    if (!fs || !fs.showOpenDialog) throw new Error("The CEP file picker is unavailable.");
    const dialog = fs.showOpenDialog(false, false, title, "", types);
    if (!dialog || dialog.err) throw new Error("The file picker could not open.");
    return dialog.data && dialog.data[0];
  }

  async function showPreview(file) {
    const request = ++previewRequest;
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = undefined;
    thumbnail.removeAttribute("src");
    imagePreview.hidden = true;
    endPan();
    zoom = 1;
    zoomCenter = [.5, .5];
    dropPrompt.hidden = false;
    get("preview-state").textContent = "Loading preview…";
    try {
      if (!/\.(jpe?g|png)$/i.test(file.name)) {
        if (!local.hasRuntime()) {
          get("preview-state").textContent = "Install the local runtime in Model Setup to preview this format.";
          return;
        }
        const png = await local.withImagePath(file, (filename) => workerCall([runtime], ["preview", filename]));
        if (request !== previewRequest) return;
        thumbnail.src = "data:image/png;base64," + png;
      } else {
        previewURL = URL.createObjectURL(file);
        thumbnail.src = previewURL;
      }
      await thumbnail.decode();
      if (request !== previewRequest) return;
      imagePreview.hidden = false;
      dropPrompt.hidden = true;
      layoutPreview();
    } catch (error) {
      if (request === previewRequest) get("preview-state").textContent = error.message || "Could not preview this image.";
    }
  }

  function selectFile(file) {
    if (!file || busy) return;
    source = file;
    scene = null;
    selecting = false;
    selectionModel = null;
    savedObjects = [];
    clearDraft();
    renderObjectList();
    get("filename").textContent = file.name;
    get("filename").title = file.name;
    showPreview(file);
    get("file").hidden = false;
    status("Ready to process.");
    refreshUI();
  }

  function browse() {
    if (!busy) picker.click();
  }

  drop.addEventListener("click", (event) => {
    if (!selecting) browse();
    else if (selectionMode === "select" && !usesYoloSelection() && event.detail) selectAtPoint(event);
  });
  get("replace").addEventListener("click", browse);
  picker.addEventListener("change", () => {
    const file = picker.files[0];
    picker.value = "";
    selectFile(file);
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  get("drop").addEventListener("dragover", () => {
    if (!busy) get("drop").classList.add("active");
  });
  get("drop").addEventListener("dragleave", () => get("drop").classList.remove("active"));
  get("drop").addEventListener("drop", (event) => {
    event.preventDefault();
    get("drop").classList.remove("active");
    selectFile(event.dataTransfer.files[0]);
  });
  window.addEventListener("beforeunload", () => {
    if (previewURL) URL.revokeObjectURL(previewURL);
  });
  function modelChanged() {
    scene = null;
    status(source ? "Ready to process with the selected model." : "Choose an image to get started.");
    refreshUI();
  }

  function clearDraft() {
    draftSelection = null;
    editingObject = null;
    stroke = null;
    selectionMode = null;
    selectionTrace.setAttribute("hidden", "");
    selectionMask.setAttribute("hidden", "");
    selectionImage.removeAttribute("href");
  }

  function showSelectionMask(overlay) {
    selectionImage.setAttribute("href", overlay);
    selectionMask.removeAttribute("hidden");
  }

  function renderObjectList() {
    const list = get("selected-objects");
    list.textContent = "";
    savedObjects.forEach((object, index) => {
      const item = document.createElement("li");
      const name = "Object " + String(index + 1).padStart(2, "0");
      item.textContent = name;
      const edit = document.createElement("button");
      edit.type = "button";
      edit.textContent = "Edit";
      edit.setAttribute("aria-label", "Edit " + name);
      edit.addEventListener("click", () => {
        editingObject = object;
        draftSelection = object.selection;
        selectionMode = null;
        showSelectionMask(object.overlay);
        selectionStatus.textContent = "Editing " + name + ". Choose Add or Remove to refine the mask, then click Update in List.";
        refreshUI();
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove from list";
      remove.setAttribute("aria-label", "Remove " + name + " from list");
      remove.addEventListener("click", () => {
        savedObjects.splice(index, 1);
        if (editingObject === object) clearDraft();
        if (!draftSelection) selectionStatus.textContent = selectionInstruction();
        renderObjectList();
        refreshUI();
      });
      item.append(edit, remove);
      list.appendChild(item);
    });
  }

  function selectionInstruction() {
    return savedObjects.length ?
      "Choose Select Object to add another mask, or click Process Selected." :
      "Choose Select Object to start masking, or choose Auto Only.";
  }

  function showMissingModel(key, role = models[key].role) {
    showSetup();
    modelSections[role].reveal(key);
    setupMessage("Download " + models[key].label + " to continue.");
  }

  function layoutPreview() {
    const bounds = imagePreview.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const scale = zoom * Math.min(bounds.width / thumbnail.naturalWidth, bounds.height / thumbnail.naturalHeight);
    const width = thumbnail.naturalWidth * scale;
    const height = thumbnail.naturalHeight * scale;
    function offset(viewSize, imageSize, center) {
      if (imageSize <= viewSize) return (viewSize - imageSize) / 2;
      return Math.max(viewSize - imageSize, Math.min(0, viewSize / 2 - center * imageSize));
    }
    const left = offset(bounds.width, width, zoomCenter[0]);
    const top = offset(bounds.height, height, zoomCenter[1]);
    Object.assign(imageContent.style, {
      left: left + "px", top: top + "px", width: width + "px", height: height + "px",
    });
    zoomCenter = [(bounds.width / 2 - left) / width, (bounds.height / 2 - top) / height];
  }

  const previewResize = new ResizeObserver(layoutPreview);
  previewResize.observe(imagePreview);

  drop.addEventListener("wheel", (event) => {
    if (!event.ctrlKey || imagePreview.hidden) return;
    event.preventDefault();
    if (stroke || pan || !event.deltaY) return;
    const bounds = imagePreview.getBoundingClientRect();
    const frame = imageContent.getBoundingClientRect();
    if (!frame.width || !frame.height) return;
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? bounds.height : 1;
    const nextZoom = Math.max(1, Math.min(8, zoom * Math.exp(-event.deltaY * unit * .002)));
    if (nextZoom === zoom) return;
    const [x, y] = imagePoint(event, frame);
    const factor = nextZoom / zoom;
    zoomCenter = [
      x + (bounds.left + bounds.width / 2 - event.clientX) / (frame.width * factor),
      y + (bounds.top + bounds.height / 2 - event.clientY) / (frame.height * factor),
    ];
    zoom = nextZoom;
    layoutPreview();
  }, { passive: false });

  function imagePoint(event, frame) {
    return [(event.clientX - frame.left) / frame.width, (event.clientY - frame.top) / frame.height];
  }

  function selectAtPoint(event) {
    if (busy || imagePreview.hidden) return;
    const point = imagePoint(event, imageContent.getBoundingClientRect());
    if (!point.every((value) => value >= 0 && value <= 1)) return;
    return previewSelection({ point, edits: [] }, "Snapping to the object…");
  }

  function previewSelection(selection, message) {
    const key = promptModel();
    if (!modelReady(key)) {
      showMissingModel(key, "prompt");
      return;
    }
    if (!local.hasRuntime()) {
      showSetup();
      setupMessage("Install the local runtime to select objects.");
      return;
    }
    return processingTask(async (signal) => {
      selectionStatus.textContent = message + " " + models[key].label;
      const overlay = await local.withImagePath(source, (filename) =>
        workerCall([runtime], ["mask", filename, "-", key], undefined, selection, signal));
      if (signal.aborted) return;
      selectionModel = key;
      showSelectionMask("data:image/png;base64," + overlay);
      draftSelection = selection;
      if (selectionMode === "select") selectionMode = null;
      selectionStatus.textContent = editingObject ? "Choose Add or Remove to refine the mask, then click Update in List." :
        "Choose Add or Remove to refine the mask, or click Add to List.";
    }, (error) => { selectionStatus.textContent = error; });
  }

  drop.addEventListener("pointerdown", (event) => {
    if (event.button === 1 && !imagePreview.hidden) {
      event.preventDefault();
      if (zoom === 1 || busy || stroke || pan) return;
      pan = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
      drop.classList.add("panning");
      drop.setPointerCapture(event.pointerId);
      return;
    }
    if (!selecting || busy || stroke || !selectionMode || imagePreview.hidden || event.button !== 0) return;
    if (selectionMode === "select" ? !usesYoloSelection() : !draftSelection) return;
    const point = imagePoint(event, imageContent.getBoundingClientRect());
    if (!point.every((value) => value >= 0 && value <= 1)) return;
    if (selectionMode !== "select" && draftSelection.edits.length >= 64) {
      selectionStatus.textContent = "This mask has 64 freehand edits. Save it or start a new mask.";
      return;
    }
    event.preventDefault();
    selectionTrace.classList.toggle("removing", selectionMode === "remove");
    selectionOutline.setAttribute("points", point.join(","));
    selectionTrace.removeAttribute("hidden");
    stroke = { pointerId: event.pointerId, mode: selectionMode, points: [point], dragged: false };
    drop.setPointerCapture(event.pointerId);
  });

  function extendStroke(event) {
    const { points } = stroke;
    const frame = imageContent.getBoundingClientRect();
    const point = imagePoint(event, frame).map((value) => Math.max(0, Math.min(1, value)));
    const [startX, startY] = points[0];
    if (Math.hypot((point[0] - startX) * frame.width, (point[1] - startY) * frame.height) >= 4) {
      stroke.dragged = true;
    }
    const [x, y] = points[points.length - 1];
    if (Math.hypot((point[0] - x) * frame.width, (point[1] - y) * frame.height) < 2) return;
    if (points.length === 512) stroke.points = points.filter((_, index) => index % 2 === 0);
    stroke.points.push(point);
    selectionOutline.setAttribute("points", stroke.points.map((position) => position.join(",")).join(" "));
  }

  drop.addEventListener("pointermove", (event) => {
    if (pan && pan.pointerId === event.pointerId) {
      if (!(event.buttons & 4)) {
        endPan();
        return;
      }
      const frame = imageContent.getBoundingClientRect();
      zoomCenter = [
        zoomCenter[0] - (event.clientX - pan.x) / frame.width,
        zoomCenter[1] - (event.clientY - pan.y) / frame.height,
      ];
      pan.x = event.clientX;
      pan.y = event.clientY;
      layoutPreview();
      return;
    }
    if (stroke && stroke.pointerId === event.pointerId) extendStroke(event);
  });

  function endPan() {
    if (!pan) return;
    const { pointerId } = pan;
    pan = null;
    drop.classList.remove("panning");
    if (drop.hasPointerCapture(pointerId)) drop.releasePointerCapture(pointerId);
  }

  function endStroke(event) {
    if (!stroke || stroke.pointerId !== event.pointerId) return;
    if (event.type === "pointerup") extendStroke(event);
    const { mode, points, dragged } = stroke;
    stroke = null;
    selectionTrace.setAttribute("hidden", "");
    if (event.type !== "pointerup") return;
    if (mode === "select") {
      if (!dragged) previewSelection({ point: points[0], edits: [] }, "Snapping to the object…");
      else if (points.length >= 3) previewSelection({ outline: points, edits: [] }, "Snapping to the outline…");
      else selectionStatus.textContent = "Drag around an object and release to close the shape.";
      return;
    }
    if (mode === "add" && !dragged) {
      previewSelection({ ...draftSelection, edits: [...draftSelection.edits, { mode, point: points[0] }] }, "Snapping to the object…");
      return;
    }
    if (!dragged || points.length < 3) {
      selectionStatus.textContent = "Drag around an area and release to close the shape.";
      return;
    }
    previewSelection({ ...draftSelection, edits: [...draftSelection.edits, { mode, points }] }, "Snapping to the outline…");
  }

  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    drop.addEventListener(type, (event) => {
      if (pan && pan.pointerId === event.pointerId) endPan();
      else endStroke(event);
    });
  }

  for (const type of ["mousedown", "auxclick"]) {
    drop.addEventListener(type, (event) => {
      if (event.button === 1 && !imagePreview.hidden) event.preventDefault();
    });
  }

  for (const [id, mode] of Object.entries(selectionModes)) {
    get(id).addEventListener("click", () => {
      selectionMode = mode;
      selectionStatus.textContent = mode === "remove" ? "Draw around the object or detail to snap and remove." :
        mode === "add" ? "Click an object to add it, or drag around the area to snap and add." :
        usesYoloSelection() ? "Click or draw around an object to start a new mask." : "Click an object to start a new mask.";
      refreshUI();
    });
  }
  get("save-object").addEventListener("click", () => {
    if (!draftSelection || (!editingObject && savedObjects.length >= 12)) return;
    const object = { selection: draftSelection, overlay: selectionImage.getAttribute("href") };
    if (editingObject) savedObjects[savedObjects.indexOf(editingObject)] = object;
    else savedObjects.push(object);
    clearDraft();
    renderObjectList();
    selectionStatus.textContent = selectionInstruction();
    refreshUI();
  });
  get("discard-selection").addEventListener("click", () => {
    clearDraft();
    selectionStatus.textContent = selectionInstruction();
    refreshUI();
  });
  get("recheck").addEventListener("click", setupCheck);
  get("python-action").addEventListener("click", () => task(async () => {
    const chosen = chooseFile("Locate python.exe", ["exe"]);
    if (!chosen) return;
    const located = await local.findPython(chosen);
    if (!located) throw new Error("Select a working 64-bit Python 3.11, 3.12 or 3.13 executable.");
    basePython = located;
    localStorage.setItem("3dscener.python", chosen);
    await inspect();
    setupMessage("Python is ready.");
  }, setupMessage));
  get("runtime-action").addEventListener("click", () => task(async () => {
    setupMessage("Installing the DirectML runtime. This may take a few minutes.");
    await local.install(basePython.command);
    await inspect();
    if (!readiness || !readiness.runtime) {
      throw new Error((readiness && readiness.runtimeError) || "The GPU runtime could not be loaded.");
    }
    setupMessage("GPU runtime is ready.");
  }, setupMessage));
  function createModelPicker(id, heading, keys, onChoose, dialog) {
    const picker = get("model-picker-template").content.firstElementChild.cloneNode(true);
    const trigger = picker.querySelector(".model-trigger");
    const name = picker.querySelector(".model-name");
    const list = picker.querySelector(".model-list");
    const rows = new Map();
    let value = "";

    trigger.id = id;
    list.id = id + "-list";
    name.id = id + "-name";
    trigger.setAttribute("aria-labelledby", heading.id + " " + name.id);
    trigger.setAttribute("aria-controls", list.id);
    list.setAttribute("aria-labelledby", heading.id);

    function closeList() {
      list.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
    }

    function enabledRows() {
      return Array.from(rows.values()).filter((row) => !row.disabled);
    }

    function openList(step = 1) {
      list.hidden = false;
      const bounds = trigger.getBoundingClientRect();
      const top = dialog ? dialog.getBoundingClientRect().top + dialog.clientTop : 0;
      const bottom = dialog ? top + dialog.clientHeight : window.innerHeight;
      const above = bounds.top - top - 8;
      const below = bottom - bounds.bottom - 8;
      const opensUp = below < Math.min(185, list.scrollHeight + 2) && above > below;
      list.classList.toggle("opens-up", opensUp);
      list.style.setProperty("--model-list-space", Math.max(0, opensUp ? above : below) + "px");
      trigger.setAttribute("aria-expanded", "true");
      const options = enabledRows();
      const selected = rows.get(value);
      const target = selected && !selected.disabled ? selected : options[step > 0 ? 0 : options.length - 1];
      if (target) target.focus();
    }

    let family;
    for (const key of keys) {
      const model = models[key];
      if (!dialog && model.family && model.family !== family) {
        family = model.family;
        const group = document.createElement("div");
        group.className = "model-list-heading";
        group.textContent = "Depth Anything " + family.toUpperCase();
        list.appendChild(group);
      }
      const row = document.createElement("button");
      row.type = "button";
      row.className = "model-choice";
      const label = document.createElement("span");
      label.textContent = model.label;
      const meta = document.createElement("span");
      meta.className = "model-meta";
      row.append(label, meta);
      row.addEventListener("click", () => {
        closeList();
        onChoose(key);
        trigger.focus();
      });
      rows.set(key, row);
      list.appendChild(row);
    }

    trigger.addEventListener("click", () => {
      if (list.hidden) openList();
      else closeList();
    });
    picker.addEventListener("focusout", (event) => {
      if (!picker.contains(event.relatedTarget)) closeList();
    });
    picker.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !list.hidden) {
        event.preventDefault();
        closeList();
        trigger.focus();
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const step = event.key === "ArrowDown" ? 1 : -1;
        if (list.hidden) {
          openList(step);
          return;
        }
        const options = enabledRows();
        if (!options.length) return;
        const index = options.indexOf(event.target);
        const next = index < 0 ? (step > 0 ? 0 : options.length - 1) : (index + step + options.length) % options.length;
        options[next].focus();
      }
    });
    if (dialog) dialog.addEventListener("close", closeList);
    heading.after(picker);

    return {
      keys, trigger, close: closeList,
      get value() { return value; },
      refresh(next, canChoose = () => true, locked = false) {
        value = next;
        name.textContent = value ? models[value].label : "Download a model in Model Setup";
        trigger.querySelector(".model-meta").hidden = !value;
        if (value) showModelMeta(trigger, value);
        else trigger.removeAttribute("aria-label");
        for (const [key, row] of rows) {
          row.disabled = !canChoose(key);
          row.setAttribute("aria-pressed", String(key === value));
          showModelMeta(row, key);
        }
        if (busy || locked) closeList();
        trigger.disabled = busy || locked;
      },
    };
  }

  function addModelSection(role, title) {
    const section = get("model-section-template").content.firstElementChild.cloneNode(true);
    const heading = section.querySelector("h3");
    const state = section.querySelector(".model-state");
    const download = section.querySelector(".download-model");
    const remove = section.querySelector(".delete-model");
    const progress = section.querySelector("progress");
    const message = section.querySelector(".model-message");
    const keys = modelKeys(role);
    let key = keys[0];

    heading.textContent = title;
    heading.id = role + "-setup-title";
    state.id = role + "-setup-state";
    section.setAttribute("aria-labelledby", heading.id);
    progress.setAttribute("aria-label", title + " download progress");
    const picker = createModelPicker(role + "-setup-picker", heading, keys, selectModel, setup);
    picker.trigger.setAttribute("aria-describedby", state.id);

    function selectModel(next) {
      if (key !== next) {
        key = next;
        progress.hidden = true;
        message.textContent = "";
      }
      picker.close();
      refresh();
    }

    function refresh() {
      const present = Boolean(readiness && readiness.models[key]);
      const pythonAvailable = basePython || local.hasRuntime();
      const needsMatSupport = key === "mat" && !(readiness && readiness.matRuntime);
      const needsLocalRuntime = needsMatSupport && !local.hasRuntime();
      if (needsLocalRuntime) {
        state.textContent = "Install the local runtime to use MAT.";
      } else if (needsMatSupport) {
        state.textContent = present ? "Model downloaded. MAT CPU support is required." :
          "Download includes MAT CPU support packages.";
      } else {
        state.textContent = present ? "Installed · checksum verified" :
          pythonAvailable ? "Ready to download" : "Python is required to download.";
      }
      picker.refresh(key);
      download.textContent = present && needsMatSupport ? "Install MAT support" : "Download";
      download.hidden = present && !needsMatSupport;
      download.disabled = busy || download.hidden || !pythonAvailable || needsLocalRuntime;
      remove.hidden = !present;
      remove.disabled = busy || !pythonAvailable;
    }

    download.addEventListener("click", () => task(async () => {
      const python = readiness && readiness.runtime ? [runtime] :
        basePython ? basePython.command : [runtime];
      progress.removeAttribute("value");
      progress.hidden = false;
      message.textContent = "";
      if (key === "mat" && !(readiness && readiness.matRuntime)) {
        state.textContent = "Installing MAT CPU support…";
        await local.installMat();
      }
      state.textContent = "Starting download…";
      await workerCall(python, ["download", key], (event) => {
        if (event.total) {
          // Show 100% only after the worker verifies the checksum.
          progress.value = Math.min(99, Math.floor(event.received / event.total * 100));
        }
        state.textContent = event.total ? "Downloading… " + progress.value + "%" : "Downloading…";
      });
      progress.value = 100;
      if (key === "mat" || !readiness) await inspect();
      else readiness.models[key] = true;
      if (key === "mat" && !modelReady(key)) throw new Error("MAT setup could not be verified. Try again.");
      message.textContent = "Download complete.";
    }, (error) => {
      progress.hidden = true;
      message.textContent = error;
    }));
    remove.addEventListener("click", () => task(async () => {
      const python = basePython ? basePython.command : [runtime];
      progress.hidden = true;
      message.textContent = "Deleting " + models[key].label + "…";
      await workerCall(python, ["delete", key]);
      readiness.models[key] = false;
      message.textContent = models[key].label + " deleted.";
    }, (error) => { message.textContent = error; }));

    get("model-sections").appendChild(section);
    return {
      refresh,
      reveal(modelKey) {
        selectModel(modelKey);
        section.scrollIntoView({ block: "nearest" });
      },
    };
  }

  function processScene(auto) {
    return processingTask(async (signal) => {
      const selected = sceneModelPickers.depth.value || "v2";
      const segmentation = sceneModelPickers.segmentation.value || "mask-rcnn-compact";
      const fill = sceneModelPickers.fill.value || "lama";
      const prompt = promptModel();
      scene = null;
      status("Checking local dependencies…");
      await inspect(false, signal);
      if (signal.aborted) return;
      const required = auto ? [segmentation, selected, fill] : [prompt, segmentation, selected, fill];
      const missing = required.find((key) => !modelReady(key));
      if (!readiness || !readiness.runtime || missing) {
        if (missing) showMissingModel(missing, !auto && missing === prompt ? "prompt" : models[missing].role);
        else {
          showSetup();
          setupMessage("Install the GPU runtime to continue.");
        }
        status("Model setup required.", true);
        return;
      }
      const selection = auto ? undefined : savedObjects.map((object) => object.selection);
      const result = await local.withImagePath(source, (filename) =>
        workerCall([runtime], ["process", selected, filename, segmentation, fill, ...(auto ? [] : ["-", prompt])],
          (event) => status(event.progress), selection, signal));
      if (signal.aborted) return;
      scene = result;
      closeSelection();
      status("Scene ready. Create AE Layer to build a new composition.");
    }, (error) => status(error, true));
  }

  get("process").addEventListener("click", () => {
    if (busy || !source) return;
    if (selecting) {
      if (savedObjects.length && !draftSelection) processScene(false);
      return;
    }
    selecting = true;
    selectionModel = null;
    savedObjects = [];
    scene = null;
    clearDraft();
    renderObjectList();
    selectionStatus.textContent = modelReady(promptModel()) ? selectionInstruction() :
      "Download a model under Selection Models in Model Setup, or choose Auto Only.";
    status("Choose a tool to start object selection.");
    refreshUI();
  });
  get("auto-objects").addEventListener("click", () => processScene(true));
  get("create").addEventListener("click", () => task(async () => {
    status("Creating the composition…");
    const payload = JSON.stringify(scene).replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
    const result = await new Promise((resolve) => host.evalScript("create3DScenerScene(" + payload + ")", resolve));
    if (result !== "OK") throw new Error(result === "EvalScript error." ? "After Effects could not run the scene script." : result);
    status("Composition created. Play the timeline to see the parallax.");
  }, (error) => status(error, true)));
  for (const [role, id] of Object.entries({ prompt: "selection-model", segmentation: "segmentation-model", depth: "model", fill: "fill-model" })) {
    const keys = modelKeys(role);
    const picker = createModelPicker(id, get(id + "-label"), keys, (next) => {
      if (role === "prompt") {
        promptPreference = next;
        localStorage.setItem(promptPreferenceKey, next);
        refreshUI();
        return;
      }
      if (next === picker.value) return;
      picker.refresh(next, modelReady);
      modelChanged();
    });
    if (role === "prompt") picker.trigger.setAttribute("aria-describedby", "selection-model-hint");
    sceneModelPickers[role] = picker;
  }
  modelSections.prompt = addModelSection("prompt", "Selection Models");
  modelSections.segmentation = addModelSection("segmentation", "Object Segmentation Models");
  modelSections.depth = addModelSection("depth", "Depth Generation Model");
  modelSections.fill = addModelSection("fill", "Fill Space Model");
  task(() => inspect(), (error) => logPython("Model check: " + error));
  return setupCheck;
}

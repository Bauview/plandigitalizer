/* PlanDigitalizer – Oberfläche (ohne Framework) */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const app = $("#app");

  const state = { loaded: false, busy: false, points: [], view: { s: 1, tx: 0, ty: 0 }, size: [1, 1], urls: [] };
  const engine = window.PlanEngine;

  const LAYER_NAMES = {
    WALLS: "Wände", DOORS: "Türen", WINDOWS: "Fenster", STAIRS: "Treppen", TEXT: "Texte",
    DIMENSIONS: "Bemassung", LINES: "Linien", HATCH: "Füllungen", SYMBOLS: "Symbole", ROOMS: "Raumstempel",
  };
  const ICON_WARN = '<svg viewBox="0 0 16 16"><path d="M8 1.8 15 14H1z"/><path d="M8 6.2v3.6M8 11.6v.4"/></svg>';
  const ICON_CHECK = '<svg viewBox="0 0 20 20"><path d="m5 10.5 3.2 3.2L15 6.8"/></svg>';

  function show(view) {
    app.dataset.view = view;
    window.scrollTo({ top: 0 });
  }

  function showError(detail) {
    $("#errorDetail").textContent = detail || "";
    show("error");
  }

  // ------------------------------------------------------------ Upload
  const drop = $("#drop");
  const fileInput = $("#file");

  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) upload(f);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); });
  // Ablegen irgendwo auf der Seite verhindern, dass der Browser die Datei öffnet
  ["dragover", "drop"].forEach((ev) => window.addEventListener(ev, (e) => e.preventDefault()));

  async function upload(file) {
    const okExt = /\.(jpe?g|png|webp|pdf|tiff?|bmp)$/i.test(file.name);
    if (!okExt) {
      showError("Dateiformat wird nicht unterstützt. Erlaubt sind JPG, PNG, WEBP und PDF.");
      return;
    }
    if (file.size > 80 * 1024 * 1024) {
      showError("Die Datei ist grösser als 80 MB. Bitte eine kleinere Datei verwenden.");
      return;
    }
    discardJob();
    drop.classList.add("busy");
    try {
      const data = await engine.load(file);
      trackUrl(data.previewUrl);
      state.loaded = true;
      $(".file-name").textContent = file.name;
      const mp = (data.width * data.height / 1e6).toFixed(1);
      let meta = `${data.width} × ${data.height} px · ${mp} MP`;
      if (data.pages > 1) meta += ` · PDF mit ${data.pages} Seiten (Seite 1 wird verarbeitet)`;
      $(".file-meta").textContent = meta;
      const img = $("#previewImg");
      img.onload = () => { resetCalibration(); };
      img.src = data.previewUrl;
      show("preview");
    } catch (err) {
      showError(err.detail || err.message);
    } finally {
      drop.classList.remove("busy");
      fileInput.value = "";
    }
  }

  function trackUrl(u) { state.urls.push(u); return u; }

  // ------------------------------------------------------------ Rechenkern laden (einmalig)
  const loadBar = $("#engineLoad");
  engine.start((text, pct) => {
    $("#engineText").textContent = pct >= 100 ? "Bereit – Pläne werden nur in diesem Browser verarbeitet." : text;
    loadBar.style.setProperty("--pct", pct + "%");
    loadBar.classList.toggle("is-ready", pct >= 100);
  }).catch((err) => {
    $("#engineText").textContent = "Der Rechenkern konnte nicht geladen werden: " + err.message +
      " Bitte die Seite neu laden oder einen aktuellen Browser (Chrome, Edge, Firefox, Safari) verwenden.";
    loadBar.classList.add("is-error");
  });

  // ------------------------------------------------------------ Kalibrierung
  const stage = $("#previewStage");
  const canvas = $("#calibCanvas");
  const calib = $("#calib");
  const calibLen = $("#calibLen");
  const calibHelp = $("#calibHelp");

  calib.addEventListener("toggle", () => {
    app.classList.toggle("calibrating", calib.open);
    if (calib.open) calibLen.focus();
    drawCalibration();
  });
  $("#calibReset").addEventListener("click", resetCalibration);
  calibLen.addEventListener("input", updateCalibHelp);

  stage.addEventListener("click", (e) => {
    if (!calib.open) return;
    const r = stage.getBoundingClientRect();
    const p = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
    if (state.points.length >= 2) state.points = [];
    state.points.push(p);
    drawCalibration();
    updateCalibHelp();
  });

  function resetCalibration() {
    state.points = [];
    drawCalibration();
    updateCalibHelp();
  }

  function parseLength() {
    const v = parseFloat(String(calibLen.value).replace(",", ".").replace(/[^\d.]/g, ""));
    return isFinite(v) && v > 0 ? v : null;
  }

  function updateCalibHelp() {
    const n = state.points.length;
    const len = parseLength();
    calibHelp.classList.remove("ok");
    if (n === 0) calibHelp.textContent = "Klicken Sie im Plan auf den Anfang und das Ende einer Strecke mit bekannter Länge.";
    else if (n === 1) calibHelp.textContent = "Jetzt den Endpunkt der Strecke anklicken.";
    else if (!len) calibHelp.textContent = "Bitte die bekannte Länge in Metern eingeben.";
    else {
      calibHelp.textContent = `Kalibrierung bereit: ${len.toFixed(2)} m zwischen den beiden Punkten.`;
      calibHelp.classList.add("ok");
    }
  }

  function drawCalibration() {
    const img = $("#previewImg");
    const w = img.clientWidth, h = img.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!calib.open || !state.points.length) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
    const pts = state.points.map(([x, y]) => [x * w, y * h]);
    ctx.strokeStyle = accent; ctx.fillStyle = "#fff"; ctx.lineWidth = 2;
    if (pts.length === 2) {
      ctx.beginPath(); ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); ctx.stroke();
    }
    pts.forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    });
  }
  window.addEventListener("resize", drawCalibration);

  // ------------------------------------------------------------ 3D-/IFC-Einstellungen
  const DEFAULT_STOREYS = [{ name: "EG", height: "2.80" }];
  const SETTINGS_KEY = "plandigitalizer.settings3d";
  const storeyBody = $("#s3Storeys");
  const planStorey = $("#s3PlanStorey");
  const autoThick = $("#s3Auto");

  function num(id) { return String($(id).value).replace(",", "."); }

  function renderStoreys(rows, selected) {
    storeyBody.innerHTML = "";
    rows.forEach((r, i) => addStoreyRow(r.name, r.height, i));
    syncPlanStorey(selected);
  }
  function addStoreyRow(name, height) {
    const tr = document.createElement("tr");
    const n = storeyBody.children.length + 1;
    tr.innerHTML = `<td><input type="text" maxlength="20" aria-label="Geschossname" id="s3n${n}-${Date.now()}"></td>
      <td><span class="input-unit"><input type="text" inputmode="decimal" aria-label="Geschosshöhe in Metern"><span>m</span></span></td>
      <td><button class="rm" type="button" aria-label="Geschoss entfernen">×</button></td>`;
    const [ni, hi] = tr.querySelectorAll("input");
    ni.value = name; hi.value = height;
    ni.addEventListener("input", () => syncPlanStorey());
    tr.querySelector(".rm").addEventListener("click", () => {
      if (storeyBody.children.length > 1) { tr.remove(); syncPlanStorey(); }
    });
    storeyBody.appendChild(tr);
  }
  function storeyRows() {
    return [...storeyBody.children].map((tr) => {
      const [ni, hi] = tr.querySelectorAll("input");
      return { name: ni.value.trim(), height: String(hi.value).replace(",", ".") };
    }).filter((r) => r.name);
  }
  function syncPlanStorey(selected) {
    const cur = selected ?? planStorey.value ?? "auto";
    const names = storeyRows().map((r) => r.name);
    planStorey.innerHTML = '<option value="auto">Automatisch erkennen</option>' +
      names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("");
    planStorey.value = names.includes(cur) ? cur : "auto";
  }
  $("#s3AddStorey").addEventListener("click", () => {
    const names = storeyRows().map((r) => r.name);
    const next = ["EG", "OG", "UG", "DG"].find((n) => !names.includes(n)) || `${names.length}.OG`;
    addStoreyRow(next, next === "UG" || next === "DG" ? "2.50" : "2.80");
    if (next === "UG") storeyBody.prepend(storeyBody.lastElementChild);   // Untergeschoss zuunterst
    syncPlanStorey();
  });
  autoThick.addEventListener("change", () => { $("#s3Thick").disabled = autoThick.checked; });

  function readSettings3d() {
    return {
      wall_height: num("#s3WallH"), auto_thickness: autoThick.checked, default_thickness: num("#s3Thick"),
      door_height: num("#s3DoorH"), window_height: num("#s3WinH"), sill_height: num("#s3Sill"),
      storeys: storeyRows(), plan_storey: planStorey.value,
      detail: $("#optAll").checked ? "all" : "essential",
    };
  }
  function saveSettings3d(v) { try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(v)); } catch (_) { /* egal */ } }
  function loadSettings3d() {
    let v = null;
    try { v = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch (_) { v = null; }
    if (v) {
      const set = (id, val) => { if (val != null && val !== "") $(id).value = val; };
      set("#s3WallH", v.wall_height); set("#s3Thick", v.default_thickness); set("#s3DoorH", v.door_height);
      set("#s3WinH", v.window_height); set("#s3Sill", v.sill_height);
      autoThick.checked = v.auto_thickness !== false;
      $("#optAll").checked = v.detail === "all";
    }
    $("#s3Thick").disabled = autoThick.checked;
    renderStoreys(v && v.storeys && v.storeys.length ? v.storeys : DEFAULT_STOREYS, v ? v.plan_storey : "auto");
  }
  loadSettings3d();

  // ------------------------------------------------------------ Verarbeitung
  const STEP_LABELS = ["Plan wird ausgerichtet, Wände und Öffnungen werden erkannt …", "Linien werden erkannt …", "Plan wird vektorisiert …",
    "Dateien werden erstellt …", "IFC- und 3D-Modell werden erstellt …"];

  function renderSteps(stages, current) {
    $("#steps").innerHTML = stages.map((label, i) => {
      const cls = i < current ? "done" : i === current ? "active" : "";
      const ico = i < current ? ICON_CHECK : i === current ? '<span class="spinner"></span>' : '<span class="dot"></span>';
      return `<li class="${cls}"><span class="ico">${ico}</span>${label}</li>`;
    }).join("");
  }

  async function startProcessing() {
    if (!state.loaded || state.busy) return;
    const opts = {};
    const len = parseLength();
    if (calib.open && state.points.length === 2 && len) {
      opts.calibration = { p1: state.points[0], p2: state.points[1], length_m: len };
    }
    opts.settings3d = readSettings3d();
    saveSettings3d(opts.settings3d);
    const labels = engine.stages.length ? engine.stages : STEP_LABELS;
    renderSteps(labels, 0);
    show("processing");
    state.busy = true;
    try {
      const r = await engine.process({ ...opts, onStage: (i) => renderSteps(labels, i) });
      r.downloads && Object.values(r.downloads).forEach((d) => trackUrl(d.url));
      trackUrl(r.processed);
      renderSteps(labels, labels.length);
      setTimeout(() => showResult(r), 300);
    } catch (err) {
      showError(err.detail || err.message);
    } finally {
      state.busy = false;
    }
  }

  // ------------------------------------------------------------ Ergebnis
  const viewer = $("#viewer");

  function showResult(r) {
    state.size = r.size;
    $("#resOriginal").src = r.processed;
    $("#resVector").innerHTML = r.svg;
    $$(".layer").forEach((el) => { el.style.width = r.size[0] + "px"; el.style.height = r.size[1] + "px"; });

    $("#notices").innerHTML = (r.warnings || []).map((w) => `<li>${ICON_WARN}<span>${escapeHtml(w)}</span></li>`).join("");

    const order = ["WALLS", "DOORS", "WINDOWS", "STAIRS", "TEXT", "DIMENSIONS", "LINES", "SYMBOLS"];
    const rec = r.info && r.info.recognition;
    const parts = rec
      ? [`${rec.walls} Wandabschnitte`, `${rec.doors} Türen`, `${rec.windows} Fenster`,
        ...["STAIRS", "TEXT", "DIMENSIONS", "LINES", "SYMBOLS"].filter((k) => r.layers[k]).map((k) => `${LAYER_NAMES[k]} ${r.layers[k]}`)]
      : order.filter((k) => r.layers[k]).map((k) => `${LAYER_NAMES[k]} ${r.layers[k]}`);
    const unit = r.unit === "mm" ? "Einheit mm" : "Einheit Pixel";
    $("#summary").textContent = [r.scale_note, unit, parts.join(" · ")].filter(Boolean).join("  ·  ");

    setDl("#dlPdf", r.downloads.pdf);
    setDl("#dlCad", r.downloads.dxf);
    $("#dlCadLabel").textContent = "DXF herunterladen";
    $("#cadHint").textContent = "Die DXF-Datei öffnet sich in jedem CAD (AutoCAD, Vectorworks, ArchiCAD …) und lässt sich dort als DWG speichern.";

    currentDownloads = { ...r.downloads, stem: r.stem };
    showBim(r.bim || { available: false, reason: "IFC/3D wurde nicht erzeugt." });

    setMode(window.matchMedia("(max-width: 720px)").matches ? "vector" : "split");
    show("result");
    requestAnimationFrame(fit);
  }

  // ------------------------------------------------------------ BIM / 3D
  let bim = null;
  let viewer3d = null;
  let currentDownloads = {};
  let exportToken = 0;

  async function prepare3dFiles(model) {
    const token = ++exportToken;
    try {
      const { exportModel } = await import("./export3d.js");
      const files = await exportModel(model);
      if (token !== exportToken) return;
      const stem = currentDownloads.stem || "Plan";
      setDl("#dlGlb", { url: trackUrl(URL.createObjectURL(files.glb)), name: `${stem}_3D.glb` });
      setDl("#dlObj", { url: trackUrl(URL.createObjectURL(files.obj)), name: `${stem}_3D.obj` });
    } catch (err) {
      console.error(err);
      $("#bimHint").textContent = "Die 3D-Dateien konnten in diesem Browser nicht erzeugt werden. Die IFC-Datei ist verfügbar.";
    }
  }

  function setDl(id, dl) {
    const a = $(id);
    const enabled = !!dl;
    a.classList.toggle("is-disabled", !enabled);
    a.setAttribute("aria-disabled", String(!enabled));
    if (enabled) { a.href = dl.url; a.download = dl.name; } else { a.removeAttribute("href"); }
  }

  function showBim(b) {
    bim = b;
    disposeViewer();
    const ok = !!b.available;
    setDl("#dlIfc", ok ? currentDownloads.ifc : null);
    setDl("#dlGlb", null);
    setDl("#dlObj", null);
    if (ok) prepare3dFiles(b.model3d);
    $("#bimHint").textContent = ok ? "" : b.reason;
    $("#no3d").hidden = ok;
    $("#no3d").textContent = ok ? "" : b.reason;
    const panel = $("#bimPanel");
    panel.hidden = !ok;
    if (!ok) return;
    const s = b.summary;
    const parts = [`Geschoss ${s.storey}`, `${s.walls} Wände (${s.walls_external} aussen)`, `${s.doors} Türen`,
      `${s.windows} Fenster`, `${s.spaces.length} Räume`];
    if (s.stairs) parts.push(`${s.stairs} Treppe${s.stairs > 1 ? "n" : ""}`);
    $("#bimLine").textContent = parts.join(" · ");
    const fmt = (a) => a.toFixed(1).replace(".", ".");
    $("#bimSpaces").innerHTML = s.spaces.length
      ? s.spaces.map((sp) => `<li><span>${escapeHtml(sp.name)}</span><span class="num">${fmt(sp.area)} m²</span></li>`).join("")
      : "<li>Keine geschlossenen Räume erkannt</li>";
    $("#bimAssumptions").innerHTML = s.assumptions.map((a) => `<li>${escapeHtml(a)}</li>`).join("");
    $("#bimNot").innerHTML = [...s.not_created, ...s.warnings].map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  }

  async function ensureViewer() {
    if (viewer3d || !bim || !bim.available) return;
    try {
      const mod = await import("./viewer3d.js");
      viewer3d = mod.mount($("#view3d"), bim.model3d);
    } catch (err) {
      $("#no3d").hidden = false;
      $("#no3d").textContent = "Die 3D-Ansicht kann in diesem Browser nicht angezeigt werden (WebGL). Die IFC- und 3D-Dateien lassen sich trotzdem herunterladen.";
    }
  }
  function disposeViewer() {
    if (viewer3d) { viewer3d.dispose(); viewer3d = null; }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  $$(".seg button").forEach((b) => b.addEventListener("click", () => { setMode(b.dataset.mode); }));
  function setMode(mode) {
    viewer.dataset.mode = mode;
    if (mode === "3d") ensureViewer();
    $$(".seg button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
    requestAnimationFrame(fit);
  }
  $("#colorLayers").addEventListener("change", (e) => viewer.classList.toggle("colored", e.target.checked));

  // Zoom & Verschieben (beide Ansichten synchron)
  function paneSize() {
    const p = $$(".pane", viewer).find((el) => el.offsetParent !== null) || $(".pane", viewer);
    return [p.clientWidth, p.clientHeight];
  }
  function apply() {
    const { s, tx, ty } = state.view;
    $$(".layer", viewer).forEach((el) => { el.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`; });
  }
  function fit() {
    const [pw, ph] = paneSize();
    const [W, H] = state.size;
    const s = Math.min(pw / W, ph / H) * 0.96;
    state.view = { s, tx: (pw - W * s) / 2, ty: (ph - H * s) / 2 };
    apply();
  }
  function zoomAt(factor, mx, my) {
    const v = state.view;
    const [W, H] = state.size;
    const [pw, ph] = paneSize();
    const minS = Math.min(pw / W, ph / H) * 0.5;
    const ns = Math.max(minS, Math.min(v.s * factor, 8));
    const f = ns / v.s;
    v.tx = mx - (mx - v.tx) * f;
    v.ty = my - (my - v.ty) * f;
    v.s = ns;
    apply();
  }
  $$(".pane", viewer).forEach((pane) => {
    pane.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = pane.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    let drag = null;
    pane.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, tx: state.view.tx, ty: state.view.ty };
      pane.setPointerCapture(e.pointerId);
      pane.classList.add("dragging");
    });
    pane.addEventListener("pointermove", (e) => {
      if (!drag) return;
      state.view.tx = drag.tx + (e.clientX - drag.x);
      state.view.ty = drag.ty + (e.clientY - drag.y);
      apply();
    });
    const end = () => { drag = null; pane.classList.remove("dragging"); };
    pane.addEventListener("pointerup", end);
    pane.addEventListener("pointercancel", end);
    pane.addEventListener("dblclick", fit);
  });
  $$(".zoom-tools button").forEach((b) => b.addEventListener("click", () => {
    const [pw, ph] = paneSize();
    if (b.dataset.zoom === "fit") fit();
    else zoomAt(b.dataset.zoom === "in" ? 1.4 : 1 / 1.4, pw / 2, ph / 2);
  }));
  window.addEventListener("resize", () => { if (app.dataset.view === "result") fit(); });

  // ------------------------------------------------------------ Aktionen
  document.addEventListener("click", (e) => {
    const a = e.target.closest("[data-action]");
    if (!a) return;
    if (a.dataset.action === "process") startProcessing();
    if (a.dataset.action === "reset") reset();
  });

  function discardJob() {
    exportToken++;
    state.loaded = false;
    state.urls.forEach((u) => URL.revokeObjectURL(u));
    state.urls = [];
    engine.cleanup();
  }

  function reset() {
    discardJob();
    calib.open = false;
    calibLen.value = "";
    $("#resVector").innerHTML = "";
    $("#resOriginal").removeAttribute("src");
    disposeViewer();
    show("upload");
  }
})();

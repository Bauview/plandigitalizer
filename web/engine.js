/* PlanDigitalizer – Verarbeitung im Browser.
 * Startet den Rechen-Worker (Python/OpenCV via Pyodide) und die Texterkennung (Tesseract.js).
 * Alle Dateien kommen von dieser Website; Pläne werden nirgendwohin übertragen.
 */
/* global Tesseract */
(() => {
  "use strict";
  const BASE = new URL("./", document.baseURI).href;
  const VERSION = document.documentElement.dataset.version || "1";

  let worker = null;
  let ready = null;
  let stages = [];
  let ocrWorker = null;
  let ocrReady = null;
  let pending = null;          // { resolve, reject, onStage }
  let loadingListener = () => {};

  function start(onLoading) {
    if (onLoading) loadingListener = onLoading;
    if (ready) return ready;
    ready = new Promise((resolve, reject) => {
      worker = new Worker(BASE + "worker.js?v=" + VERSION);
      worker.onmessage = (ev) => handle(ev.data, resolve, reject);
      worker.onerror = (e) => {
        const err = new Error("Der Rechenkern konnte nicht gestartet werden. " + (e.message || ""));
        reject(err);
        if (pending) { pending.reject(err); pending = null; }
      };
      worker.postMessage({ type: "init", version: VERSION });
    });
    ready.then(() => startOcr()).catch(() => {});
    return ready;
  }

  function handle(m, resolveInit, rejectInit) {
    if (m.type === "loading") return loadingListener(m.text, m.pct);
    if (m.type === "ready") { stages = m.stages; return resolveInit(); }
    if (!pending) {
      if (m.type === "error") rejectInit(Object.assign(new Error(m.detail || m.message), m));
      return;
    }
    const p = pending;
    if (m.type === "stage") return p.onStage && p.onStage(m.stage);
    if (m.type === "error") { pending = null; return p.reject(Object.assign(new Error(m.detail || m.message), m)); }
    if (m.type === "loaded" || m.type === "ocr" || m.type === "done") { pending = null; return p.resolve(m); }
  }

  function call(msg, onStage, transfer) {
    return ready.then(() => new Promise((resolve, reject) => {
      pending = { resolve, reject, onStage };
      worker.postMessage(msg, transfer || []);
    }));
  }

  // ---------------------------------------------------------------- Texterkennung
  function startOcr() {
    if (ocrReady) return ocrReady;
    if (typeof Tesseract === "undefined") { ocrReady = Promise.resolve(null); return ocrReady; }
    ocrReady = (async () => {
      try {
        const w = await Tesseract.createWorker(["deu", "eng"], 1, {
          workerPath: BASE + "vendor/tesseract/worker.min.js",
          corePath: BASE + "vendor/tesseract/core",
          langPath: BASE + "vendor/tesseract/lang",
          workerBlobURL: false,
          gzip: true,
        });
        await w.setParameters({ tessedit_pageseg_mode: "11", preserve_interword_spaces: "1" });
        ocrWorker = w;
        return w;
      } catch (err) {
        console.warn("Texterkennung nicht verfügbar:", err);
        return null;
      }
    })();
    return ocrReady;
  }

  async function runOcr(images, onStage) {
    const w = await startOcr();
    if (!w) return { tsvs: [], ok: false };
    const tsvs = [];
    for (const bytes of images) {
      const blob = new Blob([bytes], { type: "image/png" });
      const { data } = await w.recognize(blob, {}, { tsv: true });
      tsvs.push(data.tsv || "");
    }
    return { tsvs, ok: true };
  }

  // ---------------------------------------------------------------- öffentliche Funktionen
  async function load(file) {
    await start();
    const bytes = await file.arrayBuffer();
    const r = await call({ type: "load", name: file.name, bytes }, null, [bytes]);
    return { width: r.width, height: r.height, pages: r.pages,
      previewUrl: URL.createObjectURL(new Blob([r.preview], { type: "image/jpeg" })) };
  }

  async function process({ calibration, settings3d, onStage }) {
    const stage = onStage || (() => {});
    const prep = await call({ type: "prepare" }, stage);
    stage(1);
    const ocr = await runOcr(prep.images, stage);
    const done = await call({ type: "finish", tsvs: ocr.tsvs, ocrOk: ocr.ok, calibration, settings3d }, stage);
    const r = done.result;
    if (!ocr.ok) r.warnings = [...(r.warnings || []), "Texterkennung konnte nicht geladen werden – Texte wurden als Linien übernommen."];
    const types = { pdf: "application/pdf", dxf: "application/dxf", ifc: "application/x-step" };
    r.downloads = {};
    for (const [k, bytes] of Object.entries(done.files)) {
      const ext = k;
      const name = k === "ifc" ? `${r.stem}.ifc` : `${r.stem}_vektor.${ext}`;
      r.downloads[k] = { url: URL.createObjectURL(new Blob([bytes], { type: types[k] || "application/octet-stream" })), name };
    }
    r.processed = URL.createObjectURL(new Blob([done.processed], { type: "image/jpeg" }));
    return r;
  }

  function cleanup() {
    if (worker && ready) ready.then(() => worker.postMessage({ type: "cleanup" })).catch(() => {});
  }

  window.PlanEngine = { start, load, process, cleanup, get stages() { return stages; } };
})();

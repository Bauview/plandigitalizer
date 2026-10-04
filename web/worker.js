/* PlanDigitalizer – Rechen-Worker (läuft im Hintergrund des Browsers).
 * Lädt Python (Pyodide) und die Bildverarbeitung von DIESER Website – keine fremden Server.
 * Pläne werden nur im Arbeitsspeicher des Browsers verarbeitet.
 */
/* global importScripts, loadPyodide */
"use strict";

const BASE = new URL("./", self.location.href).href;
let py = null;
let api = null;

function post(msg, transfer) { self.postMessage(msg, transfer || []); }
function progress(text, pct) { post({ type: "loading", text, pct }); }

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function init() {
  progress("Rechenkern wird geladen …", 5);
  importScripts(BASE + "pyodide/pyodide.js");
  py = await loadPyodide({ indexURL: BASE + "pyodide/", fullStdLib: false });
  progress("Bildverarbeitung wird geladen …", 30);
  await py.loadPackage(["numpy", "opencv-python", "pillow", "pymupdf", "fonttools", "pyparsing", "typing-extensions"],
    { messageCallback: () => {}, errorCallback: (e) => console.warn(e) });
  progress("CAD-Export wird geladen …", 80);
  const ezdxf = await fetchBytes(BASE + "pyodide/ezdxf-1.4.4-py3-none-any.whl");
  py.unpackArchive(ezdxf, "wheel", { extractDir: "/lib/python3.13/site-packages" });
  progress("PlanDigitalizer wird geladen …", 92);
  const app = await fetchBytes(BASE + "py/app.zip?v=" + (self.APP_VERSION || ""));
  py.unpackArchive(app, "zip", { extractDir: "/home/pyodide/pd" });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/pd')");
  api = py.pyimport("app.web_api");
  const stages = JSON.parse(api.stages());
  progress("Bereit", 100);
  post({ type: "ready", stages });
}

function readFile(path) { return py.FS.readFile(path); }

function writeInput(name, bytes) {
  const dir = "/tmp/pd_in";
  try { py.FS.mkdirTree(dir); } catch (_) { /* existiert */ }
  const ext = (name.match(/\.[A-Za-z0-9]+$/) || [".bin"])[0].toLowerCase();
  const path = `${dir}/upload${ext}`;
  py.FS.writeFile(path, bytes);
  return path;
}

let inputPath = null;
let jobName = "Plan";

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === "init") {
      self.APP_VERSION = m.version || "";
      await init();
    } else if (m.type === "load") {
      api.cleanup();
      jobName = m.name;
      inputPath = writeInput(m.name, new Uint8Array(m.bytes));
      const r = JSON.parse(api.load(inputPath));
      if (!r.ok) return post({ type: "error", ...r });
      const preview = readFile(r.preview);
      post({ type: "loaded", width: r.width, height: r.height, pages: r.pages, preview }, [preview.buffer]);
    } else if (m.type === "prepare") {
      post({ type: "stage", stage: 0 });
      const r = JSON.parse(api.prepare(inputPath));
      if (!r.ok) return post({ type: "error", ...r });
      const images = r.ocr_images.map(readFile);
      post({ type: "ocr", images }, images.map((i) => i.buffer));
    } else if (m.type === "finish") {
      const cb = (i) => post({ type: "stage", stage: i });
      const r = JSON.parse(api.finish(JSON.stringify(m.tsvs || []), !!m.ocrOk,
        m.calibration ? JSON.stringify(m.calibration) : "", JSON.stringify(m.settings3d || {}), jobName, cb));
      if (!r.ok) return post({ type: "error", ...r });
      const files = {};
      for (const [k, p] of Object.entries(r.files)) files[k] = readFile(p);
      const processed = readFile(r.processed);
      delete r.files; delete r.processed;
      post({ type: "done", result: r, files, processed },
        [...Object.values(files).map((f) => f.buffer), processed.buffer]);
    } else if (m.type === "cleanup") {
      if (api) api.cleanup();
      inputPath = null;
    }
  } catch (err) {
    console.error(err);
    post({ type: "error", message: "Die Datei konnte nicht verarbeitet werden.",
      detail: "Technischer Fehler: " + (err && err.message ? err.message.split("\n").slice(-3).join(" ") : String(err)) });
  }
};

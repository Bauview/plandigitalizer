"""Schnittstelle für die Browser-Version (Pyodide-Worker).

Ablauf (alle Dateien liegen im Speicher-Dateisystem des Browsers, nichts verlässt das Gerät):

    load(pfad)                     -> Vorschau + Eckdaten
    prepare(pfad)                  -> Bildkorrektur; schreibt Bilder für die Texterkennung
    finish(tsvs, ...)              -> Vektorisierung, Exporte, IFC, 3D-Daten

Rückgaben sind JSON-Texte; Dateien werden nach ``OUT`` geschrieben und vom Browser gelesen.
"""
from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path

import cv2

from .bim import ModelError, Settings3D, build_bim
from .export import render_svg, write_dxf, write_pdf
from .pipeline import STAGES, PlanError, stage_finish, stage_prepare
from .pipeline.loader import load_plan, make_preview
from .pipeline.ocr import ocr_images, texts_from_tsv

OUT = Path("/tmp/pd_out")
GENERIC_ERROR = "Die Datei konnte nicht verarbeitet werden."
_state: dict = {}


def _err(exc: Exception) -> str:
    if isinstance(exc, PlanError):
        detail = f"{exc.message} {exc.detail}".strip()
    elif isinstance(exc, MemoryError):
        detail = "Nicht genügend Arbeitsspeicher im Browser – bitte eine kleinere Datei oder einen Computer statt Handy verwenden."
    else:
        detail = f"Technischer Fehler: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    return json.dumps({"ok": False, "message": GENERIC_ERROR, "detail": detail})


def stages() -> str:
    return json.dumps(STAGES)


def load(path: str) -> str:
    try:
        plan = load_plan(Path(path))
        jpg, w, h = make_preview(plan)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "preview.jpg").write_bytes(jpg)
        return json.dumps({"ok": True, "width": w, "height": h, "pages": plan.pages, "preview": str(OUT / "preview.jpg")})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def prepare(path: str) -> str:
    """Schritt 1: Laden + Bildkorrektur. Liefert die Bilder für Tesseract.js."""
    try:
        _state.clear()
        OUT.mkdir(parents=True, exist_ok=True)
        st = stage_prepare(Path(path))
        imgs, k = ocr_images(st.prep.gray)
        files = []
        for i, img in enumerate(imgs):
            ok, buf = cv2.imencode(".png", img)
            p = OUT / f"ocr{i}.png"
            p.write_bytes(buf.tobytes())
            files.append(str(p))
        _state.update(st=st, k=k, h_img=int(imgs[0].shape[0]))
        return json.dumps({"ok": True, "ocr_images": files})
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def finish(tsvs_json: str, ocr_ok: bool, calibration_json: str, settings_json: str, name: str,
           progress=None) -> str:
    """Schritt 2: Vektorisierung + alle Ausgaben."""
    report = progress or (lambda i: None)
    try:
        st = _state["st"]
        tsvs = json.loads(tsvs_json or "[]")
        texts = texts_from_tsv(tsvs, st.prep.gray.shape, _state["k"], _state["h_img"]) if ocr_ok else []
        calibration = json.loads(calibration_json) if calibration_json else None
        drawing = stage_finish(st, texts, bool(ocr_ok), calibration, report)

        report(3)
        stem = _safe_stem(name)
        files = {"pdf": OUT / f"{stem}_vektor.pdf", "dxf": OUT / f"{stem}_vektor.dxf"}
        write_pdf(drawing, files["pdf"], stem)
        write_dxf(drawing, files["dxf"])
        ok, buf = cv2.imencode(".jpg", _small(st.prep.gray), [cv2.IMWRITE_JPEG_QUALITY, 80])
        (OUT / "processed.jpg").write_bytes(buf.tobytes())
        result = {
            "svg": render_svg(drawing),
            "warnings": drawing.warnings,
            "scale_note": drawing.scale_note,
            "unit": drawing.unit,
            "layers": drawing.layer_counts(),
            "info": {k: v for k, v in drawing.info.items() if isinstance(v, (int, float, str, bool, list, type(None)))},
            "processed": str(OUT / "processed.jpg"),
            "size": [drawing.width, drawing.height],
        }

        report(4)
        try:
            bim = build_bim(drawing, Settings3D.from_dict(json.loads(settings_json or "{}")), stem)
            files["ifc"] = OUT / f"{stem}.ifc"
            files["ifc"].write_text(bim.ifc_text, encoding="utf-8")
            result["bim"] = {"available": True, "summary": bim.model.summary(), "model3d": bim.model3d}
        except ModelError as exc:
            result["bim"] = {"available": False, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            result["bim"] = {"available": False, "reason": f"IFC/3D konnte nicht erzeugt werden ({type(exc).__name__}). "
                                                           "PDF und DXF sind davon nicht betroffen."}
        result["files"] = {k: str(v) for k, v in files.items()}
        result["stem"] = stem
        result["ok"] = True
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001
        return _err(exc)


def cleanup() -> None:
    """Alle Zwischendateien im Browser-Speicher löschen."""
    _state.clear()
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree("/tmp/pd_in", ignore_errors=True)


def _small(gray):
    h, w = gray.shape
    k = min(1.0, 2000.0 / max(w, h))
    return cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray


def _safe_stem(name: str) -> str:
    import re
    stem = Path(name or "Plan").stem
    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip(" ._") or "Plan"
    return stem[:80]

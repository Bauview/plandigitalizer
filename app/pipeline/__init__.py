"""Verarbeitungs-Pipeline: Datei -> Drawing (Vektorgeometrie)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from . import cleanup
from .classify import classify
from .geometry import Drawing, Hatch, Polyline
from .loader import PlanError, load_plan
from .ocr import filter_by_ink, normalize_heights, ocr_available, remove_text_from_binary, run_ocr
from .preprocess import prepare, transform_point
from .scale import find_scale_ratio, robust_ratio
from .vectorize import vectorize

STAGES = [
    "Datei wird analysiert …",
    "Linien werden erkannt …",
    "Plan wird vektorisiert …",
    "Dateien werden erstellt …",
    "IFC- und 3D-Modell werden erstellt …",
]

SCALE_UNKNOWN = "Massstab konnte nicht zuverlässig erkannt werden. Die Zeichnung muss in CAD skaliert werden."

__all__ = ["run_pipeline", "stage_prepare", "stage_finish", "STAGES", "PlanError"]


@dataclass
class Stage1:
    """Zwischenstand nach Laden + Bildkorrektur (vor der Texterkennung)."""
    plan: object
    prep: object
    warnings: list[str]
    t_start: float


def run_pipeline(path: Path, calibration: dict | None = None,
                 progress: Callable[[int], None] = lambda i: None) -> Drawing:
    """Komplette Verarbeitung mit lokalem Tesseract (Desktop, Tests)."""
    st = stage_prepare(path, progress)
    texts = run_ocr(st.prep.gray)
    return stage_finish(st, texts, ocr_available(), calibration, progress)


def stage_prepare(path: Path, progress: Callable[[int], None] = lambda i: None) -> Stage1:
    t_start = time.time()
    progress(0)
    plan = load_plan(path)
    prep = prepare(plan.image)
    return Stage1(plan, prep, list(prep.warnings), t_start)


def stage_finish(st: Stage1, texts: list, ocr_ok: bool, calibration: dict | None = None,
                 progress: Callable[[int], None] = lambda i: None) -> Drawing:
    plan, prep, t_start = st.plan, st.prep, st.t_start
    h, w = prep.binary.shape
    warnings = list(st.warnings)
    texts = filter_by_ink(texts, prep.binary)
    if not ocr_ok:
        warnings.append("Texterkennung nicht verfügbar – Texte werden als Linien übernommen.")

    # --- Kalibrierung (Benutzer) in bearbeitete Bildkoordinaten übertragen
    calib_mm_px = None
    if calibration:
        try:
            ow, oh = plan.image.shape[1], plan.image.shape[0]
            p1 = transform_point(prep.M, calibration["p1"][0] * ow, calibration["p1"][1] * oh)
            p2 = transform_point(prep.M, calibration["p2"][0] * ow, calibration["p2"][1] * oh)
            d = math.dist(p1, p2)
            length_mm = float(calibration["length_m"]) * 1000.0
            if d > 10 and length_mm > 0:
                calib_mm_px = length_mm / d
        except (KeyError, TypeError, ValueError, IndexError):
            warnings.append("Die Kalibrierung war unvollständig und wurde ignoriert.")

    progress(1)
    binary = remove_text_from_binary(prep.binary, texts)
    normalize_heights(texts, prep.binary)
    vec = vectorize(binary)
    thin = vec.thin_w

    progress(2)
    lines = vec.lines
    cleanup.snap_axis(lines, 3.0)
    lines = cleanup.merge_collinear(lines, dist_tol=max(2.0, 1.4 * thin), gap_tol=max(3.0, 2.5 * thin))
    cleanup.connect_orthogonal(lines, tol=max(3.0, 2.5 * thin))
    cleanup.snap_free_ends(lines, tol=max(3.0, 2.0 * thin))
    lines = cleanup.merge_collinear(lines, dist_tol=1.0, gap_tol=1.0)
    lines = cleanup.drop_bridges(lines, max_len=8 * thin, tol=max(2.0, 1.2 * thin))
    lines = cleanup.drop_short(lines, max(3.0, 2.0 * thin))

    dim_cands = classify(lines, vec.arcs, vec.circles, texts, vec.wall_mask, thin, vec.comp_diag,
                         (h, w), calib_mm_px)

    # --- Massstab festlegen
    drawing = Drawing(w, h)
    paper_mm_px = None
    if plan.dpi and not prep.warped:
        paper_mm_px = 25.4 / (plan.dpi * prep.resize_factor)
    ratio = find_scale_ratio(texts)
    chain = robust_ratio(dim_cands)

    if calib_mm_px:
        drawing.mm_per_px, drawing.unit = calib_mm_px, "mm"
        drawing.scale_note = "Massstab aus Kalibrierung (bekannte Länge)."
    elif chain:
        drawing.mm_per_px, drawing.unit = chain, "mm"
        drawing.scale_note = "Massstab aus Masskette abgeleitet – bitte in CAD an einem Mass prüfen."
    elif ratio and paper_mm_px:
        drawing.mm_per_px, drawing.unit = paper_mm_px * ratio, "mm"
        drawing.scale_note = f"Massstab 1:{ratio} aus Planbeschriftung erkannt."
    else:
        warnings.append(SCALE_UNKNOWN)
        if ratio:
            drawing.scale_note = (f"Massstab 1:{ratio} im Plan gelesen, Papiergrösse aber unbekannt "
                                  "(Foto). Bitte in CAD skalieren oder Kalibrierung verwenden.")
        if paper_mm_px:
            drawing.mm_per_px, drawing.unit = paper_mm_px, "mm"
            drawing.scale_note = drawing.scale_note or "Zeichnung im Papiermass 1:1 (Planmassstab unbekannt)."
        else:
            drawing.scale_note = drawing.scale_note or "Einheit = Bildpixel. Bitte in CAD skalieren."

    # --- Entities sammeln
    for rings in vec.wall_regions:
        for ring in rings:
            drawing.entities.append(Polyline(ring, True, "WALLS"))
        drawing.entities.append(Hatch(rings, "HATCH"))
    drawing.entities.extend(lines)
    drawing.entities.extend(vec.arcs)
    drawing.entities.extend(vec.circles)
    drawing.entities.extend(texts)

    if len(lines) + len(vec.arcs) + len(vec.wall_regions) < 5:
        warnings.append("Es wurden nur sehr wenige Linien erkannt. "
                        "Die Planqualität ist möglicherweise zu gering.")

    drawing.warnings = list(dict.fromkeys(warnings))
    k = min(1.0, 2000.0 / max(w, h))
    small = cv2.resize(prep.gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else prep.gray
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    drawing.preview_jpg = buf.tobytes() if ok else b""
    drawing.info = {
        "source": plan.source,
        "pages": plan.pages,
        "size_px": [w, h],
        "deskew_deg": round(prep.deskew_deg, 2),
        "perspective_corrected": prep.warped,
        "stroke_px": round(thin, 2),
        "paper_mm_px": paper_mm_px,
        "scale_ratio_text": ratio,
        "ocr": ocr_ok,
        "seconds": round(time.time() - t_start, 1),
    }
    if plan.pages > 1:
        drawing.warnings.append(f"Die PDF hat {plan.pages} Seiten – verarbeitet wurde Seite 1.")
    return drawing

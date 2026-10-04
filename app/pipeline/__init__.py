"""Verarbeitungs-Pipeline: Datei -> Drawing (Vektorgeometrie)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2


from . import cleanup, essential, semantic
from .classify import classify
from .geometry import Drawing, Hatch, Polyline
from .loader import PlanError, load_plan
from .ocr import filter_by_ink, normalize_heights, ocr_available, remove_text_from_binary, run_ocr
from .preprocess import prepare, rotate_prepared, transform_point, warp_prepared
from .scale import find_scale_ratio, robust_ratio
from .vectorize import vectorize

STAGES = [
    "Plan wird ausgerichtet, Wände und Öffnungen werden erkannt …",
    "Linien werden erkannt …",
    "Plan wird vektorisiert …",
    "Dateien werden erstellt …",
    "IFC- und 3D-Modell werden erstellt …",
]

_ALPHA = __import__("re").compile(r"[A-Za-zÄÖÜäöüéèàç]{2,}")
_WORD = __import__("re").compile(r"[A-Za-zÄÖÜäöüéèàç]{3,}")
_TITLE = __import__("re").compile(r"grundriss|geschoss|\b(EG|OG|UG|DG|\d\.\s?OG)\b|bestand|wohnung", __import__("re").I)
_SCALE_TXT = __import__("re").compile(r"\b(m|mst\.?|massstab)?\s*1\s*:\s*\d{2,4}\b", __import__("re").I)

SCALE_UNKNOWN = "Massstab konnte nicht zuverlässig erkannt werden. Die Zeichnung muss in CAD skaliert werden."

__all__ = ["run_pipeline", "stage_prepare", "stage_finish", "STAGES", "PlanError"]


@dataclass
class Stage1:
    """Zwischenstand nach Laden + Bildkorrektur (vor der Texterkennung)."""
    plan: object
    prep: object
    warnings: list[str]
    t_start: float
    seg: object = None            # semantic.Segmentation (Netz)


def run_pipeline(path: Path, calibration: dict | None = None,
                 progress: Callable[[int], None] = lambda i: None, essential_only: bool = True,
                 use_model: bool = True) -> Drawing:
    """Komplette Verarbeitung mit lokalem Tesseract (Desktop, Tests)."""
    st = stage_prepare(path, progress, use_model=use_model)
    texts = run_ocr(st.prep.gray)
    return stage_finish(st, texts, ocr_available(), calibration, progress, essential_only=essential_only)


def stage_prepare(path: Path, progress: Callable[[int], None] = lambda i: None, use_model: bool = True) -> Stage1:
    t_start = time.time()
    progress(0)
    plan = load_plan(path)
    use_model = use_model and semantic.available()
    prep = prepare(plan.image, deskew=not use_model)
    warnings = list(prep.warnings)
    seg = None
    if use_model:
        seg = semantic.segment(prep.gray)
        band = semantic.confident_band(seg)
        # Perspektive (schräg fotografiert, Blattrand nicht sichtbar): stürzende Wandlinien gerade richten
        if not prep.warped and seg.wall_px > 0:
            first = None
            for _ in range(2):                          # zweiter Durchgang verfeinert
                pc = semantic.perspective_correction(band, seg.wall_px)
                if pc is None:
                    break
                H, size, conv = pc
                prev = (prep, seg, band)
                prep = warp_prepared(prep, H, size)
                seg = semantic.segment(prep.gray, seg.scale)
                band = semantic.confident_band(seg)
                ang, clar = semantic.dominant_angle(band)
                if clar < 0.3 or abs(ang) > 2.0:        # Entzerrung unplausibel -> verwerfen
                    prep, seg, band = prev
                    break
                first = first or conv
            if first:
                warnings.append(f"Perspektive entzerrt (Wandlinien liefen um {first:.1f}° zusammen).")
        # Ausrichtung aus den erkannten Wänden (unabhängig von Möbeln, Texten, Schraffuren);
        # nach dem Drehen erneut messen – der zweite Durchgang ist auf ~0.1° genau
        base, base_seg, total = prep, seg, 0.0
        for _ in range(3):
            angle, clarity = semantic.dominant_angle(band)
            if clarity <= 0.2 or abs(angle) <= 0.08:
                break
            total += angle                              # immer vom Ausgangsbild aus drehen (keine Unschärfe)
            prep, (labels, probs) = rotate_prepared(base, total, [base_seg.labels, base_seg.probs])
            if abs(angle) > 0.6:
                seg = semantic.segment(prep.gray, base_seg.scale)   # gerade ausgerichtet neu erkennen
            else:
                seg = semantic.Segmentation(labels, probs, base_seg.scale, base_seg.wall_px)
            band = semantic.confident_band(seg)
        if abs(prep.deskew_deg) > 0.5:
            warnings.append(f"Plan war um {prep.deskew_deg:.1f}° verdreht und wurde gerade ausgerichtet.")
    return Stage1(plan, prep, warnings, t_start, seg)


def stage_finish(st: Stage1, texts: list, ocr_ok: bool, calibration: dict | None = None,
                 progress: Callable[[int], None] = lambda i: None, essential_only: bool = True) -> Drawing:
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
    sem = None
    if st.seg is not None:
        sem = essential.reconstruct(st.seg.labels, prep.binary)
    if sem is not None:
        # Wände/Öffnungen stammen aus der Erkennung; der Rest des Plans wird separat vektorisiert
        grow = int(max(2, round(0.25 * sem.wall_px)))
        cut = cv2.dilate(sem.band, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
        residual = cv2.bitwise_and(prep.binary, cv2.bitwise_not(cut))
    else:
        residual = prep.binary
    binary = remove_text_from_binary(residual, texts)
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

    wall_mask = None if sem is not None else vec.wall_mask
    dim_cands = classify(lines, vec.arcs, vec.circles, texts, wall_mask, thin, vec.comp_diag,
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

    stamps = None
    if sem is not None and texts:
        stamps = essential.room_stamps(sem, texts, drawing.mm_per_px, prep.binary)
        if not drawing.mm_per_px and stamps[3]:
            drawing.mm_per_px, drawing.unit = stamps[3], "mm"
            drawing.scale_note = ("Massstab aus den Flächenangaben der Raumstempel abgeleitet – bitte in CAD "
                                  "an einem bekannten Mass prüfen.")
            warnings = [x for x in warnings if x != SCALE_UNKNOWN]
            stamps = essential.room_stamps(sem, texts, drawing.mm_per_px, prep.binary)
    if not drawing.mm_per_px and sem is not None:
        est = essential.fallback_scale_from_doors(sem)
        if est:
            drawing.mm_per_px, drawing.unit = est, "mm"
            drawing.scale_note = ("Massstab GESCHÄTZT aus den Türbreiten (Annahme 0.90 m) – unbedingt in CAD "
                                  "an einem bekannten Mass prüfen oder Kalibrierung verwenden.")
            warnings = [x for x in warnings if x != SCALE_UNKNOWN]
            warnings.append("Massstab nur geschätzt (aus Türbreiten).")

    # --- Entities sammeln
    if sem is not None:
        drawing.semantic = sem
        drawing.entities.extend(sem.entities)
        drawing.entities.extend(essential.opening_entities(sem, prep.binary))
        if not sem.openings:
            warnings.append("Keine Türen oder Fenster erkannt.")
        foot = essential.footprint_mask(sem)
        inside = lambda p: 0 <= int(p[1]) < h and 0 <= int(p[0]) < w and foot[int(p[1]), int(p[0])] > 0  # noqa: E731
        for l in lines:
            if l.layer == "STAIRS" and inside(l.mid):
                drawing.entities.append(l)
            elif l.layer == "DIMENSIONS":
                drawing.entities.append(l)                  # Massketten gehören zum Bestandesplan
            elif not essential_only and l.layer not in ("STAIRS",):
                if l.layer in ("WALLS", "WINDOWS", "DOORS"):
                    l.layer = "LINES"
                drawing.entities.append(l)
        if not essential_only:
            drawing.entities.extend(a for a in vec.arcs if a.layer != "DOORS")
            drawing.entities.extend(vec.circles)
            for a in vec.arcs:
                if a.layer == "DOORS":
                    a.layer = "SYMBOLS"
        used = set()
        if stamps:
            drawing.entities.extend(stamps[0])
            used = {id(q) for q in stamps[1]}
            warnings.extend(stamps[2])
        plan_title = None
        for t in texts:
            if id(t) in used:
                continue
            if essential_only and _TITLE.search(t.text) and not inside(t.center):
                # Plantitel (z.B. "Grundriss EG 1:100") -> Plankopf, ohne alte Massstabsangabe
                if plan_title is None:
                    # Wörter derselben Zeile (z.B. "GRUNDRISS ERDGESCHOSS BESTAND") zusammennehmen
                    row = [q for q in texts if q.rotation == 0 and not inside(q.center) and id(q) not in used
                           and abs(q.center[1] - t.center[1]) < 0.6 * max(1.0, t.height)
                           and _ALPHA.search(q.text) and q.conf >= 55]
                    row.sort(key=lambda q: q.box[0])
                    plan_title = _SCALE_TXT.sub("", " ".join(q.text for q in row) or t.text).strip(" ,-–")
                    used |= {id(q) for q in row}
                continue
            if essential_only and _SCALE_TXT.search(t.text):
                continue
            if essential_only and t.layer == "DIMENSIONS":
                drawing.entities.append(t)                  # Masszahl
                continue
            if essential_only and not (_WORD.search(t.text) and t.conf >= 55):
                if not ("m2" in t.text or "m²" in t.text):
                    continue                            # Fragmente (z.B. aus Möbelsymbolen) weglassen
            if essential_only:
                if not _ALPHA.search(t.text):
                    if not ("m2" in t.text or "m²" in t.text) or not inside(t.center):
                        continue
            drawing.entities.append(t)
        n_d = sum(1 for o in sem.openings if o.kind == "door")
        n_w = len(sem.openings) - n_d
        recog = {"walls": len(sem.rects), "doors": n_d, "windows": n_w, "wall_px": round(sem.wall_px, 1)}
    else:
        recog = None
        plan_title = None
        for rings in vec.wall_regions:
            for ring in rings:
                drawing.entities.append(Polyline(ring, True, "WALLS"))
            drawing.entities.append(Hatch(rings, "HATCH"))
        drawing.entities.extend(lines)
        drawing.entities.extend(vec.arcs)
        drawing.entities.extend(vec.circles)
        drawing.entities.extend(texts)

    if sem is None and len(lines) + len(vec.arcs) + len(vec.wall_regions) < 5:
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
        "recognition": recog,
        "plan_title": plan_title,
        "model": st.seg is not None,
        "seconds": round(time.time() - t_start, 1),
    }
    if plan.pages > 1:
        drawing.warnings.append(f"Die PDF hat {plan.pages} Seiten – verarbeitet wurde Seite 1.")
    return drawing

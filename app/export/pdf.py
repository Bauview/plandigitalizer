"""Vektor-PDF mit PyMuPDF. Jeder CAD-Layer wird als PDF-Ebene (OCG) angelegt,
d.h. Ebenen lassen sich z.B. in Acrobat ein- und ausblenden."""
from __future__ import annotations

import math
from pathlib import Path

import pymupdf as fitz

from ..pipeline.geometry import LAYERS, Arc, Circle, Drawing, Hatch, Line, Polyline, Text

A3 = (1190.55, 841.89)
MARGIN = 34.0  # 12 mm

# Strichfarbe (Grauwert), Strichstärke in pt
STYLE = {
    "WALLS": ((0, 0, 0), 0.7),
    "DOORS": ((0, 0, 0), 0.35),
    "WINDOWS": ((0, 0, 0), 0.35),
    "STAIRS": ((0, 0, 0), 0.3),
    "LINES": ((0, 0, 0), 0.3),
    "DIMENSIONS": ((0.25, 0.25, 0.25), 0.2),
    "SYMBOLS": ((0.2, 0.2, 0.2), 0.25),
    "HATCH": ((0.45, 0.45, 0.45), 0.15),
    "TEXT": ((0, 0, 0), 0.0),
}
HATCH_FILL = (0.55, 0.55, 0.55)


def write_pdf(d: Drawing, path: Path, title: str = "Plan") -> None:
    W, H = d.width, d.height
    paper = d.info.get("paper_mm_px")
    if paper:
        k = paper * 72.0 / 25.4          # Originalpapiergrösse beibehalten
        pw, ph = W * k, H * k
        ox, oy = 0.0, 0.0
        footer = False
    else:
        pw, ph = A3 if W >= H else (A3[1], A3[0])
        k = min((pw - 2 * MARGIN) / W, (ph - 2 * MARGIN - 10) / H)
        ox = (pw - W * k) / 2
        oy = (ph - 10 - H * k) / 2
        footer = True

    def P(p):
        return fitz.Point(ox + p[0] * k, oy + p[1] * k)

    doc = fitz.open()
    page = doc.new_page(width=pw, height=ph)
    ocg = {name: doc.add_ocg(f"{name} – {desc}", on=True) for name, (_, _, desc) in LAYERS.items()}

    by_layer: dict[str, list] = {}
    for e in d.entities:
        by_layer.setdefault(e.layer, []).append(e)

    shape = page.new_shape()
    # 1) Füllungen
    for e in by_layer.get("HATCH", []):
        if isinstance(e, Hatch):
            for ring in e.rings:
                pts = [P(p) for p in ring] + [P(ring[0])]
                shape.draw_polyline(pts)
            shape.finish(color=None, fill=HATCH_FILL, even_odd=True, width=0, closePath=True, oc=ocg["HATCH"])

    # 2) Linien je Layer
    order = ["HATCH", "LINES", "SYMBOLS", "STAIRS", "DIMENSIONS", "WINDOWS", "DOORS", "WALLS"]
    for layer in order:
        color, width = STYLE[layer]
        ents = by_layer.get(layer, [])
        open_drawn = False
        for e in ents:
            if isinstance(e, Line):
                shape.draw_line(P(e.p1), P(e.p2))
                open_drawn = True
            elif isinstance(e, Arc):
                shape.draw_polyline(_arc_points(e, P))
                open_drawn = True
        if open_drawn:
            shape.finish(color=color, width=width, lineCap=1, lineJoin=1, closePath=False, oc=ocg[layer])
        for e in ents:
            if isinstance(e, Polyline):
                shape.draw_polyline([P(p) for p in e.points])
                shape.finish(color=color, width=width, lineJoin=1, closePath=e.closed, oc=ocg[layer])
            elif isinstance(e, Circle):
                shape.draw_circle(P(e.center), e.radius * k)
                shape.finish(color=color, width=width, closePath=False, oc=ocg[layer])
    shape.commit()

    # 3) Texte
    for e in d.entities:
        if isinstance(e, Text):
            fs = max(2.5, e.height * k / 0.72)
            try:
                page.insert_text(P(e.insert), e.text, fontsize=fs, fontname="helv",
                                 color=(0, 0, 0), rotate=e.rotation, oc=ocg[e.layer])
            except Exception:  # noqa: BLE001  (z.B. nicht darstellbare Zeichen)
                continue

    if footer:
        note = f"PlanDigitalizer  ·  {title}  ·  {d.scale_note or ''}"
        page.insert_text(fitz.Point(MARGIN, ph - 16), note, fontsize=6, fontname="helv", color=(0.5, 0.5, 0.5))

    doc.set_metadata({"title": f"{title} (vektorisiert)", "creator": "PlanDigitalizer",
                      "producer": "PlanDigitalizer / PyMuPDF"})
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def _arc_points(a: Arc, P):
    n = max(8, int(abs(a.sweep) / 3))
    return [P(a.point_at(a.start + (a.end - a.start) * i / n)) for i in range(n + 1)]

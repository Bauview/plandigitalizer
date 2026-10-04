"""Vektor-PDF mit PyMuPDF – als Bestandesplan gestaltet.

* Ist der Massstab bekannt, wird massstäblich auf A3 gesetzt (1:50, 1:100, 1:200 …, der grösste,
  der passt), mit Massstabsleiste. Sonst „unmassstäblich“ auf die Seite eingepasst.
* Wände grau angelegt mit kräftigem Umriss, Fenster/Türen fein, Raumstempel mit fettem Namen.
* Plankopf mit Planbezeichnung, Massstab, Datum und Herkunft.
* Jeder CAD-Layer ist eine PDF-Ebene (in Acrobat ein-/ausblendbar).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pymupdf as fitz

from ..pipeline.geometry import LAYERS, Arc, Circle, Drawing, Hatch, Line, Polyline, Text

A3 = (1190.55, 841.89)            # pt, quer
MM = 72.0 / 25.4
MARGIN = 15 * MM
HEAD_H = 24 * MM                  # Plankopf
SCALES = [20, 50, 100, 200, 500, 1000]

# Strichfarbe, Strichstärke in pt
STYLE = {
    "WALLS": ((0, 0, 0), 0.9),
    "DOORS": ((0, 0, 0), 0.35),
    "WINDOWS": ((0, 0, 0), 0.35),
    "STAIRS": ((0, 0, 0), 0.3),
    "LINES": ((0, 0, 0), 0.25),
    "DIMENSIONS": ((0.25, 0.25, 0.25), 0.2),
    "SYMBOLS": ((0.2, 0.2, 0.2), 0.25),
    "HATCH": ((0.45, 0.45, 0.45), 0.15),
    "TEXT": ((0, 0, 0), 0.0),
    "ROOMS": ((0, 0, 0), 0.0),
}
WALL_FILL = (0.62, 0.62, 0.62)


def _bbox(d: Drawing):
    xs, ys = [], []
    for e in d.entities:
        if isinstance(e, Line):
            pts = [e.p1, e.p2]
        elif isinstance(e, Polyline):
            pts = e.points
        elif isinstance(e, (Arc, Circle)):
            pts = [(e.center[0] - e.radius, e.center[1] - e.radius), (e.center[0] + e.radius, e.center[1] + e.radius)]
        elif isinstance(e, Hatch):
            pts = [p for r in e.rings for p in r]
        elif isinstance(e, Text):
            pts = [(e.box[0], e.box[1]), (e.box[2], e.box[3])]
        else:
            continue
        for x, y in pts:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0.0, 0.0, float(d.width), float(d.height)
    return min(xs), min(ys), max(xs), max(ys)


def _layout(d: Drawing, bw: float, bh: float):
    """Seitengrösse, Massstab-Faktor (pt pro px), Massstab 1:n oder None (Inhalt bw × bh px)."""
    pw, ph = A3 if bw >= bh else (A3[1], A3[0])
    avail_w, avail_h = pw - 2 * MARGIN, ph - 2 * MARGIN - HEAD_H - 4 * MM
    if d.mm_per_px and d.unit == "mm":
        for n in SCALES:
            k = d.mm_per_px / n * MM          # pt pro px bei 1:n
            if bw * k <= avail_w and bh * k <= avail_h:
                return pw, ph, k, n
    k = min(avail_w / max(bw, 1), avail_h / max(bh, 1))
    return pw, ph, k, None


def write_pdf(d: Drawing, path: Path, title: str = "Plan") -> None:
    bx0, by0, bx1, by1 = _bbox(d)
    pad = 0.03 * max(bx1 - bx0, by1 - by0)
    bx0, by0, bx1, by1 = bx0 - pad, by0 - pad, bx1 + pad, by1 + pad
    pw, ph, k, scale_n = _layout(d, bx1 - bx0, by1 - by0)
    ox = (pw - (bx1 - bx0) * k) / 2 - bx0 * k
    oy = MARGIN + (ph - 2 * MARGIN - HEAD_H - 4 * MM - (by1 - by0) * k) / 2 - by0 * k

    def P(p):
        return fitz.Point(ox + p[0] * k, oy + p[1] * k)

    doc = fitz.open()
    page = doc.new_page(width=pw, height=ph)
    ocg = {name: doc.add_ocg(f"{name} – {desc}", on=True) for name, (_, _, desc) in LAYERS.items()}

    by_layer: dict[str, list] = {}
    for e in d.entities:
        by_layer.setdefault(e.layer if e.layer in STYLE else "LINES", []).append(e)

    shape = page.new_shape()
    # 1) Wandfüllung (Poché)
    for e in by_layer.get("HATCH", []):
        if isinstance(e, Hatch):
            for ring in e.rings:
                shape.draw_polyline([P(p) for p in ring] + [P(ring[0])])
            shape.finish(color=None, fill=WALL_FILL, even_odd=True, width=0, closePath=True, oc=ocg["HATCH"])

    # 2) Linien je Layer
    order = ["LINES", "SYMBOLS", "STAIRS", "DIMENSIONS", "WINDOWS", "DOORS", "WALLS"]
    for layer in order:
        color, width = STYLE[layer]
        ents = by_layer.get(layer, [])
        drawn = False
        for e in ents:
            if isinstance(e, Line):
                shape.draw_line(P(e.p1), P(e.p2))
                drawn = True
            elif isinstance(e, Arc):
                shape.draw_polyline(_arc_points(e, P))
                drawn = True
        if drawn:
            w_ = width * (0.6 if layer == "DOORS" else 1.0)     # Anschlagbogen feiner
            shape.finish(color=color, width=w_, lineCap=1, lineJoin=1, closePath=False, oc=ocg[layer])
        for e in ents:
            if isinstance(e, Polyline):
                shape.draw_polyline([P(p) for p in e.points])
                shape.finish(color=color, width=width, lineJoin=0 if layer == "WALLS" else 1,
                             closePath=e.closed, oc=ocg[layer])
            elif isinstance(e, Circle):
                shape.draw_circle(P(e.center), e.radius * k)
                shape.finish(color=color, width=width, closePath=False, oc=ocg[layer])
    shape.commit()

    # 3) Texte und Raumstempel (Stempelzeilen mit festem Zeilenabstand um die Stempelmitte)
    groups: dict[int, list] = {}
    for e in d.entities:
        if isinstance(e, Text) and e.layer == "ROOMS":
            groups.setdefault(e.group, []).append(e)
    stamp_y: dict[int, float] = {}
    for g, items in groups.items():
        cy = sum((q.box[1] + q.box[3]) / 2 for q in items) / len(items)
        sizes = [7.5 if "m²" in q.text else 9.0 for q in items]
        total = sum(sz * 1.35 for sz in sizes)
        y = oy + cy * k - total / 2
        for q, sz in zip(items, sizes):
            y += sz * 1.35
            stamp_y[id(q)] = y - 0.35 * sz
    for e in d.entities:
        if not isinstance(e, Text):
            continue
        is_area = "m²" in e.text
        if e.layer == "ROOMS":                      # Raumstempel: feste Papiergrösse
            fs = 7.5 if is_area else 9.0
        else:
            fs = min(10.0, max(3.0, e.height * k / 0.72))
        font = "hebo" if (e.layer == "ROOMS" and not is_area) else "helv"
        try:
            if e.layer == "ROOMS":
                cx = (e.box[0] + e.box[2]) / 2
                tw = fitz.get_text_length(e.text, fontname=font, fontsize=fs)
                yb = stamp_y.get(id(e), oy + (e.box[1] + e.box[3]) / 2 * k + fs * 0.35)
                page.insert_text(fitz.Point(ox + cx * k - tw / 2, yb), e.text, fontsize=fs,
                                 fontname=font, color=(0, 0, 0), oc=ocg["ROOMS"])
            else:
                page.insert_text(P(e.insert), e.text, fontsize=fs, fontname=font, color=(0, 0, 0),
                                 rotate=e.rotation, oc=ocg.get(e.layer, ocg["TEXT"]))
        except Exception:  # noqa: BLE001  (z.B. nicht darstellbare Zeichen)
            continue

    _title_block(page, pw, ph, title, d, scale_n)
    doc.set_metadata({"title": f"{title} (Bestandesplan, vektorisiert)", "creator": "PlanDigitalizer",
                      "producer": "PlanDigitalizer / PyMuPDF"})
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def _title_block(page, pw, ph, title, d: Drawing, scale_n):
    sh = page.new_shape()
    # Planrahmen
    sh.draw_rect(fitz.Rect(MARGIN / 2, MARGIN / 2, pw - MARGIN / 2, ph - MARGIN / 2))
    sh.finish(color=(0, 0, 0), width=0.6)
    # Plankopf rechts unten
    bw, bh = 120 * MM, HEAD_H
    x1, y1 = pw - MARGIN / 2, ph - MARGIN / 2
    x0, y0 = x1 - bw, y1 - bh
    sh.draw_rect(fitz.Rect(x0, y0, x1, y1))
    sh.draw_line(fitz.Point(x0, y0 + bh * 0.45), fitz.Point(x1, y0 + bh * 0.45))
    sh.draw_line(fitz.Point(x0 + bw * 0.5, y0 + bh * 0.45), fitz.Point(x0 + bw * 0.5, y1))
    sh.finish(color=(0, 0, 0), width=0.5)
    # Massstabsleiste (5 m) bei massstäblicher Ausgabe
    if scale_n:
        seg_m = 1.0
        seg_pt = seg_m * 1000 / scale_n * MM
        bx, by = MARGIN, ph - MARGIN / 2 - 7 * MM
        for i in range(5):
            r = fitz.Rect(bx + i * seg_pt, by, bx + (i + 1) * seg_pt, by + 1.6 * MM)
            sh.draw_rect(r)
            sh.finish(color=(0, 0, 0), fill=(0, 0, 0) if i % 2 == 0 else (1, 1, 1), width=0.3)
        for i in (0, 5):
            page.insert_text(fitz.Point(bx + i * seg_pt - 2, by - 1.2 * MM), f"{i}", fontsize=6, fontname="helv")
        page.insert_text(fitz.Point(bx + 5 * seg_pt + 2 * MM, by + 1.6 * MM), "m", fontsize=6, fontname="helv")
    sh.commit()
    page.insert_text(fitz.Point(x0 + 3 * MM, y0 + 6.5 * MM), "BESTANDESPLAN", fontsize=11, fontname="hebo")
    sub = d.info.get("plan_title") or title
    page.insert_text(fitz.Point(x0 + 3 * MM, y0 + 10.5 * MM), sub[:60], fontsize=7.5, fontname="helv")
    scale_txt = f"Massstab 1:{scale_n} (A3)" if scale_n else "unmassstäblich"
    page.insert_text(fitz.Point(x0 + 3 * MM, y0 + bh * 0.45 + 5 * MM), scale_txt, fontsize=7.5, fontname="helv")
    page.insert_text(fitz.Point(x0 + 3 * MM, y0 + bh * 0.45 + 9.5 * MM), f"Datum {_dt.date.today():%d.%m.%Y}",
                     fontsize=7, fontname="helv")
    page.insert_text(fitz.Point(x0 + bw * 0.5 + 3 * MM, y0 + bh * 0.45 + 5 * MM), "Digitalisiert mit PlanDigitalizer",
                     fontsize=6.5, fontname="helv", color=(0.35, 0.35, 0.35))
    note = (d.scale_note or "")[:70]
    page.insert_text(fitz.Point(x0 + bw * 0.5 + 3 * MM, y0 + bh * 0.45 + 9.5 * MM), note, fontsize=5,
                     fontname="helv", color=(0.35, 0.35, 0.35))


def _arc_points(a: Arc, P):
    n = max(8, int(abs(a.sweep) / 3))
    return [P(a.point_at(a.start + (a.end - a.start) * i / n)) for i in range(n + 1)]

"""2D-Vektorplan  ->  vereinfachtes Gebäudemodell.

Vorgehen (alles im Pixelraum des bereinigten Plans, am Ende in Meter umgerechnet):

1. Wandmaske aufbauen: gefüllte Wände + Fläche zwischen erkannten Wand-Doppellinien
   + dicke Einzellinien.
2. Fenster (Gruppen paralleler Fensterlinien) und Türen (Türbogen + angrenzende Wand)
   als Öffnungen bestimmen und in die Maske einfügen, damit die Wand durchläuft.
3. Wandmaske in achsparallele Rechtecke zerlegen -> Wände mit Achse und Stärke.
4. Gebäudeumriss und Räume aus den Freiflächen, Raumnamen aus der Texterkennung.
5. Treppen aus erkannten Treppenlinien, Decke aus dem Gebäudeumriss.

Es werden nur Elemente erzeugt, die im Plan erkannt wurden. Höhen stammen aus den
Einstellungen und sind als Annahme gekennzeichnet. Dächer werden nicht erzeugt.
"""
from __future__ import annotations

import math
import re

import cv2
import numpy as np

from ..pipeline.geometry import Arc, Drawing, Hatch, Line, Text
from ..pipeline.vectorize import _orthogonalize
from ..pipeline.walls import decompose, footprint as _footprint, is_external as _is_external, merge_rects, sample as _sample
from .model import (SOURCE_ASSUMED, SOURCE_MEASURED, BuildingModel, Opening, Settings3D, Slab, Space,
                    Stair, Storey, Wall)


class ModelError(Exception):
    """Modell kann nicht erzeugt werden (verständliche Meldung)."""


_STOREY_RE = re.compile(
    r"\b(UG|KG|EG|DG|[1-9]\s?\.?\s?OG|OG|Erdgeschoss|Obergeschoss|Untergeschoss|Dachgeschoss|Kellergeschoss)\b", re.I)
_STOREY_LONG = {"erdgeschoss": "EG", "obergeschoss": "OG", "untergeschoss": "UG",
                "dachgeschoss": "DG", "kellergeschoss": "UG", "kg": "UG"}
_DIM_RE = re.compile(r"^\s*[\d.,\s]+(m2|m²|m)?\s*$", re.I)


def derive_model(d: Drawing, s: Settings3D) -> BuildingModel:
    if not d.mm_per_px:
        raise ModelError("Für IFC und 3D wird ein Massstab benötigt. Bitte vor dem Vektorisieren "
                         "unter «Massstab festlegen» eine bekannte Länge angeben.")
    H, W = d.height, d.width
    mmpp = d.mm_per_px
    k = mmpp / 1000.0                                   # Meter pro Pixel

    def m(x: float, y: float) -> tuple[float, float]:
        return (round(float(x) * k, 4), round((H - float(y)) * k, 4))

    px = lambda meters: meters / k  # noqa: E731
    tmin, tmax = px(0.07), px(0.65)

    lines = [e for e in d.entities if isinstance(e, Line)]
    texts = [e for e in d.entities if isinstance(e, Text)]

    sem = getattr(d, "semantic", None)
    freestanding = 0
    door_skipped = 0
    if sem is not None and sem.rects:
        # ------------------------------------------------------------ Erkennung (Netz) übernehmen
        mask = sem.band.copy()
        openings_px = [o.as_dict() for o in sem.openings]
        rects = [r for r in sem.rects if max(r[2] - r[0], r[3] - r[1]) * k >= 0.1]
        model_notes_leftover = bool(np.count_nonzero(sem.leftover))
    else:
        model_notes_leftover = False
        # ------------------------------------------------------------ 1) Wandmaske
        mask = np.zeros((H, W), np.uint8)
        for e in d.entities:
            if isinstance(e, Hatch) and e.layer == "HATCH" and e.rings:
                cv2.fillPoly(mask, [_ring(e.rings[0])], 255)
                for hole in e.rings[1:]:
                    cv2.fillPoly(mask, [_ring(hole)], 0)
        _fill_wall_pairs(mask, [l for l in lines if l.layer == "WALLS" and (l.is_h or l.is_v)], tmin, tmax)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if np.count_nonzero(mask) * k * k < 0.5:
            raise ModelError("Im Plan wurden keine Wände erkannt – ein 3D-Modell kann nicht abgeleitet werden.")

        # ------------------------------------------------------------ 2) Öffnungen
        openings_px: list[dict] = []
        for rect in _window_rects([l for l in lines if l.layer == "WINDOWS"], px(0.45)):
            _extend_to_jambs(rect, mask, px(0.3))
            openings_px.append({"kind": "window", **rect})
        for a in (e for e in d.entities if isinstance(e, Arc) and e.layer == "DOORS"):
            rect = _door_rect(a, mask, tmax)
            if rect:
                openings_px.append({"kind": "door", **rect})
            else:
                door_skipped += 1
        ext = px(0.06)   # Öffnung leicht in die Leibung verlängern, damit die Wand durchgehend wird
        for o in openings_px:
            ex, ey = (ext, 0) if o["orient"] == "h" else (0, ext)
            cv2.rectangle(mask, (int(o["x0"] - ex), int(o["y0"] - ey)),
                          (int(math.ceil(o["x1"] + ex)), int(math.ceil(o["y1"] + ey))), 255, -1)

        # Kleine freistehende Teile (Kamin, Stütze, Schacht) sind keine Wände
        n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_c):
            if max(st_c[i, cv2.CC_STAT_WIDTH], st_c[i, cv2.CC_STAT_HEIGHT]) * k < 1.2 and st_c[i, cv2.CC_STAT_AREA] * k * k < 1.0:
                mask[lab_c == i] = 0
                freestanding += 1

        # kleine Fugen (Wandende knapp vor anschliessender Wand) schliessen
        gk = max(3, int(px(0.14)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gk, gk)))

        # ------------------------------------------------------------ 3) Wände
        rects = merge_rects(decompose(mask, px(0.6), keep_rest=False)[0], px(0.08))
    footprint = _footprint(mask, int(px(1.3)))
    outside = cv2.bitwise_not(footprint)

    model_storeys, plan_storey, detected, storey_notes = _storeys(s, texts)
    model = BuildingModel(model_storeys, plan_storey_detected=detected)
    model.assumptions.extend(storey_notes)
    wall_h = s.wall_height
    if wall_h > plan_storey.height:
        wall_h = plan_storey.height
        model.warnings.append(f"Wandhöhe auf Geschosshöhe {plan_storey.height:.2f} m begrenzt.")

    for i, r in enumerate(rects):
        x0, y0, x1, y1, orient = r
        if orient == "h":
            cy = (y0 + y1) / 2
            p0, p1, t_px = m(x0, cy), m(x1, cy), (y1 - y0)
        else:
            cx = (x0 + x1) / 2
            p0, p1, t_px = m(cx, y0), m(cx, y1), (x1 - x0)
        thickness = t_px * k
        if math.dist(p0, p1) < 0.1:
            continue
        wall = Wall(p0, p1, round(float(thickness), 3), wall_h, _is_external(outside, r))
        if not s.auto_thickness:
            wall.thickness, wall.thickness_src = s.default_thickness, SOURCE_ASSUMED
        wall._px = r  # type: ignore[attr-defined]
        model.walls.append(wall)

    # Öffnungen ihrer Wand zuordnen
    unhosted = 0
    for o in openings_px:
        host, best = None, 0.0
        for w in model.walls:
            x0, y0, x1, y1, orient = w._px  # type: ignore[attr-defined]
            if orient != o["orient"]:
                continue
            ix = min(x1, o["x1"]) - max(x0, o["x0"])
            iy = min(y1, o["y1"]) - max(y0, o["y0"])
            if ix > 0 and iy > 0 and ix * iy > best:
                host, best = w, ix * iy
        if host is None:
            unhosted += 1
            continue
        x0, y0, x1, y1, orient = host._px  # type: ignore[attr-defined]
        a0, a1 = (o["x0"], o["x1"]) if orient == "h" else (o["y0"], o["y1"])
        w0 = x0 if orient == "h" else y0
        offset = max(0.0, (a0 - w0) * k)
        width = min((a1 - a0) * k, host.length - offset)
        if width < 0.3:
            continue
        if o["kind"] == "door":
            op = Opening("door", float(offset), float(width), min(s.door_height, wall_h), 0.0, sill_src=SOURCE_MEASURED)
        else:
            sill = min(s.sill_height, max(0.0, wall_h - 0.3))
            op = Opening("window", float(offset), float(width), min(s.window_height, wall_h - sill), sill)
        host.openings.append(op)

    # Namen
    for i, w in enumerate(sorted(model.walls, key=lambda w: (not w.external, -w.length)), 1):
        w.name = f"{'Aussenwand' if w.external else 'Innenwand'} {i:02d}"
    nd = nwin = 0
    for w in model.walls:
        for o in sorted(w.openings, key=lambda o: o.offset):
            if o.kind == "door":
                nd += 1
                o.name = f"Tür {nd:02d}"
            else:
                nwin += 1
                o.name = f"Fenster {nwin:02d}"

    # ------------------------------------------------------------ 4) Räume
    free = cv2.bitwise_and(footprint, cv2.bitwise_not(cv2.dilate(mask, np.ones((3, 3), np.uint8))))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    names_by_label: dict[int, list[str]] = {}
    for t in texts:
        if t.layer not in ("TEXT", "ROOMS") or not re.search(r"[A-Za-zÄÖÜäöü]", t.text) or _DIM_RE.match(t.text):
            continue
        cx, cy = t.center
        if 0 <= int(cy) < H and 0 <= int(cx) < W and lab[int(cy), int(cx)] > 0:
            names_by_label.setdefault(int(lab[int(cy), int(cx)]), []).append(t.text.strip())
    unnamed = 0
    comps = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * k * k >= 1.0),
                   key=lambda i: (stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_LEFT]))
    for i in comps:
        comp = (lab == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        poly = _orthogonalize(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(float))
        if len(poly) < 3:
            continue
        names = names_by_label.get(i)
        if names:
            name, src = " / ".join(dict.fromkeys(names)), SOURCE_MEASURED
        else:
            unnamed += 1
            name, src = f"Raum {unnamed:02d}", SOURCE_ASSUMED
        area = float(stats[i, cv2.CC_STAT_AREA]) * k * k
        model.spaces.append(Space(name, [m(x, y) for x, y in poly], area, wall_h, src))

    # ------------------------------------------------------------ 5) Treppen + Decke
    for st in _stairs([l for l in lines if l.layer == "STAIRS"], px(0.6), m, k, plan_storey.height):
        model.stairs.append(st)

    slab_t = round(plan_storey.height - wall_h, 3)
    if slab_t >= 0.05:
        contours, _ = cv2.findContours(footprint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            poly = _orthogonalize(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(float))
            if len(poly) >= 3:
                model.slabs.append(Slab([m(x, y) for x, y in poly], slab_t))

    # ------------------------------------------------------------ Annahmen dokumentieren
    a = model.assumptions
    a.append(f"Wandhöhe {wall_h:.2f} m (Einstellung).")
    if s.auto_thickness:
        a.append("Wandstärken aus dem Plan gemessen.")
    else:
        a.append(f"Wandstärke {s.default_thickness:.2f} m für alle Wände (Einstellung).")
    if nd:
        a.append(f"Türbreiten aus dem Plan gemessen; Türhöhe {min(s.door_height, wall_h):.2f} m (Standardwert).")
    if nwin:
        a.append(f"Fensterbreiten aus dem Plan gemessen; Fensterhöhe {s.window_height:.2f} m und "
                 f"Brüstung {s.sill_height:.2f} m (Standardwerte).")
    if model.stairs:
        a.append("Treppe vereinfacht: Steigungshöhe = Geschosshöhe ÷ Anzahl Stufen, Laufrichtung angenommen.")
    if model.slabs:
        a.append(f"Bodenplatte aus dem Gebäudeumriss, Stärke {slab_t:.2f} m = Geschosshöhe − Wandhöhe.")
    if unnamed:
        a.append("Räume ohne erkannte Bezeichnung sind fortlaufend nummeriert (Raum 01 …).")
    model.not_created.append("Dach – aus einem Grundriss nicht zuverlässig ableitbar.")
    for st in model.storeys:
        if not st.is_plan:
            model.not_created.append(f"{st.name}: keine Bauteile (für dieses Geschoss wurde kein Grundriss hochgeladen).")
    if model_notes_leftover:
        model.not_created.append("Schräge oder runde Wandteile – nur im 2D-Plan enthalten.")
    if freestanding:
        model.not_created.append(f"{freestanding} kleine freistehende Element(e) (z.B. Kamin, Stütze) – nicht als Wand übernommen.")
    if door_skipped:
        model.warnings.append(f"{door_skipped} Tür(en) ohne eindeutig angrenzende Wand – nur im 2D-Plan enthalten.")
    if unhosted:
        model.warnings.append(f"{unhosted} Öffnung(en) keiner Wand zuzuordnen – nicht im Modell.")
    if not model.spaces:
        model.warnings.append("Keine geschlossenen Räume erkannt (Wände eventuell unterbrochen).")
    return model


# =============================================================================== Hilfen
def _ring(pts) -> np.ndarray:
    return np.round(np.array(pts, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _fill_wall_pairs(mask: np.ndarray, walls: list[Line], tmin: float, tmax: float) -> None:
    """Füllt den Raum zwischen zwei parallelen Wandlinien; unpaarige dicke Linien werden direkt gezeichnet."""
    paired: set[int] = set()
    for horiz in (True, False):
        cand = [l for l in walls if l.is_h == horiz]
        fx = (lambda l: l.p1[1]) if horiz else (lambda l: l.p1[0])
        iv = (lambda l: sorted((l.p1[0], l.p2[0]))) if horiz else (lambda l: sorted((l.p1[1], l.p2[1])))
        for a in cand:
            a0, a1 = iv(a)
            best, bd = None, None
            for b in cand:
                if b is a:
                    continue
                dist = abs(fx(b) - fx(a))
                if not (tmin <= dist <= tmax):
                    continue
                b0, b1 = iv(b)
                ov = min(a1, b1) - max(a0, b0)
                if ov < 0.3 * min(a1 - a0, b1 - b0):
                    continue
                if bd is None or dist < bd:
                    best, bd = b, dist
            if best is None:
                continue
            paired.add(id(a))
            paired.add(id(best))
            b0, b1 = iv(best)
            lo, hi = max(a0, b0), min(a1, b1)
            f0, f1 = sorted((fx(a), fx(best)))
            if horiz:
                cv2.rectangle(mask, (int(lo), int(f0)), (int(math.ceil(hi)), int(math.ceil(f1))), 255, -1)
            else:
                cv2.rectangle(mask, (int(f0), int(lo)), (int(math.ceil(f1)), int(math.ceil(hi))), 255, -1)
    for l in walls:
        if id(l) not in paired and l.width >= 3:
            cv2.line(mask, tuple(map(int, l.p1)), tuple(map(int, l.p2)), 255, max(1, int(round(l.width))))


def _window_rects(lines: list[Line], gap: float) -> list[dict]:
    """Gruppiert parallele Fensterlinien (über die Wandstärke verteilt) zu Fensterrechtecken."""
    out = []
    for horiz in (True, False):
        cand = [l for l in lines if (l.is_h if horiz else l.is_v)]
        groups: list[list[Line]] = []
        tol = (gap * 0.2, gap) if horiz else (gap, gap * 0.2)   # entlang knapp, quer bis Wandstärke
        for l in cand:
            bx = _bbox([l])
            hit = [g for g in groups if _boxes_touch(_bbox(g), bx, tol)]
            merged = [l] + [x for g in hit for x in g]
            groups = [g for g in groups if g not in hit] + [merged]
        for g in groups:
            if len(g) < 2:
                continue
            x0, y0, x1, y1 = _bbox(g)
            out.append({"orient": "h" if horiz else "v", "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return out


def _extend_to_jambs(o: dict, mask: np.ndarray, reach: float) -> None:
    """Fensterlinien enden oft kurz vor der Leibung – Öffnung bis zur Wand verlängern."""
    h = o["orient"] == "h"
    c = (o["y0"] + o["y1"]) / 2 if h else (o["x0"] + o["x1"]) / 2
    for end, sign in (("x0" if h else "y0", -1), ("x1" if h else "y1", 1)):
        start = o[end]
        for t in range(1, int(reach) + 1):
            x, y = (start + sign * t, c) if h else (c, start + sign * t)
            if _sample(mask, x, y):
                o[end] = start + sign * (t - 1)
                break


def _bbox(ls: list[Line]):
    xs = [p[0] for l in ls for p in (l.p1, l.p2)]
    ys = [p[1] for l in ls for p in (l.p1, l.p2)]
    return min(xs), min(ys), max(xs), max(ys)


def _boxes_touch(a, b, tol) -> bool:
    tx, ty = tol if isinstance(tol, tuple) else (tol, tol)
    return not (a[2] + tx < b[0] or b[2] + tx < a[0] or a[3] + ty < b[1] or b[3] + ty < a[1])


def _door_rect(a: Arc, mask: np.ndarray, tmax: float):
    """Bestimmt die Wandöffnung einer Tür anhand des Türbogens."""
    cx, cy = a.center
    r = a.radius
    best = None
    for ang in (a.start, a.end):
        d = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        if abs(d[0]) > 0.97:
            d = (math.copysign(1.0, d[0]), 0.0)
        elif abs(d[1]) > 0.97:
            d = (0.0, math.copysign(1.0, d[1]))
        else:
            continue
        n = (-d[1], d[0])
        # Wandquerschnitt kurz vor dem Anschlag suchen
        px0 = (cx - d[0] * 0.2 * r, cy - d[1] * 0.2 * r)
        vals = [_sample(mask, px0[0] + n[0] * t, px0[1] + n[1] * t) for t in range(-int(tmax), int(tmax) + 1)]
        run = _nearest_run(vals, int(tmax), max(4, 0.12 * r))
        if run is None:
            continue
        c0, c1 = run
        mid = (c0 + c1) / 2
        pm = (cx + n[0] * mid, cy + n[1] * mid)
        steps = np.linspace(0.05 * r, 0.45 * r, 12)
        cont = np.mean([_sample(mask, pm[0] - d[0] * t, pm[1] - d[1] * t) for t in steps])
        gapf = np.mean([not _sample(mask, pm[0] + d[0] * t, pm[1] + d[1] * t) for t in np.linspace(0.15 * r, 0.85 * r, 12)])
        if cont < 0.6 or gapf < 0.6:
            continue
        # tatsächliche Öffnungsbreite messen (bis die Wand wieder beginnt)
        length = r
        for t in np.arange(0.5 * r, 1.35 * r, 1.0):
            if _sample(mask, pm[0] + d[0] * t, pm[1] + d[1] * t):
                length = t
                break
        score = cont + gapf
        if best is None or score > best[0]:
            best = (score, d, n, c0, c1, length)
    if best is None:
        return None
    _, d, n, c0, c1, length = best
    pts = [(cx + d[0] * u + n[0] * v, cy + d[1] * u + n[1] * v) for u in (0.0, length) for v in (c0, c1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return {"orient": "h" if d[1] == 0 else "v", "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def _nearest_run(vals: list[bool], zero: int, tol: float):
    runs, start = [], None
    for i, v in enumerate(vals + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start - zero, i - 1 - zero))
            start = None
    best, bd = None, None
    for r0, r1 in runs:
        dist = 0 if r0 <= 0 <= r1 else min(abs(r0), abs(r1))
        if dist <= tol and (bd is None or dist < bd):
            best, bd = (r0, r1 + 1), dist
    return best


def _stairs(lines: list[Line], gap: float, m, k: float, storey_h: float) -> list[Stair]:
    out = []
    for horiz in (True, False):
        cand = [l for l in lines if (l.is_h if horiz else l.is_v)]
        groups: list[list[Line]] = []
        for l in cand:
            bx = _bbox([l])
            hit = [g for g in groups if _boxes_touch(_bbox(g), bx, gap)]
            groups = [g for g in groups if g not in hit] + [[l] + [x for g in hit for x in g]]
        for g in groups:
            if len(g) < 3:
                continue
            x0, y0, x1, y1 = _bbox(g)
            edges = sorted((l.p1[1] if horiz else l.p1[0]) for l in g)
            risers = len(edges)
            rise = storey_h / risers
            if horiz:   # Stufenkanten waagrecht -> Lauf in Bild-y-Richtung (CAD: nach unten)
                outline = [m(x0, y0), m(x1, y0), m(x1, y1), m(x0, y1)]
                direction = (0.0, -1.0)
                rel = [(e - y0) * k for e in edges]
                width = (x1 - x0) * k
            else:
                outline = [m(x0, y0), m(x0, y1), m(x1, y1), m(x1, y0)]
                direction = (1.0, 0.0)
                rel = [(e - x0) * k for e in edges]
                width = (y1 - y0) * k
            out.append(Stair(outline, direction, 0.0, rel, round(width, 3), risers, round(rise, 4)))
    return out


def _storeys(s: Settings3D, texts: list[Text]):
    detected = None
    best = -1
    for t in texts:
        mt = _STOREY_RE.search(t.text)
        if not mt:
            continue
        raw = mt.group(1)
        name = _STOREY_LONG.get(raw.lower(), raw.upper().replace(" ", ""))
        score = 2 if re.search(r"grundriss|geschoss", t.text, re.I) else 1
        if score > best:
            best, detected = score, name
    rows = list(s.storeys)
    notes = []
    names = [r[0] for r in rows]
    if s.plan_storey != "auto":
        plan = s.plan_storey
        notes.append(f"Geschoss «{plan}» gemäss Einstellung.")
    elif detected and detected.upper() in [n.upper() for n in names]:
        plan = names[[n.upper() for n in names].index(detected.upper())]
        notes.append(f"Geschoss «{plan}» aus der Planbeschriftung erkannt.")
    elif detected and len(rows) == 1:
        rows = [(detected, rows[0][1])]
        plan = detected
        notes.append(f"Geschoss «{plan}» aus der Planbeschriftung erkannt.")
    else:
        plan = "EG" if "EG" in names else names[0]
        if detected:
            notes.append(f"Erkanntes Geschoss «{detected}» fehlt in der Geschosstabelle – Plan wurde «{plan}» zugeordnet.")
        else:
            notes.append(f"Geschoss nicht erkennbar – Plan wurde «{plan}» zugeordnet.")
    names = [r[0] for r in rows]
    ref = names.index("EG") if "EG" in names else 0
    elev = [0.0] * len(rows)
    for i in range(ref + 1, len(rows)):
        elev[i] = elev[i - 1] + rows[i - 1][1]
    for i in range(ref - 1, -1, -1):
        elev[i] = elev[i + 1] - rows[i][1]
    storeys = [Storey(n, round(e, 3), h, n == plan) for (n, h), e in zip(rows, elev)]
    plan_storey = next(st for st in storeys if st.is_plan)
    return storeys, plan_storey, detected, notes

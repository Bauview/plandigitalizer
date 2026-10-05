"""Rekonstruktion des Wesentlichen: Wände, Fenster, Türen – sauber und CAD-gerecht.

Eingang ist die Pixel-Klassifikation des Netzes (``semantic``) und das Schwarz-Weiss-Bild.
Ergebnis sind exakte, achsparallele Wandumrisse mit sauber eingeschnittenen Öffnungen,
genormte Fenster- und Türsymbole (Türanschlag aus dem gezeichneten Bogen gelesen) und ein
``SemanticPlan`` für das IFC-/3D-Modell. Möbel, Massketten, Schraffuren und Texte fliessen
hier nicht ein.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import re

from .geometry import Arc, Hatch, Line, Polyline, Text
from .semantic import DOOR, WALL, WINDOW
from .vectorize import _orthogonalize
from .walls import decompose, footprint, median_thickness, merge_rects, raster_rects
from . import trace


@dataclass
class Opening:
    kind: str                   # "door" | "window"
    x0: float
    y0: float
    x1: float
    y1: float
    orient: str                 # Richtung der Wand: "h" | "v"
    swing: list = field(default_factory=list)   # Türflügel: [(hinge_x, hinge_y, leaf_dx, leaf_dy, radius, jamb_dx, jamb_dy)]

    @property
    def width(self) -> float:
        return (self.x1 - self.x0) if self.orient == "h" else (self.y1 - self.y0)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "orient": self.orient, "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class SemanticPlan:
    rects: list[tuple]
    leftover: np.ndarray            # nicht achsparallele Wandteile (Maske)
    openings: list[Opening]
    wall_px: float
    band: np.ndarray                # Wände + Öffnungen (Maske)
    walls: np.ndarray               # Wände ohne Öffnungen (bereinigt)
    entities: list = field(default_factory=list)


# =============================================================================== Hauptfunktion
def reconstruct(labels: np.ndarray, binary: np.ndarray, gray: np.ndarray | None = None) -> SemanticPlan | None:
    H, W = labels.shape
    wall = ((labels == WALL) * 255).astype(np.uint8)
    t = median_thickness(wall)
    if t <= 0 or np.count_nonzero(wall) < 50:
        return None
    t = max(2.0, t)
    # Wände schwarz gefüllt (CAD-Standard): Geometrie und Öffnungen direkt aus der Wandfüllung
    po = None
    if gray is not None:
        from .poche import poche_labels
        pl = poche_labels(gray, binary, labels, t)
        if pl is not None:
            labels, po = pl
            wall = ((labels == WALL) * 255).astype(np.uint8)
            t = max(2.0, median_thickness(wall) or t)
    ink = (binary > 0).astype(np.uint8)
    # dünne, nur als Doppellinie gezeichnete Wände (Leichtbau) ergänzen, die das Netz übersehen hat
    thin = trace.double_line_walls(binary, labels, t)
    thick = np.zeros_like(thin)       # (Variante für volle Wandstärke: trace.double_line_walls(..., min_fill>0))
    thin_d = None
    if np.any(thin) or np.any(thick):
        labels = labels.copy()
        labels[thin > 0] = WALL
        # dicke Ergänzungen nur dort, wo das Netz nichts erkannt hat (Öffnungen bleiben Öffnungen)
        labels[(thick > 0) & (labels == 0)] = WALL
        wall = ((labels == WALL) * 255).astype(np.uint8)
        if np.any(thin):
            thin_d = cv2.dilate(thin, np.ones((3, 3), np.uint8)) > 0

    # ------------------------------------------------------------ 1) bereinigen
    k3 = np.ones((3, 3), np.uint8)
    band0 = ((labels > 0) * 255).astype(np.uint8)
    band0 = cv2.morphologyEx(band0, cv2.MORPH_OPEN, k3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(band0, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        ext = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        if st[i, cv2.CC_STAT_AREA] >= 2.0 * t * t and ext >= 3.0 * t:
            keep[i] = 255
    # kleine, isolierte Stücke (Möbel, Symbole, Nordpfeil) gehören nicht zum Gebäude
    if n > 1:
        imax = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        amax = float(st[imax, cv2.CC_STAT_AREA])
        for i in range(1, n):
            ext = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
            if keep[i] and st[i, cv2.CC_STAT_AREA] < 0.04 * amax and ext < 10 * t:
                keep[i] = 0
    band0 = keep[lab]
    wall = cv2.bitwise_and(wall, band0)

    # ------------------------------------------------------------ 2) Öffnungen
    openings: list[Opening] = []
    for kind, cls in (("window", WINDOW), ("door", DOOR)):
        m = cv2.bitwise_and(((labels == cls) * 255).astype(np.uint8), band0)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(max(3, t * 0.6)) | 1,) * 2, np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < 0.35 * t * t or max(w, h) < 0.9 * t:
                continue
            if thin_d is not None and np.count_nonzero(thin_d[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)) > 0.5 * area:
                continue                    # liegt in einer durchgehend gezeichneten Leichtbauwand
            o = _opening_rect(kind, x, y, w, h, wall, t)
            if o is not None and thin_d is not None and _axis_cover(o, thin_d) > 0.6:
                continue                    # Mittellinie der "Öffnung" verläuft in einer Leichtbauwand
            if o is not None:
                openings.append(o)
    openings = _dedupe(openings)
    # Schein-Öffnungen (z.B. an Wandanschlüssen) sind viel schmaler als echte Türen/Fenster -> Wand
    if openings:
        med = float(np.median([o.width for o in openings]))
        def cross(o):
            return (o.y1 - o.y0) if o.orient == "h" else (o.x1 - o.x0)
        # zu schmal, oder quer viel dicker als jede Wand (Fehlerkennung an Wandanschlüssen)
        small = [o for o in openings if o.width < max(1.0 * t, 0.42 * med) or cross(o) > 2.2 * t]
        for o in small:
            _bridge(wall, o, t)
        openings = [o for o in openings if o not in small]

    # Türen prüfen: Anschlag aus dem gezeichneten Bogen suchen. Ohne Anschlag und mit durchlaufenden
    # Wandlinien (oder voller Wandfüllung) ist es keine Öffnung, sondern Wand.
    tol_ink = _symbol_ink(ink, wall, t)
    fake = []
    for o in openings:
        if o.kind != "door":
            continue
        o.swing = _door_swing(o, tol_ink)
        if (not o.swing or len(o.swing) == 2) and _faces_continuous(o, ink, t):
            fake.append(o)
    for o in fake:
        cv2.rectangle(wall, (int(o.x0), int(o.y0)), (int(math.ceil(o.x1)) - 1, int(math.ceil(o.y1)) - 1), 255, -1)
    openings = [o for o in openings if o not in fake]

    # ------------------------------------------------------------ 3) Wandrechtecke
    band = wall.copy()
    for o in openings:
        cv2.rectangle(band, (int(o.x0), int(o.y0)), (int(math.ceil(o.x1)) - 1, int(math.ceil(o.y1)) - 1), 255, -1)
    gk = max(3, int(round(0.5 * t)) | 1)
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gk, gk)))
    rects, leftover = decompose(band, max(3 * t, 12), t_est=t)
    rects = merge_rects(rects, max(2.0, 0.5 * t))
    rects = [_snap_rect(r, ink, t) for r in rects]
    rects = _extend_along_ink(rects, ink, t, openings)
    rects = _square_junctions(rects, t)
    rects = [r for r in rects if max(r[2] - r[0], r[3] - r[1]) >= 1.2 * min(r[2] - r[0], r[3] - r[1]) or
             (r[2] - r[0]) * (r[3] - r[1]) >= 1.5 * t * t]

    # Öffnungen auf die Wandstärke der tragenden Wand ausrichten + Jamben einrasten
    for o in openings:
        host = _host(o, rects)
        if host is not None:
            if o.orient == "h":
                o.y0, o.y1 = host[1], host[3]
            else:
                o.x0, o.x1 = host[0], host[2]
        _snap_jambs(o, ink, t)
    band_clean = raster_rects((H, W), rects)
    band_clean = cv2.bitwise_or(band_clean, leftover)
    _interior_windows_to_doors(openings, band_clean, t)
    # zweite Prüfung mit ausgerichteten Öffnungen: Tür ohne Anschlag, Wandkanten laufen durch -> Wand
    for o in openings:
        if o.kind == "door" and not o.swing:
            o.swing = _door_swing(o, tol_ink)
    openings = [o for o in openings
                if not (o.kind == "door" and (not o.swing or len(o.swing) == 2) and _faces_continuous(o, ink, t))]
    walls = band_clean.copy()
    for o in openings:
        ex = 2
        if o.orient == "h":
            cv2.rectangle(walls, (int(round(o.x0)), int(round(o.y0)) - ex), (int(round(o.x1)) - 1, int(round(o.y1)) - 1 + ex), 0, -1)
        else:
            cv2.rectangle(walls, (int(round(o.x0)) - ex, int(round(o.y0))), (int(round(o.x1)) - 1 + ex, int(round(o.y1)) - 1), 0, -1)
    # winzige Wandreste (z.B. zwischen zwei Öffnungen) entfernen
    n, lab, st, _ = cv2.connectedComponentsWithStats(walls, connectivity=4)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 0.6 * t * t:
            walls[lab == i] = 0

    sp = SemanticPlan(rects, leftover, openings, t, band_clean, walls)
    # Wandgeometrie 1:1 aus der gezeichneten Tinte nachzeichnen (schräge Leibungen, Nischen ...)
    sw = trace.stroke_width(binary, walls, t)
    omask = np.zeros((H, W), np.uint8)
    for o in openings:              # Öffnungen: Leibungslinien liegen darin und gehören zur Wand
        x0, y0, x1, y1 = o.x0, o.y0, o.x1, o.y1
        if x1 > x0 and y1 > y0:
            cv2.rectangle(omask, (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1), 255, -1)
    material = trace.wall_material(binary, walls, omask, t, sw, rects)
    if po is not None:
        # gefüllte Wände: die (bereinigte) Wandfläche selbst ist die Geometrie – kein Nachzeichnen von
        # angehängten Linien (Lichtschächte, Fensterrahmen, Möbel) und kein Scanrauschen
        material = walls.copy()
    regions = trace.wall_rings(material, sw, t, solid=po)
    if regions:
        traced = np.zeros((H, W), np.uint8)
        for rings in regions:
            cv2.fillPoly(traced, [np.round(np.array(r) - 0.5).astype(np.int32) for r in rings], 255)
        sp.walls = traced
        sp.stroke_px = sw
        sp.entities = _region_entities(regions)
    else:
        sp.entities = _wall_entities(walls, t)
    return sp


# =============================================================================== Öffnungen
def _axis_cover(o: "Opening", mask: np.ndarray) -> float:
    """Anteil der Öffnungs-Mittellinie (in Wandrichtung), der in ``mask`` liegt."""
    H, W = mask.shape
    if o.orient == "h":
        c = int((o.y0 + o.y1) / 2)
        xs = np.arange(int(o.x0), int(math.ceil(o.x1)))
        xs = xs[(xs >= 0) & (xs < W)]
        return float(mask[c, xs].mean()) if 0 <= c < H and xs.size else 0.0
    c = int((o.x0 + o.x1) / 2)
    ys = np.arange(int(o.y0), int(math.ceil(o.y1)))
    ys = ys[(ys >= 0) & (ys < H)]
    return float(mask[ys, c].mean()) if 0 <= c < W and ys.size else 0.0


def _frac(mask, x0, y0, x1, y1) -> float:
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(mask[y0:y1, x0:x1])) / ((x1 - x0) * (y1 - y0))


def _opening_rect(kind, x, y, w, h, wall, t) -> Opening | None:
    r = max(3, int(round(0.9 * t)))
    lr = _frac(wall, x - r, y, x, y + h) + _frac(wall, x + w, y, x + w + r, y + h)
    tb = _frac(wall, x, y - r, x + w, y) + _frac(wall, x, y + h, x + w, y + h + r)
    orient = "h" if lr >= tb else "v"
    if max(lr, tb) < 0.15 and max(w, h) < 2 * t:
        return None
    if orient == "h":
        a0, a1, c0, c1 = float(x), float(x + w), float(y), float(y + h)
        cc = _cross_extent(wall, a0, a1, c0, c1, r, t, horiz=True)
    else:
        a0, a1, c0, c1 = float(y), float(y + h), float(x), float(x + w)
        cc = _cross_extent(wall, a0, a1, c0, c1, r, t, horiz=False)
    if cc and (cc[1] - cc[0]) <= 1.8 * max(c1 - c0, 0.6 * t):
        c0, c1 = cc
    # bis zur Wand verlängern (Netz lässt an den Jamben oft 1–2 px frei)
    cm = (c0 + c1) / 2
    a0 = _extend(wall, a0, cm, -1, r, orient)
    a1 = _extend(wall, a1, cm, +1, r, orient)
    if a1 - a0 < 0.9 * t:
        return None
    if orient == "h":
        return Opening(kind, a0, c0, a1, c1, "h")
    return Opening(kind, c0, a0, c1, a1, "v")


def _cross_extent(wall, a0, a1, c0, c1, r, t, horiz):
    """Lage der Wand quer zur Öffnung: aus den Wandstücken links/rechts der Öffnung."""
    lo, hi = int(c0 - 1.5 * t), int(c1 + 1.5 * t)
    tops, bots = [], []
    for s0, s1 in ((a0 - r, a0), (a1, a1 + r)):
        if horiz:
            sub = wall[max(0, lo):max(0, hi), max(0, int(s0)):max(0, int(s1))]
        else:
            sub = wall[max(0, int(s0)):max(0, int(s1)), max(0, lo):max(0, hi)].T
        if sub.size == 0:
            continue
        prof = (sub > 0).mean(axis=1) > 0.5
        runs = _runs(prof)
        if not runs:
            continue
        cm = (c0 + c1) / 2 - max(0, lo)
        best = min(runs, key=lambda ab: 0 if ab[0] <= cm <= ab[1] else min(abs(cm - ab[0]), abs(cm - ab[1])))
        tops.append(best[0] + max(0, lo))
        bots.append(best[1] + 1 + max(0, lo))
    if not tops:
        return None
    return float(np.median(tops)), float(np.median(bots))


def _runs(prof):
    out, s = [], None
    for i, v in enumerate(list(prof) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def _extend(wall, a, cm, sign, r, orient):
    for k in range(0, r + 1):
        x, y = (a + sign * (k + 0.5), cm) if orient == "h" else (cm, a + sign * (k + 0.5))
        xi, yi = int(x), int(y)
        if 0 <= yi < wall.shape[0] and 0 <= xi < wall.shape[1] and wall[yi, xi]:
            return a + sign * k
    return a


def _bridge(wall: np.ndarray, o: Opening, t: float) -> None:
    """Füllt eine (Schein-)Öffnung mit dem Querschnitt der angrenzenden Wand."""
    H, W = wall.shape
    m = 1
    x0, y0, x1, y1 = int(o.x0), int(o.y0), int(math.ceil(o.x1)), int(math.ceil(o.y1))
    if o.orient == "v":                 # Wand läuft senkrecht: Spalten der Wand oben/unten übernehmen
        cols = np.arange(max(0, x0 - m), min(W, x1 + m))
        above = wall[max(0, y0 - 3):max(0, y0 - 1), cols].any(axis=0) if y0 > 2 else np.zeros(len(cols), bool)
        below = wall[min(H, y1 + 1):min(H, y1 + 3), cols].any(axis=0) if y1 < H - 2 else np.zeros(len(cols), bool)
        sel = _narrow(cols, above, below)
        wall[y0:y1, sel if sel.size else cols] = 255
    else:
        rows = np.arange(max(0, y0 - m), min(H, y1 + m))
        left = wall[rows, max(0, x0 - 3):max(0, x0 - 1)].any(axis=1) if x0 > 2 else np.zeros(len(rows), bool)
        right = wall[rows, min(W, x1 + 1):min(W, x1 + 3)].any(axis=1) if x1 < W - 2 else np.zeros(len(rows), bool)
        sel = _narrow(rows, left, right)
        wall[sel if sel.size else rows, x0:x1] = 255


def _narrow(idx, a, b):
    """Profil der durchlaufenden Wand: nur wo auf beiden Seiten der Öffnung Wand liegt.
    Liegt nur auf einer Seite Wand (Wandende), wird das schmalere Profil verwendet."""
    both = a & b
    if both.any():
        return idx[both]
    cand = [m for m in (a, b) if m.any()]
    if not cand:
        return idx[:0]
    return idx[min(cand, key=lambda m: int(m.sum()))]


def _symbol_ink(ink: np.ndarray, wall: np.ndarray, t: float) -> np.ndarray:
    """Tinte ausserhalb der Wände, leicht verbreitert (für die Suche nach Türbögen)."""
    out = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(wall, np.ones((3, 3), np.uint8))) // 255)
    rad = max(1, int(round(0.12 * t)))
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2))


def _faces_continuous(o: "Opening", ink: np.ndarray, t: float) -> bool:
    """Laufen beide Wandkanten (oder eine volle Füllung) durch die angebliche Öffnung?"""
    H, W = ink.shape
    d = max(2, int(round(0.18 * t)))
    if o.orient == "h":
        a0, a1 = int(o.x0 + 0.1 * (o.x1 - o.x0)), int(o.x1 - 0.1 * (o.x1 - o.x0))
        f0, f1 = int(o.y0), int(math.ceil(o.y1))
        if a1 <= a0 or f1 - f0 < 2:
            return False
        top = ink[max(0, f0 - 1):min(H, f0 + d), a0:a1].any(axis=0).mean()
        bot = ink[max(0, f1 - d):min(H, f1 + 1), a0:a1].any(axis=0).mean()
        fill = ink[f0:f1, a0:a1].mean()
    else:
        a0, a1 = int(o.y0 + 0.1 * (o.y1 - o.y0)), int(o.y1 - 0.1 * (o.y1 - o.y0))
        f0, f1 = int(o.x0), int(math.ceil(o.x1))
        if a1 <= a0 or f1 - f0 < 2:
            return False
        top = ink[a0:a1, max(0, f0 - 1):min(W, f0 + d)].any(axis=1).mean()
        bot = ink[a0:a1, max(0, f1 - d):min(W, f1 + 1)].any(axis=1).mean()
        fill = ink[a0:a1, f0:f1].mean()
    return (top >= 0.85 and bot >= 0.85) or fill >= 0.7


def _extend_along_ink(rects, ink, t, openings, max_mul=8.0):
    """Wandenden verlängern, solange die gezeichnete Wand (beide Kanten oder Füllung) weiterläuft."""
    H, W = ink.shape
    d = max(2, int(round(0.18 * t)))
    occ = raster_rects((H, W), rects)
    for o in openings:
        cv2.rectangle(occ, (int(o.x0), int(o.y0)), (int(math.ceil(o.x1)) - 1, int(math.ceil(o.y1)) - 1), 255, -1)
    out = []
    for r in rects:
        x0, y0, x1, y1, ori = r
        r = list(r)
        for end, sign in ((0, -1), (1, +1)):
            if ori == "h":
                f0, f1 = int(round(y0)), int(round(y1))
                pos = int(round(x1)) if sign > 0 else int(round(x0)) - 1
            else:
                f0, f1 = int(round(x0)), int(round(x1))
                pos = int(round(y1)) if sign > 0 else int(round(y0)) - 1
            if f1 - f0 < 2:
                continue
            steps, miss, hit = 0, 0, False
            limit = int(max_mul * t)
            p = pos
            while steps < limit:
                if ori == "h":
                    if not (0 <= p < W):
                        break
                    col_occ = occ[f0 + 1:f1 - 1, p].mean() > 0.5 * 255
                    a = ink[max(0, f0 - 1):f0 + d, p].any()
                    b = ink[f1 - d:min(H, f1 + 1), p].any()
                    fl = ink[f0:f1, p].mean()
                else:
                    if not (0 <= p < H):
                        break
                    col_occ = occ[p, f0 + 1:f1 - 1].mean() > 0.5 * 255
                    a = ink[p, max(0, f0 - 1):f0 + d].any()
                    b = ink[p, f1 - d:min(W, f1 + 1)].any()
                    fl = ink[p, f0:f1].mean()
                if col_occ:
                    hit = True
                    break
                if (a and b) or fl >= 0.7:
                    miss = 0
                else:
                    miss += 1
                    if miss > 2:
                        break
                p += sign
                steps += 1
            ext = steps - miss
            if hit or ext >= 0.5 * t:
                if hit:
                    ext = steps
                if ori == "h":
                    if sign > 0:
                        r[2] = x1 + ext
                    else:
                        r[0] = x0 - ext
                else:
                    if sign > 0:
                        r[3] = y1 + ext
                    else:
                        r[1] = y0 - ext
        out.append(tuple(r))
    return out


def _dedupe(ops: list[Opening]) -> list[Opening]:
    out: list[Opening] = []
    for o in sorted(ops, key=lambda o: -o.width):
        dup = False
        for p in out:
            ix = min(o.x1, p.x1) - max(o.x0, p.x0)
            iy = min(o.y1, p.y1) - max(o.y0, p.y0)
            if ix > 0 and iy > 0 and ix * iy > 0.3 * (o.x1 - o.x0) * (o.y1 - o.y0):
                dup = True
                break
        if not dup:
            out.append(o)
    return out


def _host(o: Opening, rects):
    best, ba = None, 0.0
    for r in rects:
        if r[4] != o.orient:
            continue
        ix = min(r[2], o.x1) - max(r[0], o.x0)
        iy = min(r[3], o.y1) - max(r[1], o.y0)
        if ix > 0 and iy > 0 and ix * iy > ba:
            best, ba = r, ix * iy
    return best


def _interior_windows_to_doors(openings: list[Opening], band: np.ndarray, t: float) -> None:
    """Fenster liegen in Aussenwänden. Öffnungen mit Innenraum auf beiden Seiten sind Durchgänge."""
    foot = footprint(band, int(max(3, 4 * t)))
    wins = [o for o in openings if o.kind == "window"]
    if not wins:
        return
    interior = []
    for o in wins:
        off = 0.5 * t + 3
        sides = []
        for sgn in (-1, 1):
            pts = []
            for f in (0.25, 0.5, 0.75):
                if o.orient == "h":
                    x = o.x0 + f * (o.x1 - o.x0)
                    y = (o.y0 - off) if sgn < 0 else (o.y1 + off)
                else:
                    y = o.y0 + f * (o.y1 - o.y0)
                    x = (o.x0 - off) if sgn < 0 else (o.x1 + off)
                xi, yi = int(x), int(y)
                pts.append(0 <= yi < foot.shape[0] and 0 <= xi < foot.shape[1] and foot[yi, xi] > 0)
            sides.append(sum(pts) >= 2)
        if all(sides):
            interior.append(o)
    if len(interior) < len(wins):          # nur wenn der Gebäudeumriss plausibel ist
        for o in interior:
            o.kind = "door"


# =============================================================================== Kanten einrasten
def _profile_edge(ink, fixed_lo, fixed_hi, pos, rng, horiz_edge, inside_sign, rng_in=None):
    """Sucht nahe ``pos`` die Kante, an der Tinte (innen) in Papier (aussen) übergeht.

    Nach aussen wird bis ``rng`` gesucht, nach innen nur bis ``rng_in`` – sonst rastet die Kante
    bei Umriss-Wänden (zwei Linien, innen weiss) auf die innere Linie ein und die Wand wird zu dünn.
    """
    H, W = ink.shape
    n = H if horiz_edge else W
    rng_in = rng if rng_in is None else rng_in
    best, bscore = pos, -1e9
    lo, hi = int(max(0, fixed_lo)), int(min(W if horiz_edge else H, fixed_hi))
    if hi - lo < 3:
        return pos
    # inside_sign > 0: Wand liegt bei grösseren Koordinaten -> "aussen" = kleinere Werte
    for d in range(-rng, rng + 1):
        inward = d * inside_sign > 0
        if inward and abs(d) > rng_in:
            continue
        p = int(round(pos)) + d
        a = p if inside_sign > 0 else p - 1          # erste Innenzeile
        b = p - 1 if inside_sign > 0 else p          # erste Aussenzeile
        if not (0 <= a < n and 0 <= b < n):
            continue
        if horiz_edge:
            vin, vout = ink[a, lo:hi].mean(), ink[b, lo:hi].mean()
        else:
            vin, vout = ink[lo:hi, a].mean(), ink[lo:hi, b].mean()
        score = vin - vout - 0.02 * abs(d)
        if score > bscore:
            best, bscore = float(p), score
    return best if bscore > 0.25 else pos


def _snap_rect(r, ink, t):
    x0, y0, x1, y1, o = r
    rg = max(2, int(round(0.45 * t)))
    ri = max(1, int(round(0.15 * t)))
    if o == "h":
        m = 0.15 * (x1 - x0)
        ny0 = _profile_edge(ink, x0 + m, x1 - m, y0, rg, True, +1, ri)
        ny1 = _profile_edge(ink, x0 + m, x1 - m, y1, rg, True, -1, ri)
        if ny1 - ny0 >= 0.75 * (y1 - y0) and ny1 > ny0 + 1:
            y0, y1 = ny0, ny1
    else:
        m = 0.15 * (y1 - y0)
        nx0 = _profile_edge(ink, y0 + m, y1 - m, x0, rg, False, +1, ri)
        nx1 = _profile_edge(ink, y0 + m, y1 - m, x1, rg, False, -1, ri)
        if nx1 - nx0 >= 0.75 * (x1 - x0) and nx1 > nx0 + 1:
            x0, x1 = nx0, nx1
    return (x0, y0, x1, y1, o)


def _square_junctions(rects, t):
    """Wandenden an querlaufende Wände anschliessen: T-Stoss bis in die Wand, L-Ecke bis zur Aussenkante."""
    out = [list(r) for r in rects]
    tol = max(2.0, 0.6 * t)
    for a in out:
        for b in out:
            if a is b or a[4] == b[4]:
                continue
            # Achsen so tauschen, dass a "waagrecht" betrachtet wird
            if a[4] == "h":
                ia0, ia1, ic0, ic1 = 0, 2, 1, 3     # a: Längsrichtung x, Querrichtung y
            else:
                ia0, ia1, ic0, ic1 = 1, 3, 0, 2
            # b in derselben Notation: b-Quer = a-Längsrichtung
            bq0, bq1 = b[ia0], b[ia1]           # b-Stärke entlang a
            bl0, bl1 = b[ic0], b[ic1]           # b-Länge quer zu a
            acm = (a[ic0] + a[ic1]) / 2
            if not (bl0 - tol <= acm <= bl1 + tol):
                continue
            for end in (ia0, ia1):
                x = a[end]
                if not (bq0 - tol <= x <= bq1 + tol):
                    continue
                through = bl0 < a[ic0] - tol and bl1 > a[ic1] + tol
                if through:                     # T-Stoss: a endet in der Mitte von b
                    a[end] = (bq0 + bq1) / 2
                else:                           # Ecke: beide bis zur Aussenkante
                    a[end] = bq1 if end == ia1 else bq0
                    if bl1 > a[ic1]:
                        b[ic0] = min(b[ic0], a[ic0])
                    if bl0 < a[ic0]:
                        b[ic1] = max(b[ic1], a[ic1])
    return [tuple(r) for r in out]


def _snap_jambs(o: Opening, ink, t):
    rg = max(2, int(round(0.3 * t)))
    if o.orient == "h":
        q0, q1 = o.y0 + 0.2 * (o.y1 - o.y0), o.y1 - 0.2 * (o.y1 - o.y0)
        o.x0 = _profile_edge(ink, q0, q1, o.x0, rg, False, -1)
        o.x1 = _profile_edge(ink, q0, q1, o.x1, rg, False, +1)
    else:
        q0, q1 = o.x0 + 0.2 * (o.x1 - o.x0), o.x1 - 0.2 * (o.x1 - o.x0)
        o.y0 = _profile_edge(ink, q0, q1, o.y0, rg, True, -1)
        o.y1 = _profile_edge(ink, q0, q1, o.y1, rg, True, +1)


# =============================================================================== Ausgabe
def _wall_entities(walls: np.ndarray, t: float) -> list:
    ents: list = []
    contours, hier = cv2.findContours(walls, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return ents
    hier = hier[0]
    eps = max(1.0, 0.12 * t)
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue
        rings = [_ring(c, eps)]
        ch = hier[i][2]
        while ch != -1:
            if cv2.contourArea(contours[ch]) > 0.5 * t * t:
                rings.append(_ring(contours[ch], eps))
            ch = hier[ch][0]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        for r in rings:
            ents.append(Polyline(r, True, "WALLS"))
        ents.append(Hatch(rings, "HATCH"))
    return ents


def _region_entities(regions) -> list:
    ents: list = []
    for rings in regions:
        for r in rings:
            ents.append(Polyline(r, True, "WALLS"))
        ents.append(Hatch(rings, "HATCH"))
    return ents


def _ring(c, eps):
    pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float) + 0.5
    return _orthogonalize(pts, tol_deg=4.0)


def opening_entities(sp: SemanticPlan, binary: np.ndarray) -> list:
    """Fenster- und Türsymbole.

    Fenster: Die im Plan gezeichneten Linien in der Öffnung (Rahmen, Glas, Leibung) und eine
    aussen gezeichnete Fensterbank werden aus dem Bild gelesen und sauber nachgezogen. Ist nichts
    erkennbar, wird ein Standardsymbol (Rahmen + Glas) gesetzt.
    Türen: Türblatt (schmales Rechteck) + Anschlagbogen, Drehpunkt und Aufschlag aus dem
    gezeichneten Bogen. Ohne erkennbaren Bogen bleibt die Öffnung ohne Symbol.
    """
    t = sp.wall_px
    ents: list = []
    ink = (binary > 0).astype(np.uint8)
    tol_ink = _symbol_ink(ink, sp.walls, t)
    outside = cv2.bitwise_not(footprint_mask(sp)) > 0
    sw = getattr(sp, "stroke_px", 2.0)
    g = int(max(1, round(sw / 2 + 1)))
    wall_cut = cv2.dilate(sp.walls, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0
    sym_ink = (ink > 0) & ~wall_cut
    for o in sp.openings:
        if o.kind == "window":
            got = _window_from_ink(o, ink, wall_cut, t, outside)
            ents.extend(got if got else _window_symbol(o, ink, outside, t))
        else:
            if not o.swing:
                o.swing = _door_swing(o, tol_ink)
            for swing in o.swing:
                hx, hy, ldx, ldy, r, jdx, jdy = swing
                (hx, hy, r), (q0, q1) = _refine_door(swing, sym_ink)
                tip = (hx + ldx * r, hy + ldy * r)
                # Türblatt wie gezeichnet (Lage/Dicke quer zur Blattrichtung), sonst ca. 4 cm
                if q1 - q0 < 1.0:
                    q0, q1 = 0.0, max(1.5, 0.045 * r)
                ents.append(Polyline([(hx + jdx * q0, hy + jdy * q0), (tip[0] + jdx * q0, tip[1] + jdy * q0),
                                      (tip[0] + jdx * q1, tip[1] + jdy * q1), (hx + jdx * q1, hy + jdy * q1)],
                                     True, "DOORS"))
                a_leaf = math.degrees(math.atan2(ldy, ldx)) % 360
                a_j = math.degrees(math.atan2(jdy, jdx)) % 360
                if (a_j - a_leaf) % 360 <= 180:
                    ents.append(Arc((hx, hy), r, a_leaf, a_leaf + ((a_j - a_leaf) % 360), "DOORS"))
                else:
                    ents.append(Arc((hx, hy), r, a_j, a_j + ((a_leaf - a_j) % 360), "DOORS"))
    return ents


def _refine_door(swing, ink: np.ndarray):
    """Drehpunkt und Radius aus dem gezeichneten Bogen (Kreisausgleich), Türblatt-Lage aus der Tinte.

    Gibt ((hx, hy, r), (q0, q1)) zurück; q0..q1 = Lage des Blattes quer zur Blattrichtung (zur Öffnung hin
    positiv), (0, 0) wenn nicht messbar."""
    hx, hy, dx, dy, r, jx, jy = swing
    H, W = ink.shape
    R = int(1.25 * r) + 2
    x0, x1, y0, y1 = max(0, int(hx) - R), min(W, int(hx) + R), max(0, int(hy) - R), min(H, int(hy) + R)
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    if xs.size < 10:
        return (hx, hy, r), (0.0, 0.0)
    px, py = xs + x0 + 0.5 - hx, ys + y0 + 0.5 - hy
    u = px * dx + py * dy                     # Blattrichtung
    v = px * jx + py * jy                     # Richtung Leibung
    rho = np.hypot(u, v)
    ang = np.degrees(np.arctan2(v, u))
    sel = (ang > 18) & (ang < 72) & (np.abs(rho - r) < 0.15 * r)
    out = (hx, hy, r)
    if np.count_nonzero(sel) >= 12:
        X, Y = px[sel] + hx, py[sel] + hy
        A = np.c_[2 * X, 2 * Y, np.ones_like(X)]
        b = X * X + Y * Y
        try:
            (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
            rr = math.sqrt(max(1e-6, c + cx * cx + cy * cy))
            res = np.abs(np.hypot(X - cx, Y - cy) - rr)
            if math.hypot(cx - hx, cy - hy) < 0.12 * r and abs(rr - r) < 0.12 * r and np.median(res) < 0.03 * r + 1:
                out = (float(cx), float(cy), float(rr))
        except np.linalg.LinAlgError:
            pass
    # Türblatt: Tinte entlang der Blattrichtung (20–85 % der Länge), quer dazu nahe am Drehpunkt
    blade = (u > 0.2 * r) & (u < 0.85 * r) & (np.abs(v) < 0.12 * r)
    if np.count_nonzero(blade) < 6:
        return out, (0.0, 0.0)
    vv = v[blade]
    lo, hi = np.percentile(vv, 3), np.percentile(vv, 97)
    if hi - lo > 0.1 * r:
        return out, (0.0, 0.0)
    # Achse = Mitte der Strichränder; Strichbreite der Linien bleibt aussen vor
    return out, (float(lo), float(hi))


def _line_positions(prof: np.ndarray, thr: float = 0.6) -> list[float]:
    """Mittellagen zusammenhängender Zeilen mit viel Tinte (= durchgehende Linien)."""
    out, start = [], None
    for i, v in enumerate(list(prof) + [0.0]):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            out.append((start + i - 1) / 2.0)
            start = None
    return out


def _runs_1d(v: np.ndarray, thr: float = 0.5):
    """(Start, Ende) zusammenhängender Abschnitte mit v >= thr."""
    out, start = [], None
    for i, x in enumerate(list(v) + [0.0]):
        if x >= thr and start is None:
            start = i
        elif x < thr and start is not None:
            out.append((start, i))
            start = None
    return out


def _window_from_ink(o: Opening, ink: np.ndarray, wall_cut: np.ndarray, t: float,
                     outside: np.ndarray | None = None) -> list:
    """Fenster 1:1 aus der Zeichnung: alle Linien parallel zur Wand (Rahmen, Glas, Fensterbank) mit ihrer
    gezeichneten Länge, dazu die Querstriche (Rahmenseiten, Bank-Enden). Nichts erkennbar -> []."""
    H, W = ink.shape
    horiz = o.orient == "h"
    a0, a1 = (o.x0, o.x1) if horiz else (o.y0, o.y1)
    f0, f1 = (o.y0, o.y1) if horiz else (o.x0, o.x1)
    T = max(2.0, f1 - f0)
    span = a1 - a0
    if span < 3:
        return []
    m_a, m_f = 0.3 * T, 0.75 * T
    A0, A1 = int(max(0, math.floor(a0 - m_a))), int(math.ceil(a1 + m_a))
    F0, F1 = int(max(0, math.floor(f0 - m_f))), int(math.ceil(f1 + m_f))
    if horiz:
        A1, F1 = min(W, A1), min(H, F1)
        sub = ink[F0:F1, A0:A1].astype(bool) & ~wall_cut[F0:F1, A0:A1]      # [f, a]
    else:
        A1, F1 = min(H, A1), min(W, F1)
        sub = (ink[A0:A1, F0:F1].astype(bool) & ~wall_cut[A0:A1, F0:F1]).T  # [f, a]
    if sub.size == 0:
        return []

    def P(a, f):
        return (a, f) if horiz else (f, a)

    out_side = None
    if outside is not None:
        def is_out(f):
            x, y = P((a0 + a1) / 2, f)
            xi, yi = int(round(x)), int(round(y))
            return 0 <= xi < W and 0 <= yi < H and bool(outside[yi, xi])
        if is_out(f0 - 0.5 * T - 2):
            out_side = -1
        elif is_out(f1 + 0.5 * T + 2):
            out_side = 1
        else:
            out_side = 0          # Innenfenster: weder Bank noch Möbel übernehmen
    # Linien parallel zur Wand: Zeilen, die im mittleren Teil der Öffnung durchgehend Tinte haben
    s0, s1 = int(a0 + 0.15 * span) - A0, int(a1 - 0.15 * span) - A0
    if s1 - s0 < 3:
        return []
    prof = sub[:, s0:s1].mean(axis=1)
    rows = _runs_1d(prof, 0.6)
    ents, pos = [], []
    for r0, r1 in rows:
        if r1 - r0 > 0.35 * T:                       # breite Fläche, keine Linie
            continue
        fc = F0 + (r0 + r1 - 1) / 2.0 + 0.5
        band = sub[r0:r1].any(axis=0)
        # zusammenhängender Strich durch die Mitte -> gezeichnete Länge
        mid = (s0 + s1) // 2
        lo = mid
        while lo > 0 and band[lo - 1:lo + 1].any():
            lo -= 1
        hi = mid
        while hi < len(band) - 1 and band[hi:hi + 2].any():
            hi += 1
        ea, eb = A0 + lo + 0.5, A0 + hi + 0.5
        # ausserhalb der Wanddicke nur auf der Aussenseite (Fensterbank), nicht im Raum (Möbel)
        if not (f0 - 1 <= fc <= f1 + 1) and out_side is not None and (fc - (f0 + f1) / 2) * out_side < 0:
            continue
        # in der Wanddicke enden Rahmen/Glas an den Leibungen
        if f0 - 1 <= fc <= f1 + 1:
            ea, eb = max(ea, a0), min(eb, a1)
        if eb - ea < 0.5 * span:
            continue
        pos.append((fc, ea, eb, r0, r1))
        if r1 - r0 >= max(6, 0.15 * T):
            # Rahmen mit Glas in grober Auflösung zu einem Band verschmolzen: beide Rahmenkanten zeichnen
            for fe in (F0 + r0 + 1.5, F0 + r1 - 1.5):
                ents.append(Line(P(ea, fe), P(eb, fe), "WINDOWS"))
        else:
            ents.append(Line(P(ea, fc), P(eb, fc), "WINDOWS"))
    inside = [p for p in pos if f0 - 1 <= p[0] <= f1 + 1]
    if not inside:
        return []
    # Querstriche zwischen benachbarten Linien (bzw. Linie und Wandflucht) an beiden Enden
    levels = sorted({p[0] for p in pos} | {f0, f1})
    for fa, fb in zip(levels[:-1], levels[1:]):
        if fb - fa < 1.5:
            continue
        ra, rb = int(round(fa - F0)) + 1, int(round(fb - F0)) - 1
        if rb - ra < 1:
            continue
        seg = sub[ra:rb]
        cols = seg.mean(axis=0)
        for end, sign in ((a0, 1), (a1, -1)):
            c0 = int(round(end - A0 - 0.1 * T)) if sign > 0 else int(round(end - A0 - 0.12 * span))
            c1 = int(round(end - A0 + 0.12 * span)) if sign > 0 else int(round(end - A0 + 0.1 * T))
            c0, c1 = max(0, c0), min(len(cols), c1)
            best = None
            for c in range(c0, c1):
                if cols[c] >= 0.8 and (best is None or abs(c - (end - A0)) < abs(best - (end - A0))):
                    best = c
            if best is not None and abs(best - (end - A0)) > max(2.0, 0.08 * T):   # an der Leibung zeichnet die Wand
                ac = A0 + best + 0.5
                ents.append(Line(P(ac, fa), P(ac, fb), "WINDOWS"))
    return ents


def _window_symbol(o: Opening, ink: np.ndarray, outside: np.ndarray, t: float) -> list:
    H, W = ink.shape
    horiz = o.orient == "h"
    a0, a1 = (o.x0, o.x1) if horiz else (o.y0, o.y1)
    f0, f1 = (o.y0, o.y1) if horiz else (o.x0, o.x1)
    T = max(1.0, f1 - f0)
    span = a1 - a0

    def P(a, f):
        return (a, f) if horiz else (f, a)

    # Querprofil über die Wandstärke (+ etwas aussen für die Fensterbank)
    m = 0.7 * T
    q0, q1 = int(math.floor(f0 - m)), int(math.ceil(f1 + m))
    s0, s1 = int(a0 + 0.12 * span), int(a1 - 0.12 * span)
    lines = []
    if s1 - s0 >= 3:
        if horiz:
            sub = ink[max(0, q0):min(H, q1), s0:s1]
        else:
            sub = ink[s0:s1, max(0, q0):min(W, q1)].T
        prof = sub.mean(axis=1) if sub.size else np.zeros(0)
        lines = [max(0, q0) + p for p in _line_positions(prof)]
    inside = [p for p in lines if f0 - 1.5 <= p <= f1 + 1.5]
    outer = [p for p in lines if p < f0 - 1.5 or p > f1 + 1.5]

    # Aussenseite bestimmen
    def is_out(f):
        x, y = P((a0 + a1) / 2, f)
        xi, yi = int(round(x)), int(round(y))
        return 0 <= xi < W and 0 <= yi < H and outside[yi, xi]
    out_sign = -1 if is_out(f0 - 0.5 * T - 2) else (1 if is_out(f1 + 0.5 * T + 2) else 0)

    ents = []
    for a, b in ((P(a0, f0), P(a0, f1)), (P(a1, f0), P(a1, f1))):   # Leibungen (Fensteranschlag)
        ents.append(Line(a, b, "WINDOWS"))
    inner_lines = [min(max(p, f0), f1) for p in inside]
    if len(inner_lines) >= 1:
        for p in inner_lines:
            ents.append(Line(P(a0, p), P(a1, p), "WINDOWS"))
    else:
        # Standard: Rahmen (ca. 8 cm) in Wandmitte + Glaslinie
        fd = min(0.38 * T, max(2.0, 0.27 * T))
        c = (f0 + f1) / 2
        ents.append(Polyline([P(a0, c - fd / 2), P(a1, c - fd / 2), P(a1, c + fd / 2), P(a0, c + fd / 2)], True, "WINDOWS"))
        ents.append(Line(P(a0, c), P(a1, c), "WINDOWS"))
    # Fensterbank aussen: gezeichnet übernehmen, sonst nichts erfinden
    if out_sign:
        sills = [p for p in outer if (p - (f1 if out_sign > 0 else f0)) * out_sign > 0 and abs(p - (f1 if out_sign > 0 else f0)) < m]
        if sills:
            p = min(sills, key=lambda q: abs(q - (f1 if out_sign > 0 else f0)))
            ov = max(1.5, 0.04 * span)
            face = f1 if out_sign > 0 else f0
            ents.append(Polyline([P(a0 - ov, face), P(a0 - ov, p), P(a1 + ov, p), P(a1 + ov, face)], False, "WINDOWS"))
    return ents


def _door_swing(o: Opening, ink) -> list:
    """Bestimmt Drehpunkt(e) und Aufschlagseite aus dem gezeichneten Türsymbol."""
    H, W = ink.shape
    if o.orient == "h":
        u, v = (1.0, 0.0), (0.0, 1.0)
        a0, a1, f0, f1 = o.x0, o.x1, o.y0, o.y1

        def P(a, f):
            return (a, f)
    else:
        u, v = (0.0, 1.0), (1.0, 0.0)
        a0, a1, f0, f1 = o.y0, o.y1, o.x0, o.x1

        def P(a, f):
            return (f, a)
    Wd = a1 - a0

    def val(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 1.0 if (0 <= x < W and 0 <= y < H and ink[y, x]) else 0.0

    def leaf_score(hinge_a, side, r, jdir):
        face = f0 if side < 0 else f1
        h = P(hinge_a, face)
        d = (v[0] * side, v[1] * side)
        j = (u[0] * jdir, u[1] * jdir)
        phis = np.radians(np.linspace(12, 78, 24))
        best = (-1.0, None)
        # Türblatt ist oft etwas schmaler gezeichnet als die Öffnung (z.B. 80 cm Blatt in 90 cm Öffnung)
        for k in np.arange(0.78, 1.071, 0.03):
            rr = r * k
            arc = np.mean([val((h[0] + rr * (math.cos(p) * d[0] + math.sin(p) * j[0]),
                                h[1] + rr * (math.cos(p) * d[1] + math.sin(p) * j[1]))) for p in phis])
            leaf = np.mean([val((h[0] + d[0] * rr * s_, h[1] + d[1] * rr * s_)) for s_ in np.linspace(0.2, 0.9, 12)])
            sc = 0.7 * arc + 0.3 * leaf if arc >= 0.5 else 0.0   # der Bogen muss klar gezeichnet sein
            if sc > best[0]:
                best = (sc, (h[0], h[1], d[0], d[1], rr, j[0], j[1]))
        return best

    cands = []
    for side in (-1, 1):
        for end, jdir in ((a0, 1), (a1, -1)):
            sc, sw = leaf_score(end, side, Wd, jdir)
            cands.append((sc, [sw]))
        s1, w1 = leaf_score(a0, side, Wd / 2, 1)
        s2, w2 = leaf_score(a1, side, Wd / 2, -1)
        cands.append(((s1 + s2) / 2 - 0.03, [w1, w2]))   # Doppeltür nur bei klarem Befund
    sc, swing = max(cands, key=lambda c: c[0])
    return swing if sc >= 0.42 else []


def fallback_scale_from_doors(sp: SemanticPlan) -> float | None:
    """Grobe Massstabs-Schätzung (mm/px) aus Türbreiten (Annahme: Rohbaulichte ~0.90 m)."""
    ws = [o.width for o in sp.openings if o.kind == "door" and len(o.swing) == 1]
    if len(ws) < 3:
        return None
    ws = np.array(ws)
    med = float(np.median(ws))
    core = ws[np.abs(ws - med) < 0.2 * med]
    if len(core) < 3 or core.std() / core.mean() > 0.15:
        return None
    return 900.0 / float(np.median(core))


def footprint_mask(sp: SemanticPlan) -> np.ndarray:
    if getattr(sp, "_foot", None) is None:
        sp._foot = footprint(sp.band, int(max(3, 4 * sp.wall_px)))  # type: ignore[attr-defined]
    return sp._foot  # type: ignore[attr-defined]


# =============================================================================== Raumstempel
_AREA_RE = re.compile(r"(?:(?:\bF|\bBF|\bNF|\bHNF|\bNGF|\bGF)\s*[=:]?\s*(\d{1,3}(?:[.,]\d{1,2})?)(?![\d.,]))|(?:(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:m\s*[2²]|qm))", re.I)
_LETTERS = re.compile(r"[A-Za-zÄÖÜäöüéèàç]{2,}")


def _is_name(q) -> bool:
    """Raumname: echtes Wort (≥ 3 Buchstaben, überwiegend Buchstaben, sicher gelesen)."""
    t = q.text.strip()
    letters = sum(c.isalpha() for c in t)
    return bool(re.search(r"[A-Za-zÄÖÜäöüéèàç]{3,}", t)) and letters >= 0.6 * len(t.replace(" ", "")) and q.conf >= 55


@dataclass
class Room:
    label: int
    area_px: float
    center: tuple
    mask_bbox: tuple


def rooms(sp: SemanticPlan) -> tuple[np.ndarray, list[Room]]:
    """Räume = Freiflächen innerhalb des Gebäudes, begrenzt durch Wände und Öffnungen."""
    t = sp.wall_px
    foot = footprint_mask(sp)
    free = cv2.bitwise_and(foot, cv2.bitwise_not(cv2.dilate(sp.band, np.ones((3, 3), np.uint8))))
    n, lab, st, cen = cv2.connectedComponentsWithStats(free, connectivity=4)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < (4 * t) ** 2:
            continue
        out.append(Room(i, float(st[i, cv2.CC_STAT_AREA]), (float(cen[i][0]), float(cen[i][1])), tuple(st[i, :4])))
    return lab, out


_NUM_PREFIX = re.compile(r"^[Oo]?(\d{1,3}(?:\.\d{1,2})?)\b\s*(.*)$")
# Umlaute, die die Texterkennung häufig verliert (häufige Raumnamen)
_UMLAUT = {"kuche": "Küche", "kueche": "Küche", "buro": "Büro", "bueroe": "Büro", "gaste": "Gäste",
           "gastezimmer": "Gästezimmer", "gaste-wc": "Gäste-WC", "geratraum": "Geräteraum", "gerate": "Geräte",
           "schlafzimmer": "Schlafzimmer", "wohnkuche": "Wohnküche", "essküche": "Essküche", "esskuche": "Essküche",
           "kuhlraum": "Kühlraum", "vorratsraum": "Vorratsraum", "waschkuche": "Waschküche", "buhne": "Bühne",
           "dachboden": "Dachboden", "trockenraum": "Trockenraum", "kinderzimmer": "Kinderzimmer"}
_BARE_AREA = re.compile(r"^(\d{1,3}[.,]\d{1,2})$")


def _clean_stamp_line(text: str) -> str:
    t = text.strip().replace("m2", "m²").replace("qm", "m²")
    t = re.sub(r"\bO(\d)", r"0\1", t)                    # "O02" -> "02" (OCR verwechselt O/0)
    t = re.sub(r"^0(\d{2})\b", r"\1", t)                    # "002" -> "02"
    words = [(_UMLAUT.get(w.lower(), w) if w.lower() in _UMLAUT else w) for w in t.split(" ")]
    return " ".join(words)


def _stamp_frame(binary: np.ndarray, box, th: float):
    """Rahmen um einen Raumstempel suchen (gezeichnetes Rechteck nahe um die Texte)."""
    H, W = binary.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    M = int(max(6, 2.2 * th))
    ink = binary > 0
    found = {}
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    for side in ("top", "bottom", "left", "right"):
        for d in range(1, M):
            if side == "top":
                yy = y0 - d
                ok = 0 <= yy < H and ink[yy, max(0, x0):min(W, x1)].mean() >= 0.85
            elif side == "bottom":
                yy = y1 + d
                ok = 0 <= yy < H and ink[yy, max(0, x0):min(W, x1)].mean() >= 0.85
            elif side == "left":
                xx = x0 - d
                ok = 0 <= xx < W and ink[max(0, y0):min(H, y1), xx].mean() >= 0.85
            else:
                xx = x1 + d
                ok = 0 <= xx < W and ink[max(0, y0):min(H, y1), xx].mean() >= 0.85
            if ok:
                found[side] = d
                break
    if len(found) < 4:
        return None
    # Strichmitte: über die Linienbreite weiterlaufen
    fx0, fy0, fx1, fy1 = x0 - found["left"], y0 - found["top"], x1 + found["right"], y1 + found["bottom"]
    return (fx0, fy0, fx1, fy1)


def room_stamps(sp: SemanticPlan, texts: list, mm_per_px: float | None, binary: np.ndarray | None = None):
    """Raumstempel aus der Texterkennung: alle lesbaren Zeilen (Nummer, Name, Fläche, Bodenbelag …).

    Die Zeilen werden in der Reihenfolge des Originals übernommen, ein gezeichneter Stempelrahmen
    ebenfalls. Passt eine Flächenangabe nicht zur gemessenen Raumfläche (±8 %), bleibt sie stehen und es
    wird ein Hinweis ausgegeben. Gibt (Entities, verwendete Texte, Hinweise, Massstab aus Raumflächen)
    zurück.
    """
    lab, rms = rooms(sp)
    H, W = lab.shape
    by_room: dict[int, list] = {}
    valid = {rm.label for rm in rms}
    # Texte in kleinen/offenen Bereichen des Gebäudes (z.B. Räume mit nicht geschlossenen Wänden)
    # bilden ebenfalls Stempel, nur ohne Flächenprüfung
    foot = footprint_mask(sp)
    extra: dict[int, Room] = {}
    for tx in texts:
        cx, cy = tx.center
        xi, yi = int(cx), int(cy)
        if 0 <= xi < W and 0 <= yi < H and foot[yi, xi] and int(lab[yi, xi]) not in valid:
            key = -(1 + len(extra)) if int(lab[yi, xi]) == 0 else int(lab[yi, xi])
            if key not in extra:
                extra[key] = Room(key, 0.0, (cx, cy), (xi, yi, 1, 1))
            by_room.setdefault(key, []).append(tx)
    rms = list(rms) + list(extra.values())
    for tx in texts:
        if tx.rotation != 0:
            continue
        cx, cy = tx.center
        xi, yi = int(cx), int(cy)
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        r = int(lab[yi, xi])
        if r in valid:
            by_room.setdefault(r, []).append(tx)
    # (Texte in offenen Bereichen sind oben schon zugeordnet)

    def area_of(q):
        m_ = _AREA_RE.search(q.text.replace(" ", "")) or _AREA_RE.search(q.text)
        if m_:
            return float((m_.group(1) or m_.group(2)).replace(",", "."))
        m_ = _BARE_AREA.match(q.text.strip())
        return float(m_.group(1).replace(",", ".")) if m_ else None

    # Massstab aus Raumflächen (nur wenn mehrere Räume übereinstimmen)
    scale_from_rooms = None
    ratios = []
    for rm in rms:
        for tx in by_room.get(rm.label, []):
            if _AREA_RE.search(tx.text.replace(" ", "")) or _AREA_RE.search(tx.text):
                a = area_of(tx)
                if a and 1.0 <= a <= 500 and rm.area_px > 0:
                    ratios.append(math.sqrt(a * 1e6 / rm.area_px))
    if len(ratios) >= 2:
        r_ = np.array(ratios)
        med = float(np.median(r_))
        core = r_[np.abs(r_ - med) < 0.04 * med]
        if len(core) >= 2 and len(core) >= 0.6 * len(r_):
            scale_from_rooms = float(np.median(core))
    mmpp = mm_per_px or scale_from_rooms
    ents, used, notes = [], [], []
    for rm in rms:
        txs = sorted(by_room.get(rm.label, []), key=lambda q: (q.box[1] + q.box[3]) / 2)
        if not txs:
            continue
        # Stempel = Gruppe dicht übereinander stehender Zeilen um die grösste/sicherste Zeile
        hs = [q.height for q in txs if q.height > 0]
        th = float(np.median(hs)) if hs else 3 * sp.wall_px
        lines = []
        for q in txs:
            t = _clean_stamp_line(q.text)
            a = area_of(q)
            m_num = _NUM_PREFIX.match(t)
            if a is not None and (_AREA_RE.search(q.text.replace(" ", "")) or (_BARE_AREA.match(q.text.strip()) and q.conf >= 80)):
                lines.append((q, "area", a))
            elif _is_name(q):
                lines.append((q, "text", t))
            elif m_num and q.conf >= 75 and not m_num.group(2):
                lines.append((q, "nr", m_num.group(1)))
            elif m_num and q.conf >= 75 and re.search(r"[A-Za-zÄÖÜäöü]", m_num.group(2) or ""):
                lines.append((q, "nr", m_num.group(1)))          # Nummer sicher, Name unvollständig gelesen
        if not lines:
            continue
        # nur Zeilen, die als Block zusammenstehen (Abstand ≤ 2.5 Zeilenhöhen)
        blocks, cur = [], [lines[0]]
        for ln in lines[1:]:
            if ln[0].box[1] - cur[-1][0].box[3] <= 2.5 * th and abs(ln[0].center[0] - cur[-1][0].center[0]) < 6 * th:
                cur.append(ln)
            else:
                blocks.append(cur)
                cur = [ln]
        blocks.append(cur)
        block = max(blocks, key=lambda b: (len(b), sum(q.conf for q, _, _ in b)))
        if len(block) == 1 and block[0][1] == "nr":
            continue
        out_lines = []
        for q, kind, val in block:
            used.append(q)
            if kind == "area":
                txt = f"{val:.1f} m²" if abs(val * 10 - round(val * 10)) < 1e-6 else f"{val:.2f} m²"
                if mmpp and rm.area_px > 0:
                    meas = rm.area_px * mmpp * mmpp / 1e6
                    if abs(val - meas) > 0.08 * meas:
                        # typischer Lesefehler: eine Ziffer zu viel/falsch ("235.5" statt "25.5")
                        raw = f"{val:.2f}".rstrip("0").rstrip(".") if "." in f"{val}" else f"{val}"
                        alts = []
                        for k_ in range(len(raw)):
                            if raw[k_].isdigit():
                                try:
                                    alts.append(float(raw[:k_] + raw[k_ + 1:]))
                                except ValueError:
                                    pass
                        good = sorted((abs(a_ - meas), a_) for a_ in alts if a_ > 0 and abs(a_ - meas) <= 0.05 * meas)
                        if good:
                            val = good[0][1]
                            txt = f"{val:.1f} m²" if abs(val * 10 - round(val * 10)) < 1e-6 else f"{val:.2f} m²"
                    if abs(val - meas) > 0.08 * meas:
                        name = " ".join(str(v) for _, k_, v in block if k_ != "area") or "?"
                        notes.append(f"Raumstempel «{name}»: Fläche {val:.1f} m² weicht von der gemessenen Fläche "
                                     f"{meas:.1f} m² ab – bitte prüfen.")
                out_lines.append((txt, "area"))
            else:
                out_lines.append((str(val), kind))
        # Nummer und Name auf eine Zeile ("02 Küche"), wie üblich
        merged = []
        for txt, kind in out_lines:
            if merged and merged[-1][1] == "nr" and kind == "text":
                merged[-1] = (merged[-1][0] + " " + txt, "title")
            else:
                merged.append((txt, kind))
        qs = [q for q, _, _ in block]
        x0 = min(q.box[0] for q in qs)
        x1 = max(q.box[2] for q in qs)
        y0 = min(q.box[1] for q in qs)
        y1 = max(q.box[3] for q in qs)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sizes = [(th if i == 0 else 0.8 * th) for i in range(len(merged))]
        total = sum(h_ * 1.5 for h_ in sizes)
        y = cy - total / 2
        for (text, kind), h_ in zip(merged, sizes):
            w_ = 0.62 * h_ * len(text)
            ents.append(Text(text, (cx - w_ / 2, y, cx + w_ / 2, y + h_), 0, 99.0, "ROOMS", cap=h_, group=rm.label))
            y += 1.5 * h_
        if binary is not None:
            fr = _stamp_frame(binary, (x0, y0, x1, y1), th)
            if fr is not None:
                fx0, fy0, fx1, fy1 = fr
                ents.append(Polyline([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)], True, "ROOMS"))
    return ents, used, notes, scale_from_rooms

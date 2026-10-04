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

from .geometry import Arc, Hatch, Line, Polyline
from .semantic import DOOR, WALL, WINDOW
from .vectorize import _orthogonalize
from .walls import decompose, footprint, median_thickness, merge_rects, raster_rects


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
def reconstruct(labels: np.ndarray, binary: np.ndarray) -> SemanticPlan | None:
    H, W = labels.shape
    wall = ((labels == WALL) * 255).astype(np.uint8)
    t = median_thickness(wall)
    if t <= 0 or np.count_nonzero(wall) < 50:
        return None
    t = max(2.0, t)
    ink = (binary > 0).astype(np.uint8)

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
        emax = float(max(st[imax, cv2.CC_STAT_WIDTH], st[imax, cv2.CC_STAT_HEIGHT]))
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
            o = _opening_rect(kind, x, y, w, h, wall, t)
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

    # ------------------------------------------------------------ 3) Wandrechtecke
    band = wall.copy()
    for o in openings:
        cv2.rectangle(band, (int(o.x0), int(o.y0)), (int(math.ceil(o.x1)) - 1, int(math.ceil(o.y1)) - 1), 255, -1)
    gk = max(3, int(round(0.5 * t)) | 1)
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gk, gk)))
    rects, leftover = decompose(band, max(3 * t, 12))
    rects = merge_rects(rects, max(2.0, 0.5 * t))
    rects = [_snap_rect(r, ink, t) for r in rects]
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
    sp.entities = _wall_entities(walls, t)
    return sp


# =============================================================================== Öffnungen
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
    if cc:
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
    m = int(max(2, t))
    x0, y0, x1, y1 = int(o.x0), int(o.y0), int(math.ceil(o.x1)), int(math.ceil(o.y1))
    if o.orient == "v":                 # Wand läuft senkrecht: Spalten der Wand oben/unten übernehmen
        cols = np.arange(max(0, x0 - m), min(W, x1 + m))
        above = wall[max(0, y0 - 3):max(0, y0 - 1), cols].any(axis=0) if y0 > 2 else np.zeros(len(cols), bool)
        below = wall[min(H, y1 + 1):min(H, y1 + 3), cols].any(axis=0) if y1 < H - 2 else np.zeros(len(cols), bool)
        sel = _narrow(cols, above, below)
        if sel.size:
            wall[y0:y1, sel] = 255
    else:
        rows = np.arange(max(0, y0 - m), min(H, y1 + m))
        left = wall[rows, max(0, x0 - 3):max(0, x0 - 1)].any(axis=1) if x0 > 2 else np.zeros(len(rows), bool)
        right = wall[rows, min(W, x1 + 1):min(W, x1 + 3)].any(axis=1) if x1 < W - 2 else np.zeros(len(rows), bool)
        sel = _narrow(rows, left, right)
        if sel.size:
            wall[sel, x0:x1] = 255


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


def _ring(c, eps):
    pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float) + 0.5
    return _orthogonalize(pts, tol_deg=4.0)


def opening_entities(sp: SemanticPlan, binary: np.ndarray) -> list:
    """Genormte Symbole: Fenster (Leibungs- und Glaslinien), Türen (Blatt + Anschlagbogen)."""
    t = sp.wall_px
    ents: list = []
    ink = (binary > 0).astype(np.uint8)
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(sp.walls, np.ones((3, 3), np.uint8))) // 255)
    rad = max(1, int(round(0.12 * t)))
    tol_ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2))
    for o in sp.openings:
        if o.kind == "window":
            ents.extend(_window_lines(o))
        else:
            o.swing = _door_swing(o, tol_ink)
            for hx, hy, ldx, ldy, r, jdx, jdy in o.swing:
                tip = (hx + ldx * r, hy + ldy * r)
                ents.append(Line((hx, hy), tip, "DOORS"))
                a_leaf = math.degrees(math.atan2(ldy, ldx)) % 360
                a_j = math.degrees(math.atan2(jdy, jdx)) % 360
                if (a_j - a_leaf) % 360 <= 180:
                    ents.append(Arc((hx, hy), r, a_leaf, a_leaf + ((a_j - a_leaf) % 360), "DOORS"))
                else:
                    ents.append(Arc((hx, hy), r, a_j, a_j + ((a_leaf - a_j) % 360), "DOORS"))
    return ents


def _window_lines(o: Opening) -> list:
    out = []
    if o.orient == "h":
        cm = (o.y0 + o.y1) / 2
        g = max(1.0, 0.08 * (o.y1 - o.y0))
        for y in (o.y0, o.y1, cm - g, cm + g):
            out.append(Line((o.x0, y), (o.x1, y), "WINDOWS"))
    else:
        cm = (o.x0 + o.x1) / 2
        g = max(1.0, 0.08 * (o.x1 - o.x0))
        for x in (o.x0, o.x1, cm - g, cm + g):
            out.append(Line((x, o.y0), (x, o.y1), "WINDOWS"))
    return out


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
            sc = 0.7 * arc + 0.3 * leaf
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
    return footprint(sp.band, int(max(3, 4 * sp.wall_px)))

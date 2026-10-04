"""Synthetische Trainingspläne aus echten Grundriss-Geometrien (ResPlan, CC BY 4.0).

Jeder ResPlan-Grundriss (Wände, Türen, Fenster, Räume in Vektorform) wird in einem
zufällig gewählten Zeichenstil gerendert – so, wie Bestandespläne in der Praxis aussehen:

* Wände: schwarz gefüllt, grau angelegt, nur Umriss, Mauerwerk-Schraffur, Kreuzschraffur,
  Handskizze (eine dicke Linie)
* Türen: Türblatt + Anschlagbogen (voll/gestrichelt), Doppeltüren, Schiebetüren
* Fenster: Rahmen, Glaslinien, Fensterbank
* Störelemente, die NICHT erkannt werden sollen: Möblierung, Sanitärapparate, Küchen,
  Massketten (Schweizer Schreibweise), Raumbeschriftungen, Treppen, Bodenbeläge,
  Balkone, Achsraster, Nordpfeil, Planrahmen, Plankopf
* Alterung/Erfassung: Papierton, Bleistift, Unschärfe, Rauschen, JPEG, Flecken, Ausbleichen,
  Schräglage, Perspektive, Handzittern

Ausgabe: Graubild + Klassenbild (0 Hintergrund, 1 Wand, 2 Fenster, 3 Tür).
"""
from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

BG, WALL, WINDOW, DOOR = 0, 1, 2, 3
SH = 4           # Subpixel-Bits für cv2-Zeichenfunktionen
F = 1 << SH

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
         cv2.FONT_HERSHEY_COMPLEX_SMALL]
ROOM_NAMES = {
    "bedroom": ["Zimmer", "Schlafen", "Zimmer 1", "Zimmer 2", "Kind", "Eltern", "ZIMMER", "Chambre"],
    "living": ["Wohnen", "Wohnen/Essen", "WOHNEN", "Essen", "Stube", "Wohnzimmer", "Séjour"],
    "kitchen": ["Küche", "KÜCHE", "Kochen", "Kü"],
    "bathroom": ["Bad", "Bad/WC", "Dusche/WC", "WC", "BAD", "Du/WC"],
    "storage": ["Reduit", "Abst.", "Réduit", "Keller", "Abstellraum"],
    "inner": ["Gang", "Korridor", "Entrée", "Vorplatz", "Diele", "Flur"],
    "balcony": ["Balkon", "Loggia", "Sitzplatz", "Terrasse"],
}
TITLES = ["GRUNDRISS EG", "GRUNDRISS 1. OG", "Grundriss Erdgeschoss", "OBERGESCHOSS", "Grundriss UG",
          "2. OBERGESCHOSS", "DACHGESCHOSS", "Erdgeschoss", "BESTAND EG", "Grundriss 1:100", "Wohnung 3.5 Zi."]


# =============================================================================== Geometrie-Hilfen
def polys(g):
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out.extend(polys(x))
        return out
    return []


@dataclass
class Frame:
    """Plan-Koordinaten (ResPlan-Einheiten) -> Pixel."""
    M: np.ndarray        # 2x2
    t: np.ndarray        # 2
    mu: float            # Meter pro Einheit

    def px(self, pts) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return p @ self.M.T + self.t

    def fx(self, pts) -> np.ndarray:
        return np.round(self.px(pts) * F).astype(np.int32)

    @property
    def s(self) -> float:          # Pixel pro Einheit
        return float(math.sqrt(abs(np.linalg.det(self.M))))

    def m(self, meters: float) -> float:   # Meter -> Einheiten
        return meters / self.mu

    def angle(self, v=(1.0, 0.0)) -> float:
        d = self.M @ np.asarray(v, float)
        return math.degrees(math.atan2(d[1], d[0]))


class Canvas:
    def __init__(self, h: int, w: int, fr: Frame, rng: random.Random):
        self.ink = np.zeros((h, w), np.uint8)
        self.h, self.w, self.fr, self.rng = h, w, fr, rng

    # --- Grundelemente (Plan-Koordinaten)
    def line(self, a, b, lw=1, v=255):
        p = self.fr.fx([a, b])
        cv2.line(self.ink, tuple(p[0]), tuple(p[1]), int(v), max(1, int(round(lw))), cv2.LINE_AA, SH)

    def poly(self, pts, closed=True, lw=1, v=255):
        cv2.polylines(self.ink, [self.fr.fx(pts)], closed, int(v), max(1, int(round(lw))), cv2.LINE_AA, SH)

    def fill(self, pts, v=255):
        cv2.fillPoly(self.ink, [self.fr.fx(pts)], int(v), cv2.LINE_AA, SH)

    def rect(self, x0, y0, x1, y1, lw=1, v=255, filled=False):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        (self.fill(pts, v) if filled else self.poly(pts, True, lw, v))

    def circle(self, c, r, lw=1, v=255, filled=False):
        p = self.fr.fx([c])[0]
        cv2.circle(self.ink, tuple(p), max(1, int(round(r * self.fr.s * F))), int(v),
                   -1 if filled else max(1, int(round(lw))), cv2.LINE_AA, SH)

    def ellipse(self, c, ax, ay, lw=1, v=255):
        n = 40
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        self.poly(np.c_[c[0] + ax * np.cos(t), c[1] + ay * np.sin(t)], True, lw, v)

    def arc(self, c, r, a0, a1, lw=1, v=255, dashed=False):
        """Bogen in Plan-Koordinaten von Winkel a0 nach a1 (Grad, kürzerer Weg)."""
        d = (a1 - a0 + 180) % 360 - 180
        n = max(8, int(abs(d) / 4))
        t = np.radians(np.linspace(a0, a0 + d, n))
        pts = np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]
        if dashed:
            for i in range(0, n - 1, 2):
                self.line(pts[i], pts[i + 1], lw, v)
        else:
            self.poly(pts, False, lw, v)

    def dashed(self, a, b, dash, gap, lw=1, v=255, dot=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.hypot(*(b - a)))
        if L < 1e-9:
            return
        u = (b - a) / L
        t = 0.0
        while t < L:
            e = min(L, t + dash)
            self.line(a + u * t, a + u * e, lw, v)
            if dot and e + gap / 2 < L:
                q = a + u * (e + gap / 2)
                self.line(q, q + u * 0.01, lw, v)
            t = e + gap

    def text(self, s, c, height_units, angle_extra=0.0, v=255, font=None, thick=None):
        """Text zentriert an c, Schrifthöhe in Plan-Einheiten, entlang der Plan-x-Achse."""
        hpx = height_units * self.fr.s
        if hpx < 4:
            return
        font = font if font is not None else self.rng.choice(FONTS)
        base_h = cv2.getTextSize("H", font, 1.0, 1)[0][1]
        sc = hpx / max(1, base_h)
        th = thick or max(1, int(round(hpx / self.rng.uniform(7, 14))))
        (tw, tht), bl = cv2.getTextSize(s, font, sc, th)
        if tw <= 0:
            return
        pad = th + 2
        patch = np.zeros((tht + bl + 2 * pad, tw + 2 * pad), np.uint8)
        cv2.putText(patch, s, (pad, pad + tht), font, sc, int(v), th, cv2.LINE_AA)
        ang = self.fr.angle() + angle_extra
        ph, pw = patch.shape
        R = cv2.getRotationMatrix2D((pw / 2, ph / 2), -ang, 1.0)
        cos, sin = abs(R[0, 0]), abs(R[0, 1])
        nw, nh = int(ph * sin + pw * cos) + 2, int(ph * cos + pw * sin) + 2
        R[0, 2] += nw / 2 - pw / 2
        R[1, 2] += nh / 2 - ph / 2
        rot = cv2.warpAffine(patch, R, (nw, nh))
        cx, cy = self.fr.px([c])[0]
        x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(self.w, x0 + nw), min(self.h, y0 + nh)
        if xb <= xa or yb <= ya:
            return
        sub = rot[ya - y0:yb - y0, xa - x0:xb - x0]
        np.maximum(self.ink[ya:yb, xa:xb], sub, out=self.ink[ya:yb, xa:xb])


# =============================================================================== Hauptfunktion
def load_resplan(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _valid(g):
    try:
        return make_valid(g) if g is not None and not g.is_valid else g
    except Exception:  # noqa: BLE001
        return g


def render(plan: dict, seed: int, max_side: int = 1600) -> dict | None:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    walls = _valid(plan["wall"])
    doors = [p for p in polys(_valid(plan.get("door"))) + polys(_valid(plan.get("front_door"))) if p.area > 0]
    wins = [p for p in polys(_valid(plan.get("window"))) if p.area > 0]
    if walls is None or walls.is_empty:
        return None

    # ------------------------------------------------------------ Massstab & Lage
    mu = rng.uniform(0.042, 0.058)                       # Meter pro Einheit
    wd = float(plan.get("wall_depth") or 4.0)
    wall_m = wd * mu
    wall_px = math.exp(rng.uniform(math.log(3.6), math.log(17)))
    ppm = wall_px / wall_m                               # Pixel pro Meter
    minx, miny, maxx, maxy = walls.bounds
    with_dims = rng.random() < 0.6
    margin_m = (rng.uniform(1.8, 3.2) if with_dims else rng.uniform(0.4, 1.5))
    sheet = rng.random() < 0.45
    ext_m = (maxx - minx) * mu + 2 * margin_m
    ext_n = (maxy - miny) * mu + 2 * margin_m
    long_m = max(ext_m, ext_n) * (1.25 if sheet else 1.0)
    if long_m * ppm > max_side:
        ppm = max_side / long_m
        if ppm * wall_m < 3.0:
            ppm = 3.0 / wall_m
    rot_mode = rng.random()
    theta = 0.0
    if rot_mode > 0.88:
        theta = rng.uniform(-45, 45)
    elif rot_mode > 0.72:
        theta = rng.uniform(-4, 4)
    theta += rng.choice([0, 90, 180, 270])
    mirror = rng.random() < 0.5
    s = mu * ppm
    c, sn = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    M = s * np.array([[c, -sn], [sn, c]]) @ np.diag([-1.0 if mirror else 1.0, 1.0])
    corners = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    pad_u = margin_m / mu
    cc = np.array([[minx - pad_u, miny - pad_u], [maxx + pad_u, miny - pad_u],
                   [maxx + pad_u, maxy + pad_u], [minx - pad_u, maxy + pad_u]]) @ M.T
    lo, hi = cc.min(0), cc.max(0)
    W, H = int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1]))
    if sheet:
        ex = int(0.12 * max(W, H))
        W, H = W + 2 * ex, H + 2 * ex + int(0.08 * max(W, H))
        t = -lo + ex
    else:
        t = -lo
    W, H = max(W, 64), max(H, 64)
    if max(W, H) > 2600:
        return None
    fr = Frame(M, t, mu)

    # ------------------------------------------------------------ Klassenbild
    lab = np.zeros((H, W), np.uint8)
    for p in polys(walls):
        _fill_poly(lab, fr, p, WALL)
    for p in wins:
        _fill_poly(lab, fr, p, WINDOW)
    for p in doors:
        _fill_poly(lab, fr, p, DOOR)
    band = lab > 0
    # Aussenwände nach aussen verstärken (ResPlan hat überall gleiche Wandstärken)
    ext_px = 0
    if rng.random() < 0.8:
        ext_px = int(round(rng.uniform(0.05, 0.25) * ppm))
    k = max(3, int(wall_px * 2.5) | 1)
    closed = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    outside = _outside(closed)
    if ext_px >= 1:
        grow = cv2.dilate(band.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ext_px + 1,) * 2))
        strip = (grow > 0) & outside & ~band
        if strip.any():
            src = np.where(band, 0, 255).astype(np.uint8)
            _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            cls_of_zero = lab[src == 0]
            idx = labels[strip] - 1
            ok = (idx >= 0) & (idx < cls_of_zero.size)
            vals = np.zeros(idx.shape, np.uint8)
            vals[ok] = cls_of_zero[idx[ok]]
            lab[strip] = vals
        band = lab > 0
        outside = _outside(cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))))

    cv_ = Canvas(H, W, fr, rng)
    lw_main = max(1, int(round(rng.uniform(0.8, 2.6) * max(1.0, wall_px / 9))))
    lw_thin = max(1, int(round(lw_main * rng.uniform(0.4, 0.8))))
    sketch = rng.random() < 0.10

    # ------------------------------------------------------------ Wände
    if sketch:
        lab = _render_sketch_walls(cv_, lab, rng, wall_px)
    else:
        _render_walls(cv_, lab, outside, rng, wall_px, lw_main)

    # ------------------------------------------------------------ Öffnungen
    for comp, kind in _openings(lab):
        if kind == DOOR:
            _render_door(cv_, comp, lab, rng, ppm, lw_thin, lw_main, sketch)
        else:
            _render_window(cv_, comp, lab, outside, rng, ppm, lw_thin, sketch)

    # ------------------------------------------------------------ Störelemente
    rooms = {k_: polys(_valid(plan.get(k_))) for k_ in ROOM_NAMES}
    rooms["stair"] = polys(_valid(plan.get("stair")))
    if rng.random() < 0.8:
        _render_furniture(cv_, rooms, rng, lw_thin)
    if rng.random() < 0.6:
        _render_floor_patterns(cv_, rooms, rng, lab)
    for st in rooms["stair"]:
        _render_stair(cv_, st, rng, lw_thin)
    if rng.random() < 0.08:
        _random_stairs(cv_, rooms, rng, lw_thin)
    for b in rooms["balcony"]:
        if rng.random() < 0.7:
            cv_.poly(np.array(b.exterior.coords), True, lw_thin)
            if rng.random() < 0.5:
                bb = b.buffer(-fr.m(0.05))
                for q in polys(bb):
                    cv_.poly(np.array(q.exterior.coords), True, lw_thin)
    if rng.random() < 0.75:
        _render_room_texts(cv_, rooms, rng)
    if with_dims:
        _render_dimensions(cv_, walls, doors, wins, rng, ext_px / ppm / mu, lw_thin)
    if rng.random() < 0.25:
        _render_inner_dims(cv_, rooms, rng, lw_thin)
    if rng.random() < 0.15:
        _render_axes(cv_, walls, rng, lw_thin)
    if rng.random() < 0.4:
        _render_north(cv_, walls, rng, lw_thin)
    if rng.random() < 0.6:
        _render_title(cv_, walls, rng)
    if rng.random() < 0.1:
        _render_scribbles(cv_, rng)
    if sheet:
        _render_sheet(cv_, rng, lw_main)
    if rng.random() < 0.05:
        _render_columns(cv_, rooms, rng)

    # ------------------------------------------------------------ Alterung
    img, lab = _degrade(cv_.ink, lab, rng, nrng)
    return {"image": img, "label": lab, "wall_px": wall_px, "theta": theta % 90 if theta % 90 <= 45 else theta % 90 - 90,
            "ppm": ppm}


# =============================================================================== Klassenbild
def _fill_poly(lab, fr, p: Polygon, v):
    if p.is_empty:
        return
    ext = fr.fx(np.array(p.exterior.coords))
    cv2.fillPoly(lab, [ext], v, cv2.LINE_8, SH)
    for hole in p.interiors:
        cv2.fillPoly(lab, [fr.fx(np.array(hole.coords))], 0, cv2.LINE_8, SH)


def _outside(closed: np.ndarray) -> np.ndarray:
    h, w = closed.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = closed
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    return ff[1:-1, 1:-1] == 128


def _openings(lab):
    out = []
    for kind in (DOOR, WINDOW):
        n, cl, st, _ = cv2.connectedComponentsWithStats((lab == kind).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 6:
                continue
            ys, xs = np.nonzero(cl == i)
            out.append(((xs, ys), kind))
    return out


def _axes(comp, lab):
    """Lage einer Öffnung: Mittelpunkt, Richtung entlang der Wand, quer, Breite, Stärke (px)."""
    xs, ys = comp
    pts = np.c_[xs, ys].astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    a = math.radians(ang)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-u[1], u[0]])
    H, W = lab.shape

    def band_at(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= x < W and 0 <= y < H and lab[y, x] in (WALL, WINDOW, DOOR)

    c = np.array([cx, cy])
    score_u = band_at(c + u * (w / 2 + 2)) + band_at(c - u * (w / 2 + 2))
    score_v = band_at(c + v * (h / 2 + 2)) + band_at(c - v * (h / 2 + 2))
    if score_v > score_u or (score_v == score_u and h > w):
        u, v, w, h = v, u, h, w
    return c, u, v, max(w, 1.0), max(h, 1.0)


# =============================================================================== Wände
def _render_walls(cv_: Canvas, lab, outside, rng, wall_px, lw):
    ink = cv_.ink
    wall = (lab == WALL).astype(np.uint8)
    style = rng.choices(["solid", "gray", "outline", "hatch", "cross", "stipple", "mixed"],
                        [0.24, 0.12, 0.2, 0.18, 0.05, 0.05, 0.16])[0]
    v_line = 255
    contours, _ = cv2.findContours(wall, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    def outline(mask_val=255):
        cv2.drawContours(ink, contours, -1, mask_val, lw, cv2.LINE_AA)

    def hatch(region, spacing, angle, val):
        hh, ww = region.shape
        tmp = np.zeros_like(ink)
        d = max(hh, ww) * 2
        cxy = (ww / 2, hh / 2)
        a = math.radians(angle)
        u = (math.cos(a), math.sin(a))
        n = (-u[1], u[0])
        for off in np.arange(-d / 2, d / 2, spacing):
            p0 = (cxy[0] + n[0] * off - u[0] * d, cxy[1] + n[1] * off - u[1] * d)
            p1 = (cxy[0] + n[0] * off + u[0] * d, cxy[1] + n[1] * off + u[1] * d)
            cv2.line(tmp, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), val, max(1, lw // 2 + rng.choice([0, 0, 1])), cv2.LINE_AA)
        tmp[region == 0] = 0
        np.maximum(ink, tmp, out=ink)

    base_ang = cv_.fr.angle()
    if style == "mixed":
        ext = wall.astype(bool) & (cv2.dilate(outside.astype(np.uint8), np.ones((int(wall_px * 3) | 1,) * 2, np.uint8)) > 0)
        inner = wall.astype(bool) & ~ext
        hatch(ext.astype(np.uint8), rng.uniform(3, 8), base_ang + rng.choice([45, 135]), v_line)
        if rng.random() < 0.5:
            ink[inner] = 255
        else:
            ink[inner] = rng.randint(90, 200)
        outline()
        return
    if style == "solid":
        ink[wall > 0] = 255
        if rng.random() < 0.3:
            outline()
    elif style == "gray":
        ink[wall > 0] = rng.randint(70, 200)
        outline()
    elif style == "outline":
        outline()
    elif style == "hatch":
        hatch(wall, rng.uniform(2.5, max(3.0, min(10.0, wall_px * 0.8))), base_ang + rng.choice([45, 135, 60]), v_line)
        outline()
    elif style == "cross":
        sp = rng.uniform(3, 9)
        hatch(wall, sp, base_ang + 45, v_line)
        hatch(wall, sp, base_ang + 135, v_line)
        outline()
    elif style == "stipple":
        ys, xs = np.nonzero(wall)
        n = int(len(xs) * rng.uniform(0.03, 0.12))
        if n:
            sel = np.random.default_rng(rng.randint(0, 1 << 30)).choice(len(xs), n, replace=False)
            for x, y in zip(xs[sel], ys[sel]):
                cv2.circle(ink, (int(x), int(y)), rng.choice([0, 1, 1]), 255, -1)
        outline()


def _render_sketch_walls(cv_: Canvas, lab, rng, wall_px):
    """Handskizze: Wände als eine (dicke) Linie entlang der Wandachse."""
    from app.pipeline.thinning import thinning
    from app.pipeline.vectorize import _trace
    band = (lab > 0).astype(np.uint8) * 255
    sk = thinning(band) > 0
    lw = max(2, int(round(rng.uniform(1.5, 4.0))))
    newlab = np.zeros_like(lab)
    # Skelett-Klasse übernehmen und auf Strichbreite verbreitern
    r = max(1, lw // 2 + 1)
    for cls in (WALL, WINDOW, DOOR):
        m = (sk & (lab == cls)).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
        newlab[(m > 0) & (newlab == 0)] = cls
    wall_sk = (sk & (lab == WALL)).astype(np.uint8)
    for path, closed in _trace(wall_sk):
        if len(path) < 3:
            continue
        p = path[:: max(1, len(path) // 40)] if len(path) > 80 else path
        p = np.vstack([p, path[-1:]])
        noise = np.cumsum(np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 0.35, (len(p), 2)), axis=0)
        noise -= np.linspace(noise[0], noise[-1], len(p))
        q = p + noise * rng.uniform(0.3, 1.0)
        cv2.polylines(cv_.ink, [np.round(q * F).astype(np.int32)], False, 255, lw, cv2.LINE_AA, SH)
    return newlab


# =============================================================================== Öffnungen
def _render_door(cv_: Canvas, comp, lab, rng, ppm, lw, lw_main, sketch):
    c, u, v, Wd, T = _axes(comp, lab)
    ink = cv_.ink
    width_m = Wd / ppm
    kind = rng.random()
    vline = 255

    def L(a, b, w=lw):
        cv2.line(ink, tuple(np.round(a * F).astype(int)), tuple(np.round(b * F).astype(int)), vline, max(1, int(w)), cv2.LINE_AA, SH)

    def arc(center, r, d0, d1, dashed=False, w=lw):
        a0 = math.degrees(math.atan2(d0[1], d0[0]))
        a1 = math.degrees(math.atan2(d1[1], d1[0]))
        d = (a1 - a0 + 180) % 360 - 180
        n = max(8, int(abs(d) / 3))
        t = np.radians(np.linspace(a0, a0 + d, n))
        pts = np.c_[center[0] + r * np.cos(t), center[1] + r * np.sin(t)]
        if dashed:
            for i in range(0, n - 1, 2):
                L(pts[i], pts[i + 1], w)
        else:
            cv2.polylines(ink, [np.round(pts * F).astype(np.int32)], False, vline, max(1, int(w)), cv2.LINE_AA, SH)

    if rng.random() < 0.15:          # Schwelle / Anschlaglinien
        for sgn in (-1, 1):
            L(c - u * Wd / 2 + sgn * v * T / 2, c + u * Wd / 2 + sgn * v * T / 2)
    if kind < 0.08:                  # nur Öffnung
        return
    if kind < 0.18:                  # Schiebetür
        off = T * 0.15
        L(c - u * Wd / 2 + v * off, c + u * Wd * 0.1 + v * off, lw + 1)
        L(c - u * Wd * 0.1 - v * off, c + u * Wd / 2 - v * off, lw + 1)
        return
    side = rng.choice([-1, 1])
    dashed = rng.random() < 0.25
    no_arc = rng.random() < 0.06
    leaf_w = lw + (1 if rng.random() < 0.4 else 0)
    hinges = [rng.choice([-1, 1])]
    if width_m > 1.25 and rng.random() < 0.7:
        hinges = [-1, 1]
    leaf_len = Wd / len(hinges)
    open_ang = 90 if rng.random() < 0.85 else rng.uniform(30, 75)
    for e in hinges:
        hinge = c + e * u * Wd / 2 + side * v * T / 2
        a = math.radians(open_ang)
        d_leaf = (-e * u) * math.cos(a) + side * v * math.sin(a)
        tip = hinge + d_leaf * leaf_len
        if rng.random() < 0.3:       # Türblatt als schmales Rechteck
            nrm = np.array([-d_leaf[1], d_leaf[0]]) * max(1.5, 0.04 * ppm)
            pts = np.array([hinge, tip, tip + nrm, hinge + nrm])
            cv2.polylines(ink, [np.round(pts * F).astype(np.int32)], True, vline, max(1, lw), cv2.LINE_AA, SH)
        else:
            L(hinge, tip, leaf_w)
        if not no_arc:
            arc(hinge, leaf_len, d_leaf, -e * u, dashed)


def _render_window(cv_: Canvas, comp, lab, outside, rng, ppm, lw, sketch):
    c, u, v, Wd, T = _axes(comp, lab)
    ink = cv_.ink

    def L(a, b, w=lw):
        cv2.line(ink, tuple(np.round(a * F).astype(int)), tuple(np.round(b * F).astype(int)), 255, max(1, int(w)), cv2.LINE_AA, SH)

    a0, a1 = c - u * Wd / 2, c + u * Wd / 2
    st = rng.random()
    if sketch:
        L(a0 + v * T * 0.35, a1 + v * T * 0.35, 1)
        L(a0 - v * T * 0.35, a1 - v * T * 0.35, 1)
        return
    jambs = rng.random() < 0.7
    if jambs:
        for p in (a0, a1):
            L(p - v * T / 2, p + v * T / 2)
    if st < 0.4:          # Leibungslinien + Glas
        L(a0 + v * T / 2, a1 + v * T / 2)
        L(a0 - v * T / 2, a1 - v * T / 2)
        g = rng.choice([1, 2])
        for k in range(g):
            off = (k - (g - 1) / 2) * max(1.5, 0.03 * ppm)
            L(a0 + v * off, a1 + v * off)
    elif st < 0.65:       # Doppellinie Mitte
        off = max(1.0, rng.uniform(0.02, 0.06) * ppm)
        L(a0 + v * off, a1 + v * off)
        L(a0 - v * off, a1 - v * off)
    elif st < 0.8:        # einfache Mittellinie
        L(a0, a1)
    else:                 # Rahmen + Glas + Fensterbank
        ins = max(1.0, 0.05 * ppm)
        q = [a0 + u * 0 + v * (T / 2 - ins), a1 + v * (T / 2 - ins), a1 - v * (T / 2 - ins), a0 - v * (T / 2 - ins)]
        cv2.polylines(ink, [np.round(np.array(q) * F).astype(np.int32)], True, 255, max(1, lw), cv2.LINE_AA, SH)
        L(a0, a1)
    if rng.random() < 0.35:   # Fensterbank aussen
        H, W = lab.shape
        for sgn in (-1, 1):
            p = c + sgn * v * (T / 2 + 3)
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H and outside[y, x]:
                ext = max(1.0, 0.05 * ppm)
                off = max(1.5, 0.04 * ppm)
                L(a0 - u * ext + sgn * v * (T / 2 + off), a1 + u * ext + sgn * v * (T / 2 + off))
                break


# =============================================================================== Möbel etc.
def _room_frame(cv_, room: Polygon):
    minx, miny, maxx, maxy = room.bounds
    return minx, miny, maxx, maxy


def _place(room: Polygon, fr: Frame, rng, w_m, d_m, against_wall=True, tries=12):
    """Rechteck w×d (Meter) im Raum platzieren. Gibt (x0,y0,x1,y1, orient) in Plan-Einheiten."""
    inner = room.buffer(-fr.m(0.03))
    if inner.is_empty:
        return None
    minx, miny, maxx, maxy = room.bounds
    for _ in range(tries):
        rot = rng.random() < 0.5
        w, d = (fr.m(w_m), fr.m(d_m)) if not rot else (fr.m(d_m), fr.m(w_m))
        if maxx - minx < w or maxy - miny < d:
            continue
        if against_wall:
            side = rng.randint(0, 3)
            if side == 0:
                x0, y0 = rng.uniform(minx, maxx - w), miny + fr.m(0.04)
            elif side == 1:
                x0, y0 = rng.uniform(minx, maxx - w), maxy - d - fr.m(0.04)
            elif side == 2:
                x0, y0 = minx + fr.m(0.04), rng.uniform(miny, maxy - d)
            else:
                x0, y0 = maxx - w - fr.m(0.04), rng.uniform(miny, maxy - d)
        else:
            x0, y0 = rng.uniform(minx, maxx - w), rng.uniform(miny, maxy - d)
        b = box(x0, y0, x0 + w, y0 + d)
        if inner.contains(b):
            return x0, y0, x0 + w, y0 + d, rot
    return None


def _render_furniture(cv_: Canvas, rooms, rng, lw):
    fr = cv_.fr
    v = 255 if rng.random() < 0.7 else rng.randint(110, 220)
    m = fr.m
    for kind, rlist in rooms.items():
        for room in rlist:
            if room.area < m(1.0) ** 2:
                continue
            items = []
            if kind == "bedroom":
                items = [("bed", 1.6 if rng.random() < 0.5 else 0.9, 2.0), ("wardrobe", rng.uniform(1.0, 2.4), 0.6),
                         ("desk", 1.2, 0.6), ("plant", 0.4, 0.4)]
            elif kind == "living":
                items = [("sofa", rng.uniform(1.8, 2.6), 0.9), ("table", 1.6, 0.9), ("armchair", 0.8, 0.8),
                         ("roundtable", 1.0, 1.0), ("tv", 1.6, 0.45), ("plant", 0.5, 0.5), ("rug", 2.0, 1.4)]
            elif kind == "kitchen":
                items = [("counter", rng.uniform(1.8, 3.6), 0.6), ("table", 1.2, 0.8)]
            elif kind == "bathroom":
                items = [("tub", 1.7, 0.75), ("shower", 0.9, 0.9), ("wc", 0.4, 0.65), ("basin", 0.6, 0.45),
                         ("washer", 0.6, 0.6)]
            elif kind == "storage":
                items = [("shelf", rng.uniform(1.0, 2.0), 0.4)]
            if not items:
                continue
            rng.shuffle(items)
            for name, w, d in items[: rng.randint(1, len(items))]:
                pl = _place(room, fr, rng, w, d, against_wall=name not in ("table", "roundtable", "rug"))
                if pl is None:
                    continue
                _draw_item(cv_, name, pl, rng, lw, v)


def _draw_item(cv_: Canvas, name, pl, rng, lw, v):
    x0, y0, x1, y1, rot = pl
    m = cv_.fr.m
    R = lambda a, b, c, d, **k: cv_.rect(a, b, c, d, lw=lw, v=v, **k)  # noqa: E731
    w, d = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if name == "bed":
        R(x0, y0, x1, y1)
        if w < d:
            ph = m(0.35)
            n = 2 if w > m(1.3) else 1
            for i in range(n):
                R(x0 + w * (i / n) + m(0.08), y0 + m(0.08), x0 + w * ((i + 1) / n) - m(0.08), y0 + ph)
            cv_.line((x0, y0 + d * 0.38), (x1, y0 + d * 0.38), lw, v)
            if rng.random() < 0.5:
                cv_.line((x1 - w * 0.4, y0 + d * 0.38), (x1, y0 + d * 0.55), lw, v)
        else:
            pw = m(0.35)
            R(x0 + m(0.08), y0 + m(0.08), x0 + pw, y1 - m(0.08))
            cv_.line((x0 + w * 0.38, y0), (x0 + w * 0.38, y1), lw, v)
    elif name in ("wardrobe", "shelf"):
        R(x0, y0, x1, y1)
        if rng.random() < 0.6:
            cv_.line((x0, y0), (x1, y1), lw, v)
            if rng.random() < 0.5:
                cv_.line((x0, y1), (x1, y0), lw, v)
    elif name == "desk":
        R(x0, y0, x1, y1)
        cv_.circle((cx, y1 + m(0.3) if w > d else cy), m(0.22), lw, v)
    elif name == "sofa":
        R(x0, y0, x1, y1)
        if w > d:
            R(x0, y0, x1, y0 + d * 0.25)
            R(x0, y0, x0 + m(0.2), y1)
            R(x1 - m(0.2), y0, x1, y1)
            cv_.line((cx, y0 + d * 0.25), (cx, y1), lw, v)
        else:
            R(x0, y0, x0 + w * 0.25, y1)
            R(x0, y0, x1, y0 + m(0.2))
            R(x0, y1 - m(0.2), x1, y1)
    elif name == "armchair":
        R(x0, y0, x1, y1)
        R(x0 + m(0.12), y0 + m(0.12), x1 - m(0.12), y1)
    elif name == "table":
        R(x0, y0, x1, y1)
        n = max(1, int((max(w, d)) / m(0.6)))
        cs = m(0.42)
        for i in range(n):
            t = (i + 0.5) / n
            if w >= d:
                xx = x0 + w * t
                R(xx - cs / 2, y0 - cs - m(0.05), xx + cs / 2, y0 - m(0.05))
                R(xx - cs / 2, y1 + m(0.05), xx + cs / 2, y1 + cs + m(0.05))
            else:
                yy = y0 + d * t
                R(x0 - cs - m(0.05), yy - cs / 2, x0 - m(0.05), yy + cs / 2)
                R(x1 + m(0.05), yy - cs / 2, x1 + cs + m(0.05), yy + cs / 2)
    elif name == "roundtable":
        r = min(w, d) / 2
        cv_.circle((cx, cy), r, lw, v)
        for k in range(4):
            a = k * math.pi / 2 + math.pi / 4
            cv_.circle((cx + (r + m(0.25)) * math.cos(a), cy + (r + m(0.25)) * math.sin(a)), m(0.2), lw, v)
    elif name == "tv":
        R(x0, y0, x1, y1)
    elif name == "plant":
        r = min(w, d) / 2
        cv_.circle((cx, cy), r, lw, v)
        for k in range(8):
            a = k * math.pi / 4
            cv_.line((cx, cy), (cx + r * math.cos(a), cy + r * math.sin(a)), 1, v)
    elif name == "rug":
        for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            cv_.dashed(a, b, m(0.15), m(0.08), 1, v)
    elif name == "counter":
        R(x0, y0, x1, y1)
        long_x = w >= d
        L = w if long_x else d
        # Spüle + Kochfeld
        def at(t, s_):
            return (x0 + L * t, cy) if long_x else (cx, y0 + L * t)
        sx, sy = at(rng.uniform(0.2, 0.4), 0)
        cv_.rect(sx - m(0.25), sy - m(0.2), sx + m(0.25), sy + m(0.2), lw=lw, v=v)
        cv_.circle((sx, sy), m(0.04), 1, v)
        kx, ky = at(rng.uniform(0.6, 0.8), 0)
        for dx in (-0.15, 0.15):
            for dy in (-0.13, 0.13):
                cv_.circle((kx + m(dx), ky + m(dy)), m(0.09), 1, v)
        if rng.random() < 0.5:
            fx, fy = at(0.95, 0)
            cv_.rect(fx - m(0.3), fy - m(0.3), fx + m(0.3), fy + m(0.3), lw=lw, v=v)
            cv_.line((fx - m(0.3), fy - m(0.3)), (fx + m(0.3), fy + m(0.3)), 1, v)
    elif name == "tub":
        R(x0, y0, x1, y1)
        cv_.ellipse((cx, cy), w / 2 - m(0.08), d / 2 - m(0.08), lw, v)
        cv_.circle((x0 + m(0.2) if w > d else cx, cy if w > d else y0 + m(0.2)), m(0.03), 1, v)
    elif name == "shower":
        R(x0, y0, x1, y1)
        cv_.line((x0, y0), (x1, y1), 1, v)
        cv_.line((x0, y1), (x1, y0), 1, v)
    elif name == "wc":
        if w < d:
            R(x0, y0, x1, y0 + d * 0.28)
            cv_.ellipse((cx, y0 + d * 0.62), w / 2 * 0.85, d * 0.35, lw, v)
        else:
            R(x0, y0, x0 + w * 0.28, y1)
            cv_.ellipse((x0 + w * 0.62, cy), w * 0.35, d / 2 * 0.85, lw, v)
    elif name == "basin":
        R(x0, y0, x1, y1)
        cv_.ellipse((cx, cy), w / 2 * 0.7, d / 2 * 0.65, lw, v)
    elif name == "washer":
        R(x0, y0, x1, y1)
        cv_.circle((cx, cy), min(w, d) * 0.35, lw, v)


def _render_floor_patterns(cv_: Canvas, rooms, rng, lab):
    fr = cv_.fr
    v = rng.randint(100, 230)
    for kind in ("bathroom", "kitchen", "balcony"):
        for room in rooms.get(kind, []):
            if rng.random() < 0.5:
                continue
            sp = fr.m(rng.choice([0.2, 0.3, 0.4]))
            minx, miny, maxx, maxy = room.bounds
            tmp = Canvas(cv_.h, cv_.w, fr, rng)
            x = minx
            while x < maxx:
                tmp.line((x, miny), (x, maxy), 1, v)
                x += sp
            y = miny
            while y < maxy:
                tmp.line((minx, y), (maxx, y), 1, v)
                y += sp
            mask = np.zeros((cv_.h, cv_.w), np.uint8)
            cv2.fillPoly(mask, [fr.fx(np.array(room.exterior.coords))], 255, cv2.LINE_8, SH)
            tmp.ink[mask == 0] = 0
            tmp.ink[lab > 0] = 0
            np.maximum(cv_.ink, tmp.ink, out=cv_.ink)
    if rng.random() < 0.15:   # Parkett im Wohnraum
        for room in rooms.get("living", []):
            sp = cv_.fr.m(0.12)
            minx, miny, maxx, maxy = room.bounds
            tmp = Canvas(cv_.h, cv_.w, fr, rng)
            y = miny
            while y < maxy:
                tmp.line((minx, y), (maxx, y), 1, v)
                y += sp
            mask = np.zeros((cv_.h, cv_.w), np.uint8)
            cv2.fillPoly(mask, [fr.fx(np.array(room.exterior.coords))], 255, cv2.LINE_8, SH)
            tmp.ink[mask == 0] = 0
            tmp.ink[lab > 0] = 0
            np.maximum(cv_.ink, tmp.ink, out=cv_.ink)


def _render_stair(cv_: Canvas, room: Polygon, rng, lw):
    fr = cv_.fr
    minx, miny, maxx, maxy = room.buffer(-fr.m(0.05)).bounds if not room.buffer(-fr.m(0.05)).is_empty else room.bounds
    w, d = maxx - minx, maxy - miny
    if w <= 0 or d <= 0:
        return
    tread = fr.m(rng.uniform(0.25, 0.3))
    along_x = w > d
    cv_.rect(minx, miny, maxx, maxy, lw=lw)
    if along_x:
        x = minx + tread
        while x < maxx - 1e-6:
            cv_.line((x, miny), (x, maxy), lw)
            x += tread
        cy = (miny + maxy) / 2
        cv_.line((minx + tread / 2, cy), (maxx - tread / 2, cy), 1)
        cv_.line((maxx - tread / 2, cy), (maxx - tread * 1.2, cy - tread * 0.5), 1)
        cv_.line((maxx - tread / 2, cy), (maxx - tread * 1.2, cy + tread * 0.5), 1)
    else:
        y = miny + tread
        while y < maxy - 1e-6:
            cv_.line((minx, y), (maxx, y), lw)
            y += tread
        cx = (minx + maxx) / 2
        cv_.line((cx, miny + tread / 2), (cx, maxy - tread / 2), 1)


def _random_stairs(cv_: Canvas, rooms, rng, lw):
    cand = rooms.get("living", []) + rooms.get("inner", [])
    if not cand:
        return
    room = rng.choice(cand)
    pl = _place(room, cv_.fr, rng, 2.8, 1.1)
    if pl:
        x0, y0, x1, y1, _ = pl
        _render_stair(cv_, box(x0, y0, x1, y1), rng, lw)


def _render_room_texts(cv_: Canvas, rooms, rng):
    fr = cv_.fr
    th = fr.m(rng.uniform(0.18, 0.4))
    font = rng.choice(FONTS)
    with_area = rng.random() < 0.6
    for kind, rl in rooms.items():
        if kind not in ROOM_NAMES:
            continue
        for room in rl:
            if room.area < fr.m(1.2) ** 2:
                continue
            p = room.representative_point()
            name = rng.choice(ROOM_NAMES[kind])
            cv_.text(name, (p.x, p.y), th, font=font)
            if with_area:
                a = room.area * fr.mu * fr.mu
                s = rng.choice([f"{a:.1f} m2", f"{a:.2f} m2", f"F = {a:.1f} m2", f"{a:.1f}"])
                cv_.text(s, (p.x, p.y + 1.7 * th * (1 if rng.random() < 0.5 else -1)), th * 0.75, font=font)


def _fmt_dim(meters: float, fmt: str) -> tuple[str, str]:
    if fmt == "cm":
        return f"{meters * 100:.0f}", ""
    if fmt == "swiss":
        cm = meters * 100
        whole = int(cm)
        frac = int(round((cm - whole) * 10))
        if frac == 10:
            whole, frac = whole + 1, 0
        main = f"{whole / 100:.2f}" if whole >= 100 else f"{whole}"
        return main, (str(frac) if frac else "")
    return f"{meters:.2f}", ""


def _render_dimensions(cv_: Canvas, walls, doors, wins, rng, ext_u, lw):
    fr = cv_.fr
    m = fr.m
    minx, miny, maxx, maxy = walls.bounds
    minx, miny, maxx, maxy = minx - ext_u, miny - ext_u, maxx + ext_u, maxy + ext_u
    fmt = rng.choice(["m", "cm", "swiss", "swiss"])
    tick = rng.choice(["slash", "slash", "dot", "arrow", "tick"])
    th = m(rng.uniform(0.16, 0.3))
    font = rng.choice(FONTS)
    gap1 = m(rng.uniform(0.5, 1.0))
    step = m(rng.uniform(0.45, 0.8))
    nlev = rng.choice([1, 2, 2, 3])
    sides = [s for s in ("b", "t", "l", "r") if rng.random() < 0.75] or ["b"]
    feats_all = []
    for p in polys(walls):
        feats_all.extend(list(p.exterior.coords))
    feats_op = []
    for p in doors + wins:
        feats_op.extend(list(p.exterior.coords))
    vline = 255 if rng.random() < 0.7 else rng.randint(120, 220)
    for side in sides:
        horiz = side in ("b", "t")
        edge = (miny if side == "b" else maxy) if horiz else (minx if side == "l" else maxx)
        sign = -1 if side in ("b", "l") else 1
        near = m(1.2)

        def coords(pts):
            out = []
            for x, y in pts:
                fpos, apos = (y, x) if horiz else (x, y)
                if abs(fpos - edge) < near + ext_u:
                    out.append(apos)
            return out

        lv_pts = [sorted(set(round(a, 1) for a in coords(feats_all) + coords(feats_op))),
                  sorted(set(round(a, 1) for a in coords(feats_all))),
                  [(minx if horiz else miny), (maxx if horiz else maxy)]]
        for lev in range(nlev):
            pts = lv_pts[min(lev, 2)] if lev < nlev - 1 or nlev == 1 else lv_pts[2]
            pts = [p for i, p in enumerate(pts) if i == 0 or p - pts[i - 1] > m(0.04)]
            if len(pts) < 2:
                continue
            off = edge + sign * (gap1 + lev * step)

            def P(a, f):
                return (a, f) if horiz else (f, a)

            cv_.line(P(pts[0] - m(0.15), off), P(pts[-1] + m(0.15), off), lw, vline)
            for a in pts:
                if rng.random() < 0.85:
                    cv_.line(P(a, edge + sign * m(0.15)), P(a, off + sign * m(0.12)), 1, vline)
                tl = m(0.09)
                if tick == "slash":
                    cv_.line(P(a - tl, off - tl * (1 if horiz else -1)), P(a + tl, off + tl * (1 if horiz else -1)), lw + 1, vline)
                elif tick == "dot":
                    cv_.circle(P(a, off), m(0.035), 1, vline, filled=True)
                elif tick == "tick":
                    cv_.line(P(a, off - tl), P(a, off + tl), lw + 1, vline)
            if tick == "arrow":
                for a0, a1 in zip(pts[:-1], pts[1:]):
                    al = m(0.12)
                    for a, dirn in ((a0, 1), (a1, -1)):
                        tip = P(a, off)
                        cv_.fill([tip, P(a + dirn * al, off - al * 0.3), P(a + dirn * al, off + al * 0.3)], vline)
            for a0, a1 in zip(pts[:-1], pts[1:]):
                main, sup = _fmt_dim((a1 - a0) * fr.mu, fmt)
                mid = (a0 + a1) / 2
                pos = P(mid, off + sign * th * 0.9 * (1 if rng.random() < 0.8 else -1))
                cv_.text(main, pos, th, 0 if horiz else -90, v=vline, font=font)
                if sup:
                    dx = len(main) * th * 0.45 + th * 0.2
                    pos2 = P(mid + dx, off + sign * th * 1.3) if horiz else P(mid - dx, off + sign * th * 1.3)
                    cv_.text(sup, pos2, th * 0.6, 0 if horiz else -90, v=vline, font=font)


def _render_inner_dims(cv_: Canvas, rooms, rng, lw):
    fr = cv_.fr
    cand = [r for k in ("living", "bedroom", "kitchen") for r in rooms.get(k, [])]
    rng.shuffle(cand)
    for room in cand[: rng.randint(1, 3)]:
        minx, miny, maxx, maxy = room.bounds
        th = fr.m(0.2)
        if rng.random() < 0.5:
            y = rng.uniform(miny + 0.3 * (maxy - miny), maxy - 0.3 * (maxy - miny))
            cv_.line((minx, y), (maxx, y), lw)
            for x in (minx, maxx):
                cv_.line((x - fr.m(0.08), y - fr.m(0.08)), (x + fr.m(0.08), y + fr.m(0.08)), lw + 1)
            cv_.text(f"{(maxx - minx) * fr.mu:.2f}", ((minx + maxx) / 2, y - th), th)
        else:
            x = rng.uniform(minx + 0.3 * (maxx - minx), maxx - 0.3 * (maxx - minx))
            cv_.line((x, miny), (x, maxy), lw)
            for y in (miny, maxy):
                cv_.line((x - fr.m(0.08), y + fr.m(0.08)), (x + fr.m(0.08), y - fr.m(0.08)), lw + 1)
            cv_.text(f"{(maxy - miny) * fr.mu:.2f}", (x - th, (miny + maxy) / 2), th, -90)


def _render_axes(cv_: Canvas, walls, rng, lw):
    fr = cv_.fr
    minx, miny, maxx, maxy = walls.bounds
    ext = fr.m(1.5)
    r = fr.m(0.3)
    xs = sorted(rng.sample([p[0] for g in polys(walls) for p in g.exterior.coords], k=min(4, len(polys(walls)) * 4)))
    for i, x in enumerate(xs[:4]):
        cv_.dashed((x, miny - ext), (x, maxy + ext), fr.m(0.6), fr.m(0.2), 1, 200, dot=True)
        cv_.circle((x, miny - ext - r), r, 1)
        cv_.text(chr(65 + i), (x, miny - ext - r), r)
    ys = sorted(rng.sample([p[1] for g in polys(walls) for p in g.exterior.coords], k=min(3, len(polys(walls)) * 4)))
    for i, y in enumerate(ys[:3]):
        cv_.dashed((minx - ext, y), (maxx + ext, y), fr.m(0.6), fr.m(0.2), 1, 200, dot=True)
        cv_.circle((minx - ext - r, y), r, 1)
        cv_.text(str(i + 1), (minx - ext - r, y), r)


def _render_north(cv_: Canvas, walls, rng, lw):
    fr = cv_.fr
    minx, miny, maxx, maxy = walls.bounds
    c = (maxx + fr.m(1.0), maxy + fr.m(0.8)) if rng.random() < 0.5 else (minx - fr.m(1.0), miny - fr.m(0.8))
    r = fr.m(rng.uniform(0.4, 0.7))
    cv_.circle(c, r, lw)
    a = rng.uniform(0, 2 * math.pi)
    tip = (c[0] + r * math.cos(a), c[1] + r * math.sin(a))
    l1 = (c[0] + 0.35 * r * math.cos(a + 2.4), c[1] + 0.35 * r * math.sin(a + 2.4))
    l2 = (c[0] + 0.35 * r * math.cos(a - 2.4), c[1] + 0.35 * r * math.sin(a - 2.4))
    cv_.fill([tip, l1, c, l2])
    cv_.text("N", (c[0] + 1.4 * r * math.cos(a), c[1] + 1.4 * r * math.sin(a)), r * 0.5)


def _render_title(cv_: Canvas, walls, rng):
    fr = cv_.fr
    minx, miny, maxx, maxy = walls.bounds
    th = fr.m(rng.uniform(0.35, 0.7))
    y = (miny - fr.m(1.6)) if rng.random() < 0.5 else (maxy + fr.m(1.6))
    cv_.text(rng.choice(TITLES), ((minx + maxx) / 2, y), th)
    if rng.random() < 0.5:
        cv_.text(rng.choice(["1:100", "M 1:100", "1:50", "Mst. 1:100", "1:200"]), ((minx + maxx) / 2, y + 1.8 * th), th * 0.6)


def _render_scribbles(cv_: Canvas, rng):
    for _ in range(rng.randint(1, 4)):
        x0, y0 = rng.uniform(0, cv_.w), rng.uniform(0, cv_.h)
        pts = [(x0, y0)]
        for _ in range(rng.randint(3, 8)):
            pts.append((pts[-1][0] + rng.uniform(-60, 60), pts[-1][1] + rng.uniform(-60, 60)))
        p = np.array(pts)
        t = np.linspace(0, 1, 60)
        idx = t * (len(p) - 1)
        q = np.c_[np.interp(idx, np.arange(len(p)), p[:, 0]), np.interp(idx, np.arange(len(p)), p[:, 1])]
        q = cv2.GaussianBlur(q.reshape(-1, 1, 2), (1, 9), 3).reshape(-1, 2)
        cv2.polylines(cv_.ink, [np.round(q * F).astype(np.int32)], False, rng.randint(150, 255), rng.randint(1, 2), cv2.LINE_AA, SH)
    if rng.random() < 0.6:
        x, y = rng.uniform(0, cv_.w * 0.7), rng.uniform(20, cv_.h)
        cv2.putText(cv_.ink, rng.choice(["neu", "abbrechen", "best.", "Kontrolle!", "ok", "Fenster neu"]),
                    (int(x), int(y)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, rng.uniform(0.6, 1.4), 255, 1, cv2.LINE_AA)


def _render_sheet(cv_: Canvas, rng, lw):
    h, w = cv_.h, cv_.w
    m = int(rng.uniform(0.015, 0.04) * max(h, w))
    t = max(1, int(lw * rng.uniform(1.0, 2.5)))
    cv2.rectangle(cv_.ink, (m, m), (w - m, h - m), 255, t)
    if rng.random() < 0.8:      # Plankopf
        bw, bh = int(w * rng.uniform(0.22, 0.35)), int(h * rng.uniform(0.08, 0.16))
        x0, y0 = w - m - bw, h - m - bh
        cv2.rectangle(cv_.ink, (x0, y0), (w - m, h - m), 255, max(1, t - 1))
        rows = rng.randint(2, 5)
        for i in range(1, rows):
            y = y0 + bh * i // rows
            cv2.line(cv_.ink, (x0, y), (w - m, y), 255, 1)
        cv2.line(cv_.ink, (x0 + bw // 3, y0), (x0 + bw // 3, h - m), 255, 1)
        fs = max(0.3, bh / rows / 40)
        for i in range(rows):
            y = y0 + bh * i // rows + int(bh / rows * 0.7)
            cv2.putText(cv_.ink, rng.choice(["Projekt", "Bauherr", "Plan Nr.", "Datum", "Gez.", "Mst.", "Architekt"]),
                        (x0 + 4, y), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 1, cv2.LINE_AA)
            cv2.putText(cv_.ink, rng.choice(["MFH Sonnenweg", "123-45", "1:100", "12.03.1978", "EFH Muster", "A3"]),
                        (x0 + bw // 3 + 6, y), cv2.FONT_HERSHEY_SIMPLEX, fs, 255, 1, cv2.LINE_AA)


def _render_columns(cv_: Canvas, rooms, rng):
    fr = cv_.fr
    for room in rooms.get("living", [])[:1]:
        p = room.representative_point()
        s = fr.m(0.25)
        cv_.rect(p.x - s / 2, p.y - s / 2, p.x + s / 2, p.y + s / 2, filled=True)


# =============================================================================== Alterung
def _degrade(ink: np.ndarray, lab: np.ndarray, rng, nrng):
    h, w = ink.shape
    ink = ink.astype(np.float32) / 255.0
    # Strichstärke verändern
    r = rng.random()
    if r < 0.08:
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
    elif r < 0.14:
        ink = cv2.erode(ink, np.ones((2, 2), np.uint8)) * 0.6 + ink * 0.4
    # Ausfälle (verblasste Linien)
    if rng.random() < 0.2:
        holes = cv2.resize(nrng.random((max(2, h // 25), max(2, w // 25))).astype(np.float32), (w, h))
        ink *= np.clip((holes - rng.uniform(0.1, 0.35)) * 4, 0.15, 1.0)
    # Handzittern / Papierverzug
    if rng.random() < 0.15:
        amp = rng.uniform(1.0, 3.5)
        sig = rng.uniform(25, 70)
        dx = cv2.GaussianBlur(nrng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sig)
        dy = cv2.GaussianBlur(nrng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sig)
        dx *= amp / (dx.std() + 1e-6)
        dy *= amp / (dy.std() + 1e-6)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        ink = cv2.remap(ink, gx + dx, gy + dy, cv2.INTER_LINEAR)
        lab = cv2.remap(lab, gx + dx, gy + dy, cv2.INTER_NEAREST)

    paper = rng.uniform(215, 255)
    ink_v = rng.uniform(0, 70) if rng.random() < 0.85 else rng.uniform(70, 150)   # Bleistift
    img = paper - ink * (paper - ink_v)
    # Papierstruktur / Vergilbung / ungleichmässige Ausleuchtung
    if rng.random() < 0.6:
        g = cv2.resize(nrng.random((4, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        img *= 1.0 - rng.uniform(0.0, 0.25) * g
    if rng.random() < 0.3:
        tex = cv2.GaussianBlur(nrng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(1, 4))
        img += tex * rng.uniform(2, 8)
    if rng.random() < 0.15:   # Flecken
        for _ in range(rng.randint(1, 4)):
            cv2.ellipse(img, (rng.randint(0, w), rng.randint(0, h)), (rng.randint(10, w // 6 + 11), rng.randint(10, h // 6 + 11)),
                        rng.uniform(0, 180), 0, 360, float(rng.uniform(150, 230)), -1)
        img = cv2.GaussianBlur(img, (0, 0), 3) * 0.3 + img * 0.7
    if rng.random() < 0.06:   # Falz
        x = rng.randint(0, w - 1)
        img[:, max(0, x - 1):x + 2] *= rng.uniform(0.75, 0.92)
    # Perspektive / Foto
    if rng.random() < 0.12:
        j = rng.uniform(0.01, 0.05)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-j, j) * w, rng.uniform(-j, j) * h] for _ in range(4)])
        Hm = cv2.getPerspectiveTransform(src, dst)
        bgv = rng.uniform(40, 160) if rng.random() < 0.5 else paper
        img = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LINEAR, borderValue=bgv)
        lab = cv2.warpPerspective(lab, Hm, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.3))
    if rng.random() < 0.6:
        img += nrng.normal(0, rng.uniform(1, 10), img.shape).astype(np.float32)
    if rng.random() < 0.1:    # Staub
        n = rng.randint(20, 400)
        ys, xs = nrng.integers(0, h, n), nrng.integers(0, w, n)
        img[ys, xs] = rng.uniform(0, 80)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 95)])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img, lab

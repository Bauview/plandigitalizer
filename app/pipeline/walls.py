"""Wandmasken-Geometrie: Zerlegung in achsparallele Wandrechtecke, Gebäudeumriss, Aussenwände.

Gemeinsam genutzt von der 2D-Rekonstruktion (``essential``) und dem Gebäudemodell (``bim.derive``).
Rechtecke: (x0, y0, x1, y1, 'h'|'v') in Pixeln.
"""
from __future__ import annotations

import cv2
import numpy as np

from .thinning import thinning


def median_thickness(mask: np.ndarray) -> float:
    """Mittlere Wandstärke (px) einer Wandmaske über die Distanztransformation auf dem Skelett."""
    if not np.any(mask):
        return 0.0
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    sk = thinning(mask) > 0
    v = 2 * dt[sk]
    return float(np.median(v)) if v.size else 0.0


def decompose(mask: np.ndarray, min_long: float, keep_rest: bool = True) -> tuple[list[tuple], np.ndarray]:
    """Zerlegt die Wandmaske in achsparallele Wandrechtecke.

    Gibt (rechtecke, rest) zurück; ``rest`` enthält Wandteile, die nicht achsparallel sind
    (schräge oder runde Wände) und als Polygon übernommen werden.
    """
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    sk = thinning(mask) > 0
    t_est = float(np.median(2 * dt[sk])) if np.any(sk) else 10.0
    L = int(max(3 * t_est, min_long))
    Hm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    Vm = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    # Kreuzungen/Ecken der dickeren Wand zuordnen (T-Stoss: Innenwand endet an der Aussenwand)
    inter = cv2.bitwise_and(Hm, Vm)
    n_i, lab_i, st_i, _ = cv2.connectedComponentsWithStats(inter, connectivity=4)
    to_v = np.zeros(n_i, bool)
    for i in range(1, n_i):
        to_v[i] = st_i[i, cv2.CC_STAT_WIDTH] > 1.25 * st_i[i, cv2.CC_STAT_HEIGHT]
    Hm[to_v[lab_i]] = 0
    Vm[(lab_i > 0) & ~to_v[lab_i]] = 0
    rest = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(Hm, Vm)))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    rects: list[tuple] = []
    for part, orient in ((Hm, "h"), (Vm, "v")):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(part, connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 0.25 * t_est * t_est:
                continue
            comp = lab[y:y + h, x:x + w] == i
            if orient == "h":
                counts = comp.sum(axis=0)
                rows = np.arange(h)[:, None]
                mids = (comp * rows).sum(axis=0) / np.maximum(counts, 1)
                ok = counts > 0
                t = float(np.median(counts[ok]))
                c = float(np.median(mids[ok])) + y + 0.5
                rects.append((float(x), c - t / 2, float(x + w), c + t / 2, "h"))
            else:
                counts = comp.sum(axis=1)
                cols = np.arange(w)[None, :]
                mids = (comp * cols).sum(axis=1) / np.maximum(counts, 1)
                ok = counts > 0
                t = float(np.median(counts[ok]))
                c = float(np.median(mids[ok])) + x + 0.5
                rects.append((c - t / 2, float(y), c + t / 2, float(y + h), "v"))
    leftover = np.zeros_like(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=4)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.5 * t_est * t_est or max(w, h) < 0.8 * t_est:
            continue
        fill = area / float(w * h)
        if fill > 0.75 or not keep_rest:
            rects.append((float(x), float(y), float(x + w), float(y + h), "h" if w >= h else "v"))
        else:
            leftover[lab == i] = 255     # schräge/runde Wand
    return rects, leftover


def merge_rects(rects: list[tuple], gap: float) -> list[tuple]:
    """Fügt kollineare, aneinanderstossende Wandstücke gleicher Richtung zusammen."""
    out = list(rects)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if a[4] != b[4]:
                    continue
                h = a[4] == "h"
                ac0, ac1 = (a[1], a[3]) if h else (a[0], a[2])
                bc0, bc1 = (b[1], b[3]) if h else (b[0], b[2])
                ta, tb = ac1 - ac0, bc1 - bc0
                if abs((ac0 + ac1) - (bc0 + bc1)) / 2 > 0.5 * min(ta, tb) or abs(ta - tb) > 0.6 * max(ta, tb):
                    continue
                aa0, aa1 = (a[0], a[2]) if h else (a[1], a[3])
                ba0, ba1 = (b[0], b[2]) if h else (b[1], b[3])
                if ba0 > aa1 + gap or aa0 > ba1 + gap:
                    continue
                la, lb = aa1 - aa0, ba1 - ba0
                c = ((ac0 + ac1) / 2 * la + (bc0 + bc1) / 2 * lb) / (la + lb)
                t = (ta * la + tb * lb) / (la + lb)
                lo, hi = min(aa0, ba0), max(aa1, ba1)
                merged = (lo, c - t / 2, hi, c + t / 2, "h") if h else (c - t / 2, lo, c + t / 2, hi, "v")
                out[i] = merged
                del out[j]
                changed = True
                break
            if changed:
                break
    return out


def footprint(mask: np.ndarray, close_px: int) -> np.ndarray:
    """Gebäudegrundfläche: Wandmaske mit geschlossenen Lücken, Innenräume gefüllt."""
    k = max(3, close_px | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    h, w = closed.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = closed
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    outside = (ff[1:-1, 1:-1] == 128)
    return np.where(outside, 0, 255).astype(np.uint8)


def sample(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi, yi = int(round(x)), int(round(y))
    return 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0


def is_external(outside: np.ndarray, r: tuple) -> bool:
    """Aussenwand, wenn eine Längsseite (mittlere 60 %) an den Aussenraum grenzt."""
    x0, y0, x1, y1, orient = r
    off = 4
    if orient == "h":
        xs = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), 15)
        sides = [[(x, y0 - off) for x in xs], [(x, y1 + off) for x in xs]]
    else:
        ys = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), 15)
        sides = [[(x0 - off, y) for y in ys], [(x1 + off, y) for y in ys]]
    return any(np.mean([sample(outside, x, y) for x, y in side]) > 0.4 for side in sides)


def raster_rects(shape, rects, value=255) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for x0, y0, x1, y1, _ in rects:
        cv2.rectangle(m, (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1), value, -1)
    return m

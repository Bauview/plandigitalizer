"""Rasterbild -> Geometrie.

Vorgehen
1. Strichbreite messen (Distanztransformation).
2. Dicke, gefüllte Bereiche (z.B. schwarz gefüllte Wände) als Fläche
   erkennen und als Umriss-Polygone ausgeben.
3. Restliche, dünne Linien skelettieren (1 px breit), das Skelett an
   Knotenpunkten in Äste zerlegen und jeden Ast als Pixelpfad verfolgen.
4. Pfade vereinfachen (Douglas-Peucker), an Ecken teilen und jedes Teilstück
   entweder als Kreisbogen (Kreisanpassung) oder als Linien ausgeben.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Arc, Circle, Line, Pt
from .thinning import thinning

OFFS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # 4er-Nachbarn zuerst


@dataclass
class VecResult:
    lines: list[Line] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    wall_regions: list[list[list[Pt]]] = field(default_factory=list)  # [Aussenring, Löcher...]
    wall_mask: np.ndarray | None = None
    thin_w: float = 2.0
    comp_diag: dict[int, float] = field(default_factory=dict)


def _thinning(img: np.ndarray) -> np.ndarray:
    return thinning(img)


def vectorize(binary: np.ndarray, tolerance: float = 1.0) -> VecResult:
    res = VecResult()
    h, w = binary.shape
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    sk_full = _thinning(binary)
    widths = 2.0 * dt[sk_full > 0]
    thin_w = float(max(1.5, np.percentile(widths, 35))) if widths.size else 2.0
    res.thin_w = thin_w

    # --- gefüllte / sehr dicke Bereiche -> Wandflächen
    k = int(round(max(5.0, 2.8 * thin_w))) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    wall = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kern)
    # Ecken rekonstruieren: wieder etwas wachsen lassen, aber nur innerhalb der Tinte
    wall = cv2.bitwise_and(binary, cv2.dilate(wall, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        ext = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if stats[i, cv2.CC_STAT_AREA] >= 4 * k * k and ext >= 4 * k:
            keep[i] = 255
    wall = keep[lab]
    res.wall_mask = wall
    if np.count_nonzero(wall):
        res.wall_regions = _wall_outlines(wall, eps=max(1.5, 0.6 * thin_w))

    # --- dünne Linien
    grow = int(math.ceil(thin_w * 0.75)) + 1
    wall_grown = cv2.dilate(wall, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    rest = cv2.bitwise_and(binary, cv2.bitwise_not(wall_grown))
    sk = (_thinning(rest) > 0).astype(np.uint8)

    ncomp, clab, cstats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    min_px = max(5, int(2.5 * thin_w))
    for i in range(1, ncomp):
        res.comp_diag[i] = math.hypot(cstats[i, cv2.CC_STAT_WIDTH], cstats[i, cv2.CC_STAT_HEIGHT])
        if res.comp_diag[i] < min_px:
            sk[clab == i] = 0

    paths = _trace(sk)
    eps = max(1.0, 0.45 * thin_w) * tolerance
    max_r = 0.6 * max(h, w)
    for path, closed in paths:
        if len(path) < 2:
            continue
        if float(np.sum(np.hypot(*np.diff(path, axis=0).T))) < min_px:
            continue
        x0, y0 = int(round(path[0][0])), int(round(path[0][1]))
        comp = int(clab[min(h - 1, max(0, y0)), min(w - 1, max(0, x0))])
        ip = np.clip(np.round(path).astype(int), [0, 0], [w - 1, h - 1])
        width = float(np.median(2.0 * dt[ip[:, 1], ip[:, 0]]))
        _emit_path(res, path, closed, eps, thin_w, max_r, width, comp, tolerance)
    return res


# ---------------------------------------------------------------------------
def _wall_outlines(mask: np.ndarray, eps: float) -> list[list[list[Pt]]]:
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    regions = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue  # Loch, wird beim Elternelement behandelt
        rings = [_orthogonalize(cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float))]
        child = hier[i][2]
        while child != -1:
            if cv2.contourArea(contours[child]) > 16:
                rings.append(_orthogonalize(cv2.approxPolyDP(contours[child], eps, True).reshape(-1, 2).astype(float)))
            child = hier[child][0]
        rings = [r for r in rings if len(r) >= 3]
        if rings:
            regions.append(rings)
    return regions


def _orthogonalize(poly: np.ndarray, tol_deg: float = 6.0) -> list[Pt]:
    """Fast waagrechte/senkrechte Polygonkanten exakt ausrichten."""
    n = len(poly)
    if n < 3:
        return [tuple(p) for p in poly]
    kinds = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
        if ang < tol_deg or ang > 180 - tol_deg:
            kinds.append("h")
        elif abs(ang - 90) < tol_deg:
            kinds.append("v")
        else:
            kinds.append("d")
    out = poly.copy()
    for i in range(n):
        a, b = (i - 1) % n, i  # Kanten vor und nach Vertex i
        ea, eb = kinds[a], kinds[b]
        for e, idx in ((ea, a), (eb, b)):
            p, q = poly[idx], poly[(idx + 1) % n]
            if e == "h":
                out[i][1] = (p[1] + q[1]) / 2
            elif e == "v":
                out[i][0] = (p[0] + q[0]) / 2
    # kollineare Zwischenpunkte entfernen
    res = []
    for i in range(n):
        p, c, q = out[i - 1], out[i], out[(i + 1) % n]
        cross = (c[0] - p[0]) * (q[1] - c[1]) - (c[1] - p[1]) * (q[0] - c[0])
        if abs(cross) > 1e-6 or len(out) <= 3:
            res.append((float(c[0]), float(c[1])))
    return res


def _trace(sk: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Zerlegt ein 1-px-Skelett in geordnete Pixelpfade."""
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    nb = cv2.filter2D(sk.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    junction = ((sk == 1) & (nb >= 3)).astype(np.uint8)
    branches = (sk & (1 - junction)).astype(np.uint8)
    nj, jlab, _, jcent = cv2.connectedComponentsWithStats(junction, connectivity=8)
    nlab, blab = cv2.connectedComponents(branches, connectivity=8)
    h, w = sk.shape

    ys, xs = np.nonzero(blab)
    labs = blab[ys, xs]
    order = np.argsort(labs, kind="stable")
    ys, xs, labs = ys[order], xs[order], labs[order]
    bounds = np.searchsorted(labs, np.arange(1, nlab + 1))
    bounds = np.append(bounds, len(labs))

    def junction_at(x: int, y: int):
        for dx, dy in OFFS:
            xx, yy = x + dx, y + dy
            if 0 <= xx < w and 0 <= yy < h and jlab[yy, xx]:
                return jlab[yy, xx]
        return 0

    paths: list[tuple[np.ndarray, bool]] = []
    for li in range(nlab - 1):
        s, e = bounds[li], bounds[li + 1]
        if e <= s:
            continue
        pix = set(zip(xs[s:e].tolist(), ys[s:e].tolist()))

        def nbrs(p):
            return [(p[0] + dx, p[1] + dy) for dx, dy in OFFS if (p[0] + dx, p[1] + dy) in pix]

        start = None
        for p in pix:
            if len(nbrs(p)) <= 1:
                start = p
                break
        loop = start is None
        if loop:
            start = next(iter(pix))
        path = [start]
        seen = {start}
        cur = start
        while True:
            nxt = [q for q in nbrs(cur) if q not in seen]
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            seen.add(cur)
        closed = loop and len(path) > 8 and (path[0] in nbrs(path[-1]) or len(nbrs(path[-1])) > 0)
        pts = [(float(x), float(y)) for x, y in path]
        if not closed:
            ja = junction_at(*path[0])
            jb = junction_at(*path[-1])
            if ja:
                pts.insert(0, (float(jcent[ja][0]), float(jcent[ja][1])))
            if jb and (jb != ja or len(path) > 1):
                pts.append((float(jcent[jb][0]), float(jcent[jb][1])))
        else:
            pts.append(pts[0])
        paths.append((np.array(pts, dtype=np.float64), closed))
    return paths


def _rdp_idx(pts: np.ndarray, eps: float) -> np.ndarray:
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = pts[s], pts[e]
        seg = b - a
        L = math.hypot(seg[0], seg[1])
        sub = pts[s + 1:e]
        if L < 1e-9:
            d = np.hypot(sub[:, 0] - a[0], sub[:, 1] - a[1])
        else:
            d = np.abs(seg[0] * (sub[:, 1] - a[1]) - seg[1] * (sub[:, 0] - a[0])) / L
        i = int(np.argmax(d))
        if d[i] > eps:
            m = s + 1 + i
            keep[m] = True
            stack.append((s, m))
            stack.append((m, e))
    return np.nonzero(keep)[0]


def _fit_circle(P: np.ndarray):
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = c[0], c[1]
    r2 = c[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return cx, cy, r, rms


def _emit_path(res: VecResult, path, closed, eps, thin_w, max_r, width, comp, tolerance):
    idx = _rdp_idx(path, eps)
    verts = path[idx]
    # Ecken bestimmen (Richtungsänderung > 40°)
    cuts = [0]
    for j in range(1, len(verts) - 1):
        v1 = verts[j] - verts[j - 1]
        v2 = verts[j + 1] - verts[j]
        a1, a2 = math.atan2(v1[1], v1[0]), math.atan2(v2[1], v2[0])
        turn = abs((math.degrees(a2 - a1) + 180) % 360 - 180)
        if turn > 40:
            cuts.append(j)
    cuts.append(len(verts) - 1)

    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        piece_v = verts[c0:c1 + 1]
        i0, i1 = idx[c0], idx[c1]
        if c1 - c0 >= 3 and i1 - i0 >= 12:
            arc = _try_arc(path[i0:i1 + 1], thin_w, max_r, tolerance)
            if arc is not None:
                if isinstance(arc, Circle):
                    res.circles.append(arc)
                else:
                    res.arcs.append(arc)
                continue
        for a, b in zip(piece_v[:-1], piece_v[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) >= 1.5:
                res.lines.append(Line((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), "LINES", width, comp))


def _try_arc(P: np.ndarray, thin_w, max_r, tolerance):
    fit = _fit_circle(P)
    if fit is None:
        return None
    cx, cy, r, rms = fit
    if r < 3 * thin_w or r > max_r:
        return None
    if rms > max(1.0, 0.02 * r) * tolerance:
        return None
    ang = np.unwrap(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    d = np.diff(ang)
    sgn = 1.0 if ang[-1] >= ang[0] else -1.0
    if np.mean(d * sgn < -0.02) > 0.1:
        return None  # nicht monoton
    sweep = math.degrees(abs(ang[-1] - ang[0]))
    if sweep < 30:
        return None
    if sweep >= 340:
        return Circle((cx, cy), r)
    a0, a1 = math.degrees(ang[0]), math.degrees(ang[-1])
    start, end = (a0, a1) if a1 >= a0 else (a1, a0)
    return Arc((cx, cy), r, start, end)

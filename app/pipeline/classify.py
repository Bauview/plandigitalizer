"""Heuristische Zuordnung der Geometrie zu Plan-Elementen / CAD-Layern.

Bewusst einfache, nachvollziehbare Regeln – keine KI, keine Cloud.
Alles, was nicht sicher zugeordnet werden kann, bleibt auf LINES bzw. SYMBOLS.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import cv2
import numpy as np

from .geometry import Arc, Circle, Line, Text
from .scale import parse_dimension_mm

_NUMERIC = re.compile(r"^\d{1,3}([.,]\d{1,3})?[⁰¹²³⁴⁵⁶⁷⁸⁹]*$")


def classify(lines: list[Line], arcs: list[Arc], circles: list[Circle], texts: list[Text],
             wall_mask: np.ndarray | None, thin_w: float, comp_diag: dict[int, float],
             shape: tuple[int, int], mm_per_px: float | None) -> list[float]:
    """Setzt ``layer`` in-place. Gibt Massstabskandidaten (mm/px) aus Massketten zurück."""
    short = float(min(shape))
    text_h = float(np.median([t.height for t in texts])) if texts else 0.012 * short
    if mm_per_px:
        dmin, dmax = 70.0 / mm_per_px, 650.0 / mm_per_px
    else:
        dmin, dmax = max(1.8 * thin_w, 0.003 * short), 0.03 * short
    lmin = max(12 * thin_w, 0.025 * short)
    tol = max(3.0, 1.5 * thin_w)

    _mark_hatch(lines, short, thin_w)
    _mark_stairs(lines, short, thin_w)
    for l in lines:
        if l.layer == "LINES" and l.width >= 1.8 * thin_w and l.length >= lmin and (l.is_h or l.is_v):
            l.layer = "WALLS"
    _mark_windows(lines, wall_mask, shape, short, dmax, tol)      # Fenster in gefüllten Wänden
    candidates = _mark_dimensions(lines, texts, text_h)
    _mark_wall_pairs(lines, dmin, dmax, lmin, tol)
    _mark_doors(arcs, lines, short, mm_per_px)
    _mark_windows(lines, wall_mask, shape, short, dmax, tol)      # Fenster in Doppellinien-Wänden

    small = 0.02 * short
    for l in lines:
        if l.layer == "LINES" and comp_diag.get(l.comp, 1e9) < small:
            l.layer = "SYMBOLS"
    for c in circles:
        c.layer = "SYMBOLS" if c.radius < small else "LINES"
    for t in texts:
        if t.layer != "DIMENSIONS":
            t.layer = "TEXT"
    return candidates


# ---------------------------------------------------------------------------
def _axis(l: Line) -> bool:
    return l.is_h or l.is_v


def _mark_hatch(lines, short, thin_w):
    diag = [l for l in lines if l.layer == "LINES" and not _axis(l) and l.length < 0.1 * short
            and 6 < (l.angle % 90) < 84]
    if len(diag) < 4:
        return
    cs = 0.03 * short
    grid = defaultdict(list)
    for i, l in enumerate(diag):
        m = l.mid
        grid[(int(m[0] // cs), int(m[1] // cs))].append(i)
    for i, l in enumerate(diag):
        m = l.mid
        th = math.radians(l.angle)
        u = (math.cos(th), math.sin(th))
        n = (-u[1], u[0])
        cnt = 0
        gx, gy = int(m[0] // cs), int(m[1] // cs)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j == i:
                        continue
                    o = diag[j]
                    da = abs(o.angle - l.angle)
                    if min(da, 180 - da) > 4:
                        continue
                    v = (o.mid[0] - m[0], o.mid[1] - m[1])
                    perp = abs(v[0] * n[0] + v[1] * n[1])
                    along = abs(v[0] * u[0] + v[1] * u[1])
                    if 0.8 * thin_w < perp < cs and along < 0.6 * max(l.length, o.length):
                        cnt += 1
        if cnt >= 3:
            l.layer = "HATCH"


def _mark_stairs(lines, short, thin_w):
    for horiz in (True, False):
        cand = [l for l in lines if l.layer == "LINES" and (l.is_h if horiz else l.is_v)
                and 0.015 * short < l.length < 0.2 * short]
        if len(cand) < 5:
            continue
        iv = lambda l: tuple(sorted((l.p1[0], l.p2[0]) if horiz else (l.p1[1], l.p2[1])))  # noqa: E731
        fx = lambda l: l.p1[1] if horiz else l.p1[0]  # noqa: E731
        used = set()
        for l in sorted(cand, key=lambda x: -x.length):
            if id(l) in used:
                continue
            a0, a1 = iv(l)
            t = 0.12 * (a1 - a0)
            grp = [o for o in cand if id(o) not in used and abs(iv(o)[0] - a0) < t and abs(iv(o)[1] - a1) < t]
            if len(grp) < 5:
                continue
            grp.sort(key=fx)
            run = _longest_regular_run(grp, fx, max_gap=0.06 * short, min_gap=2 * thin_w)
            if len(run) >= 5:
                for o in run:
                    o.layer = "STAIRS"
                    used.add(id(o))


def _longest_regular_run(grp, fx, max_gap, min_gap):
    best: list = []
    for s in range(len(grp)):
        run = [grp[s]]
        gap0 = None
        for o in grp[s + 1:]:
            g = fx(o) - fx(run[-1])
            if g < min_gap:
                continue
            if g > max_gap:
                break
            if gap0 is None:
                gap0 = g
            elif abs(g - gap0) > 0.25 * gap0:
                break
            run.append(o)
        if len(run) > len(best):
            best = run
    return best


def _mark_dimensions(lines, texts, text_h) -> list[float]:
    cands: list[float] = []
    for t in texts:
        s = t.text.strip()
        if not _NUMERIC.match(s):
            continue
        cx, cy = t.center
        th = t.height
        horiz = t.rotation == 0
        best, bd = None, 2.4 * th
        for l in lines:
            if l.layer not in ("LINES", "DIMENSIONS"):
                continue
            if horiz and l.is_h:
                d = abs(l.p1[1] - cy)
                a0, a1 = sorted((l.p1[0], l.p2[0]))
                along = cx
            elif not horiz and l.is_v:
                d = abs(l.p1[0] - cx)
                a0, a1 = sorted((l.p1[1], l.p2[1]))
                along = cy
            else:
                continue
            if a0 - th <= along <= a1 + th and d < bd and l.length > 1.5 * th:
                best, bd = l, d
        if best is None:
            continue
        best.layer = "DIMENSIONS"
        t.layer = "DIMENSIONS"
        # Begrenzungen (Masshilfslinien / Schrägstriche) auf der Masslinie sammeln
        fixed = best.p1[1] if horiz else best.p1[0]
        mt = max(4.0, 0.4 * th)
        marks = []
        for o in lines:
            if o is best or o.layer in ("WALLS", "HATCH") or o.length > 8 * th:
                continue
            if horiz:
                ys = sorted((o.p1[1], o.p2[1]))
                if ys[0] - mt <= fixed <= ys[1] + mt and abs(o.p2[1] - o.p1[1]) > 0.3 * o.length:
                    x = o.p1[0] + (o.p2[0] - o.p1[0]) * ((fixed - o.p1[1]) / ((o.p2[1] - o.p1[1]) or 1e-9))
                    marks.append((x, o))
            else:
                xs = sorted((o.p1[0], o.p2[0]))
                if xs[0] - mt <= fixed <= xs[1] + mt and abs(o.p2[0] - o.p1[0]) > 0.3 * o.length:
                    y = o.p1[1] + (o.p2[1] - o.p1[1]) * ((fixed - o.p1[0]) / ((o.p2[0] - o.p1[0]) or 1e-9))
                    marks.append((y, o))
        for _, o in marks:
            o.layer = "DIMENSIONS"
        along = cx if horiz else cy
        left = [p for p, _ in marks if p < along]
        right = [p for p, _ in marks if p > along]
        val = parse_dimension_mm(s)
        if left and right and val:
            dist = min(right) - max(left)
            if dist > 3 * th:
                cands.append(val / dist)
    _extend_dimensions(lines, text_h)
    return cands


def _extend_dimensions(lines, text_h) -> None:
    """Massketten vervollständigen: Stücke auf derselben Masslinie (auch ohne lesbare Masszahl) und
    ihre Begrenzungsstriche gehören ebenfalls zur Bemassung."""
    tol = max(2.0, 0.25 * text_h)
    for _ in range(3):
        dims = [l for l in lines if l.layer == "DIMENSIONS" and (l.is_h or l.is_v) and l.length > 1.5 * text_h]
        if not dims:
            return
        changed = False
        for l in lines:
            if l.layer != "LINES" or not (l.is_h or l.is_v):
                continue
            for d in dims:
                if d.is_h and l.is_h and abs(l.p1[1] - d.p1[1]) <= tol:
                    a0, a1 = sorted((l.p1[0], l.p2[0]))
                    b0, b1 = sorted((d.p1[0], d.p2[0]))
                elif d.is_v and l.is_v and abs(l.p1[0] - d.p1[0]) <= tol:
                    a0, a1 = sorted((l.p1[1], l.p2[1]))
                    b0, b1 = sorted((d.p1[1], d.p2[1]))
                else:
                    continue
                if a0 <= b1 + 2 * text_h and b0 <= a1 + 2 * text_h:      # anschliessend oder überlappend
                    l.layer = "DIMENSIONS"
                    changed = True
                    break
        # Begrenzungen (kurze Querstriche, Schrägstriche) auf den Masslinien
        for l in lines:
            if l.layer != "LINES" or l.length > 6 * text_h:
                continue
            for d in dims:
                if d.is_h:
                    ys = sorted((l.p1[1], l.p2[1]))
                    xs = sorted((d.p1[0], d.p2[0]))
                    hit = ys[0] - tol <= d.p1[1] <= ys[1] + tol and xs[0] - tol <= l.mid[0] <= xs[1] + tol \
                        and abs(l.p2[1] - l.p1[1]) > 0.3 * l.length
                else:
                    xs = sorted((l.p1[0], l.p2[0]))
                    ys = sorted((d.p1[1], d.p2[1]))
                    hit = xs[0] - tol <= d.p1[0] <= xs[1] + tol and ys[0] - tol <= l.mid[1] <= ys[1] + tol \
                        and abs(l.p2[0] - l.p1[0]) > 0.3 * l.length
                if hit:
                    l.layer = "DIMENSIONS"
                    changed = True
                    break
        if not changed:
            return


def _mark_wall_pairs(lines, dmin, dmax, lmin, tol):
    pairs = []
    for horiz in (True, False):
        cand = [l for l in lines if l.layer in ("LINES", "WALLS") and (l.is_h if horiz else l.is_v) and l.length >= lmin]
        fx = (lambda l: l.p1[1]) if horiz else (lambda l: l.p1[0])
        iv = (lambda l: sorted((l.p1[0], l.p2[0]))) if horiz else (lambda l: sorted((l.p1[1], l.p2[1])))
        cand.sort(key=fx)
        for i, a in enumerate(cand):
            fa = fx(a)
            a0, a1 = iv(a)
            for b in cand[i + 1:]:
                fb = fx(b)
                if fb - fa > dmax:
                    break
                if fb - fa < dmin:
                    continue
                b0, b1 = iv(b)
                ov = min(a1, b1) - max(a0, b0)
                if ov >= 0.4 * min(a1 - a0, b1 - b0):
                    a.layer = b.layer = "WALLS"
                    pairs.append((horiz, fa, fb, a0, a1, b0, b1))
    # Wandköpfe (kurze Abschlusslinien zwischen zwei Wandlinien)
    for horiz, fa, fb, a0, a1, b0, b1 in pairs:
        ends = {a0, a1, b0, b1}
        for l in lines:
            if l.layer != "LINES":
                continue
            if horiz and l.is_v:
                f, (lo, hi) = l.p1[0], sorted((l.p1[1], l.p2[1]))
            elif not horiz and l.is_h:
                f, (lo, hi) = l.p1[1], sorted((l.p1[0], l.p2[0]))
            else:
                continue
            if abs(lo - fa) <= tol and abs(hi - fb) <= tol and any(abs(f - e) <= tol for e in ends):
                l.layer = "WALLS"


def _mark_doors(arcs, lines, short, mm_per_px):
    for a in arcs:
        if mm_per_px:
            ok_r = 450 <= a.radius * mm_per_px <= 1500
        else:
            ok_r = 0.01 * short <= a.radius <= 0.12 * short
        if not (55 <= a.sweep <= 125 and ok_r):
            a.layer = "LINES"
            continue
        a.layer = "DOORS"
        r = a.radius
        ends = (a.point_at(a.start), a.point_at(a.end))
        for l in lines:
            # Türblatt: beginnt im Drehpunkt, endet am offenen Bogenende. Kann vorher fälschlich
            # als Wandlinie gepaart worden sein (parallel zu einer nahen Wand) -> hier korrigieren.
            if l.layer == "DIMENSIONS" or not (0.7 * r <= l.length <= 1.7 * r):
                continue
            if _point_seg_dist(a.center, l.p1, l.p2) > max(4.0, 0.12 * r):
                continue
            far = l.p1 if math.dist(l.p1, a.center) > math.dist(l.p2, a.center) else l.p2
            if min(math.dist(far, e) for e in ends) <= 0.3 * r:
                l.layer = "DOORS"


def _point_seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _mark_windows(lines, wall_mask, shape, short, dmax, tol):
    h, w = shape
    raster = np.zeros((h, w), np.uint8) if wall_mask is None else wall_mask.copy()
    for l in lines:
        if l.layer == "WALLS":
            cv2.line(raster, tuple(map(int, l.p1)), tuple(map(int, l.p2)), 255, 3)
    if not np.count_nonzero(raster):
        return
    r = int(max(4, 2 * tol))
    raster = cv2.dilate(raster, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    def touches(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= x < w and 0 <= y < h and raster[y, x] > 0

    cand = [l for l in lines if l.layer == "LINES" and _axis(l) and 0.012 * short <= l.length <= 0.12 * short
            and touches(l.p1) and touches(l.p2)]
    for horiz in (True, False):
        c = [l for l in cand if l.is_h == horiz]
        fx = (lambda l: l.p1[1]) if horiz else (lambda l: l.p1[0])
        iv = (lambda l: sorted((l.p1[0], l.p2[0]))) if horiz else (lambda l: sorted((l.p1[1], l.p2[1])))
        c.sort(key=fx)
        for i, a in enumerate(c):
            a0, a1 = iv(a)
            for b in c[i + 1:]:
                if fx(b) - fx(a) > dmax:
                    break
                b0, b1 = iv(b)
                if abs(a0 - b0) <= 2 * tol and abs(a1 - b1) <= 2 * tol:
                    a.layer = b.layer = "WINDOWS"

"""Geometrie-Bereinigung: Achsen ausrichten, Teilstücke zusammenführen,
Ecken und T-Stösse schliessen, Kleinstteile entfernen."""
from __future__ import annotations

import bisect
import math
from collections import defaultdict

import numpy as np

from .geometry import Line


def snap_axis(lines: list[Line], tol_deg: float = 3.0) -> None:
    for l in lines:
        a = l.angle
        if a < tol_deg or a > 180 - tol_deg:
            y = (l.p1[1] + l.p2[1]) / 2
            l.p1, l.p2 = (l.p1[0], y), (l.p2[0], y)
        elif abs(a - 90) < tol_deg:
            x = (l.p1[0] + l.p2[0]) / 2
            l.p1, l.p2 = (x, l.p1[1]), (x, l.p2[1])


def _frame(l: Line):
    th = math.radians(l.angle)
    u = (math.cos(th), math.sin(th))
    n = (-u[1], u[0])
    rho = l.p1[0] * n[0] + l.p1[1] * n[1]
    t1 = l.p1[0] * u[0] + l.p1[1] * u[1]
    t2 = l.p2[0] * u[0] + l.p2[1] * u[1]
    return u, n, rho, min(t1, t2), max(t1, t2)


def merge_collinear(lines: list[Line], dist_tol: float, gap_tol: float, ang_tol: float = 2.0) -> list[Line]:
    """Fasst kollineare, sich überlappende oder knapp getrennte Teilstücke zusammen."""
    for _ in range(3):
        before = len(lines)
        lines = _merge_pass(lines, dist_tol, gap_tol, ang_tol)
        if len(lines) == before:
            break
    return lines


def _merge_pass(lines, dist_tol, gap_tol, ang_tol):
    ab, rb = 3.0, max(1.0, 2 * dist_tol)
    info = [_frame(l) for l in lines]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (u, n, rho, t0, t1) in enumerate(info):
        buckets[(int(lines[i].angle // ab), int(rho // rb))].append(i)

    order = sorted(range(len(lines)), key=lambda i: -lines[i].length)
    used = [False] * len(lines)
    out: list[Line] = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        base = lines[i]
        u, n, rho, t0, t1 = info[i]
        ka, kr = int(base.angle // ab), int(rho // rb)
        members = [i]
        changed = True
        while changed:
            changed = False
            for da in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    for j in buckets.get(((ka + da) % 60, kr + dr), ()):
                        if used[j]:
                            continue
                        lj = lines[j]
                        dang = abs(lj.angle - base.angle)
                        if min(dang, 180 - dang) > ang_tol:
                            continue
                        d1 = abs(lj.p1[0] * n[0] + lj.p1[1] * n[1] - rho)
                        d2 = abs(lj.p2[0] * n[0] + lj.p2[1] * n[1] - rho)
                        if d1 > dist_tol or d2 > dist_tol:
                            continue
                        s1 = lj.p1[0] * u[0] + lj.p1[1] * u[1]
                        s2 = lj.p2[0] * u[0] + lj.p2[1] * u[1]
                        lo, hi = min(s1, s2), max(s1, s2)
                        if hi < t0 - gap_tol or lo > t1 + gap_tol:
                            continue
                        used[j] = True
                        members.append(j)
                        t0, t1 = min(t0, lo), max(t1, hi)
                        changed = True
        if len(members) == 1:
            out.append(base)
            continue
        wts = np.array([lines[m].length for m in members]) + 1e-6
        if base.is_h:
            y = float(np.average([lines[m].p1[1] for m in members], weights=wts))
            p1, p2 = (t0, y), (t1, y)
        elif base.is_v:
            x = float(np.average([lines[m].p1[0] for m in members], weights=wts))
            p1, p2 = (x, t0), (x, t1)
        else:
            p1 = (u[0] * t0 + n[0] * rho, u[1] * t0 + n[1] * rho)
            p2 = (u[0] * t1 + n[0] * rho, u[1] * t1 + n[1] * rho)
        width = float(np.average([lines[m].width for m in members], weights=wts))
        out.append(Line(p1, p2, base.layer, width, base.comp))
    return out


def connect_orthogonal(lines: list[Line], tol: float) -> None:
    """Verlängert/kürzt Achsenlinien, damit Ecken und T-Stösse exakt schliessen."""
    H = [l for l in lines if l.is_h]
    V = [l for l in lines if l.is_v]
    _snap_ends(H, V, tol, horizontal=True)
    _snap_ends(V, H, tol, horizontal=False)


def _snap_ends(movers: list[Line], targets: list[Line], tol: float, horizontal: bool):
    if not targets:
        return
    # targets nach ihrer festen Koordinate sortieren
    key = (lambda l: l.p1[0]) if horizontal else (lambda l: l.p1[1])
    targets = sorted(targets, key=key)
    keys = [key(t) for t in targets]
    for m in movers:
        pts = [m.p1, m.p2]
        for k in range(2):
            ex, ey = pts[k]
            along, fixed = (ex, ey) if horizontal else (ey, ex)
            lo = bisect.bisect_left(keys, along - tol)
            hi = bisect.bisect_right(keys, along + tol)
            best, bd = None, tol + 1
            for t in targets[lo:hi]:
                tf = key(t)
                a0, a1 = (sorted((t.p1[1], t.p2[1])) if horizontal else sorted((t.p1[0], t.p2[0])))
                if a0 - tol <= fixed <= a1 + tol:
                    d = abs(tf - along)
                    if d < bd:
                        best, bd = t, d
            if best is not None:
                tf = key(best)
                other = pts[1 - k][0] if horizontal else pts[1 - k][1]
                if abs(tf - other) < 1.0:
                    continue  # würde Linie auf Länge 0 kürzen
                pts[k] = (tf, ey) if horizontal else (ex, tf)
                # Ziel ggf. bis zum Stosspunkt verlängern (L-Ecke)
                a_fixed = fixed
                if horizontal:
                    y0, y1 = sorted((best.p1[1], best.p2[1]))
                    if a_fixed < y0 or a_fixed > y1:
                        _extend(best, (tf, a_fixed), vertical=True)
                else:
                    x0, x1 = sorted((best.p1[0], best.p2[0]))
                    if a_fixed < x0 or a_fixed > x1:
                        _extend(best, (a_fixed, tf), vertical=False)
        m.p1, m.p2 = pts[0], pts[1]


def _extend(l: Line, p, vertical: bool):
    if vertical:
        if abs(l.p1[1] - p[1]) < abs(l.p2[1] - p[1]):
            l.p1 = (l.p1[0], p[1])
        else:
            l.p2 = (l.p2[0], p[1])
    else:
        if abs(l.p1[0] - p[0]) < abs(l.p2[0] - p[0]):
            l.p1 = (p[0], l.p1[1])
        else:
            l.p2 = (p[0], l.p2[1])


def snap_free_ends(lines: list[Line], tol: float) -> None:
    """Schräge Linien an nahe Endpunkte anderer Linien heften."""
    grid: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    cs = max(1.0, tol)
    for i, l in enumerate(lines):
        for k, p in enumerate((l.p1, l.p2)):
            grid[(int(p[0] // cs), int(p[1] // cs))].append((i, k))
    for i, l in enumerate(lines):
        if l.is_h or l.is_v:
            continue
        pts = [l.p1, l.p2]
        for k in range(2):
            p = pts[k]
            gx, gy = int(p[0] // cs), int(p[1] // cs)
            best, bd, axis_pref = None, tol, False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j, kk in grid.get((gx + dx, gy + dy), ()):
                        if j == i:
                            continue
                        q = lines[j].p1 if kk == 0 else lines[j].p2
                        d = math.hypot(q[0] - p[0], q[1] - p[1])
                        is_axis = lines[j].is_h or lines[j].is_v
                        if d < bd or (is_axis and not axis_pref and d <= tol):
                            best, bd, axis_pref = q, d, is_axis
            if best is not None:
                pts[k] = best
        l.p1, l.p2 = pts[0], pts[1]


def drop_short(lines: list[Line], min_len: float) -> list[Line]:
    return [l for l in lines if l.length >= min_len]


def drop_bridges(lines: list[Line], max_len: float, tol: float) -> list[Line]:
    """Entfernt kurze schräge Stummel, deren beide Enden auf Achsenlinien liegen
    (typische Reste von wackeligen Handlinien)."""
    axis = [l for l in lines if l.is_h or l.is_v]

    def on_axis(p, own):
        for a in axis:
            if a is own:
                continue
            if a.is_h:
                x0, x1 = sorted((a.p1[0], a.p2[0]))
                if abs(p[1] - a.p1[1]) <= tol and x0 - tol <= p[0] <= x1 + tol:
                    return True
            else:
                y0, y1 = sorted((a.p1[1], a.p2[1]))
                if abs(p[0] - a.p1[0]) <= tol and y0 - tol <= p[1] <= y1 + tol:
                    return True
        return False

    out = []
    for l in lines:
        if not (l.is_h or l.is_v) and l.length < max_len and on_axis(l.p1, l) and on_axis(l.p2, l):
            continue
        out.append(l)
    return out

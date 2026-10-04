"""1:1-Nachzeichnen der Wände aus dem Scan.

Das Netz sagt, *wo* Wände, Fenster und Türen liegen. Die Geometrie selbst kommt aus der
gezeichneten Tinte: Wandumrisse inkl. schräger Leibungen, Nischen, Vorsprünge, Schraffur- oder
Füllflächen werden als Fläche geschlossen und als Polygon mit exakten Ecken ausgegeben.

Zusätzlich werden dünne Wände, die nur als Doppellinie gezeichnet sind (Leichtbau 10–15 cm),
direkt aus Linienpaaren erkannt – auch wenn das Netz sie übersehen hat.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .semantic import DOOR, WALL, WINDOW


# =============================================================================== dünne Wände
def double_line_walls(binary: np.ndarray, labels: np.ndarray, t: float, _pass: int = 0,
                      _anchor: np.ndarray | None = None) -> np.ndarray:
    """Wände, die nur als zwei parallele Linien gezeichnet sind (Leichtbau), als Maske (achsparallel).

    Zwei lange Linien im Abstand einer dünnen Wand werden quer geschlossen; der Streifen muss lang
    sein, eine plausible Stärke haben, an erkannte Wände anschliessen (an seinen Enden) und darf nicht
    bloss parallel neben einer erkannten Wand laufen (Küchenzeile, Fensterrahmen).
    """
    H, W = binary.shape
    ink = (binary > 0).astype(np.uint8) * 255
    known = labels > 0
    wall_lab = (labels == WALL).astype(np.uint8)
    opening = (labels == WINDOW) | (labels == DOOR)
    gap = int(max(3, round(0.55 * t)))
    long_ = int(max(25, 3.0 * t)) if _pass == 0 else int(max(12, 1.0 * t))
    out = np.zeros((H, W), np.uint8)
    side = int(max(4, round(0.6 * t)))
    off = int(max(2, round(0.2 * t)))
    for horiz in (True, False):
        k_line = np.ones((1, long_), np.uint8) if horiz else np.ones((long_, 1), np.uint8)
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k_line)
        k_close = np.ones((gap, 1), np.uint8) if horiz else np.ones((1, gap), np.uint8)
        merged = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, k_close)
        thick = cv2.morphologyEx(cv2.subtract(merged, lines), cv2.MORPH_OPEN, k_line)
        if not np.any(thick):
            continue
        cand = cv2.bitwise_or(cv2.dilate(thick, np.ones((3, 3), np.uint8)) & merged, thick)
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            length, width = (w, h) if horiz else (h, w)
            if length < long_ or width < 0.18 * t or width > 1.1 * t:
                continue
            comp = lab[y:y + h, x:x + w] == i
            if comp.mean() < 0.55:
                continue
            if opening[y:y + h, x:x + w][comp].mean() > 0.5:
                # Fensterrahmen (schmaler Streifen in breiter Öffnung) oder als Fenster verkannte Leichtbauwand?
                R = int(1.2 * t)
                if horiz:
                    win = opening[max(0, y - R):min(H, y + h + R), x:x + w]
                    blob_w = np.median(win.sum(axis=0))
                else:
                    win = opening[y:y + h, max(0, x - R):min(W, x + w + R)]
                    blob_w = np.median(win.sum(axis=1))
                if blob_w > 1.6 * width:
                    continue
                # Fenster ist meist mit Glaslinie(n) zwischen den Wandlinien gezeichnet -> mehr als 2 Linien
                sub = ink[y:y + h, x:x + w] > 0
                prof = sub.mean(axis=1) if horiz else sub.mean(axis=0)
                peaks = int(np.sum(np.diff((prof > 0.5).astype(np.int8)) == 1) + (prof[0] > 0.5))
                if peaks >= 3:
                    continue
            # läuft parallel neben einer erkannten Wand? (Mittelteil, beidseits abtasten)
            par = 0
            for f in np.linspace(0.2, 0.8, 7):
                if horiz:
                    xx = int(x + f * w)
                    above = wall_lab[max(0, y - side):max(0, y - off), xx].any()
                    below = wall_lab[min(H, y + h + off):min(H, y + h + side), xx].any()
                else:
                    yy = int(y + f * h)
                    above = wall_lab[yy, max(0, x - side):max(0, x - off)].any()
                    below = wall_lab[yy, min(W, x + w + off):min(W, x + w + side)].any()
                par += above or below
            if par >= 4:
                continue
            out[y:y + h, x:x + w][comp] = 255
    if not np.any(out):
        return out
    # nur Streifen, die an die erkannte Wandstruktur anschliessen
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    anchor = (known * 255).astype(np.uint8) if _anchor is None else _anchor
    near = cv2.dilate(anchor, np.ones((int(max(3, 0.6 * t)) | 1,) * 2, np.uint8))
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        if np.any(near[lab == i]):
            keep[i] = 255
    res = keep[lab]
    if _pass == 0 and np.any(res):
        # kurze Wandstücke (z.B. neben Türen), die an eben gefundene Wände anschliessen
        lab2 = labels.copy()
        lab2[res > 0] = WALL
        res = cv2.bitwise_or(res, double_line_walls(binary, lab2, t, 1, res))
    return res


# =============================================================================== Wandflächen
def stroke_width(binary: np.ndarray, region: np.ndarray, t: float = 20.0) -> float:
    """Typische Strichstärke (px) der Wand-Umrisslinien.

    Lange, achsparallele Linien in Wandnähe (Schraffur fällt weg) – häufigste Querschnittslänge.
    """
    ink = ((binary > 0) * 255).astype(np.uint8)
    near = cv2.dilate(region, np.ones((7, 7), np.uint8)) > 0
    L = int(max(15, 2.0 * t))
    runs = []
    for k, axis in ((np.ones((1, L), np.uint8), 0), (np.ones((L, 1), np.uint8), 1)):
        m = (cv2.morphologyEx(ink, cv2.MORPH_OPEN, k) > 0) & near
        a = m if axis == 0 else m.T            # Querschnitt: senkrecht zur Linie zählen
        a = a.astype(np.int8)
        d = np.diff(np.pad(a, ((1, 1), (0, 0))), axis=0)
        st = np.nonzero(d.T == 1)
        en = np.nonzero(d.T == -1)
        if st[0].size and st[0].size == en[0].size:
            runs.append(en[1] - st[1])
    if not runs:
        return 2.0
    r = np.concatenate(runs)
    r = r[(r >= 1) & (r <= max(3, 0.5 * t))]
    if r.size < 20:
        return 2.0
    h = np.bincount(r).astype(float)
    h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    # kleinste deutlich häufige Breite (massive Wände/Füllungen sind breiter und zählen nicht)
    cand = np.nonzero(h >= 0.5 * h.max())[0]
    i = int(cand[0])
    while i + 1 < len(h) and h[i + 1] > h[i]:
        i += 1
    return float(np.clip(i, 1, 14))


def wall_material(binary: np.ndarray, walls: np.ndarray, openings: np.ndarray, t: float, sw: float = 3.0,
                  rects: list | None = None) -> np.ndarray:
    """Wandfläche aus der Tinte: Umrisslinien + Füllung/Schraffur bzw. Raum zwischen den Umrisslinien.

    ``walls``: bereinigte Wandmaske (wo Wand ist), ``openings``: Öffnungsrechtecke (dort keine Wand).
    Ergebnis folgt den gezeichneten Kanten (schräge Leibungen, Nischen, Vorsprünge) statt Rechtecken.
    """
    H, W = walls.shape
    ink = (binary > 0).astype(np.uint8) * 255
    r = int(max(2, round(0.3 * t)))
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    rz = int(max(1, round(0.45 * sw)))
    zone = cv2.dilate(cv2.bitwise_or(walls, openings), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rz + 1,) * 2))
    ink_z = cv2.bitwise_and(ink, zone)
    # kleine Lücken in Umrisslinien schliessen (Scan), damit Innenflächen geschlossen sind
    kc = int(max(3, round(0.2 * t))) | 1
    ink_c = cv2.morphologyEx(ink_z, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    inv = cv2.bitwise_not(ink_c)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    filled = ink_z.copy()
    big = 40.0 * t * t
    wl = walls > 0
    op = openings > 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        m = lab[y:y + h, x:x + w] == i
        cw = wl[y:y + h, x:x + w][m].mean()
        co = op[y:y + h, x:x + w][m].mean()
        # Wandfläche; Flächen, die über den Öffnungsrand hinaus in die Wand reichen (Schraffur bis zur
        # gezeichneten Leibung), gehören ebenfalls dazu – das Innere der Öffnung (Rahmen, Glas) nicht
        if cw >= 0.6 or (cw + co >= 0.9 and cw >= 0.3):
            filled[y:y + h, x:x + w][m] = 255
    # Brücken über kleine Lücken der Umrisslinie (nur innerhalb der Wand)
    filled = cv2.bitwise_or(filled, cv2.bitwise_and(cv2.subtract(ink_c, ink_z),
                                                    cv2.erode(cv2.bitwise_or(walls, openings), np.ones((3, 3), np.uint8))))
    # wo die Tinte keine Wandfläche ergibt (Lücken, sehr blasse Linien), die erkannte Wand übernehmen
    grow = int(max(1, round(sw))) * 2 + 1
    lost = cv2.bitwise_and(walls, cv2.bitwise_not(cv2.dilate(filled, np.ones((grow, grow), np.uint8))))
    lost = cv2.morphologyEx(lost, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(lost, connectivity=8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) >= 1.5 * t and a >= 0.5 * t * min(w, h):
            m = lab[y:y + h, x:x + w] == i
            filled[y:y + h, x:x + w][m] = 255
            # bis an die Tinte heran schliessen
    filled = cv2.bitwise_or(filled, cv2.bitwise_and(cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((grow, grow), np.uint8)), walls))
    # Wandstücke, deren Umriss nicht geschlossen nachgezeichnet werden konnte (verblasste, unterbrochene
    # Linien): dort gilt die erkannte Wandgeometrie
    if rects:
        for x0, y0, x1, y1, _ in rects:
            xa, ya, xb, yb = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
            xa, ya, xb, yb = max(0, xa), max(0, ya), min(W, xb), min(H, yb)
            if xb - xa < 2 or yb - ya < 2:
                continue
            wsub = walls[ya:yb, xa:xb] > 0
            if wsub.sum() < 4:
                continue
            cov = (filled[ya:yb, xa:xb] > 0)[wsub].mean()
            if cov < 0.85:
                sub = filled[ya:yb, xa:xb]
                sub[wsub] = 255
    # nur zusammenhängende Teile, die substanziell Wand sind
    # In Öffnungen bleibt nur, was zur Wandmasse gehört (Leibung, Schräge, Anschlag); Rahmen- und
    # Glaslinien sind dünner und fallen weg
    if np.any(openings):
        k2 = int(max(5, round(0.35 * t))) | 1
        massive = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k2, k2), np.uint8))
        m_in = cv2.bitwise_and(massive, openings)
        # was quer durch die ganze Öffnung läuft, ist das Fenster selbst (verschwommene Rahmenlinien)
        n_o, lab_o, st_o, _ = cv2.connectedComponentsWithStats(openings, connectivity=4)
        n_m, lab_m, st_m, _ = cv2.connectedComponentsWithStats(m_in, connectivity=8)
        for j in range(1, n_m):
            x, y, w, h, _a = st_m[j]
            oi = lab_o[y + h // 2, x + w // 2] if lab_o[y + h // 2, x + w // 2] else lab_o[y, x]
            if not oi:
                continue
            ow, oh = st_o[oi, 2], st_o[oi, 3]
            along = w if ow >= oh else h
            if along > 0.5 * max(ow, oh):
                m_in[lab_m == j] = 0
        filled = cv2.bitwise_or(cv2.bitwise_and(filled, cv2.bitwise_not(openings)), m_in)
    # einzelne Striche (Fensterbank, Masslinien-Anschlüsse, Möbelkanten) sind keine Wandfläche
    k = int(max(3, round(1.6 * sw))) | 1
    inner = cv2.erode(walls, np.ones((3, 3), np.uint8))
    filled = cv2.bitwise_or(cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)),
                            cv2.bitwise_and(filled, inner))     # dünne, als Wand erkannte Striche bleiben
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 1.0 * t * t:
            continue
        m = lab[y:y + h, x:x + w] == i
        if wl[y:y + h, x:x + w][m].mean() >= 0.5:
            keep[i] = 255
    return keep[lab]


# =============================================================================== Polygone
def _snap_dir(dx: float, dy: float, tol_deg: float = 3.0, short: float = 0.0):
    ln = math.hypot(dx, dy)
    tol = max(tol_deg, math.degrees(math.atan2(1.5, max(ln, 1e-6))))   # < 1.5 px Versatz -> gerade
    tol_axis = max(tol, 12.0) if ln < short else tol   # kurze Stücke (Leibungen, Wandenden): rechtwinklig
    a = math.degrees(math.atan2(dy, dx)) % 180.0
    for target in (0.0, 90.0, 180.0, 45.0, 135.0):
        if abs(a - target) <= (tol_axis if target in (0.0, 90.0, 180.0) else min(tol, 3.0)):
            r = math.radians(target)
            return math.cos(r), math.sin(r), True
    n = ln or 1.0
    return dx / n, dy / n, False


def _intersect(p, d, q, e):
    """Schnitt der Geraden p + s·d und q + u·e (oder None bei Parallelen)."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-6:
        return None
    s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + s * d[0], p[1] + s * d[1])


def _proj(p, d, x):
    """Projektion des Punktes x auf die Gerade p + s·d."""
    s_ = (x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1]
    return (p[0] + s_ * d[0], p[1] + s_ * d[1])


def clean_ring(pts: np.ndarray, eps: float, min_edge: float, drop_len: float | None = None,
               short: float = 0.0, jog: float = 0.0) -> list[tuple[float, float]]:
    """Polygon vereinfachen, Kanten auf 0/45/90° einrasten (andere Winkel bleiben), Ecken exakt schneiden.

    Jede Kante bleibt lokal: Ecken entstehen als Schnitt benachbarter Kanten nur, wenn der Schnittpunkt
    nahe an der ursprünglichen Ecke liegt; sonst werden die Kanten mit einer kurzen Verbindung geschlossen.
    """
    if len(pts) < 3:
        return []
    drop_len = drop_len if drop_len is not None else 2.0 * min_edge + 2.0
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(float)
    if len(approx) < 3:
        return []
    n = len(approx)
    # Kante: [Punkt, Richtung, Länge, Start, Ende]
    lines = []
    for i in range(n):
        a, b = approx[i], approx[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ux, uy, snapped = _snap_dir(dx, dy, short=short)
        if (not snapped or (abs(ux) > 1e-6 and abs(uy) > 1e-6)) and L < drop_len:
            continue            # Eckabrundung aus der Rasterung, keine gezeichnete Schräge
        lines.append([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), (ux, uy), L, tuple(a), tuple(b)])
    if len(lines) < 3:
        return []

    def offset(p, q):
        return abs((q[0][0] - p[0][0]) * (-p[1][1]) + (q[0][1] - p[0][1]) * p[1][0])

    # kollineare Nachbarn zusammenfassen (gleiche Richtung, kaum Versatz)
    merged = [lines[0]]
    for ln in lines[1:]:
        p = merged[-1]
        if abs(p[1][0] * ln[1][1] - p[1][1] * ln[1][0]) < 1e-3 and p[1][0] * ln[1][0] + p[1][1] * ln[1][1] > 0 \
                and offset(p, ln) < max(1.0, eps, jog, min(0.025 * (p[2] + ln[2]), 2.0 * jog)):
            w = p[2] + ln[2]
            merged[-1] = [((p[0][0] * p[2] + ln[0][0] * ln[2]) / w, (p[0][1] * p[2] + ln[0][1] * ln[2]) / w),
                          p[1] if p[2] >= ln[2] else ln[1], w, p[3], ln[4]]
        else:
            merged.append(ln)
    if len(merged) >= 2:
        p, ln = merged[-1], merged[0]
        if abs(p[1][0] * ln[1][1] - p[1][1] * ln[1][0]) < 1e-3 and p[1][0] * ln[1][0] + p[1][1] * ln[1][1] > 0 \
                and offset(p, ln) < max(1.0, eps, jog, min(0.025 * (p[2] + ln[2]), 2.0 * jog)):
            w = p[2] + ln[2]
            merged[0] = [((p[0][0] * p[2] + ln[0][0] * ln[2]) / w, (p[0][1] * p[2] + ln[0][1] * ln[2]) / w),
                         p[1] if p[2] >= ln[2] else ln[1], w, p[3], ln[4]]
            merged.pop()

    def lim(p, q):
        return max(1.0, eps, jog, min(0.025 * (p[2] + q[2]), 2.0 * jog))

    # kleine Stufen (Kante – kurzer Versatz – parallele Kante) glätten
    changed = True
    while changed and len(merged) > 4:
        changed = False
        m = len(merged)
        for i in range(m):
            a, j, b = merged[i - 1], merged[i], merged[(i + 1) % m]
            if abs(a[1][0] * b[1][1] - a[1][1] * b[1][0]) < 1e-3 and a[1][0] * b[1][0] + a[1][1] * b[1][1] > 0:
                L = lim(a, b)
                if j[2] <= L + 1.0 and offset(a, b) < L:
                    w = a[2] + b[2]
                    new = [((a[0][0] * a[2] + b[0][0] * b[2]) / w, (a[0][1] * a[2] + b[0][1] * b[2]) / w),
                           a[1] if a[2] >= b[2] else b[1], w, a[3], b[4]]
                    ia, ib = (i - 1) % m, (i + 1) % m
                    keep = [x for k, x in enumerate(merged) if k not in (ia, i, ib)]
                    # neue Kante an die Stelle von a setzen
                    pos = sum(1 for k in range(m) if k < ia and k not in (i, ib))
                    keep.insert(pos, new)
                    merged = keep
                    changed = True
                    break
    m = len(merged)
    if m < 3:
        return []
    far = max(3.0 * eps, drop_len, 2.0 * min_edge)
    out = []
    for i in range(m):
        p, q = merged[i - 1], merged[i]
        ref = ((p[4][0] + q[3][0]) / 2, (p[4][1] + q[3][1]) / 2)
        x = _intersect(p[0], p[1], q[0], q[1])
        if x is not None and math.dist(x, ref) <= far:
            out.append((float(x[0]), float(x[1])))
        else:                                   # Versatz/Stufe: beide Kanten bis zur Ecke, kurze Verbindung
            a_ = _proj(p[0], p[1], p[4])
            b_ = _proj(q[0], q[1], q[3])
            out.append((float(a_[0]), float(a_[1])))
            if math.dist(a_, b_) > 0.5:
                out.append((float(b_[0]), float(b_[1])))
    # doppelte/sehr nahe Punkte entfernen
    clean = []
    for pt in out:
        if clean and math.dist(pt, clean[-1]) < 0.5:
            continue
        clean.append(pt)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) < 0.5:
        clean.pop()
    return clean if len(clean) >= 3 else []


def _ring_or_fallback(c, eps, min_edge, drop, short=0.0, jog=0.0):
    r = clean_ring(c, eps, min_edge, drop, short, jog)
    if r and abs(_area(r)) > 0.5 * abs(cv2.contourArea(c.astype(np.float32))):
        return r
    approx = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx] if len(approx) >= 3 else []


def _area(r):
    a = 0.0
    for i in range(len(r)):
        x0, y0 = r[i - 1]
        x1, y1 = r[i]
        a += x0 * y1 - x1 * y0
    return a / 2


def wall_rings(material: np.ndarray, sw: float, t: float):
    """Polygone (Aussenring + Löcher) der Wandflächen; Kanten in Strichmitte (halbe Strichstärke innen)."""
    k = int(round(sw / 2))
    m = material
    if k >= 1:
        ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
        m = cv2.erode(material, ker)
        # dünne Wände, die nur aus einem kräftigen Strich bestehen, behalten ihre volle Breite
        thin = cv2.subtract(material, cv2.dilate(m, ker))
        thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
        for i in range(1, n):
            if max(st[i, 2], st[i, 3]) >= max(6, 1.5 * t) and st[i, 4] >= 2 * sw * sw:
                m[lab == i] = 255
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    regions = []
    if hier is None:
        return regions
    hier = hier[0]
    eps = max(0.8, 0.35 * sw)
    min_edge = max(1.0, 0.5 * sw)
    drop = max(2.0 * sw + 2.0, 0.15 * t)
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue
        outer = _ring_or_fallback(c.reshape(-1, 2) + 0.5, eps, min_edge, drop, 0.8 * t, 0.07 * t)
        if not outer:
            continue
        rings = [outer]
        ch = hier[i][2]
        while ch != -1:
            if cv2.contourArea(contours[ch]) > 0.8 * t * t:
                r = _ring_or_fallback(contours[ch].reshape(-1, 2) + 0.5, eps, min_edge, drop, 0.8 * t, 0.07 * t)
                if r:
                    rings.append(r)
            ch = hier[ch][0]
        regions.append(rings)
    return regions


def wobbly(regions, t: float) -> float:
    """Unruhe der Umrisse: Ecken je Wandstärke Umfang. Saubere Pläne haben lange gerade Kanten
    (wenige Ecken, auch mit Leibungen), freihändige Skizzen viele kleine Knicke und Stufen."""
    n = 0
    tot = 0.0
    for rings in regions:
        for r in rings:
            n += len(r)
            for i in range(len(r)):
                (x0, y0), (x1, y1) = r[i - 1], r[i]
                tot += math.hypot(x1 - x0, y1 - y0)
    return n * t / tot if tot else 0.0

"""Pläne mit schwarz gefüllten Wänden (Poché) – die häufigste CAD-Darstellung.

Hier ist die Tinte selbst die exakte Wandgeometrie: Alles, was dunkel *und* breiter als jede Linie
oder Schrift ist, ist Wand. Öffnungen sind Unterbrüche in dieser Wandfläche; ob Tür oder Fenster,
entscheidet der gezeichnete Türbogen bzw. die Fensterlinien im Unterbruch.

Liefert eine Label-Karte (wie das Netz), damit die übrige Rekonstruktion unverändert bleibt.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .semantic import DOOR, WALL, WINDOW
from .walls import decompose, median_thickness


def poche_mask(gray: np.ndarray, t: float, wall_hint: np.ndarray | None = None) -> np.ndarray:
    """Massiv dunkle Flächen, die breiter als Linien/Schrift sind (Wandfüllung).

    Schwelle in der Mitte zwischen Wandfüllung und Papier -> Kante liegt auf der halben Helligkeit
    (auch bei unscharfen Scans an der richtigen Stelle)."""
    paper = float(np.percentile(gray, 90))
    thr = min(110.0, 0.45 * paper)
    if wall_hint is not None and np.count_nonzero(wall_hint) > 100:
        level = float(np.percentile(gray[wall_hint > 0], 30))
        if level < 0.5 * paper:
            thr = (level + paper) / 2
    dark = ((gray < thr) * 255).astype(np.uint8)
    k = int(max(5, round(0.3 * t))) | 1
    po = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(po, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) >= 2 * t and a >= 1.5 * t * t:
            keep[i] = 255
    return keep[lab]


def is_poche_plan(po: np.ndarray, labels: np.ndarray) -> bool:
    """Sind die Wände als gefüllte Flächen gezeichnet? (Poché deckt die erkannten Wände weitgehend ab)"""
    wall = labels == WALL
    a_po, a_w = np.count_nonzero(po), np.count_nonzero(wall)
    if a_po < 0.4 * max(1, a_w) or a_w == 0:
        return False
    inter = np.count_nonzero(wall & (po > 0))
    return inter >= 0.7 * a_w and inter >= 0.6 * a_po


def _ray_gap(po, x, y, dx, dy, half, t, max_len):
    """Von einem Wandende aus entlang der Wandachse bis zur nächsten Wandfläche laufen.
    Rückgabe: Länge des freien Stücks oder None (Wand läuft weiter / nichts gefunden)."""
    H, W = po.shape
    px, py = -dy, dx
    offs = np.linspace(-1.3 * half, 1.3 * half, 15)
    free_seen = False
    for s in range(1, int(max_len) + 1):
        cx, cy = x + dx * s, y + dy * s
        col = []
        for o in offs:
            qx, qy = int(round(cx + px * o)), int(round(cy + py * o))
            if not (0 <= qx < W and 0 <= qy < H):
                return None
            col.append(po[qy, qx] > 0)
        col = np.array(col)
        # längster zusammenhängender Wandabschnitt quer zur Achse
        run = best = 0
        for v in col:
            run = run + 1 if v else 0
            best = max(best, run)
        solid = best >= 0.45 * len(col)          # ~ Wandstärke getroffen (auch leicht versetzt)
        if not col.any():
            free_seen = True
        if solid:
            return s if free_seen else None
    return None


def gap_openings(po: np.ndarray, t: float):
    """Unterbrüche in der Wandfläche: (x0, y0, x1, y1, orient) je Öffnung (Wandstärke quer)."""
    rects, _ = decompose(po, max(3 * t, 12), t_est=t)
    H, W = po.shape
    out = []
    for x0, y0, x1, y1, o in rects:
        th = (y1 - y0) if o == "h" else (x1 - x0)
        if th < 0.25 * t:
            continue
        half = th / 2
        if o == "h":
            c = (y0 + y1) / 2
            ends = [(x1, c, 1, 0), (x0, c, -1, 0)]
        else:
            c = (x0 + x1) / 2
            ends = [(c, y1, 0, 1), (c, y0, 0, -1)]
        for ex, ey, dx, dy in ends:
            g = _ray_gap(po, ex, ey, dx, dy, half, t, 10 * t)
            if g is None or g < 2:
                continue
            if o == "h":
                a, b = (ex, ex + g) if dx > 0 else (ex - g, ex)
                out.append((a, c - half, b, c + half, "h"))
            else:
                a, b = (ey, ey + g) if dy > 0 else (ey - g, ey)
                out.append((c - half, a, c + half, b, "v"))
    # doppelt gefundene (von beiden Enden) zusammenfassen
    res = []
    for r in out:
        dup = False
        for i, q in enumerate(res):
            if q[4] == r[4] and abs(q[0] - r[0]) < 0.5 * t and abs(q[1] - r[1]) < 0.5 * t \
                    and abs(q[2] - r[2]) < 0.5 * t and abs(q[3] - r[3]) < 0.5 * t:
                dup = True
                break
        if not dup:
            res.append(r)
    return res


def _merge_gaps(gaps):
    """Parallele Teilstücke derselben Öffnung (Wand in zwei Streifen zerlegt) vereinigen."""
    gaps = [list(g) for g in gaps]
    changed = True
    while changed:
        changed = False
        for i in range(len(gaps)):
            for j in range(i + 1, len(gaps)):
                a, b = gaps[i], gaps[j]
                if a[4] != b[4]:
                    continue
                if a[4] == "h":
                    ov = min(a[2], b[2]) - max(a[0], b[0])
                    ln = min(a[2] - a[0], b[2] - b[0])
                    touch = max(a[1], b[1]) - min(a[3], b[3]) <= 3
                else:
                    ov = min(a[3], b[3]) - max(a[1], b[1])
                    ln = min(a[3] - a[1], b[3] - b[1])
                    touch = max(a[0], b[0]) - min(a[2], b[2]) <= 3
                if ov >= 0.5 * ln and touch:
                    gaps[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4]]
                    del gaps[j]
                    changed = True
                    break
            if changed:
                break
    return [tuple(g) for g in gaps]


def poche_labels(gray: np.ndarray, binary: np.ndarray, labels: np.ndarray, t: float):
    """(Label-Karte, Wandfüllung) aus der Tinte; None, wenn der Plan keine gefüllten Wände hat."""
    po = poche_mask(gray, t, ((labels == WALL) * 255).astype(np.uint8))
    # nur Füllflächen, die zum erkannten Gebäude gehören (kein Planrahmen, Plankopf, Nordpfeil)
    band = cv2.dilate(((labels > 0) * 255).astype(np.uint8), np.ones((int(t) | 1, int(t) | 1), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(po, connectivity=8)
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        m = lab == i
        if band[m].mean() >= 0.6:
            keep[i] = 255
    po = keep[lab]
    # Teile, die nirgends Wandstärke erreichen (kräftige Linien), sind keine Wände
    dt = cv2.distanceTransform(po, cv2.DIST_L2, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(po, connectivity=8)
    keep = np.zeros(n, np.uint8)
    tw = median_thickness(cv2.bitwise_and(po, ((labels == WALL) * 255).astype(np.uint8))) or t
    for i in range(1, n):
        m = lab == i
        if 2 * dt[m].max() >= 0.55 * tw and st[i, 4] >= 2 * tw * tw:
            # typische Stärke (Median über die Mittellinie) – kräftige Linien fallen weg
            sub = (m[st[i, 1]:st[i, 1] + st[i, 3], st[i, 0]:st[i, 0] + st[i, 2]] * 255).astype(np.uint8)
            if median_thickness(sub) >= 0.45 * tw:
                keep[i] = 255
    po = keep[lab]
    if not is_poche_plan(po, labels):
        return None
    # wirklich gefüllt (nicht Schraffur): im Inneren der Wände ist fast alles dunkel
    k = int(max(3, 0.25 * t)) | 1
    core = cv2.erode(((labels == WALL) * 255).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    paper = float(np.percentile(gray, 90))
    if not core.any() or (gray[core] < 0.35 * paper).mean() < 0.78:
        return None
    t2 = median_thickness(po) or t
    from .essential import Opening, _door_swing, _symbol_ink
    from .walls import footprint
    ink = (binary > 0).astype(np.uint8)
    tol_ink = _symbol_ink(ink, po, t2)
    H, W = po.shape
    # Netz-Ergebnis behalten (Leichtbauwände, Öffnungen darin); an der Wandfüllung gilt die Tinte
    out = labels.copy()
    r = int(max(2, round(0.5 * t2)))
    near = cv2.dilate(po, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    out[(out == WALL) & near & (po == 0)] = 0
    out[po > 0] = WALL
    decided = np.zeros(po.shape, bool)
    foot = footprint(po, int(max(3, 4 * t2)))
    for x0, y0, x1, y1, o in _merge_gaps(gap_openings(po, t2)):
        xa, ya, xb, yb = int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))
        xa, ya, xb, yb = max(0, xa), max(0, ya), min(W, xb), min(H, yb)
        free = po[ya:yb, xa:xb] == 0
        if free.size == 0 or free.mean() < 0.7:
            continue                              # kein echter Unterbruch (Wand nur versetzt)
        reg = out[ya:yb, xa:xb]
        dec = decided[ya:yb, xa:xb]
        length = (x1 - x0) if o == "h" else (y1 - y0)
        if length < 1.5 * t2:
            reg[free] = WALL                      # zu kurz für eine Öffnung: Füllung nur unterbrochen
            dec[free] = True
            continue
        band = ink[ya:yb, xa:xb]
        # Linien quer über den ganzen Unterbruch (Rahmen, Glas, Bank) -> Fenster
        prof = band.mean(axis=1 if o == "h" else 0) if band.size else np.zeros(1)
        has_lines = bool((prof >= 0.6).any())
        net = labels[ya:yb, xa:xb][free]
        # liegt die Öffnung in der Gebäudehülle (eine Seite aussen)?
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        th = (y1 - y0) if o == "h" else (x1 - x0)
        d = th / 2 + 0.8 * t2
        pts = [(cx, cy - d), (cx, cy + d)] if o == "h" else [(cx - d, cy), (cx + d, cy)]
        exterior = any(not (0 <= int(px) < W and 0 <= int(py) < H) or foot[int(py), int(px)] == 0
                       for px, py in pts)
        n_lines = int(np.sum(np.diff((prof >= 0.6).astype(np.int8)) == 1) + (prof[0] >= 0.6)) if prof.size else 0
        if (has_lines and (exterior or n_lines >= 2)) or (exterior and (net == WINDOW).mean() > 0.3):
            kind = WINDOW
        elif _door_swing(Opening("door", x0, y0, x1, y1, o), tol_ink):
            kind = DOOR
        elif ((net == DOOR) | (net == WINDOW)).mean() > 0.2:
            kind = WINDOW if (exterior and (net == WINDOW).mean() > (net == DOOR).mean()) else DOOR
        elif length < 2.5 * t2:
            kind = WALL                           # kurze Unterbrechung der Füllung (Symbol, Scanfehler)
        else:
            continue                              # unklar: so lassen, wie gezeichnet (offen)
        reg[free] = kind
        dec[free] = True
    # Netz-Öffnungen, die keinem Unterbruch der Wandfüllung entsprechen, gibt es nicht
    halo = near & (po == 0) & ~decided & ((out == DOOR) | (out == WINDOW))
    out[halo] = 0
    return out, po



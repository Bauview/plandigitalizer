"""Semantische Planerkennung mit einem kleinen, lokal laufenden neuronalen Netz.

Das Netz (``models/plannet.onnx``, ~2 MB) wurde auf synthetischen Plänen trainiert, die aus
17'000 echten Wohnungsgrundrissen (ResPlan, CC BY 4.0) in vielen Zeichenstilen erzeugt wurden.
Es unterscheidet pro Pixel: Hintergrund, Wand, Fenster, Tür. Möbel, Massketten, Texte,
Schraffuren usw. gehören zum Hintergrund.

Ausgeführt wird es mit OpenCV-DNN – im Browser (Pyodide) wie auf dem Desktop identisch,
ohne Cloud.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .walls import median_thickness

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "plannet.onnx"
BG, WALL, WINDOW, DOOR = 0, 1, 2, 3

TARGET_SIDE = 1600          # Netz-Arbeitsgrösse (lange Seite)
MAX_NET_PIXELS = 3_000_000
TILE = 1536
OVERLAP = 128

_net = None
TIMING = False


def available() -> bool:
    return MODEL_PATH.exists()


def _get_net():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        _net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        _net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return _net


@dataclass
class Segmentation:
    labels: np.ndarray        # uint8, Grösse des Arbeitsbildes
    probs: np.ndarray         # uint8 HxWx4 (0..255), Grösse des Arbeitsbildes
    scale: float              # Netz-Pixel pro Arbeits-Pixel
    wall_px: float            # mittlere Wandstärke im Arbeitsbild (px)


def _forward(x: np.ndarray) -> np.ndarray:
    """x: float32 HxW (H, W Vielfache von 32) -> Wahrscheinlichkeiten 4 x H/2 x W/2."""
    import time as _t
    t0 = _t.time()
    net = _get_net()
    net.setInput(x[None, None])
    out = net.forward()[0]
    if TIMING:
        print(f"[PD-TIMING] forward {x.shape} {_t.time() - t0:.2f}s")
    out = out - out.max(0, keepdims=True)
    e = np.exp(out)
    return e / e.sum(0, keepdims=True)


def _run(gray: np.ndarray, s: float) -> np.ndarray:
    """Netz auf dem um ``s`` skalierten Bild; Ergebnis 4 x h' x w' (halbe Netzauflösung)."""
    h, w = gray.shape
    nh, nw = max(32, int(round(h * s))), max(32, int(round(w * s)))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR) if s != 1 else gray
    x = 1.0 - img.astype(np.float32) / 255.0
    H, W = (nh + 31) // 32 * 32, (nw + 31) // 32 * 32
    pad = np.zeros((H, W), np.float32)
    pad[:nh, :nw] = x
    if H * W <= TILE * TILE * 1.3:
        p = _forward(pad)
        return p[:, : (nh + 1) // 2, : (nw + 1) // 2]
    # Kacheln mit Überlappung
    out = np.zeros((4, H // 2, W // 2), np.float32)
    step = TILE - 2 * OVERLAP
    for y0 in range(0, max(1, H - 2 * OVERLAP), step):
        for x0 in range(0, max(1, W - 2 * OVERLAP), step):
            ya, xa = min(y0, max(0, H - TILE)), min(x0, max(0, W - TILE))
            tile = pad[ya:ya + TILE, xa:xa + TILE]
            p = _forward(tile)
            # nur den inneren Bereich übernehmen (Rand des Gesamtbildes vollständig)
            iy0 = 0 if ya == 0 else OVERLAP // 2
            ix0 = 0 if xa == 0 else OVERLAP // 2
            iy1 = p.shape[1] if ya + TILE >= H else p.shape[1] - OVERLAP // 2
            ix1 = p.shape[2] if xa + TILE >= W else p.shape[2] - OVERLAP // 2
            out[:, ya // 2 + iy0: ya // 2 + iy1, xa // 2 + ix0: xa // 2 + ix1] = p[:, iy0:iy1, ix0:ix1]
    return out[:, : (nh + 1) // 2, : (nw + 1) // 2]


def _confidence(p: np.ndarray) -> float:
    am = p.argmax(0)
    sel = am > 0
    if np.count_nonzero(sel) < 50:
        return 0.0
    return float(p.max(0)[sel].mean())


def _crop(gray: np.ndarray, size: float) -> np.ndarray:
    """Quadratischer Ausschnitt (Kantenlänge ``size`` px) um den Schwerpunkt der Zeichnung."""
    h, w = gray.shape
    k = min(1.0, 400.0 / max(h, w))
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small < 128)
    if len(xs) < 20:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = float(np.median(xs)) / k, float(np.median(ys)) / k
    half = size / 2
    x0 = int(max(0, min(w - size, cx - half))) if w > size else 0
    y0 = int(max(0, min(h - size, cy - half))) if h > size else 0
    return gray[y0:y0 + int(size), x0:x0 + int(size)]


def _search_scale(gray: np.ndarray, s_max: float) -> float:
    """Netzgrösse wählen: auf einem Ausschnitt (512 Netz-Pixel) mehrere Verkleinerungen testen,
    die mit der höchsten mittleren Sicherheit gewinnt. Spart im Browser viel Rechenzeit."""
    h, w = gray.shape
    s0 = min(1.0, TARGET_SIDE / max(h, w))
    conf = {}
    side = 768
    for k in (1.0, 0.7, 0.5, 0.35):
        sc = s0 * k
        conf[sc] = _confidence(_run(_crop(gray, side / sc), sc))
    best = _pick(conf)
    extra = best * 0.7 if best == min(conf) else (best * 1.4 if best == max(conf) else None)
    if extra is not None and extra <= s_max and h * w * extra * extra > 32 * 32:
        conf[extra] = _confidence(_run(_crop(gray, side / extra), extra))
        best = _pick(conf)
    return best


def _pick(conf: dict) -> float:
    """Höchste Sicherheit; bei praktisch gleicher Sicherheit die grössere Darstellung (mehr Detail)."""
    top = max(conf.values())
    return max(sc for sc, c in conf.items() if c >= top - 0.012)


def segment(gray: np.ndarray, scale: float | None = None) -> Segmentation:
    """Erkennt Wände, Fenster und Türen.

    Ohne ``scale`` wird die Netzgrösse gesucht: Das Netz rechnet auf mehreren Verkleinerungen,
    gewählt wird die mit der höchsten mittleren Sicherheit (Wände sind dann im Bereich, in dem
    trainiert wurde). Bei erneuter Erkennung (nach Drehen/Entzerren) wird ``scale`` übernommen.
    """
    h, w = gray.shape
    s_max = math.sqrt(MAX_NET_PIXELS / float(h * w))
    if scale is not None:
        s = min(scale, s_max)
    else:
        s = _search_scale(gray, s_max)
    p = _run(gray, s)
    probs = np.zeros((h, w, 4), np.uint8)
    for c in range(4):
        probs[:, :, c] = cv2.resize((p[c] * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
    labels = probs.argmax(2).astype(np.uint8)
    wall_px = median_thickness((labels == WALL).astype(np.uint8) * 255)
    return Segmentation(labels, probs, s, wall_px)


def confident_band(seg: "Segmentation", thr: float = 0.65) -> np.ndarray:
    """Nur sicher erkannte Wand-/Öffnungspixel (für Ausrichtung und Perspektive)."""
    p = seg.probs[:, :, 1].astype(np.uint16) + seg.probs[:, :, 2] + seg.probs[:, :, 3]
    m = ((p >= thr * 255) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # nur Teile des Gebäudes: kleine, verstreute Fragmente (Schmutz, Text) verwerfen
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        amax = float(st[1:, cv2.CC_STAT_AREA].max())
        keep = np.zeros(n, np.uint8)
        keep[1:] = np.where(st[1:, cv2.CC_STAT_AREA] >= 0.05 * amax, 255, 0)
        m = keep[lab]
    return m


def dominant_angle(mask: np.ndarray) -> tuple[float, float]:
    """Hauptrichtung der Wandkanten (Grad, -45..45) und Eindeutigkeit (0..1).

    Aus langen Liniensegmenten der Wandkanten (Hough), gewichtet nach Länge – Gradientenrichtungen
    wären bei gerasterten Treppenkanten zu den Achsen hin verfälscht.
    """
    h, w = mask.shape
    k = min(1.0, 2000.0 / max(h, w))
    m = cv2.resize(mask, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else mask
    m = ((m > 127) * 255).astype(np.uint8)
    edges = cv2.Canny(m, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 1440, threshold=40,
                           minLineLength=int(0.04 * max(m.shape)), maxLineGap=4)
    if segs is None or len(segs) < 2:
        return 0.0, 0.0
    segs = segs.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 45.0) % 90.0 - 45.0
    ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    hist, edges_ = np.histogram(ang, bins=180, range=(-45, 45), weights=ln)
    sm = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, mode="same")[3:-3]
    i = int(np.argmax(sm))
    peak = (edges_[i] + edges_[i + 1]) / 2
    d = (ang - peak + 45) % 90 - 45
    near = np.abs(d) < 1.5
    if not np.any(near):
        return 0.0, 0.0
    # gewichteter Median im Gipfelbereich
    dd, ww = d[near], ln[near]
    o = np.argsort(dd)
    cw = np.cumsum(ww[o])
    peak = peak + float(dd[o][np.searchsorted(cw, cw[-1] / 2)])
    clarity = float(ln[near].sum() / ln.sum())
    return float((peak + 45) % 90 - 45), clarity


# =============================================================================== Perspektive
def _line_families(band: np.ndarray, t: float):
    """Wandkanten als Liniensegmente, gruppiert in zwei (ungefähr) rechtwinklige Richtungen."""
    edges = cv2.Canny(band, 50, 150)
    min_len = int(max(20, 3.5 * t))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=int(max(15, 2 * t)),
                           minLineLength=min_len, maxLineGap=int(max(3, 0.8 * t)))
    if segs is None:
        return None
    segs = segs.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    ln = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    hist, _ = np.histogram(ang % 90, bins=180, range=(0, 90), weights=ln)
    sm = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, mode="same")[3:-3]
    a0 = (np.argmax(sm) + 0.5) * 0.5
    fams = []
    for base in (a0, a0 + 90):
        d = (ang - base + 90) % 180 - 90
        sel = np.abs(d) < 20
        fams.append((segs[sel], ln[sel]))
    return fams


def _vanishing_point(segs, wts, N):
    """Fluchtpunkt (homogen, normierte Koordinaten) per kleinster Quadrate."""
    p1 = np.c_[segs[:, 0:2], np.ones(len(segs))] @ N.T
    p2 = np.c_[segs[:, 2:4], np.ones(len(segs))] @ N.T
    L = np.cross(p1, p2)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True) + 1e-12
    w_ = wts.astype(np.float64).copy()
    v = None
    for _ in range(5):                      # robust (Cauchy-Gewichte): Ausreisser verlieren Einfluss
        A = (L * w_[:, None]).T @ L
        _, vecs = np.linalg.eigh(A)
        v = vecs[:, 0]
        r = np.abs(L @ v)
        sc = max(1e-4, float(np.median(r)) * 2.0)
        w_ = wts / (1.0 + (r / sc) ** 2)
    resid = np.abs(L @ v)
    return v / np.linalg.norm(v), float(np.average(resid, weights=wts))


def perspective_correction(band: np.ndarray, t: float):
    """Homographie (Pixel -> entzerrte Pixel), die stürzende Linien eines schräg fotografierten
    Plans gerade richtet. ``None``, wenn keine nennenswerte Perspektive vorliegt."""
    h, w = band.shape
    fams = _line_families(band, t)
    if fams is None or any(len(f[0]) < 3 for f in fams):
        return None
    s = 2.0 / max(h, w)
    N = np.array([[s, 0, -w * s / 2], [0, s, -h * s / 2], [0, 0, 1.0]])
    vps = []
    for segs, wts in fams:
        v, res = _vanishing_point(segs, wts, N)
        vps.append(v)
    # Konvergenz: wie stark ändert sich die Richtung über das Bild? (0 bei Parallelen)
    conv = []
    for v in vps:
        if abs(v[2]) < 1e-9:
            conv.append(0.0)
        else:
            dist = np.hypot(v[0] / v[2], v[1] / v[2])      # Bildhälfte = 1
            conv.append(math.degrees(2 * math.atan(1.0 / max(dist, 1e-6))))
    if max(conv) < 0.8 or max(conv) > 35:
        return None
    l_inf = np.cross(vps[0], vps[1])
    if abs(l_inf[2]) < 1e-9:
        return None
    l_inf = l_inf / l_inf[2]
    Ha = np.array([[1, 0, 0], [0, 1, 0], [l_inf[0], l_inf[1], 1.0]])
    # Richtungen der beiden Familien nach der affinen Entzerrung -> auf die Achsen legen
    d = []
    for v in vps:
        q = Ha @ v
        dv = q[:2] / (np.linalg.norm(q[:2]) + 1e-12)
        d.append(dv)
    # die "waagrechtere" Familie auf x, die andere auf y – ohne Spiegelung oder Vierteldrehung
    if abs(d[0][0]) < abs(d[1][0]):
        d = [d[1], d[0]]
    if d[0][0] < 0:
        d[0] = -d[0]
    if d[1][1] < 0:
        d[1] = -d[1]
    D = np.c_[d[0], d[1]]
    if abs(np.linalg.det(D)) < 0.3:
        return None
    A = np.eye(3)
    A[:2, :2] = np.linalg.inv(D)
    Hn = A @ Ha
    H = np.linalg.inv(N) @ Hn @ N
    # Plausibilität: Ecken dürfen sich nicht extrem verschieben
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    c2 = H @ corners
    if np.any(c2[2] <= 0):
        return None
    c2 = (c2[:2] / c2[2]).T
    # Massstab beibehalten (gleiche Fläche) und in den positiven Bereich verschieben
    area0 = float(w * h)
    area1 = abs(cv2.contourArea(c2.astype(np.float32)))
    if area1 <= 0:
        return None
    k = math.sqrt(area0 / area1)
    S = np.diag([k, k, 1.0])
    c3 = c2 * k
    T = np.array([[1, 0, -c3[:, 0].min()], [0, 1, -c3[:, 1].min()], [0, 0, 1.0]])
    Hf = T @ S @ H
    span = c3.max(0) - c3.min(0)
    if span[0] > 1.6 * w + 10 or span[1] > 1.6 * h + 10 or span[0] < 0.6 * w or span[1] < 0.6 * h:
        return None
    return Hf, (int(math.ceil(span[0])), int(math.ceil(span[1]))), max(conv)

"""Bildbereinigung: Grösse, Perspektive, Hintergrund, Kontrast, Rauschen,
Binarisierung, Schräglage und Linienverstärkung.

Alle geometrischen Schritte werden in einer 3x3-Matrix ``M`` festgehalten
(Originalbild -> bearbeitetes Bild), damit z.B. Kalibrierpunkte aus der
Vorschau korrekt übertragen werden können.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SIDE = 4500
MIN_SIDE = 1800


@dataclass
class Prepared:
    gray: np.ndarray          # normalisiertes Graubild (weisser Hintergrund)
    binary: np.ndarray        # Tinte = 255
    M: np.ndarray             # 3x3 Transformation Original -> bearbeitet
    resize_factor: float      # nur Skalierung (für DPI)
    warped: bool = False
    deskew_deg: float = 0.0
    warnings: list[str] = field(default_factory=list)


LOW_QUALITY_MSG = ("Die Planqualität ist möglicherweise zu gering. Für bessere Ergebnisse "
                   "empfehlen wir einen Scan mit mindestens 300 dpi.")


def prepare(bgr: np.ndarray, try_perspective: bool = True, deskew: bool = True) -> Prepared:
    warnings: list[str] = []
    h, w = bgr.shape[:2]
    if min(h, w) < 300:
        from .loader import PlanError
        raise PlanError("Das Bild ist zu klein.", f"Auflösung {w}×{h} px – mindestens ca. 1000 px Kantenlänge nötig.")

    # 1) Arbeitsgrösse
    s = 1.0
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
    elif max(h, w) < MIN_SIDE:
        s = MIN_SIDE / max(h, w)
        warnings.append(LOW_QUALITY_MSG)
    if s != 1.0:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    M = np.diag([s, s, 1.0])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Unschärfe-Check
    lap = cv2.Laplacian(cv2.resize(gray, None, fx=0.5, fy=0.5), cv2.CV_64F).var()
    if lap < 40 and LOW_QUALITY_MSG not in warnings:
        warnings.append(LOW_QUALITY_MSG)

    # 2) Perspektivkorrektur (Blatt auf dunklerem Untergrund)
    warped = False
    if try_perspective:
        quad = _find_sheet(gray)
        if quad is not None:
            gray, H = _warp(gray, quad)
            M = H @ M
            warped = True

    # 3) Hintergrund ausgleichen + Kontrast (Blaupausen / Negative zuerst umkehren)
    if not warped and float(np.median(gray)) < 110 and float(np.percentile(gray, 98)) > 150:
        gray = 255 - gray
        warnings.append("Helle Linien auf dunklem Grund erkannt – Bild wurde umgekehrt.")
    norm = _normalize_background(gray)

    # 4) Rauschreduzierung
    norm = cv2.medianBlur(norm, 3)

    # 5) Binarisierung + Schräglage
    binary = _binarize(norm)
    angle = _skew_angle(binary) if deskew else 0.0
    if abs(angle) > 0.15:
        norm, R = _rotate(norm, angle)
        M = R @ M
        binary = _binarize(norm)

    # 6) Bereinigung + Linienverstärkung
    binary = _clean(binary)

    ink = float(np.count_nonzero(binary)) / binary.size
    if ink > 0.30 and LOW_QUALITY_MSG not in warnings:
        warnings.append(LOW_QUALITY_MSG)
    if ink < 0.0005:
        from .loader import PlanError
        raise PlanError("Im Bild wurden keine Planlinien gefunden.",
                        "Das Dokument scheint leer zu sein oder der Kontrast ist zu gering.")

    return Prepared(norm, binary, M, s, warped, angle, warnings)


# ---------------------------------------------------------------------------

def _find_sheet(gray: np.ndarray):
    h, w = gray.shape
    k = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    # helles Blatt vom Untergrund trennen
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    sa = small.shape[0] * small.shape[1]
    if not (0.20 * sa < area < 0.97 * sa):
        return None
    peri = cv2.arcLength(c, True)
    approx = None
    for eps in (0.02, 0.03, 0.04, 0.05):
        a = cv2.approxPolyDP(c, eps * peri, True)
        if len(a) == 4:
            approx = a
            break
    if approx is None or not cv2.isContourConvex(approx):
        return None
    quad = approx.reshape(4, 2).astype(np.float32)

    # nur entzerren, wenn ausserhalb deutlich dunkler (= Blatt auf Tisch)
    mask = np.zeros_like(small)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inside = float(np.median(small[mask > 0]))
    outside_px = small[mask == 0]
    if outside_px.size < 0.02 * sa:
        return None
    outside = float(np.median(outside_px))
    if inside - outside < 30:
        return None
    return quad / k


def _order(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _warp(gray: np.ndarray, quad: np.ndarray):
    tl, tr, br, bl = _order(quad)
    wid = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hei = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    # Perspektive verfälscht das Seitenverhältnis. Liegt es nahe am A-Format (√2), dieses verwenden.
    r = wid / hei
    if abs(r - math.sqrt(2)) / math.sqrt(2) < 0.10:
        hei = wid / math.sqrt(2)
    elif abs(1 / r - math.sqrt(2)) / math.sqrt(2) < 0.10:
        wid = hei / math.sqrt(2)
    # Blattkanten leicht wegschneiden (sonst schwarze Randlinien)
    m = 0.006 * max(wid, hei)
    dst = np.array([[-m, -m], [wid - 1 + m, -m], [wid - 1 + m, hei - 1 + m], [-m, hei - 1 + m]], np.float32)
    H = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    out = cv2.warpPerspective(gray, H, (int(wid), int(hei)), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, H


def _normalize_background(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    k = 0.25
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ks = max(15, int(max(h, w) * k / 60) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    bg = cv2.medianBlur(bg, 21 if ks < 21 else ks)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    norm = cv2.divide(gray.astype(np.float32), np.maximum(bg.astype(np.float32), 1.0), scale=255.0)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    # Kontrast strecken
    lo = float(np.percentile(norm, 0.5))
    if lo < 200:
        norm = np.clip((norm.astype(np.float32) - lo) * (255.0 / max(1.0, 255.0 - lo)), 0, 255).astype(np.uint8)
    return norm


def _binarize(norm: np.ndarray) -> np.ndarray:
    otsu_t, otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    block = max(31, (max(norm.shape) // 90) | 1)
    adapt = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    # schwache Bleistiftlinien über adaptive Schwelle, aber nur wo wirklich dunkler als Papier
    limit = min(235.0, otsu_t + 40)
    weak = cv2.bitwise_and(adapt, (norm < limit).astype(np.uint8) * 255)
    return cv2.bitwise_or(otsu, weak)


def _skew_angle(binary: np.ndarray) -> float:
    h, w = binary.shape
    k = 1500.0 / max(h, w)
    small = cv2.resize(binary, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    lines = cv2.HoughLinesP(small, 1, np.pi / 720, threshold=80,
                            minLineLength=int(0.08 * max(small.shape)), maxLineGap=5)
    if lines is None:
        return 0.0
    angs, wts = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = ((a + 45) % 90) - 45           # Abweichung von nächster Achse
        if abs(a) <= 10:
            angs.append(a)
            wts.append(math.hypot(x2 - x1, y2 - y1))
    if len(angs) < 3:
        return 0.0
    order = np.argsort(angs)
    a = np.array(angs)[order]
    wt = np.cumsum(np.array(wts)[order])
    return float(a[np.searchsorted(wt, wt[-1] / 2)])   # gewichteter Median


def _rotate(img: np.ndarray, angle: float):
    h, w = img.shape
    c = (w / 2, h / 2)
    R = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    R[0, 2] += nw / 2 - c[0]
    R[1, 2] += nh / 2 - c[1]
    out = cv2.warpAffine(img, R, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)
    R3 = np.vstack([R, [0, 0, 1]])
    return out, R3


def _clean(binary: np.ndarray) -> np.ndarray:
    # kleine Lücken schliessen (Linienverstärkung)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Staub/Punkte entfernen
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(8, int((max(binary.shape) / 1000.0) ** 2 * 5))
    keep = np.zeros(n, np.uint8)
    big = (stats[:, cv2.CC_STAT_AREA] >= min_area)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep[(big | (extent >= 12))] = 255
    keep[0] = 0
    return keep[labels]


def rotate_prepared(prep: Prepared, angle: float, extra: list[np.ndarray] | None = None):
    """Dreht das vorbereitete Bild um ``angle`` Grad (gleiche Konvention wie die Schräglagenkorrektur).

    ``extra``: weitere Bilder gleicher Grösse (z.B. Klassenbilder), werden mit 'nearest' mitgedreht.
    """
    norm, R = _rotate(prep.gray, angle)
    binary = _clean(_binarize(norm))
    h, w = norm.shape
    rot_extra = []
    for img in extra or []:
        if img.ndim == 3:
            rot_extra.append(cv2.warpAffine(img, R[:2], (w, h), flags=cv2.INTER_LINEAR, borderValue=0))
        else:
            rot_extra.append(cv2.warpAffine(img, R[:2], (w, h), flags=cv2.INTER_NEAREST, borderValue=0))
    out = Prepared(norm, binary, R @ prep.M, prep.resize_factor, prep.warped, prep.deskew_deg + angle,
                   list(prep.warnings))
    return out, rot_extra


def warp_prepared(prep: Prepared, H: np.ndarray, size: tuple[int, int]) -> Prepared:
    """Wendet eine Homographie (Perspektivkorrektur) auf das vorbereitete Bild an."""
    w, h = size
    norm = cv2.warpPerspective(prep.gray, H, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    binary = _clean(_binarize(norm))
    return Prepared(norm, binary, H @ prep.M, prep.resize_factor, True, prep.deskew_deg, list(prep.warnings))


def transform_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = M @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])

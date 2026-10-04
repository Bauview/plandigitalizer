"""Massstabsermittlung.

Reihenfolge (zuverlässigste zuerst):
1. Kalibrierung durch den Benutzer (zwei Punkte + bekannte Länge)
2. Masskette (mehrere übereinstimmende Masszahlen)
3. Massstabsangabe im Plan (z.B. "M 1:100") + bekannte Papierauflösung
"""
from __future__ import annotations

import re

import numpy as np

from .geometry import Text

_SCALE_RE = re.compile(r"(?:\b(?:M|Mst|Massstab|Maßstab|Scale)\.?\s*)?1\s*[:/]\s*(\d{1,4})\b", re.I)
_PREFIX_RE = re.compile(r"\b(?:M|Mst|Massstab|Maßstab|Scale)\.?\s*1\s*[:/]", re.I)
COMMON = {1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000}

_NUM_RE = re.compile(r"^(\d{1,3})(?:[.,](\d{1,3}))?$")


def find_scale_ratio(texts: list[Text]) -> int | None:
    best = None
    for t in texts:
        for m in _SCALE_RE.finditer(t.text):
            n = int(m.group(1))
            if n not in COMMON:
                continue
            prefixed = bool(_PREFIX_RE.search(t.text))
            score = (2 if prefixed else 1, t.conf)
            if best is None or score > best[0]:
                best = (score, n)
    return best[1] if best else None


def parse_dimension_mm(text: str) -> float | None:
    """'4.50' -> 4500 mm (Meter), '80' -> 800 mm (cm, Schweizer Konvention)."""
    t = text.strip().replace(" ", "")
    t = t.rstrip("⁰¹²³⁴⁵⁶⁷⁸⁹")
    m = _NUM_RE.match(t)
    if not m:
        return None
    whole, frac = m.group(1), m.group(2)
    if frac is not None:
        if len(frac) != 2:
            return None
        v = float(f"{whole}.{frac}") * 1000.0
    else:
        v = float(whole) * 10.0
    return v if 50 <= v <= 60000 else None


def robust_ratio(candidates: list[float], min_inliers: int = 3, rel_tol: float = 0.06) -> float | None:
    """Mind. 3 übereinstimmende Masse (±6 %) – oder 2 sehr genau übereinstimmende (±2.5 %)."""
    if len(candidates) < 2:
        return None
    arr = np.array(sorted(candidates))
    best, best_n = None, 0
    for c in arr:
        inl = arr[np.abs(arr - c) <= rel_tol * c]
        if len(inl) > best_n:
            best_n, best = len(inl), float(np.median(inl))
    if best_n >= min_inliers and best_n >= 0.5 * len(arr):
        return best
    for i, a in enumerate(arr):
        for b in arr[i + 1:]:
            if abs(a - b) <= 0.025 * a and len(arr) <= 4:
                return float((a + b) / 2)
    return None

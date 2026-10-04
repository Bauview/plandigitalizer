"""Skelettierung nur mit OpenCV + NumPy.

Läuft auch im Browser (Pyodide), wo weder opencv-contrib noch scikit-image
verfügbar sind. Die Nachbarschaft jedes Pixels wird per ``filter2D`` als 8-Bit-Code
berechnet und über zwei Nachschlagetabellen entschieden – schnell, ohne Python-Schleife
über Pixel.
"""
from __future__ import annotations

import cv2
import numpy as np

# Nachbarschaftsgewichte (Bit 0 = Osten, gegen den Uhrzeigersinn)
_KERNEL = np.array([[8, 4, 2],
                    [16, 0, 1],
                    [32, 64, 128]], np.float32)


def _luts() -> tuple[np.ndarray, np.ndarray]:
    """Lam-Lee-Suen-Verdünnung (wie scikit-image ``thin``): garantiert 1 px breite Linien."""
    def bits(n):
        return [(n >> i) & 1 for i in range(9)]

    def g1(n):
        b = bits(n)
        return sum(1 for i in (0, 2, 4, 6) if not b[i] and (b[i + 1] or b[(i + 2) % 8])) == 1

    def g2(n):
        b = bits(n)
        n1 = sum(1 for k in (1, 3, 5, 7) if b[k] or b[k - 1])
        n2 = sum(1 for k in (1, 3, 5, 7) if b[k] or b[(k + 1) % 8])
        return min(n1, n2) in (2, 3)

    def g3(n):
        b = bits(n)
        return not ((b[1] or b[2] or not b[7]) and b[0])

    def g3p(n):
        b = bits(n)
        return not ((b[5] or b[6] or not b[3]) and b[4])

    g12 = np.array([g1(n) and g2(n) for n in range(256)])
    return g12 & np.array([g3(n) for n in range(256)]), g12 & np.array([g3p(n) for n in range(256)])


_LUT1, _LUT2 = _luts()


def thin(binary: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Binärbild (Tinte > 0) -> 1-px-Skelett (uint8, 0/255)."""
    img = (binary > 0).astype(np.uint8)
    if not img.any():
        return img * 255
    ys, xs = np.nonzero(img)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
    roi[1:-1, 1:-1] = img[y0:y1, x0:x1]
    for _ in range(max_iter):
        before = int(roi.sum())
        for lut in (_LUT1, _LUT2):
            code = cv2.filter2D(roi.astype(np.float32), -1, _KERNEL, borderType=cv2.BORDER_CONSTANT)
            roi[(roi == 1) & lut[code.astype(np.uint8)]] = 0
        if int(roi.sum()) == before:
            break
    out = np.zeros_like(img)
    out[y0:y1, x0:x1] = roi[1:-1, 1:-1]
    return out * 255


def thinning(binary: np.ndarray) -> np.ndarray:
    """Einheitlich überall dieselbe Verdünnung (Desktop, Server, Browser -> gleiche Ergebnisse)."""
    return thin(binary)

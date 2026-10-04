"""Texterkennung mit Tesseract (lokal, keine Cloud).

Erkennt waagrechte und senkrechte (90°) Texte. Erkannte Textbereiche werden
anschliessend aus dem Binärbild entfernt, damit Buchstaben nicht als
Liniengewirr vektorisiert werden.
"""
from __future__ import annotations

import math
import os
import re

import cv2
import numpy as np

from .geometry import Text

try:
    import pytesseract
    if os.environ.get("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
except ImportError:  # im Browser: Tesseract.js liefert das Ergebnis (texts_from_tsv)
    pytesseract = None

OCR_MAX_SIDE = 3200
_ALNUM = re.compile(r"[A-Za-z0-9ÄÖÜäöüß]")
_lang_cache: str | None = None


def ocr_available() -> bool:
    if pytesseract is None or os.environ.get("PLANDIGITALIZER_NO_OCR"):
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _languages() -> str:
    global _lang_cache
    if _lang_cache is None:
        wanted = os.environ.get("OCR_LANG", "deu+eng").split("+")
        try:
            have = set(pytesseract.get_languages(config=""))
        except Exception:  # noqa: BLE001
            have = {"eng"}
        langs = [l for l in wanted if l in have] or (["eng"] if "eng" in have else sorted(have)[:1])
        _lang_cache = "+".join(langs)
    return _lang_cache


def ocr_images(gray: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Bilder für die Texterkennung: [waagrecht, um 90° gedreht] und Skalierungsfaktor."""
    h, w = gray.shape
    # kleine Pläne vergrössern (Raumstempel sind dort oft nur 10–15 px hoch), grosse verkleinern
    k = min(2.0, OCR_MAX_SIDE / max(h, w))
    if abs(k - 1.0) < 0.05:
        img, k = gray, 1.0
    else:
        img = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    return [img, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)], k


def run_ocr(gray: np.ndarray) -> list[Text]:
    """Desktop/Server: Tesseract über pytesseract."""
    if not ocr_available():
        return []
    imgs, k = ocr_images(gray)
    texts: list[Text] = []
    for img, rotation in zip(imgs, (0, 90)):
        try:
            d = pytesseract.image_to_data(img, lang=_languages(), config="--psm 11",
                                          output_type=pytesseract.Output.DICT)
        except Exception:  # noqa: BLE001
            continue
        texts += texts_from_data(d, k, rotation, imgs[0].shape[0], gray.shape)
    return _dedupe(texts, gray.shape)


def texts_from_tsv(tsvs: list[str], gray_shape: tuple[int, int], k: float, h_img: int) -> list[Text]:
    """Browser: Ergebnis von Tesseract.js (TSV-Ausgabe, waagrecht + 90°) in Texte umwandeln."""
    texts: list[Text] = []
    for tsv, rotation in zip(tsvs, (0, 90)):
        if tsv:
            texts += texts_from_data(parse_tsv(tsv), k, rotation, h_img, gray_shape)
    return _dedupe(texts, gray_shape)


_TSV_COLS = ["level", "page_num", "block_num", "par_num", "line_num", "word_num",
             "left", "top", "width", "height", "conf", "text"]


def parse_tsv(tsv: str) -> dict:
    d: dict[str, list] = {c: [] for c in _TSV_COLS}
    for line in tsv.splitlines():
        parts = line.split("\t")
        if len(parts) < 11 or not parts[0].strip().isdigit():
            continue   # Kopfzeile oder unvollständig
        parts = (parts + [""])[:12]
        try:
            nums = [int(float(v)) for v in parts[:10]]
            conf = float(parts[10])
        except ValueError:
            continue
        for c, v in zip(_TSV_COLS[:10], nums):
            d[c].append(v)
        d["conf"].append(conf)
        d["text"].append(parts[11])
    return d


def texts_from_data(d: dict, k: float, rotation: int, h_img: int, full_shape) -> list[Text]:
    """Wörter -> Textzeilen. ``h_img`` = Höhe des unrotierten (skalierten) OCR-Bildes."""
    H_img = h_img
    short = min(full_shape)
    lines: dict[tuple, list[int]] = {}
    for i, word in enumerate(d["text"]):
        word = (word or "").strip()
        conf = float(d["conf"][i])
        if not word or conf < 60 or not _ALNUM.search(word):
            continue
        if len(word) == 1 and conf < 85:
            continue
        hh = d["height"][i] / k
        if hh < 7 or hh > 0.08 * short:
            continue
        if d["width"][i] / max(1, d["height"][i]) > 25:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(i)

    out: list[Text] = []
    for idx in lines.values():
        idx.sort(key=lambda i: d["left"][i])
        # Wörter einer Zeile zusammenfassen, solange Lücke klein ist
        group = [idx[0]]
        for i in idx[1:]:
            prev = group[-1]
            gap = d["left"][i] - (d["left"][prev] + d["width"][prev])
            if gap < 1.6 * max(d["height"][i], d["height"][prev]):
                group.append(i)
            else:
                out.append(_make(d, group, k, rotation, H_img))
                group = [i]
        out.append(_make(d, group, k, rotation, H_img))
    return out


def _make(d, group, k, rotation, H_img) -> Text:
    x0 = min(d["left"][i] for i in group)
    y0 = min(d["top"][i] for i in group)
    x1 = max(d["left"][i] + d["width"][i] for i in group)
    y1 = max(d["top"][i] + d["height"][i] for i in group)
    txt = " ".join(d["text"][i].strip() for i in group)
    conf = float(np.mean([float(d["conf"][i]) for i in group]))
    if rotation == 90:
        # Bild wurde 90° im Uhrzeigersinn gedreht: (x', y') = (H-1-y, x)
        ox0, ox1 = y0, y1
        oy0, oy1 = H_img - x1, H_img - x0
        x0, y0, x1, y1 = ox0, oy0, ox1, oy1
    return Text(txt, (x0 / k, y0 / k, x1 / k, y1 / k), rotation, conf)


def _dedupe(texts: list[Text], shape) -> list[Text]:
    texts.sort(key=lambda t: -(t.conf * len(t.text)))
    kept: list[Text] = []
    for t in texts:
        if any(_iou(t.box, o.box) > 0.25 for o in kept):
            continue
        kept.append(t)
    return kept


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / min(area(a), area(b))


def filter_by_ink(texts: list[Text], binary: np.ndarray) -> list[Text]:
    """Verwirft Fehlerkennungen: leere/volle Bereiche, Schraffuren, Rahmen."""
    if not texts:
        return texts
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cx, cy = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    cw, ch = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    out = []
    for t in texts:
        x0, y0, x1, y1 = (int(round(v)) for v in t.box)
        roi = binary[max(0, y0):y1, max(0, x0):x1]
        if roi.size == 0:
            continue
        ink = np.count_nonzero(roi) / roi.size
        if not 0.04 <= ink <= 0.55:
            continue
        pad = max(2.0, 0.25 * t.height)
        inside = ((cx >= x0 - pad) & (cy >= y0 - pad) & (cx + cw <= x1 + pad) & (cy + ch <= y1 + pad) & (area >= 3))
        inside[0] = False
        nchars = len(t.text.replace(" ", ""))
        sure_word = t.conf >= 75 and sum(c.isalpha() for c in t.text) >= 4
        need = max(1, math.ceil((0.25 if sure_word else 0.6) * nchars))
        if np.count_nonzero(inside) < need:
            continue
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if np.any(inside & (cw > 0.7 * bw) & (ch > 0.7 * bh)) and nchars > 1:
            continue
        out.append(t)
    return out


def remove_text_from_binary(binary: np.ndarray, texts: list[Text]) -> np.ndarray:
    """Entfernt Zusammenhangskomponenten, die vollständig in Textboxen liegen."""
    if not texts:
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    x2 = x + stats[:, cv2.CC_STAT_WIDTH]
    y2 = y + stats[:, cv2.CC_STAT_HEIGHT]
    remove = np.zeros(n, bool)
    for t in texts:
        pad = max(2.0, 0.25 * t.height)
        bx0, by0, bx1, by1 = t.box[0] - pad, t.box[1] - pad, t.box[2] + pad, t.box[3] + pad
        remove |= (x >= bx0) & (y >= by0) & (x2 <= bx1) & (y2 <= by1)
    remove[0] = False
    out = binary.copy()
    out[remove[labels]] = 0
    return out


def normalize_heights(texts: list[Text], binary: np.ndarray) -> None:
    """Versalhöhe aus der engen Tintenbox bestimmen (Unterlängen abgezogen)
    und fast gleiche Schriftgrössen vereinheitlichen."""
    if not texts:
        return
    for t in texts:
        x0, y0, x1, y1 = (int(round(v)) for v in t.box)
        roi = binary[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if roi.size == 0:
            continue
        prof = (roi > 0).sum(axis=1 if t.rotation == 0 else 0).astype(float)
        if prof.max() <= 0:
            continue
        rows = np.nonzero(prof >= max(1.0, 0.03 * prof.max()))[0]   # enge Tintenbox
        a, b = int(rows[0]), int(rows[-1]) + 1
        if any(c in "gjpqy" for c in t.text):                         # Unterlänge abziehen
            b = a + int(round((b - a) * 0.78))
        if t.rotation == 0:
            t.box = (t.box[0], y0 + a, t.box[2], y0 + b)
        else:  # Grundlinie liegt rechts
            t.box = (x0 + a, t.box[1], x0 + b, t.box[3])
        t.cap = float(b - a)
    hs = np.array([t.height for t in texts])
    for t in texts:
        h = t.height
        near = hs[(hs > h / 1.2) & (hs < h * 1.2)]
        t.cap = float(np.median(near))

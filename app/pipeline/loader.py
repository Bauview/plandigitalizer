"""Laden von Bildern und PDFs in ein einheitliches BGR-Rasterbild."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pymupdf as fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXT = {".pdf"}
ALLOWED_EXT = IMAGE_EXT | PDF_EXT

PDF_RENDER_DPI = 300
MAX_RENDER_PX = 7000


class PlanError(Exception):
    """Fehler mit verständlicher Meldung für die Oberfläche."""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.message = message
        self.detail = detail


@dataclass
class LoadedPlan:
    image: np.ndarray        # BGR uint8
    dpi: float | None        # verlässliche Papier-Auflösung (PDF / Scan), sonst None
    source: str              # "pdf" | "scan" | "photo" | "image"
    pages: int = 1


def load_plan(path: Path) -> LoadedPlan:
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXT:
        raise PlanError("Dateiformat wird nicht unterstützt.",
                        f"Erlaubt sind: {', '.join(sorted(e[1:].upper() for e in ALLOWED_EXT))}.")
    if ext in PDF_EXT:
        return _load_pdf(path)
    return _load_image(path)


def _load_pdf(path: Path) -> LoadedPlan:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise PlanError("Die PDF-Datei konnte nicht geöffnet werden.",
                        "Die Datei ist beschädigt oder keine gültige PDF.") from exc
    try:
        if doc.needs_pass:
            raise PlanError("Die PDF-Datei ist passwortgeschützt.", "Bitte eine ungeschützte Version hochladen.")
        if doc.page_count == 0:
            raise PlanError("Die PDF-Datei enthält keine Seiten.")
        page = doc[0]
        long_in = max(page.rect.width, page.rect.height) / 72.0
        dpi = PDF_RENDER_DPI
        if long_in * dpi > MAX_RENDER_PX:
            dpi = MAX_RENDER_PX / long_in
        pix = page.get_pixmap(dpi=int(dpi), colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # effektive DPI aus Pixelbreite zurückrechnen (get_pixmap rundet)
        eff_dpi = pix.width / (page.rect.width / 72.0)
        return LoadedPlan(bgr, eff_dpi, "pdf", doc.page_count)
    finally:
        doc.close()


def _load_image(path: Path) -> LoadedPlan:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            is_photo = bool(exif.get(0x010F) or exif.get(0x0110))  # Make / Model -> Kamera
            im = ImageOps.exif_transpose(im)
            dpi_info = im.info.get("dpi")
            im = im.convert("RGB")
            arr = np.asarray(im)
    except Exception as exc:  # noqa: BLE001
        raise PlanError("Das Bild konnte nicht gelesen werden.",
                        "Die Datei ist beschädigt oder kein gültiges Bildformat.") from exc

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    dpi = None
    if dpi_info and not is_photo:
        try:
            d = float(dpi_info[0])
            if 150 <= d <= 1200:
                dpi = d
        except (TypeError, ValueError, IndexError):
            pass
    source = "photo" if is_photo else ("scan" if dpi else "image")
    return LoadedPlan(bgr, dpi, source)


def make_preview(plan: LoadedPlan, max_side: int = 1800) -> tuple[bytes, int, int]:
    """JPEG-Vorschau (verkleinert) + Originalmasse."""
    h, w = plan.image.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(plan.image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else plan.image
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise PlanError("Vorschau konnte nicht erstellt werden.")
    return buf.tobytes(), w, h

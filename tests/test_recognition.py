"""Tests der Planerkennung (neuronales Netz + Rekonstruktion):  pytest -q

Testplan (tests/make_samples.py): 12 × 8 m, Aussenwand 30 cm schwarz, Innenwände 12 cm als
Doppellinie, 3 Türen, 4 Fenster, Treppe, Kamin, Stütze, Massketten, Raumnamen.
Hier wird er zusätzlich gedreht, perspektivisch verzerrt und möbliert.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.pipeline import run_pipeline, semantic  # noqa: E402
from app.pipeline.geometry import Arc, Line, Polyline  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"
pytestmark = pytest.mark.skipif(not semantic.available(), reason="Erkennungsmodell fehlt")


@pytest.fixture(scope="module", autouse=True)
def samples():
    if not (SAMPLES / "scan_300dpi.png").exists():
        import runpy
        runpy.run_path(str(ROOT / "tests" / "make_samples.py"), run_name="__main__")


def _scan():
    return cv2.imread(str(SAMPLES / "scan_300dpi.png"), cv2.IMREAD_GRAYSCALE)


def _rec(d):
    r = d.info.get("recognition") or {}
    return r.get("doors"), r.get("windows")


def _walls_axis_aligned(d, tol=1e-6):
    for e in d.entities:
        if isinstance(e, Polyline) and e.layer == "WALLS":
            pts = e.points + [e.points[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                if abs(a[0] - b[0]) > tol and abs(a[1] - b[1]) > tol:
                    return False
    return True


def test_scan_essentials():
    d = run_pipeline(SAMPLES / "scan_300dpi.png")
    assert _rec(d) == (3, 4)
    layers = d.layer_counts()
    # Das Wesentliche: Bemassung ja, Restlinien/Möbel nein
    assert layers.get("DIMENSIONS", 0) >= 4 and "LINES" not in layers and "SYMBOLS" not in layers, layers
    assert layers.get("STAIRS", 0) >= 8
    assert _walls_axis_aligned(d)
    # Türen mit Anschlag (Blatt + Bogen)
    assert sum(isinstance(e, Arc) and e.layer == "DOORS" for e in d.entities) == 3


def test_all_mode_keeps_dimensions():
    d = run_pipeline(SAMPLES / "scan_300dpi.png", essential_only=False)
    assert d.layer_counts().get("DIMENSIONS", 0) > 0


@pytest.mark.parametrize("deg", [9.0, -23.0])
def test_rotated_scan_is_straightened(tmp_path, deg):
    img = _scan()
    h, w = img.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    R[0, 2] += nw / 2 - w / 2
    R[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(img, R, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)
    p = tmp_path / "rot.png"
    cv2.imwrite(str(p), rot)
    d = run_pipeline(p)
    assert abs(d.info["deskew_deg"] + deg) < 0.3 or abs(d.info["deskew_deg"] - deg) < 0.3, d.info["deskew_deg"]
    assert _rec(d) == (3, 4)
    assert _walls_axis_aligned(d)


def test_perspective_photo_without_sheet_edge(tmp_path):
    img = _scan()
    h, w = img.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.07 * w, 0.03 * h], [0.95 * w, 0.0], [w, h], [0.0, 0.94 * h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_CUBIC, borderValue=250)
    p = tmp_path / "persp.png"
    cv2.imwrite(str(p), warped)
    d = run_pipeline(p)
    assert any("Perspektive" in x for x in d.warnings), d.warnings
    assert _rec(d) == (3, 4)


def _furnished(tmp_path):
    from make_samples import P, PX_M  # type: ignore
    img = cv2.cvtColor(_scan(), cv2.COLOR_GRAY2BGR)
    lw = 3
    boxes = []

    def rect(x0, y0, x1, y1):
        cv2.rectangle(img, P(x0, y0), P(x1, y1), (0, 0, 0), lw, cv2.LINE_AA)

    # Bett (Zimmer)
    rect(2.8, 0.8, 4.4, 2.8)
    rect(2.9, 0.9, 3.55, 1.25)
    rect(3.65, 0.9, 4.3, 1.25)
    cv2.line(img, P(2.8, 1.6), P(4.4, 1.6), (0, 0, 0), lw, cv2.LINE_AA)
    boxes.append((2.8, 0.8, 4.4, 2.8))
    # Esstisch mit Stühlen (Wohnen)
    rect(8.8, 6.3, 10.2, 7.2)
    for x in (9.0, 9.6):
        rect(x, 5.8, x + 0.42, 6.2)
        rect(x, 7.3, x + 0.42, 7.65)
    boxes.append((8.8, 5.8, 10.2, 7.65))
    # Sofa
    rect(9.0, 4.8, 10.5, 5.45)
    rect(9.0, 4.8, 10.5, 4.98)
    boxes.append((9.0, 4.8, 10.5, 5.45))
    # Küchenzeile an der Aussenwand mit Spüle und Kochfeld
    rect(9.5, 0.32, 11.6, 0.92)
    rect(9.8, 0.42, 10.3, 0.82)
    for dx in (0.0, 0.3):
        for dy in (0.0, 0.25):
            cv2.circle(img, P(10.8 + dx, 0.5 + dy), int(0.09 * PX_M), (0, 0, 0), lw, cv2.LINE_AA)
    boxes.append((9.5, 0.32, 11.6, 0.92))
    p = tmp_path / "moebliert.png"
    cv2.imwrite(str(p), img)
    px_boxes = []
    for x0, y0, x1, y1 in boxes:
        a, b = P(x0, y0), P(x1, y1)
        px_boxes.append((a[0], a[1], b[0], b[1]))
    return p, px_boxes


def test_furniture_is_ignored(tmp_path):
    p, boxes = _furnished(tmp_path)
    d = run_pipeline(p)
    assert _rec(d) == (3, 4)
    def inside(pt, b, m=6):
        return b[0] - m <= pt[0] <= b[2] + m and b[1] - m <= pt[1] <= b[3] + m

    for e in d.entities:
        if e.layer not in ("WALLS", "DOORS", "WINDOWS"):
            continue
        if isinstance(e, Polyline):
            pts = e.points
        elif isinstance(e, Line):
            pts = [e.p1, e.p2]
        else:
            continue
        for b in boxes:
            assert not all(inside(q, b) for q in pts), (e.layer, pts[:4])


def test_wall_geometry_on_scan():
    """Aussenmasse 12.00 × 8.00 m und Wandstärke 30 cm (Massstab aus der Masskette)."""
    d = run_pipeline(SAMPLES / "scan_300dpi.png")
    assert d.mm_per_px
    sem = d.semantic
    xs = [x for r in sem.rects for x in (r[0], r[2])]
    ys = [y for r in sem.rects for y in (r[1], r[3])]
    width_m = (max(xs) - min(xs)) * d.mm_per_px / 1000
    height_m = (max(ys) - min(ys)) * d.mm_per_px / 1000
    assert width_m == pytest.approx(12.0, abs=0.06)
    assert height_m == pytest.approx(8.0, abs=0.06)
    ext = [r for r in sem.rects if (r[3] - r[1] if r[4] == "h" else r[2] - r[0]) * d.mm_per_px > 200]
    th = [((r[3] - r[1]) if r[4] == "h" else (r[2] - r[0])) * d.mm_per_px for r in ext]
    assert th and all(abs(t - 300) < 25 for t in th), th
    # Türöffnungen im Plan: alle 0.90 m
    widths = sorted(o.width * d.mm_per_px for o in sem.openings if o.kind == "door")
    assert len(widths) == 3 and all(abs(x - 900) < 55 for x in widths), widths   # Leibungslinie hat Strichbreite
    assert not math.isnan(sum(widths))


def test_room_stamps_verified(tmp_path):
    """Raumstempel: Flächen werden wie gezeichnet übernommen, unplausible mit Hinweis."""
    from PIL import Image, ImageDraw, ImageFont
    from make_samples import FONT, P  # type: ignore
    img = Image.open(SAMPLES / "scan_300dpi.png").convert("L")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 34)
    d.text(P(6.0, 6.45), "20.3 m2", fill=0, font=f)      # Wohnen: 6.58 × 3.08 m = 20.3 m²
    d.text(P(9.4, 2.95), "35.0 m2", fill=0, font=f)      # Küche/Bad: tatsächlich 27.6 m² -> falsch
    p = tmp_path / "stempel.png"
    img.save(p)
    dr = run_pipeline(p)
    from app.pipeline.geometry import Text
    rooms = [e.text for e in dr.entities if isinstance(e, Text) and e.layer == "ROOMS"]
    assert "20.3 m²" in rooms, rooms
    assert "35.0 m²" in rooms, rooms
    assert any("35.0 m²" in w and "prüfen" in w for w in dr.warnings), dr.warnings
    assert not any("20.3 m²" in w for w in dr.warnings), dr.warnings
    assert "Wohnen" in rooms

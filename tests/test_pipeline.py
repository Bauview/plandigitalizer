"""Automatische Tests:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import ezdxf
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.export import render_svg, write_dxf, write_pdf  # noqa: E402
from app.pipeline import PlanError, run_pipeline  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"
EXPECTED_MM_PX = 25.4 / 300 * 100   # 300 dpi, Massstab 1:100


@pytest.fixture(scope="session", autouse=True)
def samples():
    if not (SAMPLES / "handyfoto.jpg").exists():
        import runpy
        runpy.run_path(str(ROOT / "tests" / "make_samples.py"), run_name="__main__")


@pytest.mark.parametrize("name", ["scan_300dpi.png", "Bestand_Alt.pdf", "handyfoto.jpg", "handskizze.jpg"])
def test_pipeline_outputs(name, tmp_path):
    d = run_pipeline(SAMPLES / name)
    counts = d.layer_counts()
    rec = d.info.get("recognition")
    if rec:   # Erkennung mit Netz: Wände als zusammenhängende Umrisse
        assert rec["walls"] >= 4 and counts.get("WALLS", 0) >= 1, (rec, counts)
    else:
        assert counts.get("WALLS", 0) >= 4, counts
    write_dxf(d, tmp_path / "out.dxf")
    write_pdf(d, tmp_path / "out.pdf")
    svg = render_svg(d)
    assert svg.startswith("<svg")

    doc = ezdxf.readfile(tmp_path / "out.dxf")
    assert not doc.audit().has_errors
    types = {e.dxftype() for e in doc.modelspace()}
    assert "LINE" in types and "IMAGE" not in types       # echte Vektoren, kein Rasterbild

    pdf = pymupdf.open(tmp_path / "out.pdf")
    assert len(pdf[0].get_images()) == 0                   # keine eingebettete Rastergrafik
    assert len(pdf[0].get_drawings()) > 0


@pytest.mark.parametrize("name", ["scan_300dpi.png", "Bestand_Alt.pdf"])
def test_scale_detection(name):
    d = run_pipeline(SAMPLES / name)
    assert d.unit == "mm"
    assert d.mm_per_px == pytest.approx(EXPECTED_MM_PX, rel=0.02)


def test_photo_scale_from_dimension_chain():
    d = run_pipeline(SAMPLES / "handyfoto.jpg")
    assert d.unit == "mm" and "Masskette" in d.scale_note


def test_calibration_overrides():
    # Masskette unten: 5.00 m zwischen x=0 und x=5 m (Gebäudeursprung 520 px, 1 m = 118.11 px)
    W, H = 3508, 2480
    y = 480 + 8.9 * 118.11
    calib = {"p1": [520 / W, y / H], "p2": [(520 + 5 * 118.11) / W, y / H], "length_m": 5.0}
    d = run_pipeline(SAMPLES / "scan_300dpi.png", calibration=calib)
    assert d.mm_per_px == pytest.approx(EXPECTED_MM_PX, rel=0.01)
    assert "Kalibrierung" in d.scale_note


def test_blank_image_raises(tmp_path):
    import cv2
    import numpy as np
    p = tmp_path / "leer.png"
    cv2.imwrite(str(p), np.full((1500, 2000), 250, np.uint8))
    with pytest.raises(PlanError):
        run_pipeline(p)

"""Realistischer Bestandesplan (tests/make_bestand.py): Geometrie 1:1, Öffnungen, Masse, Raumstempel.

Die Wahrheit (Wandfläche, Öffnungen, Stempel) ist aus der Zeichnungserzeugung bekannt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.pipeline import run_pipeline
from app.pipeline.geometry import Line, Polyline, Text

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "tests" / "samples"


@pytest.fixture(scope="module", autouse=True)
def _samples():
    if not (SAMPLES / "bestand_scan.jpg").exists():
        import runpy
        runpy.run_path(str(ROOT / "tests" / "make_bestand.py"), run_name="__main__")


def _truth(name):
    return json.loads((SAMPLES / "bestand_truth.json").read_text())[name]


def _deviation_mm(d, name):
    """Mittlere Abweichung der gezeichneten Wandkanten von der Wahrheit (mm)."""
    T = _truth(name)
    sem = d.semantic
    h, w = sem.walls.shape
    m = cv2.imread(str(SAMPLES / T.get("wall_file", "bestand_100_wall.png")), cv2.IMREAD_GRAYSCALE)
    M = d.info["_M"] if "_M" in d.info else None
    assert M is not None
    if "H" in T:
        M = M @ np.array(T["H"])
    truth = cv2.warpPerspective(m, M, (w, h), flags=cv2.INTER_NEAREST) > 0

    def edges(a):
        return cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ep, et = edges(sem.walls > 0), edges(truth)
    d1 = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 3)[ep]
    d2 = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)[et]
    return float(np.concatenate([d1, d2]).mean()) * 1000.0 / T["px_m"]


# schwarz gefüllte Wände: die Füllkante wird übernommen; im Testplan ist sie zusätzlich mit einer 0.5-mm-
# Umrisslinie gezeichnet (Wahrheit = Linienmitte) -> systematisch eine halbe Linienbreite (~2.5 cm) aussen
@pytest.mark.parametrize("name,ext,max_mm", [("bestand_100", "png", 8.0), ("bestand_scan", "jpg", 8.0),
                                              ("bestand_solid", "png", 20.0)])
def test_bestand_1to1(name, ext, max_mm, monkeypatch):
    import app.pipeline as P
    orig = P.stage_finish

    def keep_m(st, *a, **k):
        d = orig(st, *a, **k)
        d.info["_M"] = st.prep.M
        return d
    monkeypatch.setattr(P, "stage_finish", keep_m)
    d = P.run_pipeline(SAMPLES / f"{name}.{ext}")
    r = d.info["recognition"]
    assert (r["doors"], r["windows"]) == (4, 6), r
    assert _deviation_mm(d, name) <= max_mm
    layers = d.layer_counts()
    assert layers.get("DIMENSIONS", 0) >= 15, layers
    rooms = [e.text for e in d.entities if isinstance(e, Text) and e.layer == "ROOMS"]
    for must in ("01 Zimmer", "34.3 m²", "12.1 m²", "Parkett"):
        assert must in rooms, rooms
    # Fenster werden aus der Zeichnung übernommen (Rahmen, Glas, Bank): mehrere Linien je Fenster
    assert sum(isinstance(e, Line) and e.layer == "WINDOWS" for e in d.entities) >= 18


def test_window_splays_are_traced():
    """Schräge Fensterleibungen erscheinen als schräge Wandkanten (nicht als Rechteckecke)."""
    d = run_pipeline(SAMPLES / "bestand_100.png")
    slanted = 0
    for e in d.entities:
        if isinstance(e, Polyline) and e.layer == "WALLS":
            pts = e.points + [e.points[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90
                if 20 < ang < 70 and math.dist(a, b) > 10:
                    slanted += 1
    assert slanted >= 12, slanted          # 6 Fenster × 2 Leibungen

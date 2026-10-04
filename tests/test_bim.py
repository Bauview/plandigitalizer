"""Tests für IFC- und 3D-Ausgabe:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from app.bim import ModelError, Settings3D, build_bim  # noqa: E402
from app.bim.derive import derive_model  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"


@pytest.fixture(scope="module", autouse=True)
def samples():
    if not (SAMPLES / "scan_300dpi.png").exists():
        import runpy
        runpy.run_path(str(ROOT / "tests" / "make_samples.py"), run_name="__main__")


@pytest.fixture(scope="module")
def scan():
    return run_pipeline(SAMPLES / "scan_300dpi.png")


def test_model_matches_plan(scan):
    """Testplan: 12 × 8 m, Aussenwand 30 cm, 3 Türen, 4 Fenster, 3 abgeschlossene Räume, 1 Treppe."""
    m = derive_model(scan, Settings3D())
    s = m.summary()
    assert s["doors"] == 3 and s["windows"] == 4 and s["stairs"] == 1
    assert len(s["spaces"]) == 3
    ext = [w for w in m.walls if w.external]
    assert len(ext) >= 4
    assert all(0.27 <= w.thickness <= 0.34 for w in ext)
    xs = [w.start[0] for w in ext if abs(w.start[0] - w.end[0]) < 1e-6]   # senkrechte Aussenwände
    assert max(xs) - min(xs) == pytest.approx(12.0 - 0.3, abs=0.1)        # Achsabstand 11.70 m
    doors = [o for w in m.walls for o in w.openings if o.kind == "door"]
    assert all(0.8 <= o.width <= 1.0 for o in doors)
    names = " ".join(sp["name"] for sp in s["spaces"])
    assert "Wohnen" in names and "Zimmer 01" in names
    assert any("Dach" in n for n in s["not_created"])            # kein erfundenes Dach


def test_ifc_structure_and_validity(scan, tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")   # nur zur Prüfung, nicht zur Erzeugung
    import ifcopenshell.util.element as ue
    import ifcopenshell.validate
    st = Settings3D(storeys=[("UG", 2.5), ("EG", 2.8), ("OG", 2.8)])
    res = build_bim(scan, st, "Test Küche")
    p = tmp_path / "test.ifc"
    p.write_text(res.ifc_text, encoding="utf-8")
    f = ifcopenshell.open(str(p))
    assert f.schema == "IFC4"
    count = lambda c: len(f.by_type(c))  # noqa: E731
    assert count("IfcProject") == count("IfcSite") == count("IfcBuilding") == 1
    assert [s.Name for s in f.by_type("IfcBuildingStorey")] == ["UG", "EG", "OG"]
    assert {s.Name: s.Elevation for s in f.by_type("IfcBuildingStorey")} == {"UG": -2.5, "EG": 0.0, "OG": 2.8}
    assert count("IfcDoor") == 3 and count("IfcWindow") == 4
    assert count("IfcRelVoidsElement") == count("IfcRelFillsElement") == 7
    assert count("IfcSpace") == 3 and count("IfcStair") == 1 and count("IfcSlab") == 1
    assert count("IfcRoof") == 0
    # Annahmen sind dokumentiert
    door = f.by_type("IfcDoor")[0]
    assert "Annahme" in ue.get_pset(door, "PlanDigitalizer_Herkunft")["Höhe"]
    # Schema-Validierung
    log = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, log, express_rules=True)
    assert not log.statements, log.statements[:3]
    assert f.by_type("IfcProject")[0].Name == "Test Küche"           # Umlaute korrekt kodiert
    # Geometrie aller Bauteile lässt sich erzeugen
    import ifcopenshell.geom
    settings = ifcopenshell.geom.settings()
    for e in f.by_type("IfcProduct"):
        if e.Representation:
            ifcopenshell.geom.create_shape(settings, e)
    # 3D-Daten für den Browser
    m3 = res.model3d
    assert len(m3["walls"]) == count("IfcWall")
    assert sum(len(w["openings"]) for w in m3["walls"]) == 7
    assert m3["stairs"] and m3["slabs"]


def test_settings_change_heights(scan, tmp_path):
    m = derive_model(scan, Settings3D(wall_height=3.0, auto_thickness=False, default_thickness=0.25,
                                      storeys=[("EG", 3.2)]))
    assert all(w.height == 3.0 for w in m.walls)
    assert all(w.thickness == 0.25 for w in m.walls)
    assert m.slabs and m.slabs[0].thickness == pytest.approx(0.2)


def test_no_scale_no_model():
    d = run_pipeline(SAMPLES / "handskizze.jpg")
    assert d.mm_per_px is None
    with pytest.raises(ModelError):
        derive_model(d, Settings3D())


def test_settings_from_dict_is_robust():
    s = Settings3D.from_dict({"wall_height": "2,40", "storeys": [{"name": "EG", "height": "2.9"}, {"name": ""}],
                              "plan_storey": "XX", "door_height": "abc"})
    assert s.wall_height == 2.4 and s.storeys == [("EG", 2.9)] and s.plan_storey == "auto"
    assert s.door_height == 2.0

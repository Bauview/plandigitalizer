"""BIM-Erweiterung: 2D-Plan -> Gebäudemodell -> IFC4 + 3D-Daten.

Reines Python (läuft auch im Browser). Das 3D-Modell (GLB/OBJ) wird im Browser
aus ``model_json`` mit three.js erzeugt.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..pipeline.geometry import Drawing
from .derive import ModelError, derive_model
from .ifc_writer import tread_quads, write_ifc_text
from .model import BuildingModel, Settings3D

__all__ = ["build_bim", "BimResult", "Settings3D", "ModelError", "model_json"]


@dataclass
class BimResult:
    model: BuildingModel
    ifc_text: str
    model3d: dict


def build_bim(drawing: Drawing, settings: Settings3D, name: str = "Plan") -> BimResult:
    model = derive_model(drawing, settings)
    return BimResult(model, write_ifc_text(model, name), model_json(model))


def model_json(m: BuildingModel) -> dict:
    """Geometrie für die 3D-Ansicht und den GLB/OBJ-Export (Meter, z nach oben)."""
    st = m.plan_storey
    r = lambda v: round(float(v), 4)  # noqa: E731
    return {
        "elevation": r(st.elevation),
        "walls": [{
            "name": w.name, "external": bool(w.external),
            "start": [r(w.start[0]), r(w.start[1])], "end": [r(w.end[0]), r(w.end[1])],
            "thickness": r(w.thickness), "height": r(w.height),
            "openings": [{"kind": o.kind, "name": o.name, "offset": r(o.offset), "width": r(o.width),
                          "height": r(o.height), "sill": r(o.sill)} for o in w.openings],
        } for w in m.walls],
        "slabs": [{"outline": [[r(x), r(y)] for x, y in s.outline], "thickness": r(s.thickness)} for s in m.slabs],
        "stairs": [{"treads": [[[r(x), r(y)] for x, y in q] for q in tread_quads(s)],
                    "riser": r(s.riser_height)} for s in m.stairs],
        "spaces": [{"name": s.name, "outline": [[r(x), r(y)] for x, y in s.outline], "area": round(s.area, 1)}
                   for s in m.spaces],
    }

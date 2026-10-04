"""DXF-Export mit ezdxf (MIT-Lizenz). Echte CAD-Geometrie auf getrennten Layern."""
from __future__ import annotations

from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment  # noqa: F401  (für spätere Ausrichtungen)

from ..pipeline.geometry import LAYERS, Arc, Circle, Drawing, Hatch, Line, Polyline, Text


def write_dxf(d: Drawing, path: Path) -> None:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4 if d.unit == "mm" else 0   # 4 = Millimeter, 0 = ohne Einheit
    doc.header["$MEASUREMENT"] = 1
    for name, (color, lw, desc) in LAYERS.items():
        layer = doc.layers.add(name, color=color, lineweight=lw)
        layer.description = desc

    f = d.factor()
    H = d.height

    def T(p):
        return (p[0] * f, (H - p[1]) * f)

    msp = doc.modelspace()
    for e in d.entities:
        if isinstance(e, Line):
            msp.add_line(T(e.p1), T(e.p2), dxfattribs={"layer": e.layer})
        elif isinstance(e, Polyline):
            msp.add_lwpolyline([T(p) for p in e.points], close=e.closed, dxfattribs={"layer": e.layer})
        elif isinstance(e, Arc):
            # Bild (y nach unten) -> CAD (y nach oben): Winkel spiegeln, Richtung tauschen
            msp.add_arc(T(e.center), e.radius * f, -e.end, -e.start, dxfattribs={"layer": e.layer})
        elif isinstance(e, Circle):
            msp.add_circle(T(e.center), e.radius * f, dxfattribs={"layer": e.layer})
        elif isinstance(e, Hatch):
            hatch = msp.add_hatch(dxfattribs={"layer": e.layer})
            hatch.set_solid_fill(color=256)  # BYLAYER
            for i, ring in enumerate(e.rings):
                hatch.paths.add_polyline_path([T(p) for p in ring], is_closed=True, flags=1 if i == 0 else 0)
        elif isinstance(e, Text):
            txt = msp.add_text(e.text, height=max(e.height * f, 1e-3), rotation=e.rotation,
                               dxfattribs={"layer": e.layer, "style": "Standard"})
            txt.set_placement(T(e.insert))
    doc.saveas(path)

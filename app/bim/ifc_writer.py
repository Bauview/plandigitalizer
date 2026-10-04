"""IFC4-Export ohne Fremdbibliothek (reines Python, läuft auch im Browser).

Schreibt eine IFC4-STEP-Datei (ISO 10303-21) mit:
IfcProject > IfcSite > IfcBuilding > IfcBuildingStorey,
IfcWall + IfcOpeningElement (IfcRelVoidsElement) + IfcDoor/IfcWindow (IfcRelFillsElement),
IfcSpace, IfcSlab, IfcStair, Standard-Psets/Qtos und «PlanDigitalizer_Herkunft».

Die Schemakonformität wird in den Tests mit IfcOpenShell geprüft.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime

from .model import SOURCE_ASSUMED, SOURCE_MEASURED, BuildingModel

HERKUNFT = "PlanDigitalizer_Herkunft"
_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"


def new_guid() -> str:
    n = uuid.uuid4().int
    chars = [_B64[n >> 126]]
    for i in range(21):
        chars.append(_B64[(n >> (120 - 6 * i)) & 63])
    return "".join(chars)


class E:
    """Aufzählungswert (.WERT.)."""
    def __init__(self, v: str):
        self.v = v


class T:
    """Typisierter Wert, z.B. IFCLABEL('x')."""
    def __init__(self, typ: str, v):
        self.typ, self.v = typ, v


class Ref(str):
    pass


DERIVED = object()


def _str(s: str) -> str:
    out = []
    for ch in s.replace("\\", "\\\\").replace("'", "''"):
        if 32 <= ord(ch) <= 126:
            out.append(ch)
        else:
            out.append("\\X2\\%04X\\X0\\" % ord(ch))
    return "'" + "".join(out) + "'"


def _real(v: float) -> str:
    if not math.isfinite(v):
        v = 0.0
    s = f"{v:.6f}".rstrip("0")
    return "0." if s in ("-0.", "0.") else s


def _fmt(a) -> str:
    if a is None:
        return "$"
    if a is DERIVED:
        return "*"
    if isinstance(a, Ref):
        return str(a)
    if isinstance(a, bool):
        return ".T." if a else ".F."
    if isinstance(a, E):
        return f".{a.v}."
    if isinstance(a, T):
        return f"{a.typ}({_fmt(a.v)})"
    if isinstance(a, int):
        return str(a)
    if isinstance(a, float):
        return _real(a)
    if isinstance(a, str):
        return _str(a)
    if isinstance(a, (list, tuple)):
        return "(" + ",".join(_fmt(x) for x in a) + ")"
    if hasattr(a, "item"):            # numpy-Skalar
        return _fmt(a.item())
    raise TypeError(f"IFC: nicht unterstützter Wert {a!r}")


class Step:
    def __init__(self):
        self.lines: list[str] = []

    def add(self, typ: str, *args) -> Ref:
        n = len(self.lines) + 1
        self.lines.append(f"#{n}={typ}({','.join(_fmt(a) for a in args)});")
        return Ref(f"#{n}")

    def text(self, name: str) -> str:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        head = ["ISO-10303-21;", "HEADER;",
                "FILE_DESCRIPTION(('ViewDefinition [ReferenceView_V1.2]'),'2;1');",
                f"FILE_NAME({_str(name)},'{now}',(''),(''),'PlanDigitalizer','PlanDigitalizer','');",
                "FILE_SCHEMA(('IFC4'));", "ENDSEC;", "DATA;"]
        return "\n".join(head + self.lines + ["ENDSEC;", "END-ISO-10303-21;", ""])


def write_ifc_text(model: BuildingModel, project_name: str = "Plan") -> str:
    s = Step()
    pt = lambda *c: s.add("IFCCARTESIANPOINT", tuple(float(v) for v in c))  # noqa: E731
    dr = lambda *c: s.add("IFCDIRECTION", tuple(float(v) for v in c))  # noqa: E731
    Z, X = dr(0, 0, 1), dr(1, 0, 0)

    def axis(x=0.0, y=0.0, z=0.0, angle=0.0):
        ref = X if abs(angle) < 1e-12 else dr(math.cos(angle), math.sin(angle), 0)
        return s.add("IFCAXIS2PLACEMENT3D", pt(x, y, z), Z, ref)

    def place(rel, x=0.0, y=0.0, z=0.0, angle=0.0):
        return s.add("IFCLOCALPLACEMENT", rel, axis(x, y, z, angle))

    # ------------------------------------------------------------ Kopf
    units = [s.add("IFCSIUNIT", DERIVED, E("LENGTHUNIT"), None, E("METRE")),
             s.add("IFCSIUNIT", DERIVED, E("AREAUNIT"), None, E("SQUARE_METRE")),
             s.add("IFCSIUNIT", DERIVED, E("VOLUMEUNIT"), None, E("CUBIC_METRE")),
             s.add("IFCSIUNIT", DERIVED, E("PLANEANGLEUNIT"), None, E("RADIAN"))]
    unit_ass = s.add("IFCUNITASSIGNMENT", units)
    ctx = s.add("IFCGEOMETRICREPRESENTATIONCONTEXT", None, "Model", 3, 1.0e-5, axis(), None)
    body = s.add("IFCGEOMETRICREPRESENTATIONSUBCONTEXT", "Body", "Model", DERIVED, DERIVED, DERIVED, DERIVED,
                 ctx, None, E("MODEL_VIEW"), None)
    project = s.add("IFCPROJECT", new_guid(), None, project_name,
                    "Automatisch aus 2D-Plan abgeleitet (PlanDigitalizer). Höhen sind Annahmen.",
                    None, None, None, [ctx], unit_ass)

    site_pl = place(None)
    site = s.add("IFCSITE", new_guid(), None, "Grundstück", None, None, site_pl, None, None,
                 E("ELEMENT"), None, None, None, None, None)
    bld_pl = place(site_pl)
    building = s.add("IFCBUILDING", new_guid(), None, "Gebäude", None, None, bld_pl, None, None,
                     E("ELEMENT"), None, None, None)
    s.add("IFCRELAGGREGATES", new_guid(), None, None, None, project, [site])
    s.add("IFCRELAGGREGATES", new_guid(), None, None, None, site, [building])

    storey_refs, storey_pl = [], None
    for st in model.storeys:
        pl = place(bld_pl, 0, 0, st.elevation)
        e = s.add("IFCBUILDINGSTOREY", new_guid(), None, st.name, None, None, pl, None, None,
                  E("ELEMENT"), float(st.elevation))
        storey_refs.append(e)
        _pset(s, e, HERKUNFT, {"Geschosshöhe": T("IFCTEXT", f"{st.height:.2f} m ({SOURCE_ASSUMED}, Einstellung)"),
                               "Grundriss": T("IFCTEXT", "hochgeladen" if st.is_plan else "kein Grundriss – leer")})
        if st.is_plan:
            storey, storey_pl, z_st = e, pl, st
    s.add("IFCRELAGGREGATES", new_guid(), None, None, None, building, storey_refs)

    def shape(items):
        rep = s.add("IFCSHAPEREPRESENTATION", body, "Body", "SweptSolid", items)
        return s.add("IFCPRODUCTDEFINITIONSHAPE", None, None, [rep])

    def prism(points, height, z=0.0):
        pts = [pt(float(x), float(y)) for x, y in points]
        poly = s.add("IFCPOLYLINE", pts + [pts[0]])
        prof = s.add("IFCARBITRARYCLOSEDPROFILEDEF", E("AREA"), None, poly)
        return s.add("IFCEXTRUDEDAREASOLID", prof, axis(0, 0, z), Z, float(height))

    def rect(x0, y0, x1, y1, height, z=0.0):
        return prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], height, z)

    contained = []

    # ------------------------------------------------------------ Wände
    for w in model.walls:
        L, t = w.length, w.thickness
        wpl = place(storey_pl, w.start[0], w.start[1], 0.0, w.angle)
        wall = s.add("IFCWALL", new_guid(), None, w.name, None, None, wpl,
                     shape([rect(0, -t / 2, L, t / 2, w.height)]), None, E("STANDARD"))
        contained.append(wall)
        _pset(s, wall, "Pset_WallCommon", {"IsExternal": T("IFCBOOLEAN", bool(w.external))})
        _qto(s, wall, "Qto_WallBaseQuantities", length={"Length": L, "Width": t, "Height": w.height},
             area={"GrossSideArea": L * w.height})
        _pset(s, wall, HERKUNFT, {"Lage und Länge": T("IFCTEXT", SOURCE_MEASURED),
                                  "Wandstärke": T("IFCTEXT", w.thickness_src),
                                  "Höhe": T("IFCTEXT", f"{SOURCE_ASSUMED} (Einstellung)"),
                                  "Aussen/Innen": T("IFCTEXT", "aus Gebäudeumriss abgeleitet")})
        for o in w.openings:
            pad = 0.05
            opl = place(wpl, o.offset, 0.0, o.sill)
            op = s.add("IFCOPENINGELEMENT", new_guid(), None, f"Öffnung {o.name}", None, None, opl,
                       shape([rect(0, -t / 2 - pad, o.width, t / 2 + pad, o.height)]), None, E("OPENING"))
            s.add("IFCRELVOIDSELEMENT", new_guid(), None, None, None, wall, op)
            depth = 0.05 if o.kind == "door" else min(0.08, t)
            fpl = place(opl)
            rep = shape([rect(0, -depth / 2, o.width, depth / 2, o.height)])
            if o.kind == "door":
                el = s.add("IFCDOOR", new_guid(), None, o.name, None, None, fpl, rep, None,
                           float(o.height), float(o.width), E("DOOR"), E("NOTDEFINED"), None)
                _pset(s, el, "Pset_DoorCommon", {"IsExternal": T("IFCBOOLEAN", bool(w.external))})
            else:
                el = s.add("IFCWINDOW", new_guid(), None, o.name, None, None, fpl, rep, None,
                           float(o.height), float(o.width), E("WINDOW"), E("NOTDEFINED"), None)
                _pset(s, el, "Pset_WindowCommon", {"IsExternal": T("IFCBOOLEAN", bool(w.external))})
            s.add("IFCRELFILLSELEMENT", new_guid(), None, None, None, op, el)
            contained.append(el)
            props = {"Breite": T("IFCTEXT", o.width_src),
                     "Höhe": T("IFCTEXT", f"{SOURCE_ASSUMED} (Standardwert {o.height:.2f} m)"),
                     "Lage": T("IFCTEXT", SOURCE_MEASURED)}
            if o.kind == "window":
                props["Brüstungshöhe"] = T("IFCTEXT", f"{SOURCE_ASSUMED} (Standardwert {o.sill:.2f} m)")
            _pset(s, el, HERKUNFT, props)

    # ------------------------------------------------------------ Räume
    spaces = []
    for sp in model.spaces:
        e = s.add("IFCSPACE", new_guid(), None, sp.name, None, None, place(storey_pl),
                  shape([prism(sp.outline, sp.height)]), sp.name, E("ELEMENT"), E("SPACE"), None)
        spaces.append(e)
        _qto(s, e, "Qto_SpaceBaseQuantities", length={"Height": sp.height}, area={"NetFloorArea": round(sp.area, 2)})
        _pset(s, e, HERKUNFT, {
            "Name": T("IFCTEXT", sp.name_src if sp.name_src == SOURCE_MEASURED
                      else f"{SOURCE_ASSUMED} (keine Raumbezeichnung im Plan)"),
            "Fläche": T("IFCTEXT", "aus Plan berechnet (Nettofläche zwischen Wänden)"),
            "Höhe": T("IFCTEXT", f"{SOURCE_ASSUMED} (Wandhöhe)")})
    if spaces:
        s.add("IFCRELAGGREGATES", new_guid(), None, None, None, storey, spaces)

    # ------------------------------------------------------------ Bodenplatte
    lowest = z_st is model.storeys[0]
    for sl in model.slabs:
        e = s.add("IFCSLAB", new_guid(), None, "Bodenplatte" if lowest else "Geschossdecke", None, None,
                  place(storey_pl, 0, 0, -sl.thickness), shape([prism(sl.outline, sl.thickness)]), None,
                  E("BASESLAB" if lowest else "FLOOR"))
        contained.append(e)
        _qto(s, e, "Qto_SlabBaseQuantities", length={"Depth": sl.thickness})
        _pset(s, e, HERKUNFT, {"Umriss": T("IFCTEXT", "aus Gebäudeumriss (Aussenwände)"),
                               "Stärke": T("IFCTEXT", f"{SOURCE_ASSUMED} (Geschosshöhe − Wandhöhe)")})

    # ------------------------------------------------------------ Treppen
    for i, st in enumerate(model.stairs, 1):
        items = [prism(q, (j + 1) * st.riser_height) for j, q in enumerate(tread_quads(st))]
        e = s.add("IFCSTAIR", new_guid(), None, f"Treppe {i:02d}", None, None, place(storey_pl),
                  shape(items) if items else None, None, E("STRAIGHT_RUN_STAIR"))
        contained.append(e)
        treads = max(0, len(st.tread_edges) - 1)
        going = (st.tread_edges[-1] - st.tread_edges[0]) / treads if treads else 0.0
        _pset(s, e, "Pset_StairCommon", {"NumberOfRiser": T("IFCCOUNTMEASURE", int(st.risers)),
                                         "NumberOfTreads": T("IFCCOUNTMEASURE", int(treads)),
                                         "RiserHeight": T("IFCPOSITIVELENGTHMEASURE", float(st.riser_height)),
                                         "TreadLength": T("IFCPOSITIVELENGTHMEASURE", float(max(going, 0.001)))})
        _pset(s, e, HERKUNFT, {"Lage, Breite, Auftritt": T("IFCTEXT", SOURCE_MEASURED),
                               "Steigungshöhe": T("IFCTEXT", f"{SOURCE_ASSUMED} (Geschosshöhe ÷ Anzahl Stufen)"),
                               "Laufrichtung": T("IFCTEXT", SOURCE_ASSUMED),
                               "Geometrie": T("IFCTEXT", "vereinfacht (Blockstufen)")})

    if contained:
        s.add("IFCRELCONTAINEDINSPATIALSTRUCTURE", new_guid(), None, None, None, contained, storey)
    return s.text(f"{project_name}.ifc")


def tread_quads(st) -> list[list[tuple[float, float]]]:
    """Grundrisse der einzelnen Stufen (globale Koordinaten)."""
    ox, oy = st.outline[0]
    ux, uy = st.direction
    vx, vy = st.outline[1][0] - ox, st.outline[1][1] - oy
    n = math.hypot(vx, vy) or 1.0
    vx, vy = vx / n, vy / n
    out = []
    for a, b in zip(st.tread_edges[:-1], st.tread_edges[1:]):
        p0 = (ox + ux * a, oy + uy * a)
        p1 = (ox + ux * b, oy + uy * b)
        out.append([p0, p1, (p1[0] + vx * st.width, p1[1] + vy * st.width),
                    (p0[0] + vx * st.width, p0[1] + vy * st.width)])
    return out


def _pset(s: Step, obj, name: str, props: dict) -> None:
    vals = [s.add("IFCPROPERTYSINGLEVALUE", k, None, v, None) for k, v in props.items()]
    ps = s.add("IFCPROPERTYSET", new_guid(), None, name, None, vals)
    s.add("IFCRELDEFINESBYPROPERTIES", new_guid(), None, None, None, [obj], ps)


def _qto(s: Step, obj, name: str, length: dict | None = None, area: dict | None = None) -> None:
    qs = [s.add("IFCQUANTITYLENGTH", k, None, None, float(v), None) for k, v in (length or {}).items()]
    qs += [s.add("IFCQUANTITYAREA", k, None, None, float(v), None) for k, v in (area or {}).items()]
    q = s.add("IFCELEMENTQUANTITY", new_guid(), None, name, None, None, qs)
    s.add("IFCRELDEFINESBYPROPERTIES", new_guid(), None, None, None, [obj], q)

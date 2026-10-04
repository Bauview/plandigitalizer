"""Einfaches Gebäudemodell (BIM-Zwischenformat).

Koordinaten in Metern, CAD-Ausrichtung (x nach rechts, y nach oben, z nach oben).
Das Modell liegt deckungsgleich auf der DXF-Ausgabe.

Jede Grösse trägt ihre Herkunft: ``erkannt`` (aus dem Plan gemessen) oder
``Annahme`` (Standardwert bzw. Benutzereinstellung).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

P2 = tuple[float, float]

SOURCE_MEASURED = "aus Plan erkannt"
SOURCE_ASSUMED = "Annahme"


@dataclass
class Settings3D:
    wall_height: float = 2.60
    auto_thickness: bool = True
    default_thickness: float = 0.20
    door_height: float = 2.00
    window_height: float = 1.20
    sill_height: float = 0.90
    storeys: list[tuple[str, float]] = field(default_factory=lambda: [("EG", 2.80)])
    plan_storey: str = "auto"           # Name aus ``storeys`` oder "auto"

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings3D":
        s = cls()
        if not isinstance(d, dict):
            return s

        def num(key, lo, hi):
            try:
                v = float(str(d.get(key)).replace(",", "."))
            except (TypeError, ValueError):
                return None
            return v if lo <= v <= hi else None

        for key, lo, hi in (("wall_height", 1.0, 20.0), ("default_thickness", 0.03, 2.0),
                            ("door_height", 1.5, 5.0), ("window_height", 0.2, 5.0),
                            ("sill_height", 0.0, 3.0)):
            v = num(key, lo, hi)
            if v is not None:
                setattr(s, key, v)
        if "auto_thickness" in d:
            s.auto_thickness = bool(d["auto_thickness"])
        rows = []
        for row in d.get("storeys") or []:
            try:
                name = str(row.get("name", "")).strip()[:20]
                h = float(str(row.get("height")).replace(",", "."))
            except (AttributeError, TypeError, ValueError):
                continue
            if name and 1.5 <= h <= 30 and name not in {r[0] for r in rows}:
                rows.append((name, h))
        if rows:
            s.storeys = rows[:12]
        ps = str(d.get("plan_storey") or "auto")
        s.plan_storey = ps if ps == "auto" or ps in {r[0] for r in s.storeys} else "auto"
        return s


@dataclass
class Storey:
    name: str
    elevation: float
    height: float
    is_plan: bool = False


@dataclass
class Opening:
    kind: str                  # "door" | "window"
    offset: float              # Abstand Wandanfang -> Öffnungsanfang (m, entlang Wandachse)
    width: float
    height: float
    sill: float                # Unterkante über Geschossboden
    width_src: str = SOURCE_MEASURED
    height_src: str = SOURCE_ASSUMED
    sill_src: str = SOURCE_ASSUMED
    name: str = ""


@dataclass
class Wall:
    start: P2                  # Wandachse
    end: P2
    thickness: float
    height: float
    external: bool
    thickness_src: str = SOURCE_MEASURED
    openings: list[Opening] = field(default_factory=list)
    name: str = ""

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def angle(self) -> float:
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])


@dataclass
class Space:
    name: str
    outline: list[P2]
    area: float
    height: float
    name_src: str = SOURCE_MEASURED


@dataclass
class Stair:
    outline: list[P2]          # Rechteck
    direction: P2              # Laufrichtung (Einheitsvektor)
    run_start: float           # Position der ersten Stufenkante entlang ``direction`` (m, relativ zu outline[0])
    tread_edges: list[float]   # Lage der Stufenkanten entlang ``direction`` relativ zum Rechteckanfang
    width: float
    risers: int
    riser_height: float


@dataclass
class Slab:
    outline: list[P2]
    thickness: float           # nach unten ab Geschossboden


@dataclass
class BuildingModel:
    storeys: list[Storey]
    walls: list[Wall] = field(default_factory=list)
    spaces: list[Space] = field(default_factory=list)
    stairs: list[Stair] = field(default_factory=list)
    slabs: list[Slab] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    not_created: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    plan_storey_detected: str | None = None

    @property
    def plan_storey(self) -> Storey:
        return next(s for s in self.storeys if s.is_plan)

    def summary(self) -> dict:
        doors = sum(1 for w in self.walls for o in w.openings if o.kind == "door")
        windows = sum(1 for w in self.walls for o in w.openings if o.kind == "window")
        return {
            "storey": self.plan_storey.name,
            "storey_detected": self.plan_storey_detected,
            "storeys": [{"name": s.name, "elevation": round(s.elevation, 3), "height": s.height,
                         "is_plan": s.is_plan} for s in self.storeys],
            "walls": len(self.walls),
            "walls_external": sum(1 for w in self.walls if w.external),
            "doors": doors,
            "windows": windows,
            "spaces": [{"name": s.name, "area": round(s.area, 1)} for s in self.spaces],
            "stairs": len(self.stairs),
            "slabs": len(self.slabs),
            "assumptions": self.assumptions,
            "not_created": self.not_created,
            "warnings": self.warnings,
        }

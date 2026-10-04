"""Gemeinsames Geometriemodell.

Alle Koordinaten liegen im Pixelraum des vorverarbeiteten Bildes
(Ursprung oben links, y nach unten). Die Exporter rechnen beim Schreiben
in CAD-Koordinaten (y nach oben, Einheit mm oder px) um.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Pt = tuple[float, float]

# Layer-Definitionen: Name -> (AutoCAD-Farbindex, Linienstärke in 1/100 mm, Beschreibung)
LAYERS: dict[str, tuple[int, int, str]] = {
    "WALLS": (7, 50, "Wände"),
    "DOORS": (3, 25, "Türen"),
    "WINDOWS": (4, 25, "Fenster"),
    "STAIRS": (5, 25, "Treppen"),
    "TEXT": (2, 18, "Texte"),
    "DIMENSIONS": (1, 13, "Bemassung"),
    "LINES": (8, 25, "Linien"),
    "HATCH": (9, 9, "Schraffuren / Füllungen"),
    "SYMBOLS": (6, 18, "Symbole / Unerkanntes"),
}


@dataclass
class Line:
    p1: Pt
    p2: Pt
    layer: str = "LINES"
    width: float = 1.0      # gemessene Strichbreite in px
    comp: int = -1          # Skelett-Komponente (für Gruppierung)

    @property
    def length(self) -> float:
        return math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])

    @property
    def angle(self) -> float:
        """Richtung 0..180° (ungerichtet)."""
        a = math.degrees(math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0]))
        return a % 180.0

    @property
    def is_h(self) -> bool:
        return self.p1[1] == self.p2[1] and self.p1[0] != self.p2[0]

    @property
    def is_v(self) -> bool:
        return self.p1[0] == self.p2[0] and self.p1[1] != self.p2[1]

    @property
    def mid(self) -> Pt:
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)


@dataclass
class Polyline:
    points: list[Pt]
    closed: bool = False
    layer: str = "LINES"


@dataclass
class Arc:
    center: Pt
    radius: float
    start: float            # Grad, Bildkoordinaten (y nach unten), start < end
    end: float
    layer: str = "LINES"

    def point_at(self, deg: float) -> Pt:
        r = math.radians(deg)
        return (self.center[0] + self.radius * math.cos(r), self.center[1] + self.radius * math.sin(r))

    @property
    def sweep(self) -> float:
        return self.end - self.start


@dataclass
class Circle:
    center: Pt
    radius: float
    layer: str = "SYMBOLS"


@dataclass
class Text:
    text: str
    box: tuple[float, float, float, float]   # x0, y0, x1, y1 (Bild)
    rotation: int = 0                         # 0 = waagrecht, 90 = von unten nach oben lesbar
    conf: float = 0.0
    layer: str = "TEXT"
    cap: float | None = None                  # bereinigte Versalhöhe (px)

    @property
    def height(self) -> float:
        """Schrifthöhe (Versalhöhe) in px."""
        if self.cap:
            return self.cap
        x0, y0, x1, y1 = self.box
        return (y1 - y0) if self.rotation == 0 else (x1 - x0)

    @property
    def center(self) -> Pt:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @property
    def insert(self) -> Pt:
        """Einfügepunkt links auf der Grundlinie (Bildkoordinaten)."""
        x0, y0, x1, y1 = self.box
        return (x0, y1) if self.rotation == 0 else (x1, y1)


@dataclass
class Hatch:
    rings: list[list[Pt]]   # erster Ring = Aussenkontur, weitere = Löcher
    layer: str = "HATCH"


@dataclass
class Drawing:
    width: int
    height: int
    entities: list = field(default_factory=list)
    mm_per_px: float | None = None     # reale mm pro px (Massstab bekannt)
    unit: str = "px"                   # "mm" | "px"
    scale_note: str = ""
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    preview_jpg: bytes = b""          # bereinigtes Bild (für Vergleichsansicht)
    semantic: object = None           # erkannte Wände/Öffnungen (essential.SemanticPlan) für IFC/3D

    def layer_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entities:
            counts[e.layer] = counts.get(e.layer, 0) + 1
        return counts

    def factor(self) -> float:
        """Einheiten pro Pixel für den CAD-Export."""
        return self.mm_per_px if self.mm_per_px else 1.0

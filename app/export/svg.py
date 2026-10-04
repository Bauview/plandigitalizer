"""SVG für die Vorschau im Browser."""
from __future__ import annotations

from html import escape

from ..pipeline.geometry import Arc, Circle, Drawing, Hatch, Line, Polyline, Text

ORDER = ["HATCH", "LINES", "SYMBOLS", "STAIRS", "DIMENSIONS", "WINDOWS", "DOORS", "WALLS", "TEXT", "ROOMS"]


def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def render_svg(d: Drawing) -> str:
    W, H = d.width, d.height
    groups: dict[str, list[str]] = {k: [] for k in ORDER}
    paths: dict[str, list[str]] = {k: [] for k in ORDER}
    fills: list[str] = []
    for e in d.entities:
        L = e.layer if e.layer in groups else "LINES"
        if isinstance(e, Line):
            paths[L].append(f"M{_f(e.p1[0])} {_f(e.p1[1])}L{_f(e.p2[0])} {_f(e.p2[1])}")
        elif isinstance(e, Polyline):
            pts = "L".join(f"{_f(x)} {_f(y)}" for x, y in e.points)
            paths[L].append(f"M{pts}{'Z' if e.closed else ''}")
        elif isinstance(e, Arc):
            s, t = e.point_at(e.start), e.point_at(e.end)
            large = 1 if e.sweep > 180 else 0
            paths[L].append(f"M{_f(s[0])} {_f(s[1])}A{_f(e.radius)} {_f(e.radius)} 0 {large} 1 {_f(t[0])} {_f(t[1])}")
        elif isinstance(e, Circle):
            groups[L].append(f'<circle cx="{_f(e.center[0])}" cy="{_f(e.center[1])}" r="{_f(e.radius)}"/>')
        elif isinstance(e, Hatch):
            dd = "".join("M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in ring) + "Z" for ring in e.rings)
            fills.append(dd)
        elif isinstance(e, Text):
            x, y = e.insert
            fs = max(4.0, e.height / 0.72)
            tr = f' transform="rotate(-90 {_f(x)} {_f(y)})"' if e.rotation == 90 else ""
            groups[L].append(f'<text x="{_f(x)}" y="{_f(y)}" font-size="{_f(fs)}"{tr}>{escape(e.text)}</text>')

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'preserveAspectRatio="xMidYMid meet" class="vec">',
           "<style>.vec{background:#fff}.vec path,.vec circle{fill:none;stroke:#111;"
           "vector-effect:non-scaling-stroke;stroke-width:1;stroke-linecap:round;stroke-linejoin:round}"
           ".vec .l-WALLS path{stroke-width:1.8}.vec .l-DIMENSIONS path{stroke:#555;stroke-width:.8}"
           ".vec .l-HATCH path{stroke:#888;stroke-width:.6}.vec path.fill{fill:#9a9a9a;stroke:none;fill-rule:evenodd}"
           ".vec text{font-family:Helvetica,Arial,sans-serif;fill:#111}.vec .l-ROOMS text{font-weight:600}"
           ".vec .l-WINDOWS path,.vec .l-DOORS path{stroke-width:.8}</style>",
           f'<rect width="{W}" height="{H}" fill="#fff"/>']
    if fills:
        out.append(f'<g class="l-HATCH"><path class="fill" d="{"".join(fills)}"/></g>')
    for L in ORDER:
        if not paths[L] and not groups[L]:
            continue
        out.append(f'<g class="l-{L}">')
        if paths[L]:
            out.append(f'<path d="{"".join(paths[L])}"/>')
        out.extend(groups[L])
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)

"""Verarbeitet eine Datei ohne Webserver (zum Testen).

    python tests/run_sample.py tests/samples/handyfoto.jpg  [ausgabeordner]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.export import convert_to_dwg, render_svg, write_dxf, write_pdf  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "tests/output")
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    d = run_pipeline(src, progress=lambda i: print("  Schritt", i + 1))
    stem = src.stem
    write_dxf(d, out / f"{stem}.dxf")
    write_pdf(d, out / f"{stem}.pdf", stem)
    (out / f"{stem}.svg").write_text(render_svg(d), encoding="utf-8")
    dwg = convert_to_dwg(out / f"{stem}.dxf", out / f"{stem}.dwg")
    print(json.dumps({"layers": d.layer_counts(), "unit": d.unit, "mm_per_px": d.mm_per_px,
                      "scale": d.scale_note, "warnings": d.warnings, "info": d.info, "dwg": dwg,
                      "seconds": round(time.time() - t, 1)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Packt den Python-Kern (Ordner app/) für die Browser-Version nach web/py/app.zip.

Nach jeder Änderung an Python-Dateien ausführen:   python3 tools/build_web.py
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app"
OUT = ROOT / "web" / "py" / "app.zip"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(str(p.relative_to(ROOT)), date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:10]
    # Version in index.html setzen, damit Browser die neue app.zip laden (kein veralteter Cache)
    index = ROOT / "web" / "index.html"
    html = index.read_text(encoding="utf-8")
    html = re.sub(r'data-version="[^"]*"', f'data-version="{digest}"', html, count=1)
    index.write_text(html, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(files)} Dateien, Version {digest}")


if __name__ == "__main__":
    main()

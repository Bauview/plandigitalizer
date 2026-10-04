"""Prüft, dass die Website den aktuellen Python-Code enthält (web/py/app.zip)."""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_zip_is_current():
    z = zipfile.ZipFile(ROOT / "web" / "py" / "app.zip")
    packed = {n: z.read(n) for n in z.namelist()}
    for p in (ROOT / "app").rglob("*.py"):
        rel = str(p.relative_to(ROOT))
        assert rel in packed, f"{rel} fehlt – bitte 'python3 tools/build_web.py' ausführen"
        assert packed[rel] == p.read_bytes(), f"{rel} veraltet – bitte 'python3 tools/build_web.py' ausführen"


def test_web_files_present():
    for f in ["index.html", "app.js", "engine.js", "worker.js", "_headers", "pyodide/pyodide.js",
              "pyodide/pyodide-lock.json", "vendor/tesseract/lang/deu.traineddata.gz", "vendor/three.module.min.js"]:
        assert (ROOT / "web" / f).exists(), f
    # Cloudflare Pages: max. 25 MiB pro Datei
    big = [p for p in (ROOT / "web").rglob("*") if p.is_file() and p.stat().st_size > 25 * 1024 * 1024]
    assert not big, big

"""Ende-zu-Ende-Test der Website im echten Browser (Pyodide, Tesseract.js, three.js).

    python3 tools/serve.py 8000 &            # Website lokal
    python3 tests/e2e_browser.py tests/samples/scan_300dpi.png [...]

Prüft: Rechenkern lädt, Plan wird verarbeitet, alle Downloads entstehen, keine Anfrage an fremde Server.
Benötigt: pip install playwright (Chromium vorhanden).
"""
from __future__ import annotations

import asyncio
import base64
import glob
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).parent / "output" / "e2e"


def _chromium() -> str | None:
    c = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome"))
    return c[-1] if c else None


async def main(files: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fails = 0
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path=_chromium(),
                                    args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = await b.new_page(viewport={"width": 1360, "height": 1000})
        logs, ext = [], []
        pg.on("console", lambda m: logs.append(m.type + ": " + m.text[:300]))
        pg.on("request", lambda r: ext.append(r.url) if not r.url.startswith(("http://127.0.0.1", "blob:", "data:")) else None)
        t = time.time()
        await pg.goto("http://127.0.0.1:8000/")
        await pg.wait_for_selector("#engineLoad.is-ready", timeout=180000)
        print("Rechenkern bereit", round(time.time() - t, 1), "s")
        for f in files:
            t = time.time()
            await pg.set_input_files("#file", f)
            await pg.wait_for_selector(".view-preview", state="visible", timeout=60000)
            await pg.click("[data-action=process]")
            await pg.wait_for_function("['result','error'].includes(document.querySelector('#app').dataset.view)",
                                       timeout=600000)
            if await pg.is_visible(".view-error"):
                print("FEHLER", f, await pg.inner_text("#errorDetail"))
                fails += 1
                await pg.click(".view-error [data-action=reset]")
                continue
            print("fertig", Path(f).name, round(time.time() - t, 1), "s")
            await pg.wait_for_timeout(2500)          # 3D-Dateien entstehen kurz nach dem Ergebnis
            for dl in ["dlPdf", "dlCad", "dlIfc", "dlGlb", "dlObj"]:
                href = await pg.get_attribute("#" + dl, "href")
                fn = await pg.get_attribute("#" + dl, "download")
                if not href:
                    print("  fehlt:", dl)
                    fails += 1
                    continue
                data = await pg.evaluate(
                    "async (u)=>{const b=await (await fetch(u)).arrayBuffer(); let s=''; const a=new Uint8Array(b);"
                    "for(let i=0;i<a.length;i+=0x8000) s+=String.fromCharCode.apply(null,a.subarray(i,i+0x8000)); return btoa(s);}", href)
                (OUT / fn).write_bytes(base64.b64decode(data))
            print("  ", await pg.inner_text("#summary"))
            await pg.screenshot(path=str(OUT / (Path(f).stem + "_ergebnis.png")), full_page=True)
            await pg.click(".view-result [data-action=reset]")
        await b.close()
    if ext:
        print("FREMDE ANFRAGEN:", ext)
        fails += 1
    print("\n".join(l for l in logs if "PD-TIMING" in l))
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

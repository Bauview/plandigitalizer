"""Erzeugt Testpläne: sauberer Scan (PNG 300 dpi), Scan-PDF und simuliertes Handyfoto.

    python tests/make_samples.py   ->  tests/samples/
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "samples"
DPI = 300
PX_M = 10 / 25.4 * DPI          # 1 m bei 1:100 = 10 mm Papier
W, H = 3508, 2480               # A4 quer, 300 dpi
OX, OY = 520, 480               # Gebäudeursprung (px)
LW = 3                          # dünne Linie
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def P(xm, ym):
    return (int(round(OX + xm * PX_M)), int(round(OY + ym * PX_M)))


def draw_plan() -> np.ndarray:
    img = np.full((H, W), 255, np.uint8)
    t = 0.30  # Aussenwand 30 cm, schwarz gefüllt
    cv2.rectangle(img, P(0, 0), P(12, 8), 0, -1)
    cv2.rectangle(img, P(t, t), P(12 - t, 8 - t), 255, -1)

    def gap_h(x0, x1, y0, y1):  # Öffnung in horizontaler Aussenwand
        cv2.rectangle(img, P(x0, y0), P(x1, y1), 255, -1)

    def gap_v(y0, y1, x0, x1):
        cv2.rectangle(img, P(x0, y0), P(x1, y1), 255, -1)

    def window_h(x0, x1, y0, y1):
        gap_h(x0, x1, y0, y1)
        for y in (y0, (y0 + y1) / 2, y1):
            cv2.line(img, P(x0, y), P(x1, y), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(x0, y0), P(x0, y1), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(x1, y0), P(x1, y1), 0, LW, cv2.LINE_AA)

    def window_v(y0, y1, x0, x1):
        gap_v(y0, y1, x0, x1)
        for x in (x0, (x0 + x1) / 2, x1):
            cv2.line(img, P(x, y0), P(x, y1), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(x0, y0), P(x1, y0), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(x0, y1), P(x1, y1), 0, LW, cv2.LINE_AA)

    window_h(1.2, 3.2, 0, t)
    window_h(7.0, 9.0, 0, t)
    window_h(8.5, 10.5, 8 - t, 8)
    window_v(1.0, 3.0, 12 - t, 12)

    # Haustür unten (0.9 m), Anschlag links
    gap_h(5.6, 6.5, 8 - t, 8)
    cv2.line(img, P(5.6, 8 - t), P(5.6, 8 - t - 0.9), 0, LW, cv2.LINE_AA)
    cv2.ellipse(img, P(5.6, 8 - t), (int(0.9 * PX_M), int(0.9 * PX_M)), 0, -90, 0, 0, LW, cv2.LINE_AA)

    # Innenwände 12 cm als Doppellinie
    w = 0.12
    xw = 5.0
    for x in (xw, xw + w):
        cv2.line(img, P(x, t), P(x, 2.0), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(x, 2.9), P(x, 8 - t), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(xw, 2.0), P(xw + w, 2.0), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(xw, 2.9), P(xw + w, 2.9), 0, LW, cv2.LINE_AA)
    # Tür in Innenwand (Anschlag oben, schwenkt nach rechts)
    cv2.line(img, P(xw + w, 2.0), P(xw + w + 0.8, 2.0), 0, LW, cv2.LINE_AA)
    cv2.ellipse(img, P(xw + w, 2.0), (int(0.8 * PX_M), int(0.8 * PX_M)), 0, 0, 90, 0, LW, cv2.LINE_AA)

    yw = 4.5
    for y in (yw, yw + w):
        cv2.line(img, P(xw + w, y), P(8.0, y), 0, LW, cv2.LINE_AA)
        cv2.line(img, P(8.9, y), P(12 - t, y), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(8.0, yw), P(8.0, yw + w), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(8.9, yw), P(8.9, yw + w), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(8.0, yw + w), P(8.0, yw + w + 0.8), 0, LW, cv2.LINE_AA)
    cv2.ellipse(img, P(8.0, yw + w), (int(0.8 * PX_M), int(0.8 * PX_M)), 0, 0, 90, 0, LW, cv2.LINE_AA)

    # Treppe: 11 Tritte à 27 cm
    sx0, sx1, sy0 = 0.6, 1.8, 3.2
    for i in range(12):
        y = sy0 + i * 0.27
        cv2.line(img, P(sx0, y), P(sx1, y), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(sx0, sy0), P(sx0, sy0 + 11 * 0.27), 0, LW, cv2.LINE_AA)
    cv2.line(img, P(sx1, sy0), P(sx1, sy0 + 11 * 0.27), 0, LW, cv2.LINE_AA)

    # Kamin 60x60 mit Schraffur
    cx0, cy0, s = 10.6, 5.6, 0.6
    cv2.rectangle(img, P(cx0, cy0), P(cx0 + s, cy0 + s), 0, LW, cv2.LINE_AA)
    mask = np.zeros_like(img)
    cv2.rectangle(mask, P(cx0, cy0), P(cx0 + s, cy0 + s), 255, -1)
    hatch = np.full_like(img, 255)
    step = int(0.08 * PX_M)
    for k in range(-H, W + H, step):
        cv2.line(hatch, (k, 0), (k + H, H), 0, 2, cv2.LINE_AA)
    img = np.where(mask > 0, np.minimum(img, hatch), img)

    # Stütze
    cv2.circle(img, P(3.0, 6.0), int(0.15 * PX_M), 0, LW, cv2.LINE_AA)

    # Massketten unten
    yd = 8 + 0.9
    xs = [0, 5.0, 12.0]
    cv2.line(img, P(-0.3, yd), P(12.3, yd), 0, 2, cv2.LINE_AA)
    for x in xs:
        cv2.line(img, P(x, 8 + 0.25), P(x, yd + 0.15), 0, 2, cv2.LINE_AA)
        cv2.line(img, P(x - 0.1, yd + 0.1), P(x + 0.1, yd - 0.1), 0, 3, cv2.LINE_AA)
    # links senkrecht
    xd = -0.9
    ys = [0, 4.5, 8.0]
    cv2.line(img, P(xd, -0.3), P(xd, 8.3), 0, 2, cv2.LINE_AA)
    for y in ys:
        cv2.line(img, P(-0.25, y), P(xd - 0.15, y), 0, 2, cv2.LINE_AA)
        cv2.line(img, P(xd - 0.1, y + 0.1), P(xd + 0.1, y - 0.1), 0, 3, cv2.LINE_AA)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype(FONT, 46)
    fs = ImageFont.truetype(FONT, 40)
    fb = ImageFont.truetype(FONT, 64)

    def text(xm, ym, s, font=f):
        d.text(P(xm, ym), s, fill=0, font=font)

    text(1.0, 1.0, "Zimmer 01")
    text(6.3, 1.0, "Küche")
    text(9.4, 2.4, "Bad")
    text(6.0, 6.0, "Wohnen")
    text(2.6, 5.0, "Eingang")
    for (a, b) in ((0, 5.0), (5.0, 12.0)):
        s = f"{b - a:.2f}"
        tw = d.textlength(s, font=fs)
        x, y = P((a + b) / 2, yd)
        d.text((x - tw / 2, y - 58), s, fill=0, font=fs)
    # senkrechte Masszahlen (um 90° gedreht)
    for (a, b) in ((0, 4.5), (4.5, 8.0)):
        s = f"{b - a:.2f}"
        tw = int(d.textlength(s, font=fs)) + 6
        tmp = Image.new("L", (tw, 52), 255)
        ImageDraw.Draw(tmp).text((2, 0), s, fill=0, font=fs)
        tmp = tmp.rotate(90, expand=True)
        x, y = P(xd, (a + b) / 2)
        pil.paste(Image.composite(tmp, pil.crop((x - 60, y - tw // 2, x - 60 + tmp.width, y - tw // 2 + tmp.height)),
                                  tmp.point(lambda v: 255 if v < 128 else 0)), (x - 60, y - tw // 2))
    d.text(P(0, 10.0), "Grundriss EG", fill=0, font=fb)
    d.text(P(4.0, 10.15), "M 1:100", fill=0, font=f)
    return np.array(pil)


def photo_version(img: np.ndarray) -> np.ndarray:
    """Plan als Handyfoto: Perspektive, Tischhintergrund, Lichtverlauf, Rauschen, Unschärfe."""
    rng = np.random.default_rng(1)
    paper = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    paper = cv2.resize(paper, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    ph, pw = paper.shape[:2]
    out_w, out_h = 4032, 3024
    bg = np.zeros((out_h, out_w, 3), np.uint8)
    bg[:] = (70, 90, 115)  # Holztisch
    bg = cv2.add(bg, rng.integers(0, 25, bg.shape, dtype=np.uint8))
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[380, 260], [3700, 380], [3560, 2850], [260, 2700]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, Hm, (out_w, out_h))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), Hm, (out_w, out_h))
    out = np.where(mask[..., None] > 0, warped, bg)
    # Lichtverlauf + Papierton
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    light = 0.72 + 0.28 * (xx / out_w) * (1 - 0.4 * yy / out_h)
    out = (out.astype(np.float32) * light[..., None] * np.array([0.93, 0.97, 1.0])).clip(0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (5, 5), 1.1)
    out = np.clip(out.astype(np.float32) + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    plan = draw_plan()
    Image.fromarray(plan).save(OUT / "scan_300dpi.png", dpi=(DPI, DPI))

    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=W / DPI * 72, height=H / DPI * 72)
    page.insert_image(page.rect, filename=str(OUT / "scan_300dpi.png"))
    doc.save(OUT / "Bestand_Alt.pdf")

    photo = photo_version(plan)
    im = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif[0x010F] = "TestPhone"
    exif[0x0110] = "Model X"
    im.save(OUT / "handyfoto.jpg", quality=88, exif=exif)
    print("Beispiele erstellt in", OUT)


if __name__ == "__main__":
    main()


def hand_sketch() -> np.ndarray:
    """Handskizze: wackelige Bleistiftlinien, leicht schief, ungleichmässiger Strich."""
    rng = np.random.default_rng(7)
    img = np.full((2400, 3200), 246, np.uint8)
    s = 230.0  # px pro m

    def wob(p, q, amp=2.2):
        n = max(2, int(math.dist(p, q) / 25))
        t = np.linspace(0, 1, n)
        pts = np.outer(1 - t, p) + np.outer(t, q)
        d = np.array(q) - np.array(p)
        nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        off = np.cumsum(rng.normal(0, amp * 0.35, n))
        off -= np.linspace(off[0], off[-1], n)
        pts += np.outer(off, nrm)
        return pts.astype(np.int32)

    def line(a, b, w=None):
        a = (300 + a[0] * s + rng.normal(0, 1.5), 300 + a[1] * s + rng.normal(0, 1.5))
        b = (300 + b[0] * s + rng.normal(0, 1.5), 300 + b[1] * s + rng.normal(0, 1.5))
        pts = wob(a, b)
        cv2.polylines(img, [pts], False, int(rng.integers(55, 95)), int(w or rng.integers(3, 6)), cv2.LINE_AA)

    # Aussenwände doppelt (25 cm), Innenwand doppelt (12 cm)
    for o in (0.0, 0.25):
        line((o, o), (10 - o, o)); line((10 - o, o), (10 - o, 7 - o))
        line((10 - o, 7 - o), (o, 7 - o)); line((o, 7 - o), (o, o))
    for x in (4.0, 4.12):
        line((x, 0.25), (x, 3.0)); line((x, 3.9), (x, 6.75))
    # Tür mit Bogen
    line((4.12, 3.0), (5.0, 3.0))
    c = (int(300 + 4.12 * s), int(300 + 3.0 * s))
    cv2.ellipse(img, c, (int(0.88 * s), int(0.88 * s)), 0, 0, 90, 70, 3, cv2.LINE_AA)
    # Treppe
    for i in range(8):
        line((0.6, 4.0 + i * 0.3), (1.8, 4.0 + i * 0.3), 3)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 60)
    d.text((300 + 1.0 * s, 300 + 1.5 * s), "Zimmer", fill=60, font=f)
    d.text((300 + 6.0 * s, 300 + 3.0 * s), "Wohnen", fill=60, font=f)
    img = np.array(pil)
    # leicht drehen (Scan schief eingelegt) + Rauschen
    R = cv2.getRotationMatrix2D((1600, 1200), 2.3, 1.0)
    img = cv2.warpAffine(img, R, (3200, 2400), borderValue=246)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return img


if __name__ == "__main__":
    Image.fromarray(hand_sketch()).save(OUT / "handskizze.jpg", quality=90)

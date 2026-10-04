"""Realistischer Bestandesplan-Testplan (Schweizer Darstellung) mit exakt bekannter Geometrie.

    python3 tests/make_bestand.py   ->  tests/samples/bestand_100.png (+ _50, _foto) und bestand_truth.json

* Aussenwände 36 cm Mauerwerk (Schraffur 45°, kräftiger Umriss)
* Innenwände 15 cm Mauerwerk schraffiert, 12 cm Leichtbau (nur Umriss), Betonwand 20 cm schwarz
* Fenster mit schrägen inneren Leibungen, Rahmen, Glas (Doppellinie) und Fensterbank aussen
* Türen mit Türblatt (Rechteck) und Anschlagbogen, Eingangstür
* Massketten (2 Ebenen) mit Schrägstrichen und Masszahlen, Raumstempel mit Rahmen
* Möblierung, Nordpfeil, Titel
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BW, BH = 11.0, 9.0           # Gebäude aussen (m)
TE = 0.36                    # Aussenwand
WINDOWS = [                  # (Wand, von, bis) in m entlang der Wand (aussen gemessen)
    ("top", 1.2, 2.4), ("top", 5.6, 6.8), ("top", 8.8, 10.0),
    ("right", 5.2, 6.6), ("bottom", 1.5, 3.1), ("left", 2.0, 3.2),
]
SPLAY = 0.10                 # Leibung innen je Seite breiter
FRAME_D, FRAME_T = 0.12, 0.07  # Rahmen: Abstand von aussen, Rahmentiefe
ENTRY = ("bottom", 6.0, 7.0)
INNER = [  # (x0, y0, x1, y1, Stil) – Innenwände als Rechtecke
    (4.50, TE, 4.65, BH - TE, "hatch"),          # 15 cm Mauerwerk
    (4.65, 4.20, BW - TE, 4.32, "light"),        # 12 cm Leichtbau
    (7.80, TE, 7.92, 4.20, "light"),             # 12 cm Leichtbau
    (4.65, 6.60, 6.60, 6.80, "concrete"),        # 20 cm Beton
]
IDOORS = [  # (Wand-Index, entlang von, bis, Drehpunkt-Ende 0/1, Seite -1/+1)
    (0, 5.0, 5.9, 0, +1),
    (1, 8.6, 9.5, 1, +1),
    (2, 2.6, 3.5, 0, -1),
]
ROOMS = [  # Nr, Name, Belag, Stempelposition (m), Rechteck(e) innen
    ("01", "Zimmer", "Parkett", (2.4, 6.3), [(TE, TE, 4.50, BH - TE)]),
    ("02", "Küche", "Platten", (6.2, 2.2), [(4.65, TE, 7.80, 4.20)]),
    ("03", "Bad", "Platten", (9.3, 1.6), [(7.92, TE, BW - TE, 4.20)]),
    ("04", "Wohnen", "Parkett", (8.4, 6.9), [(4.65, 4.32, BW - TE, BH - TE)]),
]


def build(px_m: float, margin_m: float = 3.2):
    W = int((BW + 2 * margin_m + 1.5) * px_m)
    H = int((BH + 2 * margin_m + 1.0) * px_m)
    ox, oy = margin_m * px_m, margin_m * px_m

    def P(x, y):
        return (int(round(ox + x * px_m)), int(round(oy + y * px_m)))

    def PF(pts):
        return np.array([[ox + x * px_m, oy + y * px_m] for x, y in pts], np.float64)

    lw_wall = 6                     # 0.5 mm Papier bei 300 dpi
    lw_thin = 3                     # 0.25 mm
    lw_hatch = 2                    # 0.18 mm

    wall = np.zeros((H, W), np.uint8)          # Wahrheit: Wandfläche
    style = np.zeros((H, W), np.uint8)         # 1 hatch, 2 light, 3 concrete
    # Aussenwände
    cv2.rectangle(wall, P(0, 0), P(BW, BH), 255, -1)
    cv2.rectangle(wall, P(TE, TE), P(BW - TE, BH - TE), 0, -1)
    style[wall > 0] = 1
    for x0, y0, x1, y1, st in INNER:
        m = np.zeros_like(wall)
        cv2.rectangle(m, P(x0, y0), P(x1, y1), 255, -1)
        wall[m > 0] = 255
        style[m > 0] = {"hatch": 1, "light": 2, "concrete": 3}[st]

    # Öffnungen ausschneiden (Fenster mit schräger Leibung)
    def wall_frame(side, a0, a1):
        """Öffnungspolygon in m für eine Aussenwand-Öffnung (mit Leibungsschräge innen)."""
        f = FRAME_D + FRAME_T
        if side == "top":
            return [(a0, -0.01), (a1, -0.01), (a1, f), (a1 + SPLAY, TE + 0.01), (a0 - SPLAY, TE + 0.01), (a0, f)]
        if side == "bottom":
            return [(a0, BH + 0.01), (a1, BH + 0.01), (a1, BH - f), (a1 + SPLAY, BH - TE - 0.01),
                    (a0 - SPLAY, BH - TE - 0.01), (a0, BH - f)]
        if side == "left":
            return [(-0.01, a0), (-0.01, a1), (f, a1), (TE + 0.01, a1 + SPLAY), (TE + 0.01, a0 - SPLAY), (f, a0)]
        return [(BW + 0.01, a0), (BW + 0.01, a1), (BW - f, a1), (BW - TE - 0.01, a1 + SPLAY),
                (BW - TE - 0.01, a0 - SPLAY), (BW - f, a0)]

    truth = {"px_m": px_m, "origin_px": [ox, oy], "windows": [], "doors": [], "rooms": [], "dims": 0}
    for side, a0, a1 in WINDOWS:
        poly = wall_frame(side, a0, a1)
        cv2.fillPoly(wall, [PF(poly).astype(np.int32)], 0)
        truth["windows"].append({"side": side, "from": a0, "to": a1})
    side, a0, a1 = ENTRY
    cv2.rectangle(wall, P(a0, BH - TE - 0.01), P(a1, BH + 0.01), 0, -1)
    truth["doors"].append({"wall": "bottom", "from": a0, "to": a1})
    for wi, a0, a1, hinge, s in IDOORS:
        x0, y0, x1, y1, _ = INNER[wi]
        if x1 - x0 < y1 - y0:   # senkrecht
            cv2.rectangle(wall, P(x0 - 0.01, a0), P(x1 + 0.01, a1), 0, -1)
        else:
            cv2.rectangle(wall, P(a0, y0 - 0.01), P(a1, y1 + 0.01), 0, -1)
        truth["doors"].append({"wall": wi, "from": a0, "to": a1})
    style[wall == 0] = 0

    img = np.full((H, W), 255, np.uint8)
    # Füllungen
    hatch = np.full((H, W), 255, np.uint8)
    step = max(4, int(0.06 * px_m))
    for k in range(-H, W + H, step):
        cv2.line(hatch, (k, 0), (k + H, H), 0, lw_hatch, cv2.LINE_AA)
    img = np.where(style == 1, np.minimum(img, hatch), img)
    img[style == 3] = 0
    # Umrisse
    cs, _ = cv2.findContours(wall, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, 0, lw_wall, cv2.LINE_AA)

    def L(a, b, w=lw_thin):
        cv2.line(img, P(*a), P(*b), 0, w, cv2.LINE_AA)

    # Fenster: Rahmen, Glas (Doppellinie), Fensterbank aussen
    for side, a0, a1 in WINDOWS:
        f0, f1 = FRAME_D, FRAME_D + FRAME_T
        g = 0.012
        if side in ("top", "bottom"):
            sgn = 1 if side == "top" else -1
            base = 0 if side == "top" else BH
            yy = lambda d: base + sgn * d  # noqa: E731
            cv2.rectangle(img, P(a0, yy(f0)), P(a1, yy(f1)), 0, lw_thin, cv2.LINE_AA)
            for d in ((f0 + f1) / 2 - g, (f0 + f1) / 2 + g):
                L((a0, yy(d)), (a1, yy(d)), 1)
            L((a0 - 0.04, yy(-0.04)), (a1 + 0.04, yy(-0.04)))
            L((a0 - 0.04, yy(-0.04)), (a0 - 0.04, yy(0)))
            L((a1 + 0.04, yy(-0.04)), (a1 + 0.04, yy(0)))
        else:
            sgn = 1 if side == "left" else -1
            base = 0 if side == "left" else BW
            xx = lambda d: base + sgn * d  # noqa: E731
            cv2.rectangle(img, P(xx(f0), a0), P(xx(f1), a1), 0, lw_thin, cv2.LINE_AA)
            for d in ((f0 + f1) / 2 - g, (f0 + f1) / 2 + g):
                L((xx(d), a0), (xx(d), a1), 1)
            L((xx(-0.04), a0 - 0.04), (xx(-0.04), a1 + 0.04))
            L((xx(-0.04), a0 - 0.04), (xx(0), a0 - 0.04))
            L((xx(-0.04), a1 + 0.04), (xx(0), a1 + 0.04))

    # Türen: Blatt (Rechteck 4 cm) + Bogen
    def door(hx, hy, leaf_dir, jamb_dir, r):
        lx, ly = leaf_dir
        jx, jy = jamb_dir
        t = 0.04
        tip = (hx + lx * r, hy + ly * r)
        pts = [(hx, hy), tip, (tip[0] + jx * t, tip[1] + jy * t), (hx + jx * t, hy + jy * t)]
        cv2.polylines(img, [PF(pts).astype(np.int32)], True, 0, lw_thin, cv2.LINE_AA)
        a0 = math.degrees(math.atan2(ly, lx))
        a1 = math.degrees(math.atan2(jy, jx))
        d = (a1 - a0 + 180) % 360 - 180
        ts = np.radians(np.linspace(a0, a0 + d, 40))
        arc = PF(np.c_[hx + r * np.cos(ts), hy + r * np.sin(ts)])
        cv2.polylines(img, [arc.astype(np.int32)], False, 0, max(1, lw_thin - 1), cv2.LINE_AA)

    door(6.0, BH - TE, (0, -1), (1, 0), 1.0)           # Eingang, schlägt nach innen
    for wi, a0, a1, hinge, s in IDOORS:
        x0, y0, x1, y1, _ = INNER[wi]
        r = a1 - a0 - 0.05
        if x1 - x0 < y1 - y0:
            fx = x1 if s > 0 else x0
            hy = a0 if hinge == 0 else a1
            door(fx, hy + (0.025 if hinge == 0 else -0.025), (s, 0), (0, 1 if hinge == 0 else -1), r)
        else:
            fy = y1 if s > 0 else y0
            hx = a0 if hinge == 0 else a1
            door(hx + (0.025 if hinge == 0 else -0.025), fy, (0, s), (1 if hinge == 0 else -1, 0), r)

    # Möblierung (dünn)
    def R(x0, y0, x1, y1):
        cv2.rectangle(img, P(x0, y0), P(x1, y1), 0, lw_hatch + 1, cv2.LINE_AA)
    R(0.7, 0.7, 2.3, 2.7); R(0.8, 0.8, 1.45, 1.1); R(1.55, 0.8, 2.2, 1.1)          # Bett
    R(TE + 0.02, 3.6, TE + 0.62, 5.6)                                                # Schrank
    R(4.7, TE + 0.02, 7.7, TE + 0.62)                                                # Küchenzeile
    for dx in (5.9, 6.2):
        for dy in (0.5, 0.75):
            cv2.circle(img, P(dx, dy), int(0.09 * px_m), 0, lw_hatch + 1, cv2.LINE_AA)
    R(8.1, 0.5, 9.8, 1.25); R(10.0, 3.2, 10.6, 3.9)                                  # Wanne, Dusche
    R(5.2, 7.6, 7.4, 8.5); R(5.2, 7.6, 7.4, 7.8)                                    # Sofa
    R(8.6, 5.4, 10.0, 6.3)                                                           # Tisch
    for x in (8.7, 9.3):
        R(x, 4.9, x + 0.42, 5.3); R(x, 6.4, x + 0.42, 6.8)

    # Massketten unten und links (Schweizer Schrägstriche)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    fs = max(10, int(0.25 * px_m))
    f = ImageFont.truetype(FONT, fs)
    fb = ImageFont.truetype(FONT_B, int(fs * 1.1))

    def tick(x, y):
        s = 0.08
        dr.line([P(x - s, y + s), P(x + s, y - s)], fill=0, width=lw_wall)

    n_dims = 0
    for level, pts in ((1, [0, 1.5, 3.1, 6.0, 7.0, BW]), (2, [0, BW])):
        y = BH + 0.7 * level + 0.1
        dr.line([P(pts[0] - 0.2, y), P(pts[-1] + 0.2, y)], fill=0, width=lw_hatch + 1)
        for x in pts:
            dr.line([P(x, BH + 0.2), P(x, y + 0.15)], fill=0, width=lw_hatch + 1)
            tick(x, y)
        for a, b in zip(pts[:-1], pts[1:]):
            s = f"{b - a:.2f}"
            tw = dr.textlength(s, font=f)
            xm, _ = P((a + b) / 2, y)
            dr.text((xm - tw / 2, P(0, y)[1] - fs - 4), s, fill=0, font=f)
            n_dims += 1
    xs = [0, 2.0, 3.2, 5.2, 6.6, BH]
    x = -0.8
    dr.line([P(x, -0.2), P(x, BH + 0.2)], fill=0, width=lw_hatch + 1)
    for v in xs:
        dr.line([P(-0.2, v), P(x - 0.15, v)], fill=0, width=lw_hatch + 1)
        tick(x, v)
    for a, b in zip(xs[:-1], xs[1:]):
        s = f"{b - a:.2f}"
        tw = int(dr.textlength(s, font=f)) + 6
        tmp = Image.new("L", (tw, fs + 8), 255)
        ImageDraw.Draw(tmp).text((3, 0), s, fill=0, font=f)
        rot = tmp.rotate(90, expand=True)
        cx, cy = P(x, (a + b) / 2)
        pil.paste(rot, (cx - rot.size[0] - 4, cy - rot.size[1] // 2))
        n_dims += 1
    truth["dims"] = n_dims

    # Raumstempel mit Rahmen
    for nr, name, belag, (sx, sy), rects in ROOMS:
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects)
        if nr == "04":     # Wohnen: Beton-Wandstück abziehen
            area -= (6.60 - 4.65) * 0.20
        lines = [(nr + "  " + name, fb), (f"{area:.1f} m2", f), (belag, f)]
        hgt = sum(int(fn.size * 1.35) for _, fn in lines)
        wid = max(int(dr.textlength(t, font=fn)) for t, fn in lines) + 24
        cx, cy = P(sx, sy)
        x0b, y0b = cx - wid // 2, cy - hgt // 2 - 8
        dr.rectangle([x0b, y0b, x0b + wid, y0b + hgt + 16], outline=0, width=lw_hatch + 1)
        yy = y0b + 8
        for t, fn in lines:
            tw = dr.textlength(t, font=fn)
            dr.text((cx - tw / 2, yy), t, fill=0, font=fn)
            yy += int(fn.size * 1.35)
        truth["rooms"].append({"nr": nr, "name": name, "area": round(area, 1), "belag": belag})

    dr.text(P(0, BH + 2.4), "GRUNDRISS ERDGESCHOSS   BESTAND", fill=0, font=fb)
    dr.text(P(0, BH + 2.85), f"M 1:{int(round(300 / 25.4 * 1000 / px_m))}", fill=0, font=f)
    img = np.array(pil)
    # Nordpfeil
    c = P(BW + 1.0, -1.2)
    r = int(0.5 * px_m)
    cv2.circle(img, c, r, 0, lw_thin, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[c[0], c[1] - r], [c[0] - r // 3, c[1] + r // 3], [c[0], c[1]],
                                 [c[0] + r // 3, c[1] + r // 3]], np.int32)], 0, cv2.LINE_AA)
    truth["wall_mask_bbox"] = [int(ox), int(oy), int(ox + BW * px_m), int(oy + BH * px_m)]
    return img, wall, truth


def photo(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    rng = np.random.default_rng(3)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0.04 * w, 0.06 * h], [0.97 * w, 0.0], [w, 0.97 * h], [0.0, h]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, Hm, (w, h), flags=cv2.INTER_LINEAR, borderValue=235).astype(np.float32)
    g = np.linspace(0.82, 1.0, w)[None, :] * np.linspace(0.9, 1.0, h)[:, None]
    out = out * g + rng.normal(0, 4, out.shape)
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    return np.clip(out, 0, 255).astype(np.uint8), Hm


def main():
    OUT.mkdir(exist_ok=True)
    all_truth = {}
    for name, px_m in (("bestand_100", 300 / 25.4 * 10), ("bestand_50", 300 / 25.4 * 20)):
        img, wall, truth = build(px_m)
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        cv2.imwrite(str(OUT / f"{name}_wall.png"), wall)
        all_truth[name] = truth
        if name == "bestand_100":
            ph, Hm = photo(img)
            cv2.imwrite(str(OUT / "bestand_foto.jpg"), ph, [cv2.IMWRITE_JPEG_QUALITY, 88])
            all_truth["bestand_foto"] = {**truth, "H": Hm.tolist()}
    (OUT / "bestand_truth.json").write_text(json.dumps(all_truth, indent=1))
    print("ok", list(all_truth))


if __name__ == "__main__":
    main()

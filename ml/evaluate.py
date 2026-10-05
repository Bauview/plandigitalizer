"""Auswertung auf dem Testdatensatz (Pläne, die das Netz nie gesehen hat).

    python3 ml/evaluate.py DATADIR [n]

Messgrössen
* wall_iou      – Übereinstimmung der rekonstruierten Wandflächen mit der Wahrheit
* door/window   – Präzision, Trefferquote, F1 der erkannten Öffnungen (Zuordnung über Überlappung)
* width_err     – mittlere Abweichung der Öffnungsbreite (in Wandstärken)
* angle_err     – Fehler der automatischen Ausrichtung (Grad) bei schräg erfassten Plänen
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline import PlanError, essential, semantic, stage_prepare  # noqa: E402


def gt_openings(lab):
    out = []
    for kind, cls in (("door", 3), ("window", 2)):
        n, cl, st, _ = cv2.connectedComponentsWithStats((lab == cls).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a >= 6:
                out.append((kind, x, y, x + w, y + h))
    return out


def match(pred, gt):
    used = set()
    tp = 0
    werr = []
    for kind, x0, y0, x1, y1 in pred:
        best, bi = 0.0, None
        for j, (k2, a0, b0, a1, b1) in enumerate(gt):
            if j in used or k2 != kind:
                continue
            ix = min(x1, a1) - max(x0, a0)
            iy = min(y1, b1) - max(y0, b0)
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            u = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if inter / u > best:
                best, bi = inter / u, j
        if bi is not None and best > 0.25:
            used.add(bi)
            tp += 1
            g = gt[bi]
            werr.append(abs(max(x1 - x0, y1 - y0) - max(g[3] - g[1], g[4] - g[2])))
    return tp, len(pred) - tp, len(gt) - tp, werr


def main():
    root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    meta = {}
    for line in open(root / "meta.jsonl"):
        r = json.loads(line)
        if r.get("split") == "test":
            meta[r["name"]] = r
    files = sorted((root / "test").glob("*_img.png"))[:n]
    stats = {"door": [0, 0, 0], "window": [0, 0, 0]}
    werrs = []
    inter = union = 0
    skipped = 0
    rinter = runion = 0
    angle_errs = []
    for f in files:
        name = f.name.split("_")[0]
        m = meta.get(name, {})
        lab0 = cv2.imread(str(f).replace("_img", "_lab"), cv2.IMREAD_GRAYSCALE)
        try:
            st = stage_prepare(f)                   # echte Kette: Grösse, Perspektive, Ausrichtung, Netz
        except PlanError:
            skipped += 1                            # zu kleines Bild – wird auch im Werkzeug abgelehnt
            continue
        h, w = st.prep.gray.shape
        lab = cv2.warpPerspective(lab0, st.prep.M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        theta = m.get("theta", 0.0)
        if abs(theta) > 0.5:
            # Restschräglage der Wahrheit nach der automatischen Ausrichtung
            ang, _ = semantic.dominant_angle(((lab > 0) * 255).astype(np.uint8))
            angle_errs.append(abs(ang))
        sp = essential.reconstruct(st.seg.labels, st.prep.binary, st.prep.gray)
        gt = gt_openings(lab)
        t = max(1.0, m.get("wall_px", 8.0) * float(np.sqrt(abs(np.linalg.det(st.prep.M[:2, :2])))))
        if sp is None:
            for k in stats:
                stats[k][2] += sum(1 for g in gt if g[0] == k)
            continue
        pred = [(o.kind, o.x0, o.y0, o.x1, o.y1) for o in sp.openings]
        for k in stats:
            tp, fp, fn, we = match([p for p in pred if p[0] == k], [g for g in gt if g[0] == k])
            stats[k][0] += tp
            stats[k][1] += fp
            stats[k][2] += fn
            werrs.extend([w_ / t for w_ in we])
        gw = lab == 1
        pw = sp.walls > 0
        inter += np.count_nonzero(gw & pw)
        union += np.count_nonzero(gw | pw)
        rinter += np.count_nonzero(gw & (st.seg.labels == 1))
        runion += np.count_nonzero(gw | (st.seg.labels == 1))
    res = {"n": len(files) - skipped, "wall_iou": round(inter / max(1, union), 4),
           "wall_iou_net": round(rinter / max(1, runion), 4)}
    for k, (tp, fp, fn) in stats.items():
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        res[k] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
                  "tp": tp, "fp": fp, "fn": fn}
    res["width_err_walls"] = round(float(np.mean(werrs)), 3) if werrs else None
    res["angle_err_deg"] = {"mean": round(float(np.mean(angle_errs)), 3), "max": round(float(np.max(angle_errs)), 3),
                            "n": len(angle_errs)} if angle_errs else None
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

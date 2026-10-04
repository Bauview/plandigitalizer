"""Erzeugt den synthetischen Trainingsdatensatz.

    python3 ml/make_dataset.py RESPLAN.pkl split.json OUTDIR [n_train n_val n_test]

Bilder:  OUTDIR/<split>/<id>_img.png, Klassen: OUTDIR/<split>/<id>_lab.png, Metadaten: meta.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.synth import load_resplan, render  # noqa: E402

DATA = None


def _init(pkl):
    global DATA
    DATA = load_resplan(pkl)
    cv2.setNumThreads(1)


def _job(args):
    split, idx, seed, out = args
    try:
        r = render(DATA[idx], seed)
    except Exception as exc:  # noqa: BLE001
        return {"split": split, "idx": idx, "error": repr(exc)}
    if r is None:
        return None
    name = f"{seed:07d}"
    d = Path(out) / split
    cv2.imwrite(str(d / f"{name}_img.png"), r["image"])
    cv2.imwrite(str(d / f"{name}_lab.png"), r["label"])
    return {"split": split, "name": name, "idx": idx, "wall_px": round(r["wall_px"], 2),
            "theta": round(r["theta"], 2), "shape": list(r["image"].shape)}


def main():
    pkl, split_json, out = sys.argv[1:4]
    n = [int(x) for x in sys.argv[4:7]] if len(sys.argv) >= 7 else [7000, 300, 300]
    data_ids = {p["id"]: i for i, p in enumerate(load_resplan(pkl))}
    splits = json.load(open(split_json))
    jobs = []
    off = int(os.environ.get("SEED_OFFSET", "0"))
    for (split, count), base in zip(zip(("train", "val", "test"), n), (off, 5_000_000 + off, 6_000_000 + off)):
        (Path(out) / split).mkdir(parents=True, exist_ok=True)
        ids = [data_ids[i] for i in splits[split] if i in data_ids]
        for k in range(count):
            jobs.append((split, ids[(k * 7919) % len(ids)], base + k, out))
    with Pool(int(os.environ.get("WORKERS", "2")), initializer=_init, initargs=(pkl,)) as pool, \
            open(Path(out) / "meta.jsonl", "a") as meta:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            if r:
                meta.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                print(i, "/", len(jobs), flush=True)
                meta.flush()


if __name__ == "__main__":
    main()

"""Training des PlanNet auf den synthetischen Plänen.

    python3 ml/train.py DATADIR OUT.pt [--iters N] [--resume CKPT]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pipeline.preprocess import _normalize_background  # noqa: E402
from ml.model import N_CLASSES, PlanNet  # noqa: E402

CROP = 288


def to_input(gray: np.ndarray) -> np.ndarray:
    """Netz-Eingabe: Tinte positiv, 0..1."""
    return 1.0 - gray.astype(np.float32) / 255.0


class Crops(IterableDataset):
    def __init__(self, root: Path, split: str, seed: int = 0):
        self.files = sorted((root / split).glob("*_img.png"))
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (info.id if info else 0) * 7777 + int(time.time()))
        while True:
            f = rng.choice(self.files)
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            lab = cv2.imread(str(f).replace("_img", "_lab"), cv2.IMREAD_GRAYSCALE)
            if img is None or lab is None:
                continue
            if rng.random() < 0.5:
                img = _normalize_background(img)
            # Skalierung
            s = math.exp(rng.uniform(math.log(0.75), math.log(1.3)))
            if abs(s - 1) > 0.03:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            for _ in range(6):     # mehrere Ausschnitte pro Bild
                yield self._crop(img, lab, rng)

    def _crop(self, img, lab, rng):
        h, w = img.shape
        if h < CROP or w < CROP:
            ph, pw = max(0, CROP - h), max(0, CROP - w)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=int(np.median(img)))
            lab = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape
        if rng.random() < 0.8:
            ys, xs = np.nonzero(lab[::4, ::4])
            if len(xs):
                i = rng.randrange(len(xs))
                cx, cy = xs[i] * 4, ys[i] * 4
            else:
                cx, cy = rng.randrange(w), rng.randrange(h)
        else:
            cx, cy = rng.randrange(w), rng.randrange(h)
        x0 = min(max(0, cx - CROP // 2 + rng.randint(-60, 60)), w - CROP)
        y0 = min(max(0, cy - CROP // 2 + rng.randint(-60, 60)), h - CROP)
        im = img[y0:y0 + CROP, x0:x0 + CROP]
        lb = lab[y0:y0 + CROP, x0:x0 + CROP]
        k = rng.randrange(4)
        im, lb = np.rot90(im, k), np.rot90(lb, k)
        if rng.random() < 0.5:
            im, lb = im[:, ::-1], lb[:, ::-1]
        x = to_input(np.ascontiguousarray(im))
        if rng.random() < 0.3:     # Kontrast/Helligkeit
            x = np.clip(x * rng.uniform(0.6, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(x[None].copy()), torch.from_numpy(np.ascontiguousarray(lb).astype(np.int64))


def loss_fn(logits, target):
    # Ausgabe hat halbe Auflösung -> Ziel verkleinern (Mehrheitsklasse über 'nearest' ist ausreichend)
    t = target[:, ::2, ::2]
    w = torch.tensor([1.0, 1.5, 3.0, 3.0], device=logits.device)
    ce = Fn.cross_entropy(logits, t, weight=w)
    p = logits.softmax(1)
    oh = Fn.one_hot(t, N_CLASSES).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - (2 * inter[1:] + 1) / (den[1:] + 1)
    return ce + dice.mean()


@torch.no_grad()
def evaluate(model, root: Path, split="val", n=60, max_side=1100):
    model.eval()
    files = sorted((root / split).glob("*_img.png"))[:n]
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(f).replace("_img", "_lab"), cv2.IMREAD_GRAYSCALE)
        k = min(1.0, max_side / max(img.shape))
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        x = np.zeros((H, W), np.float32)
        x[:h, :w] = to_input(img)
        out = model(torch.from_numpy(x)[None, None])
        pred = out.argmax(1)[0].numpy().astype(np.uint8)
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)[:h, :w]
        for c in range(1, N_CLASSES):
            inter[c] += np.sum((pred == c) & (lab == c))
            union[c] += np.sum((pred == c) | (lab == c))
    model.train()
    return {name: round(float(inter[c] / max(1, union[c])), 4) for c, name in ((1, "wall"), (2, "window"), (3, "door"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--resume")
    a = ap.parse_args()
    torch.set_num_threads(2)
    root = Path(a.data)
    model = PlanNet().to(memory_format=torch.channels_last)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        start = ck.get("iter", 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05,
                                                last_epoch=start - 1 if start else -1) if not start else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.iters - start), eta_min=a.lr * 0.01)
    dl = DataLoader(Crops(root, "train"), batch_size=a.bs, num_workers=1, prefetch_factor=4)
    log = open(Path(a.out).with_suffix(".log"), "a")
    t0 = time.time()
    run = 0.0
    best = -1.0
    for it, (x, y) in enumerate(dl, start=start):
        if it >= a.iters:
            break
        loss = loss_fn(model(x.to(memory_format=torch.channels_last)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > start else loss.item()
        if it % 100 == 0:
            msg = f"it {it} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if (it + 1) % 1500 == 0 or it + 1 == a.iters:
            ev = evaluate(model, root)
            score = sum(ev.values())
            msg = f"EVAL it {it + 1} {json.dumps(ev)}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "iter": it + 1, "eval": ev}, a.out)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "iter": it + 1, "eval": ev},
                           str(Path(a.out).with_suffix("")) + "_best.pt")


if __name__ == "__main__":
    main()

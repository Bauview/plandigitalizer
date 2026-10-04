"""Exportiert ein Trainings-Checkpoint als ONNX für OpenCV-DNN (Browser + Desktop).

    python3 ml/export_onnx.py CKPT.pt app/models/plannet.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.model import PlanNet  # noqa: E402


def main():
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    m = PlanNet()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, torch.zeros(1, 1, 256, 256), str(out), opset_version=13, input_names=["x"],
                      output_names=["y"], dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h2", 3: "w2"}},
                      dynamo=False, do_constant_folding=True)
    import cv2
    net = cv2.dnn.readNetFromONNX(str(out))
    x = np.random.rand(1, 1, 320, 448).astype(np.float32)
    net.setInput(x)
    o = net.forward()
    with torch.no_grad():
        r = m(torch.from_numpy(x)).numpy()
    print("ONNX", out, f"{out.stat().st_size / 1e6:.2f} MB", "max diff", float(np.abs(o - r).max()))


if __name__ == "__main__":
    main()

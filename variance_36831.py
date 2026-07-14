#!/usr/bin/env python3
"""#36831 — the dynamic-shape GPU path is not stably wrong, it is unstable.

Compiles the SAME dynamic model N times (fresh compile, no model cache) and runs
the SAME fixed input through it. A correct plugin returns the same value every time.
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov

N = 3
CASES = [
    ("detection_scrfd_det_10g", "det_10g.onnx", (1, 3, 640, 640)),
    ("recognition_arcface_w600k_r50", "w600k_r50.onnx", (1, 3, 112, 112)),
]

root = Path(sys.argv[1])
core = ov.Core()

for name, fname, shape in CASES:
    base = ov.convert_model(str(root / fname))
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    static = base.clone()
    static.reshape({0: ov.PartialShape(list(shape))})
    cpu = list(core.compile_model(static, "CPU")(x).values())[0]
    print(f"{name}\n  CPU (ground truth)      mean|out| = {np.abs(cpu).mean():.6f}")

    for i in range(N):
        model = ov.convert_model(str(root / fname))  # fresh, dynamic, as shipped
        out = list(core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})(x).values())[0]
        print(
            f"  GPU dynamic, compile #{i + 1}  mean|out| = {np.abs(out).mean():.6f}"
            f"   diff vs CPU = {np.abs(out - cpu).mean():.6f}"
        )
    print()

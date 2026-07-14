#!/usr/bin/env python3
"""Load the shipped IR back from disk and confirm they still reproduce the bug on GPU."""
import sys
from pathlib import Path

import numpy as np
import openvino as ov

ir = Path(sys.argv[1])
core = ov.Core()

CASES = [
    ("detection_scrfd_det_10g", (1, 3, 640, 640)),
    ("recognition_arcface_w600k_r50", (1, 3, 112, 112)),
]

for name, shape in CASES:
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    ref = None
    for variant in ("dynamic", "bounded", "static"):
        xml = ir / f"{name}.{variant}.xml"
        m = core.read_model(str(xml))
        print(f"{name}.{variant:<8} input partial shape from disk: {m.inputs[0].partial_shape}")
        if ref is None:  # CPU ground truth from the static IR
            st = core.read_model(str(ir / f"{name}.static.xml"))
            ref = list(core.compile_model(st, "CPU")(x).values())[0]
        out = list(core.compile_model(m, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})(x).values())[0]
        print(
            f"    GPU mean|out| = {np.abs(out).mean():<28.6f} diff vs CPU = {np.abs(out - ref).mean():.6f}"
        )
    print()

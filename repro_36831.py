#!/usr/bin/env python3
"""
openvinotoolkit/openvino#36831 — self-contained repro + IR exporter.

GPU plugin returns garbage/near-zero output for dynamic-shape conv models
(InsightFace SCRFD det_10g / ArcFace w600k_r50) on a Gen11 iGPU (arch v11.2.0).
Static reshape fixes it. CPU is always correct.

Usage (inside a container with /dev/dri and the Intel OpenCL ICD):
    pip install openvino==2025.4.1 numpy
    python repro_36831.py /path/to/buffalo_l [--export-ir OUTDIR]
"""
import sys
from pathlib import Path

import numpy as np
import openvino as ov

MODELS = {
    "detection_scrfd_det_10g": {
        "file": "det_10g.onnx",
        "static": (1, 3, 640, 640),
        "bounded": [1, 3, (320, 640), (320, 640)],
    },
    "recognition_arcface_w600k_r50": {
        "file": "w600k_r50.onnx",
        "static": (1, 3, 112, 112),
        "bounded": [(1, 32), 3, 112, 112],
    },
}

FP32 = {"INFERENCE_PRECISION_HINT": "f32"}


def variants(onnx_path, spec):
    """dynamic (as shipped) / bounded dynamic / fully static."""
    dynamic = ov.convert_model(onnx_path)

    bounded = dynamic.clone()
    bounded.reshape({0: ov.PartialShape(spec["bounded"])})

    static = dynamic.clone()
    static.reshape({0: ov.PartialShape(list(spec["static"]))})

    return {"dynamic": dynamic, "bounded": bounded, "static": static}


def main():
    root = Path(sys.argv[1])
    outdir = None
    if "--export-ir" in sys.argv:
        outdir = Path(sys.argv[sys.argv.index("--export-ir") + 1])
        outdir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    print("openvino          ", ov.get_version())
    print("available devices ", core.available_devices)
    for prop in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "OPTIMIZATION_CAPABILITIES"):
        print(f"GPU {prop:<26}", core.get_property("GPU", prop))
    print()

    header = f"{'model':<30} {'variant':<9} {'GPU mean|out|':>14} {'mean abs diff vs CPU':>21}"
    print(header)
    print("-" * len(header))

    for name, spec in MODELS.items():
        onnx_path = root / spec["file"]
        vs = variants(str(onnx_path), spec)

        # fixed seeded input at the primary (static) shape — identical for all variants
        x = np.random.default_rng(0).standard_normal(spec["static"]).astype(np.float32)

        # ground truth: CPU plugin on the static model
        cpu = list(core.compile_model(vs["static"], "CPU")(x).values())[0]

        for variant in ("dynamic", "bounded", "static"):
            model = vs[variant]
            out = list(core.compile_model(model, "GPU", FP32)(x).values())[0]
            print(
                f"{name:<30} {variant:<9} {np.abs(out).mean():>14.6f} "
                f"{np.abs(out - cpu).mean():>21.6f}"
            )

            if outdir:
                ov.save_model(model, str(outdir / f"{name}.{variant}.xml"), compress_to_fp16=False)
        print()


if __name__ == "__main__":
    main()

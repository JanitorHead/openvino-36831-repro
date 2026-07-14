# Repro pack for [openvinotoolkit/openvino#36831](https://github.com/openvinotoolkit/openvino/issues/36831)

> GPU plugin returns garbage / near-zero output for **dynamic-shape conv models**
> (InsightFace SCRFD `det_10g` + ArcFace `w600k_r50`) on a **Gen11 iGPU**.
> A full static reshape fixes it. The CPU plugin is always correct.

Everything here was produced on the affected machine. If you don't have Gen11
hardware, open an issue here (or ping me on the upstream thread) and I'll run
whatever you need on it.

## Environment

| | |
|---|---|
| CPU / GPU | Intel Celeron N5105 — Jasper Lake, UHD Graphics 24 EU, PCI `8086:4e61` |
| `DEVICE_ARCHITECTURE` | `GPU: vendor=0x8086 arch=v11.2.0` (**Gen11**) |
| OpenVINO | `2025.4.1-20426-82bbf0292c5-releases/2025/4` (pip `openvino==2025.4.1`) |
| Optimization caps | `['FP32', 'BIN', 'FP16', 'EXPORT_IMPORT']` |
| OpenCL ICD | the image registers two: `libigdrcl.so` and `libigdrcl_legacy1.so`. **The bug reproduces identically under either** (pin one with `OCL_ICD_FILENAMES=...`), so it is not an artifact of the legacy driver build. |
| Host | Linux 6.18.38, container with `--device=/dev/dri` |

No onnxruntime is involved anywhere in this pack — it is pure OpenVINO.

## Models

Public InsightFace `buffalo_l` pack ([v0.7 release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip),
no login needed). The `.onnx` files sit at the archive root:

| file | bytes | sha256 | input |
|---|---|---|---|
| `buffalo_l.zip` | 288,621,354 | `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` | — |
| `det_10g.onnx` | 16,923,827 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` | SCRFD, `[1,3,?,?]` |
| `w600k_r50.onnx` | 174,383,860 | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` | ArcFace, `['batch',3,112,112]` |

```bash
mkdir -p models && cd models
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip
```

Neither `insightface` nor `onnx` is required — OpenVINO reads the `.onnx` directly.

## Run it

```bash
pip install openvino==2025.4.1 numpy
python repro_36831.py    models              # dynamic / bounded / static vs CPU  → repro.log
python variance_36831.py models              # same input, 3 fresh compiles       → variance.log
python verify_ir.py      ir                  # same check, straight from the shipped IR
```

## What it shows

`repro.log` — dynamic and bounded-dynamic are wrong, static matches CPU:

```
model                          variant    GPU mean|out|  mean abs diff vs CPU
detection_scrfd_det_10g        dynamic         0.000078              0.040975
detection_scrfd_det_10g        bounded         0.000078              0.040975
detection_scrfd_det_10g        static          0.040906              0.000000
recognition_arcface_w600k_r50  dynamic         0.000000              0.318564
recognition_arcface_w600k_r50  bounded   89881201052972948992456268284166144.000000   (see below)
recognition_arcface_w600k_r50  static          0.318564              0.000001
```

Bounded shapes (`[1,3,320..640,320..640]` / `[1..32,3,112,112]`) do **not** help. Bounded
detection is bit-identical to unbounded dynamic; bounded recognition is wrong with a value
that changes between runs (I have seen `0`, `4.6e-5` and `8.99e34` for the same input).

`variance.log` — the dynamic path is not *stably* wrong, it is **unstable**. Same
process, same fixed input, three fresh compiles of the same model:

```
recognition_arcface_w600k_r50
  CPU (ground truth)       mean|out| = 0.318564
  GPU dynamic, compile #1  mean|out| = 0.000000                          diff vs CPU = 0.318564
  GPU dynamic, compile #2  mean|out| = 25816417527110520140518702673035264.000000
  GPU dynamic, compile #3  mean|out| = 25816714632719948631784678462849024.000000
```

Detection is stably ~0; recognition alternates between exactly 0 and ~2.6e34.

## Pre-exported IR

The [release](../../releases/latest) carries the six IR variants (dynamic / bounded / static
× detection + recognition, exported with OpenVINO 2025.4.1 on this machine,
`compress_to_fp16=False`), so you can skip the ONNX conversion:

```python
core.compile_model(core.read_model("recognition_arcface_w600k_r50.dynamic.xml"), "GPU")
```

Verified: loaded back from disk, the shipped IR reproduce the bug and keep their partial
shapes (`verify_ir.py`):

```
detection_scrfd_det_10g.dynamic   input partial shape from disk: [1,3,?,?]
    GPU mean|out| = 0.000078                     diff vs CPU = 0.040975
detection_scrfd_det_10g.static    input partial shape from disk: [1,3,640,640]
    GPU mean|out| = 0.040906                     diff vs CPU = 0.000000
recognition_arcface_w600k_r50.dynamic  input partial shape from disk: [?,3,112,112]
    GPU mean|out| = 0.000000                     diff vs CPU = 0.318564
recognition_arcface_w600k_r50.static   input partial shape from disk: [1,3,112,112]
    GPU mean|out| = 0.318564                     diff vs CPU = 0.000001
```

## Notes

- `OV_GPU_Verbose` prints nothing on the release wheels (they are built without
  `ENABLE_DEBUG_CAPS`), so no plugin traces are included. I can build OpenVINO with debug
  caps on this machine and capture whatever traces are useful — just say which.
- I did not keep the compiled Gen11 `.blob`s (they are tied to this GPU + driver version
  anyway). I can regenerate and upload them on request.

## Credits / licence

The IR in the release are a format conversion of the **InsightFace `buffalo_l`** models
(https://github.com/deepinsight/insightface, model zoo). The weights are theirs, not mine —
they are redistributed here **unmodified in content**, purely so this GPU bug can be
reproduced, and InsightFace's terms apply: **"ALL models are available for non-commercial
research purposes only."** Nothing here is re-licensed. The scripts in this repo are MIT.

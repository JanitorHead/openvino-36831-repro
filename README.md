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
| GPU | Intel UHD Graphics, PCI `8086:4e61` — Jasper Lake, Celeron N5105, 24 EU |
| `DEVICE_ARCHITECTURE` | `GPU: vendor=0x8086 arch=v11.2.0` (**Gen11**) |
| OpenCL ICD | `libigdrcl.so` **and `libigdrcl_legacy1.so`** — Gen11 is a *legacy* platform for compute-runtime, the legacy ICD serves it |
| OpenVINO | `2025.4.1-20426-82bbf0292c5-releases/2025/4` (pip `openvino==2025.4.1`) |
| Optimization caps | `['FP32', 'BIN', 'FP16', 'EXPORT_IMPORT']` |
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
```

## What it shows

`repro.log` — dynamic and bounded-dynamic are wrong, static matches CPU:

```
model                          variant    GPU mean|out|  mean abs diff vs CPU
detection_scrfd_det_10g        dynamic         0.000078              0.040975
detection_scrfd_det_10g        bounded         0.000078              0.040975
detection_scrfd_det_10g        static          0.040906              0.000000
recognition_arcface_w600k_r50  dynamic         0.000000              0.318564
recognition_arcface_w600k_r50  bounded   89881201052972948992456268284166144.000000   (same)
recognition_arcface_w600k_r50  static          0.318564              0.000001
```

Bounded shapes (`[1,3,320..640,320..640]` / `[1..32,3,112,112]`) do **not** help —
they are as wrong as unbounded, and on ArcFace the bounded compile blew up to ~9e34.

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

The release attached to this repo has the six IR variants (dynamic / bounded / static
× detection + recognition, `compress_to_fp16=False`), exported on this machine with
OpenVINO 2025.4.1, so you can skip the ONNX conversion entirely:
`core.read_model("detection_scrfd_det_10g.dynamic.xml")` → `compile_model(..., "GPU")`.

## Notes

- `OV_GPU_Verbose` produces no output on the release wheels (no `ENABLE_DEBUG_CAPS`).
  Happy to build OpenVINO with debug caps on this machine and capture whatever traces
  are useful — just say which.
- The compiled Gen11 `.blob`s exist but are tied to this GPU + driver, so they are not
  included; ask if you want them anyway.

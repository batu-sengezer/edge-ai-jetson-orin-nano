# Edge AI on the NVIDIA Jetson Orin Nano

This project is part of a larger series called Edge AI Across the Compute Spectrum, which examines how the same small machine learning workload performs across different classes of hardware. Tier 1 targets a microcontroller, an STM32F407 running FreeRTOS. Tier 2, covered in this repository, targets the NVIDIA Jetson Orin Nano, an embedded GPU board common in drones and robotics. Tier 3 targets a laptop-class Apple silicon chip.

The central question for this tier is when GPU acceleration on a board like this actually helps, and when it does not. Both outcomes are reported below, including a case where the result ran counter to the initial expectation.

## Overview

The project is composed of five sections, which are connected rather than independent and are ordered so that each builds on, or complicates, the one before it.

- **Section 1** deploys the Tier 1 HAR model through TensorRT and compares CPU and GPU execution on the same small workload, establishing that GPU acceleration is not automatic and depends on the size of the problem being solved.
- **Section 2** moves underneath TensorRT's abstraction, using a hand-written CUDA kernel to test the opposite case, a large, compute-bound workload for which GPU parallelism is expected to matter.
- **Section 3** returns to a deployed setting, running a real vision model against a live camera feed to check whether inference time or the surrounding pipeline dominates total latency in something closer to real use.
- **Section 4** takes the Section 1 benchmark and re-runs it under artificial system load, since a deployed model rarely has the board to itself, and checks whether average latency alone is a safe way to characterize the workload.
- **Section 5** closes with power draw, measured on the same benchmark and directly relevant to any platform that is not plugged into mains power, such as a drone.

The contrast between Sections 1 and 2 is the central argument of the tier: the same hardware behaves very differently depending on workload size and arithmetic intensity, rather than one processor holding a fixed advantage over the other regardless of context.

Sections 1, 3, and 4 all use the HAR model as a common reference point, which keeps their results comparable to one another rather than each section starting from a different baseline. Section 2 deliberately uses a separate, larger workload, so that the small-workload conclusion from Section 1 has a contrasting case to be checked against rather than standing alone.

## Summary of results

These are single-board observations from a limited number of runs and should be read at that scale rather than as general claims about the hardware.

| Section | What was measured | Result |
|---|---|---|
| 1 | HAR model, CPU versus GPU, FP16 versus FP32 | For a model this small, the GPU showed no measurable advantage, and the two precisions performed about the same |
| 2 | Hand-written CUDA matrix multiply | A simple GPU kernel substantially outperformed a single-threaded CPU loop, with the gap widening as matrix size increased |
| 2 | Tiled kernel versus cuBLAS | The tiled kernel reached roughly a fifth of cuBLAS's throughput |
| 3 | Live camera pipeline | Model inference was fast; the surrounding pipeline, mainly camera capture, was far slower and appears to be the actual bottleneck |
| 4 | CPU and memory contention | Median latency was largely unaffected under load; the observed worst-case latency increased sharply |
| 5 | Power draw at two power modes | The lower power mode used noticeably less energy for a modest reduction in throughput |

The overall pattern suggests that parallel hardware provides a large benefit on compute-heavy workloads and little to none on small workloads where GPU call overhead outweighs the computation itself. This was not the expected outcome at the start of the project; the HAR model result in particular ran against the initial assumption that GPU acceleration would help across the board.

## Hardware and setup

The Jetson Orin Nano is built around a system-on-module with a unified memory architecture: the ARM CPU and the NVIDIA GPU sit on the same die and address the same physical LPDDR5 pool, rather than each holding a private memory space connected by a discrete PCIe link, as on a desktop with a dedicated GPU. Boards of this class are aimed at robotics and drone applications, where meaningful GPU compute is needed within a much smaller power and size budget than a discrete desktop GPU allows.

| Component | Detail |
|---|---|
| Board | NVIDIA Jetson Orin Nano Developer Kit, Super capable, part number 945-13766-0005-000, model P3766 |
| Module | Jetson Orin Nano, 8GB variant |
| GPU | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores, compute capability 8.7, 8 streaming multiprocessors |
| CPU | 6-core Arm Cortex-A78AE, 64-bit |
| Memory | 8GB of 128-bit LPDDR5, shared between CPU and GPU |
| Software stack | JetPack 6.2.1, L4T 36.4.7 (Ubuntu 22.04 base), CUDA 12.6, TensorRT 10.3.0 |
| Boot media | 128GB SanDisk Extreme PRO microSD, UHS-I, A2 rated |
| Camera | UVC USB webcam, 1920x1080 |
| Development setup | macOS host over SSH throughout. No Ubuntu host PC and no NVIDIA SDK Manager were used, unlike most official Jetson setup documentation. |

Unless stated otherwise, all measurements were taken in `MAXN_SUPER` power mode with `jetson_clocks` locked, which pins CPU and GPU clocks at their maximum for that mode rather than allowing dynamic scaling. Locking clocks removes one source of run-to-run variability from the measurements below.

---

## 1. Running the Tier 1 model through TensorRT

The Tier 1 human activity recognition (HAR) classifier, a small dense network with 561 input features and 6 output classes, was exported from Keras to ONNX and compiled into two TensorRT engines, one at FP32 and one at FP16. TensorRT compiles a model into a serialized engine tailored to the target GPU through steps such as layer and tensor fusion and kernel auto-tuning, where the builder profiles several candidate kernel implementations (tactics) per layer and selects the fastest. FP16 halves the per-value memory footprint and, on hardware with dedicated Tensor Cores such as this GPU's 32, can substantially increase throughput for kernels that are compute-bound rather than memory- or overhead-bound.

| Precision | Throughput | Mean latency | Median | p90 | p95 | p99 | Observed max | CoV |
|---|---|---|---|---|---|---|---|---|
| FP16 | 16,093 qps | 0.0452 ms | 0.0449 ms | 0.0458 ms | 0.0468 ms | 0.0579 ms | 0.365 ms | 6.9% |
| FP32 | 16,232 qps | 0.0450 ms | 0.0447 ms | 0.0457 ms | 0.0466 ms | 0.0569 ms | 0.0879 ms | 4.8% |

**Interpretation.** FP16 produced no measurable improvement over FP32 and was marginally slower on this run. The likely explanation is that this model is too small for precision to matter: FP16 helps mainly when a model performs enough matrix arithmetic that halving numeric precision meaningfully reduces the work involved. This network finishes its arithmetic almost instantly at either precision, so measured latency is dominated by fixed overhead, kernel launch and host-device data movement, rather than by arithmetic throughput. In roofline terms, a workload this small sits well to the overhead-bound side of the curve, where raising arithmetic throughput has little left to act on.

The elevated FP16 maximum (0.365 ms against a p99 of 0.058 ms) is a single outlier across 48,281 samples rather than a repeating pattern.

### CPU versus GPU, measured several ways

Given the small size of the model, a single measurement method risked being misleading, so the comparison was checked from several angles.

| Method | Mean latency | What is included |
|---|---|---|
| CPU only (ONNX Runtime CPU execution provider) | 0.030 ms | Compute only, no device transfer |
| GPU compute only (`trtexec`) | 0.031 ms | GPU kernel execution |
| GPU end to end (`trtexec`) | 0.045 ms | Kernel plus host-to-device and device-to-host transfer |
| GPU via Python (ONNX Runtime, TensorRT execution provider) | 0.169 ms | Kernel, transfer, and Python dispatch overhead |

**Interpretation.** Raw compute is essentially identical on CPU and GPU, 0.030 ms versus 0.031 ms. Every GPU measurement that includes realistic call overhead is slower than the CPU figure, with the Python path slowest due to interpreter dispatch and tensor conversion on top of transfer cost. This model does not appear to benefit from the GPU, and the apparent GPU disadvantage scales with how much of the surrounding software stack is included in the measurement.

---

## 2. Hand-written CUDA: matrix multiplication

TensorRT provides GPU acceleration without requiring an understanding of the underlying hardware. This section demonstrates that layer directly, through a hand-written CUDA implementation of dense matrix multiplication, a fundamental operation in most machine learning workloads.

Three implementations were compared: a single-threaded CPU baseline, a naive CUDA kernel where each thread computes one output element by reading its operands directly from global memory on every iteration, and a tiled CUDA kernel using shared memory. Repeated global memory access is comparatively slow and largely redundant across neighboring threads, so the tiled version has each thread block cooperatively stage a tile of the input matrices into faster on-chip shared memory once, synchronize with `__syncthreads()`, then reuse that tile across several multiply-accumulate steps before loading the next one.

| N | CPU | Naive CUDA | Tiled CUDA | Naive vs CPU | Tiled vs naive |
|---|---|---|---|---|---|
| 256 | 41.576 ms | 0.250 ms | 0.161 ms | ~166x | ~1.55x |
| 512 | 345.241 ms | 1.710 ms | 1.090 ms | ~202x | ~1.57x |
| 1024 | 3468.711 ms | 13.061 ms | 8.643 ms | ~266x | ~1.51x |
| 2048 | not run | 102.020 ms | 69.716 ms | n/a | ~1.46x |

**Interpretation.** Two patterns stand out beyond the largest single figure. The speedup over CPU grows with problem size, from roughly 166x at N=256 to roughly 266x at N=1024, consistent with larger matrices exposing more parallelism while the CPU loop's cost scales cubically. Tiling delivers a consistent ~1.5x over the naive kernel at every size tested, consistent with reduced global memory traffic from data reuse, though memory traffic was not profiled directly to confirm this.

### Comparison against cuBLAS

| Implementation | N = 1024 | Relative to cuBLAS |
|---|---|---|
| Naive CUDA | 13.061 ms | ~13% |
| Tiled CUDA | 8.643 ms | ~20% |
| cuBLAS `Sgemm` | 1.706 ms | 100% |

**Interpretation.** The tiled kernel reached approximately 20% of cuBLAS's throughput, a real and expected gap rather than a close result. cuBLAS reflects extensive architecture-specific tuning, including register-level blocking, warp-level scheduling, and a heuristic search over multiple candidate kernels per GEMM shape, well beyond the scope of this kernel. The purpose of this section is to demonstrate understanding of GPU parallelism and the memory hierarchy, not to match a vendor-tuned library. Production workloads should rely on libraries such as cuBLAS or TensorRT rather than hand-written kernels.

---

## 3. Live camera pipeline with MobileNetV2

An off-the-shelf ImageNet MobileNetV2 model, not trained as part of this project, was exported to ONNX and compiled with TensorRT, then run against a live USB camera feed. The purpose of this section is to demonstrate and measure a deployed perception pipeline rather than to evaluate the model itself.

The compiled engine was first benchmarked in isolation.

| Metric | Value |
|---|---|
| Throughput | 1,232.8 qps |
| Mean latency | 0.850 ms |
| Median | 0.850 ms |
| p90 / p95 / p99 | 0.854 / 0.855 / 0.858 ms |
| Observed min / max | 0.841 / 0.937 ms |
| GPU compute time (mean) | 0.808 ms |

**Interpretation.** Unlike the HAR model in Section 1, this run produced no timing-instability warning, consistent with MobileNetV2 doing enough real computation per call that launch jitter becomes negligible relative to the workload.

The pipeline was then run live, with inference time and full pipeline time measured separately per frame.

| Stage | Measured |
|---|---|
| Model inference alone | 2.2 to 2.3 ms |
| Full pipeline (capture, resize, colour convert, infer) | 178 to 184 ms |
| Sustained frame rate | 5.4 to 5.6 FPS |
| Ratio | ~78x |

**Interpretation.** Model inference accounts for a small fraction of total frame time. The most likely source of the remaining time is frame capture and decode from the USB camera at 1920x1080, with preprocessing contributing a smaller share; the pipeline was not broken down stage by stage to confirm the exact split, but the values were consistent across frames, which points to a structural bottleneck rather than measurement noise. In effect the pipeline is I/O-bound rather than compute-bound. On a real platform, further model optimization would likely yield little benefit, since the model is already a small share of total latency; the larger opportunity is in the surrounding data path.

Live classifications were frequently incorrect under informal handheld conditions. ImageNet's 1,000-class taxonomy maps poorly onto arbitrary desk objects, and handheld framing, motion, background clutter, and inconsistent lighting all diverge from the curated single-subject photographs used in training. A more controlled pass correctly identified a computer keyboard, a remote control, and an analog clock.

---

## 4. Contention and interference

Real embedded systems rarely run a single task in isolation. The Section 1 benchmark was run on a quiet system and then re-run under synthetic CPU and memory load (`stress-ng --cpu 4 --vm 2 --vm-bytes 1G`).

| Metric | Quiet | Under contention | Change |
|---|---|---|---|
| Throughput | 16,272.6 qps | 13,134.9 qps | -19.3% |
| Mean latency | 0.0448 ms | 0.0539 ms | +20.3% |
| Median latency | 0.0441 ms | 0.0444 ms | +0.7% |
| p90 | 0.0450 ms | 0.0458 ms | +1.8% |
| p95 | 0.0459 ms | 0.0476 ms | +3.7% |
| p99 | 0.0581 ms | 0.0660 ms | +13.6% |
| Observed maximum | 2.84 ms | 20.35 ms | +617% |
| H2D observed maximum | 2.807 ms | 16.367 ms | +483% |
| CoV | 7.9% | 750.9% | |

**Interpretation.** The median moved by under 1%, indicating most individual inferences were unaffected by the background load. The observed maximum latency rose by roughly 7x, and variability increased by two orders of magnitude. This pattern is consistent with most calls proceeding normally while a small number are delayed, likely by CPU scheduling contention or memory bandwidth contention and cache pollution from the `stress-ng` virtual memory workers; the parallel rise in host-to-device transfer time points toward the memory subsystem rather than the GPU kernel itself, though this was not confirmed directly with a profiler.

A design validated only against average latency would appear unaffected by this load. The same design could still miss a real-time deadline because of the rare, substantially longer delays that an average obscures.

---

## 5. Power and efficiency

Board power was read via `jtop` (`VDD_IN`) while the Section 1 benchmark ran. The two readings below were captured in the same run and time window rather than estimated separately.

| Power mode | VDD_IN under load | GPU utilization | Throughput | GPU compute mean | CoV | Inferences/sec/watt |
|---|---|---|---|---|---|---|
| `MAXN_SUPER` | 6.9 W | 46.2% | 16,272.6 qps | 0.0304 ms | 7.9% | ~2,358 |
| `15W` | 4.9 W | 64.7% | 14,037.5 qps | 0.0439 ms | 4.0% | ~2,865 |

**Interpretation.** At `MAXN_SUPER`, the board draws under 7 W while sustaining over 16,000 inferences per second, with GPU utilization under 50%, consistent with the workload not saturating the hardware.

The 15 W mode reduces power draw by roughly 30% for a 14% reduction in throughput, yielding roughly 21% more inferences per second per watt. Dynamic power in CMOS logic scales roughly with the square of supply voltage and linearly with clock frequency, and DVFS (dynamic voltage and frequency scaling) modes typically require higher voltage to sustain higher clocks, so a capped power budget can improve energy efficiency even as peak throughput falls. This comparison reflects two specific power modes on one board running one small model and should not be generalized without further testing.

The 15 W run was also more stable (CoV 4.0% versus 7.9% at `MAXN_SUPER`), plausibly because a hard clock ceiling removes some jitter from dynamic clock scaling at the uncapped mode; this was not investigated further.

During the 15 W run, `jtop` displayed `Jetson Clocks: inactive` despite `jetson_clocks` exiting successfully. This was confirmed as a display inconsistency rather than a measurement problem: `jetson_clocks --show` reported all six CPU cores pinned at a fixed frequency and the GPU pinned at 612 MHz, matching the clock rate `trtexec` reported for the same run.

---

## Measurement scope and limitations

Latency maximums reported throughout this document are the highest value observed during a limited sampling window, not a proven worst case. No static timing analysis was performed, and the term worst-case execution time is deliberately avoided throughout.

The contention results in Section 4 describe behavior under one synthetic load pattern on one board. They illustrate a trend, that tail latency is more sensitive to interference than average latency, without establishing a bound under other conditions.

CUDA timings in Section 2 are single-run kernel compute times and exclude host-to-device and device-to-host transfer. They demonstrate scaling behavior across implementations rather than serving as a rigorous benchmark suite.

The custom kernel in Section 2 does not outperform cuBLAS and is not presented as doing so.

---

## Repository contents

```
.
├── camera_demo.py          # Live camera pipeline, with inference-only and full-pipeline timing
├── export_mobilenet.py     # Exports MobileNetV2 to ONNX, run on a host machine
├── matmul.cu                # CPU, naive CUDA, and tiled CUDA matrix multiply
├── cublas_test.cu           # cuBLAS Sgemm reference for the Section 2 comparison
├── run_cpu.py               # HAR inference, ONNX Runtime CPU execution provider
├── run_gpu.py               # HAR inference, ONNX Runtime TensorRT execution provider
├── imagenet_classes.txt     # ImageNet label list used by the camera demo
└── media/                   # Demo video and representative annotated frames
```

Compiled model files (`*.onnx`, `*.onnx.data`) and TensorRT engine files (`*.engine`) are not included. Engine files are built ahead-of-time for a specific GPU's streaming multiprocessor (SM) architecture and TensorRT version, so they are not portable across hardware or TensorRT releases; ONNX files can be regenerated from the included export scripts.

### Reproducing

```bash
# Build and benchmark a TensorRT engine
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# Build and run the CUDA comparison
nvcc -O3 matmul.cu -o matmul && ./matmul
nvcc -O3 cublas_test.cu -o cublas_test -lcublas && ./cublas_test

# Live camera demo
python3 camera_demo.py

# Contention test, in two terminals
stress-ng --cpu 4 --vm 2 --vm-bytes 1G --timeout 30s   # terminal A
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --fp16  # terminal B
```

Locking the board's clocks before any timed run improves repeatability:

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks && nvpmodel -q
```

`jetson_clocks` does not persist across a reboot or a power mode change and must be reapplied each time.

---

## Future work

- Sparse matrix operations as a natural extension of the CUDA work in Section 2, since real workloads often involve matrices that are mostly zero, where the dominant cost shifts from raw arithmetic to handling an irregular memory access pattern that tiling alone does not address.
- Determining how much of the Section 3 pipeline's 180 ms is recoverable, for example through lower capture resolution, GPU-side preprocessing, or overlapping capture and inference on separate threads.
- Runtime instrumentation to detect the tail-latency events characterized in Section 4 as they occur, rather than only in post hoc statistics.

---

## Related work in this series

- **Tier 1: STM32F407.** INT8-quantized HAR classifier under FreeRTOS, cycle-accurate DWT timing over 1,000 runs. 786.89 µs observed maximum latency, 11.3x model size reduction, 94.98% accuracy after quantization.
- **Tier 3: Apple silicon.** The same workload at laptop-class power and performance.
- **Cross-tier comparison.** The same inference task across a microcontroller, a drone-class embedded GPU board, and a laptop-class chip, reflecting the tradeoffs an autonomous platform's onboard compute has to make.

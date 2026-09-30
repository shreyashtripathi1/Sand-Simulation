# CUDA-Accelerated Falling Sand: 3D, 2D and CPU versions

Real-time falling-sand simulation on the GPU with **CUDA** and **OpenGL**. The main version is a **3D sand cube**: millions of grains fall inside a transparent glass cube that you rotate with the mouse, and a keyboard-controlled dropper pours sand anywhere inside it. The original **2D** version and a **CPU** port of the 3D simulation (for a GPU-vs-CPU comparison) are kept alongside it.

| Version | Source | What it is |
|---|---|---|
| 3D cube | `main.cu` | CUDA + OpenGL, voxel grid (192³ by default), rotating view, keyboard dropper |
| 2D | `main_2d.cu` | Original CUDA + OpenGL simulator, mouse-painted sand and blast effect |
| CPU | `cpu_sim.cpp` | The 3D simulation on the CPU with OpenMP, no graphics; used for benchmarking only |

---

## 3D version

### Features

- **Thread-per-cell CUDA kernels** over an N×N×N voxel grid (default 192³ = 7.1 million cells, configurable 16³ to 320³)
- **3D cellular-automaton physics:** sand falls straight down, otherwise slides diagonally into any of the 8 surrounding columns, giving a stable angle of repose
- **Race-free updates:** `atomicCAS` claim flags, plus a flip-flop scheme extended to 3D so the pile does not drift to one side
- **GPU back-face culling:** only grains with an exposed face turned towards the camera are drawn
- **Pixel-sized grains**, lit from their surface normal, inside a transparent glass cube with a floor grid
- **Keyboard dropper** with an on-screen marker, guide line and floor footprint

### Controls

| Input | Action |
|---|---|
| Left mouse (hold + drag) | Rotate the cube |
| Mouse wheel | Zoom |
| Arrow keys | Move the dropper horizontally, relative to your view (Up = away from you, Right = to your right) |
| Space / Ctrl | Move the dropper up / down |
| Enter (hold) | Drop sand (outline is yellow while dropping, red when idle) |
| `[` / `]` | Shrink / grow the dropper |
| C / R / Esc | Clear the cube / reset the camera / quit |

The window title shows the dropper position, visible grains, GPU time per frame (simulation and culling) and FPS.

### How it works

Each frame runs on the GPU; only the final draw goes through OpenGL:

```
AddSandKernel → SimulateParticlesKernel (x2) → BuildVisibleVoxelsKernel → copy visible list → OpenGL draw
 (Enter held)    gravity + atomic claims         cull + shade + compact      (a few MB)        (GL_POINTS)
```

**Grid layout.** One flat `int` array indexed `(y * N + z) * N + x`. `x` is the fastest axis, so the 32 threads of a warp read 32 consecutive words (coalesced). Blocks are 32×4×2 = 256 threads. `0` means empty; a grain stores its packed colour with alpha set, so it is never `0`.

**Simulation.** Ping-pong buffers (`d_gridInput` / `d_gridOutput`) separate reads from writes. A grain tries the cell below, then the 8 diagonals below it. The first thread to win `atomicCAS` on a target cell's claim flag moves there, so no grain is lost or duplicated.

**Removing bias.** The atomic winner is usually the lower thread index, so the kernel mirrors the thread-to-cell mapping in x and z on alternate steps and alternates the direction and hashed start of the diagonal search.

**Back-face culling.** `BuildVisibleVoxelsKernel` keeps a grain only if at least one face is exposed to air **and** points towards the camera (`normal · (camera − grain) > 0`). Survivors are lit and appended to a compact vertex list with one `atomicAdd`. Only that list is copied to the CPU and drawn as depth-tested `GL_POINTS`, sized to one cell on screen and rounded up so neighbours leave no gaps.

---

## 2D version

`main_2d.cu` is the original simulator on a 1280×720 grid (one thread per cell). The simulation, render and colour-buffer kernels are unchanged from the original; the host code gained command-line options and a benchmark mode.

Options: `size=WxH` (default 1280x720), `block=AxB` (default 16x8, the fastest shape in the sweep below; the original was 8x8), `pinned` (pinned host memory for the colour-buffer copy), `novsync`, `bench`. The title bar shows sim / render / copy time in ms and FPS.

| Input | Action |
|---|---|
| Left mouse button (hold) | Spawn sand at the cursor |
| Right mouse button (hold) | Blast: erupts nearby sand outwards |

Each frame runs `AddSandKernel → SimulateParticlesKernel → RenderToColorKernel → OpenGL texture`, with the same ping-pong buffers and `atomicCAS` claims as the 3D version. Its dominant per-frame cost is the device-to-host copy of the whole colour buffer (about 3.7 MB at 1280×720), which is what the 3D version's GPU-side culling avoids.

---

## CPU version

`cpu_sim.cpp` runs the same algorithm as the 3D version on the processor: the same sand drop, two simulation steps per frame with the same hashed flip-flop order and compare-and-swap claim flags, and the same culling pass, parallelised with OpenMP across grid rows. It has no graphics. CUDA cannot run on the CPU (unplugging a laptop does not change that), so this port is what makes a fair GPU-vs-CPU comparison possible. Every run checks that no grain is created or lost, and `cpu_sim selftest` checks the culling logic.

---

## Requirements

| Requirement | Details |
|---|---|
| GPU | NVIDIA GPU with CUDA (3D and 2D versions). About 85 MB of GPU memory at 192³, about 375 MB at 320³ |
| CUDA Toolkit | Any recent version |
| Compiler | MSVC from Visual Studio 2022 with the "Desktop development with C++" workload (Windows) |
| Libraries | OpenGL and GLFW (included under `glfw/`) |
| Python (optional) | Python 3 for the benchmark summary scripts (standard library only) |

---

## Build & run (Windows)

`nvcc` needs the MSVC compiler (`cl.exe`) on `PATH`, which is only the case in the **x64 Native Tools Command Prompt for VS 2022**. `build.bat` finds Visual Studio itself, so it works from any terminal:

```bat
build.bat                 :: build and run the 3D cube (PowerShell: .\build.bat)
build.bat 256             :: a different grid size
build.bat 256 novsync     :: vsync off: uncapped FPS
```

The program accepts `[gridSize] [novsync] [block=AxBxC] [bench]` (`bench` runs a fixed scripted scene, appends a row to `benchmark_results.csv` and exits).

Manual builds, from the x64 Native Tools prompt:

```bat
:: 3D cube
nvcc main.cu -o sim3d -allow-unsupported-compiler -Xcompiler "/MD" -I"glfw\include" -L"glfw\lib-vc2022" -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32

:: 2D version: same command with main_2d.cu and -o sim2d

:: CPU version
cl /nologo /O2 /openmp /std:c++17 /EHsc cpu_sim.cpp /Fe:cpu_sim.exe
```

### Benchmarking

```bat
run_benchmarks.bat         :: GPU: grid-size and block-shape sweeps -> benchmark_results.csv
run_cpu_benchmarks.bat     :: CPU: grid-size and thread-count sweeps -> benchmark_cpu_results.csv
python compare_cpu_gpu.py  :: GPU vs CPU tables
run_2d_benchmarks.bat      :: 2D: resolution, block-shape and pinned-memory sweeps -> benchmark_2d_results.csv
```

Both scripts accept `quick` (short sweep) and `nobuild`; the CPU one also takes `battery` or `ac` to label the power state. `summarize_benchmarks.py` and `summarize_2d_benchmarks.py` print the GPU results as markdown tables (medians of repeated runs). The 2D `bench` mode runs a fixed 700-frame scripted scene (moving spawner plus periodic blasts), so results are repeatable without mouse input.

---

## Performance

Test system: NVIDIA GeForce GTX 1650 and Intel Core i5-12450H (8 cores, 12 threads). Times are per frame. The scene is a fixed scripted run: sand dropped continuously while the camera orbits slowly, vsync off, 100 warm-up frames then 600 measured (GPU) or 100 measured (CPU). Repeated runs agree to within about 1%.

### 3D on the GPU

**Grid size** (block 32×4×2). *Sim ms* covers the two simulation steps plus sand drop, *Cull ms* the visibility pass, *Frame ms* is wall-clock per frame including the copy and draw.

| Grid | Cells | Visible grains | Sim ms | Cull ms | GPU ms | Frame ms | FPS |
|---|---|---|---|---|---|---|---|
| 64³ | 262,144 | 752 | 0.198 | 0.022 | 0.219 | 1.31 | 762 |
| 96³ | 884,736 | 2,400 | 0.372 | 0.043 | 0.415 | 1.34 | 746 |
| 128³ | 2,097,152 | 2,782 | 0.658 | 0.087 | 0.745 | 1.51 | 662 |
| **192³ (default)** | 7,077,888 | 10,681 | 2.055 | 0.293 | 2.349 | 2.67 | 375 |
| 256³ | 16,777,216 | 16,298 | 4.415 | 0.632 | 5.046 | 5.42 | 185 |
| 320³ | 32,768,000 | 25,176 | 8.182 | 1.242 | 9.424 | 9.89 | 101 |

**Block shape** at 256³:

| Block | Sim ms | Cull ms | GPU ms | FPS | vs default |
|---|---|---|---|---|---|
| **32×4×2 (default)** | 4.415 | 0.632 | 5.046 | 185 | — |
| 32×8×1 | 4.494 | 0.655 | 5.149 | 181 | +2.0% |
| 16×4×4 | 4.625 | 0.709 | 5.334 | 175 | +5.7% |
| 16×16×1 | 4.740 | 0.742 | 5.482 | 170 | +8.6% |
| 8×8×4 | 4.792 | 0.777 | 5.569 | 167 | +10.4% |
| 8×8×8 | 5.051 | 0.861 | 5.913 | 158 | +17.2% |

- **Memory bandwidth is the limit.** The simulation reads and writes about four grid-sized arrays per step. At 320³ that works out to roughly 119 GB/s, close to the GTX 1650's peak (128 GB/s for the GDDR5 model), so it is memory-bound, not compute-bound. The figure is derived from buffer sizes, not profiled.
- **Time scales with cell count** from 192³ up (cells ×4.6, time ×4.0). Small grids (up to about 128³) are limited by launch and draw overhead instead: at 64³ the GPU takes 0.22 ms but a frame takes 1.3 ms.
- **The default 192³ runs at 375 FPS uncapped.** With vsync on, the 144 Hz display is the limit; only 320³ (101 FPS) falls below it. On battery, FPS is capped to about 30 regardless of grid size, most likely by NVIDIA Battery Boost; CUDA still runs on the GPU.
- **Culling and the copy are cheap:** the visibility pass is 10–13% of GPU time, and frame time is only 0.3–0.5 ms above GPU time from 192³ up.
- **Wide x-blocks are fastest.** 32×4×2 beats 8×8×8 by 17%, because a block 32 wide reads one contiguous 128-byte line per warp.

The pile is small in this scene (750 to 25,000 visible grains). Empty cells exit early, so a fuller cube may cost more.

### 2D on the GPU

GTX 1650, `bench` mode (700 frames, vsync off), median of 3 runs, 8×8 blocks unless stated.

| Resolution | Sim ms | Render ms | Copy ms | GPU ms | Copy share | FPS |
|---|---|---|---|---|---|---|
| 640x360 | 0.175 | 0.020 | 0.402 | 0.594 | 68% | 668 |
| 1280x720 | 1.003 | 0.055 | 0.894 | 1.946 | 46% | 404 |
| 1920x1080 | 1.230 | 0.117 | 1.890 | 3.237 | 58% | 237 |
| 2560x1440 | 1.434 | 0.251 | 3.296 | 4.982 | 66% | 151 |
| 3840x2160 | 3.157 | 0.515 | 6.648 | 10.312 | 64% | 74 |

- The device-to-host copy of the colour buffer is 46-68% of GPU time at every size, so the 2D version is limited by PCIe transfer, not by the simulation kernel.
- Block shape at 1920x1080 (kernels only): 16x8 is fastest (0.91 ms, 32% faster than 8x8); 16x16, 32x4, 64x4 and 32x8 are 4-8% faster than 8x8.
- Pinned host memory speeds up the copy by only about 1.1× (frame time 2.48 → 2.26 ms at 1280x720), so the copy is bandwidth-bound rather than staging-bound.

### GPU vs CPU

CUDA `gpu_ms` (simulation + culling) against CPU `total_ms` (simulation + culling), median of 2 to 3 runs. The CPU sweep ran on battery; the GPU sweep ran earlier with the charger connected.

| Grid | Cells | CPU 1 thread | CPU 12 threads | CPU thread speedup | GPU | GPU vs 1 thread | GPU vs 12 threads |
|---|---|---|---|---|---|---|---|
| 64³ | 262,144 | 1.462 | 0.403 | 3.6× | 0.219 | 7× | 2× |
| 96³ | 884,736 | 4.272 | 1.107 | 3.9× | 0.415 | 10× | 3× |
| 128³ | 2,097,152 | 10.7 | 2.864 | 3.7× | 0.745 | 14× | 4× |
| 192³ | 7,077,888 | 36.6 | 12.2 | 3.0× | 2.349 | 16× | 5× |
| 256³ | 16,777,216 | 87.0 | 33.1 | 2.6× | 5.046 | 17× | 7× |
| 320³ | 32,768,000 | 168.9 | 58.6 | 2.9× | 9.424 | 18× | 6× |

CPU thread scaling at 192³:

| Threads | ms per frame | Speedup vs 1 thread | Parallel efficiency |
|---|---|---|---|
| 1 | 36.6 | 1.00× | 100% |
| 2 | 20.5 | 1.79× | 89% |
| 4 | 14.2 | 2.57× | 64% |
| 6 | 14.9 | 2.46× | 41% |
| 8 | 13.6 | 2.69× | 34% |
| 12 | 12.2 | 2.99× | 25% |

- **The GPU is 16–18× faster than one CPU thread and 5–7× faster than all 12** at 192³ and above. At 192³: GPU 2.35 ms, CPU 12.2 ms with every core busy (about 82 FPS of simulation alone, before drawing). At 320³ the CPU manages about 17 FPS against the GPU's 101.
- **More threads stop helping early:** 2 threads are 89% efficient, but 12 threads give only 3.0×, and going from 8 to 12 gains about 11%. The cause is memory bandwidth again: the CPU moves about 7 GB/s with one thread and 19 GB/s with 12 at 320³, against the GPU's 119 GB/s. The GPU's roughly 6× lead tracks that gap.
- **Small grids favour the CPU relatively:** at 64³ the GPU is only 2× faster than all 12 threads, because launch overhead eats its advantage.

The CPU ran on battery, where processor power limits may be lower, so a plugged-in CPU may be somewhat faster. The CPU scene ran 120 frames instead of 700 and so had a smaller pile (about 7,900 visible grains at 192³ against 10,700); cost is dominated by scanning the whole grid, so this matters little.

---

## Project structure

```
├── main.cu                     # 3D sand cube (CUDA + OpenGL)
├── main_2d.cu                  # 2D simulator (with benchmark mode)
├── cpu_sim.cpp                 # CPU (OpenMP) port of the 3D simulation
├── build.bat                   # build + run the 3D version from any terminal
├── run_benchmarks.bat          # GPU benchmark sweeps
├── summarize_benchmarks.py     # GPU results as markdown tables
├── run_cpu_benchmarks.bat      # CPU benchmark sweeps
├── compare_cpu_gpu.py          # GPU vs CPU tables
├── run_2d_benchmarks.bat       # 2D benchmark sweeps
├── summarize_2d_benchmarks.py  # 2D results as markdown tables
├── benchmark_results.csv       # GPU results
├── benchmark_cpu_results.csv   # CPU results
├── CUDA_SandSim_Report.pdf     # project report (2D version)
└── glfw/                       # GLFW headers and lib-vc2022
```

---

## Limitations and future work

- Only sand is implemented; water, fire and smoke would be natural additions
- The 2D blast effect is not ported to 3D
- The simulation runs at a fixed 2 steps per frame, so its speed follows the frame rate
- Grains are screen-aligned squares, so they look flat when zoomed in close; instanced cubes would look better at higher cost
- CUDA–OpenGL interoperability would remove the last device-to-host copy (it needs an OpenGL extension loader such as GLAD)
- Skipping empty regions (per-column pile height or empty bricks) and copying only changed cells would cut the memory traffic that limits the simulation
- Grid state is not saved between sessions

---

## References

1. J. Nickolls et al., "Scalable parallel programming with CUDA," *ACM Queue*, vol. 6, 2008
2. C. McIvor and B. H. Kaye, "Cellular automata simulation of granular materials," *Powder Technology*, vol. 190, 2009
3. D. B. Kirk and W.-M. W. Hwu, *Programming Massively Parallel Processors*, Morgan Kaufmann, 2016
4. J. Sanders and E. Kandrot, *CUDA by Example*, Addison-Wesley Professional, 2010
5. D. Shreiner et al., *OpenGL Programming Guide*, 8th ed., Addison-Wesley Professional, 2013

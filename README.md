# CUDA-Accelerated Falling Sand Simulation

A real-time falling sand simulation powered by **CUDA** and **OpenGL**, implementing GPU-parallel cellular automata with interactive physics effects.

> Project Component for **Parallel and Distributing Computing (UCS645)**  
> Thapar Institute of Engineering and Technology, Patiala — May 2025

---
## Showcase
[sand-simulation-demo.webm](https://github.com/user-attachments/assets/ad24490b-6b5d-4096-b7b6-8bae3b29be4b)

---
## Features

- **GPU-parallel simulation** — 480,000 cells (Scalable) updated simultaneously via CUDA kernels
- **Cellular automata physics** — gravity-driven sand movement with stable angle-of-repose
- **Race-condition-free updates** — atomic conflict resolution using `atomicCAS`
- **Interactive controls**
  - Left mouse button — spawn sand particles
  - Right mouse button — trigger physics-based explosion/blast effect
- **Dynamic colour visualisation** — phase-shifted sine wave RGB cycling per simulation step
- **Performance logging** — FPS, kernel time, and memory transfer time recorded to CSV

---

## How It Works

Each frame runs a pipeline of three CUDA kernels:

```
AddSandKernel → SimulateParticlesKernel → RenderToColorKernel → OpenGL display
```

A **ping-pong double-buffer** (`d_gridInput` / `d_gridOutput`) decouples reads and writes each frame. Concurrent write conflicts are resolved with `atomicCAS` on a dedicated claims buffer.

---

## Project Structure

```
Sand-Simulation/
│
├── main.cu                     # CUDA + OpenGL simulation source
├── plot.py                     # Per-configuration performance graphs
├── compare.py                  # Cross-configuration comparison plots
│
├── performance_log_8x8.csv     # Metrics for 8×8 block config
├── performance_log_16x16.csv   # Metrics for 16×16 block config
├── performance_log_32x8.csv    # Metrics for 32×8 block config
│
├── 8x8/                        # Output graphs — 8×8 configuration
├── 16x16/                      # Output graphs — 16×16 configuration
├── 32x8/                       # Output graphs — 32×8 configuration
├── block_size_comparision/     # Cross-config comparison plots
│
├── glfw/                       # GLFW library (include + lib-vc2022)
├── CUDA_SandSim_Report.pdf     # Full project report
├── .gitignore
└── README.md
```

---

## Requirements

| Requirement | Details |
|---|---|
| GPU | NVIDIA GPU with CUDA support |
| CUDA Toolkit | Any recent version (tested with Visual Studio 2022) |
| Compiler | MSVC via Visual Studio 2022 (Windows) |
| Graphics API | OpenGL |
| Windowing | GLFW (included under `glfw/`) |
| Python (optional) | Python 3.x with `matplotlib`, `pandas` for plotting |

---

## Build & Run (Windows)

### 1. Open a Developer Command Prompt

```bat
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### 2. Compile

```bat
nvcc main.cu -o sim -allow-unsupported-compiler -Xcompiler "/MD" ^
  -I"glfw\include" -L"glfw\lib-vc2022" ^
  -lglfw3 -lopengl32 -lgdi32 -luser32 -lshell32
```

### 3. Run with a block size configuration

```bat
sim 8 8      # 64 threads/block  — baseline
sim 16 16    # 256 threads/block — best 2D spatial locality
sim 32 8     # 256 threads/block — optimal memory coalescing
```

Performance metrics are automatically written to the corresponding CSV file on exit.

---

## Controls

| Input | Action |
|---|---|
| Left mouse button (hold) | Spawn sand at cursor |
| Right mouse button (hold) | Trigger blast / eruption effect |
| Mouse movement | Aim cursor |

---

## Performance Analysis

After running the simulation, generate graphs using the provided Python scripts.

**Per-configuration graphs** (frame time, FPS, kernel time):
```bash
python plot.py
```

**Cross-configuration comparison** (all three block sizes overlaid):
```bash
python compare.py
```

### Results Summary

| Block Size | Threads/Block | Mean FPS | Mean Frame Time | Key Advantage |
|---|---|---|---|---|
| 8×8 | 64 | Lowest | Highest | Simple baseline; minimal register pressure |
| 16×16 | 256 | High | Low | 2D spatial locality; better cache utilisation |
| 32×8 | 256 | Highest | Lowest | Warp-aligned coalesced row reads |

### Key Observations

- **32×8** achieves the lowest kernel times due to fully coalesced 128-byte cache-line reads aligned with CUDA's 32-thread warp width
- **16×16** improves two-dimensional spatial locality, beneficial when diagonal neighbour accesses are the bottleneck
- **8×8** is the simplest baseline but is limited by low per-block warp count (2 warps/block)
- The dominant per-frame cost is `cudaMemcpy` of the 1.92 MB colour buffer from device to host — a known bottleneck targeted for future work

---

## Limitations

- Only sand is implemented as a particle type
- 1.92 MB device-to-host memory transfer every frame (colour buffer)
- Grid resolution is fixed at 800×600
- Simulation state is not persisted between sessions

---

## Future Work

- **CUDA–OpenGL PBO interoperability** — eliminate the per-frame `cudaMemcpy` entirely
- **Multiple particle types** — water (horizontal spread), fire (upward rise), smoke
- **Dynamic grid resizing** — reallocate GPU buffers on window resize
- **State persistence** — serialise `d_gridInput` to disk for save/load support

---

## Contributors

| Name | Roll No |
|---|---|
| Shreyash Tripathi | 102303328 |
| Harpuneet Singh | 102483073 |
| Aunish Kumar Yadav | 102303306 |

**Guided by:** Dr. Saif Nalband (Assistant Professor, DCSE)  
**Department of Computer Science and Engineering**  
Thapar Institute of Engineering and Technology (Deemed to be University), Patiala — 147004

---

## References

1. J. Nickolls et al., "Scalable parallel programming with CUDA," *ACM Queue*, vol. 6, 2008
2. C. McIvor and B. H. Kaye, "Cellular automata simulation of granular materials," *Powder Technology*, vol. 190, 2009
3. D. B. Kirk and W.-M. W. Hwu, *Programming Massively Parallel Processors*, Morgan Kaufmann, 2016
4. J. Sanders and E. Kandrot, *CUDA by Example*, Addison-Wesley Professional, 2010
5. D. Shreiner et al., *OpenGL Programming Guide*, 8th ed., Addison-Wesley Professional, 2013

---

## Acknowledgement

This project was developed as part of academic coursework for **UCS645 — Parallel and Distributing Computing** to demonstrate real-time GPU-based simulation, parallel cellular automata, and CUDA performance analysis techniques.

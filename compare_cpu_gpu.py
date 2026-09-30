"""Compare the CUDA (GPU) and OpenMP (CPU) benchmark results as markdown tables.

Reads benchmark_results.csv (written by run_benchmarks.bat) and benchmark_cpu_results.csv
(written by run_cpu_benchmarks.bat). Repeated runs are combined with the median.

GPU time = "gpu_ms" (simulation + culling kernels). CPU time = "total_ms" (the same two
passes on the processor). Drawing is excluded on both sides, so the two are comparable.

    python compare_cpu_gpu.py [gpu.csv] [cpu.csv]
"""
import csv
import statistics
import sys
from collections import defaultdict

GPU_BLOCK = "32x4x2"


def load(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"{path} not found")


def median(values):
    return statistics.median(values)


def table(title, header, lines):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for line in lines:
        print("| " + " | ".join(line) + " |")


def fmt_ms(v):
    return f"{v:.3f}" if v < 10 else f"{v:.1f}"


def main():
    gpu_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    cpu_path = sys.argv[2] if len(sys.argv) > 2 else "benchmark_cpu_results.csv"
    gpu_rows, cpu_rows = load(gpu_path), load(cpu_path)

    gpu = defaultdict(list)
    for r in gpu_rows:
        if r["block"] == GPU_BLOCK:
            gpu[int(r["grid"])].append(float(r["gpu_ms"]))
    gpu_ms = {g: median(v) for g, v in gpu.items()}
    gpu_name = gpu_rows[0]["gpu"] if gpu_rows else "?"
    cpu_name = cpu_rows[0]["cpu"] if cpu_rows else "?"
    print(f"GPU: {gpu_name}\nCPU: {cpu_name}")

    # (power, grid, threads) -> median total_ms
    cpu = defaultdict(list)
    for r in cpu_rows:
        cpu[(r["power"], int(r["grid"]), int(r["threads"]))].append(float(r["total_ms"]))
    cpu_ms = {k: median(v) for k, v in cpu.items()}

    for power in sorted({k[0] for k in cpu_ms}):
        grids = sorted({k[1] for k in cpu_ms if k[0] == power})
        lines = []
        for grid in grids:
            threads = sorted(t for (p, g, t) in cpu_ms if p == power and g == grid)
            t1, tmax = threads[0], threads[-1]
            c1, cm = cpu_ms[(power, grid, t1)], cpu_ms[(power, grid, tmax)]
            g_ms = gpu_ms.get(grid)
            multi = tmax != t1
            row = [f"{grid}³", f"{grid ** 3:,}", fmt_ms(c1),
                   fmt_ms(cm) if multi else "–", f"{c1 / cm:.1f}×" if multi else "–"]
            if g_ms:
                row += [fmt_ms(g_ms), f"{c1 / g_ms:.0f}×", f"{cm / g_ms:.0f}×" if multi else "–"]
            else:
                row += ["–", "–", "–"]
            lines.append(row)
        tmax_all = max(k[2] for k in cpu_ms if k[0] == power)
        table(f"GPU vs CPU by grid size (CPU on {power}; ms per frame, simulation + culling)",
              ["Grid", "Cells", "CPU 1 thread", f"CPU {tmax_all} threads", "CPU thread speedup",
               "GPU", "GPU vs 1 thread", f"GPU vs {tmax_all} threads"], lines)

        # thread scaling at the grid that has the most thread counts
        by_grid = defaultdict(set)
        for (p, g, t) in cpu_ms:
            if p == power:
                by_grid[g].add(t)
        g_scale = max(by_grid, key=lambda g: len(by_grid[g]))
        if len(by_grid[g_scale]) > 1:
            ts = sorted(by_grid[g_scale])
            base = cpu_ms[(power, g_scale, ts[0])]
            lines = []
            for t in ts:
                v = cpu_ms[(power, g_scale, t)]
                lines.append([str(t), fmt_ms(v), f"{base / v:.2f}×", f"{base / v / t * 100:.0f}%"])
            table(f"CPU thread scaling at {g_scale}³ ({power})",
                  ["Threads", "ms per frame", "Speedup vs 1 thread", "Parallel efficiency"], lines)


if __name__ == "__main__":
    main()

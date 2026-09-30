"""Summarise benchmark_results.csv (written by `sim3d bench`) as markdown tables.

Repeated runs of the same configuration are combined with the median, which ignores
the occasional slow run (driver hiccup, background app).

    python summarize_benchmarks.py [benchmark_results.csv]
"""
import csv
import statistics
import sys
from collections import defaultdict

DEFAULT_BLOCK = "32x4x2"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["gpu"], int(r["grid"]), r["block"])].append(r)
    return g


def med(rows, key):
    return statistics.median(float(r[key]) for r in rows)


def table(title, header, lines):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for line in lines:
        print("| " + " | ".join(line) + " |")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    try:
        rows = load(path)
    except FileNotFoundError:
        sys.exit(f"{path} not found - run run_benchmarks.bat first")
    if not rows:
        sys.exit(f"{path} has no data rows")

    g = group(rows)
    print(f"GPU: {rows[0]['gpu']}   ({len(rows)} runs)")

    # Sweep 1: grid size at the default block shape
    lines = []
    for (gpu, grid, block), rs in sorted(g.items(), key=lambda kv: kv[0][1]):
        if block != DEFAULT_BLOCK:
            continue
        cells = int(rs[0]["cells"])
        lines.append([
            f"{grid}³", f"{cells:,}", f"{med(rs, 'visible_grains_avg'):,.0f}",
            f"{med(rs, 'sim_ms'):.3f}", f"{med(rs, 'cull_ms'):.3f}",
            f"{med(rs, 'gpu_ms'):.3f}", f"{med(rs, 'frame_ms'):.2f}", f"{med(rs, 'fps'):.0f}",
        ])
    if lines:
        table(f"Grid size (block {DEFAULT_BLOCK})",
              ["Grid", "Cells", "Visible grains", "Sim ms", "Cull ms", "GPU ms", "Frame ms", "FPS"], lines)

    # Sweep 2: block shape at the largest grid that has more than one block shape
    by_grid = defaultdict(set)
    for (gpu, grid, block) in g:
        by_grid[grid].add(block)
    multi = [n for n, b in by_grid.items() if len(b) > 1]
    if multi:
        n = max(multi)
        lines = []
        for (gpu, grid, block), rs in sorted(g.items(), key=lambda kv: med(kv[1], "gpu_ms")):
            if grid != n:
                continue
            lines.append([block, f"{med(rs, 'sim_ms'):.3f}", f"{med(rs, 'cull_ms'):.3f}",
                          f"{med(rs, 'gpu_ms'):.3f}", f"{med(rs, 'fps'):.0f}"])
        table(f"Block shape at {n}³ (fastest first)",
              ["Block", "Sim ms", "Cull ms", "GPU ms", "FPS"], lines)


if __name__ == "__main__":
    main()

"""Summarise benchmark_2d_results.csv (written by `sim2d bench`) as markdown tables.

Repeated runs of the same configuration are combined with the median.

    python summarize_2d_benchmarks.py [benchmark_2d_results.csv]

Columns of interest: sim_ms (sand spawn + simulation kernel), render_ms (colour kernel),
copy_ms (device-to-host copy of the colour buffer), gpu_ms (their sum), frame_ms
(wall-clock per frame, including the OpenGL upload and draw) and fps.
"""
import csv
import statistics
import sys
from collections import defaultdict

DEFAULT_BLOCK = "8x8"


def load(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"{path} not found - run run_2d_benchmarks.bat first")


def table(title, header, lines):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for line in lines:
        print("| " + " | ".join(line) + " |")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_2d_results.csv"
    rows = load(path)
    if not rows:
        sys.exit(f"{path} has no data rows")

    groups = defaultdict(list)
    for r in rows:
        groups[(int(r["width"]), int(r["height"]), r["block"], int(r["pinned"]))].append(r)

    def med(key, name):
        return statistics.median(float(r[name]) for r in groups[key])

    print(f"GPU: {rows[0]['gpu']}   ({len(rows)} runs)")

    # Sweep 1: resolution (default block, pageable memory)
    keys = sorted((k for k in groups if k[2] == DEFAULT_BLOCK and k[3] == 0), key=lambda k: k[0] * k[1])
    if keys:
        lines = []
        for k in keys:
            gpu = med(k, "gpu_ms")
            lines.append([f"{k[0]}x{k[1]}", f"{k[0] * k[1]:,}", f"{med(k, 'sim_ms'):.3f}",
                          f"{med(k, 'render_ms'):.3f}", f"{med(k, 'copy_ms'):.3f}", f"{gpu:.3f}",
                          f"{med(k, 'copy_ms') / gpu * 100:.0f}%", f"{med(k, 'frame_ms'):.2f}",
                          f"{med(k, 'fps'):.0f}"])
        table(f"Resolution (block {DEFAULT_BLOCK}, pageable host memory)",
              ["Resolution", "Cells", "Sim ms", "Render ms", "Copy ms", "GPU ms", "Copy share",
               "Frame ms", "FPS"], lines)

    # Sweep 2: block shape at the resolution with the most block shapes
    by_res = defaultdict(set)
    for (w, h, b, p) in groups:
        if p == 0:
            by_res[(w, h)].add(b)
    multi = [r for r, b in by_res.items() if len(b) > 1]
    if multi:
        res = max(multi, key=lambda r: len(by_res[r]))
        ks = [k for k in groups if (k[0], k[1]) == res and k[3] == 0]
        ks.sort(key=lambda k: med(k, "sim_ms") + med(k, "render_ms"))
        base = None
        for k in ks:
            if k[2] == DEFAULT_BLOCK:
                base = med(k, "sim_ms") + med(k, "render_ms")
        lines = []
        for k in ks:
            kern = med(k, "sim_ms") + med(k, "render_ms")
            rel = f"{(kern / base - 1) * 100:+.1f}%" if base else "–"
            lines.append([k[2], f"{med(k, 'sim_ms'):.3f}", f"{med(k, 'render_ms'):.3f}", f"{kern:.3f}", rel])
        table(f"Block shape at {res[0]}x{res[1]} (kernels only, fastest first)",
              ["Block", "Sim ms", "Render ms", "Kernels ms", f"vs {DEFAULT_BLOCK}"], lines)

    # Sweep 3: pinned vs pageable host memory (default block)
    lines = []
    for (w, h, b, p) in sorted(groups, key=lambda k: k[0] * k[1]):
        if b != DEFAULT_BLOCK or p != 1 or (w, h, b, 0) not in groups:
            continue
        a, c = (w, h, b, 0), (w, h, b, 1)
        lines.append([f"{w}x{h}", f"{med(a, 'copy_ms'):.3f}", f"{med(c, 'copy_ms'):.3f}",
                      f"{med(a, 'copy_ms') / med(c, 'copy_ms'):.1f}×", f"{med(a, 'frame_ms'):.2f}",
                      f"{med(c, 'frame_ms'):.2f}", f"{med(a, 'fps'):.0f}", f"{med(c, 'fps'):.0f}"])
    if lines:
        table("Pageable vs pinned host memory for the colour-buffer copy",
              ["Resolution", "Copy ms pageable", "Copy ms pinned", "Copy speedup", "Frame ms pageable",
               "Frame ms pinned", "FPS pageable", "FPS pinned"], lines)


if __name__ == "__main__":
    main()

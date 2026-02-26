#!/usr/bin/env python3
"""
Find delta such that P_fail <= 2^(-δ * n^0.99)
for punctured self-dual Reed-Muller codes RM*(r, 2r+1) on BEC,
bit 0 always erased.

Sweeps r (rmin..rmax), with m = 2r+1 auto-computed.
Code length n = 2^m - 1 = 2^(2r+1) - 1.

Usage:
    python3 run_delta_fit.py --rmin 2 --rmax 7 --frames 50000 --eps 0.4

Requirements:
    - rm_bec_sim compiled in current directory
"""

import subprocess, sys, os, math, argparse, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


def binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n - k:
        k = n - k
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def rm_dim(r, m):
    return sum(binom(m, i) for i in range(r + 1))


def run_sim(r, m, eps, nframes, nthr, pbar=None):
    """Run rm_bec_sim at a single epsilon. Returns P_amb or None.
    If pbar (tqdm bar) is given, updates it with frame-level progress."""
    cmd = ["./rm_bec_sim",
           "-r", str(r), "-m", str(m), "-f", str(nframes),
           "-s", f"{eps:.6f}", "-e", f"{eps:.6f}", "-d", "0.01",
           "-t", str(nthr), "-p"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    # Read stderr in a thread to capture progress without blocking stdout
    import threading, io
    stderr_lines = []
    prev_done = 0

    def read_stderr():
        nonlocal prev_done
        for line in proc.stderr:
            line = line.strip()
            stderr_lines.append(line)
            if pbar and line.startswith("PROGRESS "):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        done = int(parts[1])
                        delta = done - prev_done
                        if delta > 0:
                            pbar.update(delta)
                            prev_done = done
                    except ValueError:
                        pass

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    stdout_data = proc.stdout.read()
    proc.wait()
    t.join(timeout=2)

    # Ensure pbar reaches 100%
    if pbar and prev_done < nframes:
        pbar.update(nframes - prev_done)

    if proc.returncode != 0:
        err = "\n".join(stderr_lines[:5])
        tqdm.write(f"    SIM ERROR RM*({r},{m}): {err[:200]}")
        return None

    for line in stdout_data.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(s) for s in
               ["Built", "Building", "Code", "Frame", "SIMD", "eps",
                "---", "Done", "Stored", "On-the-fly"]):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                e = float(parts[0])
                if abs(e - eps) < 0.001:
                    return float(parts[1])
            except ValueError:
                continue

    tqdm.write(f"    PARSE FAIL RM*({r},{m}), stdout:")
    for line in stdout_data.strip().split("\n")[-5:]:
        tqdm.write(f"      [{line}]")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fit δ for RM*(r, 2r+1): P_amb ≤ 2^(-δ·n^0.99)")
    parser.add_argument("--rmin", type=int, default=2,
                        help="minimum r (default 2)")
    parser.add_argument("--rmax", type=int, default=7,
                        help="maximum r (default 7)")
    parser.add_argument("--frames", type=int, default=50000,
                        help="Monte Carlo frames per point (default 50000)")
    parser.add_argument("--eps", type=float, default=0.4,
                        help="erasure probability (default 0.4)")
    parser.add_argument("--threads", type=int, default=0,
                        help="OpenMP threads (default: all)")
    args = parser.parse_args()

    nthr = args.threads if args.threads > 0 else os.cpu_count() or 4
    eps = args.eps

    if not os.path.exists("./rm_bec_sim"):
        print("ERROR: ./rm_bec_sim not found. Compile first:")
        print("  gcc -O3 -march=native -fopenmp -o rm_bec_sim rm_bec_sim.c -lm")
        return 1

    rs = list(range(args.rmin, args.rmax + 1))
    if not rs:
        print(f"ERROR: no valid r in [{args.rmin}, {args.rmax}]")
        return 1

    codes = []  # (r, m, n, k)
    for r in rs:
        m = 2 * r + 1
        n = (1 << m) - 1
        k = rm_dim(r, m)
        codes.append((r, m, n, k))

    print(f"Punctured self-dual RM*(r, 2r+1) δ fit")
    print(f"{'='*60}")
    print(f"  r = {rs[0]}..{rs[-1]} ({len(rs)} codes), m = 2r+1, n = 2^m - 1")
    print(f"  ε = {eps}, frames = {args.frames}, threads = {nthr}")
    print()

    # Code info table
    print(f"  {'r':>3}  {'m':>3}  {'n':>7}  {'k':>7}  {'n-k':>7}  {'R':>7}")
    print(f"  {'---':>3}  {'---':>3}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>7}")
    for r, m, n, k in codes:
        print(f"  {r:>3}  {m:>3}  {n:>7}  {k:>7}  {n-k:>7}  {k/n:>7.4f}")
    print()

    # Step 1: Run simulations
    print(f"STEP 1: Simulate at ε = {eps}")
    print("-" * 60)
    print(f"  {'r':>3}  {'m':>3}  {'n':>7}  {'P_amb':>14}  {'time':>8}")
    print(f"  {'---':>3}  {'---':>3}  {'---':>7}  {'---':>14}  {'---':>8}")

    results = []  # (r, m, n, k, P_amb)
    stopped_early = False
    for idx, (r, m, n, k) in enumerate(codes):
        label = f"[{idx+1}/{len(codes)}] RM*({r},{2*r+1}) n={n}"
        if stopped_early:
            results.append((r, m, n, k, 0.0))
            tqdm.write(f"  {r:>3}  {m:>3}  {n:>7}  {'0 (skipped)':>14}  {'--':>8}")
            continue
        pbar = tqdm(total=args.frames, desc=label, unit="fr",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        t0 = time.time()
        pamb = run_sim(r, m, eps, args.frames, nthr, pbar=pbar)
        dt = time.time() - t0
        pbar.close()
        results.append((r, m, n, k, pamb))
        ps = f"{pamb:.6e}" if pamb is not None else "N/A"
        tqdm.write(f"  {r:>3}  {m:>3}  {n:>7}  {ps:>14}  {dt:>7.1f}s")
        if pamb is not None and pamb == 0.0:
            tqdm.write(f"  >>> P_amb = 0 at r={r} (n={n}), skipping larger r")
            stopped_early = True

    # Save raw CSV
    with open("delta_raw.csv", "w") as f:
        f.write("r,m,n,k,eps,P_amb\n")
        for r, m, n, k, pamb in results:
            ps = f"{pamb:.12e}" if pamb is not None else ""
            f.write(f"{r},{m},{n},{k},{eps},{ps}\n")
    print(f"\nRaw data saved to delta_raw.csv\n")

    # Step 2: Fit delta
    EXP = 0.99
    print(f"STEP 2: Fit δ  (model: P_amb ≤ 2^(-δ·n^{EXP}))")
    print("-" * 60)

    points = []  # (r, m, n, secbits, pamb)
    for r, m, n, k, pamb in results:
        if pamb is not None and pamb > 0:
            secbits = -math.log2(2 * pamb)
            points.append((r, m, n, secbits, pamb))

    if len(points) == 0:
        print("  No data points with P_amb > 0.")
        print("  Try increasing --frames.")
        return 1

    print(f"\n  {'r':>3}  {'m':>3}  {'n':>7}  {'P_amb':>14}  {'sec bits':>12}  {'bits/n^'+str(EXP):>14}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*7}  {'-'*14}  {'-'*12}  {'-'*14}")
    for r, m, n, sb, pf in points:
        ne = n ** EXP
        print(f"  {r:>3}  {m:>3}  {n:>7}  {pf:>14.6e}  {sb:>12.4f}  {sb/ne:>14.6f}")

    # Save fit results
    with open("delta_fit.csv", "w") as f:
        f.write(f"r,m,n,P_amb,sec_bits,sec_bits_over_n^{EXP}\n")
        for r, m, n, sb, pf in points:
            ne = n ** EXP
            f.write(f"{r},{m},{n},{pf:.12e},{sb:.10f},{sb/ne:.10f}\n")
    print(f"\n  Fit data saved to delta_fit.csv")

    if len(points) < 2:
        print(f"\n  Only {len(points)} data point(s) — skipping fit and plot.")
        return 0

    # Linear regression: sec_bits = δ·n^EXP + c
    sx  = sum(n ** EXP for _, _, n, _, _ in points)
    sy  = sum(sb for _, _, _, sb, _ in points)
    sxx = sum((n ** EXP) ** 2 for _, _, n, _, _ in points)
    sxy = sum((n ** EXP) * sb for _, _, n, sb, _ in points)
    nn  = len(points)

    denom = nn * sxx - sx * sx
    if abs(denom) > 1e-15:
        delta_fit = (nn * sxy - sx * sy) / denom
        intercept = (sy - delta_fit * sx) / nn
    else:
        delta_fit, intercept = sxy / sxx, 0

    # R²
    ss_res = sum((sb - delta_fit * (n ** EXP) - intercept)**2 for _, _, n, sb, _ in points)
    ss_tot = sum((sb - sy / nn)**2 for _, _, _, sb, _ in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # δ_min: if c >= 0, δ_fit·x is already below δ_fit·x+c, so δ_min = δ_fit
    #        if c <  0, use min ratio across all points
    if intercept >= 0:
        delta_min = delta_fit
    else:
        delta_min = min(sb / (n ** EXP) for _, _, n, sb, _ in points)

    print(f"\n  Results ({len(points)} data points):")
    print(f"  ─────────────────────────────────────────")
    print(f"  δ_min = {delta_min:.8f}  ({'= δ_fit (c≥0)' if intercept >= 0 else 'min ratio (c<0)'})")
    print(f"  δ_fit = {delta_fit:.8f}  (c = {intercept:.4f}, R² = {r2:.6f})")
    print()

    # Target security levels
    print(f"  Target security (using δ_min = {delta_min:.6f}):")
    for target in [40, 60, 80, 128]:
        n_req = (target / delta_min) ** (1.0 / EXP)
        print(f"    {target:>3}-bit security:  n ≥ {n_req:.0f}")
    print()

    # ── Plot ──
    print(f"  Generating plot ...")

    ne_arr = np.array([d[2] ** EXP for d in points])
    sbs    = np.array([d[3] for d in points])

    ymax = 40
    n_max = max(d[2] for d in points)
    ne_max = (n_max * 1.15) ** EXP          # 15% padding beyond largest n
    ne_range = np.linspace(0, ne_max, 400)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(ne_arr, sbs, s=60, c="#2563eb", zorder=5,
               label="Simulated", edgecolors="white", linewidth=0.5)
    # clip hypothesis lines to y <= ymax
    y_min_line = delta_min * ne_range
    y_fit_line = delta_fit * ne_range + intercept
    ax.plot(ne_range[y_min_line <= ymax], y_min_line[y_min_line <= ymax],
            "--", color="#ef4444", linewidth=2,
            label=f"δ_min·n^{EXP}  (δ_min = {delta_min:.6f})")
    ax.plot(ne_range[y_fit_line <= ymax], y_fit_line[y_fit_line <= ymax],
            "-.", color="#f59e0b", linewidth=2,
            label=f"δ_fit·n^{EXP} + c  (δ_fit = {delta_fit:.6f}, c={intercept:.2f}, R²={r2:.3f})")
    for r, m, n, sb, _ in points:
        ax.annotate(f"n={n}", (n ** EXP, sb), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), color="#64748b")
    for sec, col in [(20, "#cbd5e1"), (40, "#94a3b8")]:
        ax.axhline(sec, color=col, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(ne_max * 0.98, sec, f"{sec}-bit", fontsize=8, color=col, va="bottom", ha="right")

    ax.set_xlabel(f"$n^{{{EXP}}}$", fontsize=12)
    ax.set_ylabel("Security bits  (−log₂ P_fail)", fontsize=12)
    ax.set_title(f"RM*(r, 2r+1) on BEC (ε = {eps})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0, right=ne_max)
    ax.set_ylim(top=ymax)

    plt.tight_layout()
    plt.savefig("delta_fit.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to delta_fit.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())

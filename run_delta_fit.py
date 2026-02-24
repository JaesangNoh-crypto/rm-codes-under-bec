#!/usr/bin/env python3
"""
Find delta such that P_fail <= 2^(-δ₂ * √n)
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


def run_sim(r, m, eps, nframes, nthr):
    """Run rm_bec_sim at a single epsilon. Returns P_amb or None."""
    cmd = ["./rm_bec_sim",
           "-r", str(r), "-m", str(m), "-f", str(nframes),
           "-s", f"{eps:.6f}", "-e", f"{eps:.6f}", "-d", "0.01",
           "-t", str(nthr)]
    ret = subprocess.run(cmd, capture_output=True, text=True)

    if ret.returncode != 0:
        print(f"    SIM ERROR RM*({r},{m}): {ret.stderr[:200]}")
        return None

    for line in ret.stdout.strip().split("\n"):
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

    print(f"    PARSE FAIL RM*({r},{m}), stdout:")
    for line in ret.stdout.strip().split("\n")[-5:]:
        print(f"      [{line}]")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fit δ₂ for RM*(r, 2r+1): P_amb ≤ 2^(-δ₂·√n)")
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

    print(f"Punctured self-dual RM*(r, 2r+1) δ₂ fit")
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
    for r, m, n, k in codes:
        if stopped_early:
            results.append((r, m, n, k, 0.0))
            print(f"  {r:>3}  {m:>3}  {n:>7}  {'0 (skipped)':>14}  {'--':>8}")
            continue
        t0 = time.time()
        pamb = run_sim(r, m, eps, args.frames, nthr)
        dt = time.time() - t0
        results.append((r, m, n, k, pamb))
        ps = f"{pamb:.6e}" if pamb is not None else "N/A"
        print(f"  {r:>3}  {m:>3}  {n:>7}  {ps:>14}  {dt:>7.1f}s")
        if pamb is not None and pamb == 0.0:
            print(f"  >>> P_amb = 0 at r={r} (n={n}), skipping larger r")
            stopped_early = True

    # Save raw CSV
    with open("delta_raw.csv", "w") as f:
        f.write("r,m,n,k,eps,P_amb\n")
        for r, m, n, k, pamb in results:
            ps = f"{pamb:.12e}" if pamb is not None else ""
            f.write(f"{r},{m},{n},{k},{eps},{ps}\n")
    print(f"\nRaw data saved to delta_raw.csv\n")

    # Step 2: Fit delta
    print("STEP 2: Fit δ₂  (model: P_amb ≤ 2^(-δ₂·√n))")
    print("-" * 60)

    points = []  # (r, m, n, sqn, secbits, pamb)
    for r, m, n, k, pamb in results:
        if pamb is not None and pamb > 0:
            sqn = math.sqrt(n)
            secbits = -math.log2(2 * pamb)
            points.append((r, m, n, sqn, secbits, pamb))

    if len(points) == 0:
        print("  No data points with P_amb > 0.")
        print("  Try increasing --frames.")
        return 1

    print(f"\n  {'r':>3}  {'m':>3}  {'n':>7}  {'√n':>8}  {'P_amb':>14}  {'sec bits':>12}  {'bits/√n':>12}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*7}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*12}")
    for r, m, n, sqn, sb, pf in points:
        print(f"  {r:>3}  {m:>3}  {n:>7}  {sqn:>8.3f}  {pf:>14.6e}  {sb:>12.4f}  {sb/sqn:>12.6f}")

    # Save fit results
    with open("delta_fit.csv", "w") as f:
        f.write("r,m,n,sqrt_n,P_amb,sec_bits,sec_bits_over_sqrt_n\n")
        for r, m, n, sqn, sb, pf in points:
            f.write(f"{r},{m},{n},{sqn:.6f},{pf:.12e},{sb:.10f},{sb/sqn:.10f}\n")
    print(f"\n  Fit data saved to delta_fit.csv")

    if len(points) < 2:
        print(f"\n  Only {len(points)} data point(s) — skipping fit and plot.")
        return 0

    # δ₂_min: guaranteed bound for all tested n (in bits)
    delta2_min = min(sb / sqn for _, _, _, sqn, sb, _ in points)

    # Linear regression: sec_bits = δ₂·√n + c
    sx  = sum(sq for _, _, _, sq, _, _ in points)
    sy  = sum(sb for _, _, _, _, sb, _ in points)
    sxx = sum(sq * sq for _, _, _, sq, _, _ in points)
    sxy = sum(sq * sb for _, _, _, sq, sb, _ in points)
    nn  = len(points)

    denom = nn * sxx - sx * sx
    if abs(denom) > 1e-15:
        delta2_fit = (nn * sxy - sx * sy) / denom
        intercept = (sy - delta2_fit * sx) / nn
    else:
        delta2_fit, intercept = sxy / sxx, 0

    # R²
    ss_res = sum((sb - delta2_fit * sq - intercept)**2 for _, _, _, sq, sb, _ in points)
    ss_tot = sum((sb - sy / nn)**2 for _, _, _, _, sb, _ in points)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"\n  Results ({len(points)} data points):")
    print(f"  ─────────────────────────────────────────")
    print(f"  δ₂_min = {delta2_min:.6f}  (guaranteed: P ≤ 2^(-{delta2_min:.4f}·√n) ∀n)")
    print(f"  δ₂_fit = {delta2_fit:.6f}  (c = {intercept:.4f}, R² = {r2:.6f})")
    print()

    # Target security levels
    print(f"  Target security (using δ₂_min = {delta2_min:.4f}):")
    for target in [40, 60, 80, 128]:
        n_req = (target / delta2_min) ** 2
        print(f"    {target:>3}-bit security:  n ≥ {n_req:.0f}  (√n ≥ {math.sqrt(n_req):.1f})")
    print()

    # ── Plot ──
    print(f"  Generating plot ...")

    sqns = np.array([d[3] for d in points])
    sbs  = np.array([d[4] for d in points])

    sq_max = max(max(sqns) * 1.15, 62 / delta2_min * 1.05)
    sq_range = np.linspace(0, sq_max, 200)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(sqns, sbs, s=60, c="#2563eb", zorder=5,
               label="Simulated", edgecolors="white", linewidth=0.5)
    ax.plot(sq_range, delta2_min * sq_range,
            "--", color="#ef4444", linewidth=2,
            label=f"δ₂_min = {delta2_min:.4f}")
    ax.plot(sq_range, delta2_fit * sq_range + intercept,
            "-.", color="#f59e0b", linewidth=2,
            label=f"δ₂_fit = {delta2_fit:.4f}, c={intercept:.2f} (R²={r2:.3f})")
    for r, m, n, sq, sb, _ in points:
        ax.annotate(f"r={r}", (sq, sb), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), color="#64748b")
    for sec, col in [(20, "#cbd5e1"), (40, "#94a3b8"), (60, "#64748b")]:
        ax.axhline(sec, color=col, linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(sq_max * 0.98, sec, f"{sec}-bit", fontsize=8, color=col, va="bottom", ha="right")

    ax.set_xlabel("√n", fontsize=12)
    ax.set_ylabel("Security bits  (−log₂ P_fail)", fontsize=12)
    ax.set_title(f"RM*(r, 2r+1) on BEC (ε = {eps})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig("delta_fit.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved to delta_fit.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())

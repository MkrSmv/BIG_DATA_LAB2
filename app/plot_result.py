"""
plot_result.py — Hadoop/Spark Performance Visualization
Generates 6-panel comparison chart from experiment result files.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import sys
import os

# ── Dark theme ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#c9d1d9",
    "ytick.color":       "#c9d1d9",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.8,
    "font.family":       "DejaVu Sans",
})

# ── Experiment definitions ───────────────────────────────────
EXPERIMENTS = [
    ("results/res_one_node.txt",      "1DN\nNormal"),
    ("results/res_one_node_opt.txt",  "1DN\nOptimized"),
    ("results/res_three_node.txt",    "3DN\nNormal"),
    ("results/res_three_node_opt.txt","3DN\nOptimized"),
]

COLORS   = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff"]
PHASE_C  = {"Load": "#58a6ff", "AGG": "#ffa657", "ML": "#ff7b72", "Other": "#8b949e"}
METRICS  = ["TIME", "MEMORY", "LOAD_TIME", "AGG_TIME", "ML_TIME", "ACCURACY", "ROWS"]


# ── Parsing ──────────────────────────────────────────────────
def parse_results():
    results = []
    for fname, label in EXPERIMENTS:
        if not os.path.exists(fname):
            print(f"[WARN] Missing: {fname}")
            continue
        data = {"label": label}
        with open(fname, "r") as f:
            for line in f:
                for m in METRICS:
                    if f"[{m}]" in line:
                        raw = line.strip().split("]", 1)[1].lower().replace("mb", "").strip()
                        try:
                            data[m] = float(raw)
                        except ValueError:
                            pass
        results.append(data)
    return results


# ── Helper: value labels on bars ─────────────────────────────
def label_bars(ax, bars, fmt="{:.1f}", unit="", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + h * 0.015 + 0.01,
            fmt.format(h) + unit,
            ha="center", va="bottom", fontsize=fontsize,
            color="white", fontweight="bold",
        )


def style_ax(ax, title, ylabel, xlabel_labels, colors, xs):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="white")
    ax.set_ylabel(ylabel, fontsize=9, color="#8b949e")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabel_labels, fontsize=9, color="#c9d1d9")
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Main ─────────────────────────────────────────────────────
def main():
    results = parse_results()
    if not results:
        print("No result files found. Run the experiments first.")
        sys.exit(1)

    labels = [r["label"] for r in results]
    colors = COLORS[: len(labels)]
    xs     = np.arange(len(labels))

    # ── Derived metrics ──────────────────────────────────────
    times     = [r.get("TIME",        0) for r in results]
    memories  = [r.get("MEMORY",      0) for r in results]
    load_ts   = [r.get("LOAD_TIME",   0) for r in results]
    agg_ts    = [r.get("AGG_TIME",    0) for r in results]
    ml_ts     = [r.get("ML_TIME",     0) for r in results]
    accuracies= [r.get("ACCURACY",    0) * 100 for r in results]
    rows      = [r.get("ROWS",        1) for r in results]

    # Throughput (rows per second)
    throughput = [r / t if t > 0 else 0 for r, t in zip(rows, times)]

    # Speedup vs first experiment
    baseline = times[0] if times[0] > 0 else 1
    speedup  = [baseline / t if t > 0 else 0 for t in times]

    # Other time (preprocessing + evaluation + feature eng)
    other_ts = [
        max(t - l - a - m, 0)
        for t, l, a, m in zip(times, load_ts, agg_ts, ml_ts)
    ]

    # ── Figure layout: 2 rows × 3 cols ───────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Hadoop / Spark Cluster — Performance Comparison\n"
        "Dataset: NYC Yellow Taxi 2023  |  Task: Logistic Regression (payment_type)",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # ── Chart 1: Total Execution Time ────────────────────────
    bars1 = ax1.bar(xs, times, color=colors, width=0.55, zorder=2)
    style_ax(ax1, "① Total Execution Time", "Seconds", labels, colors, xs)
    label_bars(ax1, bars1, "{:.1f}", "s")

    # ── Chart 2: Phase Breakdown (stacked bar) ───────────────
    b_load  = ax2.bar(xs, load_ts,  label="Load",      color=PHASE_C["Load"],  width=0.55, zorder=2)
    b_agg   = ax2.bar(xs, agg_ts,   label="Agg.",      color=PHASE_C["AGG"],   width=0.55, bottom=load_ts, zorder=2)
    b_ml    = ax2.bar(xs, ml_ts,    label="ML train",  color=PHASE_C["ML"],    width=0.55,
                      bottom=[l + a for l, a in zip(load_ts, agg_ts)], zorder=2)
    b_other = ax2.bar(xs, other_ts, label="Other",     color=PHASE_C["Other"], width=0.55,
                      bottom=[l + a + m for l, a, m in zip(load_ts, agg_ts, ml_ts)], zorder=2)
    style_ax(ax2, "② Stage Breakdown (Stacked)", "Seconds", labels, colors, xs)
    ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white", loc="upper right")

    # ── Chart 3: Speedup vs Baseline (1DN Normal) ────────────
    bars3 = ax3.bar(xs, speedup, color=colors, width=0.55, zorder=2)
    ax3.axhline(y=1.0, color="#f0883e", linestyle="--", linewidth=1.2, label="Baseline (1×)")
    style_ax(ax3, "③ Speedup vs 1DN Normal", "Speedup (×)", labels, colors, xs)
    label_bars(ax3, bars3, "{:.2f}", "×")
    ax3.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    # ── Chart 4: Throughput (rows/sec) ───────────────────────
    tput_k = [t / 1000 for t in throughput]
    bars4 = ax4.bar(xs, tput_k, color=colors, width=0.55, zorder=2)
    style_ax(ax4, "④ Throughput", "Rows / Second (×1000)", labels, colors, xs)
    label_bars(ax4, bars4, "{:.0f}", "k")

    # ── Chart 5: Model Accuracy ───────────────────────────────
    bars5 = ax5.bar(xs, accuracies, color=colors, width=0.55, zorder=2)
    style_ax(ax5, "⑤ Model Accuracy (Logistic Regression)", "Accuracy (%)", labels, colors, xs)
    label_bars(ax5, bars5, "{:.2f}", "%")
    lo = max(min(accuracies) - 5, 0) if accuracies else 0
    ax5.set_ylim(lo, 100)

    # ── Chart 6: Driver Memory Usage ─────────────────────────
    bars6 = ax6.bar(xs, memories, color=colors, width=0.55, zorder=2)
    style_ax(ax6, "⑥ Driver Memory Usage (psutil)", "MB", labels, colors, xs)
    label_bars(ax6, bars6, "{:.0f}", "MB")

    # ── Legend patch row at bottom ────────────────────────────
    patches = [
        mpatches.Patch(color=c, label=lb.replace("\n", " "))
        for c, lb in zip(colors, labels)
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=len(patches),
        fontsize=10, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white", framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    os.makedirs("charts", exist_ok=True)
    plt.savefig("charts/comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Saved: charts/comparison.png")

    # ── Second figure: Phase detail lines ─────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.patch.set_facecolor("#0d1117")
    fig2.suptitle("Stage-Level Time Breakdown per Experiment", fontsize=13, fontweight="bold",
                  color="white", y=1.02)

    for ax, (phase_vals, title, color) in zip(
        axes2,
        [
            (load_ts, "Load Time", PHASE_C["Load"]),
            (agg_ts,  "Aggregation Time", PHASE_C["AGG"]),
            (ml_ts,   "ML Training Time", PHASE_C["ML"]),
        ]
    ):
        bars = ax.bar(xs, phase_vals, color=color, width=0.55, zorder=2)
        style_ax(ax, title, "Seconds", labels, [color] * len(labels), xs)
        label_bars(ax, bars, "{:.2f}", "s")
        ax.grid(axis="y", zorder=0)

    fig2.tight_layout()
    plt.savefig("charts/stages_breakdown.png", dpi=150, bbox_inches="tight",
                facecolor=fig2.get_facecolor())
    print("Saved: charts/stages_breakdown.png")


if __name__ == "__main__":
    main()

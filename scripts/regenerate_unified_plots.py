#!/usr/bin/env python3
"""Regenerate unified, continuous plots across all training runs.

The training loop's PlotGenerator only plots the CURRENT process's
in-memory history. When a run is resumed from a checkpoint, a fresh
logger starts empty, so the plots show only the resumed epochs.

This script walks every `run_*.jsonl` under `checkpoints/logs/`,
merges their epoch / test entries into one continuous timeline, and
writes unified plots to `checkpoints/plots/`.

Policy for duplicates (when two runs both recorded the same epoch):
    the LATER run's data wins. This lets a bad Stage-2 run be
    transparently overwritten by a subsequent resumed run.

Run it any time — even while training is active (reads only).

Usage:
    python scripts/regenerate_unified_plots.py
    # optional: exclude specific runs (e.g. a broken attempt)
    python scripts/regenerate_unified_plots.py --skip-run run_20260419_191237
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_runs(log_dir: Path, skip_runs: set,
                  run_epoch_caps: dict = None) -> dict:
    """Return dict with 'epochs' (list) and 'tests' (list), merged across
    all runs in chronological order. Later runs overwrite earlier ones
    when epoch indices collide.

    Args:
        run_epoch_caps: optional {run_stem: max_epoch}. For listed runs,
            epoch (and test) records with epoch > max_epoch are dropped.
    """
    run_epoch_caps = run_epoch_caps or {}
    epoch_map, test_map = {}, {}
    run_files = sorted(log_dir.glob("run_*.jsonl"))

    for f in run_files:
        if f.stem in skip_runs:
            print(f"  skipping {f.stem}")
            continue
        cap = run_epoch_caps.get(f.stem)  # None → no cap
        dropped_tail = 0
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = rec.get("epoch")
                if cap is not None and ep is not None and ep > cap:
                    dropped_tail += 1
                    continue
                if rec.get("type") == "epoch":
                    epoch_map[rec["epoch"]] = {**rec, "_run": f.stem}
                elif rec.get("type") == "test":
                    test_map[rec["epoch"]] = {**rec, "_run": f.stem}
        if cap is not None and dropped_tail > 0:
            print(f"  capping {f.stem} at epoch {cap} "
                  f"(dropped {dropped_tail} records beyond cap)")

    epochs = [epoch_map[k] for k in sorted(epoch_map)]
    tests = [test_map[k] for k in sorted(test_map)]
    return {"epochs": epochs, "tests": tests, "runs": [f.stem for f in run_files]}


# Stage-1 ends at this epoch (inclusive). Matches curriculum config
# projector_warmup_epochs = 8. Oracle points at or before this epoch are
# measured under "frozen Qwen + Bridge only"; points after are measured under
# "Qwen LoRA + SEG-row trained with Shifted-Zero weighting" — a structurally
# different training regime, so we plot them in a different colour.
STAGE_1_LAST_EPOCH = 8

# Runs to skip entirely when regenerating plots (architecturally obsolete):
#   run_20260419_191237 — pre-Shifted-Zero (weight=0 on SEG target; no SEG
#                         learning at all)
#   run_20260420_070021 — Shifted-Zero only; lm_head[SEG] still frozen random
#                         (Oracle high, live 0.0 — the <|tool_call>/<|im_end|>
#                         substitution bug)
# Pass --include-run <stem> to override and include one explicitly.
DEFAULT_SKIP_RUNS = {
    "run_20260419_191237",
    "run_20260420_070021",
}

# Per-run MAX epoch to keep. Runs that overshot past a "golden anchor" have
# their tail trimmed here — because we resume from that anchor and the tail
# is discarded. Epochs strictly greater than the cap are dropped.
#   run_20260418_205734 — initial run went ep1–10 but ep8 is the Stage-1
#                         peak (RWTD Oracle 0.6691); ep9–10 overfitted
#                         ADE20K (Oracle dropped to 0.5455 by ep10). We
#                         keep only 1–8.
DEFAULT_RUN_EPOCH_CAPS = {
    "run_20260418_205734": 8,
}


def load_oracle_results(test_results_dir: Path) -> list:
    """Scan test_results/ for *_bridge_oracle.json files (teacher-forced
    Oracle mIoU per checkpoint). Returns sorted list of dicts with epoch +
    mean_iou + stage so they can be overlaid with colour by stage."""
    out = []
    if not test_results_dir.exists():
        return out
    for f in sorted(test_results_dir.glob("*_bridge_oracle.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        ep = d.get("displayed_epoch")
        if ep is None:
            ep = (d.get("epoch_index", 0) or 0) + 1
        miou = d.get("mean_iou")
        if miou is None:
            continue
        ep = int(ep)
        stage = 1 if ep <= STAGE_1_LAST_EPOCH else 2
        out.append({
            "epoch": ep, "mean_iou": float(miou),
            "median_iou": d.get("median_iou"),
            "n_samples": d.get("n_samples"),
            "source": f.name, "stage": stage,
        })
    out.sort(key=lambda r: r["epoch"])
    return out


def load_e2e_results(test_results_dir: Path) -> list:
    """Scan for end-to-end (live-inference, exactly-2) results from
    ablation_exact_k2_rwtd.py. Expected filenames:
        <stem>_e2e_exactly_2.json    (preferred — single-file summary)
      OR  <stem>_live_ablation/summary.json
    We always parse the condition_B_exactly_2 (honest) branch as the E2E
    headline number, since "1 to 6" is Hungarian-inflated.

    Returns sorted list of dicts: {epoch, mean_iou, stage, source}.
    """
    out = []
    if not test_results_dir.exists():
        return out

    # ---- Format A: flat JSONs with ablation summary shape --------------- #
    for f in sorted(test_results_dir.glob("*_e2e*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        # Accept either the ablation summary format or a flat {mean_iou,...}.
        if "condition_B_exactly_2" in d:
            body = d["condition_B_exactly_2"]
            miou = body.get("mean_iou")
        else:
            miou = d.get("mean_iou")
        ep = d.get("displayed_epoch") or d.get("epoch")
        if ep is None or miou is None:
            continue
        ep = int(ep); stage = 1 if ep <= STAGE_1_LAST_EPOCH else 2
        out.append({"epoch": ep, "mean_iou": float(miou),
                    "source": f.name, "stage": stage})

    # ---- Format B: nested ablation directories (*_live_ablation/summary) #
    for f in sorted(test_results_dir.glob("*_live_ablation/summary.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        body = d.get("condition_B_exactly_2", {})
        miou = body.get("mean_iou")
        if miou is None:
            continue
        # Infer epoch from parent dir name "<stem>_live_ablation"
        parent = f.parent.name
        ep = None
        for tok in parent.replace("_live_ablation", "").split("_"):
            if tok.startswith("epoch") or tok.startswith("ep"):
                try:
                    ep = int("".join(c for c in tok if c.isdigit()))
                    break
                except ValueError:
                    continue
        if ep is None:
            continue
        stage = 1 if ep <= STAGE_1_LAST_EPOCH else 2
        out.append({"epoch": ep, "mean_iou": float(miou),
                    "source": f"{parent}/summary.json", "stage": stage})

    out.sort(key=lambda r: r["epoch"])
    return out


def plot_val_miou(epochs, tests, oracle, plots_dir: Path):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [e["epoch"] for e in epochs]
    ys = [e["val_miou"] for e in epochs]
    ax.plot(xs, ys, "-o", color="red", label="Val mIoU (ADE20K)", linewidth=2)

    # Mark stage transitions based on phase field if present
    phases = [e.get("phase", 1) for e in epochs]
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(xs[i] - 0.5, color="gray", linestyle="--", alpha=0.5,
                       label=f"Stage {phases[i]} start" if i == 1 or
                             phases[i] != phases[max(0, i - 2)] else None)

    # Overlay live-inference test (RWTD) points if available
    if tests:
        t_xs = [t["epoch"] for t in tests]
        t_ys = [t.get("test_miou", 0) for t in tests]
        ax.plot(t_xs, t_ys, "s", color="blue", markersize=8,
                label="Test mIoU (RWTD, live)", markeredgecolor="black")

    # Overlay Oracle (teacher-forced) RWTD points, split by stage
    if oracle:
        s1 = [o for o in oracle if o["stage"] == 1]
        s2 = [o for o in oracle if o["stage"] == 2]
        if s1:
            ax.plot([o["epoch"] for o in s1], [o["mean_iou"] for o in s1],
                    "-^", color="purple", markersize=11, linewidth=1.4,
                    markeredgecolor="black",
                    label="Oracle RWTD — Stage 1 (Bridge + frozen Qwen)")
        if s2:
            ax.plot([o["epoch"] for o in s2], [o["mean_iou"] for o in s2],
                    "-D", color="#2ca02c", markersize=11, linewidth=1.4,
                    markeredgecolor="black",
                    label="Oracle RWTD — Stage 2 (Shifted-Zero + Qwen LoRA)")

    best = max(epochs, key=lambda e: e["val_miou"])
    ax.annotate(f"Best val: {best['val_miou']:.4f} (ep {best['epoch']})",
                xy=(best["epoch"], best["val_miou"]),
                xytext=(best["epoch"] + 0.3, best["val_miou"] + 0.01),
                fontsize=10, color="red")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("mIoU", fontsize=11)
    ax.set_title("Segmentation Quality (Val + Test) — Unified across runs",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "val_miou.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss_components(epochs, plots_dir: Path):
    if not epochs:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    xs = [e["epoch"] for e in epochs]

    ce = [e.get("train_mask_ce", 0) for e in epochs]
    dice = [e.get("train_mask_dice", 0) for e in epochs]
    axes[0].plot(xs, ce, "-o", color="red", label="CE")
    axes[0].plot(xs, dice, "-o", color="blue", label="Dice")
    axes[0].set_title("Mask Losses (CE + Dice)", fontsize=11)
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    lm = [e.get("train_lm_loss", 0) for e in epochs]
    axes[1].plot(xs, lm, "-s", color="orange", label="LM loss ([SEG]-weighted)")
    axes[1].set_title("LM Loss (Exponential + Shifted-Zero)", fontsize=11)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LM Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Training Loss Components — Unified across runs", fontsize=12)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_components.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_train_total(epochs, plots_dir: Path):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = [e["epoch"] for e in epochs]
    ys = [e.get("train_total", 0) for e in epochs]
    ax.plot(xs, ys, "-o", color="blue", label="Train Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss — Unified across runs", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_train.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_lr_schedule(epochs, plots_dir: Path):
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = [e["epoch"] for e in epochs]
    ys = [e.get("lr", 0) for e in epochs]
    ax.plot(xs, ys, "-o", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule — Unified across runs", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(plots_dir / "lr_schedule.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_test_metrics(tests, oracle, e2e, plots_dir: Path):
    """RWTD metrics panel:
      - Oracle (teacher-forced) — Stage 1 purple triangles, Stage 2 green diamonds
      - E2E live-inference exactly-2 — Stage 1 lightblue squares, Stage 2 orange stars
      - Training-loop live "1 to 6" — blue dots (Hungarian-inflated, flagged in legend)
    """
    if not tests and not oracle and not e2e:
        return
    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Oracle ---- #
    if oracle:
        s1 = [o for o in oracle if o["stage"] == 1]
        s2 = [o for o in oracle if o["stage"] == 2]
        if s1:
            ax.plot([o["epoch"] for o in s1], [o["mean_iou"] for o in s1],
                    "-^", color="purple", markersize=12, linewidth=1.5,
                    markeredgecolor="black",
                    label="Oracle S1 (Bridge + frozen Qwen, teacher-forced)")
            for o in s1:
                ax.annotate(f"{o['mean_iou']:.3f}",
                            xy=(o["epoch"], o["mean_iou"]),
                            xytext=(6, 10), textcoords="offset points",
                            fontsize=9, color="purple")
        if s2:
            ax.plot([o["epoch"] for o in s2], [o["mean_iou"] for o in s2],
                    "-D", color="#2ca02c", markersize=12, linewidth=1.5,
                    markeredgecolor="black",
                    label="Oracle S2 (+Qwen LoRA + SEG row, teacher-forced)")
            for o in s2:
                ax.annotate(f"{o['mean_iou']:.3f}",
                            xy=(o["epoch"], o["mean_iou"]),
                            xytext=(6, 10), textcoords="offset points",
                            fontsize=9, color="#2ca02c")

    # ---- E2E (live exactly-2) ---- #
    if e2e:
        s1 = [o for o in e2e if o["stage"] == 1]
        s2 = [o for o in e2e if o["stage"] == 2]
        if s1:
            ax.plot([o["epoch"] for o in s1], [o["mean_iou"] for o in s1],
                    "-s", color="#1f77b4", markersize=11, linewidth=1.2,
                    markeredgecolor="black",
                    label="E2E live exactly-2, S1 (honest; 0.0 until SEG learned)")
            for o in s1:
                ax.annotate(f"{o['mean_iou']:.3f}",
                            xy=(o["epoch"], o["mean_iou"]),
                            xytext=(6, -14), textcoords="offset points",
                            fontsize=9, color="#1f77b4")
        if s2:
            ax.plot([o["epoch"] for o in s2], [o["mean_iou"] for o in s2],
                    "-*", color="#ff7f0e", markersize=16, linewidth=1.4,
                    markeredgecolor="black",
                    label="E2E live exactly-2, S2 (honest end-to-end)")
            for o in s2:
                ax.annotate(f"{o['mean_iou']:.3f}",
                            xy=(o["epoch"], o["mean_iou"]),
                            xytext=(6, -14), textcoords="offset points",
                            fontsize=9, color="#ff7f0e")

    # ---- Training-loop test eval (live "1 to 6" — inflated) ---- #
    if tests:
        xs = [t["epoch"] for t in tests]
        mious = [t.get("test_miou", 0) for t in tests]
        ax.plot(xs, mious, "o", color="gray", markersize=8, alpha=0.6,
                label="Training-loop live \"1 to 6\" (Hungarian-inflated)")

    # Stage separator
    ax.axvline(STAGE_1_LAST_EPOCH + 0.5, color="gray",
               linestyle="--", alpha=0.5)
    ax.text(STAGE_1_LAST_EPOCH + 0.6, 0.02,
            "Stage 1 | Stage 2", fontsize=9, color="gray")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("RWTD mIoU", fontsize=11)
    ax.set_title("RWTD Evaluation — Oracle vs End-to-End Live Inference",
                 fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "test_metrics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir",
                   default="/home/aviad/Detecture/Qwen2SAM_Detecture/checkpoints/logs")
    p.add_argument("--plots-dir",
                   default="/home/aviad/Detecture/Qwen2SAM_Detecture/checkpoints/plots")
    p.add_argument("--test-results-dir",
                   default="/home/aviad/Detecture/Qwen2SAM_Detecture/checkpoints/test_results")
    p.add_argument("--skip-run", action="append", default=[],
                   help="Run stem to skip IN ADDITION to DEFAULT_SKIP_RUNS. "
                        "Can repeat.")
    p.add_argument("--include-run", action="append", default=[],
                   help="Run stem to force-include (removes from default "
                        "skip list). Can repeat.")
    p.add_argument("--no-default-skips", action="store_true",
                   help="Disable default skip list entirely (includes all runs).")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    plots_dir = Path(args.plots_dir)
    test_results_dir = Path(args.test_results_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    skip = set() if args.no_default_skips else set(DEFAULT_SKIP_RUNS)
    skip.update(args.skip_run)
    skip.difference_update(args.include_run)
    if skip:
        print(f"Runs being skipped: {sorted(skip)}")

    data = load_all_runs(log_dir, skip, run_epoch_caps=DEFAULT_RUN_EPOCH_CAPS)
    oracle = load_oracle_results(test_results_dir)
    e2e = load_e2e_results(test_results_dir)

    print(f"Loaded {len(data['epochs'])} epoch records "
          f"and {len(data['tests'])} test records from "
          f"{len(data['runs'])} run file(s).")
    if data["epochs"]:
        origins = {}
        for e in data["epochs"]:
            origins.setdefault(e["_run"], []).append(e["epoch"])
        for run, epochs in origins.items():
            print(f"    {run}: epochs {epochs}")

    if oracle:
        print(f"Loaded {len(oracle)} Oracle result(s):")
        for o in oracle:
            tag = "S1" if o["stage"] == 1 else "S2"
            print(f"    [{tag}] ep{o['epoch']:2d}  Oracle_mIoU={o['mean_iou']:.4f}  ({o['source']})")
    else:
        print(f"No Oracle results found in {test_results_dir}.")

    if e2e:
        print(f"Loaded {len(e2e)} E2E (live exactly-2) result(s):")
        for o in e2e:
            tag = "S1" if o["stage"] == 1 else "S2"
            print(f"    [{tag}] ep{o['epoch']:2d}  E2E_mIoU={o['mean_iou']:.4f}  ({o['source']})")
    else:
        print(f"No E2E results found (expected *_e2e*.json "
              f"or *_live_ablation/summary.json in {test_results_dir}).")

    plot_val_miou(data["epochs"], data["tests"], oracle, plots_dir)
    plot_loss_components(data["epochs"], plots_dir)
    plot_train_total(data["epochs"], plots_dir)
    plot_lr_schedule(data["epochs"], plots_dir)
    plot_test_metrics(data["tests"], oracle, e2e, plots_dir)

    print(f"Plots written to {plots_dir}")


if __name__ == "__main__":
    main()

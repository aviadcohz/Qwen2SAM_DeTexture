#!/usr/bin/env python3
"""
Full training pipeline: generate GT embeddings, then train.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python run_full_pipeline.py --config configs/detexture.yaml
"""

import subprocess
import sys
from pathlib import Path

CONFIG = "configs/detexture.yaml"


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n", flush=True)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFAILED: {desc} (exit code {result.returncode})")
        sys.exit(1)
    print(f"\nDONE: {desc}\n")


def main():
    config = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else CONFIG

    # Step 1: Generate Qwen GT embeddings
    run(
        [sys.executable, "scripts/prepare_qwen_gt_embeds.py", "--config", config],
        "Step 1/2: Generating Qwen GT embeddings (offline teacher)",
    )

    # Step 2: Train
    run(
        [sys.executable, "-m", "training.train", "--config", config],
        "Step 2/2: Training with self-distillation curriculum",
    )

    print("=" * 60)
    print("  FULL PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

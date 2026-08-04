#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Install a trained v2 checkpoint into the edge ModelRegistry layout.
#
# Usage:
#   python3 scripts/install_edge_checkpoint.py \
#     --checkpoint models/lydlr_compressor_v2_full_latest.pth \
#     --version 2_full \
#     --node-id node_0
"""
Copy / symlink a RD-trained checkpoint into models/<node_id>/ so
edge_compressor_node.ModelRegistry can hot-load it as lydlr_compressor_v{version}.pth.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Install checkpoint for edge ModelRegistry")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--version", default="2_full", help="Registry version string (no leading v)")
    p.add_argument("--node-id", default="node_0")
    p.add_argument("--models-root", type=Path, default=Path("models"))
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy")
    p.add_argument("--smoke", action="store_true", help="Load via ModelRegistry after install")
    args = p.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    dest_dir = args.models_root / args.node_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"lydlr_compressor_v{args.version}.pth"
    meta = dest_dir / f"metadata_lydlr_compressor_v{args.version}.json"

    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if args.symlink:
        dest.symlink_to(args.checkpoint.resolve())
    else:
        shutil.copy2(args.checkpoint, dest)

    meta.write_text(
        json.dumps(
            {
                "version": args.version,
                "source": str(args.checkpoint.resolve()),
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "notes": "RD v2 weights for edge hot-swap / Jetson Orin demo",
            },
            indent=2,
        )
    )
    print(f"installed {dest}")
    print(f"metadata {meta}")

    if args.smoke:
        import sys

        import torch

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "ros2" / "src" / "lydlr_ai"))
        from lydlr_ai.model.compressor import EnhancedMultimodalCompressor

        # Avoid importing edge_compressor_node (needs rclpy). Mirror ModelRegistry load.
        device = torch.device("cpu")
        model = EnhancedMultimodalCompressor().to(device)
        ckpt = torch.load(dest, map_location=device)
        state = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        print(
            f"smoke OK version={args.version} params={n} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )


if __name__ == "__main__":
    main()

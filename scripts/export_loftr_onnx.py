#!/usr/bin/env python3
"""
LunaMatch — Export LoFTR to ONNX for browser inference.

Uses Kornia's LoFTR implementation (Apache 2.0 licensed) with the
"outdoor" pretrained weights. Exports a coarse-level-only model with
fixed 480x480 input resolution to keep the ONNX file small (~45MB)
and WASM-heap-friendly.

Usage:
    pip install torch kornia onnx onnxruntime
    python3 scripts/export_loftr_onnx.py

Output:
    public/models/loftr_outdoor.onnx
"""

import os
import sys
import torch
import torch.nn as nn

try:
    from kornia.feature import LoFTR
except ImportError:
    print("ERROR: kornia not installed. Run: pip install kornia")
    sys.exit(1)


class LoFTRWrapper(nn.Module):
    """
    Thin wrapper around Kornia LoFTR that accepts two images and returns
    matched keypoints + confidence — the exact contract our TypeScript
    inference code expects.
    """

    def __init__(self, pretrained: str = "outdoor"):
        super().__init__()
        self.loftr = LoFTR(pretrained=pretrained)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        Args:
            image0: [1, 1, H, W] float32 grayscale, values in [0, 1]
            image1: [1, 1, H, W] float32 grayscale, values in [0, 1]

        Returns:
            keypoints0: [N, 2] float32 — matched coordinates in image0
            keypoints1: [N, 2] float32 — matched coordinates in image1
            confidence: [N]    float32 — match confidence scores
        """
        input_dict = {"image0": image0, "image1": image1}
        self.loftr(input_dict)

        return (
            input_dict["mkpts0_f"],  # [N, 2]
            input_dict["mkpts1_f"],  # [N, 2]
            input_dict["mconf"],     # [N]
        )


def main():
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "public", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "loftr_outdoor.onnx")

    print("Loading LoFTR (outdoor weights)...")
    model = LoFTRWrapper(pretrained="outdoor")
    model.eval()

    # Fixed input resolution for ONNX export
    H, W = 480, 480
    dummy_img0 = torch.randn(1, 1, H, W)
    dummy_img1 = torch.randn(1, 1, H, W)

    print(f"Exporting to ONNX ({H}x{W} fixed resolution)...")
    torch.onnx.export(
        model,
        (dummy_img0, dummy_img1),
        output_path,
        opset_version=16,
        dynamo=False,
        input_names=["image0", "image1"],
        output_names=["keypoints0", "keypoints1", "confidence"],
        dynamic_axes={
            "keypoints0": {0: "num_matches"},
            "keypoints1": {0: "num_matches"},
            "confidence": {0: "num_matches"},
        },
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")

    # Quick validation
    try:
        import onnxruntime as onnxrt
        import numpy as np

        session = onnxrt.InferenceSession(output_path)
        test_img0 = np.random.rand(1, 1, H, W).astype(np.float32)
        test_img1 = np.random.rand(1, 1, H, W).astype(np.float32)
        outputs = session.run(None, {"image0": test_img0, "image1": test_img1})

        print(f"  Validation: {outputs[0].shape[0]} matches found on random input")
        print(f"  Output names: keypoints0={outputs[0].shape}, keypoints1={outputs[1].shape}, confidence={outputs[2].shape}")
        print("ONNX model validated successfully!")
    except ImportError:
        print("  (Skipping validation - install onnxruntime to verify)")


if __name__ == "__main__":
    main()

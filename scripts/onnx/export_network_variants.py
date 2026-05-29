#!/usr/bin/env python3
"""
export_network_variants.py
Generates a grid of ProfileMRPNN proxy networks with varying depth/width,
exports each in FP32/FP16/FP8-sim/FP4-sim/ONNX, and writes a master JSON manifest.
"""

import os
import sys

# Fix Windows GBK console encoding issue with Unicode characters (e.g. emoji in torch.onnx logs)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
import math
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

# Allow importing from scripts/ regardless of CWD
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from quantize_performance_analyzer import (
        apply_fp8_to_model,
        apply_fp4_to_model,
        deterministic_init,
    )
except Exception:
    # Inline fallback so the script is self-contained
    def deterministic_init(m):
        import torch
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def apply_fp8_to_model(model):
        import torch
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                with torch.no_grad():
                    w = m.weight.data
                    # Simulate E4M3FN: max ~448, 4 exp + 3 mantissa
                    scale = w.abs().max().clamp_min(1e-12) / 448.0
                    w_quant = (w / scale).round().clamp(-448, 448) * scale
                    m.weight.data.copy_(w_quant)
        return model

    def apply_fp4_to_model(model):
        import torch
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                with torch.no_grad():
                    w = m.weight.data
                    # Simulate E2M1: max ~6.0, 2 exp + 1 mantissa
                    scale = w.abs().max().clamp_min(1e-12) / 6.0
                    w_quant = (w / scale).round().clamp(-6, 6) * scale
                    m.weight.data.copy_(w_quant)
        return model

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Proxy network
# ---------------------------------------------------------------------------

class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w


class ResBlock(nn.Module):
    def __init__(self, dim: int, use_se: bool = True, se_dim: int = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.use_se = use_se
        if use_se:
            self.se = SELayer(dim, reduction=max(1, dim // (se_dim or max(1, dim // 4))))
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.se(out)
        return x + out


class ProfileMRPNN(nn.Module):
    """
    Simplified proxy preserving operator mix (Linear, ReLU, SE gating, residuals)
    for scale profiling. Input signature matches production MRPNN expectations.
    """

    def __init__(
        self,
        main_depth: int = 3,
        dir_depth: int = 2,
        fusion_depth: int = 2,
        hidden_dim: int = 64,
        se_dim: int = 16,
    ):
        super().__init__()
        self.main_depth = main_depth
        self.dir_depth = dir_depth
        self.fusion_depth = fusion_depth
        self.hidden_dim = hidden_dim
        self.se_dim = se_dim

        # Monkey-patch so we can rebuild cleanly later
        self.init_kwargs = {
            "main_depth": main_depth,
            "dir_depth": dir_depth,
            "fusion_depth": fusion_depth,
            "hidden_dim": hidden_dim,
            "se_dim": se_dim,
        }

        # Input projection (production has slice-specific concatenation; proxy uses flat dims)
        self.main_in = nn.Linear(16, hidden_dim)
        self.dir_in = nn.Linear(8, hidden_dim)

        # Main branch: SE-gated residual stack
        self.main_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, use_se=True, se_dim=se_dim) for _ in range(main_depth)]
        )

        # Direction branch: residual stack (no SE for variety)
        self.dir_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, use_se=False) for _ in range(dir_depth)]
        )

        # Fusion tail
        fusion_layers = []
        in_dim = hidden_dim * 2
        for i in range(fusion_depth):
            out_dim = hidden_dim if i < fusion_depth - 1 else 1
            fusion_layers.append(nn.Linear(in_dim, out_dim))
            if i < fusion_depth - 1:
                fusion_layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.fusion = nn.Sequential(*fusion_layers)

    def forward(self, main_feat: torch.Tensor, dir_feat: torch.Tensor) -> torch.Tensor:
        m = self.main_in(main_feat)
        m = self.main_blocks(m)
        d = self.dir_in(dir_feat)
        d = self.dir_blocks(d)
        fused = torch.cat([m, d], dim=-1)
        out = self.fusion(fused)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def theoretical_sizes_mb(model: nn.Module) -> Dict[str, float]:
    """Conservative sizes: weights compress, biases stay FP32."""
    total = param_count(model)
    weight_elems = sum(p.numel() for n, p in model.named_parameters() if "weight" in n)
    bias_elems = total - weight_elems
    return {
        "fp32": (total * 4) / (1024 * 1024),
        "fp16": (weight_elems * 2 + bias_elems * 4) / (1024 * 1024),
        "fp8": (weight_elems * 1 + bias_elems * 4) / (1024 * 1024),
        "fp4": (weight_elems * 0.5 + bias_elems * 4) / (1024 * 1024),
    }


def export_variant(
    variant_dir: Path,
    kwargs: dict,
    device: torch.device = torch.device("cpu"),
):
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Fresh model with deterministic init
    model = ProfileMRPNN(**kwargs).to(device)
    model.apply(deterministic_init)
    model.eval()

    # Dummy inputs for tracing
    dummy_main = torch.randn(2, 16, device=device)
    dummy_dir = torch.randn(2, 8, device=device)

    # FP32
    fp32_path = variant_dir / "model_fp32.pth"
    torch.save(model.state_dict(), fp32_path)

    # ONNX
    onnx_path = variant_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_main, dummy_dir),
        str(onnx_path),
        input_names=["main_feat", "dir_feat"],
        output_names=["output"],
        dynamic_shapes={
            "main_feat": {0: "batch"},
            "dir_feat": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=21,
        do_constant_folding=True,
    )

    # FP16
    model_fp16 = ProfileMRPNN(**kwargs).to(device).half()
    # Copy fp32 weights then convert
    model_fp16.load_state_dict(model.state_dict())
    model_fp16.eval()
    fp16_path = variant_dir / "model_fp16.pth"
    torch.save(model_fp16.state_dict(), fp16_path)

    # FP8-sim
    model_fp8 = ProfileMRPNN(**kwargs).to(device)
    model_fp8.load_state_dict(model.state_dict())
    model_fp8 = apply_fp8_to_model(model_fp8)
    model_fp8.eval()
    fp8_path = variant_dir / "model_fp8_sim.pth"
    torch.save(model_fp8.state_dict(), fp8_path)

    # FP4-sim
    model_fp4 = ProfileMRPNN(**kwargs).to(device)
    model_fp4.load_state_dict(model.state_dict())
    model_fp4 = apply_fp4_to_model(model_fp4)
    model_fp4.eval()
    fp4_path = variant_dir / "model_fp4_sim.pth"
    torch.save(model_fp4.state_dict(), fp4_path)

    sizes = theoretical_sizes_mb(model)
    return {
        "param_count": param_count(model),
        "theoretical_sizes_mb": sizes,
        "exports": {
            "fp32": str(fp32_path.relative_to(variant_dir.parent.parent)),
            "fp16": str(fp16_path.relative_to(variant_dir.parent.parent)),
            "fp8_sim": str(fp8_path.relative_to(variant_dir.parent.parent)),
            "fp4_sim": str(fp4_path.relative_to(variant_dir.parent.parent)),
            "onnx": str(onnx_path.relative_to(variant_dir.parent.parent)),
        },
    }


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------

def make_variant_id(main_depth, dir_depth, fusion_depth, hidden_dim, se_dim) -> str:
    return f"md{main_depth}_dd{dir_depth}_fd{fusion_depth}_hd{hidden_dim}_se{se_dim}"


def generate_matrix(
    output_root: Path,
    main_depths: Tuple[int, ...] = (1, 2, 3, 4, 5),
    dir_depths: Tuple[int, ...] = (1, 2, 3),
    fusion_depths: Tuple[int, ...] = (1, 2, 3),
    hidden_dims: Tuple[int, ...] = (16, 32, 64, 128, 256),
    se_dims: Tuple[int, ...] = (4, 8, 16),
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    variants: List[dict] = []

    total = (
        len(main_depths)
        * len(dir_depths)
        * len(fusion_depths)
        * len(hidden_dims)
        * len(se_dims)
    )
    done = 0

    for main_depth, dir_depth, fusion_depth, hidden_dim, se_dim in itertools.product(
        main_depths, dir_depths, fusion_depths, hidden_dims, se_dims
    ):
        vid = make_variant_id(main_depth, dir_depth, fusion_depth, hidden_dim, se_dim)
        variant_dir = output_root / vid
        kwargs = {
            "main_depth": main_depth,
            "dir_depth": dir_depth,
            "fusion_depth": fusion_depth,
            "hidden_dim": hidden_dim,
            "se_dim": se_dim,
        }
        meta = export_variant(variant_dir, kwargs)
        variants.append(
            {
                "variant_id": vid,
                "type": "proxy",
                **kwargs,
                **meta,
            }
        )
        done += 1
        print(f"[{done}/{total}] Exported {vid}")

    matrix = {
        "output_root": str(output_root.resolve()),
        "variant_count": len(variants),
        "variants": variants,
    }
    config_path = output_root / "matrix_config.json"
    with open(config_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nMatrix config written to {config_path}")
    return matrix


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT = Path(__file__).parent.parent / "output" / "network_variants"
    generate_matrix(
        output_root=OUT,
        main_depths=(1, 2, 3, 4, 5),
        dir_depths=(1, 2, 3),
        fusion_depths=(1, 2, 3),
        hidden_dims=(16, 32, 64, 128, 256),
        se_dims=(4, 8, 16),
    )

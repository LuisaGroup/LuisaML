"""
End-to-end test script: create a PyTorch MLP and export it to
ONNX JSON + safetensors for the LuisaML runtime.

This script runs the full loop:
  1. Create model (from scripts/simple_mlp.py)
  2. Export to ONNX JSON + safetensors (from scripts/export_onnx.py)
"""
import sys
from pathlib import Path

# Add repo root to path so we can import scripts/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from simple_mlp import create_model, get_sample_input
from export_onnx import export_model


def main():
    output_dir = ROOT / "tests" / "onnx" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Create model ---
    print("=" * 60)
    print("Step 1: Creating PyTorch model")
    print("=" * 60)
    model = create_model(input_dim=784, hidden_dim=256, output_dim=10, num_hidden=2)
    sample_input = get_sample_input(input_dim=784, batch_size=1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")

    # Verify forward pass
    with torch.no_grad():
        out = model(sample_input)
    print(f"Forward pass OK, output shape: {out.shape}")

    # --- Step 1.5: Save reference input/output for C++ test ---
    input_ref_path = output_dir / "input_ref.bin"
    output_ref_path = output_dir / "output_ref.bin"
    input_ref_path.write_bytes(sample_input.numpy().astype("float32").tobytes())
    output_ref_path.write_bytes(out.numpy().astype("float32").tobytes())
    print(f"Reference input  : {input_ref_path} ({sample_input.numel()} floats)")
    print(f"Reference output : {output_ref_path} ({out.numel()} floats)")

    # --- Step 2: Export ---
    print("\n" + "=" * 60)
    print("Step 2: Exporting to ONNX JSON + safetensors")
    print("=" * 60)
    onnx_json_path, safetensors_path = export_model(
        model=model,
        sample_input=sample_input,
        output_dir=output_dir,
        model_name="simple_mlp",
        opset_version=14,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"ONNX JSON      : {onnx_json_path}")
    print(f"Safetensors    : {safetensors_path}")
    print(f"\nYou can now load these in LuisaML via:")
    print(f"  auto model = Model::load_from_json(<json_string>);")
    print(f"  net.set_weight_buffer(<byte_buffer_from_safetensors>);")


if __name__ == "__main__":
    import torch
    main()

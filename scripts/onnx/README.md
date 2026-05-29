# ONNX Export Scripts

This directory contains the PyTorch → ONNX export pipeline for the LuisaML ONNX importer.

## Files

- `export_onnx.py` — Full ONNX export pipeline with JSON conversion and safetensors weight extraction.
- `mrpnn.py` — MRPNN (Multi-feature Fusion Radiance Prediction Network) PyTorch model.
- `simple_mrpnn_v{1,2,3}.py` — Simplified MRPNN variants for testing.
- `export_quantized_onnx.py` — FP8 E4M3FN quantization export.
- `export_network_variants.py` — Grid profiler for generating network variant sweeps.
- `test_export_simple.py` — Simple test that exports a tiny 2-layer MLP to ONNX.

## Usage

```bash
# Export a simple MLP for testing
python scripts/onnx/test_export_simple.py
```

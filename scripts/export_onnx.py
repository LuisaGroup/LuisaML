"""
Export a PyTorch model to ONNX JSON + safetensors for LuisaML runtime.

The ONNX JSON format is used by LuisaML (not raw protobuf).
Weights are stored externally in a .safetensors file.
ONNX JSON initializers use `data_offsets: [start, end]` to reference
the weight blob instead of embedding raw_data.

NOTE: Protobuf JSON uses camelCase and integer enum values, but LuisaML's
C++ parser expects snake_case and string enum names. This script converts
the JSON to the LuisaML-compatible format.
"""
import os
import json
import base64
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.onnx
from google.protobuf.json_format import MessageToJson
from safetensors.torch import save_file


# ONNX TensorProto.DataType enum -> string name mapping
_ONNX_DTYPE_MAP = {
    0: "UNDEFINED",
    1: "FLOAT",
    2: "UINT8",
    3: "INT8",
    4: "UINT16",
    5: "INT16",
    6: "INT32",
    7: "INT64",
    8: "STRING",
    9: "BOOL",
    10: "FLOAT16",
    11: "DOUBLE",
    12: "UINT32",
    13: "UINT64",
    14: "COMPLEX64",
    15: "COMPLEX128",
    16: "BFLOAT16",
}


def _convert_onnx_json_for_luisaml(obj: Any) -> Any:
    """Recursively convert protobuf JSON to LuisaML-compatible format.

    NOTE: This is called AFTER preserving_proto_field_name=True, so most keys
    are already snake_case. We only need to convert integer dtype values to
    strings and skip unused fields.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k in ("metadata_props", "data_location"):
                # Skip fields not used by LuisaML
                continue
            # Convert integer dtype values to string names
            if k in ("data_type", "elem_type") and isinstance(v, int):
                v = _ONNX_DTYPE_MAP.get(v, str(v))
            result[k] = _convert_onnx_json_for_luisaml(v)
        return result
    elif isinstance(obj, list):
        return [_convert_onnx_json_for_luisaml(x) for x in obj]
    else:
        return obj


def export_model(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_dir: str | Path,
    model_name: str = "mlp",
    opset_version: int = 14,
) -> Tuple[Path, Path]:
    """
    Export a PyTorch model to ONNX JSON + safetensors.

    Returns:
        (onnx_json_path, safetensors_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_proto_path = output_dir / f"{model_name}.onnx"
    onnx_json_path = output_dir / f"{model_name}.onnx.json"
    safetensors_path = output_dir / f"{model_name}.safetensors"

    # --- Step 1: Export PyTorch model to ONNX protobuf ---
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        str(onnx_proto_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # --- Step 2: Convert ONNX protobuf to JSON (LuisaML format) ---
    import onnx
    model_proto = onnx.load(str(onnx_proto_path))
    json_str = MessageToJson(model_proto, preserving_proto_field_name=True)
    model_json = json.loads(json_str)
    model_json = _convert_onnx_json_for_luisaml(model_json)

    # --- Step 3: Extract weights into a raw binary blob ---
    state_dict = model.state_dict()
    # Sort keys for deterministic offset ordering
    sorted_keys = sorted(state_dict.keys())

    # Build offset table: name -> (dtype, shape, offset, length)
    offset_table: Dict[str, Dict] = {}
    offset = 0

    for key in sorted_keys:
        tensor = state_dict[key]
        offset_table[key] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "offset": offset,
            "length": tensor.numel() * tensor.element_size(),
        }
        offset += tensor.numel() * tensor.element_size()

    # Write raw binary blob (no safetensors header) so data_offsets in JSON
    # match exactly the byte offsets in the file.
    with open(safetensors_path, "wb") as f:
        for key in sorted_keys:
            tensor = state_dict[key]
            f.write(tensor.detach().cpu().numpy().tobytes())

    # --- Step 4: Patch ONNX JSON initializers to use data_offsets ---
    graph = model_json.get("graph", {})
    initializers = graph.get("initializer", [])

    for init in initializers:
        name = init.get("name", "")
        if name in offset_table:
            # Remove raw_data if present
            init.pop("rawData", None)
            init.pop("raw_data", None)
            # Add data_offsets for LuisaML runtime
            info = offset_table[name]
            init["data_offsets"] = [info["offset"], info["offset"] + info["length"]]

    # --- Step 5: Save ONNX JSON ---
    with open(onnx_json_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)

    # Cleanup protobuf file (optional)
    onnx_proto_path.unlink(missing_ok=True)

    print(f"[export_onnx] ONNX JSON -> {onnx_json_path}")
    print(f"[export_onnx] Safetensors -> {safetensors_path}")
    print(f"[export_onnx] Total weight bytes: {offset}")
    return onnx_json_path, safetensors_path


if __name__ == "__main__":
    from simple_mlp import create_model, get_sample_input

    model = create_model()
    sample_input = get_sample_input()
    export_model(model, sample_input, output_dir="output", model_name="mlp")

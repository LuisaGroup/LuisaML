"""
Export MRPNN model to ONNX and JSON format, and analyze the exported model
"""

import struct
from safetensors.numpy import save_file as save_safetensors_numpy
from genson import SchemaBuilder
from onnxsim import simplify
from google.protobuf.json_format import MessageToDict
from collections import Counter
import json
from onnx import TensorProto, numpy_helper
import numpy as np
import onnx
import torch
import os
import sys

# Fix Windows GBK console encoding issue with Unicode characters (e.g. emoji in torch.onnx logs)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def convert_elem_type_to_string(obj, elem_type_names):
    """Recursively convert all elemType values from int to string name"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (key == "elem_type" or key == "data_type") and isinstance(value, int):
                obj[key] = elem_type_names.get(value, f"UNKNOWN({value})")
            else:
                convert_elem_type_to_string(value, elem_type_names)
    elif isinstance(obj, list):
        for item in obj:
            convert_elem_type_to_string(item, elem_type_names)


def process_initializers(model_dict, safetensors_offsets=None):
    """Process initializers: add 'dynamic' field, safetensors offset, and conditionally remove 'raw_data'

    Args:
        model_dict: The ONNX model dictionary
        safetensors_offsets: Dict mapping tensor names to their offsets in safetensors file
                            Format: {name: {"data_offsets": [begin, end], "dtype": str, "shape": list}}
    """
    if safetensors_offsets is None:
        safetensors_offsets = {}

    if "graph" in model_dict and "initializer" in model_dict["graph"]:
        for init in model_dict["graph"]["initializer"]:
            name = init.get("name", "")
            if name in safetensors_offsets:
                init["data_offsets"] = safetensors_offsets[name]["data_offsets"]
                # Remove raw_data if exists
                if "raw_data" in init:
                    del init["raw_data"]


def export_to_onnx(model, dummy_inputs, onnx_path, input_names, output_names, opset_version=21):
    """Export PyTorch model to ONNX format and simplify it"""
    print(f"\nExporting model to ONNX: {onnx_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        external_data=False,
        dynamo=True
    )

    print(f"Model exported to {onnx_path}")

    # Simplify ONNX with onnxsim
    print(f"\nSimplifying ONNX model with onnxsim...")
    onnx_model = onnx.load(onnx_path)
    simplified_model, check = simplify(onnx_model)

    if check:
        print("ONNX model simplified successfully!")
        onnx.save(simplified_model, onnx_path)
        print(f"Simplified model saved to {onnx_path}")
    else:
        print("Warning: Simplified ONNX model could not be validated, using original model")
        simplified_model = onnx_model

    return simplified_model


def convert_onnx_to_json(onnx_model, json_path, safetensors_offsets=None):
    """Convert ONNX model to JSON format

    Args:
        onnx_model: The ONNX model object
        json_path: Path to save the JSON file
        safetensors_offsets: Dict mapping tensor names to their offsets in safetensors file
    """
    print(f"\nConverting ONNX to JSON: {json_path}")

    # Validate ONNX model
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Convert to JSON
    model_dict = MessageToDict(onnx_model, preserving_proto_field_name=True)

    # Get elemType name mapping from TensorProto
    elem_type_names = {v: k for k, v in TensorProto.DataType.items()}

    # Convert all elemType values to string names
    convert_elem_type_to_string(model_dict, elem_type_names)

    # Process initializers: add dynamic field, safetensors offset and conditionally remove raw_data
    process_initializers(model_dict, safetensors_offsets)

    # Save as JSON file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=2, ensure_ascii=False)

    print(f"ONNX model converted to JSON: {json_path}")

    return model_dict


def export_weights_to_safetensors(onnx_model, safetensors_path):
    """Export weights from ONNX model initializers to safetensors format and return offset information

    This function extracts weights directly from the optimized ONNX model's initializers,
    so only the ONNX model is needed (no PyTorch model dependency).

    Args:
        onnx_model: The ONNX model object (onnx.ModelProto)
        safetensors_path: Path to save the safetensors file

    Returns:
        dict: Mapping of tensor names to their metadata including offsets
              Format: {name: {"data_offsets": [begin, end], "dtype": str, "shape": list}}
    """
    print(
        f"\nExporting weights to safetensors from ONNX initializers: {safetensors_path}")

    # Extract weights from ONNX initializers (only weight/bias tensors)
    weights_dict = {}
    for initializer in onnx_model.graph.initializer:
        name = initializer.name
        # Only export actual weight/bias tensors, skip other constants
        if not name.endswith('.weight') and not name.endswith('.bias'):
            continue
        # Convert ONNX tensor to numpy array
        np_array = numpy_helper.to_array(initializer)
        weights_dict[name] = np_array

    # Save to safetensors format (using numpy variant)
    save_safetensors_numpy(weights_dict, safetensors_path)

    print(f"Exported {len(weights_dict)} tensors to {safetensors_path}")

    # Read back the safetensors file to get offset information
    safetensors_offsets = get_safetensors_offsets(safetensors_path)

    return safetensors_offsets


def get_safetensors_offsets(safetensors_path):
    """Read safetensors file header to get tensor offset information

    Args:
        safetensors_path: Path to the safetensors file

    Returns:
        dict: Mapping of tensor names to their metadata
              Format: {name: {"data_offsets": [begin, end], "dtype": str, "shape": list}}
    """
    with open(safetensors_path, 'rb') as f:
        # Read first 8 bytes to get header length (little-endian 64-bit integer)
        header_length_bytes = f.read(8)
        header_length = struct.unpack('<Q', header_length_bytes)[0]

        # Read the JSON header
        header_json = f.read(header_length)
        metadata = json.loads(header_json.decode('utf-8'))

    # Remove __metadata__ key if exists (it's not a tensor)
    metadata.pop('__metadata__', None)

    # print(f"\nSafetensors offsets loaded from {safetensors_path}")
    # for name, info in metadata.items():
    #     offsets = info.get('data_offsets', [])
    #     print(f"  {name}: offsets={offsets}")

    return metadata


# ============ Analysis Functions ============

def extract_elem_types_recursive(obj):
    """Recursively search for elemType and dataType keys in any nested structure"""
    types = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "elem_type" or key == "data_type":
                types.append(value)
            else:
                types.extend(extract_elem_types_recursive(value))
    elif isinstance(obj, list):
        for item in obj:
            types.extend(extract_elem_types_recursive(item))
    return types


def extract_dim_lengths_recursive(obj):
    """Recursively search for 'dim' keys and collect their list lengths"""
    lengths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "dim" and isinstance(value, list):
                lengths.append(len(value))
            else:
                lengths.extend(extract_dim_lengths_recursive(value))
    elif isinstance(obj, list):
        for item in obj:
            lengths.extend(extract_dim_lengths_recursive(item))
    return lengths


def extract_dim_products_recursive(obj):
    """Recursively search for 'dim' keys and collect the product of their values"""
    products = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "dim" and isinstance(value, list):
                # Calculate product of all dim values
                product = 1
                for dim_item in value:
                    # Handle object format: {"dim_value": "1"}
                    if isinstance(dim_item, dict) and "dim_value" in dim_item:
                        dim_val = dim_item["dim_value"]
                        if isinstance(dim_val, (int, float)):
                            product *= int(dim_val)
                        elif isinstance(dim_val, str) and dim_val.isdigit():
                            product *= int(dim_val)
                    # Handle direct value format
                    elif isinstance(dim_item, (int, float)):
                        product *= int(dim_item)
                products.append(product)
            else:
                products.extend(extract_dim_products_recursive(value))
    elif isinstance(obj, list):
        for item in obj:
            products.extend(extract_dim_products_recursive(item))
    return products


def analyze_onnx_json(json_path, schema_output_path=None):
    """
    Analyze ONNX JSON file and print statistics.

    Args:
        json_path: Path to the ONNX JSON file
        schema_output_path: Optional path to save JSON schema (if None, derived from json_path)

    Returns:
        dict containing analysis results
    """
    print(f"\n{'=' * 50}")
    print(f"Analyzing ONNX JSON: {json_path}")
    print(f"{'=' * 50}")

    # Load JSON file
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}

    # Get all nodes
    nodes = data.get("graph", {}).get("node", [])

    # 1. Count opType values
    op_types = [node.get("op_type", "Unknown") for node in nodes]
    op_type_counts = Counter(op_types)
    results['op_type_counts'] = op_type_counts

    print(f"\nTotal nodes: {len(nodes)}")
    print(f"Total unique opType values: {len(op_type_counts)}")
    print("\n" + "-" * 50)
    print("opType statistics (sorted by count):")
    print("-" * 50)
    for op_type, count in op_type_counts.most_common():
        print(f"{op_type:20s} : {count}")

    # 2. Count attribute type values
    attribute_types = []
    nodes_with_attributes = 0
    total_attributes = 0

    for node in nodes:
        attributes = node.get("attribute", [])
        if attributes:
            nodes_with_attributes += 1
            for attr in attributes:
                total_attributes += 1
                attr_type = attr.get("type", "Unknown")
                attribute_types.append(attr_type)

    attribute_type_counts = Counter(attribute_types)
    results['attribute_type_counts'] = attribute_type_counts

    print("\n" + "-" * 50)
    print("Attribute type statistics (sorted by count):")
    print("-" * 50)
    print(f"Nodes with attributes: {nodes_with_attributes}")
    print(f"Total attributes: {total_attributes}")
    print(f"Total unique attribute types: {len(attribute_type_counts)}")
    for attr_type, count in attribute_type_counts.most_common():
        print(f"{attr_type:20s} : {count}")

    # 3. Count elemType and dataType values
    elem_types = extract_elem_types_recursive(data)
    elem_type_counts = Counter(elem_types)
    results['elem_type_counts'] = elem_type_counts

    print("\n" + "-" * 50)
    print("elemType/dataType statistics (sorted by count):")
    print("-" * 50)
    print(f"Total elemType/dataType values found: {len(elem_types)}")
    print(f"Total unique values: {len(elem_type_counts)}")
    for elem_type, count in elem_type_counts.most_common():
        print(f"{elem_type:20s} : {count}")

    # 4. Count dim list length distribution
    dim_lengths = extract_dim_lengths_recursive(data)
    dim_length_counts = Counter(dim_lengths)
    results['dim_length_counts'] = dim_length_counts

    print("\n" + "-" * 50)
    print("dim list length distribution (sorted by count):")
    print("-" * 50)
    print(f"Total dim lists found: {len(dim_lengths)}")
    print(f"Total unique lengths: {len(dim_length_counts)}")
    for length, count in dim_length_counts.most_common():
        print(f"length {length:5d} : {count}")

    # 5. Count dim products (total elements) distribution
    dim_products = extract_dim_products_recursive(data)
    dim_product_counts = Counter(dim_products)
    results['dim_product_counts'] = dim_product_counts

    print("\n" + "-" * 50)
    print("dim product (total elements) distribution (sorted by count):")
    print("-" * 50)
    print(f"Total dim products found: {len(dim_products)}")
    print(f"Total unique products: {len(dim_product_counts)}")
    for product, count in dim_product_counts.most_common():
        print(f"elements {product:12d} : {count}")

    # 6. Generate JSON Schema using genson
    if schema_output_path is None:
        schema_output_path = json_path.replace('.json', '_schema.json')

    print("\n" + "-" * 50)
    print("Generating JSON Schema using genson...")
    print("-" * 50)

    builder = SchemaBuilder()
    builder.add_object(data)
    schema = builder.to_schema()
    results['schema'] = schema

    with open(schema_output_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"JSON Schema saved to: {schema_output_path}")

    return results


def simplify_and_export_onnx(onnx_model_path, export_dir, model_name="model"):
    """Simplify ONNX model and export it to JSON and safetensors format

    Args:
        onnx_model_path: Path to the ONNX model file
        export_dir: Directory to save exported files
        model_name: Base name for exported files (default: "model")

    Returns:
        dict: Containing paths to all generated files
    """
    print("\n" + "=" * 50)
    print("Processing ONNX Model")
    print("=" * 50)

    # Create output directory
    os.makedirs(export_dir, exist_ok=True)

    # Auto-generate file paths
    json_path = os.path.join(export_dir, f"{model_name}.json")
    safetensors_path = os.path.join(export_dir, f"{model_name}.safetensors")
    schema_path = os.path.join(export_dir, f"{model_name}_schema.json")

    # Load ONNX model
    print(f"\nLoading ONNX model from: {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)

    # Simplify ONNX model
    print(f"Simplifying ONNX model with onnxsim...")
    simplified_model, check = simplify(onnx_model)

    if check:
        print("ONNX model simplified successfully!")
        onnx.save(simplified_model, onnx_model_path)
        print(f"Simplified model saved to {onnx_model_path}")
    else:
        print("Warning: Simplified ONNX model could not be validated, using original model")
        simplified_model = onnx_model

    # Export weights to safetensors
    print(f"\nExporting weights to safetensors...")
    safetensors_offsets = export_weights_to_safetensors(
        simplified_model, safetensors_path)

    # Convert ONNX to JSON
    print(f"Converting ONNX to JSON...")
    convert_onnx_to_json(simplified_model, json_path, safetensors_offsets)

    # Analyze ONNX JSON
    print(f"Analyzing ONNX model...")
    analyze_onnx_json(json_path, schema_path)

    print("\n" + "=" * 50)
    print("Export completed! Generated files:")
    print("=" * 50)
    print(f"  - {onnx_model_path} (ONNX model)")
    print(f"  - {json_path} (JSON model)")
    print(f"  - {safetensors_path} (Model weights)")
    print(f"  - {schema_path} (JSON schema)")

    return {
        "onnx_model": onnx_model_path,
        "json": json_path,
        "safetensors": safetensors_path,
        "schema": schema_path
    }


def main():
    """Load MRPNN weights from binary file, test with fixed inputs, and export to ONNX"""
    import sys
    from pathlib import Path
    from mrpnn_pytorch_net_infer_exp import read_weights_from_pth, create_mrpnn_model, DEFAULT_PTH_PATH

    import argparse
    parser = argparse.ArgumentParser(description='Export MRPNN to ONNX')
    parser.add_argument('--load_pth', type=str, default=DEFAULT_PTH_PATH,
                       help='Path to .pth checkpoint file')
    parser.add_argument('--weights_path', type=str, default=None,
                       help='(Deprecated) Path to binary weights file')
    args = parser.parse_args()

    print(f"Loading weights from: {args.load_pth}")
    if not os.path.exists(args.load_pth):
        print(f"Error: weights file not found: {args.load_pth}")
        print("Usage: python export_onnx.py --load_pth <path_to_pth_file>")
        sys.exit(1)

    # Create MRPNN model and load weights from pth file
    model = create_mrpnn_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mrpnn_weights = read_weights_from_pth(model, args.load_pth, device)
    model.load_state_dict(mrpnn_weights)
    model.eval()
    print("Model weights loaded from pth.")

    # Create fixed test inputs with deterministic values (no randomness)
    batch_size = 1
    # 192 = 3 channels * 64 levels; fill with linearly spaced values in [0.01, 0.5]
    fixed_vals = torch.linspace(
        0.01, 0.5, 192, dtype=torch.float32).unsqueeze(0)       # [1, 192]
    dummy_X_Val = fixed_vals.clone()                                 # optical depth
    # sub-layer optical depth
    dummy_X_Val_Sub = fixed_vals.clone() * 0.8
    # Henyey-Greenstein optical depth
    dummy_X_Val_Hg = fixed_vals.clone() * 0.6
    dummy_scatterrate = torch.tensor(
        [[0.8, 0.6, 0.5]], dtype=torch.float32)             # albedo
    # scattering asymmetry
    dummy_g = torch.tensor(0.1, dtype=torch.float32)
    # angle parameter
    dummy_gamma = torch.tensor(0.04, dtype=torch.float32)

    dummy_inputs = (dummy_X_Val, dummy_X_Val_Sub, dummy_X_Val_Hg,
                    dummy_scatterrate, dummy_g, dummy_gamma)

    # Test forward pass with fixed inputs
    print("\n" + "=" * 50)
    print("Testing forward pass with fixed inputs:")
    print("=" * 50)
    print(
        f"  X_Val shape:       {dummy_X_Val.shape}, range: [{dummy_X_Val.min():.4f}, {dummy_X_Val.max():.4f}]")
    print(
        f"  X_Val_Sub shape:   {dummy_X_Val_Sub.shape}, range: [{dummy_X_Val_Sub.min():.4f}, {dummy_X_Val_Sub.max():.4f}]")
    print(
        f"  X_Val_Hg shape:    {dummy_X_Val_Hg.shape}, range: [{dummy_X_Val_Hg.min():.4f}, {dummy_X_Val_Hg.max():.4f}]")
    print(f"  scatterrate:       {dummy_scatterrate.squeeze().tolist()}")
    print(f"  g:                 {dummy_g.item()}")
    print(f"  gamma:             {dummy_gamma.item()}")

    with torch.no_grad():
        output = model(*dummy_inputs)
        print(f"\n  Output shape: {output.shape}")
        print(f"  Output value: {output.squeeze().tolist()}")

    # Input/output names for ONNX
    input_names = ['X_Val', 'X_Val_Sub',
                   'X_Val_Hg', 'scatterrate', 'g', 'gamma']
    output_names = ['output']

    # Export to ONNX
    output_dir = "output/mrpnn"
    onnx_path = os.path.join(output_dir, "mrpnn.onnx")

    export_to_onnx(
        model, dummy_inputs, onnx_path,
        input_names, output_names, opset_version=21
    )

    # Use the unified function to simplify and export
    result = simplify_and_export_onnx(onnx_path, output_dir, "mrpnn")

    print("\n" + "=" * 50)
    print("Export and analysis completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()

# ==================================================
# Testing forward pass with fixed inputs:
# ==================================================
#   X_Val shape:       torch.Size([1, 192]), range: [0.0100, 0.5000]
#   X_Val_Sub shape:   torch.Size([1, 192]), range: [0.0080, 0.4000]
#   X_Val_Hg shape:    torch.Size([1, 192]), range: [0.0060, 0.3000]
#   scatterrate:       [0.800000011920929, 0.6000000238418579, 0.5]
#   g:                 0.10000000149011612
#   gamma:             0.03999999910593033

#   Output shape: torch.Size([1, 3])
#   Output value: [0.0199737548828125, 0.00658416748046875, 0.00323486328125]

# Total nodes: 1512
# Total unique opType values: 23

# --------------------------------------------------
# opType statistics (sorted by count):
# --------------------------------------------------
# Transpose            : 402
# Slice                : 285
# ScatterND            : 204
# Gemm                 : 86
# Unsqueeze            : 77
# ReduceMean           : 72
# ScatterElements      : 72
# ReduceMax            : 72
# Relu                 : 72
# Add                  : 66
# Mul                  : 43
# Concat               : 19
# Cast                 : 14
# Sigmoid              : 13
# Gather               : 3
# Expand               : 2
# Equal                : 2
# ReduceMin            : 2
# If                   : 2
# Pow                  : 1
# Exp                  : 1
# Sub                  : 1
# Max                  : 1

# --------------------------------------------------
# Attribute type statistics (sorted by count):
# --------------------------------------------------
# Nodes with attributes: 948
# Total attributes: 1426
# Total unique attribute types: 5
# INT                  : 572
# INTS                 : 402
# STRING               : 276
# FLOAT                : 172
# GRAPH                : 4

# --------------------------------------------------
# elemType/dataType statistics (sorted by count):
# --------------------------------------------------
# Total elemType/dataType values found: 1928
# Total unique values: 4
# FLOAT16              : 1799
# INT64                : 115
# FLOAT                : 10
# BOOL                 : 4

# --------------------------------------------------
# dim list length distribution (sorted by count):
# --------------------------------------------------
# Total dim lists found: 1658
# Total unique lengths: 2
# length     2 : 1572
# length     1 : 86

# --------------------------------------------------
# dim product (total elements) distribution (sorted by count):
# --------------------------------------------------
# Total dim products found: 1658
# Total unique products: 15
# elements          160 : 402
# elements            8 : 354
# elements            1 : 302
# elements           16 : 258
# elements           32 : 195
# elements           75 : 75
# elements            3 : 36
# elements          128 : 12
# elements            6 : 6
# elements           64 : 6
# elements          192 : 6
# elements           72 : 3
# elements            2 : 1
# elements           96 : 1
# elements           24 : 1
import os, sys, json, math, struct
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import TensorProto, helper, numpy_helper
from google.protobuf.json_format import MessageToDict
from onnxsim import simplify
from safetensors.numpy import save_file as save_safetensors_numpy

sys.path.insert(0, os.path.dirname(__file__))
from export_onnx import convert_onnx_to_json


# ==========================================================================
# FP8 E4M3FN conversion
# ==========================================================================
FP8_E4M3_MAX_FINITE = 448.0

def fp8e4m3_from_float(v: float) -> int:
    if v == 0.0:
        return 0
    sign = 0
    if v < 0.0:
        sign = 1
        v = -v
    if v > FP8_E4M3_MAX_FINITE:
        return 0xFE if sign else 0x7E
    mant, exp = math.frexp(v)
    mant *= 2.0
    exp -= 1
    e = exp + 7
    if e <= 0:
        if e < -3:
            return 0x80 if sign else 0
        shift = 1 - e
        mant = mant / (1 << shift)
        e = 0
    mant_scaled = mant * 8
    mant_q = math.floor(mant_scaled)
    frac = mant_scaled - mant_q
    m = int(mant_q) & 0x07
    if frac > 0.5 or (frac == 0.5 and (m & 1)):
        m += 1
        if m > 0x07:
            m = 0
            e += 1
    if e >= 15:
        e = 15
        if m >= 0x07:
            m = 0x06
    if e == 0 and m == 0:
        return 0x80 if sign else 0
    return (sign << 7) | (e << 3) | m


def fp8e4m3_to_float(bits: int) -> float:
    sign = (bits >> 7) & 1
    e = (bits >> 3) & 0x0F
    m = bits & 0x07
    if e == 0 and m == 0:
        return -0.0 if sign else 0.0
    if e == 0:
        value = (m / 8.0) * (2.0 ** (1 - 7))
    else:
        value = (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return -value if sign else value


def quantize_fp8(arr: np.ndarray) -> np.ndarray:
    flat = arr.astype(np.float32).flatten()
    q = np.array([fp8e4m3_from_float(float(x)) for x in flat], dtype=np.uint8)
    return q.reshape(arr.shape)


def dequantize_fp8(q: np.ndarray) -> np.ndarray:
    flat = q.astype(np.int32).flatten()
    d = np.array([fp8e4m3_to_float(int(x)) for x in flat], dtype=np.float32)
    return d.reshape(q.shape)


# ==========================================================================
# FP4 E2M1 conversion
# ==========================================================================
FP4_E2M1_MAX_FINITE = 6.0

def fp4e2m1_from_float(v: float) -> int:
    if v == 0.0:
        return 0
    sign = 0
    if v < 0.0:
        sign = 1
        v = -v
    if v > FP4_E2M1_MAX_FINITE:
        return (sign << 3) | (3 << 1) | 1
    mant, exp = math.frexp(v)
    mant *= 2.0
    exp -= 1
    e = exp + 1
    if e <= 0:
        if e < -1:
            return sign << 3
        shift = 1 - e
        mant = mant / (1 << shift)
        e = 0
    mant_scaled = mant * 2
    mant_q = math.floor(mant_scaled)
    frac = mant_scaled - mant_q
    m = int(mant_q) & 0x01
    if frac > 0.5 or (frac == 0.5 and (m & 1)):
        m += 1
        if m > 0x01:
            m = 0
            e += 1
    if e > 3:
        e = 3
        m = 1
    if e == 0 and m == 0:
        return sign << 3
    return (sign << 3) | (e << 1) | m


def fp4e2m1_to_float(bits: int) -> float:
    sign = (bits >> 3) & 1
    e = (bits >> 1) & 0x03
    m = bits & 0x01
    if e == 0 and m == 0:
        return -0.0 if sign else 0.0
    if e == 0:
        value = (m / 2.0) * (2.0 ** (1 - 1))
    else:
        value = (1.0 + m / 2.0) * (2.0 ** (e - 1))
    return -value if sign else value


def quantize_fp4(arr: np.ndarray) -> np.ndarray:
    flat = arr.astype(np.float32).flatten()
    n = len(flat)
    q = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(n):
        nibble = fp4e2m1_from_float(float(flat[i]))
        if i % 2 == 0:
            q[i // 2] |= (nibble << 4)
        else:
            q[i // 2] |= (nibble & 0x0F)
    return q


def dequantize_fp4(q: np.ndarray, total_elements: int) -> np.ndarray:
    d = np.zeros(total_elements, dtype=np.float32)
    for i in range(total_elements):
        byte = int(q[i // 2])
        nibble = (byte >> 4) if (i % 2 == 0) else (byte & 0x0F)
        d[i] = fp4e2m1_to_float(nibble)
    return d


# ==========================================================================
# MLP Model
# ==========================================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, 512))
        layers.append(nn.ReLU())
        for _ in range(7):
            layers.append(nn.Linear(512, 512))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(512, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================================================
# ONNX helpers
# ==========================================================================
def sanitize_onnx_names(onnx_model):
    graph = onnx_model.graph
    name_map = {}
    def sanitize(name):
        return name.replace('.', '_').replace('/', '_')
    for init in graph.initializer:
        if sanitize(init.name) != init.name:
            name_map[init.name] = sanitize(init.name)
    for inp in graph.input:
        if sanitize(inp.name) != inp.name:
            name_map[inp.name] = sanitize(inp.name)
    for out in graph.output:
        if sanitize(out.name) != out.name:
            name_map[out.name] = sanitize(out.name)
    for vi in graph.value_info:
        if sanitize(vi.name) != vi.name:
            name_map[vi.name] = sanitize(vi.name)
    for node in graph.node:
        for n in node.input:
            if sanitize(n) != n and n not in name_map:
                name_map[n] = sanitize(n)
        for n in node.output:
            if sanitize(n) != n and n not in name_map:
                name_map[n] = sanitize(n)
    if not name_map:
        return onnx_model
    new_initializers = []
    for init in graph.initializer:
        new_init = TensorProto()
        new_init.CopyFrom(init)
        if init.name in name_map:
            new_init.name = name_map[init.name]
        new_initializers.append(new_init)
    new_inputs = []
    for inp in graph.input:
        new_inp = onnx.ValueInfoProto()
        new_inp.CopyFrom(inp)
        if inp.name in name_map:
            new_inp.name = name_map[inp.name]
        new_inputs.append(new_inp)
    new_outputs = []
    for out in graph.output:
        new_out = onnx.ValueInfoProto()
        new_out.CopyFrom(out)
        if out.name in name_map:
            new_out.name = name_map[out.name]
        new_outputs.append(new_out)
    new_value_info = []
    for vi in graph.value_info:
        new_vi = onnx.ValueInfoProto()
        new_vi.CopyFrom(vi)
        if vi.name in name_map:
            new_vi.name = name_map[vi.name]
        new_value_info.append(new_vi)
    new_nodes = []
    for node in graph.node:
        new_node = onnx.NodeProto()
        new_node.CopyFrom(node)
        new_node.input[:] = [name_map.get(n, n) for n in node.input]
        new_node.output[:] = [name_map.get(n, n) for n in node.output]
        new_nodes.append(new_node)
    new_graph = helper.make_graph(
        nodes=new_nodes, name=graph.name,
        inputs=new_inputs, outputs=new_outputs,
        initializer=new_initializers, value_info=new_value_info,
    )
    new_model = helper.make_model(new_graph, opset_imports=onnx_model.opset_import)
    new_model.ir_version = onnx_model.ir_version
    new_model.producer_name = onnx_model.producer_name
    new_model.producer_version = onnx_model.producer_version
    new_model.domain = onnx_model.domain
    new_model.model_version = onnx_model.model_version
    new_model.doc_string = onnx_model.doc_string
    return new_model


def export_to_onnx(model, dummy_input, onnx_path, opset_version=21):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=opset_version,
            input_names=["input"], output_names=["output"],
            dynamo=True
        )
    onnx_model = onnx.load(onnx_path)
    simplified, check = simplify(onnx_model)
    if check:
        onnx.save(simplified, onnx_path)
    onnx_model = onnx.load(onnx_path)
    return sanitize_onnx_names(onnx_model)


def get_safetensors_offsets(path):
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        metadata = json.loads(f.read(header_len).decode('utf-8'))
    metadata.pop('__metadata__', None)
    return metadata


def export_weights_to_safetensors(onnx_model, safetensors_path):
    weights = {}
    for init in onnx_model.graph.initializer:
        name = init.name
        if init.data_type == TensorProto.FLOAT8E4M3FN:
            raw = init.raw_data
            n = len(raw)
            n_padded = ((n + 3) // 4) * 4
            if n_padded > n:
                raw = raw + b'\x00' * (n_padded - n)
            weights[name] = np.frombuffer(raw, dtype=np.uint8)
        elif init.data_type == TensorProto.FLOAT4E2M1:
            raw = init.raw_data
            weights[name] = np.frombuffer(raw, dtype=np.uint8)
        else:
            weights[name] = numpy_helper.to_array(init)
    save_safetensors_numpy(weights, safetensors_path)
    return get_safetensors_offsets(safetensors_path)


def export_onnx_with_safetensors(onnx_model, out_dir, base_name):
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{base_name}.onnx")
    json_path = os.path.join(out_dir, f"{base_name}.json")
    safetensors_path = os.path.join(out_dir, f"{base_name}.safetensors")
    onnx.save(onnx_model, onnx_path)
    offsets = export_weights_to_safetensors(onnx_model, safetensors_path)
    convert_onnx_to_json(onnx_model, json_path, offsets)
    return {"onnx": onnx_path, "json": json_path, "safetensors": safetensors_path}


def convert_to_fp16_onnx(fp32_model):
    graph = fp32_model.graph
    new_initializers = []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        new_init = numpy_helper.from_array(arr, name=init.name)
        new_initializers.append(new_init)
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
    fp32_model.ir_version = 7
    return fp32_model


def _build_quantized_onnx(fp32_model, quantize_fn, target_dtype):
    graph = fp32_model.graph
    init_map = {init.name: init for init in graph.initializer}
    new_nodes = []
    new_initializers = []
    for node in graph.node:
        if node.op_type == "Gemm":
            a_name = node.input[0]
            b_name = node.input[1]
            c_name = node.input[2] if len(node.input) > 2 else ""
            out_name = node.output[0]
            b_init = init_map[b_name]
            b_arr = numpy_helper.to_array(b_init)
            q_arr = quantize_fn(b_arr)
            q_init = TensorProto()
            q_init.name = b_name + "_q"
            q_init.data_type = target_dtype
            q_init.dims.extend(list(b_arr.shape))
            q_init.raw_data = q_arr.tobytes()
            new_initializers.append(q_init)
            cast_out = b_name + "_f16"
            cast_node = helper.make_node(
                "Cast", inputs=[q_init.name], outputs=[cast_out], to=TensorProto.FLOAT16,
                name=cast_out + "_cast"
            )
            new_nodes.append(cast_node)
            if c_name:
                c_init = init_map[c_name]
                c_arr = numpy_helper.to_array(c_init)
                if c_arr.dtype == np.float32:
                    c_arr = c_arr.astype(np.float16)
                c_init_new = numpy_helper.from_array(c_arr, name=c_name)
                new_initializers.append(c_init_new)
            new_gemm = helper.make_node(
                "Gemm",
                inputs=[a_name, cast_out, c_name] if c_name else [a_name, cast_out],
                outputs=[out_name],
                name=out_name + "_gemm",
                transA=next((a.i for a in node.attribute if a.name == "transA"), 0),
                transB=next((a.i for a in node.attribute if a.name == "transB"), 0),
                alpha=next((a.f for a in node.attribute if a.name == "alpha"), 1.0),
                beta=next((a.f for a in node.attribute if a.name == "beta"), 1.0),
            )
            new_nodes.append(new_gemm)
        elif node.op_type == "Relu":
            new_nodes.append(node)
        else:
            new_nodes.append(node)
    new_inputs = list(graph.input)
    for inp in new_inputs:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
    new_outputs = list(graph.output)
    for out in new_outputs:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
    new_value_info = []
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        new_value_info.append(vi)
    # Add value_info for Cast outputs (weight_f16 tensors)
    for node in graph.node:
        if node.op_type == "Gemm":
            b_name = node.input[1]
            b_init = init_map[b_name]
            cast_out = b_name + "_f16"
            cast_vi = helper.make_tensor_value_info(
                cast_out, TensorProto.FLOAT16, list(b_init.dims))
            new_value_info.append(cast_vi)
    new_graph = helper.make_graph(
        nodes=new_nodes, name=graph.name,
        inputs=new_inputs, outputs=new_outputs,
        initializer=new_initializers, value_info=new_value_info,
    )
    new_model = helper.make_model(new_graph, opset_imports=[helper.make_opsetid("", 21)])
    new_model.ir_version = 10
    new_model.producer_name = "pytorch"
    new_model.producer_version = "2.10.0"
    new_model.domain = ""
    return new_model


def convert_to_fp8_onnx(fp32_model):
    return _build_quantized_onnx(fp32_model, quantize_fp8, TensorProto.FLOAT8E4M3FN)


def convert_to_fp4_onnx(fp32_model):
    return _build_quantized_onnx(fp32_model, quantize_fp4, TensorProto.FLOAT4E2M1)


# ==========================================================================
# Reference output computation
# ==========================================================================
def compute_fp32_reference(model, input_val=0.5):
    x = torch.tensor([[input_val]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return float(model(x).item())


def compute_fp16_reference(model, input_val=0.5):
    x = torch.tensor([[input_val]], dtype=torch.float16)
    model_h = model.half()
    model_h.eval()
    with torch.no_grad():
        return float(model_h(x).item())


def compute_quantized_reference(model, quantize_fn, dequantize_fn, packed=False, input_val=0.5):
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    model_copy.eval()
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if "weight" in name:
                arr = param.detach().cpu().numpy()
                q = quantize_fn(arr)
                if packed:
                    dq = dequantize_fp4(q, arr.size).reshape(arr.shape)
                else:
                    dq = dequantize_fn(q)
                param.copy_(torch.from_numpy(dq).to(param.dtype))
    model_h = model_copy.half()
    x = torch.tensor([[input_val]], dtype=torch.float16)
    with torch.no_grad():
        return float(model_h(x).item())


# ==========================================================================
# Main
# ==========================================================================
def main():
    out_dir = "output/mlp_8x512"
    os.makedirs(out_dir, exist_ok=True)

    model = MLP()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    dummy_input = torch.tensor([[0.5]], dtype=torch.float32)

    print("Exporting FP32...")
    fp32_model = export_to_onnx(model, dummy_input, os.path.join(out_dir, "mlp_fp32.onnx"))
    export_onnx_with_safetensors(fp32_model, out_dir, "mlp_fp32")

    print("Building FP16...")
    fp16_model = convert_to_fp16_onnx(onnx.load(os.path.join(out_dir, "mlp_fp32.onnx")))
    export_onnx_with_safetensors(fp16_model, out_dir, "mlp_fp16")

    print("Building FP8...")
    fp8_model = convert_to_fp8_onnx(onnx.load(os.path.join(out_dir, "mlp_fp32.onnx")))
    export_onnx_with_safetensors(fp8_model, out_dir, "mlp_fp8")

    print("Building FP4...")
    fp4_model = convert_to_fp4_onnx(onnx.load(os.path.join(out_dir, "mlp_fp32.onnx")))
    export_onnx_with_safetensors(fp4_model, out_dir, "mlp_fp4")

    print("Computing reference outputs...")
    refs = {
        "input": 0.5,
        "fp32": compute_fp32_reference(model),
        "fp16": compute_fp16_reference(model),
        "fp8": compute_quantized_reference(model, quantize_fp8, dequantize_fp8, packed=False),
        "fp4": compute_quantized_reference(model, quantize_fp4, dequantize_fp4, packed=True),
    }
    ref_path = os.path.join(out_dir, "ref_outputs.json")
    with open(ref_path, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"Reference outputs saved to {ref_path}")
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()

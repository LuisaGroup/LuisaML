---
name: read_onnx
description: >
  Navigate and understand the LuisaML ONNX runtime (include/onnx, src/onnx).
  Use when: (1) user asks about ONNX model loading, execution, or kernel generation
  in LuisaML, (2) user wants to add/modify an ONNX operator, (3) user asks about
  safetensors/weight-buffer integration, (4) user needs to trace how ONNX JSON
  becomes a LuisaCompute GPU kernel, (5) user references files under include/onnx
  or src/onnx.
---

# Read ONNX (LuisaML)

Guide for navigating the LuisaML ONNX embedded-DSL runtime.

## Architecture at a Glance

```
ONNX JSON  ──►  Model::load_from_json()  ──►  Graph + Nodes + Variables
                                                    │
                                            NetworkInstance
                                                    │
                              Kernel1D/Kernel2D lambda wrapping net.forward()
                                                    │
                              LuisaCompute AST  ──►  GPU Shader
```

Key design: **C++ runtime builds DSL AST**, not offline source translation.
Calls like `Var<T>`, `$if`, `dynamic_range`, `ByteBuffer::read` inside `Operator::forward()` generate AST nodes. When wrapped in a `Kernel1D` lambda, the whole graph becomes a GPU kernel.

## Directory Map

| Path | Role |
|---|---|
| `include/onnx/onnx.h` | Data layer: `Model`, `Graph`, `Node`, `Variable`, `Attribute`, `DataType` |
| `include/onnx/operator.h` | Base class `Operator`, registration macros |
| `include/onnx/tensor.h` | `ITensor`, `Tensor<T>`, `ConstTensor<T>` |
| `include/onnx/tensor_table.h` | `TensorTable` / `TensorEntry`: owning + borrowing storage |
| `include/onnx/network_instance.h` | `NetworkInstance`, `PreparedGraph`: execution engine |
| `include/onnx/operators/common.h` | Type dispatch, vectorized read/write, broadcast, shape utils |
| `include/onnx/dynamic_array/` | `DynamicArray<T>` backends: Local/Buffer/View/Scalar/Linear/FP4/FP8 |
| `include/onnx/register_allocator/` | Register allocator (graph-coloring memory reuse) |
| `src/onnx/onnx.cpp` | JSON parsing (`yyjson`) |
| `src/onnx/network_instance.cpp` | Tensor creation, lifetime analysis, operator dispatch |
| `src/onnx/operators/*.cpp` | Per-operator DSL implementations |

## Core Classes

### Model / Graph / Node / Variable (`onnx.h`)
- **Model**: top-level; entry `Model::load_from_json(json_str)`.
- **Graph**: holds `variables`, ordered `nodes`, `input`/`output` refs. Supports `set_parent` for subgraphs (e.g. `If`).
- **Node**: `op_type`, `inputs`, `outputs`, `attributes`.
- **Variable**: name, `DataType`, shape, and either:
  - `raw_data` (base64 CPU bytes), or
  - `data_offsets` → `buffer_start/buffer_end` into an external `ByteBuffer`.
- **Attribute**: `luisa::variant` of float/int/string/graph/floats/ints/strings/graphs.

### Operator (`operator.h`)
Pure virtual interface:
```cpp
virtual void forward(luisa::span<std::reference_wrapper<ITensor>> inputs,
                     luisa::span<std::reference_wrapper<ITensor>> outputs) = 0;
```
Optional hooks:
- `is_output_view()` / `can_operate_inplace()` / `need_outline()`
- `set_environment(NetworkInstance&)`

Registration macro:
```cpp
REGISTER_TO_DEFAULT_OPSET(Gemm) {
    // read node.attributes...
    return luisa::make_unique<Gemm>(...);
}
```

### Tensor (`tensor.h`)
- `ITensor`: shape, stride, element type, view/const flags.
- `Tensor<T, Container>`: default container `DynamicArray<T>`.
- Multidim `operator()` supports both host constants and DSL `Var<uint>` indices.

### DynamicArray (`dynamic_array/`)
Variant container bridging tensor to LuisaCompute storage:

| Mode | DSL behavior |
|---|---|
| `LocalData<T>` | `Local<T>` AST node |
| `BufferData<T>` | `byte_buffer->read<T>(offset)` / `write(offset, val)` |
| `ViewData<T>` | offset into another `LocalData` |
| `ScalarData<T>` | zero storage; all indices return same constant |
| `LinearData<T>` | zero storage; value = `start + idx * delta` |
| `FP4Data<T>` / `FP8Data<T>` | quantized ByteBuffer with packed offset logic |

## Execution Flow

### 1. Load
```cpp
auto model = Model::load_from_json(json_string);
```
- `yyjson` parses JSON.
- `Graph::mark_constants()` flags variables never produced by a node.

### 2. Configure NetworkInstance
```cpp
NetworkInstance net;
net.set_model(std::move(model));
net.set_input("input", input_tensor);
net.set_output("output", output_tensor);
net.set_weight_buffer(weight_byte_buffer);   // safetensors blob as ByteBuffer
net.set_warp_size(32);
```

### 3. Forward (happens at DSL capture time, not CPU runtime)
Inside `NetworkInstance::forward_graph()`:

**Phase 0** — Create operators from `OperatorSet`.
**Phase 1** — `build_last_use_map()`: record last node index each variable is used as input; extend lifetimes for subgraphs.
**Phase 2** — `create_intermediate_tensors_pooled()`: register allocator builds interference graph, colors it, assigns `PhantomStorage` slots. Intermediate tensors become `DynamicArray::ViewData` into slots.
**Phase 3** — `execute_operators()`:
```cpp
op->set_environment(*this, tensor_table);
$outline_with_name(op->get_name()) {
    op->forward(op_inputs, op_outputs);
};
```
Each `forward()` issues DSL constructs (`dynamic_range`, `Var<T>`, `$if`, etc.).

### 4. Compile & Run
```cpp
Kernel1D kernel = [&] { net.forward(); };
auto shader = device.compile(kernel);
stream << shader().dispatch(1) << synchronize();
```

## Weight Input (safetensors)

The runtime **does not parse .safetensors directly**. Instead:
1. Caller parses `.safetensors` (Python or C++), sorts by name, concatenates bytes into a blob.
2. Upload blob as LuisaCompute `Buffer<uint8_t>` / `ByteBuffer` → `Var<ByteBuffer>`.
3. `NetworkInstance::set_weight_buffer(byte_buffer_var)`.
4. ONNX JSON initializers use `data_offsets: [start, end]` instead of `raw_data`.

At tensor creation (`network_instance.cpp`):
```cpp
if (var.is_trainable_weight()) {
    auto [buf_start, buf_end] = var.get_buffer_range();
    tensor = luisa::make_unique<NNTensor<T>>(
        shape, typename NNTensor<T>::container_type{
            num_elements, weight_buffer_, buf_start});
}
```
This binds a `BufferData<T>` backed by the shared `ByteBuffer`.

For small embedded weights, `create_tensor_for_var` detects:
- **All-equal** → `ScalarData`
- **Arithmetic progression** → `LinearData`
- **General** → `LocalData` or `BufferData`

## Adding an Operator

1. Create `src/onnx/operators/<op_name>.cpp`.
2. Inherit `Operator`, implement `forward()` using DSL (`Var<T>`, `dynamic_range`, `$if`).
3. Use utilities from `include/onnx/operators/common.h` for type dispatch, vectorized loads, broadcast indexing.
4. Register:
   ```cpp
   REGISTER_TO_DEFAULT_OPSET(MyOp) {
       // parse node.attributes...
       return luisa::make_unique<MyOp>(...);
   }
   ```
5. Include new cpp in build system.

## Key Optimization Paths

When reading operator implementations, look for these DSL-level optimizations:
- **CooperativeVector**: `use_coop_vec_`, `CoopVector<T>`, `cooperative_mat_mul_add`
- **Warp vectorization**: `warp_size_`, `warp_active_sum`, `float4`/`half4` chunking
- **Normal vectorization**: `float4`/`half4` reads from `BufferData` when memory contiguous
- **Scalar fallback**: plain `fma` loops
- **Constant compression**: `ScalarData`, `LinearData`
- **Register allocation**: intermediate tensors pooled via graph coloring in `register_allocator/`

## Quick Reference: File → Question

| Question | Go to |
|---|---|
| How is ONNX JSON parsed? | `src/onnx/onnx.cpp` |
| How are tensors created and pooled? | `src/onnx/network_instance.cpp` |
| How does a specific op work? | `src/onnx/operators/<op>.cpp` |
| How do I add a new op? | `include/onnx/operator.h` + any `src/onnx/operators/*.cpp` |
| How are weights bound to ByteBuffer? | `src/onnx/network_instance.cpp` (`create_tensor_for_var`) |
| How is memory reused between intermediates? | `include/onnx/register_allocator/` + `network_instance.cpp` |
| What DSL types/backends exist? | `include/onnx/dynamic_array/` |

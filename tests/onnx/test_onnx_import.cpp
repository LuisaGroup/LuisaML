#include "onnx/onnx.h"
#include "onnx/network_instance.h"
#include <luisa/luisa-compute.h>
#include <cstdio>
#include <string>

using namespace luisa::compute;
using namespace lcml::onnx;

// Minimal ONNX JSON model: a single 4x8 Gemm with ReLU -> 8x2 Gemm
static const char *k_simple_mlp_json = R"({
  "ir_version": "8",
  "producer_name": "test",
  "producer_version": "1.0",
  "graph": {
    "name": "simple_mlp",
    "input": [
      {
        "name": "input",
        "type": {
          "tensor_type": {
            "elem_type": "FLOAT",
            "shape": {
              "dim": [
                {"dim_value": "1"},
                {"dim_value": "4"}
              ]
            }
          }
        }
      }
    ],
    "output": [
      {
        "name": "output",
        "type": {
          "tensor_type": {
            "elem_type": "FLOAT",
            "shape": {
              "dim": [
                {"dim_value": "1"},
                {"dim_value": "2"}
              ]
            }
          }
        }
      }
    ],
    "value_info": [
      {
        "name": "gemm1_out",
        "type": {
          "tensor_type": {
            "elem_type": "FLOAT",
            "shape": {
              "dim": [
                {"dim_value": "1"},
                {"dim_value": "8"}
              ]
            }
          }
        }
      },
      {
        "name": "relu1_out",
        "type": {
          "tensor_type": {
            "elem_type": "FLOAT",
            "shape": {
              "dim": [
                {"dim_value": "1"},
                {"dim_value": "8"}
              ]
            }
          }
        }
      }
    ],
    "initializer": [
      {
        "name": "W1",
        "data_type": "FLOAT",
        "dims": ["4", "8"],
        "raw_data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      },
      {
        "name": "B1",
        "data_type": "FLOAT",
        "dims": ["1", "8"],
        "raw_data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      },
      {
        "name": "W2",
        "data_type": "FLOAT",
        "dims": ["8", "2"],
        "raw_data": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
      },
      {
        "name": "B2",
        "data_type": "FLOAT",
        "dims": ["1", "2"],
        "raw_data": "AAAAAAAAAAAAAAAA"
      }
    ],
    "node": [
      {
        "name": "Gemm1",
        "op_type": "Gemm",
        "input": ["input", "W1", "B1"],
        "output": ["gemm1_out"],
        "attribute": [
          {"name": "alpha", "type": "FLOAT", "f": 1.0},
          {"name": "beta", "type": "FLOAT", "f": 1.0},
          {"name": "transA", "type": "INT", "i": "0"},
          {"name": "transB", "type": "INT", "i": "0"}
        ]
      },
      {
        "name": "Relu1",
        "op_type": "Relu",
        "input": ["gemm1_out"],
        "output": ["relu1_out"],
        "attribute": []
      },
      {
        "name": "Gemm2",
        "op_type": "Gemm",
        "input": ["relu1_out", "W2", "B2"],
        "output": ["output"],
        "attribute": [
          {"name": "alpha", "type": "FLOAT", "f": 1.0},
          {"name": "beta", "type": "FLOAT", "f": 1.0},
          {"name": "transA", "type": "INT", "i": "0"},
          {"name": "transB", "type": "INT", "i": "0"}
        ]
      }
    ]
  }
})";

int main() {
    // ===================== ONNX Import Test =====================
    std::printf("=== ONNX Import Test ===\n");
    std::fflush(stdout);
    
    std::printf("Opset has Gemm: %d\n", (int)OperatorSet::get_default().has_operator("Gemm"));
    std::fflush(stdout);
    auto model = Model::load_from_json(k_simple_mlp_json);
    std::printf("Model loaded successfully.\n");
    std::fflush(stdout);
    std::printf("  IR version: %d\n", model.get_ir_version());
    std::fflush(stdout);
    std::printf("  Producer: %s\n", model.get_producer_name().c_str());
    std::fflush(stdout);
    
    auto const &graph = model.get_graph();
    std::printf("  Graph name: %s\n", graph.get_name().c_str());
    std::fflush(stdout);
    std::printf("  Inputs: %zu\n", graph.get_inputs().size());
    std::fflush(stdout);
    std::printf("  Outputs: %zu\n", graph.get_outputs().size());
    std::fflush(stdout);
    std::printf("  Nodes: %zu\n", graph.get_nodes().size());
    std::fflush(stdout);
    
    if (graph.get_nodes().size() != 3) {
        std::printf("ERROR: Expected 3 nodes, got %zu\n", graph.get_nodes().size());
        return 1;
    }
    
    std::printf("  Node types: ");
    for (auto const &node : graph.get_nodes()) {
        std::printf("%s ", node.get_op_type().c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
    
    // ===================== Shader Compile Test =====================
    std::printf("\n=== Shader Compile Test ===\n");
    std::fflush(stdout);
    
    // Define a kernel using Luisa DSL (AST-level shader construction)
    Kernel1D kernel = [](Var<uint> idx) {
        auto x = def(1.0f);
        auto y = def(2.0f);
        auto z = x + y;
    };
    std::printf("Kernel AST defined successfully.\n");
    std::fflush(stdout);
    
    // Note: device compilation requires a GPU backend runtime.
    // In CI or headless environments this may not be available,
    // so we verify AST construction only here.
    
    std::printf("\nAll tests passed!\n");
    std::fflush(stdout);
    return 0;
}

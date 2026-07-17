// Comprehensive ONNX import + inference test for LuisaML.
//
// This test:
//  1. Loads an ONNX JSON model (produced by tests/onnx/generate_onnx.py)
//  2. Loads weights from a safetensors file
//  3. Creates a LuisaCompute kernel that runs the model via NetworkInstance
//  4. Compiles and dispatches the kernel on dx and vk backends
//  5. Compares GPU output against a CPU reference (from PyTorch)

#include "onnx/onnx.h"
#include "onnx/network_instance.h"
#include "onnx/tensor.h"
#include "onnx/dynamic_array/dynamic_array.h"

#include <luisa/luisa-compute.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace luisa::compute;
using namespace lcml::onnx;

// ============================================================
// Helpers: file I/O
// ============================================================

static std::vector<std::filesystem::path> guess_data_paths(std::string_view filename) {
    std::vector<std::filesystem::path> candidates;
    // From current working directory (project root)
    candidates.emplace_back(std::filesystem::path("tests/onnx/output") / filename);
    // From build directory (one level up)
    candidates.emplace_back(std::filesystem::path("../tests/onnx/output") / filename);
    // From build directory (two levels up)
    candidates.emplace_back(std::filesystem::path("../../tests/onnx/output") / filename);
    // From build directory (three levels up)
    candidates.emplace_back(std::filesystem::path("../../../tests/onnx/output") / filename);
    // From build directory (four levels up)
    candidates.emplace_back(std::filesystem::path("../../../../tests/onnx/output") / filename);
    // From executable directory
    {
        char exe_path[MAX_PATH];
        if (GetModuleFileNameA(nullptr, exe_path, MAX_PATH) > 0) {
            auto exe_dir = std::filesystem::path(exe_path).parent_path();
            candidates.emplace_back(exe_dir / "../tests/onnx/output" / filename);
            candidates.emplace_back(exe_dir / "../../tests/onnx/output" / filename);
            candidates.emplace_back(exe_dir / "../../../tests/onnx/output" / filename);
            candidates.emplace_back(exe_dir / "../../../../tests/onnx/output" / filename);
            candidates.emplace_back(exe_dir / "../../../../../tests/onnx/output" / filename);
        }
    }
    return candidates;
}

static std::filesystem::path find_data_file(std::string_view filename) {
    for (const auto &p : guess_data_paths(filename)) {
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    std::fprintf(stderr, "Data file not found: %s\n", std::string(filename).c_str());
    std::exit(1);
}

static std::string read_text_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Failed to open: %s\n", path.string().c_str());
        std::exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

static std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::fprintf(stderr, "Failed to open: %s\n", path.string().c_str());
        std::exit(1);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
}

/// Read the raw weight binary blob.
static std::vector<uint8_t> read_safetensors_data(const std::filesystem::path &path) {
    return read_binary_file(path);
}

static std::vector<float> read_float_bin(const std::filesystem::path &path) {
    auto bytes = read_binary_file(path);
    if (bytes.size() % sizeof(float) != 0) {
        std::fprintf(stderr, "Float bin file size not aligned\n");
        std::exit(1);
    }
    size_t n = bytes.size() / sizeof(float);
    std::vector<float> result(n);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

// ============================================================
// Test a single backend
// ============================================================

static bool test_backend(const char *backend_name,
                         const std::string &json_text,
                         const std::vector<uint8_t> &weight_data,
                         const std::vector<float> &input_ref,
                         const std::vector<float> &output_ref) {
    std::printf("\n=== Testing backend: %s ===\n", backend_name);
    std::fflush(stdout);

    // 1. Create context and device
    Context context{[]() -> const char * {
        char path[MAX_PATH];
        if (GetModuleFileNameA(nullptr, path, MAX_PATH) > 0) {
            static std::string s(path);
            return s.c_str();
        }
        return "";
    }()};

    // Check if backend is installed before creating device
    {
        auto backends = context.installed_backends();
        bool found = false;
        for (auto &b : backends) {
            if (b == backend_name) { found = true; break; }
        }
        if (!found) {
            std::printf("Backend %s not installed, skipping.\n", backend_name);
            std::fflush(stdout);
            return true;
        }
    }

    Device device = context.create_device(backend_name);
    std::printf("Device created successfully.\n");
    std::fflush(stdout);

    // 2. Load ONNX model
    auto model = Model::load_from_json(json_text);
    std::printf("Model loaded: %s (IR v%d)\n",
                model.get_graph().get_name().c_str(),
                model.get_ir_version());
    std::fflush(stdout);

    // 3. Create GPU buffers
    std::printf("Creating GPU buffers...\n");
    std::fflush(stdout);
    auto weight_buffer = device.create_byte_buffer(weight_data.size());
    auto input_buffer = device.create_byte_buffer(input_ref.size() * sizeof(float));
    auto output_buffer = device.create_byte_buffer(output_ref.size() * sizeof(float));

    Stream stream = device.create_stream();
    stream << weight_buffer.copy_from(weight_data.data())
           << input_buffer.copy_from(input_ref.data())
           << synchronize();
    std::printf("Buffers created and data uploaded.\n");
    std::fflush(stdout);

    // 4. Define kernel with NetworkInstance
    std::printf("Setting up NetworkInstance...\n");
    std::fflush(stdout);
    NetworkInstance net;
    net.set_model(std::move(model));
    net.set_warp_size(1); // Disable warp optimization for single-thread dispatch
    std::printf("NetworkInstance setup done. Compiling kernel...\n");
    std::fflush(stdout);

    auto kernel = device.compile<1>([&](ByteBufferVar weight_bb,
                                        ByteBufferVar input_bb,
                                        ByteBufferVar output_bb) {
        std::printf("  Inside kernel lambda, setting up tensors...\n");
        std::fflush(stdout);
        net.set_weight_buffer(weight_bb);

        // Input tensor backed by input ByteBuffer
        ITensor::shape_type input_shape;
        input_shape.push_back(1u);
        input_shape.push_back(784u);
        Tensor<float, DynamicArray<float>> input_tensor(
            input_shape,
            DynamicArray<float>(784u, &input_bb, 0u));
        net.set_input("input", input_tensor);

        // Output tensor as LocalData (shader-local), then copy to output ByteBuffer
        ITensor::shape_type output_shape;
        output_shape.push_back(1u);
        output_shape.push_back(10u);
        Tensor<float, DynamicArray<float>> output_tensor(
            output_shape,
            DynamicArray<float>(10u));
        net.set_output("output", output_tensor);

        // Run the network
        std::printf("  Calling net.forward()...\n");
        std::fflush(stdout);
        net.forward();
        std::printf("  net.forward() completed.\n");
        std::fflush(stdout);

        // Copy output from local memory to ByteBuffer
        for (auto i : dynamic_range(10u)) {
            auto val = output_tensor(0u, i);
            output_bb.write(i * static_cast<uint>(sizeof(float)), val);
        }
    });

    // 5. Dispatch and synchronize
    std::printf("Dispatching kernel...\n");
    std::fflush(stdout);
    stream << kernel(weight_buffer, input_buffer, output_buffer).dispatch(1)
           << synchronize();

    // 6. Download and compare
    std::vector<float> gpu_output(output_ref.size());
    stream << output_buffer.copy_to(gpu_output.data())
           << synchronize();

    bool pass = true;
    constexpr float tolerance = 1e-3f;
    for (size_t i = 0; i < output_ref.size(); ++i) {
        float diff = std::abs(gpu_output[i] - output_ref[i]);
        if (diff > tolerance) {
            pass = false;
            std::printf("MISMATCH at %zu: GPU=%.6f  Ref=%.6f  diff=%.6f\n",
                        i, gpu_output[i], output_ref[i], diff);
            std::fflush(stdout);
        }
    }

    if (pass) {
        std::printf("Backend %s: PASSED (all %zu values within tolerance %.1e)\n",
                    backend_name, output_ref.size(), tolerance);
    } else {
        std::printf("Backend %s: FAILED\n", backend_name);
    }
    std::fflush(stdout);
    return pass;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    std::printf("=== LuisaML ONNX Inference Test ===\n");
    std::fflush(stdout);

    // Load reference data
    auto json_path = find_data_file("simple_mlp.onnx.json");
    auto safetensors_path = find_data_file("simple_mlp.safetensors");
    auto input_ref_path = find_data_file("input_ref.bin");
    auto output_ref_path = find_data_file("output_ref.bin");

    std::printf("Using data files:\n");
    std::printf("  ONNX JSON : %s\n", json_path.string().c_str());
    std::printf("  Weights   : %s\n", safetensors_path.string().c_str());
    std::printf("  Input ref : %s\n", input_ref_path.string().c_str());
    std::printf("  Output ref: %s\n", output_ref_path.string().c_str());
    std::fflush(stdout);

    auto json_text = read_text_file(json_path);
    auto weight_data = read_safetensors_data(safetensors_path);
    auto input_ref = read_float_bin(input_ref_path);
    auto output_ref = read_float_bin(output_ref_path);

    std::printf("Loaded %zu input floats, %zu output floats, %zu weight bytes.\n",
                input_ref.size(), output_ref.size(), weight_data.size());
    std::fflush(stdout);

    // Test backends
    bool dx_ok = test_backend("dx", json_text, weight_data, input_ref, output_ref);
    bool vk_ok = test_backend("vk", json_text, weight_data, input_ref, output_ref);

    std::printf("\n=== Summary ===\n");
    std::printf("DX: %s\n", dx_ok ? "PASS/SKIP" : "FAIL");
    std::printf("VK: %s\n", vk_ok ? "PASS/SKIP" : "FAIL");
    std::fflush(stdout);

    if (dx_ok && vk_ok) {
        std::printf("\nAll tests passed!\n");
        return 0;
    } else {
        std::printf("\nSome tests failed.\n");
        return 1;
    }
}

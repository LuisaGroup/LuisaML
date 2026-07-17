target("test-lcml-onnx")
    set_basename("test-lcml-onnx")
    _config_project({
        project_kind = "binary"
    })
    add_deps("lcml-onnx", "lc-runtime", "lc-dsl", "lc-backends-dummy")
    add_includedirs("../../include")
    add_files("test_onnx_import.cpp")
target_end()

target("test-lcml-onnx-inference")
    set_basename("test-lcml-onnx-inference")
    _config_project({
        project_kind = "binary"
    })
    add_deps("lcml-onnx", "lc-runtime", "lc-dsl", "lc-backends-dummy")
    add_includedirs("../../include")
    add_files("test_onnx_inference.cpp")
target_end()

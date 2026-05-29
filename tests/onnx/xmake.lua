target("test-lcml-onnx")
    set_basename("test-lcml-onnx")
    _config_project({
        project_kind = "binary"
    })
    add_deps("lcml-onnx", "lc-runtime", "lc-dsl", "lc-backends-dummy")
    add_includedirs("../../include")
    add_files("**.cpp")
target_end()

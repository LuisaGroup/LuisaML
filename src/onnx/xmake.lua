target("lcml-onnx")
    set_basename("lcml-onnx")
    _config_project({
        project_kind = "shared",
        batch_size = 8
    })
    add_defines('LUISA_ONNX_EXPORT')
    add_deps("lc-runtime", "lc-dsl", "lc-yyjson")
    add_includedirs("../../include", {
        public = true
    })
    add_files("**.cpp")
target_end()

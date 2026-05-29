target("lcml-onnx")
    set_basename("lcml-onnx")
    _config_project({
        project_kind = "static",
        batch_size = 8
    })
    add_deps("lc-runtime", "lc-dsl", "lc-yyjson")
    add_includedirs("../../include", {
        public = true
    })
    add_files("**.cpp")
target_end()

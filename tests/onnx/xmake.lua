target("test-lcml-onnx")
    set_basename("test-lcml-onnx")
    _config_project({
        project_kind = "binary"
    })
    add_deps("lcml-onnx", "lc-runtime", "lc-dsl", "lc-backends-dummy")
    add_includedirs("../../include")
    add_files("**.cpp")
    if is_plat("windows") then
        add_ldflags("/WHOLEARCHIVE:lcml-onnx.lib", {force = true})
    else
        add_ldflags("-Wl,--whole-archive", "-llcml-onnx", "-Wl,--no-whole-archive", {force = true})
    end
target_end()

set_xmakever("3.0.6")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)

lc_options = {
	toolchain = "clang-cl",
	lc_enable_unity_build = false,
	lc_enable_pch = false,
	lc_safe_mode = true,
	lc_dx_cuda_interop = true,
	lc_vk_cuda_interop = true,
	lc_enable_xir = true,
	lc_enable_clangcxx = true,
	lc_vk_backend_use_xir_spirv = true,
    lc_enable_tests = false
}


if os.exists("xmake/options.lua") then
    includes("xmake/options.lua")
end

-- Include LuisaCompute build system and targets
includes("LuisaCompute")

-- Include our extension sources and tests
includes("src")
includes("tests")

set_xmakever("3.0.6")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_policy("build.ccache", not is_plat("windows"))
set_policy("check.auto_ignore_flags", false)

lc_options = {
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

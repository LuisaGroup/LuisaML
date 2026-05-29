file(GLOB_RECURSE LCML_ONNX_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_library(lcml-onnx STATIC ${LCML_ONNX_SOURCES})
set_target_properties(lcml-onnx PROPERTIES
    OUTPUT_NAME lcml-onnx
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 8
)

target_include_directories(lcml-onnx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(lcml-onnx PUBLIC
    luisa-compute-runtime
    luisa-compute-dsl
    luisa-compute-ext-yyjson-interface
)

target_compile_features(lcml-onnx PUBLIC cxx_std_20)

add_executable(test-lcml-onnx test_onnx_import.cpp)

target_include_directories(test-lcml-onnx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(test-lcml-onnx PRIVATE
    lcml-onnx
    luisa-compute-runtime
    luisa-compute-dsl
    luisa-compute-backends
)

if(WIN32)
    target_link_options(test-lcml-onnx PRIVATE "LINKER:/WHOLEARCHIVE:$<TARGET_FILE:lcml-onnx>")
elseif(APPLE)
    target_link_options(test-lcml-onnx PRIVATE "-Wl,-force_load,$<TARGET_FILE:lcml-onnx>")
else()
    target_link_options(test-lcml-onnx PRIVATE
        "LINKER:--whole-archive"
        "$<TARGET_FILE:lcml-onnx>"
        "LINKER:--no-whole-archive"
    )
endif()

set_target_properties(test-lcml-onnx PROPERTIES
    UNITY_BUILD OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #ifdef LUISA_ONNX_EXPORT
        #define LUISA_ONNX_API __declspec(dllexport)
    #else
        #define LUISA_ONNX_API __declspec(dllimport)
    #endif
#else
    #define LUISA_ONNX_API
#endif

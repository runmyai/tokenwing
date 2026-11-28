# TokenWing ABI v2 quantization plugin for WV-v6.3 (Inkling-2-Mini: EXT0..EXT3 = 112..115).
# Built into build/bin/ alongside llama-server whenever shared libraries are enabled, so both
# Linux (libwv_inkling_plugin.so) and Windows (wv_inkling_plugin.dll) release archives carry it.

set(WV_INK_PLUGIN_SRCS
    wv_ink_plugin_api.cpp
    wv_ink_cpu_kernels.cpp
)

set(WV_INK_PLUGIN_WITH_CUDA OFF)
if (GGML_CUDA AND TARGET ggml-cuda)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" _wv_ink_platform_lwr)
    if (MSVC AND _wv_ink_platform_lwr STREQUAL "arm64ec")
        find_library(CUDA_CUDART
            NAMES cudart
            HINTS ${CUDAToolkit_ROOT} ENV CUDA_PATH
            PATH_SUFFIXES lib/x64
        )
    endif()
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
        list(APPEND WV_INK_PLUGIN_SRCS wv_ink_cuda_kernels.cu)
        set(WV_INK_PLUGIN_WITH_CUDA ON)
    endif()
endif()

add_library(wv_inkling_plugin SHARED ${WV_INK_PLUGIN_SRCS})

set_target_properties(wv_inkling_plugin PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

target_include_directories(wv_inkling_plugin PRIVATE ../../include)
target_compile_definitions(wv_inkling_plugin PRIVATE WV_OBFUSCATE_SYMBOLS=1)

if (MSVC)
    target_compile_options(wv_inkling_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:precise /EHsc /GR- /DNOMINMAX /DWIN32_LEAN_AND_MEAN>)
else()
    target_compile_options(wv_inkling_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fno-rtti>)
endif()

if (WV_INK_PLUGIN_WITH_CUDA)
    get_target_property(_wv_ink_cuda_archs ggml-cuda CUDA_ARCHITECTURES)
    if (NOT _wv_ink_cuda_archs)
        set(_wv_ink_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    set_target_properties(wv_inkling_plugin PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_VISIBILITY_PRESET hidden
        CUDA_ARCHITECTURES "${_wv_ink_cuda_archs}"
    )
    target_compile_definitions(wv_inkling_plugin PRIVATE WV_INK_PLUGIN_ENABLE_CUDA=1)
    set(_wv_ink_cuda_flags -use_fast_math --expt-relaxed-constexpr)
    if (MSVC)
        list(APPEND _wv_ink_cuda_flags -Xcompiler /Zc:preprocessor)
    endif()
    target_compile_options(wv_inkling_plugin PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_wv_ink_cuda_flags}>")
    target_link_libraries(wv_inkling_plugin PRIVATE CUDA::cudart)
endif()

install(TARGETS wv_inkling_plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

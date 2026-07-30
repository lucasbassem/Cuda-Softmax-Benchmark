cmake_minimum_required(VERSION 3.24)

project(cuda_softmax_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(SOFTMAX_ENABLE_FAST_MATH "Compile CUDA kernels with --use_fast_math" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(softmax_benchmark
    src/main.cu
    src/cpu_softmax.cpp
    src/softmax_kernels.cu
)

target_include_directories(softmax_benchmark PRIVATE include)
target_link_libraries(softmax_benchmark PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_options(softmax_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
    )
else()
    target_compile_options(softmax_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

if(SOFTMAX_ENABLE_FAST_MATH)
    target_compile_options(softmax_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    target_compile_definitions(softmax_benchmark PRIVATE SOFTMAX_FAST_MATH=1)
endif()

set_target_properties(softmax_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME softmax_smoke_test
        COMMAND softmax_benchmark
            --sizes 16x31,32x257
            --warmup 2
            --iterations 5
            --output ${CMAKE_BINARY_DIR}/smoke_results.csv
    )
endif()

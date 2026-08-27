cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" CACHE PATH "Path to the CUDA Toolkit")
set(ENV{CUDA_PATH} "${CUDAToolkit_ROOT}")
set(ENV{CUDA_PATH_V13_3} "${CUDAToolkit_ROOT}")
set(CMAKE_VS_GLOBALS "CudaToolkitDir=${CUDAToolkit_ROOT}" CACHE STRING "CUDA Toolkit directory passed to Visual Studio CUDA targets")
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90" CACHE STRING "CUDA architectures compiled into the YoloTrtCuda preprocessing kernels")
project(YoloTrtCuda VERSION 1.0.0 LANGUAGES CXX CUDA)

set(TENSORRT_PATH "D:/Code/Libs/TensorRT-Shared" CACHE PATH "Path to the TensorRT installation")
set(CUDNN_PATH "C:/Program Files/NVIDIA/CUDNN/v9.24" CACHE PATH "Path to cuDNN (not copied or linked directly by YoloTrtCuda)")

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInferRuntime.h PATHS "${TENSORRT_PATH}/include" NO_DEFAULT_PATH REQUIRED)
find_library(TENSORRT_IMPORT_PATH NAMES nvinfer_11 nvinfer_10 nvinfer PATHS "${TENSORRT_PATH}/lib" NO_DEFAULT_PATH REQUIRED)

include(CMakePackageConfigHelpers)

file(GLOB YOLOTRTCUDA_SRC_FILES "YoloTrtCuda/src/*.cpp" "YoloTrtCuda/src/*.cu")
file(GLOB YOLOTRTCUDA_HEADER_FILES "YoloTrtCuda/include/*.h")

add_library(YoloTrtCuda SHARED)
add_library(YoloTrtCuda::YoloTrtCuda ALIAS YoloTrtCuda)
target_sources(YoloTrtCuda
    PRIVATE
        ${YOLOTRTCUDA_SRC_FILES}
        ${YOLOTRTCUDA_HEADER_FILES}
)
target_include_directories(YoloTrtCuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/YoloTrtCuda/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/YoloTrtCuda/src
        ${TENSORRT_INCLUDE_DIR}
)
target_compile_definitions(YoloTrtCuda
    PRIVATE
        YOLOTRTCUDA_BUILD_SHARED
    INTERFACE
        YOLOTRTCUDA_USE_SHARED
)
set_target_properties(YoloTrtCuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
)
target_link_libraries(YoloTrtCuda
    PRIVATE
        ${TENSORRT_IMPORT_PATH}
        CUDA::cudart
)

install(TARGETS YoloTrtCuda
    EXPORT YoloTrtCudaTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES YoloTrtCuda/include/YoloTrtCuda.h DESTINATION include)
install(EXPORT YoloTrtCudaTargets
    FILE YoloTrtCudaTargets.cmake
    NAMESPACE YoloTrtCuda::
    DESTINATION lib/cmake/YoloTrtCuda
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YoloTrtCudaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloTrtCudaConfig.cmake"
    INSTALL_DESTINATION lib/cmake/YoloTrtCuda
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/YoloTrtCudaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/YoloTrtCudaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloTrtCudaConfigVersion.cmake"
    DESTINATION lib/cmake/YoloTrtCuda
)

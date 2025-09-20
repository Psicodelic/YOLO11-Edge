cmake_minimum_required(VERSION 3.10)
project(YOLO11-Edge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# 命令行参数
option(ENABLE_TESTS "Build tests" OFF)

if(EXISTS "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc")
    set(CROSS_COMPILE "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-")
else()
    set(CROSS_COMPILE "aarch64-linux-gnu-")
endif()

# Set cross-compilation toolchain
set(CMAKE_C_COMPILER "${CROSS_COMPILE}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_COMPILE}g++")
set(CMAKE_AR "${CROSS_COMPILE}ar")
set(CMAKE_AS "${CROSS_COMPILE}as")
set(CMAKE_LD "${CROSS_COMPILE}ld")
set(CMAKE_STRIP "${CROSS_COMPILE}strip")
set(CMAKE_RANLIB "${CROSS_COMPILE}ranlib")
set(CMAKE_OBJCOPY "${CROSS_COMPILE}objcopy")
set(CMAKE_OBJDUMP "${CROSS_COMPILE}objdump")
set(CMAKE_READELF "${CROSS_COMPILE}readelf")
set(CMAKE_NM "${CROSS_COMPILE}nm")

# Set compiler flags
add_compile_options(-Werror) # remove Werror
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Equivalent to -fPIC
set(CMAKE_CXX_STANDARD 17)

# Define paths
set(HR_TOP_DIR "/usr/hobot")
if(NOT DEFINED HR_BUILD_OUTPUT_DIR)
    set(HR_BUILD_OUTPUT_DIR ${HR_TOP_DIR})
endif()

if(NOT DEFINED APP_DEPLOY_DIR)
    set(APP_DEPLOY_DIR "/app/")
endif()

set(PLATFORM_SAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLATFORM_SAMPLES_DEPLOY_DIR "${APP_DEPLOY_DIR}/platform_samples")

# Enable verbose output based on BUILD_VERBOSE
option(BUILD_VERBOSE "Enable verbose build output" OFF)
if(BUILD_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 添加 dnn 库
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")
include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

# 添加包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "include"
)

# 添加链接目录
link_directories(
    ${PLATFORM_SAMPLES_DIR}/lib
    /usr/hobot/lib
)

# 添加可执行文件
add_executable(yolo11-edge src/yolo11-edge.cpp
                src/preprocess.cpp 
                src/postprocess.cpp)

# 添加链接库
target_link_libraries(yolo11-edge
        ${OpenCV_LIBS}
        dnn
        gflags
        )

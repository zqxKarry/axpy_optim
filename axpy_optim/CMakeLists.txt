cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(AXPY_OPTIM)

#添加C++11标准支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# 添加头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# 添加源代码文件路径
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

# 指定生成目标 
add_executable(AXPY_OPTIM ${DIR_SRCS})
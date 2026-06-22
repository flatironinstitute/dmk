cmake_minimum_required(VERSION 3.18)
project(test_binsort CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

find_package(OpenMP REQUIRED)

set(XSIMD_INCLUDE_DIR "/mnt/home/ipopa/finufft/build/_deps/xsimd-src/include"
    CACHE PATH "xsimd include dir")

add_executable(test_binsort main.cpp)

target_include_directories(test_binsort PRIVATE
    ${XSIMD_INCLUDE_DIR}
)

target_link_libraries(test_binsort PRIVATE
    OpenMP::OpenMP_CXX
)
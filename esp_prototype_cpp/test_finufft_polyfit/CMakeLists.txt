cmake_minimum_required(VERSION 3.18)
project(test_poly CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3)

find_package(OpenMP REQUIRED)

set(DMK_INCLUDE_DIR "/mnt/home/ipopa/dmk/include" CACHE PATH "dmk include dir")
set(DMK_LIB         "/mnt/home/ipopa/dmk/build/libdmk.a" CACHE FILEPATH "dmk library")
set(FINUFFT_DIR     "/mnt/home/ipopa/finufft" CACHE PATH "finufft root dir")

set(FINUFFT_INCLUDE_DIR "${FINUFFT_DIR}/include")
set(FINUFFT_LIB         "${FINUFFT_DIR}/build/src/libfinufft.a")
set(FINUFFT_COMMON_LIB  "${FINUFFT_DIR}/build/src/common/libfinufft_common.a")

add_executable(test_poly test_poly_pswf.cpp)

target_include_directories(test_poly PRIVATE
    ${DMK_INCLUDE_DIR}
    ${FINUFFT_INCLUDE_DIR}
)

target_link_libraries(test_poly PRIVATE
    ${DMK_LIB}
    ${FINUFFT_LIB}
    ${FINUFFT_COMMON_LIB}
    fftw3
    fftw3f
    fftw3_omp
    fftw3f_omp
    OpenMP::OpenMP_CXX
    m
)
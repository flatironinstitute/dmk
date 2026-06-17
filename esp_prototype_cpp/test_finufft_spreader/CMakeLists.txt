cmake_minimum_required(VERSION 3.15)
project(test_finufft_spreader)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

set(FINUFFT_DIR "$ENV{HOME}/finufft")

add_executable(test_finufft_spreader test_finufft_spreader.cpp)

target_include_directories(test_finufft_spreader PRIVATE
    ${FINUFFT_DIR}/include
)

target_link_libraries(test_finufft_spreader PRIVATE
    ${FINUFFT_DIR}/build/src/libfinufft.a
    ${FINUFFT_DIR}/build/src/common/libfinufft_common.a
    fftw3
    fftw3f
    fftw3_omp
    fftw3f_omp
    OpenMP::OpenMP_CXX
    m
)
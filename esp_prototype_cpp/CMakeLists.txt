cmake_minimum_required(VERSION 3.18)

project(esp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3)

find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

include(FetchContent)

set(FINUFFT_USE_DUCC0 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  finufft
  GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git
  GIT_TAG v2.5.0
)

FetchContent_MakeAvailable(finufft)

set(DMK_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../include" CACHE PATH "dmk include dir")
set(DMK_LIB         "${CMAKE_SOURCE_DIR}/../build/libdmk.a" CACHE FILEPATH "dmk library")
set(NDA_C_LIB       "${CMAKE_SOURCE_DIR}/../build/extern/nda/c++/nda/libnda_c.a" CACHE FILEPATH "nda_c library")
#set(DMK_REF_LIB     "${CMAKE_SOURCE_DIR}/../build/libdmk_ref.a" CACHE FILEPATH "dmk ref library")

# --- main esp executable ---

add_executable(esp
    main.cpp
    esp.cpp
    fft_fftw.cpp
)

target_include_directories(esp PRIVATE
    ${DMK_INCLUDE_DIR}
    ${FFTW_INCLUDE_DIRS}
)

target_link_libraries(esp PRIVATE
    ${DMK_LIB}
    ${NDA_C_LIB}
    finufft::finufft
    ${FFTW_LIBRARIES}
    fftw3f
    fftw3_omp
    fftw3f_omp
    OpenMP::OpenMP_CXX
    BLAS::BLAS
    LAPACK::LAPACK
    gfortran
    MPI::MPI_CXX
    m
)
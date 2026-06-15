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
pkg_check_modules(FFTW REQUIRED fftw3)

set(DMK_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../include" CACHE PATH "dmk include dir")
set(DMK_LIB         "${CMAKE_SOURCE_DIR}/../build/libdmk.a" CACHE FILEPATH "dmk library")

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
    ${FFTW_LIBRARIES}
    OpenMP::OpenMP_CXX
    m
)
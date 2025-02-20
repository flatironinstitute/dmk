cmake_minimum_required(VERSION 3.18)

project(dmk LANGUAGES CXX Fortran)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
  add_compile_options("-ffast-math")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options("-DSCTL_HAVE_SVML")
  add_link_options("-lstdc++")
endif()

option(DMK_BUILD_TESTS "Build tests" ON)

if (NOT ${DMK_BUILD_TESTS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDOCTEST_CONFIG_DISABLE")
endif()

# FIXME: BLAS hack on SCTL to avoid header conflict. Eigen's BLAS define conflicts with SCTL's BLAS, so we don't include SCTL's BLAS header
add_compile_options("-march=native;-DSCTL_HAVE_MPI;-DSCTL_MAX_DEPTH=62;-DEIGEN_USE_BLAS;-D_SCTL_BLAS_H_;-funroll-loops")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
  ${PROJECT_SOURCE_DIR}/extern/spdlog/include
  ${PROJECT_SOURCE_DIR}/extern/doctest
)

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/nanobench EXCLUDE_FROM_ALL)
add_subdirectory(extern/doctest)

file(GLOB_RECURSE DMK_REF_COMMON_SRC "src/common/*.f*" "vec-kernels/src/libkernels2.cpp")
add_library(dmk_ref_common OBJECT ${DMK_REF_COMMON_SRC})
set_target_properties(dmk_ref_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dmk_ref_common PRIVATE "extern/VCL/version2" "vec-kernels/include")

file(GLOB_RECURSE DMK_REF_PDMK_SRC "src/pdmk/*.f")
list(REMOVE_ITEM DMK_REF_PDMK_SRC "${CMAKE_SOURCE_DIR}/src/pdmk/kernelevaluation/l3dkernels_fast.f" "${CMAKE_SOURCE_DIR}/src/pdmk/pdmk.f")
add_library(dmk_ref_pdmk OBJECT ${DMK_REF_PDMK_SRC})
set_target_properties(dmk_ref_pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE DMK_REF_BDMK_SRC "src/bdmk/*.f")
add_library(dmk_ref_bdmk OBJECT ${DMK_REF_BDMK_SRC})
set_target_properties(dmk_ref_bdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(dmk_ref STATIC $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
add_library(dmk_ref_shared SHARED $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
set_target_properties(dmk_ref_shared PROPERTIES OUTPUT_NAME dmk_ref)

set(DMK_SRC src/dmk.cpp src/proxy.cpp src/tree.cpp src/logger.cpp src/util.cpp
            src/fourier_data.cpp src/tensorprod.cpp src/planewave.cpp src/direct.cpp src/doctest.cpp)
add_library(DMKOBJS OBJECT ${DMK_SRC})
set_target_properties(DMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(DMKOBJS PUBLIC ${DMK_INCLUDES} ${MPI_CXX_INCLUDE_DIRS})

add_library(dmk STATIC)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX DMKOBJS dmk_ref_common spdlog::spdlog)

add_library(dmk_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared DMKOBJS dmk_ref_common OpenMP::OpenMP_CXX MPI::MPI_CXX BLAS::BLAS spdlog doctest)

if(${DMK_BUILD_TESTS})
  add_subdirectory(test)
endif()

cmake_minimum_required(VERSION 3.20)

project(dmk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DMK_BUILD_TESTS "Build tests" ON)
option(DMK_BUILD_EXAMPLES "Build examples" ON)
option(DMK_INSTRUMENT "Instrument the code" OFF)
option(DMK_BUILD_PVFMM "Build PVFMM for comparison. Useless without DMK_BUILD_EXAMPLES" OFF)
option(DMK_HAVE_MPI "Use MPI for parallelism" ON)
option(DMK_HAVE_OPENMP "Use OpenMP for parallelism" ON)
option(DMK_BUILD_REFERENCE "Build reference Fortran implementation" OFF)
option(DMK_USE_JIT "Use JIT compilation where supported (requires LLVM)" OFF)

if (DMK_BUILD_REFERENCE)
  enable_language(Fortran)
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(DMK_VECLIB_DEFINE "SCTL_HAVE_SVML")
else()
  set(DMK_VECLIB_DEFINE "SCTL_HAVE_LIBMVEC")
endif()

# Disable google tests and HDF5 for NDA
set(Build_Tests OFF CACHE BOOL "" FORCE)
set(HDF5Support OFF CACHE BOOL "" FORCE)

set(DMK_CXX_FLAGS "-DSCTL_MAX_DEPTH=62;-D${DMK_VECLIB_DEFINE}")
if (DMK_BUILD_TESTS)
  set(DMK_TEST_SRC src/doctest.cpp)
else()
  set(DMK_TEST_SRC)
  list(APPEND DMK_CXX_FLAGS "-DDOCTEST_CONFIG_DISABLE")
endif()

if (DMK_INSTRUMENT)
  list(APPEND DMK_CXX_FLAGS "-DSCTL_PROFILE=5" "-DDMK_INSTRUMENT")
endif()

set(DMK_OPENMP_TARGET "")
if (DMK_HAVE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(DMK_OPENMP_TARGET OpenMP::OpenMP_CXX)
  list(APPEND DMK_CXX_FLAGS "-DDMK_HAVE_OPENMP")
  set(OpenMPSupport ON CACHE BOOL "" FORCE)
else()
  set(OpenMPSupport OFF CACHE BOOL "" FORCE)
endif()

set(DMK_MPI_TARGET "")
if (DMK_HAVE_MPI)
  find_package(MPI REQUIRED)
  set(DMK_MPI_TARGET MPI::MPI_CXX)
  list(APPEND DMK_CXX_FLAGS "-DSCTL_HAVE_MPI" "-DDMK_HAVE_MPI")
  set(MPISupport ON CACHE BOOL "" FORCE)
else()
  set(MPISupport OFF CACHE BOOL "" FORCE)
endif()
find_package(BLAS REQUIRED)

if (DMK_BUILD_PVFMM)
  include(FetchContent)
  FetchContent_Declare(
    pvfmm_lib
    GIT_REPOSITORY https://github.com/dmalhotra/pvfmm.git
    GIT_TAG develop
  )
  FetchContent_MakeAvailable(pvfmm_lib)
endif()

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
)

include(FetchContent)
FetchContent_Declare(
  polyfit
  GIT_REPOSITORY https://github.com/DiamonDinoia/polyfit.git
  GIT_TAG        a2994771155ed4cd22c2b792b463c42eefca3067
)
set(MONOFIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MONOFIT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(polyfit)

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/nanobench EXCLUDE_FROM_ALL)
add_subdirectory(extern/doctest)
add_subdirectory(extern/nda)
if (DMK_USE_JIT)
  list(APPEND DMK_CXX_FLAGS "-DDMK_USE_JIT")
  set(RUFUS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(RUFUS_LINK_LLVM_SHARED ON CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  add_subdirectory(extern/RuFuS)

  rufus_embed_ir_as_header(jit_kernels src/jit_kernels.cpp
    INCLUDES ${DMK_INCLUDES}
    DEFINITIONS ${DMK_VECLIB_DEFINE}
    FROM_TARGET nda::nda_c
  )
endif()

if (DMK_BUILD_REFERENCE)
  list(APPEND DMK_CXX_FLAGS "-DDMK_HAVE_REFERENCE")
  file(GLOB_RECURSE DMK_REF_COMMON_SRC CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/common/*.f*"
    "${PROJECT_SOURCE_DIR}/vec-kernels/src/libkernels2.cpp")
  add_library(dmk_ref_common OBJECT ${DMK_REF_COMMON_SRC})
  set_target_properties(dmk_ref_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(dmk_ref_common PRIVATE
    "${PROJECT_SOURCE_DIR}/extern/VCL/version2"
    "${PROJECT_SOURCE_DIR}/vec-kernels/include")

  file(GLOB_RECURSE DMK_REF_PDMK_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/pdmk/*.f")
  list(REMOVE_ITEM DMK_REF_PDMK_SRC
    "${PROJECT_SOURCE_DIR}/src/pdmk/kernelevaluation/l3dkernels_fast.f"
    "${PROJECT_SOURCE_DIR}/src/pdmk/pdmk.f"
    "${PROJECT_SOURCE_DIR}/src/pdmk/pdmk3.f"
  )
  add_library(dmk_ref_pdmk OBJECT ${DMK_REF_PDMK_SRC})
  set_target_properties(dmk_ref_pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  file(GLOB_RECURSE DMK_REF_BDMK_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/bdmk/*.f")
  list(REMOVE_ITEM DMK_REF_BDMK_SRC
    "${PROJECT_SOURCE_DIR}/src/bdmk/bdmk4.f"
  )
  add_library(dmk_ref_bdmk OBJECT ${DMK_REF_BDMK_SRC})
  set_target_properties(dmk_ref_bdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(dmk_ref STATIC
    $<TARGET_OBJECTS:dmk_ref_common>
    $<TARGET_OBJECTS:dmk_ref_pdmk>
    $<TARGET_OBJECTS:dmk_ref_bdmk>)
endif()

set(DMK_SRC src/dmk.cpp src/proxy.cpp src/tree.cpp src/logger.cpp src/util.cpp src/legeexps.cpp
  src/fourier_data.cpp src/tensorprod.cpp src/planewave.cpp src/direct.cpp src/prolate.cpp src/prolate0_eval.cpp
  src/timing.cpp src/aot_kernels.cpp
  ${DMK_TEST_SRC}
)
add_library(DMKOBJS OBJECT ${DMK_SRC})
set_target_properties(DMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(DMKOBJS PUBLIC ${DMK_INCLUDES})
target_link_libraries(DMKOBJS PUBLIC ${DMK_OPENMP_TARGET} ${DMK_MPI_TARGET}
  nda::nda_c spdlog::spdlog_header_only doctest::doctest BLAS::BLAS polyfit::polyfit
)
if (DMK_USE_JIT)
  target_link_libraries(DMKOBJS PUBLIC RuFuS::rufus jit_kernels)
endif()
target_compile_options(DMKOBJS PUBLIC "${DMK_CXX_FLAGS}")

add_library(dmk STATIC)
target_link_libraries(dmk PUBLIC DMKOBJS)

add_library(dmk_shared SHARED)
set_target_properties(dmk_shared PROPERTIES OUTPUT_NAME dmk)
target_link_libraries(dmk_shared PUBLIC DMKOBJS)

if (DMK_BUILD_TESTS)
  add_subdirectory(test)
endif()

if (DMK_BUILD_EXAMPLES)
  add_subdirectory(examples)
  add_subdirectory(scripts)
endif()

if (DMK_USE_JIT)
  add_subdirectory(perf)
endif()

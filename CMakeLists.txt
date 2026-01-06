cmake_minimum_required(VERSION 3.18)

project(dmk LANGUAGES CXX Fortran)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_compile_options("-DSCTL_HAVE_SVML")
else()
  add_compile_options("-DSCTL_HAVE_LIBMVEC")
endif()

option(DMK_BUILD_TESTS "Build tests" ON)
option(DMK_BUILD_EXAMPLES "Build examples" ON)
option(DMK_INSTRUMENT "Instrument the code. Requires PAPI" OFF)
option(DMK_BUILD_PVFMM "Build PVFMM for comparison. Useless without DMK_BUILD_EXAMPLES" OFF)
option(DMK_HAVE_MPI "Use MPI for parallelism" ON)
option(DMK_HAVE_OPENMP "Use OpenMP for parallelism" ON)
option(DMK_BUILD_REFERENCE "Build reference Fortran implementation" OFF)

# Disable google tests and HDF5 for NDA
set(Build_Tests OFF CACHE BOOL "" FORCE)
set(HDF5Support OFF CACHE BOOL "" FORCE)

if (NOT ${DMK_BUILD_TESTS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDOCTEST_CONFIG_DISABLE")
  set(DMK_TEST_SRC)
else()
  set(DMK_TEST_SRC src/doctest.cpp)
endif()

set(DMK_CXX_FLAGS "-march=native;-DSCTL_MAX_DEPTH=62")
if(${DMK_INSTRUMENT})
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(PAPI REQUIRED IMPORTED_TARGET papi)
  set(PAPI_TARGET papi)
  add_compile_options("-DSCTL_PROFILE=5;-DDMK_INSTRUMENT")
else()
  set(PAPI_TARGET "")
endif()

set(DMK_OPENMP_TARGET "")
if (DMK_HAVE_OPENMP)
  find_package(OpenMP REQUIRED)
  set(DMK_OPENMP_TARGET OpenMP::OpenMP_CXX)
  set(DMK_CXX_FLAGS "${DMK_CXX_FLAGS};-DDMK_HAVE_OPENMP")
  option(OpenMPSupport ON)
else()
  option(OpenMPSupport OFF)
endif()

set(DMK_MPI_TARGET "")
if (DMK_HAVE_MPI)
  find_package(MPI REQUIRED)
  set(DMK_CXX_FLAGS "${DMK_CXX_FLAGS};-DSCTL_HAVE_MPI;-DDMK_HAVE_MPI")
  set(DMK_MPI_TARGET MPI::MPI_CXX)
  # Enable MPI in NDA
  option(MPISupport ON)
else()
  # Disable MPI in NDA
  option(MPISupport OFF)
endif()
find_package(BLAS REQUIRED)

if(${DMK_BUILD_PVFMM})
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

add_subdirectory(extern/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(extern/nanobench EXCLUDE_FROM_ALL)
add_subdirectory(extern/doctest)
add_subdirectory(extern/nda)

set(DMK_REFERENCE_TARGET "")
if (DMK_BUILD_REFERENCE)
  set(DMK_CXX_FLAGS "${DMK_CXX_FLAGS};-DDMK_HAVE_REFERENCE")
  set(DMK_REFERENCE_TARGET dmk_ref)
  file(GLOB_RECURSE DMK_REF_COMMON_SRC "src/common/*.f*" "vec-kernels/src/libkernels2.cpp")
  add_library(dmk_ref_common OBJECT ${DMK_REF_COMMON_SRC})
  set_target_properties(dmk_ref_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(dmk_ref_common PRIVATE "extern/VCL/version2" "vec-kernels/include")


  file(GLOB_RECURSE DMK_REF_PDMK_SRC "src/pdmk/*.f")
  list(REMOVE_ITEM DMK_REF_PDMK_SRC
    "${CMAKE_SOURCE_DIR}/src/pdmk/kernelevaluation/l3dkernels_fast.f"
    "${CMAKE_SOURCE_DIR}/src/pdmk/pdmk.f"
    "${CMAKE_SOURCE_DIR}/src/pdmk/pdmk4.f"
  )
  add_library(dmk_ref_pdmk OBJECT ${DMK_REF_PDMK_SRC})
  set_target_properties(dmk_ref_pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  file(GLOB_RECURSE DMK_REF_BDMK_SRC "src/bdmk/*.f")
  list(REMOVE_ITEM DMK_REF_BDMK_SRC
    "${CMAKE_SOURCE_DIR}/src/bdmk/bdmk4.f"
  )
  add_library(dmk_ref_bdmk OBJECT ${DMK_REF_BDMK_SRC})
  set_target_properties(dmk_ref_bdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_library(dmk_ref STATIC $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
  add_library(dmk_ref_shared SHARED $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
  set_target_properties(dmk_ref_shared PROPERTIES OUTPUT_NAME dmk_ref)
endif()

set(DMK_SRC src/dmk.cpp src/proxy.cpp src/tree.cpp src/logger.cpp src/util.cpp src/legeexps.cpp
  src/fourier_data.cpp src/tensorprod.cpp src/planewave.cpp src/direct.cpp src/prolate.cpp src/prolate0_eval.cpp
  src/timing.cpp
  ${DMK_TEST_SRC}
)
add_library(DMKOBJS OBJECT ${DMK_SRC})
set_target_properties(DMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(DMKOBJS PUBLIC ${DMK_INCLUDES} ${PAPI_INCLUDEDIR})
target_link_libraries(DMKOBJS PUBLIC ${DMK_OPENMP_TARGET} ${DMK_MPI_TARGET} ${PAPI_TARGET}
  nda::nda_c spdlog::spdlog_header_only doctest::doctest BLAS::BLAS
)
target_compile_options(DMKOBJS PUBLIC "${DMK_CXX_FLAGS}")

add_library(dmk STATIC)
target_link_libraries(${PROJECT_NAME} PRIVATE ${DMK_OPENMP_TARGET} DMKOBJS)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(${PROJECT_NAME} PUBLIC "${DMK_CXX_FLAGS}")

add_library(dmk_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared PRIVATE DMKOBJS)
target_include_directories(${PROJECT_NAME}_shared PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(${PROJECT_NAME}_shared PUBLIC "${DMK_CXX_FLAGS}")

if(${DMK_BUILD_TESTS})
  add_subdirectory(test)
endif()

if(${DMK_BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()

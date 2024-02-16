cmake_minimum_required(VERSION 3.9)

project(dmk LANGUAGES CXX Fortran)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

add_compile_options("-march=native;-DSCTL_HAVE_MPI;-DEIGEN_USE_BLAS")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
)

file(GLOB DMK_REF_COMMON_SRC "src/common/*.f" "vec-kernels/src/libkernels.cpp")
add_library(dmk_ref_common OBJECT ${DMK_REF_COMMON_SRC})
set_target_properties(dmk_ref_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dmk_ref_common PRIVATE "extern/VCL/version2" "vec-kernels/include")

file(GLOB DMK_REF_PDMK_SRC "src/pdmk/**.f")
list(REMOVE_ITEM DMK_REF_PDMK_SRC "${CMAKE_SOURCE_DIR}/src/pdmk/kernelevaluation/l3dkernels_fast.f")
add_library(dmk_ref_pdmk OBJECT ${DMK_REF_PDMK_SRC})
set_target_properties(dmk_ref_pdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB DMK_REF_BDMK_SRC "src/bdmk/**.f")
add_library(dmk_ref_bdmk OBJECT ${DMK_REF_BDMK_SRC})
set_target_properties(dmk_ref_bdmk PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(dmk_ref STATIC $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
add_library(dmk_ref_shared SHARED $<TARGET_OBJECTS:dmk_ref_common> $<TARGET_OBJECTS:dmk_ref_pdmk> $<TARGET_OBJECTS:dmk_ref_bdmk>)
set_target_properties(dmk_ref_shared PROPERTIES OUTPUT_NAME dmk_ref)

set(DMK_SRC src/dmk.cpp src/proxy.cpp)
add_library(DMKOBJS OBJECT ${DMK_SRC})
set_target_properties(DMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(DMKOBJS PUBLIC ${DMK_INCLUDES})

add_library(dmk STATIC)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX DMKOBJS)

add_library(dmk_shared SHARED)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared DMKOBJS OpenMP::OpenMP_CXX MPI::MPI_CXX ${BLAS_LIBRARIES})

add_executable(test_chebyshev_poly_perf test/test_chebyshev_poly_perf.cpp)
target_link_libraries(test_chebyshev_poly_perf dmk_shared)

add_executable(test_chebyshev_eval_perf test/test_chebyshev_eval_perf.cpp)
target_link_libraries(test_chebyshev_eval_perf dmk_shared)

add_executable(test_chebyshev test/test_chebyshev.cpp)
target_link_libraries(test_chebyshev dmk_shared)

add_executable(test_proxy test/test_proxy.cpp)
target_link_libraries(test_proxy dmk_shared)

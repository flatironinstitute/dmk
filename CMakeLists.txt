cmake_minimum_required(VERSION 3.9)

project(dmk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options("-march=native;-DSCTL_HAVE_MPI;-DSCTL_HAVE_BLAS;-DSCTL_HAVE_LAPACK")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

set(DMK_INCLUDES
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/extern/eigen
  ${PROJECT_SOURCE_DIR}/extern/SCTL/include
)

set(DMK_SRC src/dmk.cpp)
add_library(DMKOBJS OBJECT ${DMK_SRC})
set_target_properties(DMKOBJS PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(DMKOBJS PUBLIC ${DMK_INCLUDES})

add_library(dmk STATIC $<TARGET_OBJECTS:DMKOBJS>)
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

add_library(dmk_shared SHARED $<TARGET_OBJECTS:DMKOBJS>)
set_target_properties(${PROJECT_NAME}_shared PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_shared OpenMP::OpenMP_CXX MPI::MPI_CXX ${BLAS_LIBRARIES})

add_executable(test_tree test/test_tree.cpp)
target_include_directories(test_tree PUBLIC ${DMK_INCLUDES})
target_link_libraries(test_tree dmk MPI::MPI_CXX ${BLAS_LIBRARIES})

add_executable(test_chebyshev_perf test/test_chebyshev_perf.cpp)
target_include_directories(test_chebyshev_perf PUBLIC ${DMK_INCLUDES})
target_link_libraries(test_chebyshev_perf dmk MPI::MPI_CXX ${BLAS_LIBRARIES})

add_executable(test_chebyshev test/test_chebyshev.cpp)
target_include_directories(test_chebyshev PUBLIC ${DMK_INCLUDES})
target_link_libraries(test_chebyshev dmk MPI::MPI_CXX ${BLAS_LIBRARIES})

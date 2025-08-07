add_executable(dmk_example main.cpp)
target_link_libraries(dmk_example dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET})

add_executable(ewald_total ewald_total.cpp)
target_link_libraries(ewald_total dmk dmk_ref BLAS::BLAS MPI::MPI_CXX nanobench OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})

# add an ewald library
add_library(ewald SHARED ewald_total.cpp)
target_link_libraries(ewald
    PUBLIC dmk dmk_ref BLAS::BLAS MPI::MPI_CXX
           nanobench OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})
target_include_directories(ewald PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fft_parallel_2D_f fft_parallel_2D_factored.cpp)
target_link_libraries(fft_parallel_2D_f MPI::MPI_CXX OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})

add_executable(fft_parallel_3D fft_parallel_3D.cpp)
target_link_libraries(fft_parallel_3D MPI::MPI_CXX OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})

add_executable(transp parallel_transpose.cpp)
target_link_libraries(transp MPI::MPI_CXX OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(transp3D parallel_transpose_3D.cpp)
target_link_libraries(transp3D MPI::MPI_CXX OpenMP::OpenMP_CXX ${PAPI_TARGET})

if(${DMK_BUILD_PVFMM})
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET} pvfmmStatic)
endif()
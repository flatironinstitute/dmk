add_executable(dmk_example main.cpp)
target_link_libraries(dmk_example dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET})

add_library(ewald_ex STATIC ewald_total.cpp)
target_link_libraries(ewald_ex PUBLIC dmk dmk_ref BLAS::BLAS MPI::MPI_CXX nanobench OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})
target_include_directories(ewald_ex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fft_parallel_2D_f fft_parallel_2D_factored.cpp)
target_link_libraries(fft_parallel_2D_f MPI::MPI_CXX OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})

add_executable(fft_parallel_3D fft_parallel_3D.cpp)
target_link_libraries(fft_parallel_3D MPI::MPI_CXX OpenMP::OpenMP_CXX ducc0 ${PAPI_TARGET})

add_executable(transp parallel_transpose.cpp)
target_link_libraries(transp MPI::MPI_CXX OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(transp3D parallel_transpose_3D.cpp)
target_link_libraries(transp3D MPI::MPI_CXX OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(ewald_final_test_ex test_ewald.cpp)
target_link_libraries(ewald_final_test_ex ewald_ex)

if(${DMK_BUILD_PVFMM})
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET} pvfmmStatic)
endif()
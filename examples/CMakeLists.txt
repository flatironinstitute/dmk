add_executable(dmk_example main.cpp)
target_link_libraries(dmk_example dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET})

if(${DMK_BUILD_PVFMM})
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET} pvfmm)
endif()

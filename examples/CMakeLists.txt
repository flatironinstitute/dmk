add_executable(dmk_example main.cpp)
target_link_libraries(dmk_example dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET})

add_executable(ewald_test ewald_test.cpp)
target_link_libraries(ewald_test BLAS::BLAS OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(ewald_short ewald_short.cpp)
target_link_libraries(ewald_short BLAS::BLAS OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(ewald_shorter ewald_shorter.cpp)
target_link_libraries(ewald_shorter BLAS::BLAS OpenMP::OpenMP_CXX ${PAPI_TARGET})

add_executable(ewald_total ewald_total.cpp)
target_link_libraries(ewald_total BLAS::BLAS OpenMP::OpenMP_CXX ${PAPI_TARGET})

if(${DMK_BUILD_PVFMM})
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET} pvfmmStatic)
endif()
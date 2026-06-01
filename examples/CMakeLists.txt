if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example PRIVATE dmk)

  add_executable(measure_error measure_error.cpp)
  target_link_libraries(measure_error PRIVATE dmk)

  add_executable(example_c example_c.c)
  target_link_libraries(example_c PRIVATE dmk MPI::MPI_C)
endif()

if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP AND DMK_BUILD_PVFMM)
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm PRIVATE dmk pvfmmStatic)
endif()

add_executable(benchmark_kernels benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels PRIVATE dmk)


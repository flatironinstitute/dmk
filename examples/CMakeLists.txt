if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example dmk ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})

  add_executable(measure_error measure_error.cpp)
  target_link_libraries(measure_error dmk ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})
  target_include_directories(measure_error PUBLIC ${DMK_INCLUDES})
endif()

if (DMK_HAVE_MPI AND DMK_BUILD_PVFMM)
    add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
    target_link_libraries(compare_dmk_pvfmm dmk pvfmmStatic ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})
endif()

add_executable(benchmark_kernels benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels dmk ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} ${DMK_REFERENCE_TARGET})

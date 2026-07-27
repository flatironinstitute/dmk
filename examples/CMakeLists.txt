if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example PRIVATE dmk)

  add_executable(measure_error measure_error.cpp)
  target_link_libraries(measure_error PRIVATE dmk)

  add_executable(measure_error_esp measure_error_esp.cpp)
  target_link_libraries(measure_error_esp PRIVATE dmk)
  target_include_directories(measure_error_esp PRIVATE ${CMAKE_SOURCE_DIR}/test)
endif()

if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP AND DMK_BUILD_PVFMM)
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm PRIVATE dmk pvfmmStatic)
endif()

add_executable(benchmark_kernels benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels PRIVATE dmk)

add_executable(benchmark_esp benchmark_esp.cpp)
target_link_libraries(benchmark_esp PRIVATE dmk)
target_include_directories(benchmark_esp PRIVATE ${CMAKE_SOURCE_DIR}/test)

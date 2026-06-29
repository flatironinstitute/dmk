if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP)
  add_executable(dmk_example main.cpp)
  target_link_libraries(dmk_example PRIVATE dmk)

  add_executable(measure_error measure_error.cpp)
  target_link_libraries(measure_error PRIVATE dmk)
endif()

if (DMK_HAVE_MPI AND DMK_HAVE_OPENMP AND DMK_BUILD_PVFMM)
  add_executable(compare_dmk_pvfmm compare_dmk_pvfmm.cpp)
  target_link_libraries(compare_dmk_pvfmm PRIVATE dmk pvfmmStatic)
endif()

add_executable(benchmark_kernels benchmark_kernels.cpp)
target_link_libraries(benchmark_kernels PRIVATE dmk)

if (DMK_BUILD_ESP)
  add_executable(benchmark_esp benchmark_esp.cpp)
  target_link_libraries(benchmark_esp PRIVATE dmk)
  target_compile_definitions(benchmark_esp PRIVATE
    VERIFY_SCRIPT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/verify_esp.py"
    PERILAP3D_DIR="${CMAKE_SOURCE_DIR}/scripts/perilap3d"
  )
endif()

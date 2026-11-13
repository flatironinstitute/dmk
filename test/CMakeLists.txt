set(DMK_TEST_SRC test_all.cpp test_pbc_periodic.cpp test_direct.cpp test_esp.cpp test_yukawa_levels.cpp)
if (DMK_GPU_OFFLOAD)
  list(APPEND DMK_TEST_SRC test_gpu_parity.cpp test_esp_gpu.cpp)
endif()

add_executable(test_all ${DMK_TEST_SRC})
target_link_libraries(test_all PRIVATE dmk_shared dmk_internal)
if (DMK_GPU_OFFLOAD)
  target_link_libraries(test_all PRIVATE ${DMK_GPU_TARGET})
endif()

if (DMK_HAVE_MPI)
  add_test(NAME test_all COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 --bind-to none
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_all> ${MPIEXEC_POSTFLAGS} -nc)
else()
  add_test(NAME test_all COMMAND test_all -nc)
endif()
# Autotuning a GPU build dominates the run time of the small test problems. The whole suite runs
# well past ctest's 1500 s default. Two threads keeps the MPI ranks threaded, so the hybrid path is
# what gets tested, without letting ranks times threads run away on a shared machine. The ranks are
# left unbound, since a rank bound to one core runs both its threads there.
set_tests_properties(test_all PROPERTIES
  ENVIRONMENT "DMK_JIT_AUTOTUNE_DISABLE=1;OMP_NUM_THREADS=2"
  TIMEOUT 5400)

add_executable(test_all test_all.cpp test_pbc_periodic.cpp test_direct.cpp)
target_link_libraries(test_all dmk_shared ${DMK_REFERENCE_TARGET} doctest::doctest ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} BLAS::BLAS spdlog::spdlog_header_only)
target_include_directories(test_all PRIVATE ${DMK_INCLUDES} ${PROJECT_SOURCE_DIR}/extern/nda/c++
  ${PROJECT_BINARY_DIR}/extern/nda/deps/itertools_src/c++)

add_executable(benchmark_gradient ../scripts/benchmark_gradient.cpp)
target_link_libraries(benchmark_gradient dmk_shared nda::nda_c spdlog::spdlog_header_only ${DMK_OPENMP_TARGET} BLAS::BLAS ${DMK_MPI_TARGET})
target_include_directories(benchmark_gradient PRIVATE ${DMK_INCLUDES})
target_compile_options(benchmark_gradient PRIVATE "${DMK_CXX_FLAGS}")
target_compile_definitions(benchmark_gradient PRIVATE DOCTEST_CONFIG_DISABLE SCTL_PROFILE=-1)

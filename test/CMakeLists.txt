
# add_executable(test_chebyshev_poly_perf test_chebyshev_poly_perf.cpp)
# target_link_libraries(test_chebyshev_poly_perf dmk_shared)

# add_executable(test_chebyshev_eval_perf test_chebyshev_eval_perf.cpp)
# target_link_libraries(test_chebyshev_eval_perf dmk_shared)

# add_executable(test_tree test_tree.cpp)
# target_link_libraries(test_tree dmk_shared)

# add_executable(test_pdmk test_pdmk.cpp)
# target_link_libraries(test_pdmk dmk_shared dmk_ref)

add_executable(test_all test_all.cpp)
target_link_libraries(test_all dmk_shared ${DMK_REFERENCE_TARGET} doctest::doctest ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} BLAS::BLAS)

add_executable(benchmark_gradient ../scripts/benchmark_gradient.cpp)
target_link_libraries(benchmark_gradient dmk_shared nda::nda_c spdlog::spdlog_header_only ${DMK_OPENMP_TARGET} BLAS::BLAS ${DMK_MPI_TARGET})
target_include_directories(benchmark_gradient PRIVATE ${DMK_INCLUDES})
target_compile_options(benchmark_gradient PRIVATE "${DMK_CXX_FLAGS}")
target_compile_definitions(benchmark_gradient PRIVATE DOCTEST_CONFIG_DISABLE SCTL_PROFILE=-1)

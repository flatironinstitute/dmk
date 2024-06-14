
# add_executable(test_chebyshev_poly_perf test_chebyshev_poly_perf.cpp)
# target_link_libraries(test_chebyshev_poly_perf dmk_shared)

# add_executable(test_chebyshev_eval_perf test_chebyshev_eval_perf.cpp)
# target_link_libraries(test_chebyshev_eval_perf dmk_shared)

# add_executable(test_tree test_tree.cpp)
# target_link_libraries(test_tree dmk_shared)

# add_executable(test_pdmk test_pdmk.cpp)
# target_link_libraries(test_pdmk dmk_shared dmk_ref)

add_executable(test_all test_all.cpp)
target_link_libraries(test_all dmk_shared dmk_ref doctest)

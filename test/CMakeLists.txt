add_executable(test_all test_all.cpp test_pbc_periodic.cpp test_direct.cpp)
target_link_libraries(test_all PRIVATE dmk_shared)

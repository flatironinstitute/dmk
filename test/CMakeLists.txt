add_executable(test_all test_all.cpp)
target_link_libraries(test_all dmk_shared ${DMK_REFERENCE_TARGET} doctest::doctest ${DMK_MPI_TARGET} ${DMK_OPENMP_TARGET} BLAS::BLAS)

add_executable(dmk_example main.cpp)
target_link_libraries(dmk_example dmk dmk_ref BLAS::BLAS OpenMP::OpenMP_CXX MPI::MPI_CXX ${PAPI_TARGET})

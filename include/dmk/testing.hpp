#ifndef TESTING_HPP
#define TESTING_HPP

#ifdef DMK_HAVE_MPI
#include <doctest/extensions/doctest_mpi.h>
#define TEST_CASE_GENERIC(name, arg) MPI_TEST_CASE(name, arg)
// Communicator for single-rank C API calls made from helpers, where test_comm is
// out of scope.
#define DMK_TEST_COMM_SELF MPI_COMM_SELF
#else
#include <doctest/doctest.h>
#define TEST_CASE_GENERIC(name, arg) TEST_CASE(name)
#define DMK_TEST_COMM_SELF nullptr
#endif

#endif

#ifndef TESTING_HPP
#define TESTING_HPP

#ifdef DMK_HAVE_MPI
#include <doctest/extensions/doctest_mpi.h>
#define TEST_CASE_GENERIC(name, arg) MPI_TEST_CASE(name, arg)
#else
#include <doctest/doctest.h>
#define TEST_CASE_GENERIC(name, arg) TEST_CASE(name)
#endif


#endif

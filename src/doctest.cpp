#ifdef DMK_HAVE_MPI
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/extensions/doctest_mpi.h>
#else
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#endif

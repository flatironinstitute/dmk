#ifdef DMK_HAVE_MPI
#include <doctest/extensions/doctest_mpi.h>

int main(int argc, char **argv) {
    doctest::mpi_init_thread(argc, argv, MPI_THREAD_SINGLE);

    doctest::Context ctx;
    ctx.setOption("reporters", "MpiConsoleReporter");
    ctx.setOption("reporters", "MpiFileReporter");
    ctx.setOption("force-colors", true);
    ctx.applyCommandLine(argc, argv);

    int test_result = ctx.run();

    doctest::mpi_finalize();

    return test_result;
}
#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#endif

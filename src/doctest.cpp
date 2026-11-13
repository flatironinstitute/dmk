#ifdef DMK_HAVE_MPI
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/extensions/doctest_mpi.h>

#include <iostream>
#include <sstream>

namespace {
// MpiConsoleReporter reports a failed assertion in an MPI test case by its expression and values alone, dropping the
// subcase and any INFO, CAPTURE or *_MESSAGE text, so the failing rank prints those itself.
struct DmkMpiReporter : doctest::MpiConsoleReporter {
    using doctest::MpiConsoleReporter::MpiConsoleReporter;

    void log_assert(const doctest::AssertData &rb) override {
        MpiConsoleReporter::log_assert(rb);
        if (!rb.m_failed || !is_mpi_test_case())
            return;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::ostringstream msg;
        msg << "On rank [" << rank << "] : " << file_line_to_string(rb.m_file, rb.m_line, "\n");
        for (const auto &subcase : subcasesStack)
            msg << "  in subcase: " << subcase.m_name << "\n";
        const auto contexts = doctest::IReporter::get_active_contexts();
        for (int i = 0; i < doctest::IReporter::get_num_active_contexts(); ++i) {
            msg << "  logged: ";
            contexts[i]->stringify(&msg);
            msg << "\n";
        }
        std::cerr << msg.str();
    }
};
REGISTER_REPORTER("DmkMpiReporter", 1, DmkMpiReporter);
} // namespace
#else
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#endif

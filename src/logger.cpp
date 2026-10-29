#include <dmk.h>
#include <dmk/logger.h>
#include <sctl.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace dmk {
std::shared_ptr<spdlog::logger> logger_;
std::shared_ptr<spdlog::logger> rank_logger_;
auto log_level_ = spdlog::level::off;

namespace {
// Keyed on rank 0 rather than the MPI_Comm, since sctl::Comm dups its handle.
bool is_root_ = true;

void make_logger(bool is_root) {
    is_root_ = is_root;
    spdlog::cfg::load_env_levels();
    spdlog::sink_ptr sink;
    if (is_root)
        sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
    else
        sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    logger_ = std::make_shared<spdlog::logger>("DMK", sink);
#ifdef DMK_HAVE_MPI
    logger_->set_pattern("[%T.%e] [%8i] [%n] [%l] %v");
#else
    logger_->set_pattern("[%T.%e] [%n] [%l] %v");
#endif
}
} // namespace

std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm) {
    if (!logger_ || (comm.Rank() == 0) != is_root_)
        make_logger(comm.Rank() == 0);
    logger_->set_level(log_level_);
    return logger_;
}

std::shared_ptr<spdlog::logger> &get_logger() {
    if (!logger_)
        make_logger(true);
    logger_->set_level(log_level_);
    return logger_;
}

std::shared_ptr<spdlog::logger> &get_logger(int level) {
    log_level_ = spdlog::level::level_enum(level);
    return get_logger();
}

std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm, int level) {
    log_level_ = spdlog::level::level_enum(level);
    return get_logger(comm);
}

std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm) {
#ifdef DMK_HAVE_MPI
    static bool first_call = true;
    static int rank = -1;

    if (first_call || comm.Rank() != rank) {
        first_call = false;
        rank = comm.Rank();
        spdlog::cfg::load_env_levels();

        rank_logger_ = std::make_shared<spdlog::logger>(spdlog::logger(
            "DMK-" + std::to_string(comm.Rank()), std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>()));
        rank_logger_->set_pattern("[%8i] [%n] [%l] %v");
    }

    rank_logger_->set_level(log_level_);
    return rank_logger_;
#else
    return get_logger(comm);
#endif
}

std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm, int level) {
    log_level_ = spdlog::level::level_enum(level);
    return get_rank_logger(comm);
}

} // namespace dmk

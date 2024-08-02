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
MPI_Comm comm_;

std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm) {
    bool first_call = true;

    if (first_call || comm.GetMPI_Comm() != comm_) {
        first_call = false;
        spdlog::cfg::load_env_levels();
        comm_ = comm.GetMPI_Comm();

        if (comm.Rank() == 0)
            logger_ = std::make_shared<spdlog::logger>(
                spdlog::logger("DMK", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>()));
        else
            logger_ = std::make_shared<spdlog::logger>(
                spdlog::logger("DMK", std::make_shared<spdlog::sinks::null_sink_st>()));
        logger_->set_pattern("[%8i] [%n] [%l] %v");
    }

    logger_->set_level(log_level_);
    return logger_;
}

std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm, int level) {
    log_level_ = spdlog::level::level_enum(level);
    return get_logger(comm);
}

std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm) {
    bool first_call = true;

    if (first_call || comm.GetMPI_Comm() != comm_) {
        first_call = false;
        comm_ = comm.GetMPI_Comm();
        spdlog::cfg::load_env_levels();

        rank_logger_ = std::make_shared<spdlog::logger>(spdlog::logger(
            "DMK-" + std::to_string(comm.Rank()), std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>()));
        rank_logger_->set_pattern("[%8i] [%n] [%l] %v");
    }

    rank_logger_->set_level(log_level_);
    return rank_logger_;
}

std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm, int level) {
    log_level_ = spdlog::level::level_enum(level);
    return get_rank_logger(comm);
}

} // namespace dmk

#include "spdlog/common.h"
#include <dmk/logger.h>
#include <sctl.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

namespace dmk {
std::shared_ptr<spdlog::logger> &get_logger() {
    bool first_call = true;
    static std::shared_ptr<spdlog::logger> logger;
    if (first_call) {
        first_call = false;
        spdlog::cfg::load_env_levels();

        auto comm = sctl::Comm::World();
        if (comm.Rank() == 0)
            logger = std::make_shared<spdlog::logger>(
                spdlog::logger("DMK", std::make_shared<spdlog::sinks::ansicolor_stderr_sink_st>()));
        else
            logger = std::make_shared<spdlog::logger>(
                spdlog::logger("DMK", std::make_shared<spdlog::sinks::null_sink_st>()));
        logger->set_pattern("[%6o] [%n] [%l] %v");
    }

    return logger;
}
std::shared_ptr<spdlog::logger> &get_logger(int level) {
    auto &logger = get_logger();
    logger->set_level(spdlog::level::level_enum(level));
    return logger;
}
} // namespace dmk

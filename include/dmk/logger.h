#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <spdlog/spdlog.h>

namespace dmk {
std::shared_ptr<spdlog::logger> &get_logger();
std::shared_ptr<spdlog::logger> &get_logger(int level);
std::shared_ptr<spdlog::logger> &get_rank_logger();
std::shared_ptr<spdlog::logger> &get_rank_logger(int level);
} // namespace dmk

#endif

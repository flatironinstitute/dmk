#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <sctl.hpp>
#include <spdlog/spdlog.h>

namespace dmk {
std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm);
std::shared_ptr<spdlog::logger> &get_logger(const sctl::Comm &comm, int level);
std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm);
std::shared_ptr<spdlog::logger> &get_rank_logger(const sctl::Comm &comm, int level);
} // namespace dmk

#endif

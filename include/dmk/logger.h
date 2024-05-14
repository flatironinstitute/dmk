#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <sctl.hpp>
#include <spdlog/spdlog.h>

namespace dmk {
std::shared_ptr<spdlog::logger> &get_logger();
std::shared_ptr<spdlog::logger> &get_logger(int level);
std::shared_ptr<spdlog::logger> &get_rank_logger();
std::shared_ptr<spdlog::logger> &get_rank_logger(int level);
} // namespace dmk

template <>
struct fmt::formatter<sctl::Morton<2>> : fmt::formatter<std::string> {
    auto format(sctl::Morton<2> my, format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream stream;
        stream << my;
        return format_to(ctx.out(), stream.str());
    }
};

template <>
struct fmt::formatter<sctl::Morton<3>> : fmt::formatter<std::string> {
    auto format(sctl::Morton<3> my, format_context &ctx) const -> decltype(ctx.out()) {
        std::stringstream stream;
        stream << my;
        return format_to(ctx.out(), stream.str());
    }
};

#endif

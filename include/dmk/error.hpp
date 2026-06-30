#ifndef DMK_ERROR_HPP
#define DMK_ERROR_HPP

#include <dmk.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace dmk {

/// Exception carrying a dmk_error code, used at the C API boundary (e.g. input
/// validation) so the guard can return a specific code. Internal code may keep
/// throwing plain std::exception; those map to DMK_ERR_INTERNAL.
struct api_error : std::runtime_error {
    dmk_error code;
    api_error(dmk_error c, std::string msg) : std::runtime_error(std::move(msg)), code(c) {}
};

/// Store a human-readable message for the most recent failing call on this
/// thread. Retrievable through pdmk_last_error_message().
void set_last_error(const std::string &msg);
const char *last_error_message();

/// Run f() and convert any escaping C++ exception into a dmk_error, stashing the
/// message in the thread-local last-error buffer. Used to wrap every extern "C"
/// entry point so exceptions never cross the FFI boundary.
template <class F>
dmk_error dmk_guard(F &&f) noexcept {
    try {
        f();
        return DMK_SUCCESS;
    } catch (const api_error &e) {
        set_last_error(e.what());
        return e.code;
    } catch (const std::exception &e) {
        set_last_error(e.what());
        return DMK_ERR_INTERNAL;
    } catch (...) {
        set_last_error("unknown C++ exception");
        return DMK_ERR_INTERNAL;
    }
}

} // namespace dmk

#endif

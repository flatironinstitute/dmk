#pragma once
#ifdef DMK_WITH_FINUFFT

#include <array>
#include <vector>

namespace dmk {

using Vec3 = std::array<double, 3>;

struct ESPResult { //we might only want to return total potential
    std::vector<double> total;
    std::vector<double> short_range_pot;
    std::vector<double> long_range_pot;
    std::vector<double> self_pot;
};

ESPResult esp_potential(const std::vector<Vec3> &r_src,
                        const std::vector<double> &charges,
                        double L, double r_c, double eps);

} // namespace dmk

#endif // DMK_WITH_FINUFFT

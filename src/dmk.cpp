#include <sctl.hpp>

extern "C" {
void init() { sctl::PtTree<double, 3> tree; }
}

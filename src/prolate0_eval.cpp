#include <dmk/prolate0_eval.hpp>

namespace dmk {
template float prolate0_eval<float>(float c, float x);
template double prolate0_eval<double>(double c, double x);
template float prolate0_int_eval<float>(float c, float r);
template double prolate0_int_eval<double>(double c, double r);
} // namespace dmk

#ifndef NVTX_WRAPPER_H
#define NVTX_WRAPPER_H

#define OMP_WRAPPER_H

#ifdef DMK_GPU_OFFLOAD
#include <nvtx3/nvToolsExt.h>
#else
static inline void nvtxRangePush [[maybe_unused]] (auto &s) { return; }
static inline void nvtxRangePop [[maybe_unused]] () { return; }
#endif

#endif

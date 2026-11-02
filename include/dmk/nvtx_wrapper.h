#ifndef NVTX_WRAPPER_H
#define NVTX_WRAPPER_H

#ifdef DMK_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
static inline void nvtxRangePush [[maybe_unused]] (auto &s) { return; }
static inline void nvtxRangePop [[maybe_unused]] () { return; }
#endif

#endif

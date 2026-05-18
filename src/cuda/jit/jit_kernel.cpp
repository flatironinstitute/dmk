#include "jit_kernel.hpp"

namespace dmk::cuda::jit {

void JitKernel::unload() {
    if (module_) {
        cuModuleUnload(module_);
        module_ = nullptr;
        function_ = nullptr;
    }
}


}

// API should eventually look sth like this:
/*
auto kernel = jit.get_kernel({
    .name = "charge2proxy",
    .real = "float",
    .arch = current_sm,
    .params = {
        {"N_ORDER", args.n_order},
        {"N_CHARGE_DIM", args.n_charge_dim},
        {"CHUNK", 128},
        {"I_TILE", 3},
        {"J_TILE", 3},
        {"K_TILE", 4},
    }
});

kernel.launch(
    grid,
    block,
    shared_bytes,
    stream,
    args,
    group_perm
);
*/


#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "jit_compiler.hpp"
#include "jit_kernel.hpp"
#include "jit_types.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace dmk::cuda::jit {

class JitCache {
  public:
    JitCache();
    explicit JitCache(std::vector<std::string> include_dirs);

    std::shared_ptr<JitKernel> get_kernel(const JitKey &key);

    std::shared_ptr<JitKernel> get_kernel_from_source(const JitKey &key, const std::string &source,
                                                      const std::string &name_expression = {});
    int sm_major() const { return sm_major_; }
    int sm_minor() const { return sm_minor_; }

  private:
    std::shared_ptr<JitKernel> compile_and_load(const JitKey &key);

    std::string make_source(const JitKey &key) const;

    std::vector<std::string> make_nvrtc_options() const;

    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<JitKernel>> cache_;
    JitCompiler compiler_;

    std::vector<std::string> include_dirs_;
    std::vector<std::string> extra_options_;

    int device_ = 0;
    int sm_major_ = 0;
    int sm_minor_ = 0;
};

} // namespace dmk::cuda::jit
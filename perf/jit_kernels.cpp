#include <sctl.hpp>

#define ALWAYS_INLINE __attribute__((always_inline))

template <typename T, size_t StackSize>
class StackOrHeapBuffer {
    alignas(sizeof(T)) std::array<T, StackSize> stack_buffer_;
    T *heap_buffer_ = nullptr;
    T *data_;

  public:
    StackOrHeapBuffer(size_t required_size) {
        if (required_size <= StackSize) {
            data_ = stack_buffer_.data();
        } else {
            heap_buffer_ = new T[required_size];
            data_ = heap_buffer_;
        }
    }

    T *data() { return data_; }
    const T *data() const { return data_; }

    ~StackOrHeapBuffer() {
        if (heap_buffer_) {
            delete[] heap_buffer_;
        }
    }
};

template <class F, class... Args>
ALWAYS_INLINE void dispatch_digits(int n_digits, F &&f, Args &&...args) {
    switch (n_digits) {
    case 2:
        return f.template operator()<2>(std::forward<Args>(args)...);
    case 3:
        return f.template operator()<3>(std::forward<Args>(args)...);
    case 4:
        return f.template operator()<4>(std::forward<Args>(args)...);
    case 5:
        return f.template operator()<5>(std::forward<Args>(args)...);
    case 6:
        return f.template operator()<6>(std::forward<Args>(args)...);
    case 7:
        return f.template operator()<7>(std::forward<Args>(args)...);
    case 8:
        return f.template operator()<8>(std::forward<Args>(args)...);
    case 9:
        return f.template operator()<9>(std::forward<Args>(args)...);
    case 10:
        return f.template operator()<10>(std::forward<Args>(args)...);
    case 11:
        return f.template operator()<11>(std::forward<Args>(args)...);
    case 12:
        return f.template operator()<12>(std::forward<Args>(args)...);
    default:
        throw std::runtime_error("Unsupported digits");
    }
}

// So far this almost always loses, and wwhen it wins, it's not by a lot
template <typename VecType>
ALWAYS_INLINE VecType blocked_horner(const VecType &x, const typename VecType::ScalarType *coeffs, int n_coeffs) {
    VecType x2 = x * x;
    VecType x4 = x2 * x2;

    VecType result = VecType::Zero();

    // Process 4 coefficients at a time with Estrin
    int i = n_coeffs - 1;
    for (; i >= 3; i -= 4) {
        VecType c0 = VecType::Load1(&coeffs[i - 3]);
        VecType c1 = VecType::Load1(&coeffs[i - 2]);
        VecType c2 = VecType::Load1(&coeffs[i - 1]);
        VecType c3 = VecType::Load1(&coeffs[i]);

        // 2-level Estrin for these 4 coeffs
        VecType p01 = FMA(c1, x, c0);
        VecType p23 = FMA(c3, x, c2);
        VecType block = FMA(p23, x2, p01);

        // Combine with previous result
        result = FMA(result, x4, block);
    }

    // Handle remaining coefficients with Horner
    for (; i >= 0; --i) {
        result = FMA(result, x, VecType::Load1(&coeffs[i]));
    }

    return result;
}

template <typename VecType>
ALWAYS_INLINE VecType horner(const VecType &x, const typename VecType::ScalarType *coeffs, int n_coeffs) {
    VecType poly = VecType::Zero();
    for (int i = n_coeffs - 1; i >= 0; --i)
        poly = FMA(poly, x, VecType::Load1(&coeffs[i]));
    return poly;
}

template <class Real, int digits, int KERNEL_INPUT_DIM, int KERNEL_OUTPUT_DIM, int SPATIAL_DIM, int NORMAL_DIM,
          int VecLen, class uKernelEvaluator>
void EvalPairs(int Ns, const Real *__restrict__ r_src, const Real *__restrict__ v_src,
               const Real *__restrict__ src_normals, int Nt, const Real *__restrict__ r_trg, Real *__restrict__ v_trg,
               uKernelEvaluator uKernel) {
    using namespace sctl;
    static constexpr Integer digits_ = (digits == -1 ? (Integer)(TypeTraits<Real>::SigBits * 0.3010299957) : digits);
    using RealVec = Vec<Real, VecLen>;

    Real scale_factor = 1.0;
    if constexpr (requires {
                      { uKernel.scale_factor() };
                  }) {
        scale_factor = static_cast<Real>(uKernel.scale_factor());
    }

    const Long NNt = ((Nt + VecLen - 1) / VecLen) * VecLen;
    if (NNt == VecLen) {
        RealVec xt[SPATIAL_DIM], vt[KERNEL_OUTPUT_DIM], xs[SPATIAL_DIM], ns[NORMAL_DIM], vs[KERNEL_INPUT_DIM];
        for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
            vt[k] = RealVec::Zero();
        for (Integer k = 0; k < SPATIAL_DIM; k++) {
            alignas(sizeof(RealVec)) std::array<Real, VecLen> Xt;
            RealVec::Zero().StoreAligned(&Xt[0]);
            for (Integer i = 0; i < Nt; i++)
                Xt[i] = r_trg[i * SPATIAL_DIM + k];
            xt[k] = RealVec::LoadAligned(&Xt[0]);
        }
        for (Long s = 0; s < Ns; s++) {
            for (Integer k = 0; k < SPATIAL_DIM; k++)
                xs[k] = RealVec::Load1(&r_src[s * SPATIAL_DIM + k]);
            for (Integer k = 0; k < NORMAL_DIM; k++)
                ns[k] = RealVec::Load1(&src_normals[s * NORMAL_DIM + k]);
            for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                vs[k] = RealVec::Load1(&v_src[s * KERNEL_INPUT_DIM + k]);

            RealVec dX[SPATIAL_DIM], U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
            for (Integer i = 0; i < SPATIAL_DIM; i++)
                dX[i] = xt[i] - xs[i];
            if constexpr (NORMAL_DIM > 0)
                uKernel(U, dX, ns);
            else
                uKernel(U, dX);

            for (Integer k0 = 0; k0 < KERNEL_INPUT_DIM; k0++) {
                for (Integer k1 = 0; k1 < KERNEL_OUTPUT_DIM; k1++) {
                    vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
                }
            }
        }
        for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++) {
            alignas(sizeof(RealVec)) std::array<Real, VecLen> out;
            vt[k].StoreAligned(&out[0]);
            for (Long t = 0; t < Nt; t++) {
                v_trg[t * KERNEL_OUTPUT_DIM + k] += out[t] * scale_factor;
            }
        }
    } else {
        const Real *__restrict__ Xs_ = r_src;
        const Real *__restrict__ Ns_ = src_normals;
        const Real *__restrict__ Vs_ = v_src;

        constexpr Integer Nbuff = 16 * 1024;
        constexpr Integer alignment = sizeof(RealVec) / sizeof(Real);

        // Round up Xt size to alignment boundary
        const Integer Xt_size = SPATIAL_DIM * NNt;
        const Integer required_size = Xt_size + NNt * KERNEL_OUTPUT_DIM;

        StackOrHeapBuffer<Real, Nbuff> buffer(required_size);
        Real *buff = buffer.data();
        Real *__restrict__ Xt_ = buff;
        Real *__restrict__ Vt_ = buff + Xt_size;

        for (Long k = 0; k < SPATIAL_DIM; k++) {
            for (Long i = 0; i < Nt; i++) {
                Xt_[k * NNt + i] = r_trg[i * SPATIAL_DIM + k];
            }
            for (Long i = Nt; i < NNt; i++) {
                Xt_[k * NNt + i] = 0;
            }
        }

        for (Long t = 0; t < NNt; t += VecLen) {
            RealVec xt[SPATIAL_DIM], vt[KERNEL_OUTPUT_DIM], xs[SPATIAL_DIM], ns[NORMAL_DIM], vs[KERNEL_INPUT_DIM];
            for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
                vt[k] = RealVec::Zero();
            for (Integer k = 0; k < SPATIAL_DIM; k++)
                xt[k] = RealVec::LoadAligned(&Xt_[k * NNt + t]);
            for (Long s = 0; s < Ns; s++) {
                for (Integer k = 0; k < SPATIAL_DIM; k++)
                    xs[k] = RealVec::Load1(&Xs_[s * SPATIAL_DIM + k]);
                for (Integer k = 0; k < NORMAL_DIM; k++)
                    ns[k] = RealVec::Load1(&Ns_[s * NORMAL_DIM + k]);
                for (Integer k = 0; k < KERNEL_INPUT_DIM; k++)
                    vs[k] = RealVec::Load1(&Vs_[s * KERNEL_INPUT_DIM + k]);

                RealVec dX[SPATIAL_DIM], U[KERNEL_INPUT_DIM][KERNEL_OUTPUT_DIM];
                for (Integer i = 0; i < SPATIAL_DIM; i++)
                    dX[i] = xt[i] - xs[i];
                if constexpr (NORMAL_DIM > 0) {
                    uKernel(U, dX, ns);
                } else {
                    uKernel(U, dX);
                }

                for (Integer k0 = 0; k0 < KERNEL_INPUT_DIM; k0++) {
                    for (Integer k1 = 0; k1 < KERNEL_OUTPUT_DIM; k1++) {
                        vt[k1] = FMA(U[k0][k1], vs[k0], vt[k1]);
                    }
                }
            }
            for (Integer k = 0; k < KERNEL_OUTPUT_DIM; k++)
                vt[k].StoreAligned(&Vt_[k * NNt + t]);
        }

        for (Long k = 0; k < KERNEL_OUTPUT_DIM; k++) {
            for (Long i = 0; i < Nt; i++) {
                v_trg[i * KERNEL_OUTPUT_DIM + k] += Vt_[k * NNt + i] * scale_factor;
            }
        }
    }
    // Profile::IncrementCounter(ProfileCounter::FLOP, Ns * Nt * uKernelEvaluator::FLOPS());
}

template <int digits, class Real, int MaxVecLen>
void laplace_pswf_all_pairs_jit_impl(int nd, Real rsc, Real cen, Real d2max, Real thresh2, int n_coeffs,
                                     const Real *coeffs, int n_src, const Real *r_src, const Real *charge, int n_trg,
                                     const Real *r_trg, Real *pot) {
    using VecType = sctl::Vec<Real, MaxVecLen>;

    struct Evaluator {
        VecType thresh2_vec, d2max_vec, rsc_vec, cen_vec;
        const Real *coeffs;
        int n_coeffs;

        static constexpr Real scale_factor() { return Real{1.0}; }

        Evaluator(VecType thresh2_vec, VecType d2max_vec, VecType rsc_vec, VecType cen_vec, const Real *coeffs,
                  int n_coeffs)
            : thresh2_vec(thresh2_vec), d2max_vec(d2max_vec), rsc_vec(rsc_vec), cen_vec(cen_vec), coeffs(coeffs),
              n_coeffs(n_coeffs) {}

        void operator()(VecType (&u)[1][1], const VecType (&dX)[3]) const {
            const VecType R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
            const auto mask = (R2 > thresh2_vec) & (R2 < d2max_vec);

            if constexpr (digits >= 6) {
                if (!sctl::mask_any(mask)) {
                    u[0][0] = VecType::Zero();
                    return;
                }
            }
            const VecType Rinv = sctl::approx_rsqrt<digits>(R2, mask);

            const VecType x = sctl::FMA(R2, Rinv, cen_vec) * rsc_vec;
            u[0][0] = horner(x, coeffs, n_coeffs) * Rinv;
        }
    };

    Evaluator evaluator(thresh2, d2max, rsc, cen, coeffs, n_coeffs);

    constexpr int KERNEL_INPUT_DIM = 1;
    constexpr int KERNEL_OUTPUT_DIM = 1;
    constexpr int SPATIAL_DIM = 3;
    constexpr int NORMAL_DIM = 0;

    EvalPairs<Real, digits, KERNEL_INPUT_DIM, KERNEL_OUTPUT_DIM, SPATIAL_DIM, NORMAL_DIM, MaxVecLen>(
        n_src, r_src, charge, nullptr, n_trg, r_trg, pot, evaluator);
}

template <class Real, int MaxVecLen>
void laplace_pswf_all_pairs_jit(int nd, int n_digits, Real rsc, Real cen, Real d2max, Real thresh2, int n_coeffs,
                                const Real *coeffs, int n_src, const Real *r_src, const Real *charge, int n_trg,
                                const Real *r_trg, Real *pot) {
    auto kernel = [&]<int DIGITS>() {
        laplace_pswf_all_pairs_jit_impl<DIGITS, Real, MaxVecLen>(nd, rsc, cen, d2max, thresh2, n_coeffs, coeffs, n_src,
                                                                 r_src, charge, n_trg, r_trg, pot);
    };
    dispatch_digits(n_digits, kernel);
}

template void laplace_pswf_all_pairs_jit<float, 16>(int nd, int n_digits, float rsc, const float cen, float d2max,
                                                    float thresh2, int n_coeffs, const float *coeffs, int n_src,
                                                    const float *r_src, const float *charge, int n_trg,
                                                    const float *r_trg, float *pot);
template void laplace_pswf_all_pairs_jit<double, 8>(int nd, int n_digits, double rsc, const double cen, double d2max,
                                                    double thresh2, int n_coeffs, const double *coeffs, int n_src,
                                                    const double *r_src, const double *charge, int n_trg,
                                                    const double *r_trg, double *pot);

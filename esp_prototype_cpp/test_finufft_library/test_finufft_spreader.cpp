#include <finufft.h>
#include <complex>
#include <vector>
#include <cstdio>
#include <cmath>

int main() {
    // 10 source points and charges from get_test2_input
    int M = 10;
    std::vector<double> r_src_flat = {
        0.131538-0.5, 0.686773-0.5, 0.98255-0.5,
        0.45865-0.5,  0.930436-0.5, 0.753356-0.5,
        0.218959-0.5, 0.526929-0.5, 0.0726859-0.5,
        0.678865-0.5, 0.653919-0.5, 0.884707-0.5,
        0.934693-0.5, 0.701191-0.5, 0.436411-0.5,
        0.519416-0.5, 0.762198-0.5, 0.477732-0.5,
        0.0345721-0.5,0.0474645-0.5,0.274907-0.5,
        0.5297-0.5,   0.328234-0.5, 0.166507-0.5,
        0.00769819-0.5,0.75641-0.5, 0.897656-0.5,
        0.0668422-0.5, 0.365339-0.5, 0.0605643-0.5
    };
    std::vector<double> charges = {0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.1, -0.1};

    // grid parameters - must match your ESPParams
    int n_f = 32; // grid size, adjust to match your params.n_f
    int N = n_f;  // same in all 3 dims

    // FINUFFT requires positions in [-pi, pi)
    // your positions are in [-0.5, 0.5], so scale by 2*pi
    double scale = 2.0 * M_PI;
    std::vector<double> x(M), y(M), z(M);
    for (int j = 0; j < M; j++) {
        x[j] = r_src_flat[j*3 + 0] * scale;
        y[j] = r_src_flat[j*3 + 1] * scale;
        z[j] = r_src_flat[j*3 + 2] * scale;
    }

    // complex strengths (charges are real, imaginary part = 0)
    std::vector<std::complex<double>> c(M), c_out(M);
    for (int j = 0; j < M; j++)
        c[j] = std::complex<double>(charges[j], 0.0);

    // output grid (complex)
    std::vector<std::complex<double>> F(n_f * n_f * n_f, 0.0);

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.spread_kerformula = 8;  // use PSWF kernel
    //opts.upsampfac        = 1.0; // no upsampling since we're doing spread/interp only
    double tol            = 1e-6;
    int unused            = +1;

    // STEP 1: spread (type 1) - NU pts -> uniform grid
    printf("Spreading %d NU pts to %dx%dx%d grid...\n", M, n_f, n_f, n_f);
    int ier = finufft3d1(M, x.data(), y.data(), z.data(),
                         c.data(), unused, tol,
                         n_f, n_f, n_f,
                         F.data(), &opts);
    if (ier > 1) {
        printf("spread error ier=%d\n", ier);
        return ier;
    }

    // STEP 2: interpolate (type 2) - uniform grid -> NU pts
    printf("Interpolating back to %d NU pts...\n", M);
    ier = finufft3d2(M, x.data(), y.data(), z.data(),
                     c_out.data(), unused, tol,
                     n_f, n_f, n_f,
                     F.data(), &opts);
    if (ier > 1) {
        printf("interp error ier=%d\n", ier);
        return ier;
    }

    printf("\nResults (ratio recovered/input should be constant = kernel mass):\n");
    for (int j = 0; j < M; j++) {
        double ratio = c_out[j].real() / c[j].real();
        printf("  pt %d: in=%.4f  out=%.4f  ratio=%.6f\n",
            j, charges[j], c_out[j].real(), ratio);
    }
    return 0;
}
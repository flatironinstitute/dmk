#!/usr/bin/env python3
"""Generate include/dmk/bessel.hpp from Bessels.jl reference source.

Ports besselk0/k1/j0/j1 (orders 0 and 1 only) from JuliaMath/Bessels.jl (MIT).
The coefficient tables are parsed directly from the reference .jl files so the
minimax constants are reproduced verbatim (no hand transcription).

By default this clones JuliaMath/Bessels.jl at a pinned commit into a temporary
directory and reads the reference files from there (requires network + git), so
no third-party .jl files need to live in this repo. To build offline from a local
checkout instead, set BESSELS_REF to its src/BesselFunctions directory.

Run:    python3 scripts/gen_bessel.py
Writes: include/dmk/bessel.hpp
"""
import ast
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_URL = "https://github.com/JuliaMath/Bessels.jl"
REPO_COMMIT = "009b615"  # pinned; bump deliberately and re-review coefficients


def _clean(julia: str) -> str:
    # Julia float literals: 2.62f-1 -> 2.62e-1, 1.0f0 -> 1.0e0
    s = re.sub(r"f([-+]?\d)", r"e\1", julia)
    s = s.replace("one(Float32)", "1.0").replace("zero(Float32)", "0.0")
    s = s.replace("one(Float64)", "1.0").replace("zero(Float64)", "0.0")
    # strip trailing line comments inside multi-line tuples
    s = re.sub(r"#[^\n]*", "", s)
    return s


def _balanced(text: str, start: int):
    """Return the substring of the balanced (...) group beginning at/after start."""
    i = text.index("(", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
    raise ValueError("unbalanced")


def extract(text: str, name: str, typ):
    """Extract a const definition as a python object.

    typ in {'Float64','Float32',None}. Returns nested tuples/floats.
    """
    if typ is None:
        pat = re.compile(r"\b" + re.escape(name) + r"\s*=")
    else:
        pat = re.compile(r"\b" + re.escape(name) + r"\s*\(\s*::\s*Type\{\s*" + typ + r"\s*\}\s*\)\s*=")
    m = pat.search(text)
    if not m:
        raise KeyError(f"{name} [{typ}] not found")
    rhs_start = m.end()
    # value may be a scalar or a (...) tuple
    rest = text[rhs_start:]
    # find first non-space
    k = len(rest) - len(rest.lstrip())
    if rest[k] == "(":
        grp = _balanced(text, rhs_start)
        return ast.literal_eval(_clean(grp))
    else:
        # scalar: read to end of line
        line = rest.splitlines()[0]
        return ast.literal_eval(_clean(line).strip())


def fmt(x):
    return repr(float(x))


def c_array(name, ctype, values):
    body = ", ".join(fmt(v) for v in values)
    return f"static constexpr {ctype} {name}[{len(values)}] = {{{body}}};"


def c_array2(name, ctype, rows):
    ncol = len(rows[0])
    lines = []
    for r in rows:
        assert len(r) == ncol, f"{name} ragged"
        lines.append("    {" + ", ".join(fmt(v) for v in r) + "}")
    body = ",\n".join(lines)
    return f"static constexpr {ctype} {name}[{len(rows)}][{ncol}] = {{\n{body}}};"


def read_refs():
    """Return (constants.jl, besselj_polys.jl) text from Bessels.jl.

    Uses BESSELS_REF (a local src/BesselFunctions dir) if set; otherwise clones
    the pinned commit into a temp dir and removes it afterward.
    """
    def load(bf):
        consts = open(os.path.join(bf, "constants.jl")).read()
        polys = open(os.path.join(bf, "Polynomials", "besselj_polys.jl")).read()
        return consts, polys

    ref = os.environ.get("BESSELS_REF")
    if ref:
        return load(ref)

    tmp = tempfile.mkdtemp(prefix="bessels_ref_")
    try:
        print(f"cloning {REPO_URL}@{REPO_COMMIT} ...")
        subprocess.run(["git", "clone", "--quiet", REPO_URL, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", REPO_COMMIT], check=True)
        return load(os.path.join(tmp, "src", "BesselFunctions"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    consts, polys = read_refs()

    out = []
    A = out.append

    # --- K0/K1 coefficient tables (float64 + float32) ---
    k_names = ["P1_k0", "Q1_k0", "P2_k0", "P3_k0", "Q3_k0",
               "P1_k1", "Q1_k1", "P2_k1", "Q2_k1", "P3_k1", "Q3_k1"]
    for nm in k_names:
        A(c_array(nm + "_d", "double", extract(consts, nm, "Float64")))
        A(c_array(nm + "_f", "float", extract(consts, nm, "Float32")))
    # scalars
    A(f"static constexpr double Y_k0_d = {fmt(extract(consts, 'Y_k0', None))};")
    A(f"static constexpr float  Y_k0_f = {fmt(extract(consts, 'Y_k0', None))}f;")
    A(f"static constexpr double Y_k1_d = {fmt(extract(consts, 'Y_k1', 'Float64'))};")
    A(f"static constexpr float  Y_k1_f = {fmt(extract(consts, 'Y_k1', 'Float32'))}f;")
    A(f"static constexpr double Y2_k1_d = {fmt(extract(consts, 'Y2_k1', 'Float64'))};")
    A(f"static constexpr float  Y2_k1_f = {fmt(extract(consts, 'Y2_k1', 'Float32'))}f;")

    # --- J0/J1 float32 Cephes-style tables ---
    A(c_array("JP_j0_f", "float", extract(consts, "JP_j0", "Float32")))
    A(c_array("MO_j0_f", "float", extract(consts, "MO_j0", "Float32")))
    A(c_array("PH_j0_f", "float", extract(consts, "PH_j0", "Float32")))
    A(c_array("JP_j1_f", "float", extract(consts, "JP32", None)))
    A(c_array("MO_j1_f", "float", extract(consts, "MO132", None)))
    A(c_array("PH_j1_f", "float", extract(consts, "PH132", None)))

    # --- J0/J1 float64 Harrison tables ---
    j0_roots = extract(polys, "J0_ROOTS", "Float64")
    j0_polys = extract(polys, "J0_POLYS", "Float64")
    j0_pio2 = extract(polys, "J0_POLY_PIO2", "Float64")
    j1_roots = extract(polys, "J1_ROOTS", "Float64")
    j1_polys = extract(polys, "J1_POLYS", "Float64")
    j1_pio2 = extract(polys, "J1_POLY_PIO2", "Float64")
    A(c_array2("J0_ROOTS_d", "double", j0_roots))
    A(c_array2("J0_POLYS_d", "double", j0_polys))
    A(c_array("J0_PIO2_d", "double", j0_pio2))
    A(c_array2("J1_ROOTS_d", "double", j1_roots))
    A(c_array2("J1_POLYS_d", "double", j1_polys))
    A(c_array("J1_PIO2_d", "double", j1_pio2))

    coeff_block = "\n".join(out)

    header = TEMPLATE.replace("@@COEFFS@@", coeff_block)
    dst = os.path.join(ROOT, "include", "dmk", "bessel.hpp")
    with open(dst, "w") as f:
        f.write(header)
    print("wrote", dst)
    print(f"  J0_POLYS {len(j0_polys)}x{len(j0_polys[0])}, J1_POLYS {len(j1_polys)}x{len(j1_polys[0])}")


TEMPLATE = r'''// include/dmk/bessel.hpp -- GENERATED by scripts/gen_bessel.py; do not edit by hand.
//
// Scalar Bessel functions J0,J1,K0,K1 (orders 0 and 1 only), ported from
// JuliaMath/Bessels.jl (MIT License), commit 009b615. Coefficient tables are
// reproduced verbatim from the Bessels.jl reference source.
//
// Bessels.jl is Copyright (c) 2021 Michael Helton and contributors, MIT License:
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction ... (full MIT text at
//   https://github.com/JuliaMath/Bessels.jl/blob/master/LICENSE ).
//
// K0/K1 use the two-branch rational approximations from Holoborodko / boost math.
// J0/J1 (double) use Harrison's root-expansion minimax polynomials for |x|<26 and
// an asymptotic amplitude/phase form for |x|>=26. J0/J1 (float) use the Cephes
// small-arg + asymptotic forms. Only nu in {0,1} is provided (DMK needs no more).
#ifndef DMK_BESSEL_HPP
#define DMK_BESSEL_HPP

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace dmk::bessel {
namespace detail {

@@COEFFS@@

// Horner evaluation of a polynomial with ascending coefficients (evalpoly).
template <class T, std::size_t N>
inline T evalpoly(T x, const T (&c)[N]) {
    T r = c[N - 1];
    for (std::size_t i = N - 1; i-- > 0;)
        r = std::fma(r, x, c[i]);
    return r;
}

// sin(x + phi) with x possibly large: keep x exact (libm reduces internally),
// combine with the small phase phi via the angle-addition identity. Equivalent
// in intent to Bessels.jl sin_sum (full accuracy independent of |x|).
template <class T>
inline T sin_x_plus_phi(T x, T phi) {
    return std::sin(x) * std::cos(phi) + std::cos(x) * std::sin(phi);
}

} // namespace detail

template <class T>
inline T k0(T x) {
    using namespace detail;
    if (x <= T(0))
        return std::numeric_limits<T>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        if (x <= 1.0) {
            double a = x * x / 4;
            double s = evalpoly(a, P1_k0_d) / evalpoly(a, Q1_k0_d) + Y_k0_d;
            a = std::fma(s, a, 1.0);
            return std::fma(-a, std::log(x), evalpoly(x * x, P2_k0_d));
        } else {
            double s = std::exp(-x / 2);
            double a = (evalpoly(1.0 / x, P3_k0_d) / evalpoly(1.0 / x, Q3_k0_d) + 1.0) * s / std::sqrt(x);
            return a * s;
        }
    } else {
        if (x <= 1.0f) {
            float a = x * x / 4;
            float s = evalpoly(a, P1_k0_f) / evalpoly(a, Q1_k0_f) + Y_k0_f;
            a = std::fma(s, a, 1.0f);
            return std::fma(-a, std::log(x), evalpoly(x * x, P2_k0_f));
        } else {
            float s = std::exp(-x / 2);
            float a = (evalpoly(1.0f / x, P3_k0_f) / evalpoly(1.0f / x, Q3_k0_f) + 1.0f) * s / std::sqrt(x);
            return a * s;
        }
    }
}

template <class T>
inline T k1(T x) {
    using namespace detail;
    if (x <= T(0))
        return std::numeric_limits<T>::quiet_NaN();
    if constexpr (std::is_same_v<T, double>) {
        if (x <= 1.0) {
            double z = x * x, a = z / 4;
            double pq = evalpoly(a, P1_k1_d) / evalpoly(a, Q1_k1_d) + Y_k1_d;
            pq = std::fma(pq * a, a, (a / 2 + 1));
            double aa = pq * x / 2;
            pq = std::fma(evalpoly(z, P2_k1_d) / evalpoly(z, Q2_k1_d), x, 1.0 / x);
            return std::fma(aa, std::log(x), pq);
        } else {
            double s = std::exp(-x / 2);
            double a = (evalpoly(1.0 / x, P3_k1_d) / evalpoly(1.0 / x, Q3_k1_d) + Y2_k1_d) * s / std::sqrt(x);
            return a * s;
        }
    } else {
        if (x <= 1.0f) {
            float z = x * x, a = z / 4;
            float pq = evalpoly(a, P1_k1_f) / evalpoly(a, Q1_k1_f) + Y_k1_f;
            pq = std::fma(pq * a, a, (a / 2 + 1));
            float aa = pq * x / 2;
            pq = std::fma(evalpoly(z, P2_k1_f) / evalpoly(z, Q2_k1_f), x, 1.0f / x);
            return std::fma(aa, std::log(x), pq);
        } else {
            float s = std::exp(-x / 2);
            float a = (evalpoly(1.0f / x, P3_k1_f) / evalpoly(1.0f / x, Q3_k1_f) + Y2_k1_f) * s / std::sqrt(x);
            return a * s;
        }
    }
}

template <class T>
inline T j0(T x) {
    using namespace detail;
    x = std::abs(x);
    if constexpr (std::is_same_v<T, double>) {
        constexpr double PIO2 = 1.5707963267948966;
        constexpr double TWOOPI = 0.6366197723675814;
        constexpr double SQ2OPI = 0.79788456080286535588;
        constexpr double PIO4 = 0.78539816339744830962;
        if (x < 26.0) {
            if (x < PIO2)
                return evalpoly(x * x, J0_PIO2_d);
            int idx = int(TWOOPI * x) - 1;
            if (idx < 0) idx = 0;
            if (idx > 15) idx = 15;
            double r = x - J0_ROOTS_d[idx][0] - J0_ROOTS_d[idx][1];
            return evalpoly(r, J0_POLYS_d[idx]);
        } else {
            double xinv = 1.0 / x;
            if (xinv == 0.0)
                return 0.0;
            double x2 = xinv * xinv, p, q;
            if (x < 120.0) {
                const double p1[] = {1.0, -1.0 / 16, 53.0 / 512, -4447.0 / 8192, 3066403.0 / 524288,
                                     -896631415.0 / 8388608, 796754802993.0 / 268435456,
                                     -500528959023471.0 / 4294967296};
                const double q1[] = {-1.0 / 8, 25.0 / 384, -1073.0 / 5120, 375733.0 / 229376,
                                     -55384775.0 / 2359296, 24713030909.0 / 46137344, -7780757249041.0 / 436207616};
                p = evalpoly(x2, p1);
                q = evalpoly(x2, q1);
            } else {
                const double p2[] = {1.0, -1.0 / 16, 53.0 / 512, -4447.0 / 8192};
                const double q2[] = {-1.0 / 8, 25.0 / 384, -1073.0 / 5120, 375733.0 / 229376};
                p = evalpoly(x2, p2);
                q = evalpoly(x2, q2);
            }
            double a = SQ2OPI * std::sqrt(xinv) * p;
            double xn = xinv * q;
            return a * sin_x_plus_phi(x, PIO4 + xn);
        }
    } else {
        constexpr float PIO4 = 0.78539816339744830962f;
        if (x <= 2.0f) {
            float z = x * x;
            if (x < 1.0e-3f)
                return 1.0f - 0.25f * z;
            constexpr float DR1 = 5.78318596294678452118f;
            return (z - DR1) * evalpoly(z, JP_j0_f);
        } else {
            float q = 1.0f / x;
            if (q == 0.0f)
                return 0.0f;
            float w = std::sqrt(q);
            float p = w * evalpoly(q, MO_j0_f);
            w = q * q;
            float xn = q * evalpoly(w, PH_j0_f) - PIO4;
            return p * std::cos(xn + x);
        }
    }
}

template <class T>
inline T j1(T x) {
    using namespace detail;
    T s = (x < T(0)) ? T(-1) : T(1);
    x = std::abs(x);
    if constexpr (std::is_same_v<T, double>) {
        constexpr double PIO2 = 1.5707963267948966;
        constexpr double TWOOPI = 0.6366197723675814;
        constexpr double SQ2OPI = 0.79788456080286535588;
        constexpr double PIO4 = 0.78539816339744830962;
        if (x <= 26.0) {
            if (x <= PIO2)
                return x * evalpoly(x * x, J1_PIO2_d) * s;
            int idx = int(TWOOPI * x) - 1;
            if (idx < 0) idx = 0;
            if (idx > 15) idx = 15;
            double r = x - J1_ROOTS_d[idx][0] - J1_ROOTS_d[idx][1];
            return evalpoly(r, J1_POLYS_d[idx]) * s;
        } else {
            double xinv = 1.0 / x;
            if (xinv == 0.0)
                return 0.0;
            double x2 = xinv * xinv, p, q;
            if (x < 120.0) {
                const double p1[] = {1.0, 3.0 / 16, -99.0 / 512, 6597.0 / 8192, -4057965.0 / 524288,
                                     1113686901.0 / 8388608, -951148335159.0 / 268435456,
                                     581513783771781.0 / 4294967296};
                const double q1[] = {3.0 / 8, -21.0 / 128, 1899.0 / 5120, -543483.0 / 229376,
                                     8027901.0 / 262144, -30413055339.0 / 46137344, 9228545313147.0 / 436207616};
                p = evalpoly(x2, p1);
                q = evalpoly(x2, q1);
            } else {
                const double p2[] = {1.0, 3.0 / 16, -99.0 / 512, 6597.0 / 8192};
                const double q2[] = {3.0 / 8, -21.0 / 128, 1899.0 / 5120, -543483.0 / 229376};
                p = evalpoly(x2, p2);
                q = evalpoly(x2, q2);
            }
            double a = SQ2OPI * std::sqrt(xinv) * p;
            double xn = xinv * q;
            return a * sin_x_plus_phi(x, -PIO4 + xn) * s;
        }
    } else {
        constexpr float THPIO4 = 2.35619449019234492885f;
        if (x <= 2.0f) {
            float z = x * x;
            constexpr float Z1 = 1.46819706421238932572e1f;
            return (z - Z1) * x * evalpoly(z, JP_j1_f) * s;
        } else {
            float q = 1.0f / x;
            if (q == 0.0f)
                return 0.0f;
            float w = std::sqrt(q);
            float p = w * evalpoly(q, MO_j1_f);
            w = q * q;
            float xn = q * evalpoly(w, PH_j1_f) - THPIO4;
            return p * std::cos(xn + x) * s;
        }
    }
}

} // namespace dmk::bessel

#endif // DMK_BESSEL_HPP
'''

if __name__ == "__main__":
    main()

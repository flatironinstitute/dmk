# Third-party notices

DMK is distributed under the Apache License 2.0 (see `LICENSE`). Binary distributions of DMK —
`libdmk.a`, `libdmk.so`, and any language-binding artifact built from them — statically incorporate
the components below. `libdmk.a` additionally contains the object files of the archives marked
"merged archive".

| Component | License | Copyright | Incorporated as |
|---|---|---|---|
| [FINUFFT](https://github.com/flatironinstitute/finufft) | Apache-2.0 | Copyright (C) 2017-2026 The Simons Foundation, Inc. | merged archive |
| [ducc0](https://github.com/mreineck/ducc) (FFT components) | BSD-3-Clause OR GPL-2.0-or-later | Copyright (C) Martin Reinecke | merged archive |
| [nda](https://github.com/TRIQS/nda) | Apache-2.0 | Copyright (c) 2019--present, The Simons Foundation | merged archive |
| [xsimd](https://github.com/xtensor-stack/xsimd) | BSD-3-Clause | Copyright (c) 2016 Johan Mabille, Sylvain Corlay, Wolf Vollprecht and Martin Renou; Copyright (c) 2016 QuantStack; Copyright (c) 2018 Serge Guelton | headers, compiled into FINUFFT |
| [SCTL](https://github.com/dmalhotra/SCTL) | Apache-2.0 | Copyright (c) Dhairya Malhotra | headers, compiled in |
| [spdlog](https://github.com/gabime/spdlog) | MIT | Copyright (c) 2016 - present, Gabi Melman and spdlog contributors | headers, compiled in |
| [polyfit](https://github.com/DiamonDinoia/polyfit) | MIT | Copyright 2025 Marco Barbone, Simons Foundation | headers, compiled in |
| [doctest](https://github.com/doctest/doctest) | MIT | Copyright (c) 2016-2023 Viktor Kirilov | headers, compiled in |

ducc0's FFT components are dual-licensed; DMK elects the BSD-3-Clause option.

## Included numerical routines

DMK's Legendre expansion and prolate spheroidal wave function routines are C++ ports of Fortran by
Vladimir Rokhlin. The original sources are in this repository:
`src/common/specialfunctions/legeexps.f`, `prolcrea.f` and `prolaterouts.f`.

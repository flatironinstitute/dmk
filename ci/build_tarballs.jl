# Run through ci/build_jll.sh, which sets up the Yggdrasil checkout, depot and working directory.
using BinaryBuilder, BinaryBuilderBase, Pkg
using Base.BinaryPlatforms

const YGGDRASIL_DIR = ENV["YGGDRASIL_DIR"]
include(joinpath(YGGDRASIL_DIR, "fancy_toys.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "cuda.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "macos_sdks.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "microarchitectures.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "mpi.jl"))

name = "DMK"
dmk_root = dirname(@__DIR__)
version = VersionNumber(readchomp(joinpath(dmk_root, "VERSION.txt")))

# DirectorySource cannot exclude anything and a checkout holds build trees, so stage the tracked files.
dmk_src = mktempdir()
for f in readlines(`git -C $dmk_root ls-files --recurse-submodules`)
    mkpath(dirname(joinpath(dmk_src, f)))
    cp(joinpath(dmk_root, f), joinpath(dmk_src, f))
end

sources = [
    DirectorySource(dmk_src; target="DMK"),
    # finufft and polyfit, and what they fetch in turn, are pinned and downloaded at configure time;
    # nda instead clones these from branch heads unless they are already in its deps/
    GitSource("https://github.com/google/googletest.git", "dd0a37b8432bf8a0c3e33938ae9807d413db0cbd"),
    GitSource("https://github.com/TRIQS/itertools.git", "d32d2b20a1beda709a43bd56ac85ea6d41efcdef"),
    GitSource("https://github.com/TRIQS/mpi.git", "c760e19bd79b3fb84fca2b1e861a3478047f6a42"),
]

script = raw"""
cd ${WORKSPACE}/srcdir
apk del cmake

mv -v googletest DMK/extern/nda/deps/GTest
mv -v itertools DMK/extern/nda/deps/itertools
mv -v mpi DMK/extern/nda/deps/mpi

if [[ "${target}" == *-apple-* ]]; then
    as_needed=-Wl,-dead_strip_dylibs
else
    as_needed=-Wl,--as-needed
fi

cuda_args=()
if [[ "${bb_full_target}" != *cuda+none* ]]; then
    # nvcc writes to /tmp, which is a small tmpfs in the sandbox
    export TMPDIR=${WORKSPACE}/tmpdir
    mkdir -v ${TMPDIR}
    cuda_args=(
        -DDMK_GPU_OFFLOAD=ON
        -DCMAKE_CUDA_COMPILER=${prefix}/cuda/bin/nvcc
        -DCMAKE_CUDA_HOST_COMPILER=${CXX}
        -DCMAKE_CUDA_FLAGS="-L${prefix}/cuda/lib"
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
    )
fi

# nda's TRIQS mpi has FindMPI read the library version string, which needs try_run (impossible when
# cross-compiling); it only uses the string to pick mpiexec flags for its tests. It also links
# MPI::MPI_CXX, so as_needed drops the unused libmpicxx, which an MPICH configured with --disable-cxx lacks.
cmake -S DMK -B build \
    -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_FIND_ROOT_PATH=${prefix} \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI_C_COMPILER=${bindir}/mpicc \
    -DMPI_CXX_COMPILER=${bindir}/mpicxx \
    -DMPI_CXX_SKIP_MPICXX=ON \
    -DMPI_RUN_RESULT_CXX_libver_mpi_normal=0 \
    -DMPI_RUN_RESULT_CXX_libver_mpi_normal__TRYRUN_OUTPUT= \
    -DCMAKE_SHARED_LINKER_FLAGS=${as_needed} \
    -DBLA_VENDOR=libblastrampoline \
    -DDMK_BUILD_EXAMPLES=OFF \
    -DFINUFFT_ARCH_FLAGS= \
    "${cuda_args[@]}"
cmake --build build --parallel ${nproc}
cmake --install build
# Julia loads only libdmk.so. The CMake config names the archive and dmk.pc holds sandbox MPI paths.
rm -r ${libdir}/libdmk.a ${libdir}/cmake ${libdir}/pkgconfig
# The audit looks for licenses under the JLL's name
mv -v ${prefix}/share/licenses/dmk ${prefix}/share/licenses/DMK
"""

augment_platform_block = """
    using Base.BinaryPlatforms

    module __CUDA
        $(CUDA.augment)
    end

    $(MPI.augment)

    $(MicroArchitectures.augment)

    function augment_platform!(platform::Platform)
        augment_mpi!(platform)
        @static if Sys.ARCH === :x86_64
            augment_microarchitecture!(platform)
        end
        __CUDA.augment_platform!(platform)
    end
    """

# SCTL's vector width is fixed at compile time, hence the microarchitecture expansion.
platforms = filter(p -> (Sys.islinux(p) && libc(p) == "glibc" && arch(p) in ("x86_64", "aarch64")) ||
                        (Sys.isapple(p) && arch(p) == "aarch64"), supported_platforms())
platforms = filter(p -> cxxstring_abi(p) != "cxx03", expand_cxxstring_abis(platforms))
platforms = expand_microarchitectures(platforms, ["x86_64", "avx", "avx2", "avx512"])
foreach(p -> p["cuda"] = "none", platforms)
platforms, mpi_dependencies = MPI.augment_platforms(platforms)

# An artifact serves every later minor of its CUDA major. 12.4 is the first nvcc that accepts GCC 13,
# 12.8 the first that targets Blackwell.
cuda_platforms = filter(p -> arch(p) == "x86_64" && p["cuda"] in ("12.4", "12.8", "13.0"), CUDA.supported_platforms(; min_version=v"12.4"))
cuda_platforms = filter(p -> cxxstring_abi(p) == "cxx11", expand_cxxstring_abis(cuda_platforms))
cuda_platforms = expand_microarchitectures(cuda_platforms, ["avx2", "avx512"])
cuda_platforms, cuda_mpi_dependencies = MPI.augment_platforms(cuda_platforms)

products = [
    LibraryProduct("libdmk", :libdmk),
]

dependencies = AbstractDependency[
    Dependency("CompilerSupportLibraries_jll"),
    Dependency("libblastrampoline_jll"; compat="5.4.0"),
    # clang's OpenMP runtime on macOS; clang emits calls that only a libomp of its own version provides
    Dependency("LLVMOpenMP_jll"; compat="23.1.1", platforms=filter(Sys.isapple, platforms)),
    HostBuildDependency(PackageSpec(; name="CMake_jll", version="3.31.9")),
]

# Don't look for mpiwrapper.so when the audit dlopens libdmk
ENV["MPITRAMPOLINE_DELAY_INIT"] = "1"

for platform in [platforms; cuda_platforms]
    should_build_platform(triplet(platform)) || continue

    cuda = platform["cuda"] != "none"
    platform_dependencies = AbstractDependency[dependencies; cuda ? cuda_mpi_dependencies : mpi_dependencies]
    if cuda
        append!(platform_dependencies, CUDA.required_dependencies(platform; static_sdk=true))
        push!(platform_dependencies, RuntimeDependency("CUDA_Compiler_jll"))
    end
    archs = !cuda ? "" : VersionNumber(platform["cuda"]) >= v"12.8" ? "80-real;90-real;100-real;120" : "80-real;90"
    platform_sources, platform_script = sources, "CUDA_ARCHS=\"$archs\"\n" * script
    if Sys.isapple(platform)
        # The default SDK's libc++ predates the C++20 <concepts>/<ranges> nda needs; 15.0 is the first SDK
        # Yggdrasil ships with its own libc++. 11.0 is the first macOS on Apple Silicon.
        platform_sources, platform_script = require_macos_sdk("15.0", sources, platform_script; deployment_target="11.0")
    end

    # The sandbox has no GPU driver, so the audit cannot dlopen a CUDA build.
    build_tarballs(ARGS, name, version, platform_sources, platform_script, [platform],
                   products, platform_dependencies;
                   augment_platform_block, julia_compat="1.10", preferred_gcc_version=v"13",
                   lazy_artifacts=true, dont_dlopen=cuda)
end

# Writes bindings/DMK_jll for the tarballs in a products directory, with artifacts downloaded from
# <url prefix>/<version>+<build>/. Run through ci/publish_jll.sh.
import SHA, TOML
url_prefix, products_dir = ARGS

# The recipe's per-platform objects, as Yggdrasil's registration step reads them
meta_json = tempname()
empty!(ARGS)
push!(ARGS, "--meta-json=$meta_json")
include(joinpath(@__DIR__, "build_tarballs.jl"))

objs = [BinaryBuilder.JSON.parse(line; dicttype=Dict) for line in eachline(meta_json)]
foreach(BinaryBuilder.cleanup_merged_object!, objs)
built = Set(m[:platform_triplet] for m in match.(BinaryBuilder.TARBALL_FILENAME_REGEX, readdir(products_dir)) if m !== nothing)
filter!(obj -> triplet(only(obj["platforms"])) in built, objs)
# Project.toml takes the dependencies of the last object, and a CUDA build's are a superset.
sort!(objs; by = obj -> only(obj["platforms"])["cuda"] != "none")

# A checkout downloads exactly the tarballs its Artifacts.toml pins, so new content for a published file takes the
# next build number, and with it a new URL directory.
jll_dir = joinpath(dmk_root, "bindings", "DMK_jll")
build = 0
if isfile(joinpath(jll_dir, "Project.toml"))
    old = VersionNumber(TOML.parsefile(joinpath(jll_dir, "Project.toml"))["version"])
    if VersionNumber(old.major, old.minor, old.patch) == version
        published = Dict(basename(d["url"]) => d["sha256"]
                         for a in TOML.parsefile(joinpath(jll_dir, "Artifacts.toml"))[name] for d in a["download"])
        changed = any(published) do (f, h)
            p = joinpath(products_dir, f)
            isfile(p) && bytes2hex(open(SHA.sha256, p)) != h
        end
        build = only(old.build) + changed
    end
end
build_version = VersionNumber("$(version)+$(build)")
upload_prefix = "$url_prefix/$build_version"

BinaryBuilder.devdir[] = joinpath(dmk_root, "bindings")
for (i, obj) in enumerate(objs)
    BinaryBuilder.rebuild_jll_package(obj; download_dir=products_dir, upload_prefix, build_version,
                                      from_scratch=i == 1)
end
println("DMK_jll $build_version: copy $(joinpath(products_dir, "$name.v$version.*.tar.gz")) to the directory serving ",
        upload_prefix)

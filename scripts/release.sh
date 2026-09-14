#!/usr/bin/env bash
# Cut a release: stamp VERSION.txt, sync the binding manifests, commit and tag. Does not push.
set -euo pipefail

usage() { echo "usage: ${0##*/} <major.minor.patch>" >&2; exit 2; }
[ $# -eq 1 ] || usage
version=$1
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || usage

cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is dirty; commit or stash before releasing" >&2
    exit 1
fi
if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
    echo "tag v$version already exists" >&2
    exit 1
fi

readme_notices=$(awk '/^# Third-party notices/{f=1} f' README.md 2>/dev/null)
if [ -n "$readme_notices" ] && [ "$readme_notices" != "$(cat THIRD_PARTY_NOTICES.md)" ]; then
    echo "README.md third-party section has drifted from THIRD_PARTY_NOTICES.md" >&2
    exit 1
fi

if grep -q '^## \[Unreleased\]' CHANGELOG.md; then
    echo "CHANGELOG.md still has an [Unreleased] heading; retitle it to [$version]" >&2
    exit 1
fi
if ! grep -q "^## \[$version\]" CHANGELOG.md; then
    echo "CHANGELOG.md has no [$version] section" >&2
    exit 1
fi

# Read the Docs overrides docs/conf.py on tag builds, so it only has to pin the right X.Y series.
docs_version=$(sed -nE 's/^version = release = "([^"]+)".*/\1/p' docs/conf.py)
if [ "$docs_version" != "$version" ] && [ "$docs_version" != "${version%.*}" ]; then
    echo "docs/conf.py version '$docs_version' is neither $version nor ${version%.*}" >&2
    exit 1
fi

echo "$version" > VERSION.txt
git add VERSION.txt

# Tracked-ness, not existence, decides what gets stamped: the bindings are developed in the tree
# but released through BinaryBuilder, so an untracked manifest must not be swept into the release
# commit. Stamping resumes on its own if one is ever committed.
for manifest in bindings/DMK.jl/Project.toml bindings/python/pyproject.toml; do
    git ls-files --error-unmatch "$manifest" >/dev/null 2>&1 || continue
    sed -i.bak -E "s/^version *= *\".*\"/version = \"$version\"/" "$manifest"
    rm -f "$manifest.bak"
    if ! grep -q "^version = \"$version\"$" "$manifest"; then
        echo "could not stamp the version into $manifest; nothing committed or tagged" >&2
        exit 1
    fi
    git add "$manifest"
done

git commit -qm "release: v$version"
git tag -a "v$version" -m "v$version"

echo "tagged v$version. push with:"
echo "  git push origin $(git rev-parse --abbrev-ref HEAD) v$version"

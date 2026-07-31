#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="goss-build-base:local"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the base image if it does not exist yet.
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    docker build --target build-base -t "${IMAGE_TAG}" "${REPO_ROOT}"
fi

# Run the given command inside the container with the repo mounted.
docker run --rm -t \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    "${IMAGE_TAG}" \
    bash -c "${*:-bash}"

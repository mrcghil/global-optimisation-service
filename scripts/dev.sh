#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="goss-build-base:local"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# On machines behind a Zscaler TLS-inspection proxy the container needs the
# corporate root CA injected so git/curl can reach GitHub over HTTPS.  Default
# to true on this dev machine; override with INJECT_ZSCALER_CA=false in
# environments without TLS inspection (CI outside the corporate network, etc.).
INJECT_ZSCALER_CA="${INJECT_ZSCALER_CA:-true}"

# Build the base image if it does not exist yet.
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    docker build --target build-base \
        --build-arg "INJECT_ZSCALER_CA=${INJECT_ZSCALER_CA}" \
        -t "${IMAGE_TAG}" "${REPO_ROOT}"
fi

# Run the given command inside the container with the repo mounted.
docker run --rm -t \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    "${IMAGE_TAG}" \
    bash -c "${*:-bash}"

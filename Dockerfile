# Dockerfile
FROM ubuntu:24.04 AS build-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ gcc make cmake git ca-certificates libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: inject a corporate TLS-inspection root CA (e.g. Zscaler) so
# git/curl can reach GitHub over HTTPS when the host intercepts TLS.
# Disabled by default — only activated when INJECT_ZSCALER_CA=true so the
# image stays portable for teammates and CI runners outside the corporate
# network.  Set to "true" on any machine behind a Zscaler proxy.
ARG INJECT_ZSCALER_CA=false
COPY zscaler-root-ca.crt /tmp/zscaler-root-ca.crt
RUN if [ "${INJECT_ZSCALER_CA}" = "true" ]; then \
        cp /tmp/zscaler-root-ca.crt /usr/local/share/ca-certificates/zscaler-root-ca.crt \
        && update-ca-certificates; \
    fi

# CppAD (headers + optional lib) from source
ARG CPPAD_TAG=20240000.7
RUN git clone --depth 1 --branch ${CPPAD_TAG} \
        https://github.com/coin-or/CppAD.git /tmp/cppad \
    && cmake -S /tmp/cppad -B /tmp/cppad/build \
        -Dcppad_prefix=/usr/local -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/cppad/build --target install \
    && rm -rf /tmp/cppad

# CppADCodeGen (header-only install)
RUN git clone --depth 1 https://github.com/joaoleal/CppADCodeGen.git /tmp/cppadcg \
    && cmake -S /tmp/cppadcg -B /tmp/cppadcg/build \
        -DCMAKE_INSTALL_PREFIX=/usr/local -DGOOGLETEST_GIT=OFF \
    && cmake --build /tmp/cppadcg/build --target install \
    && rm -rf /tmp/cppadcg

WORKDIR /work

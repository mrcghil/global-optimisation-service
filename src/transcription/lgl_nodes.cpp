#include "goss/transcription/lgl_nodes.hpp"

#include <cmath>
#include <string>

namespace goss::transcription {

namespace {

/// Evaluate the Legendre polynomial P_{k}(x) and its derivative P'_{k}(x)
/// via the three-term recurrence relation:
///   P_0(x) = 1,  P_1(x) = x
///   P_{k+1}(x) = ((2k+1)*x*P_k(x) - k*P_{k-1}(x)) / (k+1)
/// The derivative satisfies the same recurrence displaced by one index:
///   P'_{j+1}(x) = (2j+1)*P_j(x) + P'_{j-1}(x)
///
/// Returns {P_k(x), P'_k(x)}.
std::pair<double, double> legendre_and_deriv(std::size_t k, double x) {
    if (k == 0) {
        return {1.0, 0.0};
    }
    if (k == 1) {
        return {x, 1.0};
    }
    double p_prev = 1.0;   // P_{j-1}
    double p_curr = x;     // P_j
    double dp_prev = 0.0;  // P'_{j-1}
    double dp_curr = 1.0;  // P'_j
    for (std::size_t j = 1; j < k; ++j) {
        // Recurrence: P_{j+1}(x) = ((2j+1)*x*P_j - j*P_{j-1}) / (j+1)
        const double p_next =
            (static_cast<double>(2 * j + 1) * x * p_curr
             - static_cast<double>(j) * p_prev)
            / static_cast<double>(j + 1);
        // Derivative recurrence: P'_{j+1}(x) = (2j+1)*P_j(x) + P'_{j-1}(x)
        const double dp_next =
            static_cast<double>(2 * j + 1) * p_curr + dp_prev;
        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }
    return {p_curr, dp_curr};
}

}  // namespace

void lgl_nodes_and_weights(std::size_t n,
                           std::vector<double>& nodes_out,
                           std::vector<double>& weights_out) {
    if (n < 2) {
        throw TranscriptionError(
            "lgl_nodes_and_weights requires n >= 2, got n = "
            + std::to_string(n));
    }

    nodes_out.resize(n);
    weights_out.resize(n);

    // Endpoints are always ±1.
    nodes_out[0] = -1.0;
    nodes_out[n - 1] = 1.0;

    if (n == 2) {
        // w_i = 2 / (n*(n-1)) = 2 / (2*1) = 1 for both endpoints.
        weights_out[0] = 1.0;
        weights_out[1] = 1.0;
        return;
    }

    // The polynomial degree is N = n-1.  The interior nodes are the n-2 roots of
    // P'_N(x).  We seed with Chebyshev-Gauss-Lobatto interior points (which are
    // good approximations to the LGL nodes) and refine by Newton's method.
    //
    // CGL interior seeds (ascending order): cos(j*pi/(n-1)), j = n-2, n-3, …, 1.
    // So for the k-th interior node (k=0 is leftmost/most negative):
    //   seed = cos((n-2-k)*pi/(n-1))
    //
    // Because nodes are symmetric about 0, we compute only the first half
    // (left side, including the centre if n is odd) and mirror the rest.
    const std::size_t degree = n - 1;
    const std::size_t n_interior = n - 2;
    // ceil(n_interior / 2): how many nodes to compute directly
    const std::size_t half = (n_interior + 1) / 2;
    const double pi = std::acos(-1.0);

    for (std::size_t k = 0; k < half; ++k) {
        // Seed with the k-th CGL interior point (most negative first).
        double x = std::cos(static_cast<double>(n_interior - k) * pi
                            / static_cast<double>(n - 1));

        // Newton iteration: find root of P'_N(x).
        // Use P''_N from the identity: (1-x²)*P''_N = 2x*P'_N - N*(N+1)*P_N.
        for (int iter = 0; iter < 100; ++iter) {
            const auto [p_n, dp_n] = legendre_and_deriv(degree, x);
            if (std::abs(dp_n) < 1e-15) {
                // Already at a root of P'_N.
                break;
            }
            const double denom = 1.0 - x * x;
            double ddp_n = 0.0;
            if (std::abs(denom) > 1e-15) {
                ddp_n = (2.0 * x * dp_n
                         - static_cast<double>(degree)
                               * static_cast<double>(degree + 1) * p_n)
                        / denom;
            } else {
                // Near the endpoints ±1, use the known formula:
                // P''_N(±1) = (1/12)*N*(N-1)*(N+1)*(N+2) * (±1)^N  (not needed
                // in practice since interior nodes are never near ±1).
                break;
            }
            if (std::abs(ddp_n) < 1e-15) {
                break;
            }
            const double dx = dp_n / ddp_n;
            x -= dx;
            if (std::abs(dx) < 1e-15 * (1.0 + std::abs(x))) {
                break;
            }
        }

        // Store this node at index k+1 (interior nodes start at index 1).
        nodes_out[k + 1] = x;
        // Mirror: the symmetric node about 0 is at index n-2-k.
        // When k == n_interior/2 and n_interior is odd (the centre node, x≈0),
        // both indices are the same and we write 0 twice — that is harmless.
        nodes_out[n - 2 - k] = -x;
    }

    // Force the exact midpoint to 0 when n is odd (n_interior is odd), avoiding
    // tiny rounding errors from the Newton iteration near 0.
    if (n_interior % 2 == 1) {
        const std::size_t mid_idx = n_interior / 2 + 1;  // (n-1)/2 among all n nodes
        nodes_out[mid_idx] = 0.0;
    }

    // Compute weights: w_i = 2 / (n*(n-1) * [P_N(x_i)]²).
    // At the endpoints P_N(±1) = ±1, so w_endpoint = 2/(n*(n-1)).
    const double weight_scale =
        2.0 / (static_cast<double>(n) * static_cast<double>(n - 1));

    weights_out[0] = weight_scale;
    weights_out[n - 1] = weight_scale;

    for (std::size_t i = 1; i < n - 1; ++i) {
        const auto [p_val, dp_val] = legendre_and_deriv(degree, nodes_out[i]);
        (void)dp_val;  // derivative not needed for weights
        weights_out[i] = weight_scale / (p_val * p_val);
    }
}

std::vector<double> lgl_differentiation_matrix(const std::vector<double>& nodes) {
    const std::size_t n = nodes.size();
    if (n < 2) {
        throw TranscriptionError(
            "lgl_differentiation_matrix requires at least 2 nodes, got "
            + std::to_string(n));
    }

    const std::size_t degree = n - 1;
    std::vector<double> D(n * n, 0.0);

    // Precompute P_N(x_i) at every node.
    std::vector<double> p_vals(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto [p_val, dp_val] = legendre_and_deriv(degree, nodes[i]);
        (void)dp_val;
        p_vals[i] = p_val;
    }

    // Off-diagonal entries:
    //   D[i,j] = P_N(x_i) / (P_N(x_j) * (x_i - x_j)),  i ≠ j
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) {
                D[i * n + j] =
                    p_vals[i] / (p_vals[j] * (nodes[i] - nodes[j]));
            }
        }
    }

    // Exact corner diagonal entries:
    //   D[0,0]     = -(N*(N+1))/4
    //   D[N,N]     = +(N*(N+1))/4
    const double corner =
        static_cast<double>(degree) * static_cast<double>(degree + 1) / 4.0;
    D[0] = -corner;
    D[n * n - 1] = corner;

    // Interior diagonal entries: D[i,i] = -sum_{j≠i} D[i,j]  (partition of unity)
    for (std::size_t i = 1; i < n - 1; ++i) {
        double off_diag_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j != i) {
                off_diag_sum += D[i * n + j];
            }
        }
        D[i * n + i] = -off_diag_sum;
    }

    return D;
}

}  // namespace goss::transcription

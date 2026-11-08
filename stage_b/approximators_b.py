"""
Stage B approximators: M0, M0I, M0F, M1, M2 -- estimate the body-level
disturbance theta (6-D).

Common interface:
    estimate(xi, emag)        -> theta_hat in R^6
    adapt_rate(xi, s, emag)   -> flat weight derivative
    pack() / unpack(w)        -> flat weight vector
    get_weights / set_weights -> copy out / load (warm start)
    lyapunov_weight_term()    -> weight part of V

    xi   : feature vector (adaptive.features())
    s    : Lyapunov driving signal = -E^T P e_tilde in R^6
    emag : regressor magnitude ||e|| (used only by M0 and M0F)

Driving signal (derived in adaptive.py):
    theta_hat_dot = -gamma E^T P e_tilde = gamma s  cancels the cross term,
    so Vdot = -e_tilde^T Q_p e_tilde <= 0.

References:
    M0  Sombolestan & Nguyen, IEEE T-RO 40 (2024), Eqs. (27)/(30), (40)/(41)
    M1  Lewis, Yesildirek & Liu, IEEE TNN 7(2), 1996 + sigma-modification
    M2  Patil, Le, Greene & Dixon, IEEE L-CSS 6 (2022), Eqs. (3)-(8)
    Projection: Lavretsky & Gibson 2011, Appendix E
"""

import numpy as np

N_THETA = 6          # body-level disturbance: 3 linear + 3 angular accelerations


# ---------------------------------------------------------------------------
# Projection operator
# ---------------------------------------------------------------------------

def projection(theta, y, bound):
    """
    Ball projection (Lavretsky & Gibson 2011, App. E).

    Keeps ||theta|| <= bound invariant by removing the outward component on
    the boundary. Satisfies (theta_hat - theta*)^T (Proj(theta_hat, y) - y)
    <= 0, so it does not break the Lyapunov argument.
    """
    n = np.linalg.norm(theta)
    if n < bound:
        return y
    outward = float(theta @ y)
    if outward <= 0.0:
        return y
    return y - outward * theta / (n ** 2)


# ===========================================================================
# M0 -- hand-written regressor:  theta_hat = alpha_hat*||e|| + beta_hat
# ===========================================================================

class M0_LinearRegressor:
    """
    Parameterisation of the main paper. Regressor [||e||, 1]; two parameter
    vectors in R^6, i.e. 12 parameters. Assumes the structure of the
    uncertainty is known.
    """

    name = "M0 (alpha*||e|| + beta)"

    def __init__(self, n_in=None, n_out=N_THETA, gamma=50.0, bound=500.0,
                 seed=0):
        self.n = n_out
        self.gamma = gamma
        self.bound = bound
        rng = np.random.default_rng(seed)
        self.alpha = rng.uniform(0.0, 0.01, size=n_out)
        self.beta = rng.uniform(0.0, 0.01, size=n_out)

    @property
    def n_weights(self):
        return 2 * self.n

    def get_weights(self):
        """All adapted parameters as a flat float64 vector (copy of pack())."""
        return np.array(self.pack(), dtype=np.float64).copy()

    def set_weights(self, w):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != self.n_weights:
            raise ValueError(f"{self.name}: got {w.size} weights, "
                             f"expected {self.n_weights}")
        self.unpack(w.copy())

    def pack(self):
        return np.concatenate([self.alpha, self.beta])

    def unpack(self, w):
        self.alpha = w[: self.n].copy()
        self.beta = w[self.n:].copy()

    def estimate(self, xi, emag):
        return self.alpha * emag + self.beta

    def adapt_rate(self, xi, s, emag):
        y_a = self.gamma * s * emag
        y_b = self.gamma * s
        return np.concatenate([
            projection(self.alpha, y_a, self.bound),
            projection(self.beta, y_b, self.bound),
        ])

    def lyapunov_weight_term(self):
        return (self.alpha @ self.alpha + self.beta @ self.beta) / self.gamma


# ===========================================================================
# M0I -- pure integral action:  theta_hat = beta_hat  (first rung of the ladder)
# ===========================================================================

class M0I_PureIntegral(M0_LinearRegressor):
    """
    M0 without the alpha*||e|| term: theta_hat = beta_hat, 6 parameters,
    beta_dot = gamma * s. Tests whether M0 is in effect pure integral action.

    Initial beta is bit-identical to M0's: the same rng draws alpha first
    and then beta, and alpha is discarded here.
    """

    name = "M0I (beta, pure integral)"

    def __init__(self, n_in=None, n_out=N_THETA, gamma=50.0, bound=500.0,
                 seed=0):
        super().__init__(n_in, n_out, gamma, bound, seed)
        self.alpha = np.zeros(n_out)     # term not present

    @property
    def n_weights(self):
        return self.n

    def pack(self):
        return self.beta.copy()

    def unpack(self, w):
        self.beta = w.copy()

    def estimate(self, xi, emag):
        return self.beta.copy()

    def adapt_rate(self, xi, s, emag):
        return projection(self.beta, self.gamma * s, self.bound)

    def lyapunov_weight_term(self):
        return (self.beta @ self.beta) / self.gamma


# ===========================================================================
# M0F -- phase-aware classical regressor
#   theta_hat = alpha*||e|| + beta + c*sin(2 pi phi) + d*cos(2 pi phi)
# ===========================================================================

class M0F_PhaseRegressor(M0_LinearRegressor):
    """
    M0 plus the gait phase. Regressor [||e||, 1, sin phi, cos phi], 24
    parameters. Still classical adaptive control: linearly parameterised,
    regressor written by hand by someone who knows the disturbance is
    gait-periodic. Tests whether M2's advantage on the swinging payload
    comes from universal approximation or only from knowing the phase.

    Each linear parameter uses the same gradient law
        c_dot = gamma * s * sin(phi),  d_dot = gamma * s * cos(phi),
    so the Lyapunov cross term cancels as for M0. sin/cos are the last two
    entries of features() and are unscaled (O(1)). The extra parameters
    start at zero so the seed affects alpha/beta exactly as in M0.
    """

    name = "M0F (alpha*||e|| + beta + phase)"

    def __init__(self, n_in=None, n_out=N_THETA, gamma=50.0, bound=500.0,
                 seed=0):
        super().__init__(n_in, n_out, gamma, bound, seed)
        self.c = np.zeros(n_out)
        self.d = np.zeros(n_out)

    @property
    def n_weights(self):
        return 4 * self.n

    def pack(self):
        return np.concatenate([self.alpha, self.beta, self.c, self.d])

    def unpack(self, w):
        n = self.n
        self.alpha = w[:n].copy()
        self.beta = w[n:2*n].copy()
        self.c = w[2*n:3*n].copy()
        self.d = w[3*n:].copy()

    @staticmethod
    def _sc(xi):
        return float(xi[-2]), float(xi[-1])

    def estimate(self, xi, emag):
        sn, cs = self._sc(xi)
        return self.alpha * emag + self.beta + self.c * sn + self.d * cs

    def adapt_rate(self, xi, s, emag):
        sn, cs = self._sc(xi)
        g = self.gamma
        return np.concatenate([
            projection(self.alpha, g * s * emag, self.bound),
            projection(self.beta, g * s, self.bound),
            projection(self.c, g * s * sn, self.bound),
            projection(self.d, g * s * cs, self.bound),
        ])

    def lyapunov_weight_term(self):
        return (self.alpha @ self.alpha + self.beta @ self.beta
                + self.c @ self.c + self.d @ self.d) / self.gamma


# ===========================================================================
# M1 -- RBF network:  theta_hat = W_hat^T phi(xi)
# ===========================================================================

class M1_RBF:
    """
    Radial-basis-function network, linear in its output weights:

        theta_hat(xi) = W_hat^T phi(xi),
        phi_i(xi) = exp( -||xi/scale - c_i||^2 / (2 width^2) )

    Update with sigma-modification (prevents weight drift without
    persistent excitation; the error converges to a small neighbourhood):

        W_hat_dot = Gamma phi(xi) s^T - sigma Gamma W_hat

    sigma was swept in verify_adaptive.py test 4b (constant disturbance,
    excited reference, 20 s): 0 -> error 0.060, 0.0005 -> 0.066,
    0.002 -> 0.130, 0.01 -> 0.468. 0.0005 keeps almost the sigma=0 accuracy
    and still protects against drift.

    Centres: a grid is impossible in 11 dimensions, so centres are placed
    at random (seeded) in the normalised box [-1.5, 1.5]^d.

    Bias term: without it the steady-state features sit near zero, all
    Gaussian bases stay at phi ~ 0.08 and none is excited. M0 has a
    constant term and M2 has bias neurons, so M1 gets one too (phi_0 = 1).
    """

    name = "M1 (RBF)"

    def __init__(self, n_in, n_out=N_THETA, n_centers=64, width=1.0,
                 gamma=30.0, sigma=0.0005, scale=None, seed=0,
                 centers=None, bias=True):
        self.n_in = n_in
        self.n = n_out
        self.width = width
        self.gamma = gamma
        self.sigma = sigma
        self.scale = np.ones(n_in) if scale is None else np.asarray(scale, float)
        rng = np.random.default_rng(seed)
        if centers is None:
            # random in the normalised box [-1.5, 1.5]^d
            self.centers = rng.uniform(-1.5, 1.5, size=(n_centers, n_in))
        else:
            self.centers = np.asarray(centers, float)
        self.bias = bool(bias)
        # L = number of Gaussians (+1 constant term)
        self.L = self.centers.shape[0] + (1 if self.bias else 0)
        self.W = rng.uniform(0.0, 0.01, size=(self.L, n_out))

    @property
    def n_weights(self):
        return self.L * self.n

    def get_weights(self):
        """All adapted parameters as a flat float64 vector (copy of pack())."""
        return np.array(self.pack(), dtype=np.float64).copy()

    def set_weights(self, w):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != self.n_weights:
            raise ValueError(f"{self.name}: got {w.size} weights, "
                             f"expected {self.n_weights}")
        self.unpack(w.copy())

    def pack(self):
        return self.W.ravel().copy()

    def unpack(self, w):
        self.W = w.reshape(self.L, self.n).copy()

    def _phi(self, xi):
        z = np.asarray(xi, float) / self.scale
        d2 = np.sum((self.centers - z) ** 2, axis=1)
        g = np.exp(-d2 / (2.0 * self.width ** 2))
        return np.concatenate([[1.0], g]) if self.bias else g

    def estimate(self, xi, emag=None):
        return self.W.T @ self._phi(xi)

    def adapt_rate(self, xi, s, emag=None):
        phi = self._phi(xi)
        Wdot = self.gamma * np.outer(phi, s) - self.sigma * self.gamma * self.W
        return Wdot.ravel()

    def lyapunov_weight_term(self):
        return float(np.sum(self.W * self.W)) / self.gamma


# ===========================================================================
# M2 -- Lyapunov-based deep neural network (Lb-DNN)
# ===========================================================================

class M2_LbDNN:
    """
    Patil, Le, Greene & Dixon, IEEE L-CSS 2022.

    Architecture (Eq. 3):
        Phi(xi, V0..Vk) = Vk^T phi_k o ... o V1^T phi_1 ( V0^T xia ),
        xia = [xi^T, 1]^T
    Recursion (Eq. 4):
        Phi_0 = V0^T xia,   Phi_j = Vj^T phi_j(Phi_{j-1}),  j = 1..k

    Update (Eqs. 7-8) with driving signal s:
        vec(V0_dot) = proj( G0 [ (prod_{l=1..k} Vl^T phi'_l)(I_{L1} kron xia^T) ]^T s )
        vec(Vj_dot) = proj( Gj [ (prod_{l=j+1..k} Vl^T phi'_l)(I_{Lj+1} kron phi_j^T) ]^T s )

    This is not backpropagation: the weight derivative is chosen to cancel
    the cross term of the Lyapunov derivative. There is no data set, no
    batch update and no learning-rate schedule. The Kronecker Jacobian is
    checked by finite differences in verify_adaptive.py.
    """

    name = "M2 (Lb-DNN)"

    def __init__(self, n_in, n_out=N_THETA, n_hidden=2, width=8,
                 activation="tanh", gamma=10.0, bound=5000.0, seed=0,
                 switch_gain=True, switch_thresh=None, gamma_low=None,
                 scale=None):
        self.n_in = n_in
        self.n = n_out
        self.k = n_hidden
        self.activation = activation
        self.gamma = gamma
        self.bound = bound
        self.switch_gain = switch_gain
        self.scale = np.ones(n_in) if scale is None else np.asarray(scale, float)

        # Layer sizes: L0 = n_in+1 (augmented), ..., L_{k+1} = n_out
        self.L = [n_in + 1] + [width] * self.k + [n_out]

        rng = np.random.default_rng(seed)
        # Paper: inner layers U(0, 0.5), output layer U(0, 0.01)
        self.V = []
        for j in range(self.k + 1):
            hi = 0.01 if j == self.k else 0.5
            self.V.append(rng.uniform(0.0, hi, size=(self.L[j], self.L[j+1])))
        self._shapes = [v.shape for v in self.V]
        self._sizes = [v.size for v in self.V]

        # Switched gain. The paper uses Gamma = 10 I if ||V_tilde|| <= 5,
        # else I, where V_tilde is the (bounded) weight error. V* is unknown,
        # so the observable ||V_hat|| is used as a proxy. That proxy grows
        # monotonically, so with a fixed threshold the gain locks at the low
        # value. Hence: threshold relative to the initial norm (3 * w0), and
        # gamma_low = gamma / 10 (keeps the paper's 10:1 ratio).
        w0 = np.sqrt(sum(float(np.sum(v * v)) for v in self.V))
        self.switch_thresh = (3.0 * w0 if switch_thresh is None
                              else float(switch_thresh))
        self.gamma_low = (gamma / 10.0 if gamma_low is None
                          else float(gamma_low))
        self.w0 = w0

    @property
    def n_weights(self):
        return sum(self._sizes)

    def get_weights(self):
        """All adapted parameters as a flat float64 vector (copy of pack())."""
        return np.array(self.pack(), dtype=np.float64).copy()

    def set_weights(self, w):
        w = np.asarray(w, dtype=np.float64).ravel()
        if w.size != self.n_weights:
            raise ValueError(f"{self.name}: got {w.size} weights, "
                             f"expected {self.n_weights}")
        self.unpack(w.copy())

    def pack(self):
        return np.concatenate([v.ravel() for v in self.V])

    def unpack(self, w):
        idx = 0
        for j, (shp, sz) in enumerate(zip(self._shapes, self._sizes)):
            self.V[j] = w[idx: idx + sz].reshape(shp).copy()
            idx += sz

    # --- activation ----------------------------------------------------
    def _act(self, y):
        if self.activation == "lrelu":
            out = np.where(y >= 0.0, y, 0.01 * y)
        else:
            out = np.tanh(y)
        out = out.copy()
        out[-1] = 1.0                 # bias neuron
        return out

    def _act_grad(self, y):
        if self.activation == "lrelu":
            dd = np.where(y >= 0.0, 1.0, 0.01)
        else:
            dd = 1.0 - np.tanh(y) ** 2
        dd = dd.copy()
        dd[-1] = 0.0                  # bias neuron has zero gradient
        return np.diag(dd)

    # --- forward pass --------------------------------------------------
    def _forward(self, xi):
        z = np.asarray(xi, float) / self.scale
        xia = np.concatenate([z, [1.0]])
        Phi_list = [self.V[0].T @ xia]
        phi_list, dphi_list = [], []
        for j in range(1, self.k + 1):
            y = Phi_list[j - 1]
            phi_list.append(self._act(y))
            dphi_list.append(self._act_grad(y))
            Phi_list.append(self.V[j].T @ phi_list[-1])
        return Phi_list[-1], Phi_list, phi_list, dphi_list, xia

    def estimate(self, xi, emag=None):
        return self._forward(xi)[0]

    # --- adaptation ----------------------------------------------------
    def _omega(self, j, phi_list, dphi_list):
        """
        Omega_j = prod_{l=j+1..k} (V_l^T phi'_l), right to left.
        Empty product (j = k) is the identity, n_out x n_out.
        """
        M = np.eye(self.L[self.k + 1])
        for l in range(self.k, j, -1):
            M = M @ (self.V[l].T @ dphi_list[l - 1])
        return M

    def jacobian_block(self, xi, j):
        """
        Lambda_j = Omega_j (I_{L_{j+1}} kron v^T),  (n_out, L_{j+1}*L_j);
        d(theta_hat)/d(vec(V_j^T)). Used by verify_adaptive.py.
        """
        _, _, phi_list, dphi_list, xia = self._forward(xi)
        Omega = self._omega(j, phi_list, dphi_list)
        v = xia if j == 0 else phi_list[j - 1]
        return np.kron(Omega, v.reshape(1, -1))

    def adapt_rate(self, xi, s, emag=None):
        _, _, phi_list, dphi_list, xia = self._forward(xi)
        rates = []
        gam = self.gamma
        if self.switch_gain:
            w_norm = np.sqrt(sum(float(np.sum(v * v)) for v in self.V))
            if w_norm > self.switch_thresh:
                gam = self.gamma_low
        for j in range(self.k + 1):
            Omega = self._omega(j, phi_list, dphi_list)
            v = xia if j == 0 else phi_list[j - 1]
            Lam = np.kron(Omega, v.reshape(1, -1))
            g = gam * (Lam.T @ s)
            # vec is row-major; Lambda's columns follow V_j^T, so transpose
            # back before flattening
            g = g.reshape(self.L[j + 1], self.L[j]).T.ravel()
            rates.append(projection(self.V[j].ravel(), g, self.bound))
        return np.concatenate(rates)

    def lyapunov_weight_term(self):
        return sum(float(np.sum(v * v)) for v in self.V) / self.gamma


# ---------------------------------------------------------------------------

def make(kind, n_in, **kw):
    """Factory: 'M0' | 'M0I' | 'M0F' | 'M1' | 'M2'."""
    table = {"M0": M0_LinearRegressor, "M0I": M0I_PureIntegral,
             "M0F": M0F_PhaseRegressor,
             "M1": M1_RBF, "M2": M2_LbDNN}
    if kind not in table:
        raise ValueError(f"unknown approximator: {kind}")
    return table[kind](n_in=n_in, **kw)

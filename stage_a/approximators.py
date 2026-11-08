"""
Asama A - Uc fonksiyon yaklasiklayici: M0, M1, M2.

Hepsi ayni arayuzu uygular:
    estimate(x, xd, e)  -> Phi_hat in R^n        (belirsizlik kestirimi)
    adapt_rate(x, xd, e)-> agirlik turevi        (uyarlama kurali)
    pack()/unpack()     -> agirliklari duz vektore cevirme (RK4 icin)
    lyapunov_weight_term() -> V'nin agirlik parcasi

Kaynaklar:
    M0  Sombolestan & Nguyen, IEEE T-RO 2024, denklem (30), (40), (41)
    M1  Lewis, Yesildirek & Liu, IEEE TNN 7(2), 1996 + sigma-modifikasyonu
    M2  Patil, Le, Greene & Dixon, IEEE L-CSS 6, 2022, denklem (3)-(8)
"""

import numpy as np
from plant import projection


# ============================================================================
# M0 - Ana makalenin parametrelendirmesi:  theta_hat = alpha_hat*||e|| + beta_hat
# ============================================================================

class M0_LinearRegressor:
    """
    Sombolestan & Nguyen (2024), denklem (30):
        theta_hat = alpha_hat * ||e|| + beta_hat

    Regresor yalnizca [||e||, 1]. Iki parametre vektoru, her biri R^n.

    Uyarlama kurallari makalenin (40)-(41) denklemlerinin bu tesise uyarlanmasi.
    Orada izleme hatasi yerine model-arasi fark e_tilde kullaniliyor ve
    y_alpha = -B^T P e_tilde ||e||, y_beta = -B^T P e_tilde.
    Burada referans model olmadigi icin dogrudan e kullaniliyor; isaret,
    Lyapunov turevindeki capraz terimlerin gotrulmesinden geliyor
    (asagidaki verify_lyapunov.py bunu sayisal olarak dogrular).
    """

    name = "M0 (alpha*||e|| + beta)"

    def __init__(self, n=2, gamma=50.0, bound=500.0, seed=0):
        self.n = n
        self.gamma = gamma
        self.bound = bound
        rng = np.random.default_rng(seed)
        self.alpha = rng.uniform(0.0, 0.01, size=n)
        self.beta = rng.uniform(0.0, 0.01, size=n)

    @property
    def n_weights(self):
        return 2 * self.n

    def pack(self):
        return np.concatenate([self.alpha, self.beta])

    def unpack(self, w):
        self.alpha = w[: self.n].copy()
        self.beta = w[self.n :].copy()

    def estimate(self, x, xd, e):
        return self.alpha * np.linalg.norm(e) + self.beta

    def adapt_rate(self, x, xd, e):
        e_norm = np.linalg.norm(e)
        y_a = self.gamma * e * e_norm
        y_b = self.gamma * e
        return np.concatenate([
            projection(self.alpha, y_a, self.bound),
            projection(self.beta, y_b, self.bound),
        ])

    def lyapunov_weight_term(self, alpha_star=None, beta_star=None):
        """V'nin agirlik parcasi: (1/gamma)*(||alpha||^2 + ||beta||^2).
        Ideal degerler bilinmedigi icin tam hata normu yerine agirlik normu."""
        return (self.alpha @ self.alpha + self.beta @ self.beta) / self.gamma


# ============================================================================
# M1 - RBF agi:  Phi_hat = W_hat^T * phi(z)
# ============================================================================

class M1_RBF:
    """
    Radyal tabanli fonksiyon agi, cikista lineer parametrelendirilmis:
        Phi_hat(z) = W_hat^T phi(z),   phi_i(z) = exp(-||z - c_i||^2 / (2*s_i^2))

    Uyarlama kurali (Lyapunov turevinden, sigma-modifikasyonlu):
        W_hat_dot = Gamma * phi(z) * e^T - sigma * Gamma * W_hat

    sigma-modifikasyonu kalici uyarim olmadiginda agirlik kaymasini engeller
    (Ioannou & Kokotovic). Bedeli: hata tam sifira degil, kucuk bir komsuluga
    yakinsar -> SGUUB.

    Merkezler: bu asamada izgara. Quadruped asamasinda on simulasyon verisine
    k-means ile yerlestirilecek (boyut laneti notu, proje dokumani).
    """

    name = "M1 (RBF)"

    def __init__(self, n=2, n_centers_per_dim=5, z_range=(-3.0, 3.0),
                 width=1.0, gamma=30.0, sigma=0.01, seed=0):
        self.n = n
        grid = np.linspace(z_range[0], z_range[1], n_centers_per_dim)
        # Girdi z = x (2 boyut). Izgara merkezleri.
        cx, cy = np.meshgrid(grid, grid, indexing="ij")
        self.centers = np.stack([cx.ravel(), cy.ravel()], axis=1)  # (L, 2)
        self.L = self.centers.shape[0]
        self.width = width
        self.gamma = gamma
        self.sigma = sigma
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(0.0, 0.01, size=(self.L, n))

    @property
    def n_weights(self):
        return self.L * self.n

    def pack(self):
        return self.W.ravel().copy()

    def unpack(self, w):
        self.W = w.reshape(self.L, self.n).copy()

    def _phi(self, z):
        d2 = np.sum((self.centers - z) ** 2, axis=1)
        return np.exp(-d2 / (2.0 * self.width ** 2))

    def estimate(self, x, xd, e):
        # Girdi olarak xd kullaniliyor: Lb-DNN ile ADIL kiyas icin.
        # Dixon makalesi agi Phi(xd, V...) seklinde xd ile besliyor; f(x)-f(xd)
        # farki rho(||e||)e terimiyle karsilaniyor.
        return self.W.T @ self._phi(xd)

    def adapt_rate(self, x, xd, e):
        phi = self._phi(xd)
        Wdot = self.gamma * np.outer(phi, e) - self.sigma * self.gamma * self.W
        return Wdot.ravel()

    def lyapunov_weight_term(self):
        return float(np.sum(self.W * self.W)) / self.gamma


# ============================================================================
# M2 - Lyapunov tabanli derin sinir agi (Lb-DNN)
# ============================================================================

class M2_LbDNN:
    """
    Patil, Le, Greene & Dixon, IEEE L-CSS 2022.

    Mimari (denklem 3):
        Phi(xd, V0..Vk) = Vk^T phi_k o ... o V1^T phi_1 ( V0^T xda )
        xda = [xd^T, 1]^T   (bias icin genisletilmis)

    Ozyinelemeli gosterim (denklem 4):
        Phi_0 = V0^T xda
        Phi_j = Vj^T phi_j(Phi_{j-1}),  j = 1..k

    Uyarlama kurallari (denklem 7-8):
        vec(V0_dot) = proj( G0 * [ (prod_{l=1..k} Vl^T phi'_l) (I_{L1} kron xda^T) ]^T e )
        vec(Vj_dot) = proj( Gj * [ (prod_{l=j+1..k} Vl^T phi'_l)(I_{Lj+1} kron phi_j^T) ]^T e )

    prod sagdan sola: prod_{p=1..m} A_p = A_m ... A_2 A_1.

    vec(.) satir-oncelikli (makalenin notasyonu), numpy'nin varsayilan C duzeni.

    Aktivasyon: LReLU (makalede DNN1/DNN3) veya tanh (DNN2/DNN4).
    Makale LReLU'nun kalici durumda daha iyi oldugunu buluyor; gradyani
    sureksiz oldugu icin duzgun olmayan Lyapunov analizi gerekiyor.
    """

    name = "M2 (Lb-DNN)"

    def __init__(self, n=2, n_hidden=2, width=6, activation="lrelu",
                 gamma=10.0, bound=5000.0, seed=0,
                 switch_gain=True, switch_thresh=5.0, gamma_low=1.0):
        """
        switch_gain : makalenin anahtarlamali kazanc kurali.
            "The adaptation gain ... is selected using the switched rule:
             Gamma_j = 10*I if ||[vec(V0_tilde) ... vec(Vk_tilde)]|| <= 5,
             and Gamma_j = I otherwise."
            Makale agirlik HATASI V_tilde kullaniyor; bu simulasyonda ideal
            agirliklar bilinmedigi icin gozlenebilir ||V_hat|| vekili kullanildi.
            Amac ayni: agirliklar buyurken kazanci dusurup iraksamayi onlemek.
        """
        self.n = n
        self.k = n_hidden                      # gizli katman sayisi
        self.activation = activation
        self.gamma = gamma
        self.bound = bound
        self.switch_gain = switch_gain
        self.switch_thresh = switch_thresh
        self.gamma_low = gamma_low

        # Katman boyutlari: L0 = n+1 (genisletilmis), ..., L_{k+1} = n
        self.L = [n + 1] + [width] * self.k + [n]

        rng = np.random.default_rng(seed)
        # Makale: ic katmanlar U(0,0.5), cikis katmani U(0,0.01)
        self.V = []
        for j in range(self.k + 1):
            hi = 0.01 if j == self.k else 0.5
            self.V.append(rng.uniform(0.0, hi, size=(self.L[j], self.L[j + 1])))

        self._shapes = [v.shape for v in self.V]
        self._sizes = [v.size for v in self.V]

    @property
    def n_weights(self):
        return sum(self._sizes)

    def pack(self):
        return np.concatenate([v.ravel() for v in self.V])

    def unpack(self, w):
        idx = 0
        for j, (shp, sz) in enumerate(zip(self._shapes, self._sizes)):
            self.V[j] = w[idx: idx + sz].reshape(shp).copy()
            idx += sz

    # --- aktivasyon ve gradyani ------------------------------------------
    def _act(self, y):
        """phi_j(y): son eleman bias icin 1'e sabitlenir."""
        if self.activation == "lrelu":
            out = np.where(y >= 0.0, y, 0.01 * y)
        else:
            out = np.tanh(y)
        out = out.copy()
        out[-1] = 1.0
        return out

    def _act_grad(self, y):
        """phi'_j(y) = d phi_j / d y, kosegen matris. Bias satiri sifir."""
        if self.activation == "lrelu":
            d = np.where(y >= 0.0, 1.0, 0.01)
        else:
            d = 1.0 - np.tanh(y) ** 2
        d = d.copy()
        d[-1] = 0.0
        return np.diag(d)

    # --- ileri gecis -------------------------------------------------------
    def _forward(self, xd):
        """
        Donenler:
            Phi      : ag cikisi, R^n
            Phi_list : [Phi_0, ..., Phi_k]  (ara aktivasyon ONCESI degerler)
            phi_list : [phi_1(Phi_0), ..., phi_k(Phi_{k-1})]
            dphi_list: [phi'_1(Phi_0), ..., phi'_k(Phi_{k-1})]
            xda      : genisletilmis girdi
        """
        xda = np.concatenate([xd, [1.0]])
        Phi_list = [self.V[0].T @ xda]            # Phi_0
        phi_list = []
        dphi_list = []
        for j in range(1, self.k + 1):
            y = Phi_list[j - 1]
            phi_list.append(self._act(y))
            dphi_list.append(self._act_grad(y))
            Phi_list.append(self.V[j].T @ phi_list[-1])
        return Phi_list[-1], Phi_list, phi_list, dphi_list, xda

    def estimate(self, x, xd, e):
        Phi, *_ = self._forward(xd)
        return Phi

    # --- uyarlama kurallari ------------------------------------------------
    def _omega(self, j, phi_list, dphi_list):
        """
        Omega_j = prod_{l=j+1..k} ( V_l^T phi'_l )   (sagdan sola)
                = (Vk^T phi'_k) ... (V_{j+1}^T phi'_{j+1})
        Bos carpim (j = k) icin birim matris, boyut L_{k+1} x L_{k+1} = n x n.
        """
        M = np.eye(self.L[self.k + 1])
        for l in range(self.k, j, -1):
            # phi'_l -> dphi_list[l-1],  V_l -> self.V[l]
            M = M @ (self.V[l].T @ dphi_list[l - 1])
        return M

    def adapt_rate(self, x, xd, e):
        _, Phi_list, phi_list, dphi_list, xda = self._forward(xd)
        rates = []

        gam = self.gamma
        if self.switch_gain:
            w_norm = np.sqrt(sum(float(np.sum(v * v)) for v in self.V))
            if w_norm > self.switch_thresh:
                gam = self.gamma_low

        for j in range(self.k + 1):
            Omega = self._omega(j, phi_list, dphi_list)   # (n, L_{j+1})
            # Girdi vektoru: j=0 ise xda, degilse phi_j
            v = xda if j == 0 else phi_list[j - 1]        # R^{L_j}
            # Lambda_j = Omega_j (I_{L_{j+1}} kron v^T) :  (n, L_{j+1}*L_j)
            Lam = np.kron(Omega, v.reshape(1, -1))
            g = gam * (Lam.T @ e)                  # R^{L_{j+1}*L_j}
            # vec satir-oncelikli: V_j in R^{L_j x L_{j+1}} -> vec uzunlugu ayni
            # Lambda sutun sirasi (L_{j+1}, L_j) -> V_j^T duzenine karsilik gelir
            g = g.reshape(self.L[j + 1], self.L[j]).T.ravel()
            Vj_flat = self.V[j].ravel()
            rates.append(projection(Vj_flat, g, self.bound))

        return np.concatenate(rates)

    def lyapunov_weight_term(self):
        return sum(float(np.sum(v * v)) for v in self.V) / self.gamma

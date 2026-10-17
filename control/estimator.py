"""
Оценка состояния — инерциальный EKF по схеме PX4 EKF2 (РЕШ-32; docs/control.md, раздел 9.3).

    StateEstimator(sensor_params).update(meas, dt) -> est     — звено `estimator` класса SAU

Фильтр с вектором ошибок (error-state EKF). IMU — не измерение, а вход прогноза
(инерциальное счисление); ориентацию, скорость и положение исправляют GPS, барометр,
магнитометр (три оси, поле Земли известно), воздушная скорость и синтетическое
скольжение β = 0.

Номинальное состояние: q — кватернион поворота связанная СК → NED (Гамильтон, [w, x, y, z]),
v — скорость NED, м/с; p — положение NED, м (p_D = −h); b_g — смещение гироскопа, рад/с;
b_a — смещение акселерометра, м/с²; w — ветер (w_N, w_E), м/с.

Прогноз (на шаге IMU Δt; ω_изм, f_изм — гироскоп и акселерометр):
    ω = ω_изм − b_g,  f = f_изм − b_a,  R = R(q)
    q ← q ⊗ Exp(ω·Δt),   v ← v + (R·f + g)·Δt,   p ← p + v·Δt,   g = (0, 0, g)
Вектор ошибок δx = [δθ, δv, δp, δb_g, δb_a, δw] (17). Ошибка ориентации — в осях NED:
R_ист = Exp(δθ)·R. Линеаризация:
    δθ' = −R·δb_g − R·n_g
    δv' = −[R·f]×·δθ − R·δb_a − R·n_a
    δp' = δv,   δb_g' = n_bg,   δb_a' = n_ba,   δw' = n_w
    P ← F·P·Fᵀ + Q,  F = I + A·Δt,  Q = diag((σ_g·Δt)², (σ_a·Δt)², 0, q_bg·Δt, q_ba·Δt, q_w·Δt)
σ_g, σ_a — шум гироскопа и акселерометра на отсчёт; q_bg = 2σ_b²/τ_b — интенсивность
дрейфа гироскопа (марковский процесс датчика). Все — из SensorParams.

Коррекция (по группам, K = P·Hᵀ·S⁻¹, P — форма Джозефа), затем перенос ошибки в
номинальное состояние: q ← Exp(δθ) ⊗ q, остальное — сложение.
Отбраковка (как EKF2_*_GATE): отношение e_i² / (gate²·S_ii) > 1 хотя бы по одной
составляющей — группа не используется. Защита от блокировки: группа отбраковывается
дольше lockout_s — измерение принимается без проверки (фильтр выставляется по нему). Сброс к GPS (как EKF2_NOAID_TOUT): и скорость, и
положение GPS отбраковываются дольше gps_reset_s — фильтр без коррекции, сброс обоих;
только положение (скорость принимается) — это выброс GPS, положение сбрасывается
лишь через gps_pos_reset_s.

Запаздывание (как в PX4): фильтр работает на задержанном времени t − delay; отсчёты IMU
копятся в буфере, каждое измерение используется в момент своего отсчёта (по отметке
времени t_gps, t_h, t_Va, t_mag). Текущая оценка — повторный прогноз номинального
состояния по буферу IMU от задержанного времени до текущего.
"""

from dataclasses import dataclass

import numpy as np

G = 9.81
I3 = np.eye(3)

# Индексы вектора ошибок
TH, VEL, POS, BG, BA, WND = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12),
                             slice(12, 15), slice(15, 17))
N_ERR = 17


# ---------------------------------------------------------------------------
# Кватернионы и повороты
# ---------------------------------------------------------------------------

def skew(a):
    """Кососимметричная матрица: skew(a)·b = a × b."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def quat_mult(a, b):
    """Произведение Гамильтона a ⊗ b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_exp(rv):
    """Кватернион поворота на вектор rv (ось · угол, рад)."""
    a = np.linalg.norm(rv)
    if a < 1e-12:
        return np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]])
    s = np.sin(0.5 * a) / a
    return np.array([np.cos(0.5 * a), s * rv[0], s * rv[1], s * rv[2]])


def quat_to_R(q):
    """Матрица поворота связанная СК → NED."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_to_quat(phi, theta, psi):
    """Углы Эйлера (ψ → θ → φ) → кватернион."""
    cf, sf = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cp, sp = np.cos(psi / 2), np.sin(psi / 2)
    return np.array([cf * ct * cp + sf * st * sp,
                     sf * ct * cp - cf * st * sp,
                     cf * st * cp + sf * ct * sp,
                     cf * ct * sp - sf * st * cp])


def R_to_euler(R):
    """Матрица связанная → NED → (φ, θ, ψ). При |θ| → 90° φ и ψ не определены, но конечны."""
    theta = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    return np.arctan2(R[2, 1], R[2, 2]), theta, np.arctan2(R[1, 0], R[0, 0])


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Модели измерений: h(x) и H = ∂h/∂δx (по вектору ошибок).
# Проверяются конечными разностями в checks/check_estimator.py.
# ---------------------------------------------------------------------------

def meas_mag(q, m_ned):
    """
    Магнитометр по трём осям: m_b = Rᵀ·m_n. R_ист = Exp(δθ)·R → m_b ≈ Rᵀ·m_n + Rᵀ·[m_n]×·δθ.
    (Курс из магнитометра через оценённый наклон, как в режиме «heading» EKF2, усиливает
    ошибку наклона в tg(наклонения) ≈ 3.3 раза и коррелирован с ней — поэтому три оси.)
    """
    R = quat_to_R(q)
    H = np.zeros((3, N_ERR))
    H[:, TH] = R.T @ skew(m_ned)
    return R.T @ m_ned, H


def air_body(q, v, w):
    """Воздушная скорость в связанной СК a = Rᵀ·(v − w), w = (w_N, w_E, 0), и ∂a/∂δx."""
    R = quat_to_R(q)
    va = v - np.array([w[0], w[1], 0.0])
    a = R.T @ va
    da = np.zeros((3, N_ERR))
    da[:, TH] = R.T @ skew(va)          # R_ист = Exp(δθ)·R → a_ист ≈ a + Rᵀ·[va]×·δθ
    da[:, VEL] = R.T
    da[:, WND] = -R.T[:, :2]
    return a, da


def meas_airspeed(q, v, w):
    """Va = |v − w|; H = (v − w)ᵀ/Va по δv и −(v − w)_NE/Va по δw."""
    va = v - np.array([w[0], w[1], 0.0])
    Va = max(np.linalg.norm(va), 1e-3)
    H = np.zeros((1, N_ERR))
    H[0, VEL] = va / Va
    H[0, WND] = -va[:2] / Va
    return np.array([Va]), H


def meas_sideslip(q, v, w):
    """Синтетическое скольжение β ≈ a_y / a_x (измерение — 0, EKF2_FUSE_BETA)."""
    a, da = air_body(q, v, w)
    ax = a[0] if abs(a[0]) > 1.0 else np.copysign(1.0, a[0])
    H = ((da[1] * ax - a[1] * da[0]) / ax**2).reshape(1, -1)
    return np.array([a[1] / ax]), H


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

@dataclass
class EstimatorParams:
    """Настройки, которых нет в SensorParams. Шумы датчиков берутся из SensorParams."""

    delay:         float = 0.25    # с, задержка горизонта фильтра (≥ запаздывания GPS)
    gps_pos_sigma: float = 2.0     # м — СКО положения GPS для фильтра (как hAcc приёмника;
                                   #     коррелированный уход GPS фильтр не наблюдает)
    q_ba:          float = 1e-6    # (м/с²)²/с — блуждание смещения акселерометра
    q_wind:        float = 0.5     # (м/с)²/с — блуждание ветра (турбулентность; NEES ветра ≈ 1…3)
    beta_sigma:    float = 0.3     # рад — синтетическое β = 0 (EKF2_BETA_NOISE)
    gate_gps:      float = 5.0     # σ
    gate_baro:     float = 5.0
    gate_mag:      float = 5.0
    gate_tas:      float = 3.0
    gate_beta:     float = 5.0
    gps_reset_s:   float = 5.0     # с — скорость и положение GPS отбраковываются дольше → сброс
    gps_pos_reset_s: float = 20.0  # с — только положение отбраковывается дольше → сброс положения
    lockout_s:     tuple = (("mag", 2.0), ("baro", 5.0), ("tas", 5.0))
                                   # с — группа отбраковывается дольше → измерение принимается
                                   #     без проверки (иначе ошибка больше порога не исправится)
    use_airdata:   bool = True     # воздушная скорость и β = 0 (оценка ветра)


# ---------------------------------------------------------------------------
# Фильтр
# ---------------------------------------------------------------------------

class StateEstimator:
    """Инерциальный EKF для SAU: update(meas, dt) -> est."""

    GROUPS = ("gps_pos", "gps_vel", "baro", "mag", "tas", "beta")

    def __init__(self, sensor_params, params: EstimatorParams = None):
        self.sp, self.pr = sensor_params, params or EstimatorParams()
        sp = self.sp
        I, D = np.radians(sp.mag_incl), np.radians(sp.mag_decl)
        self.decl = D
        self.mag_ned = sp.mag_field * np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), np.sin(I)])
        self.q_bg = 2.0 * sp.gyro_drift_sigma**2 / sp.gyro_drift_tau     # (рад/с)²/с
        self.ready = False
        self.stats = {}            # по группам: [использовано, отбраковано]
        self.test_ratio = {}       # последнее отношение невязки к порогу
        self.n_gps_reset = 0

    # --- инициализация ------------------------------------------------------
    def _init(self, m):
        sp, pr = self.sp, self.pr
        for k in ("ax", "mx", "vn", "h"):
            if k not in m:
                raise ValueError(f"StateEstimator: нет измерения '{k}' (нужен RealisticSensors с accel_fn)")
        f = np.array([m["ax"], m["ay"], m["az"]])
        phi = np.arctan2(-f[1], -f[2])                    # наклон — по «кажущейся вертикали»
        theta = np.arcsin(np.clip(f[0] / G, -1.0, 1.0))
        psi = self._mag_heading(m, phi, theta)
        self.q = euler_to_quat(phi, theta, psi)
        self.v = np.array([m["vn"], m["ve"], m["vd"]])
        self.p = np.array([m["pn"], m["pe"], -m["h"]])
        self.bg, self.ba = np.zeros(3), np.zeros(3)
        Va = max(m["Va"], 1.0)
        self.w = self.v[:2] - Va * np.array([np.cos(psi), np.sin(psi)])  # β = 0
        # P0 ориентации — с запасом на выставку в полёте (наклон по акселерометру при
        # скольжении или вираже ошибается на градусы, курс по магнитометру — в tg I раз больше)
        self.P = np.diag(np.r_[[0.1**2] * 2, [0.3**2],
                               [0.5**2] * 3,
                               [pr.gps_pos_sigma**2] * 2, [sp.baro_noise**2],
                               [sp.gyro_bias0**2 + sp.gyro_drift_sigma**2] * 3,
                               [sp.accel_bias0**2] * 3,
                               [2.0**2] * 2])
        self.t_f = m["t"]                      # время, к которому относится оценка фильтра
        self.imu = []                          # буфер IMU: (t, ω_изм, f_изм, Δt)
        self.obs = []                          # очередь измерений: (t_отсчёта, вид, данные)
        self.last = {}                         # отметки времени последних принятых отсчётов
        self.t_vel_ok = self.t_pos_ok = m["t"]     # последний принятый отсчёт GPS-скорости / положения
        self.stats = {k: [0, 0] for k in self.GROUPS}
        self.t_ok = {k: m["t"] for k in self.GROUPS}      # последний принятый отсчёт по группам
        self.n_forced = 0
        self.ready = True

    def _mag_heading(self, m, phi, theta):
        """Курс по магнитометру при заданном наклоне (для начальной выставки) + склонение."""
        mx, my, mz = m["mx"], m["my"], m["mz"]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(theta), np.cos(theta)
        bx = mx * ct + (my * sf + mz * cf) * st
        by = my * cf - mz * sf
        return _wrap(np.arctan2(-by, bx) + self.decl)

    # --- очередь измерений --------------------------------------------------
    def _queue(self, m):
        """Новые отсчёты (по отметкам времени) — в очередь к моменту их взятия."""
        for kind, key in (("gps", "t_gps"), ("baro", "t_h"), ("tas", "t_Va"), ("mag", "t_mag")):
            ts = m.get(key)
            if ts is not None and ts != self.last.get(kind):
                self.last[kind] = ts
                self.obs.append((ts, kind, dict(m)))
        self.obs.sort(key=lambda o: o[0])

    # --- шаг ---------------------------------------------------------------------
    def update(self, meas, dt):
        m = meas
        if not self.ready or m["t"] < self.t_last:          # первый шаг или сброс полёта
            self._init(m)
        self.t_last = m["t"]
        self.imu.append((m["t"], np.array([m["p"], m["q"], m["r"]]),
                         np.array([m["ax"], m["ay"], m["az"]]), dt))
        self._queue(m)

        # задержанный горизонт: прогноз по отсчётам IMU старше t − delay, затем измерения
        while self.imu and self.imu[0][0] <= m["t"] - self.pr.delay + 1e-9:
            t_i, om, f, dti = self.imu.pop(0)
            self._predict(om, f, dti)
            self.t_f = t_i + dti
            while self.obs and self.obs[0][0] <= self.t_f + 1e-9:
                self._fuse(*self.obs.pop(0))

        # текущая оценка: повторный прогноз по буферу (до текущего отсчёта)
        q, v, p = self.q, self.v, self.p
        for _, om, f, dti in self.imu[:-1]:
            q, v, p = self._propagate(q, v, p, om, f, dti)
        return self._output(m, q, v, p)

    # --- прогноз ---------------------------------------------------------------
    def _propagate(self, q, v, p, om, f, dt):
        """Номинальное состояние на шаг Δt по отсчёту IMU."""
        R = quat_to_R(q)
        a = R @ (f - self.ba) + np.array([0.0, 0.0, G])
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
        q = quat_mult(q, quat_exp((om - self.bg) * dt))
        return q / np.linalg.norm(q), v, p

    def error_dynamics(self, f):
        """Матрица A уравнения ошибок δx' = A·δx (f — показание акселерометра)."""
        R = quat_to_R(self.q)
        A = np.zeros((N_ERR, N_ERR))
        A[TH, BG] = -R
        A[VEL, TH] = -skew(R @ (f - self.ba))
        A[VEL, BA] = -R
        A[POS, VEL] = I3
        return A

    def _predict(self, om, f, dt):
        sp, pr = self.sp, self.pr
        F = np.eye(N_ERR) + self.error_dynamics(f) * dt
        Q = np.zeros(N_ERR)
        Q[TH] = (sp.gyro_noise * dt)**2
        Q[VEL] = (sp.accel_noise * dt)**2
        Q[BG] = self.q_bg * dt
        Q[BA] = pr.q_ba * dt
        Q[WND] = pr.q_wind * dt
        self.q, self.v, self.p = self._propagate(self.q, self.v, self.p, om, f, dt)
        self.P = F @ self.P @ F.T + np.diag(Q)
        self.P = 0.5 * (self.P + self.P.T)

    # --- коррекция ---------------------------------------------------------------
    def _fuse(self, t_obs, kind, m):
        sp, pr = self.sp, self.pr
        if kind == "gps":
            self._gps(m)
        elif kind == "baro":
            H = np.zeros((1, N_ERR))
            H[0, 8] = -1.0
            self._correct("baro", np.array([m["h"]]), np.array([-self.p[2]]), H,
                          np.array([sp.baro_noise**2]), pr.gate_baro)
        elif kind == "mag":
            h, H = meas_mag(self.q, self.mag_ned)
            self._correct("mag", np.array([m["mx"], m["my"], m["mz"]]), h, H,
                          np.full(3, sp.mag_noise**2), pr.gate_mag)
        elif kind == "tas" and pr.use_airdata and m["Va"] > 5.0:
            h, H = meas_airspeed(self.q, self.v, self.w)
            self._correct("tas", np.array([m["Va"]]), h, H,
                          np.array([sp.airspeed_noise**2]), pr.gate_tas)
            h, H = meas_sideslip(self.q, self.v, self.w)       # вместе с ПВД (EKF2_FUSE_BETA)
            self._correct("beta", np.zeros(1), h, H, np.array([pr.beta_sigma**2]), pr.gate_beta)

    def _gps(self, m):
        sp, pr = self.sp, self.pr
        H = np.zeros((3, N_ERR))
        H[:, VEL] = I3
        ok_v = self._correct("gps_vel", np.array([m["vn"], m["ve"], m["vd"]]), self.v, H,
                             np.full(3, sp.gps_vel_noise**2), pr.gate_gps)
        H = np.zeros((2, N_ERR))
        H[0, 6] = H[1, 7] = 1.0
        ok_p = self._correct("gps_pos", np.array([m["pn"], m["pe"]]), self.p[:2], H,
                             np.full(2, pr.gps_pos_sigma**2), pr.gate_gps)
        t = self.t_f
        if ok_v:
            self.t_vel_ok = t
        if ok_p:
            self.t_pos_ok = t
        no_aid = t - self.t_vel_ok > pr.gps_reset_s and t - self.t_pos_ok > pr.gps_reset_s
        if no_aid or t - self.t_pos_ok > pr.gps_pos_reset_s:
            self.p[:2] = [m["pn"], m["pe"]]
            self._reset_cov(np.arange(6, 8), pr.gps_pos_sigma**2)
            self.t_pos_ok = t
            if no_aid:
                self.v = np.array([m["vn"], m["ve"], m["vd"]])
                self._reset_cov(np.arange(3, 6), sp.gps_vel_noise**2)
                self.t_vel_ok = t
            self.n_gps_reset += 1

    def _reset_cov(self, idx, var):
        """Сброс состояний idx к измерению: корреляции — ноль, дисперсия — var."""
        self.P[idx, :] = 0.0
        self.P[:, idx] = 0.0
        self.P[idx, idx] = var

    def _correct(self, name, y, h, H, Rdiag, gate, wrap=False):
        """Коррекция группой: отбраковка по σ, K = P·Hᵀ·S⁻¹, Джозеф, перенос ошибки."""
        e = np.atleast_1d(y - h)
        if wrap:
            e = _wrap(e)
        R = np.diag(Rdiag)
        S = H @ self.P @ H.T + R
        ratio = e**2 / (gate**2 * np.diag(S))
        self.test_ratio[name] = float(ratio.max())
        lock = dict(self.pr.lockout_s).get(name)
        forced = lock is not None and self.t_f - self.t_ok[name] > lock
        if np.any(ratio > 1.0) and not forced:
            self.stats[name][1] += 1
            return False
        self.n_forced += bool(forced and np.any(ratio > 1.0))
        self.t_ok[name] = self.t_f
        self.stats[name][0] += 1
        K = self.P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(N_ERR) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(K @ e)
        return True

    def _inject(self, dx):
        """Перенос ошибки в номинальное состояние."""
        self.q = quat_mult(quat_exp(dx[TH]), self.q)
        self.q = self.q / np.linalg.norm(self.q)
        self.v = self.v + dx[VEL]
        self.p = self.p + dx[POS]
        self.bg = self.bg + dx[BG]
        self.ba = self.ba + dx[BA]
        self.w = self.w + dx[WND]

    # --- выход ---------------------------------------------------------------------
    def _output(self, m, q, v, p):
        phi, theta, psi = R_to_euler(quat_to_R(q))
        om = np.array([m["p"], m["q"], m["r"]]) - self.bg
        Va = meas_airspeed(q, v, self.w)[0][0] if self.pr.use_airdata else m["Va"]
        est = dict(m)
        est.update({"phi": phi, "theta": theta, "psi": psi,
                    "p": om[0], "q": om[1], "r": om[2],
                    "h": -p[2], "hdot": -v[2], "pn": p[0], "pe": p[1],
                    "Vg": float(np.hypot(v[0], v[1])), "chi": float(np.arctan2(v[1], v[0])),
                    "Va": float(Va), "wn": self.w[0], "we": self.w[1],
                    "b_gyro": self.bg.copy(), "b_acc": self.ba.copy(),
                    "vn": v[0], "ve": v[1], "vd": v[2], "quat": q})
        return est

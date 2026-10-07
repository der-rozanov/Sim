"""
Проверка САУ: переходные процессы на полной нелинейной модели 6DOF против
расчётных линейных (control/tuning.py, B&M гл. 6).

Регуляторы — классы control/controllers.py с коэффициентами по умолчанию
(рассчитаны design_sau для FPV-самолёта на 16 м/с). Датчики без шума — проверяется
расчёт, а не фильтрация. Ступеньки малые, чтобы не упираться в ограничения:
  1. тангаж  θc: +3°          — PitchController (каскад θ → q → δe)
  2. высота  hc: +5 м          — П-контур KH + тангаж + скорость
  3. скорость Vc: 16 → 18 м/с  — SpeedController
  4. крен    φ: 20° → 0        — RollHold (φ_ref = 0)
  5. курс    χc: +20°          — LateralController
  6. большой манёвр: курс +90° и высота +30 м одновременно (с насыщениями)
Метрики: перерегулирование и время установления (5 %) — модель и теория.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import THETA, Q, H, P, PHI, PSI, earth_velocity
from runner import run, compute_trim, trim_state
from control.controllers import (PitchController, PitchControlParams, SpeedController,
                                 SpeedControlParams, LateralController,
                                 LateralControlParams, RollHold, wrap_angle,
                                 AltitudeHold, AltitudeHoldParams)
from control.tuning import design_sau

ac  = AircraftParams()
VA  = 16.0
cfg = SimConfig(Va0=VA, h0=200.0, dt=0.01, t_end=20.0)
d   = design_sau(ac, VA)
k, gn = d.coeffs, d.gains
alpha_t, de_t, thr_t = k["alpha_trim"], k["de_trim"], k["thr_trim"]
s0  = trim_state(ac, cfg)
NOISE0 = SensorParams(gyro_noise=0.0, ins_angle_noise=0.0)


# ---------------------------------------------------------------------------
# Линейная теория: переходная функция TF num/den (numpy, Эйлер с мелким шагом)
# ---------------------------------------------------------------------------
def tf_step(num, den, t, amp=1.0):
    num, den = np.atleast_1d(num) / den[0], np.atleast_1d(den) / den[0]
    n = len(den) - 1
    num = np.r_[np.zeros(n + 1 - len(num)), num]
    A = np.zeros((n, n)); A[:-1, 1:] = np.eye(n - 1); A[-1] = -den[:0:-1]
    C = num[:0:-1] - num[0] * den[:0:-1]
    x, y, h = np.zeros(n), np.empty(len(t)), 1e-4
    ti = 0.0
    for i, tt in enumerate(t):
        while ti < tt - 1e-12:
            x = x + h * (A @ x + np.r_[np.zeros(n - 1), 1.0] * amp); ti += h
        y[i] = C @ x + num[0] * amp
    return y


def metrics(t, y, y0, y1):
    """Перерегулирование, % и время установления в трубку 5 % от ступеньки."""
    step = y1 - y0
    over = max(0.0, (np.max((y - y0) / step) - 1.0) * 100)
    out = np.where(np.abs(y - y1) > 0.05 * abs(step))[0]
    ts = t[out[-1]] if len(out) else 0.0
    return over, ts


# ---------------------------------------------------------------------------
# Замкнутый контур на полной модели
# ---------------------------------------------------------------------------
def closed_loop(theta_ref=None, h_ref=None, Va_ref=VA, chi_ref=None, state0=None, t_end=20.0):
    """h_ref, chi_ref, Va_ref — число или функция времени t."""
    f = lambda v, t: v(t) if callable(v) else v
    pitch = PitchController(ac, PitchControlParams())
    pitch.set_trim_elevator(de_t); pitch.set_trim_throttle(thr_t)
    speed = SpeedController(ac, SpeedControlParams())
    speed.set_trim_throttle(thr_t); speed.set_Va_ref(Va_ref)
    lat = LateralController(ac, LateralControlParams(beta_hold=True))
    hold = RollHold(ac, NOISE0, np.random.default_rng(0))
    alt = AltitudeHold(AltitudeHoldParams(), alpha_t)

    def fn(t, s, Va, al):
        speed.set_Va_ref(f(Va_ref, t))
        if h_ref is not None:
            th_c = alt.step(f(h_ref, t), s[H], cfg.dt)
        else:
            th_c = theta_ref
        pitch.theta_ref = th_c
        de = pitch.step(t, {'theta': s[THETA], 'q': s[Q], 'h': s[H], 'Va': Va}, cfg.dt)[0]
        dt_ = speed.step(Va, cfg.dt)
        if chi_ref is None:
            da, dr = hold.step(s[PHI], s[P], cfg.dt), 0.0
        else:
            Vx, Vy, _ = earth_velocity(s)
            lat.set_course(f(chi_ref, t))
            da, dr = lat.step({'chi': np.arctan2(Vy, Vx), 'phi': s[PHI], 'p': s[P],
                               'beta': np.arcsin(np.clip(s[6] / Va, -1, 1))}, cfg.dt)
        return np.array([de, dt_, da, dr])

    c = SimConfig(Va0=VA, h0=cfg.h0, dt=cfg.dt, t_end=t_end)
    return run(fn, ac, WindParams(), c, state0=s0.copy() if state0 is None else state0)


pitch_KH = PitchControlParams().KH
g_V = ac.g / VA
results = []

# 1. Тангаж
log = closed_loop(theta_ref=alpha_t + np.radians(3.0), t_end=4.0)
y = np.degrees(log.state[:, THETA] - alpha_t)
lin = tf_step([gn["kp_theta"] * k["a_theta3"]],
              [1, k["a_theta1"] + gn["kd_theta"] * k["a_theta3"], k["a_theta2"] + gn["kp_theta"] * k["a_theta3"]],
              log.t, 3.0)
results.append(("Тангаж θ, ° (θc +3°; теория B&M: α ≡ θ)", log.t, y, lin, 3.0 * gn["K_theta_DC"], 0.0))

# 2. Высота
log = closed_loop(h_ref=cfg.h0 + 5.0, t_end=15.0)
y = log.state[:, H] - cfg.h0
lin = tf_step([gn["kp_h"] * VA, gn["ki_h"] * VA], [1, gn["kp_h"] * VA, gn["ki_h"] * VA], log.t, 5.0)
results.append(("Высота Δh, м (ступенька +5 м)", log.t, y, lin, 5.0, 0.0))

# 3. Скорость
log = closed_loop(theta_ref=alpha_t, Va_ref=18.0, h_ref=cfg.h0, t_end=15.0)
y = log.Va - VA
lin = tf_step([gn["kp_V"] * k["a_V2"], gn["ki_V"] * k["a_V2"]],
              [1, k["a_V1"] + k["a_V2"] * gn["kp_V"], k["a_V2"] * gn["ki_V"]], log.t, 2.0)
results.append(("Скорость ΔVa, м/с (16 → 18)", log.t, y, lin, 2.0, 0.0))

# 4. Крен: из φ = 20° к 0 (RollHold) — та же ступенька −20°
s_r = s0.copy(); s_r[PHI] = np.radians(20.0)
log = closed_loop(theta_ref=alpha_t, h_ref=cfg.h0, state0=s_r, t_end=2.0)
y = np.degrees(log.state[:, PHI])
lin = 20.0 - tf_step([gn["kp_phi"] * k["a_phi2"], gn["ki_phi"] * k["a_phi2"]],
                     [1, k["a_phi1"] + gn["kd_phi"] * k["a_phi2"], gn["kp_phi"] * k["a_phi2"],
                      gn["ki_phi"] * k["a_phi2"]], log.t, 20.0)
results.append(("Крен φ, ° (из 20° к 0)", log.t, y, lin, 0.0, 20.0))

# 5. Курс
log = closed_loop(h_ref=cfg.h0, chi_ref=np.radians(20.0), t_end=12.0)
chi = np.degrees([np.arctan2(*earth_velocity(s)[1::-1]) for s in log.state])
lin = tf_step([gn["kp_chi"] * g_V, gn["ki_chi"] * g_V],
              [1, gn["kp_chi"] * g_V, gn["ki_chi"] * g_V], log.t, 20.0)
results.append(("Курс χ, ° (ступенька +20°)", log.t, chi, lin, 20.0, 0.0))

# 6. Большой манёвр (без теории — с насыщениями)
log6 = closed_loop(h_ref=cfg.h0 + 30.0, chi_ref=np.radians(90.0), t_end=25.0)

# Умолчания регуляторов = расчёт design_sau(AircraftParams(), 16 м/с)?
from dataclasses import asdict
bad = []
for name, obj, ref in (("PitchControlParams", PitchControlParams(), d.pitch),
                       ("SpeedControlParams", SpeedControlParams(), d.speed),
                       ("LateralControlParams", LateralControlParams(), d.lateral),
                       ("AltitudeHoldParams", AltitudeHoldParams(), d.altitude)):
    for f, v in asdict(ref).items():
        dv = getattr(obj, f)
        if isinstance(v, float) and not np.isclose(v, dv, rtol=0.01, atol=1e-6):
            bad.append(f"{name}.{f}: умолчание {dv:.4g}, расчёт {v:.4g}")

# ---------------------------------------------------------------------------
print("=" * 78)
print(f"Проверка САУ FPV-самолёта, Va = {VA} м/с (полная модель vs линейная теория)")
print("=" * 78)
print(f"{'контур':<36} {'перерег. мод/теор, %':>21} {'t_уст 5 % мод/теор, с':>22}")
for name, t, y, lin, y1, y0 in results:
    om, tm = metrics(t, y, y0, y1 if "Тангаж" not in name else y[-1])
    ol, tl = metrics(t, lin, y0, y1)
    print(f"{name:<36} {om:9.1f} / {ol:<9.1f} {tm:10.2f} / {tl:<9.2f}")
print("\nУмолчания регуляторов совпадают с расчётом control/tuning.py" if not bad
      else "\nРАСХОЖДЕНИЯ умолчаний с расчётом:\n  " + "\n  ".join(bad))
chi6 = np.degrees([np.arctan2(*earth_velocity(s)[1::-1]) for s in log6.state])
print(f"\nМанёвр: курс +90°, высота +30 м за 25 с: χ = {chi6[-1]:.1f}°, Δh = {log6.state[-1, H] - cfg.h0:.1f} м, "
      f"|φ|max = {np.degrees(np.abs(log6.state[:, PHI]).max()):.1f}°, Va ∈ [{log6.Va.min():.1f}, {log6.Va.max():.1f}] м/с, "
      f"α_max = {np.degrees(log6.alpha.max()):.1f}°")

# 7. Как в 3D-тренажёре: Va = 26 м/с, высота +100 м, затем −150 м
VA_G = 26.0
log7 = closed_loop(h_ref=lambda t: cfg.h0 + (100.0 if t < 60 else -50.0), Va_ref=VA_G, t_end=120.0)
i60 = np.searchsorted(log7.t, 60.0)
print(f"\nВысота на Va = {VA_G:.0f} м/с: при 60 с h = {log7.state[i60, H]:.1f} м (уставка {cfg.h0 + 100:.0f}), "
      f"в конце h = {log7.state[-1, H]:.1f} м (уставка {cfg.h0 - 50:.0f}),  α_max = {np.degrees(log7.alpha.max()):.1f}°")

# 8. Серия разворотов: +90°, −90° (180°), +45° — выход крена за ±30°
chi_seq = lambda t: np.radians(90.0 if t < 15 else (-90.0 if t < 35 else -45.0))
log8 = closed_loop(h_ref=cfg.h0, chi_ref=chi_seq, t_end=50.0)
phi8 = np.degrees(log8.state[:, PHI])
chi8 = np.degrees([np.arctan2(*earth_velocity(s)[1::-1]) for s in log8.state])
print(f"Развороты +90 → −90 → −45°: |φ|max = {np.abs(phi8).max():.1f}° (предел уставки 30°), "
      f"χ в конце = {chi8[-1]:.1f}°, |Δh|max = {np.abs(log8.state[:, H] - cfg.h0).max():.1f} м")

# ---------------------------------------------------------------------------
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898989"
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
for ax, (name, t, y, lin, y1, y0) in zip(axes.flat, results):
    ax.axhline(y1, color=GRAY, lw=1, ls=":")
    ax.plot(t, y, color=BLUE, lw=2, label="полная модель")
    ax.plot(t, lin, color=ORANGE, lw=2, ls="--", label="линейная теория")
    ax.set_title(name, fontsize=10); ax.set_xlabel("t, с")
    ax.grid(alpha=0.25)
axes[0, 0].legend(fontsize=8, frameon=False)
ax = axes[1, 2]
ax.plot(log6.t, chi6, color=BLUE, lw=2, label="курс χ, °")
ax.plot(log6.t, log6.state[:, H] - cfg.h0, color=ORANGE, lw=2, label="Δh, м")
ax.plot(log6.t, np.degrees(log6.state[:, PHI]), color="#1baf7a", lw=2, label="крен φ, °")
ax.set_title("Манёвр: χc +90° и hc +30 м", fontsize=10); ax.set_xlabel("t, с")
ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
fig.suptitle(f"САУ FPV-самолёта, Va = {VA} м/с: расчёт control/tuning.py vs полная модель 6DOF")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "sau_steps.png")
fig.savefig(out, dpi=120)
print(f"\nГрафик: {out}")
plt.show()

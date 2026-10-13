# -*- coding: utf-8 -*-
"""
С7: LQR — набор высоты → горизонтальный полёт → снижение.

Три фазы:
  0        – T_CRUISE  с — набор:       h_ref = H_CRUISE (150 м)
  T_CRUISE – T_DESCEND с — горизонт:   h_ref = H_CRUISE
  T_DESCEND – T_END    с — снижение:   h_ref = H_INIT   (100 м)

LQR-регулятор: численная линеаризация + DARE (scipy).
Измерения: гироскоп q, барометр h, СВС Va, ИНС theta.
u, w реконструируются из Va и theta (приближение без зонда УА).

Запуск:  python scenarios/s7_lqr_climb.py
         python scenarios/s7_lqr_climb.py results/s7.gif   -- сохранить анимацию
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.config import AircraftParams, WindParams, SensorParams, SimConfig
from sim.state import Q, THETA, H
from sim.wind import wind as _wind_base
from control.lqr import LQRController, LQRParams
from control.sensors import measure_gyro, measure_altitude, measure_airspeed
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import with_roll_hold   # САУ по крену (РЕШ-19)
from flight_logger import FlightLogger
from viz.viewer import show_flightlog


# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg = SimConfig(dt=0.01, t_end=145.0, Va0=30.0, h0=100.0)

VA_REF    = 16.0
H_INIT    = 100.0
H_CRUISE  = 150.0
T_CRUISE  = 50.0    # с, начало горизонтального полёта
T_DESCEND = 100.0   # с, начало снижения


# ------------------------------------------------------------------
# Балансировка
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, VA_REF)
s0            = trim_state(aircraft, cfg)
controls_trim = np.array([de_trim, thr_trim])

def wind_fn(h, t):
    return _wind_base(h, t, wind_params)

# ------------------------------------------------------------------
# LQR-регулятор
# ------------------------------------------------------------------
lqr_params = LQRParams(
    Q_diag=[1.0, 1.0, 10.0, 100.0],
    R_diag=[10.0, 1.0],
    h_Kp=0.05,
    h_ref=H_INIT,
    Va_ref=VA_REF,
)
lqr = LQRController(aircraft, s0, controls_trim, wind_fn, cfg.dt, lqr_params)
lqr.print_gains()

print(f"Трим: alpha={np.degrees(alpha_trim):.2f}°  "
      f"delta_e={np.degrees(de_trim):.2f}°  thr={thr_trim:.3f}")

# ------------------------------------------------------------------
# Лётный журнал
# ------------------------------------------------------------------
logger = FlightLogger(
    scenario="С7: LQR набор/горизонт/снижение",
    description=(f"LQR h_ref: {H_INIT:.0f}→{H_CRUISE:.0f}→{H_INIT:.0f} м, "
                 f"Va={VA_REF:.0f} м/с"),
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_CRUISE,  "label": "набор→горизонт",    "color": "steelblue"},
        {"t": T_DESCEND, "label": "горизонт→снижение",  "color": "darkorange"},
    ],
)

# ------------------------------------------------------------------
# Закон управления
# ------------------------------------------------------------------
rng           = np.random.default_rng(42)
h_ref_buf     = []
theta_ref_buf = []   # эффективная уставка тангажа из внешнего P-контура LQR

# Экспоненциальный фильтр барометра: h_filt = α·h_filt + (1−α)·h_meas
# α=0.9 при dt=0.01 с → постоянная времени τ = dt/(1-α) ≈ 0.1 с
H_FILT_ALPHA = 0.9
h_filt       = [s0[H]]   # список — мутабельный контейнер для замыкания

def controls_fn(t, state, Va, alpha):  # noqa: ARG001 — alpha required by runner interface
    h_ref = H_CRUISE if t < T_DESCEND else H_INIT
    lqr.set_altitude_ref(h_ref)
    h_ref_buf.append(h_ref)

    q_meas     = measure_gyro(state[Q],     sp.gyro_bias,     sp.gyro_noise,     rng)
    h_meas     = measure_altitude(state[H], sp.baro_bias,     sp.baro_noise,     rng)
    Va_meas    = measure_airspeed(Va,       sp.airspeed_bias, sp.airspeed_noise, rng)
    theta_meas = state[THETA]

    # Фильтрация высоты перед подачей во внешний P-контур
    h_filt[0] = H_FILT_ALPHA * h_filt[0] + (1.0 - H_FILT_ALPHA) * h_meas

    # Сохраняем эффективную уставку тангажа (зеркало внешнего контура LQR)
    theta_ref_eff = lqr.x_trim[3] + lqr.params.h_Kp * (h_ref - h_filt[0])
    theta_ref_eff = np.clip(theta_ref_eff, np.radians(-25.0), np.radians(25.0))
    theta_ref_buf.append(theta_ref_eff)

    u_meas = Va_meas * np.cos(theta_meas)
    w_meas = Va_meas * np.sin(theta_meas)

    meas = {
        "u": u_meas, "w": w_meas,
        "q": q_meas, "theta": theta_meas,
        "h": h_filt[0], "Va": Va_meas,
    }
    return lqr.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(with_roll_hold(controls_fn, aircraft, sp, cfg.dt), aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С7  LQR набор/горизонт/снижение")
path = logger.save(log, h_ref=h_ref_buf, theta_ref=theta_ref_buf)

# ------------------------------------------------------------------
# Метрики точности
# ------------------------------------------------------------------
t_arr = log.t
h_err = log.state[:, H] - np.array(h_ref_buf[:len(t_arr)])
mask_climb   = t_arr <  T_CRUISE
mask_cruise  = (t_arr >= T_CRUISE) & (t_arr < T_DESCEND)
mask_descent = t_arr >= T_DESCEND

print("\n  СКО удержания высоты по фазам:")
print(f"    Набор    (0–{T_CRUISE:.0f} с)           : {np.sqrt(np.mean(h_err[mask_climb]**2)):.2f} м")
print(f"    Горизонт ({T_CRUISE:.0f}–{T_DESCEND:.0f} с) : {np.sqrt(np.mean(h_err[mask_cruise]**2)):.2f} м")
print(f"    Снижение ({T_DESCEND:.0f}–{cfg.t_end:.0f} с) : {np.sqrt(np.mean(h_err[mask_descent]**2)):.2f} м")

# ------------------------------------------------------------------
# Показ (viz/viewer.py); аргумент командной строки — сохранить анимацию в .gif/.mp4
# ------------------------------------------------------------------
show_flightlog(path, speed=8.0,
               save_path=sys.argv[1] if len(sys.argv) > 1 else None)

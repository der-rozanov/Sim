# -*- coding: utf-8 -*-
"""
С4: Управляемый срыв и вывод — индикация закритического УА.

Сценарий:
  0 .. T_SWITCH  с → тримовый полёт
  T_SWITCH .. T_RECOVER с → тангаж 20°, газ=0 → alpha растёт → срыв
  T_RECOVER .. конец  → нос вниз −5°, газ=трим → вывод из срыва

Ключевой элемент: сигнал индикатора УА (0=НОРМ / 1=ПРЕД / 2=КРИТ / 3=СРЫВ)
вычисляется как функция alpha и отображается отдельным субплотом.
Фон info_box меняет цвет вместе с уровнем тревоги.

Запуск:  python scenarios/s4_stall.py
         python scenarios/s4_stall.py results/s4.gif
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import with_roll_hold   # САУ по крену (РЕШ-19)
from control.controllers import PitchController, PitchControlParams
from control.sensors import measure_gyro, measure_altitude, measure_airspeed
from sim.state import Q, H, euler
from flight_logger import FlightLogger
from viz.viewer import show_flightlog


# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg         = SimConfig(Va0=16.0, h0=300.0, theta0=0.0, dt=0.01, t_end=55.0)

THETA_POS     = np.radians(1.9)    # тримовый полёт (фаза 1)
THETA_NEG     = np.radians(20.0)   # тангаж + газ=0 → срыв (фаза 2)
THETA_RECOVER = np.radians(-5.0)   # нос вниз + газ → вывод из срыва (фаза 3)
T_SWITCH      = 8.0                # с, фаза 1 → 2
T_RECOVER     = 25.0               # с, фаза 2 → 3


# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

logger = FlightLogger(
    scenario="С4: Управляемый срыв",
    description="Трим → тангаж 20°/газ=0 → вывод −5°",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_SWITCH,  "label": "срыв",  "color": "red"},
        {"t": T_RECOVER, "label": "вывод", "color": "dodgerblue"},
    ],
)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")

# ------------------------------------------------------------------
# Контроллер
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.set_trim_elevator(de_trim)   # упреждающий балансировочный δe
controller.reset({'theta': euler(s0)[1], 'q': 0.0, 'h': cfg.h0})

rng = np.random.default_rng(seed=7)
theta_ref_buf = []

def controls_fn(t, state, Va, alpha):
    if t < T_SWITCH:
        # Фаза 1: тримовый полёт
        controller.set_trim_throttle(thr_trim)
        theta_ref = THETA_POS
    elif t < T_RECOVER:
        # Фаза 2: газ в ноль + агрессивный тангаж → срыв
        # Пропеллер тормозит → Va падает → alpha растёт до срыва
        controller.set_trim_throttle(0.0)
        theta_ref = THETA_NEG
    else:
        # Фаза 3: нос вниз + газ → вывод из срыва
        # Va растёт при снижении → CL восстанавливается → выход из срыва
        controller.set_trim_throttle(thr_trim)
        theta_ref = THETA_RECOVER

    controller.set_pitch_setpoint(theta_ref)
    theta_ref_buf.append(theta_ref)

    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = euler(state)[1] + rng.normal(0.0, sp.gyro_noise)
    h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    return controller.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(with_roll_hold(controls_fn, aircraft, sp, cfg.dt), aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С4  Срыв и вывод")
path = logger.save(log, theta_ref=theta_ref_buf)

alpha_all = np.degrees(log.alpha)

# ------------------------------------------------------------------
# Индикатор УА: 0=НОРМ, 1=ПРЕДУПРЕЖДЕНИЕ, 2=КРИТИЧЕСКИЙ, 3=СРЫВ
# ------------------------------------------------------------------
_warn_deg  = np.degrees(aircraft.alpha_warning)
_crit_deg  = np.degrees(aircraft.alpha_crit)
_stall_deg = np.degrees(aircraft.alpha_stall)

def _alert_level(a_deg):
    if a_deg >= _stall_deg:
        return 3
    if a_deg >= _crit_deg:
        return 2
    if a_deg >= _warn_deg:
        return 1
    return 0

alert_all = np.array([_alert_level(a) for a in alpha_all])

# Числовые итоги
t_above_warn  = np.sum(alert_all >= 1) * cfg.dt
t_above_crit  = np.sum(alert_all >= 2) * cfg.dt
t_above_stall = np.sum(alert_all >= 3) * cfg.dt
alpha_max     = alpha_all.max()

print(f"\n  Индикатор УА:")
print(f"    alpha_max            = {alpha_max:.2f} deg")
print(f"    t >= ПРЕД ({_warn_deg:.0f} deg) = {t_above_warn:.2f} s")
print(f"    t >= КРИТ ({_crit_deg:.0f} deg) = {t_above_crit:.2f} s")
print(f"    t >= СРЫВ ({_stall_deg:.0f} deg) = {t_above_stall:.2f} s")

# ------------------------------------------------------------------
# Показ (viz/viewer.py); аргумент командной строки — сохранить анимацию в .gif/.mp4
# ------------------------------------------------------------------
show_flightlog(path, speed=5.0,
               save_path=sys.argv[1] if len(sys.argv) > 1 else None)

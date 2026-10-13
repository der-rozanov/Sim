# -*- coding: utf-8 -*-
"""
С5: Контроль высоты — набор и снижение по уставке.

Структура управления:
  h_ref → [P_h] → theta_ref → [каскадный ПИД theta+q] → delta_e
  Va_ref → [ПИ Va] → throttle            (SpeedController, удержание крейсерской скорости)

Сценарий:
   0 .. T_CLIMB   с → h_ref = H_TRIM  (горизонтальный трим)
  T_CLIMB .. T_DESCEND с → h_ref = H_HIGH  (набор высоты)
  T_DESCEND .. конец   с → h_ref = H_TRIM  (снижение обратно)

Запуск:  python scenarios/s5_altitude_control.py
         python scenarios/s5_altitude_control.py results/s5.gif
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
from control.controllers import (PitchController, PitchControlParams,
                                 SpeedController, SpeedControlParams)
from control.sensors import measure_gyro, measure_altitude, measure_airspeed
from sim.state import THETA, Q, H
from flight_logger import FlightLogger
from viz.viewer import show_flightlog


# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg         = SimConfig(Va0=16.0, h0=100.0, theta0=0.0, dt=0.01, t_end=60.0)

H_TRIM    = 100.0   # начальная высота, м
H_HIGH    = 200.0   # уставка набора, м
T_CLIMB   = 10.0    # с, команда набора
T_DESCEND = 40.0    # с, команда снижения

# P-коэффициент внешнего контура: ошибка высоты → поправка тангажа
# 50 м ошибки → theta_ref = alpha_trim + 50*KH (ограничено ±15°)
KH = PitchControlParams().KH   # рад/м, расчёт control/tuning.py


# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

logger = FlightLogger(
    scenario="С5: Контроль высоты",
    description=f"Набор h={H_HIGH:.0f} м, Va={cfg.Va0:.0f} м/с",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_CLIMB,   "label": "набор",    "color": "orange"},
        {"t": T_DESCEND, "label": "снижение", "color": "dodgerblue"},
    ],
)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")

# ------------------------------------------------------------------
# Внутренний каскадный ПИД тангажа
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.set_trim_elevator(de_trim)   # упреждающий балансировочный δe
controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': H_TRIM})

# Скорость — газом (без этого набор на балансировочном газе ведёт к сваливанию)
spd_ctrl = SpeedController(aircraft, SpeedControlParams())
spd_ctrl.set_trim_throttle(thr_trim)
spd_ctrl.set_Va_ref(cfg.Va0)

rng = np.random.default_rng(seed=42)
h_ref_buf     = []
theta_ref_buf = []

# ------------------------------------------------------------------
# Функция управления
# ------------------------------------------------------------------
def controls_fn(t, state, Va, alpha):
    # Уставка высоты по расписанию
    if t < T_CLIMB:
        h_ref = H_TRIM
    elif t < T_DESCEND:
        h_ref = H_HIGH
    else:
        h_ref = H_TRIM

    # Внешний контур: P по высоте → theta_ref
    h_meas    = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    h_err     = h_ref - h_meas
    theta_ref = np.clip(alpha_trim + KH * h_err,
                        np.radians(-15.0), np.radians(15.0))

    controller.set_pitch_setpoint(theta_ref)
    h_ref_buf.append(h_ref)
    theta_ref_buf.append(theta_ref)

    # Внутренний контур: каскадный ПИД
    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    delta_e = controller.step(t, meas, cfg.dt)[0]
    return np.array([delta_e, spd_ctrl.step(Va_meas, cfg.dt)])

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(with_roll_hold(controls_fn, aircraft, sp, cfg.dt), aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С5  Контроль высоты")
path = logger.save(log, h_ref=h_ref_buf, theta_ref=theta_ref_buf, Va_ref=cfg.Va0)

# ------------------------------------------------------------------
# Показ (viz/viewer.py); аргумент командной строки — сохранить анимацию в .gif/.mp4
# ------------------------------------------------------------------
show_flightlog(path, speed=2.0,
               save_path=sys.argv[1] if len(sys.argv) > 1 else None)

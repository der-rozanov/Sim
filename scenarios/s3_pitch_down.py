# -*- coding: utf-8 -*-
"""
С3: Пикирование — выход на установившийся тангаж вниз.

Сценарий:
  0 .. T_TRIM с      → горизонтальный полёт (трим, h0=500 м)
  T_TRIM .. T_RETURN → уставка тангажа −5° (пикирование)
  T_RETURN .. конец  → возврат на тримовый тангаж

Тяга фиксирована на тримовом значении. Контроллер тангажа —
каскадный ПИД (theta → q → delta_e).

На графике выделены:
  - Переходный процесс по theta (уставка vs факт)
  - Угол атаки alpha с зонами предупреждения / критического УА
  - Высота (ЛА снижается при отрицательном тангаже)

Запуск:  python scenarios/s3_pitch_down.py
         python scenarios/s3_pitch_down.py results/s3.gif   -- сохранить анимацию
"""

import sys
import os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import with_roll_hold   # САУ по крену (РЕШ-19)
from control.controllers import PitchController, PitchControlParams
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
cfg = SimConfig(Va0=16.0, h0=500.0, theta0=0.0, dt=0.01, t_end=40.0)

T_TRIM       = 10.0             # с, длительность тримового участка
T_RETURN     = 20.0             # с, момент возврата на тримовый тангаж
THETA_CLIMB  = np.radians(-5.0) # рад, уставка кабрирования


# ------------------------------------------------------------------
# Балансировка
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

logger = FlightLogger(
    scenario="С3: Пикирование",
    description=f"Трим → θ_ref={np.degrees(THETA_CLIMB):.0f}° → возврат, Va={cfg.Va0:.0f} м/с",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_TRIM,   "label": "пикирование", "color": "steelblue"},
        {"t": T_RETURN, "label": "возврат",      "color": "darkorange"},
    ],
)

print(f"Трим:  alpha={np.degrees(alpha_trim):.2f}°  "
      f"delta_e={np.degrees(de_trim):.2f}°  throttle={thr_trim:.3f}")
print(f"Уставка пикирования:  theta_ref = {np.degrees(THETA_CLIMB):.1f}°")
print(f"Возврат на трим:      t = {T_RETURN:.0f} с")

# ------------------------------------------------------------------
# Контроллер
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.set_trim_elevator(de_trim)   # упреждающий балансировочный δe
controller.reset({"theta": s0[THETA], "q": 0.0, "h": cfg.h0})

rng            = np.random.default_rng(seed=42)
theta_ref_buf  = []

def controls_fn(t, state, Va, alpha):
    # Три фазы: трим → пикирование → возврат на трим
    if t < T_TRIM:
        theta_ref = alpha_trim
    elif t < T_RETURN:
        theta_ref = THETA_CLIMB
    else:
        theta_ref = alpha_trim
    controller.set_pitch_setpoint(theta_ref)
    theta_ref_buf.append(theta_ref)

    q_meas     = measure_gyro(state[Q],     sp.gyro_bias,    sp.gyro_noise,    rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    h_meas     = measure_altitude(state[H], sp.baro_bias,    sp.baro_noise,    rng)
    Va_meas    = measure_airspeed(Va,       sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {"q": q_meas, "theta": theta_meas, "h": h_meas, "Va": Va_meas}
    return controller.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(with_roll_hold(controls_fn, aircraft, sp, cfg.dt), aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С3  Пикирование -5°")
path = logger.save(log, theta_ref=theta_ref_buf)

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n          = len(log.t)
t_arr      = log.t
h_arr      = log.state[:, H]
theta_arr  = np.degrees(log.state[:, THETA])
alpha_arr  = np.degrees(log.alpha)

# Числовые итоги переходного процесса
# t_reach: первый момент, когда theta впервые входит в ±0.5° от уставки
# (фугоидная составляющая не мешает определению быстрого переходного процесса)
theta_target = np.degrees(THETA_CLIMB)
band_reach   = 0.5    # °, полоса первого достижения

t_reach = None
for i in range(n):
    if t_arr[i] < T_TRIM:
        continue
    if abs(theta_arr[i] - theta_target) <= band_reach:
        t_reach = t_arr[i]
        break

t_reach_str = f"{t_reach - T_TRIM:.2f} с" if t_reach else "—"
alpha_max_after = np.max(alpha_arr[t_arr >= T_TRIM])
dh_total = h_arr[-1] - h_arr[0]

print(f"\n  Переходный процесс:")
print(f"    Время первого достижения ±0.5°: {t_reach_str} от момента команды")
print(f"    Макс. alpha после команды: {alpha_max_after:.2f}°")
print(f"    Набор высоты за прогон:   {dh_total:.1f} м")

# ------------------------------------------------------------------
# Показ (viz/viewer.py); аргумент командной строки — сохранить анимацию в .gif/.mp4
# ------------------------------------------------------------------
show_flightlog(path, speed=4.0,
               save_path=sys.argv[1] if len(sys.argv) > 1 else None)

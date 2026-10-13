# -*- coding: utf-8 -*-
"""
С1: Установившийся (балансировочный) полёт.

Горизонтальный полёт с фиксированными балансировочными управлениями.
Управление вычисляется аналитически из условий равновесия (compute_trim),
затем остаётся неизменным на протяжении всего прогона.

Что демонстрирует:
  - Все параметры состояния (Va, alpha, theta, h, q) постоянны → модель корректна.
  - Энергия ЛА не дрейфует → интегратор (RK4) работает верно.
  - Балансировочные значения alpha_trim / delta_e_trim / throttle_trim
    соответствуют аналитическому расчёту.

Запуск:  python scenarios/s1_steady_flight.py
         python scenarios/s1_steady_flight.py results/s1.png   -- сохранить
"""

import sys
import os
import numpy as np

# Русский вывод в консоль Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Добавить корень проекта в путь поиска модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import with_roll_hold   # САУ по крену (РЕШ-19)
from sim.state import THETA, Q, H
from flight_logger import FlightLogger
from viz.viewer import show_flightlog


# ------------------------------------------------------------------
# Параметры прогона
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()          # нулевой ветер
cfg = SimConfig(Va0=16.0, h0=100.0, theta0=0.0, dt=0.01, t_end=30.0)

# ------------------------------------------------------------------
# Балансировочные условия (аналитическое решение системы 2×2)
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)

logger = FlightLogger(
    scenario="С1: Установившийся полёт",
    description=f"Балансировочный полёт Va={cfg.Va0:.0f} м/с, h={cfg.h0:.0f} м",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    trim=(alpha_trim, de_trim, thr_trim),
)

print("=" * 56)
print("С1: Установившийся полёт — балансировочные условия")
print("=" * 56)
print(f"  Va_trim    = {cfg.Va0:.1f} м/с")
print(f"  alpha_trim = {np.degrees(alpha_trim):.3f} °")
print(f"  delta_e    = {np.degrees(de_trim):.3f} °")
print(f"  throttle   = {thr_trim:.4f}")
print()

# ------------------------------------------------------------------
# Прогон с фиксированными trim-управлениями
# ------------------------------------------------------------------
ctrl_arr = np.array([de_trim, thr_trim])

s0  = trim_state(aircraft, cfg)
# Продольный канал — фиксированный трим; крен держит САУ (парирует момент винта)
log = run(with_roll_hold(lambda t, s, Va, alpha: ctrl_arr, aircraft, SensorParams(), cfg.dt),
          aircraft, wind_params, cfg, state0=s0)

print_summary(log, aircraft, label="С1  Установившийся полёт")
path = logger.save(log)

# ------------------------------------------------------------------
# Вычислить отклонения от балансировочного режима
# ------------------------------------------------------------------
Va_arr    = log.Va
alpha_arr = log.alpha
theta_arr = log.state[:, THETA]
q_arr     = log.state[:, Q]
h_arr     = log.state[:, H]

dVa_max    = np.max(np.abs(Va_arr    - cfg.Va0))
dalpha_max = np.max(np.abs(np.degrees(alpha_arr) - np.degrees(alpha_trim)))
dtheta_max = np.max(np.abs(np.degrees(theta_arr) - np.degrees(alpha_trim)))
dq_max     = np.max(np.abs(np.degrees(q_arr)))
dh_max     = np.max(np.abs(h_arr - cfg.h0))
dE_pct     = 100.0 * abs(log.E_total[-1] - log.E_total[0]) / log.E_total[0]

print("  Макс. отклонения от тримового режима:")
print(f"    |ΔVa|   ≤ {dVa_max:.4f} м/с")
print(f"    |Δα|    ≤ {dalpha_max:.4f} °")
print(f"    |Δθ|    ≤ {dtheta_max:.4f} °")
print(f"    |q|max  ≤ {dq_max:.4f} °/с")
print(f"    |Δh|    ≤ {dh_max:.4f} м")
print(f"    ΔE/E₀   = {dE_pct:.4f} %")
print()

# ------------------------------------------------------------------
# Показ (viz/viewer.py); аргумент командной строки — сохранить анимацию в .gif/.mp4
# ------------------------------------------------------------------
show_flightlog(path, speed=2.0, static=True,
               save_path=sys.argv[1] if len(sys.argv) > 1 else None)

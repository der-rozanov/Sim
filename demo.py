# -*- coding: utf-8 -*-
"""
demo.py -- запуск симулятора и отображение графиков.

Запуск:  python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from config import AircraftParams, WindParams, SimConfig
from runner import run, print_summary, compute_trim, trim_state
from plotting import (plot_dynamics, plot_trajectory,
                      plot_energy, plot_integrator_check)

# ------------------------------------------------------------------
# Параметры прогона
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
cfg         = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.1, t_end=20.0)

# ------------------------------------------------------------------
# Точные балансировочные условия (решение системы 2x2)
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")

ctrl_arr = np.array([de_trim, thr_trim])
def fixed_trim(t, state, Va, alpha):
    return ctrl_arr

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
s0  = trim_state(aircraft, cfg)
log = run(fixed_trim, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="Free flight (fixed trim controls)")

# ------------------------------------------------------------------
# Графики
# ------------------------------------------------------------------
plot_dynamics  (log, aircraft,    title="Free flight -- fixed trim")
plot_trajectory(log, wind_params, title="2D trajectory")
plot_energy    (log, aircraft,    title="Mechanical energy")
plot_integrator_check(aircraft, wind_params, cfg, t_end=20.0)

plt.show()

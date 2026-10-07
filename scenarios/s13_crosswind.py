# -*- coding: utf-8 -*-
"""
С13: Удержание путевого угла при боковом ветре.

Сценарий:
   0 .. T_WIND   с → штиль, путевой угол chi_ref = 0° (на север)
  T_WIND         с → ступенька бокового ветра VW м/с на восток (слева направо)
  T_TURN         с → chi_ref = 90° (на восток: ветер становится попутным)

Ожидание (кинематика треугольника скоростей, при β ≈ 0):
  - на путевом угле 0° ЛА летит с углом упреждения сноса:
      ψ = −arcsin(VW / Va),   Vg = √(Va² − VW²)
  - на путевом угле 90° ветер попутный: ψ ≈ χ, Vg ≈ Va + VW.
САУ держит путевой угол (GPS), а не курс — поэтому линия пути не сносится.

Ступенька ветра реализована порывом с большой длительностью (WindParams.gust_vwy).

Два прогона с одинаковым шумом: А) β = 0 по зонду, Б) руль направления в нейтрали.
Запуск:  python scenarios/s13_crosswind.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import PHI, PSI, H, X, Y, DA, DR, earth_velocity
from sim.wind import wind
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import LateralControlParams
from flight_logger import FlightLogger
from lateral_common import FullSAU, course_of

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft = AircraftParams()
sp       = SensorParams()
cfg      = SimConfig(Va0=16.0, h0=100.0, dt=0.01, t_end=80.0)

VW     = 5.0     # м/с, боковой ветер на восток  [МОДЕЛЬ: значение условное]
T_WIND = 20.0    # с, ступенька ветра
T_TURN = 50.0    # с, разворот на восток
CHI2   = np.radians(90.0)
deg    = np.degrees

wind_params = WindParams(gust_vwy=VW, gust_t0=T_WIND, gust_dur=1e6)


def chi_ref_fn(t):
    return 0.0 if t < T_TURN else CHI2


def simulate(beta_hold: bool):
    rng = np.random.default_rng(seed=42)
    sau = FullSAU(aircraft, sp, cfg, rng, chi_ref_fn,
                  lat_params=LateralControlParams(beta_hold=beta_hold))
    sau.bind_wind(lambda h, t: wind(h, t, wind_params))
    log = run(sau, aircraft, wind_params, cfg, state0=trim_state(aircraft, cfg))
    return log, sau


log_a, sau_a = simulate(beta_hold=True)
log_b, sau_b = simulate(beta_hold=False)
t = log_a.t
chi_a, chi_b = course_of(log_a), course_of(log_b)


def ground_speed(log):
    return np.array([np.hypot(*earth_velocity(s)[:2]) for s in log.state])

Vg_a = ground_speed(log_a)

print("=" * 92)
print(f"С13: Боковой ветер {VW:.0f} м/с (ступенька при t={T_WIND:.0f} с), прогон А — β по зонду")
print("=" * 92)
print(f"{'t,с':>5} {'χref°':>6} {'χ°':>7} {'ψ°':>7} {'ψ−χ°':>6} {'φ°':>6} {'β°':>6} "
      f"{'δa°':>6} {'δr°':>6} {'Vg':>5} {'h,м':>6} {'x,м':>6} {'y,м':>6}")
for tt in [0, 19, 20, 20.2, 20.5, 21, 22, 25, 30, 40, 50, 52, 55, 60, 70, 79.99]:
    i = min(int(np.searchsorted(t, tt - 1e-9)), len(t) - 1)
    s = log_a.state[i]
    print(f"{t[i]:5.1f} {deg(sau_a.chi_ref_buf[i]):6.0f} {deg(chi_a[i]):7.2f} "
          f"{deg(s[PSI]):7.2f} {deg(s[PSI] - chi_a[i]):6.2f} {deg(s[PHI]):6.2f} "
          f"{deg(log_a.beta[i]):6.2f} {deg(log_a.controls[i, DA]):6.2f} "
          f"{deg(log_a.controls[i, DR]):6.2f} {Vg_a[i]:5.1f} {s[H]:6.1f} "
          f"{s[X]:6.0f} {s[Y]:6.1f}")

# ------------------------------------------------------------------
# Сравнение с треугольником скоростей
# ------------------------------------------------------------------
m_hold = (t > T_TURN - 10) & (t < T_TURN)       # установившийся полёт на север при ветре
m_east = t > cfg.t_end - 10                     # установившийся полёт на восток
Va_m   = np.mean(log_a.Va[m_hold])
print()
print("Установившийся режим                  моделирование    треугольник скоростей")
print(f"  χ=0°:  ψ (угол упреждения), °     {deg(np.mean(log_a.state[m_hold, PSI])):12.2f}"
      f"     {deg(-np.arcsin(VW / Va_m)):12.2f}")
print(f"  χ=0°:  Vg, м/с                    {np.mean(Vg_a[m_hold]):12.2f}"
      f"     {np.sqrt(Va_m**2 - VW**2):12.2f}")
print(f"  χ=90°: ψ − χ, °                   {deg(np.mean(log_a.state[m_east, PSI] - chi_a[m_east])):12.2f}"
      f"     {0.0:12.2f}")
print(f"  χ=90°: Vg, м/с                    {np.mean(Vg_a[m_east]):12.2f}"
      f"     {np.mean(log_a.Va[m_east]) + VW:12.2f}")


# ------------------------------------------------------------------
# Метрики реакции на ступеньку ветра: А vs Б
# ------------------------------------------------------------------
def metrics(log, chi):
    m = (t >= T_WIND) & (t < T_TURN)
    y_track = log.state[m, Y]            # отклонение от линии пути на север
    return {
        "chi":  deg(np.max(np.abs(chi[m]))),
        "beta": deg(np.max(np.abs(log.beta[m]))),
        "beta_ss": deg(np.mean(log.beta[(t > T_WIND + 10) & (t < T_TURN)])),
        "y":    np.max(np.abs(y_track)),
        "phi":  deg(np.max(np.abs(log.state[m, PHI]))),
        "dr":   deg(np.max(np.abs(log.controls[m, DR]))),
        "dh":   np.max(np.abs(log.state[m, H] - cfg.h0)),
    }

ma, mb = metrics(log_a, chi_a), metrics(log_b, chi_b)
rows = [
    ("max |χ| после ступеньки ветра, °",          "chi",     ".2f"),
    ("max |β| после ступеньки ветра, °",          "beta",    ".2f"),
    ("β установившийся (среднее), °",             "beta_ss", ".3f"),
    ("max снос с линии пути |y|, м",              "y",       ".1f"),
    ("max |φ|, °",                                "phi",     ".2f"),
    ("max |δr|, °",                               "dr",      ".2f"),
    ("max |h − h0|, м",                           "dh",      ".1f"),
]
print()
print(f"{'Метрика (t = 20…50 с)':<44} {'А: β по зонду':>14} {'Б: δr = 0':>12}")
print("-" * 72)
for name, k, f in rows:
    print(f"{name:<44} {ma[k]:>14{f}} {mb[k]:>12{f}}")

print()
print_summary(log_a, aircraft, label="С13 прогон А (β по зонду)")

logger = FlightLogger(
    scenario="С13: Боковой ветер",
    description=f"Ступенька бокового ветра {VW:.0f} м/с, удержание путевого угла, разворот на 90°",
    aircraft=aircraft, wind_params=wind_params, cfg=cfg, sp=sp,
    trim=compute_trim(aircraft, cfg.Va0),
    events=[{"t": T_WIND, "label": "боковой ветер", "color": "orange"},
            {"t": T_TURN, "label": "курс 90°",      "color": "dodgerblue"}],
)
logger.save(log_a, h_ref=sau_a.h_ref_buf, theta_ref=sau_a.theta_ref_buf)

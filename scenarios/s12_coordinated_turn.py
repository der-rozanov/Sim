# -*- coding: utf-8 -*-
"""
С12: Координированный разворот — смена курса боковой САУ.

Расписание уставки курса chi_ref:
   0 .. T_TURN1  с → 0°    (полёт на север; первые ~15 с — переходный процесс
                            продольной САУ по высоте, как в С6, — манёвр позже)
  T_TURN1 .. T_TURN2 с → 90°   (разворот вправо на восток)
  T_TURN2 .. конец  с → −45°  (разворот влево на 135°, на северо-запад)

Высота 100 м и Va = 30 м/с удерживаются продольной САУ (как в С6).

Два прогона с ОДИНАКОВЫМ шумом датчиков, различие — только контур скольжения:
  А) beta_hold=True  — руль направления удерживает β = 0 по зонду (координация)
  Б) beta_hold=False — руль направления в нейтрали
Это показывает вклад измерения УС в координацию разворота.

Вывод: таблица по времени, метрики переходного процесса, .flightlog (прогон А).
Запуск:  python scenarios/s12_coordinated_turn.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import PHI, PSI, H, X, Y, P, R, DA, DR
from sim.wind import wind
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import LateralControlParams
from flight_logger import FlightLogger
from lateral_common import FullSAU, course_of, settle_time

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg         = SimConfig(Va0=16.0, h0=100.0, dt=0.01, t_end=75.0)

T_TURN1 = 20.0                  # с, команда разворота вправо
T_TURN2 = 45.0                  # с, команда разворота влево
CHI1    = np.radians(90.0)
CHI2    = np.radians(-45.0)
TOL     = np.radians(2.0)       # трубка установления по курсу
deg     = np.degrees


def chi_ref_fn(t):
    if t < T_TURN1:
        return 0.0
    return CHI1 if t < T_TURN2 else CHI2


def simulate(beta_hold: bool):
    rng = np.random.default_rng(seed=42)          # одинаковый шум в обоих прогонах
    sau = FullSAU(aircraft, sp, cfg, rng, chi_ref_fn,
                  lat_params=LateralControlParams(beta_hold=beta_hold))
    sau.bind_wind(lambda h, t: wind(h, t, wind_params))
    log = run(sau, aircraft, wind_params, cfg, state0=trim_state(aircraft, cfg))
    return log, sau


log_a, sau_a = simulate(beta_hold=True)
log_b, sau_b = simulate(beta_hold=False)
t = log_a.t

# ------------------------------------------------------------------
# Таблица по времени (прогон А)
# ------------------------------------------------------------------
chi_a = course_of(log_a)
chi_b = course_of(log_b)
print("=" * 96)
print("С12: Координированный разворот (прогон А — с контуром скольжения по зонду)")
print("=" * 96)
print(f"{'t,с':>5} {'χref°':>6} {'χ°':>7} {'ψ°':>7} {'φref°':>6} {'φ°':>6} {'β°':>6} "
      f"{'δa°':>6} {'δr°':>6} {'h,м':>6} {'Va':>5} {'x,м':>6} {'y,м':>6}")
for tt in [0, 5, 10, 20, 21, 22, 23, 25, 27, 30, 35, 45, 46, 48, 50, 53, 57, 65, 74.99]:
    i = min(int(np.searchsorted(t, tt - 1e-9)), len(t) - 1)
    s = log_a.state[i]
    print(f"{t[i]:5.1f} {deg(sau_a.chi_ref_buf[i]):6.0f} {deg(chi_a[i]):7.1f} "
          f"{deg(s[PSI]):7.1f} {deg(sau_a.phi_ref_buf[i]):6.1f} {deg(s[PHI]):6.1f} "
          f"{deg(log_a.beta[i]):6.2f} {deg(log_a.controls[i, DA]):6.2f} "
          f"{deg(log_a.controls[i, DR]):6.2f} {s[H]:6.1f} {log_a.Va[i]:5.1f} "
          f"{s[X]:6.0f} {s[Y]:6.0f}")

# ------------------------------------------------------------------
# Метрики
# ------------------------------------------------------------------
def metrics(log, chi, sau):
    chi_ref = np.array(sau.chi_ref_buf)
    err = (chi_ref - chi + np.pi) % (2 * np.pi) - np.pi
    m1 = (t >= T_TURN1) & (t < T_TURN2)
    m2 = t >= T_TURN2
    # Перерегулирование: выход за уставку в сторону движения
    over1 = max(0.0, deg(np.max(chi[m1]) - CHI1))
    over2 = max(0.0, deg(CHI2 - np.min(chi[m2])))
    return {
        "ts1":   settle_time(t[t < T_TURN2], err[t < T_TURN2], TOL, T_TURN1),
        "ts2":   settle_time(t, err, TOL, T_TURN2),
        "over1": over1,
        "over2": over2,
        "beta":  deg(np.max(np.abs(log.beta[t >= T_TURN1]))),
        "phi":   deg(np.max(np.abs(log.state[:, PHI]))),
        "dh":    np.max(np.abs(log.state[t >= T_TURN1, H] - cfg.h0)),
        "dVa":   np.max(np.abs(log.Va - cfg.Va0)),
        "da":    deg(np.max(np.abs(log.controls[:, DA]))),
        "dr":    deg(np.max(np.abs(log.controls[:, DR]))),
    }

ma, mb = metrics(log_a, chi_a, sau_a), metrics(log_b, chi_b, sau_b)
rows = [
    ("Время установления χ (±2°), разворот 90°, с",  "ts1",   ".1f"),
    ("Время установления χ (±2°), разворот 135°, с", "ts2",   ".1f"),
    ("Перерегулирование χ, разворот 90°, °",         "over1", ".1f"),
    ("Перерегулирование χ, разворот 135°, °",        "over2", ".1f"),
    ("max |β| в разворотах, °",                       "beta",  ".2f"),
    ("max |φ|, °",                                    "phi",   ".1f"),
    ("max |h − h0| с начала манёвра, м",              "dh",    ".1f"),
    ("max |Va − Va0|, м/с",                           "dVa",   ".2f"),
    ("max |δa|, °",                                   "da",    ".1f"),
    ("max |δr|, °",                                   "dr",    ".1f"),
]
print()
print(f"{'Метрика':<48} {'А: β по зонду':>14} {'Б: δr = 0':>12}")
print("-" * 76)
for name, k, f in rows:
    print(f"{name:<48} {ma[k]:>14{f}} {mb[k]:>12{f}}")

# Кинематика установившегося разворота: ψ̇ = g·tg φ / Va
i_ss = int(np.searchsorted(t, T_TURN1 + 3.0))
s = log_a.state[i_ss]
# ψ̇ — по окну ±0.5 с (руль направления с шумом зонда даёт мелкую дрожь r)
k = int(round(0.5 / cfg.dt))
psi_dot = (log_a.state[i_ss + k, PSI] - log_a.state[i_ss - k, PSI]) / (2 * k * cfg.dt)
print(f"\nРазворот при t={t[i_ss]:.1f} с: φ = {deg(s[PHI]):.1f}°, "
      f"ψ̇ = {deg(psi_dot):.2f} °/с, g·tgφ/Va = {deg(aircraft.g*np.tan(s[PHI])/log_a.Va[i_ss]):.2f} °/с")
R_turn = log_a.Va[i_ss]**2 / (aircraft.g * np.tan(abs(s[PHI])))
print(f"Радиус разворота R = Va²/(g·tgφ) = {R_turn:.0f} м")

print()
print_summary(log_a, aircraft, label="С12 прогон А (β по зонду)")

logger = FlightLogger(
    scenario="С12: Координированный разворот",
    description="Курс 0° → 90° → −45°, β=0 по зонду, h=100 м, Va=30 м/с",
    aircraft=aircraft, wind_params=wind_params, cfg=cfg, sp=sp,
    trim=compute_trim(aircraft, cfg.Va0),
    events=[{"t": T_TURN1, "label": "курс 90°",  "color": "orange"},
            {"t": T_TURN2, "label": "курс −45°", "color": "dodgerblue"}],
)
logger.save(log_a, h_ref=sau_a.h_ref_buf, theta_ref=sau_a.theta_ref_buf)

# -*- coding: utf-8 -*-
"""
С14: Полёт по точкам (B&M гл. 10–11).

Маршрут — ломаная из WAYPOINTS (N, E, h), старт — из начала координат курсом
на север. Уставки курса и высоты даёт WaypointNavigator (control/navigation.py):
следование по прямой векторным полем, переключение участков по полуплоскости
(биссектриса угла), после последней точки — кружение радиусом R_orbit.
Остальная САУ — FullSAU (как в s12/s13): χ → φ → δa, β → δr, h → θ → δe, Va → δt.

Два прогона: А) штиль, Б) постоянный ветер VW м/с на восток.
Вывод: моменты прохождения точек, боковое отклонение от линии на прямых,
ошибка радиуса на окружности; .flightlog — в results/.

Запуск:  python scenarios/s14_waypoints.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import H, X, Y, PHI
from sim.wind import wind
from runner import run, trim_state
from control.navigation import WaypointNavigator, NavParams
from flight_logger import FlightLogger
from lateral_common import FullSAU

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft = AircraftParams()
sp       = SensorParams()
cfg      = SimConfig(Va0=16.0, h0=100.0, dt=0.01, t_end=260.0)

# (N, E, h), м — квадрат 1 км с острым углом в конце  [МОДЕЛЬ: маршрут условный]
WAYPOINTS = [(1000, 0, 100), (1000, 1000, 130), (0, 1000, 130), (600, 300, 100)]
VW  = 5.0    # м/с, ветер на восток в прогоне Б  [МОДЕЛЬ: значение условное]
deg = np.degrees


def simulate(wind_params, label, nav_params=None):
    rng = np.random.default_rng(seed=42)
    nav = WaypointNavigator(nav_params or NavParams())
    nav.set_route(WAYPOINTS, start=(0.0, 0.0, cfg.h0))
    sau = FullSAU(aircraft, sp, cfg, rng, chi_ref_fn=lambda t: 0.0, nav=nav)
    sau.bind_wind(lambda h, t: wind(h, t, wind_params))

    # режим навигатора и отклонение — на каждом шаге (обёртка САУ)
    epy, mode, idx = [], [], []

    def controls_fn(t, state, Va, alpha):
        c = sau(t, state, Va, alpha)
        epy.append(nav.e_py)
        mode.append(nav.mode)
        idx.append(nav.idx)
        return c

    log = run(controls_fn, aircraft, wind_params, cfg, state0=trim_state(aircraft, cfg))
    mode, idx = np.array(mode), np.array(idx)
    # точка k пройдена — навигатор ушёл с участка к ней (на следующий или на кружение)
    passed = [int(np.argmax((idx > k) | (mode == "orbit"))) for k in range(len(WAYPOINTS))]
    events = [{"t": round(log.t[i], 2), "label": f"точка {k + 1}", "color": "dodgerblue"}
              for k, i in enumerate(passed) if i > 0]
    FlightLogger(scenario="С14", description=f"полёт по точкам — {label}",
                 aircraft=aircraft, wind_params=wind_params, cfg=cfg,
                 events=events).save(log)
    return log, sau, np.array(epy), mode, idx


def report(label, log, sau, epy, mode, idx):
    t, st = log.t, log.state
    print("=" * 78)
    print(f"С14: полёт по точкам — {label}")
    print("=" * 78)
    print(f"{'точка':>6} {'N, м':>7} {'E, м':>7} {'h_ref':>6} | {'мин. расст., м':>14} "
          f"{'t, с':>6} {'h ЛА':>6}")
    for k, wp in enumerate(WAYPOINTS):
        d = np.hypot(st[:, X] - wp[0], st[:, Y] - wp[1])
        i = int(np.argmin(d))
        print(f"{k + 1:6d} {wp[0]:7.0f} {wp[1]:7.0f} {wp[2]:6.0f} | {d[i]:14.1f} "
              f"{t[i]:6.1f} {st[i, H]:6.1f}")
    print()
    print("Участок (прямая; 10 с после входа отброшены)   |e_py| max, м  |e_py| конец, м  "
          "|h − h_ref| конец, м")
    h_ref = np.array(sau.h_ref_buf)
    for k in range(len(WAYPOINTS)):
        m = (mode == "line") & (idx == k)
        if not m.any():
            continue
        m &= t > t[m][0] + 10.0
        if not m.any():
            continue
        j = np.where(m)[0][-1]
        name = "старт" if k == 0 else f"{k}"
        print(f"  {name:>5} → {k + 1:<36} {np.max(np.abs(epy[m])):12.1f} {abs(epy[j]):15.1f}"
              f" {abs(st[j, H] - h_ref[j]):20.1f}")
    m = mode == "fillet"
    if m.any():
        print(f"  дуги скругления R = {sau.nav.params.R_fillet:.0f} м: |d − R| max = "
              f"{np.max(np.abs(epy[m])):.1f} м;  φ max = {deg(np.max(np.abs(st[m, PHI]))):.1f}°")
    m = mode == "orbit"
    if m.any():
        mm = m & (t > t[m][0] + 40)
        if mm.any():
            print(f"  кружение R = {sau.nav.params.R_orbit:.0f} м (через 40 с после входа): "
                  f"|d − R| max = {np.max(np.abs(epy[mm])):.1f} м, СКО = {np.std(epy[mm]):.1f} м")
    print()


if __name__ == "__main__":
    for wp, label in [(WindParams(), "штиль"),
                      (WindParams(Vw_cross=VW), f"ветер {VW:.0f} м/с на восток")]:
        report(label, *simulate(wp, label))

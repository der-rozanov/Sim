# -*- coding: utf-8 -*-
"""
С11: пилотажный комплекс C-11 в автономном режиме (побочная разработка, docs/DEV_PLAN.md).

Фигуры 1–8 схемы Starting Schedule C-11 (январь 2010): петля; горка с разворотом (вместо
поворота на вертикали — нет обдувки РН винтом); квадратная петля; подъём 45° с полубочкой
и 5/8 петли; бочка; вертикаль — полупетля — вертикаль с полубочкой; кубинская восьмёрка;
Иммельман. Штопор (фигура 9) не выполняется — нет АДХ за срывом (решение автора).

Траектория комплекса — control/aerobatics.py (C11, programme): фигуры подряд, между ними
прямые 60 м. Регулятор — AerobaticController (docs/control.md, раздел 13), истинные
измерения, вход на 25 м/с, 150 м, курс на север.

Два прогона: А) штиль, Б) ветер 4 м/с на юг — встречный на фигуре 1, как на схеме
(«Windrichtung»). Траектория задана в земной СК: при ветре петли круглые относительно земли.
Вывод: показатели по фигурам; .flightlog — в results/ (h_ref — высота опорной точки
траектории, события — начала фигур). 3D: python viz/viewer3d.py results/<файл>.flightlog

Это моделирование.

Запуск:  python scenarios/s11_aerobatics.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import H
from sim.wind import make_wind_fn
from runner import run, trim_state
from control.aerobatics import AerobaticController, AeroParams, Figure, programme, C11
from flight_logger import FlightLogger
from viz.viewer import show_flightlog

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft = AircraftParams()
params   = AeroParams()                        # Va_ref = 25 м/с
cfg      = SimConfig(Va0=params.Va_ref, h0=150.0, dt=0.01)
VW       = -4.0    # м/с, ветер на север (< 0 — на юг) в прогоне Б  [МОДЕЛЬ: значение условное]
deg = np.degrees

SEGS = programme(C11)
# номер фигуры для каждого отрезка программы: −1 — прямые входа и между фигурами
FIG_OF_SEG = [-1]
for k, (_, fsegs) in enumerate(C11):
    FIG_OF_SEG += [k] * len(fsegs) + [-1]
FIG_OF_SEG += [-1]                             # после конца программы


def simulate(wind_params, label):
    s0 = trim_state(aircraft, cfg)
    ctl = AerobaticController(aircraft, params, wind_fn=make_wind_fn(wind_params))
    fig = Figure.from_state("C-11", SEGS, s0)
    ctl.set_figure(fig)
    seg, e, att, h_ref = [], [], [], []

    def controls_fn(t, state, Va, alpha):
        u = ctl.step(t, state, Va, alpha, cfg.dt)
        lg = ctl.log
        seg.append(lg["seg"])
        e.append(np.hypot(lg["e_y"], lg["e_z"]))
        att.append(lg["att_err"])
        h_ref.append(-fig.ref(ctl.i, ctl.s)[0][2])
        return u

    t_end = fig.length / params.Va_ref * 1.1 + 2.0     # в фигурах Va < Va_ref: конец ≈ 59 с
    log = run(controls_fn, aircraft, wind_params, SimConfig(Va0=cfg.Va0, h0=cfg.h0, dt=cfg.dt,
                                                            t_end=t_end), state0=s0)
    seg = np.array(seg)
    fidx = np.array([FIG_OF_SEG[min(i, len(FIG_OF_SEG) - 1)] for i in seg])
    events = []
    for k, (name, _) in enumerate(C11):
        i = np.where(fidx == k)[0]
        if len(i):
            events.append({"t": round(float(log.t[i[0]]), 2), "label": name.split(".")[0],
                           "color": "dodgerblue"})
    path = FlightLogger(scenario="С11", description=f"пилотажный комплекс C-11 — {label}",
                        aircraft=aircraft, wind_params=wind_params, cfg=cfg,
                        events=events).save(log, h_ref=h_ref, Va_ref=params.Va_ref)
    return path, log, fidx, np.array(e), deg(np.array(att))


def report(label, log, fidx, e, att):
    n = len(fidx)
    Va, al, be = log.Va[:n], deg(log.alpha[:n]), deg(np.abs(log.beta[:n]))
    print("=" * 100)
    print(f"С11: комплекс C-11 — {label};  t = {log.t[n - 1]:.1f} с,  h: {cfg.h0:.0f} → "
          f"{log.state[n - 1, H]:.1f} м")
    print("=" * 100)
    print(f"{'фигура':52s} {'|e| max, м':>10} {'ориент., °':>10} {'Va, м/с':>12} {'α, °':>12} {'|β| max, °':>10}")
    for k, (name, _) in enumerate(C11):
        m = fidx == k
        if not m.any():
            print(f"{name:52s}   не выполнена")
            continue
        print(f"{name:52s} {e[m].max():10.2f} {att[m].max():10.1f} {Va[m].min():5.1f}…{Va[m].max():5.1f} "
              f"{al[m].min():5.1f}…{al[m].max():5.1f} {be[m].max():10.1f}")
    m = fidx == -1
    print(f"{'прямые между фигурами':52s} {e[m].max():10.2f}")
    print()


if __name__ == "__main__":
    paths = []
    for wp, label in [(WindParams(), "штиль"),
                      (WindParams(Vw_const=VW), f"ветер {abs(VW):.0f} м/с на юг")]:
        path, *res = simulate(wp, label)
        report(label, *res)
        paths.append(path)
    for p in paths:
        print("3D-просмотр: python viz/viewer3d.py " + os.path.relpath(p))
    show_flightlog(*paths, speed=5.0)

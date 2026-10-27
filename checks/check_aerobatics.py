# -*- coding: utf-8 -*-
"""
Пилотаж (control/aerobatics.py, побочная разработка «Пилотажный комплекс C-11»):
фигуры по одной на полной модели, FPV-самолёт, штиль, истинные измерения.

  1. Петля R = 15 м.
  2. Бочка: 360° по крену за 50 м прямой (≈ 180 °/с на 25 м/с).
  3. Квадратная петля: стороны 20 м, углы R = 10 м.
  + геометрия всех фигур комплекса C-11: выход горизонтально, «на ногах», в нужную сторону
    (полёт всего комплекса — scenarios/s11_aerobatics.py).

Вход — горизонтальный полёт на 25 м/с, 150 м. Показатели: отклонение от траектории
в плоскости, перпендикулярной касательной (|e| = √(e_y² + e_z²)), — на фигуре и после
неё; ошибка ориентации; изменение высоты на выходе; перегрузка n = −f_z/g (удельная
сила вдоль −z связанной СК); диапазоны Va и α; рули в упоре. Это моделирование.

Запуск: python checks/check_aerobatics.py  → checks/aerobatics.png
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dataclasses import replace
import matplotlib.pyplot as plt

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import H, X, Y
from sim.dynamics import specific_force
from runner import run, trim_state
from control.aerobatics import AerobaticController, AeroParams, Figure, line, arc

ac = AircraftParams()
PR = AeroParams()
H0 = 150.0
calm = lambda h, t: (0.0, 0.0, 0.0)
ok_all = True


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {info}")


FIGURES = [
    ("Петля R = 15 м", [line(40), arc(15, 360), line(100)], 1.0),
    ("Бочка 360° за 50 м", [line(40), line(50, 360), line(100)], 4.0),
    ("Квадратная петля", [line(40), arc(10, 90), line(20), arc(10, 90), line(20),
                          arc(10, 90), line(20), arc(10, 90), line(100)], 1.5),
]


def fly(segs):
    cfg = SimConfig(Va0=PR.Va_ref, h0=H0, dt=0.01)
    s0 = trim_state(ac, cfg)
    ctl = AerobaticController(ac, PR)
    fig = Figure.from_state("", segs, s0)
    ctl.set_figure(fig)
    rec, ref = [], []

    def fn(t, s, Va, al):
        u = ctl.step(t, s, Va, al, cfg.dt)
        rec.append(dict(ctl.log))
        ref.append(fig.ref(ctl.i, ctl.s)[0])
        return u

    log = run(fn, ac, WindParams(), replace(cfg, t_end=fig.length / PR.Va_ref + 3.0), state0=s0)
    return log, rec, np.array(ref), fig


# --- геометрия фигур комплекса C-11 -------------------------------------------
print("\nГеометрия фигур C-11 (control/aerobatics.py): выход горизонтально, «на ногах»")
from control.aerobatics import C11
EXIT = [+1, -1, +1, -1, +1, -1, +1, -1]     # направление выхода относительно входа (схема C-11)
for (name, segs), d in zip(C11, EXIT):
    f = Figure(name, segs, [0.0, 0.0, 0.0], np.eye(3))
    t_end, z_end = f.R_end[:, 0], f.R_end[:, 2]
    check(name, np.allclose(t_end, [d, 0, 0], atol=1e-9) and np.allclose(z_end, [0, 0, 1], atol=1e-9),
          f"выход t = {np.round(t_end, 3)}, Δh = {-f.p_end[2]:+.1f} м")

results = []
for name, segs, e_lim in FIGURES:
    print(f"\n{name}")
    log, rec, ref, fig = fly(segs)
    n = len(rec)
    seg = np.array([r["seg"] for r in rec])
    e = np.hypot([r["e_y"] for r in rec], [r["e_z"] for r in rec])
    att = np.degrees([r["att_err"] for r in rec])
    infig = (seg >= 1) & (seg < len(segs) - 1)
    after = seg >= len(segs) - 1
    nz = np.array([-specific_force(log.state[k], log.controls[k], log.t[k], ac, calm)[2] / ac.g
                   for k in range(n)])
    dh = log.state[n - 1, H] - H0
    sat = [np.mean(np.abs(log.controls[:n, k]) >= ac.delta_e_max - 1e-6) * 100 for k in (0, 2, 3)]
    print(f"   |e| на фигуре max {e[infig].max():.2f} м, СКО {np.sqrt(np.mean(e[infig]**2)):.2f} м; "
          f"после фигуры max {e[after].max():.2f} м; ошибка ориентации max {att[infig].max():.1f}°")
    print(f"   Δh на выходе {dh:+.1f} м; n ∈ [{nz.min():.1f}, {nz.max():.1f}]; "
          f"Va ∈ [{log.Va.min():.1f}, {log.Va.max():.1f}] м/с; α ∈ [{np.degrees(log.alpha.min()):.1f}, "
          f"{np.degrees(log.alpha.max()):.1f}]°; |β| max {np.degrees(np.abs(log.beta).max()):.1f}°")
    print(f"   рули в упоре, % времени: δe {sat[0]:.1f}, δa {sat[1]:.1f}, δr {sat[2]:.1f}; "
          f"газ 1.0 — {np.mean(log.controls[:n, 1] >= 1 - 1e-6) * 100:.0f} %")
    check("фигура пройдена до конца", seg[-1] >= len(segs) - 1)
    check(f"|e| на фигуре < {e_lim} м", e[infig].max() < e_lim, f"{e[infig].max():.2f} м")
    check("после фигуры |e| < 1 м", e[after][-50:].max() < 1.0, f"{e[after][-50:].max():.2f} м")
    check("α < α_warning, Va > 1.5·Va_stall (7.5 м/с)",
          log.alpha.max() < ac.alpha_warning and log.Va.min() > 1.5 * 7.5)
    results.append((name, log, rec, ref, nz, n))

# --- рисунок -----------------------------------------------------------------
BLUE, ORANGE, GRAY, GREEN = "#2a78d6", "#eb6834", "#898989", "#1baf7a"
fig_, axs = plt.subplots(2, 3, figsize=(15, 9))
for j, (name, log, rec, ref, nz, n) in enumerate(results):
    ax = axs[0, j]
    x, y, h = log.state[:n, X], log.state[:n, Y], log.state[:n, H]
    seg = np.array([r["seg"] for r in rec])
    nseg = len(FIGURES[j][1])
    m = (seg >= 1) & (seg <= nseg - 2)
    x0, x1 = ref[m, 0].min() - 20, ref[m, 0].max() + 20
    if j == 1:                                       # бочка: отклонения в увеличенном масштабе
        ax.plot(ref[:, 0], -ref[:, 2] - H0, color=GRAY, lw=4, alpha=0.5, label="траектория фигуры")
        ax.plot(x, h - H0, color=BLUE, lw=1.6, label="Δh, м")
        ax.plot(x, y, color=ORANGE, lw=1.6, label="боковое отклонение y, м")
        ax.set_ylabel("м")
    else:
        ax.plot(ref[:, 0], -ref[:, 2], color=GRAY, lw=4, alpha=0.5, label="траектория фигуры")
        ax.plot(x, h, color=BLUE, lw=1.6, label="полёт (вид сбоку)")
        ax.set_aspect("equal")
        ax.set_ylabel("h, м")
    ax.set_xlim(x0, x1)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("x (север), м")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax = axs[1, j]
    t = log.t[:n]
    e = np.hypot([r["e_y"] for r in rec], [r["e_z"] for r in rec])
    ax.plot(t, e, color=BLUE, lw=1.6, label="|e|, м")
    ax.plot(t, nz, color=ORANGE, lw=1.2, label="перегрузка n")
    ax.plot(t, log.Va[:n] / 5, color=GREEN, lw=1.2, label="Va / 5, м/с")
    ax.plot(t, np.degrees(log.alpha[:n]), color=GRAY, lw=1.0, label="α, °")
    ax.set_xlabel("t, с")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig_.suptitle("Пилотаж по траектории (control/aerobatics.py), FPV-самолёт, 25 м/с, штиль — моделирование",
              fontsize=11)
fig_.tight_layout()
out = os.path.join(os.path.dirname(__file__), "aerobatics.png")
fig_.savefig(out, dpi=110)
print(f"\nрисунок: {out}")
print("\nALL AEROBATIC CHECKS PASSED" if ok_all else "\nFAILURES FOUND")

# -*- coding: utf-8 -*-
"""
Проверка турбулентности Драйдена (sim/turbulence.py, docs/physics.md, «Турбулентность»):

  1. СКО порывов по трём осям против σ табл. 4.1 B&M — все уровни и высоты
     (ряд 4000 с × 5 seed'ов; время корреляции L/Va до 33 с — нужен длинный ряд).
  2. Спектральная плотность мощности (СПМ) ряда против аналитической |H(jω)|²
     в полосе 0.05…20 рад/с (усреднённая периодограмма, окно Ханна) —
     checks/turbulence_psd.png.
  3. Одно t → одно значение при любом порядке вызовов; до t0 порыв нулевой.
  4. turb_level = "off" → wind() побитно совпадает с прежней моделью.
  5. Замкнутый контур (SAU, TECS + курс, FPV, Va = 16 м/с, h = 100 м, 120 с):
     разброс h, Va, α, φ, χ при слабой и умеренной турбулентности — для сведения,
     критерий — только «полёт не прерван».

Запуск: python checks/check_turbulence.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scenarios"))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import H, earth_velocity
from sim.turbulence import DrydenTurbulence, DRYDEN_TABLE, dryden_params
from sim.wind import wind, make_wind_fn
from runner import run, trim_state
from lateral_common import FullSAU

ok_all = True


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {info}")


def series(level, alt, seed, T):
    """Ряд порывов (N, 3) на сетке turb_dt длиной T с."""
    tb = DrydenTurbulence(WindParams(turb_level=level, turb_alt=alt, turb_seed=seed))
    tb(T)
    return tb.series[1:int(T / tb.dt)], tb.dt


# ---------------------------------------------------------------------------
print("1. СКО порывов против табл. 4.1 B&M (5 × 4000 с), м/с")
print(f"   {'уровень':>18} {'σu':>6} {'σv':>6} {'σw':>6}   {'табл. σu,σv,σw':>16}")
for (alt, level) in DRYDEN_TABLE:
    wp = WindParams(turb_level=level, turb_alt=alt)
    _, _, _, su, sv, sw = dryden_params(wp)
    sig = np.sqrt(np.mean([np.var(series(level, alt, s, 4000.0)[0], axis=0)
                           for s in range(5)], axis=0))
    ref = np.array([su, sv, sw])
    err = np.max(np.abs(sig / ref - 1.0))
    print(f"   {alt + '/' + level:>18} {sig[0]:6.3f} {sig[1]:6.3f} {sig[2]:6.3f}   "
          f"{su:5.2f} {sv:5.2f} {sw:5.2f}")
    check(f"{alt}/{level}: |σ/σ_табл − 1| < 8 %", err < 0.08, f"макс. {err * 100:.1f} %")


# ---------------------------------------------------------------------------
print("\n2. СПМ против |H(jω)|² (low/light, 20 000 с)")


def psd_analytic(w, L, sigma, V, second_order):
    """|H(jω)|² формирующего фильтра Драйдена (вход — белый шум единичной интенсивности)."""
    a = V / L
    if not second_order:
        return sigma**2 * 2.0 * a / (w**2 + a**2)
    return sigma**2 * 3.0 * a * (w**2 + a**2 / 3.0) / (w**2 + a**2)**2


def psd_welch(y, dt, nseg=2**14):
    """Усреднённая периодограмма с окном Ханна: P(ω) с ∫P dω/2π (−∞…∞) = дисперсия."""
    win = np.hanning(nseg)
    segs = [y[i:i + nseg] for i in range(0, len(y) - nseg + 1, nseg // 2)]
    P = np.mean([np.abs(np.fft.rfft((s - s.mean()) * win))**2 for s in segs], axis=0)
    P *= dt / np.sum(win**2)
    w = 2.0 * np.pi * np.fft.rfftfreq(nseg, dt)
    return w[1:], P[1:]


wp = WindParams(turb_level="light", turb_alt="low")
Lu, Lv, Lw, su, sv, sw = dryden_params(wp)
y, dt = series("light", "low", 7, 20000.0)
names = [("ug (север, H_u)", Lu, su, False), ("vg (восток, H_v)", Lv, sv, True),
         ("wg (вверх, H_w)", Lw, sw, True)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
edges = np.logspace(np.log10(0.05), np.log10(20.0), 13)      # полосы по ω, рад/с
for j, (name, L, sig, so) in enumerate(names):
    w, P = psd_welch(y[:, j], dt)
    Pa = psd_analytic(w, L, sig, wp.turb_Va, so)
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (w >= lo) & (w < hi)
        ratios.append(np.mean(P[m]) / np.mean(Pa[m]))
    ratios = np.array(ratios)
    err = np.max(np.abs(ratios - 1.0))
    check(f"{name}: СПМ / |H|² в 12 полосах 0.05…20 рад/с — в пределах 25 %",
          err < 0.25, f"от {ratios.min():.2f} до {ratios.max():.2f}")
    ax = axes[j]
    ax.loglog(w, P, color="#7fa7d9", lw=3.0, label="моделирование (периодограмма)")
    ax.loglog(w, Pa, color="#222222", lw=1.0, ls="--", label="|H(jω)|², B&M (4.4)")
    ax.set_title(f"{name}: L = {L:.0f} м, σ = {sig:.2f} м/с", fontsize=10)
    ax.set_xlabel("ω, рад/с")
    ax.set_xlim(0.01, 50)
    ax.grid(True, which="major", color="#dddddd", lw=0.6)
axes[0].set_ylabel("СПМ, (м/с)²·с/рад")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Турбулентность Драйдена, табл. 4.1 B&M: малая высота, слабая; "
             "Va = 16 м/с (моделирование)", fontsize=11)
fig.tight_layout()
png = os.path.join(os.path.dirname(__file__), "turbulence_psd.png")
fig.savefig(png, dpi=110)
print(f"   график: {os.path.relpath(png, ROOT)}")


# ---------------------------------------------------------------------------
print("\n3. Детерминированность и момент включения t0")
wp = WindParams(turb_level="moderate", turb_seed=3)
ts = np.array([5.0, 0.123, 60.0, 5.0, 33.3333, 0.123])
a = DrydenTurbulence(wp)
b = DrydenTurbulence(wp)
va = np.array([a(t) for t in ts])
vb = np.array([b(t) for t in np.sort(ts)])[np.argsort(np.argsort(ts))]
check("одно t → одно значение при любом порядке вызовов", np.array_equal(va, vb))
c = DrydenTurbulence(wp, t0=100.0)
check("до t0 порыв нулевой, через t после t0 — как у генератора с t0 = 0",
      c(50.0) == (0.0, 0.0, 0.0) and np.allclose(c(105.0), a(5.0)))


# ---------------------------------------------------------------------------
print("\n4. Турбулентность выключена → прежняя модель ветра")
wp = WindParams(Vw_const=3.0, Vw_cross=-2.0, dV_shear=4.0, gust_amp=2.0, gust_vwh=1.0,
                gust_vwy=-1.5, gust_t0=10.0, gust_dur=2.0)
fn = make_wind_fn(wp)
pts = [(h, t) for h in (0.0, 60.0, 120.0) for t in (0.0, 10.5, 30.0)]
check("make_wind_fn(off) ≡ wind(h, t, params)",
      all(fn(h, t) == wind(h, t, wp) for h, t in pts))


# ---------------------------------------------------------------------------
print("\n5. Замкнутый контур: SAU (TECS + курс), χ_ref = 0, h_ref = 100 м, 120 с, "
      "шум датчиков — SensorParams()")
ac, sp = AircraftParams(), SensorParams()
cfg = SimConfig(Va0=16.0, h0=100.0, dt=0.01, t_end=120.0)
print(f"   {'турбулентность':>14} {'σh, м':>6} {'|Δh|max':>8} {'σVa':>5} {'Va min…max':>11} "
      f"{'α max, °':>8} {'σφ, °':>6} {'|φ|max':>7} {'|χ|max, °':>9}")
for level in ("off", "light", "moderate"):
    wp = WindParams(turb_level=level, turb_seed=1)
    sau = FullSAU(ac, sp, cfg, np.random.default_rng(42), lambda t: 0.0)
    sau.bind_wind(make_wind_fn(wp))
    log = run(sau, ac, wp, cfg, state0=trim_state(ac, cfg))
    m = log.t > 10.0
    dh = log.state[m, H] - 100.0
    chi = np.array([np.arctan2(*earth_velocity(s)[1::-1]) for s in log.state[m]])
    full = log.t[-1] > cfg.t_end - 0.05
    print(f"   {level:>14} {dh.std():6.2f} {np.abs(dh).max():8.2f} {log.Va[m].std():5.2f} "
          f"{log.Va[m].min():5.1f}…{log.Va[m].max():4.1f} {np.degrees(log.alpha[m].max()):8.1f} "
          f"{np.degrees(log.phi[m].std()):6.2f} "
          f"{np.degrees(np.abs(log.phi[m]).max()):7.1f} {np.degrees(np.abs(chi).max()):9.1f}")
    check(f"{level}: полёт 120 с не прерван", full)
print(f"   (α_warning = {np.degrees(ac.alpha_warning):.1f}°, "
      f"α_crit = {np.degrees(ac.alpha_crit):.1f}°)")

print("\nALL TURBULENCE CHECKS PASSED" if ok_all else "\nSOME TURBULENCE CHECKS FAILED")

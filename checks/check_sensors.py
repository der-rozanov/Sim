# -*- coding: utf-8 -*-
"""
Проверка реалистичного набора датчиков (control/sau.RealisticSensors, docs/SENSORS.md, раздел 4):

  1. Удельная сила (sim.dynamics.specific_force): в балансировочном горизонтальном
     полёте |f| = g.
  2. Канал SampledChannel: частота отсчётов и запаздывание — выход равен отсчёту,
     взятому на сетке period не позже t − delay.
  3. Марковский процесс GaussMarkov: установившееся СКО и корреляция e^(−1) через τ.
  4. Замкнутый контур (SAU, TECS + курс, FPV, Va = 16 м/с, штиль, 180 с):
     разворот χ 0 → 90° (20 с), набор h 100 → 130 м (80 с) на трёх наборах датчиков —
     истина / NoisySensors (s8–s10) / RealisticSensors. Критерий — полёт не прерван,
     развороты и набор выполнены; числа — для сведения. График — checks/sensors_real.png.

Запуск: python checks/check_sensors.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import H, X, Y, PHI, air_data, earth_velocity
from sim.dynamics import specific_force
from sim.wind import make_wind_fn
from runner import run, compute_trim, trim_state
from control.controllers import PitchControlParams, wrap_angle
from control.sensors import GaussMarkov, SampledChannel
from control.sau import SAU, SAUParams, Mode, TruthSensors, NoisySensors, RealisticSensors

ok_all = True
deg = np.degrees


def check(name, ok, info=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}  {info}")


ac, sp = AircraftParams(), SensorParams()
cfg = SimConfig(Va0=16.0, h0=100.0, dt=0.01, t_end=180.0)
wp = WindParams()
wind_fn = make_wind_fn(wp)
trim = compute_trim(ac, cfg.Va0)

# ---------------------------------------------------------------------------
print("1. Удельная сила (акселерометр без шума)")
s0 = trim_state(ac, cfg)
f = specific_force(s0, [trim[1], trim[2], 0.0, 0.0], 0.0, ac, wind_fn)
check("балансировка: |f| = g", abs(np.linalg.norm(f) / ac.g - 1.0) < 0.01,
      f"f = ({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f}) м/с², |f|/g = {np.linalg.norm(f) / ac.g:.4f}")


# ---------------------------------------------------------------------------
print("\n2. Канал: частота отсчётов и запаздывание")
for period, delay in ((0.05, 0.0), (0.2, 0.2), (0.2, 0.15)):
    ch, err = SampledChannel(period, delay), 0.0
    for k in range(1000):
        t = k * 0.01
        y = ch.update(t, lambda: t)                    # отсчёт = момент его взятия
        if t >= delay + period:
            expect = np.floor((t - delay) / period + 1e-6) * period
            err = max(err, abs(y - expect))
    check(f"период {period} с, задержка {delay} с: выход = отсчёт на сетке не позже t − delay",
          err < 1e-9, f"макс. расхождение {err:.1e} с")


# ---------------------------------------------------------------------------
print("\n3. Марковский процесс 1-го порядка (σ = 1, τ = 3 с, шаг 0.01 с, 5000 с)")
gm = GaussMarkov(1.0, 3.0, np.random.default_rng(5))
b = np.array([gm.step(0.01)[0] for _ in range(500_000)])
lag = int(3.0 / 0.01)
rho = np.corrcoef(b[:-lag], b[lag:])[0, 1]
check("установившееся СКО = σ", abs(b.std() - 1.0) < 0.05, f"{b.std():.3f}")
check("корреляция через τ = e^(−1) = 0.368", abs(rho - np.exp(-1)) < 0.05, f"{rho:.3f}")


# ---------------------------------------------------------------------------
print("\n4. Замкнутый контур: разворот χ 0 → 90° (20 с), набор h 100 → 130 м (80 с)")
T_TURN, T_CLIMB, CHI2, H2 = 20.0, 80.0, np.radians(90.0), 130.0


def flight(kind):
    rng = np.random.default_rng(42)
    params = SAUParams(pitch=PitchControlParams(Va_ref=cfg.Va0))
    holder = {}
    if kind == "truth":
        sens = TruthSensors()
    elif kind == "noisy":
        sens = NoisySensors(sp, rng, gps_pos=True)
    else:
        sens = RealisticSensors(sp, rng, accel_fn=lambda t, s: specific_force(
            s, holder["sau"].controls, t, ac, wind_fn))
    sau = SAU(ac, trim, sens, params)
    holder["sau"] = sau
    sau.engage(theta=trim[0], h=cfg.h0, Va=cfg.Va0, chi=0.0)
    sau.set_mode(Mode.ALTITUDE)
    rec = {"meas": [], "bias": [], "drift": []}

    def controls_fn(t, state, Va, alpha):
        sau.refs.chi = 0.0 if t < T_TURN else CHI2
        sau.refs.h = cfg.h0 if t < T_CLIMB else H2
        _, _, beta = air_data(state, wind_fn(state[H], t))
        c = sau.step(t, state, Va, alpha, beta, cfg.dt)
        rec["meas"].append(dict(sau.est))
        if kind == "real":
            rec["bias"].append(sens.gyro_bias().copy())
            rec["drift"].append(sens.gps_drift.b.copy())
        return c

    log = run(controls_fn, ac, wp, cfg, state0=trim_state(ac, cfg))
    return log, rec


res = {k: flight(k) for k in ("truth", "noisy", "real")}
t = res["truth"][0].t
print(f"   {'датчики':>8} {'|e_χ| 60…80 с, °':>17} {'φ max, °':>9} {'|Δh| 140…180, м':>16} "
      f"{'σδa, °':>7} {'σδe, °':>7} {'полёт':>6}")
for k, (log, rec) in res.items():
    Vx, Vy = np.array([earth_velocity(s)[:2] for s in log.state]).T
    chi = np.arctan2(Vy, Vx)
    m1 = (log.t > 60) & (log.t < 80)
    m2 = log.t > 140
    e_chi = deg(np.abs(wrap_angle(chi[m1] - CHI2))).max()
    dh = np.abs(log.state[m2, H] - H2).max()
    sda = deg(np.std(np.diff(log.controls[:, 2])))
    sde = deg(np.std(np.diff(log.controls[:, 0])))
    full = log.t[-1] > cfg.t_end - 0.05
    print(f"   {k:>8} {e_chi:17.2f} {deg(np.abs(log.state[:, PHI]).max()):9.1f} {dh:16.2f} "
          f"{sda:7.3f} {sde:7.3f} {'да' if full else 'НЕТ':>6}")
    check(f"{k}: полёт 180 с, |e_χ| < 3°, |Δh| < 3 м", full and e_chi < 3.0 and dh < 3.0)
print("   σδa, σδe — СКО приращения руля за шаг 0.01 с: «дёрганье» рулей от шума")

# ошибки измерений реалистичного набора
log, rec = res["real"]
meas = rec["meas"]
pn = np.array([m["pn"] for m in meas]); pe = np.array([m["pe"] for m in meas])
e_pos = np.hypot(pn - log.state[:, X], pe - log.state[:, Y])
drift = np.array(rec["drift"])
e_drift = np.hypot(*drift.T)
bias = deg(np.array(rec["bias"]))
print(f"   real: ошибка GPS-положения относительно текущей истины (уход + запаздывание) — "
      f"{e_pos.mean():.1f} м в среднем, {e_pos.max():.1f} м макс.; из них уход — "
      f"{e_drift.mean():.1f} м в среднем")
print(f"   real: смещение гироскопа (p, q, r) в начале {np.round(bias[0], 2)} °/с, "
      f"изменение за полёт {np.round(np.abs(bias[-1] - bias[0]), 3)} °/с")

# ---------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 7.5))
C_TRUE, C_MEAS, C_REAL = "#222222", "#d9822b", "#3b75af"
m = (t >= 79.8) & (t <= 81.2)
axs[0, 0].plot(t[m], log.state[m, H], color=C_TRUE, lw=1.5, label="истина")
axs[0, 0].step(t[m], [meas[i]["h"] for i in np.where(m)[0]], where="post", color=C_MEAS,
               lw=1.2, label=f"барометр {sp.baro_rate:.0f} Гц, σ = {sp.baro_noise} м")
axs[0, 0].set_title("Высота: отсчёты барометра с удержанием", fontsize=10)
axs[0, 0].set_xlabel("t, с"); axs[0, 0].set_ylabel("h, м"); axs[0, 0].legend(fontsize=8)

m = (t >= 19.5) & (t <= 23.0)
chi_true = deg(np.unwrap(np.arctan2(*np.array([earth_velocity(s)[:2] for s in log.state[m]]).T[::-1])))
axs[0, 1].plot(t[m], chi_true, color=C_TRUE, lw=1.5, label="истина")
axs[0, 1].step(t[m], deg(np.unwrap([meas[i]["chi"] for i in np.where(m)[0]])), where="post",
               color=C_MEAS, lw=1.2, label=f"GPS {sp.gps_rate:.0f} Гц, задержка {sp.gps_delay} с")
axs[0, 1].set_title("Путевой угол в развороте: частота и запаздывание GPS", fontsize=10)
axs[0, 1].set_xlabel("t, с"); axs[0, 1].set_ylabel("χ, °"); axs[0, 1].legend(fontsize=8)

axs[1, 0].plot(t, pn - log.state[:, X], color=C_REAL, lw=0.6, alpha=0.45,
               label="полная ошибка N (уход + запаздывание)")
axs[1, 0].plot(t, pe - log.state[:, Y], color=C_MEAS, lw=0.6, alpha=0.45,
               label="полная ошибка E (уход + запаздывание)")
axs[1, 0].plot(t, drift[:, 0], color=C_REAL, lw=2.0, label="уход N")
axs[1, 0].plot(t, drift[:, 1], color=C_MEAS, lw=2.0, label="уход E")
axs[1, 0].set_title(f"Ошибка GPS-положения: уход σ = {sp.gps_pos_drift_sigma} м, "
                    f"τ = {sp.gps_pos_drift_tau:.0f} с (B&M 7.5)", fontsize=10)
axs[1, 0].set_xlabel("t, с"); axs[1, 0].set_ylabel("м"); axs[1, 0].legend(fontsize=7, ncol=2)

for k, col in (("truth", C_TRUE), ("real", C_REAL)):
    lg = res[k][0]
    mm = (lg.t > 15) & (lg.t < 35)
    axs[1, 1].plot(lg.t[mm], deg(lg.state[mm, PHI]), color=col, lw=1.3,
                   label={"truth": "САУ на истине", "real": "САУ на RealisticSensors"}[k])
axs[1, 1].set_title("Крен в развороте: истина против реалистичных датчиков", fontsize=10)
axs[1, 1].set_xlabel("t, с"); axs[1, 1].set_ylabel("φ, °"); axs[1, 1].legend(fontsize=8)
for ax in axs.flat:
    ax.grid(True, color="#dddddd", lw=0.6)
fig.suptitle("Реалистичные датчики: FPV, Va = 16 м/с, штиль (моделирование)", fontsize=11)
fig.tight_layout()
png = os.path.join(os.path.dirname(__file__), "sensors_real.png")
fig.savefig(png, dpi=110)
print(f"\n   график: {os.path.relpath(png, ROOT)}")

print("\nALL SENSOR CHECKS PASSED" if ok_all else "\nSOME SENSOR CHECKS FAILED")

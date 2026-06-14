# -*- coding: utf-8 -*-
"""
Проверка точности источников УА: зонд vs косвенная оценка (ИНС+GPS).

Что измеряем:
  err_probe[t] = alpha_probe[t] - alpha_true[t]
  err_est[t]   = alpha_est[t]   - alpha_true[t]

Разложение ошибки на два компонента:
  Систематическая = |mean(err)|   — постоянное смещение (bias)
  Стохастическая  = std(err)      — случайный шум (σ)

Физика систематической ошибки оценщика:
  alpha_est = theta_meas − gamma_gps,
  gamma_gps = atan(Vh_earth / Vx_earth) — по ЗЕМНОЙ скорости.
  При горизонтальном ветре Vwx:
    Vx_earth = Vx_air + Vwx  →  gamma_gps ≠ gamma_air
    delta_alpha ≈ Vwx · Vh_air / Va²

  При снижении (Vh < 0) и встречном ветре (Vwx < 0):
    delta_alpha > 0  → оценщик ЗАВЫШАЕТ УА.

Схема:
  Одни и те же seed'ы (42 — общие датчики, 99 — зонд), как в s9.
  Три режима ветра (Vwx = 0, −5, −10 м/с) → три группы баров.

Запуск:  python checks/check_alpha_error.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import THETA, Q, H, U, W
from runner import run, compute_trim, trim_state
from control.sensors import (measure_gyro, measure_altitude, measure_airspeed,
                              measure_angle_of_attack, measure_gps_velocity_earth)
from control.estimators import estimate_alpha_indirect

plt.rcParams["font.family"] = "DejaVu Sans"

# ══════════════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ
# ══════════════════════════════════════════════════════════════════════════════
aircraft = AircraftParams()
sp       = SensorParams()
cfg      = SimConfig(Va0=30.0, h0=150.0, dt=0.01, t_end=60.0)

# Угол тангажа: чуть ниже тримового → устойчивое снижение (Vh ≠ 0)
# При Vh ≠ 0 и Vwx ≠ 0 оценщик получает систематическое смещение
THETA_DESCENT_OFFSET = np.radians(-3.0)   # рад ниже тримового

WIND_CASES = [
    ("Штиль\n(Vwx = 0)",    0.0),
    ("Встречный\n(Vwx = −5)",  -5.0),
    ("Встречный\n(Vwx = −10)", -10.0),
]

SEED_COMMON = 42
SEED_PROBE  = 99


# ══════════════════════════════════════════════════════════════════════════════
#  ФУНКЦИЯ ОДНОГО ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════
def run_case(Vwx: float):
    """
    Запускает прогон с фиксированными управлениями (трим + коррекция тангажа).
    Возвращает массивы ошибок (градусы) за все шаги прогона.
    """
    wind_params = WindParams(Vw_const=Vwx)

    alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)

    # Начальное состояние: трим, но theta чуть ниже → снижение
    s0 = trim_state(aircraft, cfg)
    s0[THETA] += THETA_DESCENT_OFFSET

    rng_common = np.random.default_rng(seed=SEED_COMMON)
    rng_probe  = np.random.default_rng(seed=SEED_PROBE)

    err_probe_list = []
    err_est_list   = []

    def controls_fn(t, state, Va, alpha_true):
        # ── Общие датчики (оба метода используют одинаково) ──────────────────
        theta_meas = state[THETA] + rng_common.normal(0.0, sp.gyro_noise)
        Vx_gps, Vh_gps = measure_gps_velocity_earth(
            state[U], state[W], state[THETA],
            bias=0.0, noise_std=sp.gps_vel_noise, rng=rng_common)

        # ── Зонд (прямое измерение) ───────────────────────────────────────────
        alpha_probe = measure_angle_of_attack(
            alpha_true, sp.probe_bias, sp.probe_noise, rng_probe)

        # ── Оценщик (ИНС + GPS) ───────────────────────────────────────────────
        alpha_est = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)

        # ── Ошибки в градусах ─────────────────────────────────────────────────
        err_probe_list.append(np.degrees(alpha_probe - alpha_true))
        err_est_list.append(np.degrees(alpha_est   - alpha_true))

        # Управление: фиксированный трим (без контроллера — чистый сбор данных)
        return np.array([de_trim, thr_trim])

    run(controls_fn, aircraft, wind_params, cfg, state0=s0)

    return np.array(err_probe_list), np.array(err_est_list)


# ══════════════════════════════════════════════════════════════════════════════
#  ПРОГОНЫ
# ══════════════════════════════════════════════════════════════════════════════
results = []  # list of (label, err_probe, err_est)
for label, Vwx in WIND_CASES:
    print(f"Прогон: {label.replace(chr(10), ' ')} ...")
    ep, ee = run_case(Vwx)
    results.append((label, ep, ee))


# ══════════════════════════════════════════════════════════════════════════════
#  ВЫВОД ЧИСЛОВЫХ ИТОГОВ
# ══════════════════════════════════════════════════════════════════════════════
print()
print("══════════════════════════════════════════════════════════")
print("  ОШИБКИ УА (°)     │  Систематическая  │  Стохастическая")
print("──────────────────────────────────────────────────────────")
for label, ep, ee in results:
    lbl = label.replace('\n', ' ')
    print(f"  {lbl}")
    print(f"    Зонд:      sys={np.mean(ep):+.3f}°  stoch={np.std(ep):.3f}°")
    print(f"    Оценщик:   sys={np.mean(ee):+.3f}°  stoch={np.std(ee):.3f}°")
    # Проверка: систематическая ошибка оценщика должна расти с ветром
print("══════════════════════════════════════════════════════════")


# ══════════════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1 + len(WIND_CASES), 1,
                         figsize=(10, 4 + 3 * len(WIND_CASES)),
                         gridspec_kw={"height_ratios": [3] + [1] * len(WIND_CASES)})
fig.suptitle(
    "Сравнение погрешностей измерения УА: зонд vs оценщик (ИНС+GPS)\n"
    f"Va={cfg.Va0:.0f} м/с, снижение, θ offset={np.degrees(THETA_DESCENT_OFFSET):.0f}°",
    fontsize=11, fontweight="bold"
)

# ── Столбчатая диаграмма (верхняя панель) ────────────────────────────────────
ax_bar = axes[0]

n_cases = len(results)
x       = np.arange(n_cases)
w       = 0.32     # ширина одного бара

# Цвета
C_SYS   = "#e57373"    # систематическая — красноватый
C_STOCH = "#ffb74d"    # стохастическая  — оранжевый
C_SYS_P = "#64b5f6"    # систематическая зонда — голубой (≈0, едва видно)
C_STH_P = "#81c784"    # стохастическая зонда  — зелёный

for i, (label, ep, ee) in enumerate(results):
    sys_p  = abs(np.mean(ep))
    stch_p = np.std(ep)
    sys_e  = abs(np.mean(ee))
    stch_e = np.std(ee)

    # Зонд: sys (низ) + stoch (верх)
    ax_bar.bar(x[i] - w/2, sys_p,  width=w, color=C_SYS_P, label="зонд: систем." if i==0 else "")
    ax_bar.bar(x[i] - w/2, stch_p, width=w, bottom=sys_p,
               color=C_STH_P, hatch="///", label="зонд: стохаст." if i==0 else "")

    # Оценщик: sys (низ) + stoch (верх)
    ax_bar.bar(x[i] + w/2, sys_e,  width=w, color=C_SYS,   label="оценщик: систем." if i==0 else "")
    ax_bar.bar(x[i] + w/2, stch_e, width=w, bottom=sys_e,
               color=C_STOCH, hatch="...", label="оценщик: стохаст." if i==0 else "")

    # Числа на барах
    total_p = sys_p + stch_p
    total_e = sys_e + stch_e
    ax_bar.text(x[i] - w/2, total_p + 0.01, f"{total_p:.2f}°",
                ha="center", va="bottom", fontsize=8, color="navy")
    ax_bar.text(x[i] + w/2, total_e + 0.01, f"{total_e:.2f}°",
                ha="center", va="bottom", fontsize=8, color="darkred")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([r[0] for r in results], fontsize=9)
ax_bar.set_ylabel("|ошибка|, °", fontsize=9)
ax_bar.set_title("Суммарная погрешность = |систематическая| + стохастическая (СКО)", fontsize=9)
ax_bar.legend(fontsize=8, ncol=2, loc="upper left")
ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.15)
ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

# Подписи под осью: зонд / оценщик
for i in range(n_cases):
    ax_bar.text(x[i] - w/2, -ax_bar.get_ylim()[1]*0.08, "зонд",
                ha="center", fontsize=7, color="navy")
    ax_bar.text(x[i] + w/2, -ax_bar.get_ylim()[1]*0.08, "оценщ.",
                ha="center", fontsize=7, color="darkred")

# ── Временные ряды ошибок (нижние панели) ────────────────────────────────────
t_arr = np.arange(len(results[0][1])) * cfg.dt

for ax, (label, ep, ee) in zip(axes[1:], results):
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.plot(t_arr, ep, color="steelblue", lw=1.0, alpha=0.8, label="Зонд")
    ax.plot(t_arr, ee, color="tomato",    lw=1.0, alpha=0.8, label="Оценщик")
    # Средние (систематические) линии
    ax.axhline(np.mean(ep), color="steelblue", lw=1.5, ls="--",
               label=f"mean зонда {np.mean(ep):+.3f}°")
    ax.axhline(np.mean(ee), color="tomato",    lw=1.5, ls="--",
               label=f"mean оценщ. {np.mean(ee):+.3f}°")
    ax.set_ylabel("err, °", fontsize=8)
    ax.set_title(label.replace('\n', ' '), fontsize=8)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=7)

axes[-1].set_xlabel("Время, с", fontsize=9)

plt.tight_layout()
plt.show()

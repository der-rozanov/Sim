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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Русский вывод в консоль Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Добавить корень проекта в путь поиска модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from runner import run, compute_trim, trim_state, print_summary
from sim.state import THETA, Q, H, X, U, W
from flight_logger import FlightLogger

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры прогона
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()          # нулевой ветер
cfg = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=30.0)

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
log = run(lambda t, s, Va, alpha: ctrl_arr,
          aircraft, wind_params, cfg, state0=s0)

print_summary(log, aircraft, label="С1  Установившийся полёт")
logger.save(log)

# ------------------------------------------------------------------
# Вычислить отклонения от балансировочного режима
# ------------------------------------------------------------------
Va_arr    = log.Va
alpha_arr = log.alpha
theta_arr = log.state[:, THETA]
q_arr     = log.state[:, Q]
h_arr     = log.state[:, H]
de_arr    = log.controls[:, 0]
thr_arr   = log.controls[:, 1]
t_arr     = log.t

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
# Построение графика
# ------------------------------------------------------------------
warn_deg  = np.degrees(aircraft.alpha_warning)
crit_deg  = np.degrees(aircraft.alpha_crit)
alpha_deg = np.degrees(alpha_arr)
theta_deg = np.degrees(theta_arr)
q_deg     = np.degrees(q_arr)
de_deg    = np.degrees(de_arr)
dE_kJ     = (log.E_total - log.E_total[0]) / 1000.0

fig = plt.figure(figsize=(13, 10))
fig.suptitle(
    "С1: Установившийся полёт (балансировочный режим)\n"
    f"Va = {cfg.Va0:.0f} м/с,  α_трим = {np.degrees(alpha_trim):.2f}°,"
    f"  δe = {np.degrees(de_trim):.2f}°,  газ = {thr_trim:.3f}",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38)

# ---- Воздушная скорость ----
ax_Va = fig.add_subplot(gs[0, 0])
ax_Va.plot(t_arr, Va_arr, color="steelblue", lw=1.5)
ax_Va.axhline(cfg.Va0, color="black", lw=0.9, ls="--", alpha=0.5,
              label=f"Va_трим = {cfg.Va0:.1f} м/с")
ax_Va.set_ylabel("Va, м/с", fontsize=9)
ax_Va.legend(fontsize=7)
ax_Va.grid(True, ls="--", alpha=0.5)
ax_Va.set_title(f"Макс. |ΔVa| = {dVa_max:.2e} м/с", fontsize=8)

# ---- Угол атаки ----
ax_al = fig.add_subplot(gs[0, 1])
ax_al.plot(t_arr, alpha_deg, color="crimson", lw=1.5)
ax_al.axhline(np.degrees(alpha_trim), color="black", lw=0.9, ls="--", alpha=0.5,
              label=f"α_трим = {np.degrees(alpha_trim):.2f}°")
ax_al.axhline(warn_deg, color="orange", lw=1.0, ls=":", alpha=0.8,
              label=f"пред. {warn_deg:.0f}°")
ax_al.axhline(crit_deg, color="red", lw=1.0, ls=":", alpha=0.8,
              label=f"крит. {crit_deg:.0f}°")
ax_al.set_ylabel("α, °", fontsize=9)
ax_al.legend(fontsize=7)
ax_al.grid(True, ls="--", alpha=0.5)
ax_al.set_title(f"Макс. |Δα| = {dalpha_max:.2e} °", fontsize=8)

# ---- Угол тангажа ----
ax_th = fig.add_subplot(gs[1, 0])
ax_th.plot(t_arr, theta_deg, color="seagreen", lw=1.5)
ax_th.axhline(np.degrees(alpha_trim), color="black", lw=0.9, ls="--", alpha=0.5,
              label=f"θ_трим = {np.degrees(alpha_trim):.2f}°")
ax_th.set_ylabel("θ, °", fontsize=9)
ax_th.legend(fontsize=7)
ax_th.grid(True, ls="--", alpha=0.5)
ax_th.set_title(f"Макс. |Δθ| = {dtheta_max:.2e} °", fontsize=8)

# ---- Угловая скорость тангажа ----
ax_q = fig.add_subplot(gs[1, 1])
ax_q.plot(t_arr, q_deg, color="mediumpurple", lw=1.5)
ax_q.axhline(0, color="gray", lw=0.8, ls=":")
ax_q.set_ylabel("q, °/с", fontsize=9)
ax_q.grid(True, ls="--", alpha=0.5)
ax_q.set_title(f"Макс. |q| = {dq_max:.2e} °/с", fontsize=8)

# ---- Высота ----
ax_h = fig.add_subplot(gs[2, 0])
ax_h.plot(t_arr, h_arr, color="royalblue", lw=1.5)
ax_h.axhline(cfg.h0, color="black", lw=0.9, ls="--", alpha=0.5,
             label=f"h₀ = {cfg.h0:.0f} м")
ax_h.set_ylabel("h, м", fontsize=9)
ax_h.legend(fontsize=7)
ax_h.grid(True, ls="--", alpha=0.5)
ax_h.set_title(f"Макс. |Δh| = {dh_max:.2e} м", fontsize=8)

# ---- Отклонение полной энергии ----
ax_E = fig.add_subplot(gs[2, 1])
ax_E.plot(t_arr, dE_kJ * 1000, color="black", lw=1.5)   # Дж
ax_E.axhline(0, color="gray", lw=0.8, ls=":")
ax_E.set_ylabel("ΔE_полн., Дж", fontsize=9)
ax_E.grid(True, ls="--", alpha=0.5)
ax_E.set_title(f"Дрейф энергии = {dE_pct:.3f} %  (верификация RK4)", fontsize=8)

# ---- Управление: руль высоты + газ ----
ax_ctrl = fig.add_subplot(gs[3, :])
color_de  = "saddlebrown"
color_thr = "darkgreen"
ax_ctrl.plot(t_arr, de_deg,  color=color_de,  lw=1.4,
             label=f"δe = {np.degrees(de_trim):.2f}° (фиксированный)")
ax_ctrl.plot(t_arr, thr_arr, color=color_thr, lw=1.4, ls="--",
             label=f"газ = {thr_trim:.3f} (фиксированный)")
ax_ctrl.set_ylabel("Управление", fontsize=9)
ax_ctrl.set_xlabel("Время, с", fontsize=9)
ax_ctrl.legend(fontsize=8, loc="center right")
ax_ctrl.grid(True, ls="--", alpha=0.5)
ax_ctrl.set_title("Управление: балансировочные значения (константа)", fontsize=8)

# Метка времени — ось x для верхних строк
for ax in (ax_Va, ax_al, ax_th, ax_q, ax_h, ax_E):
    ax.set_xlabel("Время, с", fontsize=8)
    ax.tick_params(labelsize=8)
ax_ctrl.tick_params(labelsize=8)

# Вставка с числовыми итогами
summary_text = (
    f"Балансировочный режим:\n"
    f"  Va    = {cfg.Va0:.1f} м/с\n"
    f"  α     = {np.degrees(alpha_trim):.3f}°\n"
    f"  δe    = {np.degrees(de_trim):.3f}°\n"
    f"  газ   = {thr_trim:.4f}\n"
    f"\nМакс. отклонения за {cfg.t_end:.0f} с:\n"
    f"  |ΔVa| ≤ {dVa_max:.1e} м/с\n"
    f"  |Δα|  ≤ {dalpha_max:.1e}°\n"
    f"  |Δh|  ≤ {dh_max:.1e} м\n"
    f"  ΔE/E₀ = {dE_pct:.3f} %"
)
fig.text(0.99, 0.01, summary_text,
         ha="right", va="bottom", fontsize=8,
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                   edgecolor="goldenrod", alpha=0.9))

fig.subplots_adjust(top=0.89, bottom=0.07, left=0.08, right=0.82, hspace=0.55, wspace=0.38)

# ------------------------------------------------------------------
# Сохранение / показ
# ------------------------------------------------------------------
save_path = sys.argv[1] if len(sys.argv) > 1 else None

if save_path is not None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {save_path}")

plt.show()

"""
Проверка 02: Силовая установка — тяговооружённость и диапазон скоростей.

Модель тяги (Beard & McLain, упрощённая):
  T = 0.5 · ρ · S_prop · C_prop · ((k_motor · δt)² − Va²)

Что считаем:
  1. T(δt) при фиксированных Va — полные тяговые характеристики.
  2. T(Va) при фиксированных δt — «скоростные» кривые.
  3. Потребная тяга уровня (L=W) vs Va — баланс сил.
  4. Диапазон допустимых скоростей: Va_stall ≤ Va ≤ Va_max.
  5. Тяговооружённость: T_max / (m·g).
  6. Балансировочный газ δt_trim(Va): трим по скорости.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AircraftParams
from aero import coef_CL, coef_CD

params = AircraftParams()

# ---------------------------------------------------------------------------
# Формула тяги
# ---------------------------------------------------------------------------

def thrust(throttle: float, Va: float) -> float:
    Vmotor = params.k_motor * throttle
    return 0.5 * params.rho * params.S_prop * params.C_prop * (Vmotor**2 - Va**2)


# ---------------------------------------------------------------------------
# Потребная тяга для горизонтального полёта при данной Va
# Условие балансировки: L = W  →  CL = 2mg / (ρ Va² S)
# T_req = D = q·S·CD(alpha_eq)
# ---------------------------------------------------------------------------

def level_flight_required_thrust(Va: float) -> tuple:
    """
    При Va найти alpha_eq из L=W, затем T_req = D.
    Возвращает (T_req, alpha_eq, CL_eq, CD_eq) или NaN если нет решения.
    """
    W = params.mass * params.g
    q_dyn = 0.5 * params.rho * Va**2
    CL_needed = W / (q_dyn * params.S)

    # Ищем alpha в [-15°, 30°] такой что coef_CL(alpha, 0, 0, Va) ≈ CL_needed
    alpha_arr = np.linspace(np.deg2rad(-15), np.deg2rad(30), 2000)
    cl_arr = np.array([coef_CL(a, 0.0, 0.0, Va, params) for a in alpha_arr])

    # Ищем пересечение
    diff = cl_arr - CL_needed
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) == 0:
        return np.nan, np.nan, CL_needed, np.nan

    # Линейная интерполяция в точке пересечения (первый корень)
    i = idx[0]
    t = -diff[i] / (diff[i+1] - diff[i])
    alpha_eq = alpha_arr[i] + t * (alpha_arr[i+1] - alpha_arr[i])
    CD_eq = coef_CD(alpha_eq, params)
    T_req = q_dyn * params.S * CD_eq
    return T_req, alpha_eq, CL_needed, CD_eq


# ---------------------------------------------------------------------------
# Числовые характеристики
# ---------------------------------------------------------------------------

# --- Максимальная скорость: T=0 при δt=1  →  Va_max = k_motor
Va_max_thrust = params.k_motor  # м/с

# --- Скорость сваливания: CL_max из модели
alpha_arr = np.linspace(np.deg2rad(-5), np.deg2rad(35), 5000)
CL_arr = np.array([coef_CL(a, 0.0, 0.0, 30.0, params) for a in alpha_arr])
idx_CLmax = np.argmax(CL_arr)
CL_max = CL_arr[idx_CLmax]
alpha_stall_actual = np.rad2deg(alpha_arr[idx_CLmax])
Va_stall = np.sqrt(2 * params.mass * params.g / (params.rho * params.S * CL_max))

# --- Максимальная тяга (δt=1, Va=0 — статическая)
T_static_max = thrust(1.0, 0.0)

# --- Тяговооружённость
weight = params.mass * params.g
TWR = T_static_max / weight

# --- Диапазон скоростей горизонтального полёта
Va_sweep = np.linspace(Va_stall + 0.5, Va_max_thrust - 1, 300)
T_available = np.array([thrust(1.0, Va) for Va in Va_sweep])
T_req_arr, alpha_eq_arr = [], []
for Va in Va_sweep:
    Tr, aeq, _, _ = level_flight_required_thrust(Va)
    T_req_arr.append(Tr)
    alpha_eq_arr.append(np.rad2deg(aeq) if not np.isnan(aeq) else np.nan)
T_req_arr   = np.array(T_req_arr)
alpha_eq_arr = np.array(alpha_eq_arr)

# Диапазон, где T_available > T_req (мотор обеспечивает горизонт)
valid_mask = T_available > T_req_arr
Va_min_motor = Va_sweep[valid_mask][0]  if valid_mask.any() else np.nan
Va_max_motor = Va_sweep[valid_mask][-1] if valid_mask.any() else np.nan

# Балансировочный газ δt_trim(Va)
# T_req = 0.5·ρ·S_prop·C_prop·((k_motor·δt)² − Va²)  →  δt = sqrt(Va² + T_req/(0.5·ρ·S_prop·C_prop)) / k_motor
coeff = 0.5 * params.rho * params.S_prop * params.C_prop
delta_t_trim = np.sqrt(np.clip(Va_sweep**2 + T_req_arr / coeff, 0, None)) / params.k_motor
delta_t_trim = np.clip(delta_t_trim, 0.0, 1.0)

print("=" * 58)
print("  Силовая установка (параметры-аналог Aerosonde)")
print("=" * 58)
print(f"  k_motor              = {params.k_motor:.1f} м/с")
d_prop = np.sqrt(4 * params.S_prop / np.pi) * 100
print(f"  S_prop               = {params.S_prop:.4f} m^2  (d~{d_prop:.0f} cm)")
print(f"  C_prop               = {params.C_prop:.2f}")
print()
print(f"  Статическая тяга (dt=1, Va=0)  = {T_static_max:.1f} N")
print(f"  Масса ЛА                       = {params.mass:.1f} кг  (вес {weight:.1f} N)")
print(f"  Тяговооруженность T/W (static) = {TWR:.2f}")
print()
print(f"  Va_stall                       = {Va_stall:.1f} м/с  ({Va_stall*3.6:.0f} км/ч)")
print(f"    CL_max = {CL_max:.3f}  при alpha_stall ~ {alpha_stall_actual:.1f} град")
print(f"  Va_max (T=0, dt=1)             = {Va_max_thrust:.1f} м/с  ({Va_max_thrust*3.6:.0f} км/ч)")
print()
print(f"  Диапазон горизонт. полета (dt <= 1):")
print(f"    Va_min (мотор >= сопротивл.) = {Va_min_motor:.1f} м/с  ({Va_min_motor*3.6:.0f} км/ч)")
print(f"    Va_max (тяга -> 0)           = {Va_max_motor:.1f} м/с  ({Va_max_motor*3.6:.0f} км/ч)")
print()
# Тяга и трим на Va=30
T30_max   = thrust(1.0, 30.0)
T30_req, a30, _, _ = level_flight_required_thrust(30.0)
dt30_trim = np.sqrt(30.0**2 + T30_req / coeff) / params.k_motor
print(f"  Рабочая точка Va = 30 м/с:")
print(f"    T_max (dt=1) = {T30_max:.1f} N")
print(f"    T_req (L=W)  = {T30_req:.1f} N")
print(f"    alpha_eq     = {np.rad2deg(a30):.2f} grad")
print(f"    dt_trim      = {dt30_trim:.3f}")
print("=" * 58)

# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Силовая установка ЛА (параметры-аналог Aerosonde)", fontsize=13)

# --- 1. T(δt) при фиксированных Va ---
ax = axes[0, 0]
for Va_fix in [0, 15, 30, 45, 60]:
    dt_arr = np.linspace(0, 1, 200)
    T_arr  = np.array([thrust(d, Va_fix) for d in dt_arr])
    ax.plot(dt_arr, T_arr, label=f"Va={Va_fix} м/с")
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.axhline(weight, color="gray", linewidth=1, linestyle=":", label=f"Вес ЛА {weight:.0f} Н")
ax.set_xlabel("Газ δt")
ax.set_ylabel("Тяга T, Н")
ax.set_title("Тяга vs газ при фиксированных Va")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# --- 2. T(Va) при фиксированных δt ---
ax = axes[0, 1]
Va_range = np.linspace(0, params.k_motor, 300)
for dt_fix in [0.25, 0.5, 0.75, 1.0]:
    T_arr = np.array([thrust(dt_fix, Va) for Va in Va_range])
    ax.plot(Va_range, T_arr, label=f"δt={dt_fix:.2f}")
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.plot(Va_sweep, T_req_arr, "k--", linewidth=2, label="T_req (L=W)")
ax.axvline(Va_stall,        color="orange", linestyle=":", linewidth=1.5, label=f"Va_stall={Va_stall:.0f}")
ax.axvline(Va_max_thrust,   color="red",    linestyle=":", linewidth=1.5, label=f"Va_max={Va_max_thrust:.0f}")
ax.set_xlim(0, params.k_motor + 5)
ax.set_xlabel("Воздушная скорость Va, м/с")
ax.set_ylabel("Тяга T, Н")
ax.set_title("Тяга vs Va при фиксированных δt")
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.4)

# --- 3. Запас тяги и балансировочный газ ---
ax = axes[1, 0]
thrust_margin = T_available - T_req_arr
ax.plot(Va_sweep * 3.6, thrust_margin, "b-", linewidth=2, label="Запас тяги (δt=1 - T_req)")
ax.fill_between(Va_sweep * 3.6, 0, thrust_margin,
                where=(thrust_margin > 0), alpha=0.15, color="blue", label="Избыток")
ax.fill_between(Va_sweep * 3.6, 0, thrust_margin,
                where=(thrust_margin < 0), alpha=0.2, color="red", label="Дефицит")
ax.axhline(0, color="k", linewidth=1)
ax.axvline(Va_stall * 3.6,      color="orange", linestyle=":", label=f"Va_stall={Va_stall*3.6:.0f} км/ч")
ax.axvline(Va_max_motor * 3.6,  color="red",    linestyle="--", label=f"Va_max={Va_max_motor*3.6:.0f} км/ч")
ax.set_xlabel("Воздушная скорость, км/ч")
ax.set_ylabel("Запас тяги, Н")
ax.set_title("Запас тяги для горизонтального полёта (δt=1)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# --- 4. Балансировочный газ и угол атаки ---
ax = axes[1, 1]
l1, = ax.plot(Va_sweep * 3.6, delta_t_trim, "g-", linewidth=2, label="δt_trim")
ax.set_xlabel("Воздушная скорость, км/ч")
ax.set_ylabel("Балансировочный газ δt_trim", color="g")
ax.tick_params(axis="y", labelcolor="g")
ax.axvline(Va_stall * 3.6,     color="orange", linestyle=":", linewidth=1.5, label=f"Va_stall")
ax.axvline(30.0 * 3.6,         color="purple", linestyle="--", linewidth=1.2, label="Va=30 м/с")
ax.set_ylim(0, 1.1)

ax2 = ax.twinx()
ax2.plot(Va_sweep * 3.6, alpha_eq_arr, "b--", linewidth=1.5, label="α_eq (L=W)")
ax2.set_ylabel("УА равновесия α_eq, °", color="b")
ax2.tick_params(axis="y", labelcolor="b")
ax2.axhline(np.rad2deg(params.alpha_warning), color="orange", linestyle=":", linewidth=1)
ax2.axhline(np.rad2deg(params.alpha_crit),    color="red",    linestyle=":", linewidth=1)

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=7.5, loc="upper right")
ax.set_title("Балансировочный газ и равновесный УА vs Va")
ax.grid(True, alpha=0.4)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "motor.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  График сохранён: {out_path}")
plt.show()

"""
Проверка: оптимальная скорость горизонтального полёта.

Цель: найти Va, при котором мощность мотора P = T·Va минимальна
(наилучшая продолжительность полёта), и Va при максимальном L/D
(наилучшая дальность).

Метод:
  В установившемся горизонтальном полёте: T = D, L = W.
  Из L = W: CL(Va) = 2·m·g / (ρ·Va²·S)
  alpha(Va) обратным ходом из CL: alpha = (CL − CL0) / CLa  (линейный участок)
  D(Va)  = 0.5·ρ·Va²·S·CD(alpha)
  P(Va)  = D·Va  [Вт]

Две ключевые точки аналитически (квадратичная поляра):
  Макс. качество (мин. D, макс. дальность):
    CL_md  = sqrt(CDp·π·e·AR)
    Va_md  = sqrt(2·m·g / (ρ·S·CL_md))

  Мин. мощность (макс. продолжительность):
    CL_mp  = sqrt(3·CDp·π·e·AR)  =  √3 · CL_md
    Va_mp  = sqrt(2·m·g / (ρ·S·CL_mp))  =  (1/3^0.25) · Va_md  ≈ 0.76 · Va_md

Запуск: python checks/check_optimal_speed.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams
from sim.aero import coef_CL, coef_CD
from sim.dynamics import thrust as motor_thrust

params = AircraftParams()

# ---------------------------------------------------------------------------
# Аналитические точки (квадратичная поляра)
# ---------------------------------------------------------------------------
AR     = params.b**2 / params.S
CDp    = params.CDp
e      = params.e_oswald
m, g   = params.mass, params.g
rho, S = params.rho, params.S

CL_md = np.sqrt(CDp * np.pi * e * AR)          # оптимум дальности
CL_mp = np.sqrt(3 * CDp * np.pi * e * AR)       # оптимум продолжительности

Va_md = np.sqrt(2 * m * g / (rho * S * CL_md))  # скорость макс. качества
Va_mp = np.sqrt(2 * m * g / (rho * S * CL_mp))  # скорость мин. мощности

# alpha при этих режимах (линейная модель)
alpha_md_deg = np.degrees((CL_md - params.CL0) / params.CLa)
alpha_mp_deg = np.degrees((CL_mp - params.CL0) / params.CLa)

CD_md = CDp + CL_md**2 / (np.pi * e * AR)
CD_mp = CDp + CL_mp**2 / (np.pi * e * AR)

D_md = 0.5 * rho * Va_md**2 * S * CD_md   # мин. тяга (Н)
D_mp = 0.5 * rho * Va_mp**2 * S * CD_mp   # тяга при мин. мощности (Н)
P_md = D_md * Va_md                         # мощность при макс. качестве (Вт)
P_mp = D_mp * Va_mp                         # мин. мощность (Вт)

K_md = CL_md / CD_md                        # максимальное качество

# ---------------------------------------------------------------------------
# Стояночная (минимальная) скорость: Va_stall при CL_max
# ---------------------------------------------------------------------------
# Ищем CL_max численно (sigmoid-модель)
alpha_arr = np.linspace(-0.1, params.alpha_stall * 1.1, 300)
CL_arr = np.array([coef_CL(a, 0.0, 0.0, Va_md, params) for a in alpha_arr])
CL_max = CL_arr.max()
Va_stall = np.sqrt(2 * m * g / (rho * S * CL_max))

# ---------------------------------------------------------------------------
# Численный свип Va → P(Va)
# ---------------------------------------------------------------------------
Va_sweep = np.linspace(Va_stall * 1.05, 50.0, 800)

D_arr     = np.full_like(Va_sweep, np.nan)
P_arr     = np.full_like(Va_sweep, np.nan)
alpha_arr_deg = np.full_like(Va_sweep, np.nan)
K_arr     = np.full_like(Va_sweep, np.nan)

for i, Va in enumerate(Va_sweep):
    CL_req = 2 * m * g / (rho * Va**2 * S)
    if CL_req > CL_max or CL_req < -1.0:
        continue
    # Обратная задача: alpha из линейной модели CL
    alpha = (CL_req - params.CL0) / params.CLa
    if alpha < -0.3 or alpha > params.alpha_stall:
        continue
    CD = coef_CD(alpha, params)
    D  = 0.5 * rho * Va**2 * S * CD
    D_arr[i] = D
    P_arr[i] = D * Va
    alpha_arr_deg[i] = np.degrees(alpha)
    K_arr[i] = CL_req / CD

# Численные экстремумы
valid     = ~np.isnan(P_arr)
i_P_min   = np.nanargmin(P_arr)
i_D_min   = np.nanargmin(D_arr)

Va_P_min  = Va_sweep[i_P_min]
P_min     = P_arr[i_P_min]
Va_D_min  = Va_sweep[i_D_min]
D_min_num = D_arr[i_D_min]
K_max_num = K_arr[i_D_min]

# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------
sep = "=" * 60
print(sep)
print("  Оптимальные режимы горизонтального полёта")
print(f"  ЛА: m={m} кг  S={S} м²  b={params.b} м  AR={AR:.2f}")
print(f"  rho={rho} кг/м³  Va_stall ≈ {Va_stall:.1f} м/с")
print(sep)
print(f"  МАКСимальное качество  (мин. тяга, макс. дальность)")
print(f"    Va_md   = {Va_md:6.2f} м/с  (аналит.)  /  {Va_D_min:.2f} м/с  (числ.)")
print(f"    alpha*  = {alpha_md_deg:6.2f}°")
print(f"    CL_md   = {CL_md:.4f}")
print(f"    CD_md   = {CD_md:.4f}")
print(f"    K_max   = {K_md:.2f}  (числ. {K_max_num:.2f})")
print(f"    T_min   = {D_md:.2f} Н")
print(f"    P @md   = {P_md:.1f} Вт")
print(sep)
print(f"  МИНимальная мощность  (макс. продолжительность)")
print(f"    Va_mp   = {Va_mp:6.2f} м/с  (аналит.)  /  {Va_P_min:.2f} м/с  (числ.)")
print(f"    alpha*  = {alpha_mp_deg:6.2f}°")
print(f"    CL_mp   = {CL_mp:.4f}")
print(f"    CD_mp   = {CD_mp:.4f}")
print(f"    T @mp   = {D_mp:.2f} Н")
print(f"    P_min   = {P_mp:.1f} Вт  (числ. {P_min:.1f} Вт)")
print(sep)
print(f"  Соотношение скоростей  Va_mp / Va_md = {Va_mp/Va_md:.3f}  (теор. {1/3**0.25:.3f})")
print(f"  Экономия мощности: P_md → P_mp:  {(P_md-P_mp)/P_md*100:.1f}%")
print(sep)

# ---------------------------------------------------------------------------
# What-if: сдвиг Va_md → Va_target путём изменения одного параметра
# ---------------------------------------------------------------------------
Va_target = 30.0   # целевая скорость макс. качества, м/с
ratio2 = (Va_target / Va_md) ** 2   # во сколько раз нужно увеличить Va_md²

# Va_md = sqrt(2mg / (rho * S * CL_md)),  CL_md = sqrt(CDp*pi*e*AR)
# Va_md^2 = 2mg / (rho * sqrt(CDp*pi*e*b^2*S))

# Вариант А: изменить массу  (Va_md ∝ sqrt(m))
m_new = m * ratio2

# Вариант Б: изменить CDp  (Va_md ∝ CDp^(-1/4))
CDp_new = CDp / ratio2**2

# Вариант В: изменить площадь крыла S  (Va_md ∝ S^(-1/4))
S_new = S / ratio2**2

# Проверка Va_md для каждого варианта
def va_md_calc(m_, S_, CDp_, b_, e_, rho_, g_):
    AR_ = b_**2 / S_
    CL_md_ = np.sqrt(CDp_ * np.pi * e_ * AR_)
    return np.sqrt(2 * m_ * g_ / (rho_ * S_ * CL_md_))

Va_check_A = va_md_calc(m_new, S,     CDp,     params.b, e, rho, g)
Va_check_B = va_md_calc(m,     S,     CDp_new, params.b, e, rho, g)
Va_check_C = va_md_calc(m,     S_new, CDp,     params.b, e, rho, g)

# Новые alpha* и P_trim для каждого варианта
def alpha_and_P(m_, S_, CDp_, b_, e_, rho_, g_):
    AR_    = b_**2 / S_
    CL_md_ = np.sqrt(CDp_ * np.pi * e_ * AR_)
    CD_md_ = CDp_ + CL_md_**2 / (np.pi * e_ * AR_)
    Va_    = np.sqrt(2 * m_ * g_ / (rho_ * S_ * CL_md_))
    alpha_ = np.degrees((CL_md_ - params.CL0) / params.CLa)
    T_     = 0.5 * rho_ * Va_**2 * S_ * CD_md_
    P_     = T_ * Va_
    return Va_, alpha_, T_, P_

_, alpha_A, T_A, P_A = alpha_and_P(m_new, S,     CDp,     params.b, e, rho, g)
_, alpha_B, T_B, P_B = alpha_and_P(m,     S,     CDp_new, params.b, e, rho, g)
_, alpha_C, T_C, P_C = alpha_and_P(m,     S_new, CDp,     params.b, e, rho, g)

sep2 = "-" * 60
print(f"\n  ЧТО НАДО ИЗМЕНИТЬ чтобы Va_md = {Va_target:.0f} м/с")
print(sep2)
print(f"  {'Вариант':<30} {'Новое знач.':<14} {'alpha*':>7} {'P_trim':>9}")
print(sep2)
print(f"  {'А: масса m (кг)':<30} {m_new:<14.2f} {alpha_A:>6.1f}° {P_A:>8.0f} Вт")
print(f"     Jy тоже масштабировать: {params.Jy * m_new/m:.3f} кг·м²")
print(f"  {'Б: CDp (чище аэродин.)':<30} {CDp_new:<14.4f} {alpha_B:>6.1f}° {P_B:>8.0f} Вт")
print(f"     (текущий CDp={CDp}, было бы {CDp_new/CDp*100:.0f}% от него)")
print(f"  {'В: площадь крыла S (м²)':<30} {S_new:<14.4f} {alpha_C:>6.1f}° {P_C:>8.0f} Вт")
print(f"     b={params.b} м остаётся, AR → {params.b**2/S_new:.2f}")
print(sep2)
print(f"  Текущий оптимум Va_md={Va_md:.1f} м/с: alpha={alpha_md_deg:.1f}°  P_trim={P_md:.0f} Вт")
print(sep)

# ---------------------------------------------------------------------------
# Графики (3 субплота)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Оптимальные режимы горизонтального полёта\n"
             f"m={m} кг, S={S} м², AR={AR:.1f}, Va_stall≈{Va_stall:.1f} м/с",
             fontsize=12)

colors = {"range": "steelblue", "endurance": "darkorange", "stall": "red"}

# ---- 1. Тяга D(Va) — мин. тяга = макс. дальность ----
ax = axes[0]
ax.plot(Va_sweep[valid], D_arr[valid], "k-", lw=2)
ax.axvline(Va_md, color=colors["range"], ls="--", lw=1.5,
           label=f"Va_md={Va_md:.1f} м/с  (K_max={K_md:.1f})")
ax.axvline(Va_mp, color=colors["endurance"], ls="--", lw=1.5,
           label=f"Va_mp={Va_mp:.1f} м/с  (мин. P)")
ax.axvline(Va_stall, color=colors["stall"], ls=":", lw=1.2,
           label=f"Va_stall≈{Va_stall:.1f} м/с")
ax.scatter([Va_md], [D_md], color=colors["range"],    s=60, zorder=5)
ax.scatter([Va_mp], [D_mp], color=colors["endurance"],s=60, zorder=5)
ax.set_xlabel("Va, м/с")
ax.set_ylabel("T = D, Н")
ax.set_title("Требуемая тяга T(Va)\n(гориз. полёт, T = D)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
ax.set_xlim(Va_stall * 0.9, 50)
ax.set_ylim(0, D_arr[valid].max() * 1.15)

# ---- 2. Мощность P(Va) — мин. мощность = макс. продолжительность ----
ax = axes[1]
ax.plot(Va_sweep[valid], P_arr[valid], "k-", lw=2)
ax.axvline(Va_mp, color=colors["endurance"], ls="--", lw=1.5,
           label=f"Va_mp={Va_mp:.1f} м/с  P_min={P_mp:.0f} Вт")
ax.axvline(Va_md, color=colors["range"], ls="--", lw=1.5,
           label=f"Va_md={Va_md:.1f} м/с  P={P_md:.0f} Вт")
ax.axvline(Va_stall, color=colors["stall"], ls=":", lw=1.2,
           label=f"Va_stall≈{Va_stall:.1f} м/с")
ax.scatter([Va_mp], [P_mp], color=colors["endurance"],s=60, zorder=5)
ax.scatter([Va_md], [P_md], color=colors["range"],    s=60, zorder=5)
ax.set_xlabel("Va, м/с")
ax.set_ylabel("P = T·Va, Вт")
ax.set_title("Мощность мотора P(Va)\n(гориз. полёт)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
ax.set_xlim(Va_stall * 0.9, 50)
ax.set_ylim(0, P_arr[valid].max() * 1.15)

# ---- 3. Качество K(Va) и УА alpha(Va) ----
ax  = axes[2]
ax2 = ax.twinx()

ax.plot(Va_sweep[valid], K_arr[valid],       "b-",  lw=2, label="K = CL/CD")
ax2.plot(Va_sweep[valid], alpha_arr_deg[valid], "m--", lw=1.5, label="α, °")

ax.axvline(Va_md, color=colors["range"],     ls="--", lw=1.5)
ax.axvline(Va_mp, color=colors["endurance"], ls="--", lw=1.5)
ax.axvline(Va_stall, color=colors["stall"],  ls=":", lw=1.2)
ax.scatter([Va_md], [K_md], color=colors["range"],    s=60, zorder=5)

ax.set_xlabel("Va, м/с")
ax.set_ylabel("K = CL/CD", color="b")
ax2.set_ylabel("α, °",     color="m")
ax.set_title(f"Качество K(Va) и УА α(Va)\nK_max={K_md:.1f} при Va={Va_md:.1f} м/с")
ax.tick_params(axis="y", labelcolor="b")
ax2.tick_params(axis="y", labelcolor="m")

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
ax.grid(True, alpha=0.4)
ax.set_xlim(Va_stall * 0.9, 50)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "optimal_speed.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  График сохранён: {out_path}")
plt.show()

"""
Проверка 01: Аэродинамическая поляра ЛА.

Что строим:
  1. CL(alpha) — подъёмная сила.
  2. CD(alpha) — сопротивление.
  3. CL/CD(alpha) — аэродинамическое качество; находим угол максимального качества.
  4. Классическая поляра: CL vs CD.

Управление: delta_e = 0, q = 0, Va = 30 м/с (только для CLq — на коэффициент влияет мало).
Это «чистая» аэродинамика профиля без управляющих добавок.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Путь к корню симулятора
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams
from sim.aero import coef_CL, coef_CD

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

params = AircraftParams()

alpha_deg = np.linspace(-15, 30, 500)          # диапазон УА, градусы
alpha_rad = np.deg2rad(alpha_deg)

Va      = 30.0   # м/с — рабочая точка (влияет только через CLq=0 → не влияет)
q       = 0.0    # нет угловой скорости тангажа
delta_e = 0.0    # нейтральное положение руля высоты

# ---------------------------------------------------------------------------
# Вычисление коэффициентов
# ---------------------------------------------------------------------------

CL = np.array([coef_CL(a, q, delta_e, Va, params) for a in alpha_rad])
CD = np.array([coef_CD(a, params)                  for a in alpha_rad])

# Аэродинамическое качество (избегаем деление на 0 при CD → 0)
quality = np.where(CD > 1e-6, CL / CD, np.nan)

# ---------------------------------------------------------------------------
# Угол максимального качества
# ---------------------------------------------------------------------------

idx_best = np.nanargmax(quality)
alpha_best_deg = alpha_deg[idx_best]
alpha_best_rad = alpha_rad[idx_best]
CL_best  = CL[idx_best]
CD_best  = CD[idx_best]
K_best   = quality[idx_best]

print("=" * 55)
print("  Аэродинамическое качество ЛА (параметры-аналог Aerosonde)")
print("=" * 55)
print(f"  Угол макс. качества alpha*  = {alpha_best_deg:.2f}°  ({alpha_best_rad:.4f} рад)")
print(f"  CL при alpha*               = {CL_best:.4f}")
print(f"  CD при alpha*               = {CD_best:.4f}")
print(f"  max CL/CD (K_max)           = {K_best:.2f}")
print(f"  Угол предупреждения         = {np.rad2deg(params.alpha_warning):.1f}°")
print(f"  Критический УА              = {np.rad2deg(params.alpha_crit):.1f}°")
print("=" * 55)

# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Аэродинамическая поляра ЛА (параметры-аналог Aerosonde, δₑ=0)", fontsize=13)

# --- 1. CL(alpha) ---
ax = axes[0, 0]
ax.plot(alpha_deg, CL, "b-", linewidth=2)
ax.axvline(alpha_best_deg, color="g", linestyle="--", label=f"α* = {alpha_best_deg:.1f}°")
ax.axvline(np.rad2deg(params.alpha_warning), color="orange", linestyle=":", label=f"α_warn = {np.rad2deg(params.alpha_warning):.0f}°")
ax.axvline(np.rad2deg(params.alpha_crit),    color="red",    linestyle=":", label=f"α_crit = {np.rad2deg(params.alpha_crit):.0f}°")
ax.set_xlabel("УА α, °")
ax.set_ylabel("CL")
ax.set_title("Коэффициент подъёмной силы CL(α)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# --- 2. CD(alpha) ---
ax = axes[0, 1]
ax.plot(alpha_deg, CD, "r-", linewidth=2)
ax.axvline(alpha_best_deg, color="g", linestyle="--", label=f"α* = {alpha_best_deg:.1f}°")
ax.axvline(np.rad2deg(params.alpha_warning), color="orange", linestyle=":", label=f"α_warn")
ax.axvline(np.rad2deg(params.alpha_crit),    color="red",    linestyle=":", label=f"α_crit")
ax.set_xlabel("УА α, °")
ax.set_ylabel("CD")
ax.set_title("Коэффициент лобового сопротивления CD(α)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# --- 3. CL/CD(alpha) — качество ---
ax = axes[1, 0]
ax.plot(alpha_deg, quality, "k-", linewidth=2)
ax.axvline(alpha_best_deg, color="g", linestyle="--",
           label=f"α* = {alpha_best_deg:.1f}°,  K_max = {K_best:.1f}")
ax.axvline(np.rad2deg(params.alpha_warning), color="orange", linestyle=":", label=f"α_warn")
ax.axvline(np.rad2deg(params.alpha_crit),    color="red",    linestyle=":", label=f"α_crit")
ax.scatter([alpha_best_deg], [K_best], color="g", zorder=5, s=60)
ax.annotate(f"  K={K_best:.1f}\n  α={alpha_best_deg:.1f}°",
            xy=(alpha_best_deg, K_best), fontsize=8, color="g")
ax.set_xlabel("УА α, °")
ax.set_ylabel("CL / CD")
ax.set_title("Аэродинамическое качество K = CL/CD(α)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# --- 4. Классическая поляра CL(CD) ---
ax = axes[1, 1]
ax.plot(CD, CL, "m-", linewidth=2)
ax.scatter([CD_best], [CL_best], color="g", zorder=5, s=60,
           label=f"α* = {alpha_best_deg:.1f}°  (K_max)")
# Касательная из начала координат через точку максимального качества
k_slope = CL_best / CD_best
cd_line = np.linspace(0, max(CD) * 1.1, 50)
ax.plot(cd_line, k_slope * cd_line, "g--", linewidth=1, alpha=0.7, label=f"K = {K_best:.1f}")
ax.set_xlabel("CD")
ax.set_ylabel("CL")
ax.set_title("Классическая поляра CL(CD)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

plt.tight_layout()

# Сохранение рядом со скриптом
out_path = os.path.join(os.path.dirname(__file__), "polar.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n  График сохранён: {out_path}")

plt.show()

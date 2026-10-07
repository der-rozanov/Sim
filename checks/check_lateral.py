# -*- coding: utf-8 -*-
"""
Проверка адекватности бокового канала модели 6DOF в открытом контуре (без САУ).

Четыре опыта из балансировочного горизонтального полёта Va=20 м/с:
  1. Импульс элеронами δa=+5° на 1 с     — апериодика крена, разворот, спираль
  2. Дублет рулём направления ±5° по 0.3 с — голландский шаг (период, затухание)
  3. Связь крена и угловой скорости курса — ψ̇ ≈ g·tg φ / Va при β ≈ 0
  4. Ступенька бокового ветра 5 м/с        — флюгерный эффект: нос в ветер, β → 0

Запуск: python checks/check_lateral.py
Вывод — только текст (таблицы и сравнение с линейной теорией).
"""

import sys
import os
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import V, P, R, PHI, PSI, Y, H, X, Q, THETA, U, W
from sim.dynamics import derivatives
from runner import run, compute_trim, trim_state

ap  = AircraftParams()
cfg = SimConfig(Va0=20.0, h0=300.0, dt=0.01, t_end=30.0)
alpha_tr, de_tr, thr_tr = compute_trim(ap, cfg.Va0)
s0 = trim_state(ap, cfg)
D5 = np.radians(5.0)
deg = np.degrees


def table(log, times, title):
    print(f"\n--- {title} ---")
    print(f"{'t, с':>6} {'φ, °':>7} {'p, °/с':>8} {'r, °/с':>8} {'β, °':>7} "
          f"{'ψ, °':>8} {'h, м':>7} {'Va, м/с':>8} {'x, м':>7} {'y, м':>7}")
    for tt in times:
        i = min(int(np.searchsorted(log.t, tt - 1e-9)), len(log.t) - 1)
        s = log.state[i]
        print(f"{log.t[i]:6.1f} {deg(s[PHI]):7.2f} {deg(s[P]):8.2f} {deg(s[R]):8.2f} "
              f"{deg(log.beta[i]):7.2f} {deg(s[PSI]):8.2f} {s[H]:7.1f} {log.Va[i]:8.2f} "
              f"{s[X]:7.0f} {s[Y]:7.1f}")


# ------------------------------------------------------------------
# Линейная теория: собственные значения бокового канала [v, p, r, phi]
# ------------------------------------------------------------------
calm = lambda h, t: (0.0, 0.0, 0.0)

# Боковой трим: δa, δr, при которых p_dot = r_dot = 0 — парируют реактивный
# момент винта (РЕШ-19). Без них опыты в открытом контуре дрейфуют по крену.
c0 = np.array([de_tr, thr_tr, 0.0, 0.0])
d0 = derivatives(s0, c0, 0.0, ap, calm)[[P, R]]
B = np.column_stack([(derivatives(s0, c0 + e, 0.0, ap, calm)[[P, R]] - d0) / 1e-6
                     for e in (np.array([0, 0, 1e-6, 0]), np.array([0, 0, 0, 1e-6]))])
da_tr, dr_tr = np.linalg.solve(B, -d0)
c_tr = np.array([de_tr, thr_tr, da_tr, dr_tr])
idx = [V, P, R, PHI]
f0 = derivatives(s0, c_tr, 0.0, ap, calm)
A = np.zeros((4, 4))
for j, k in enumerate(idx):
    sp = s0.copy(); sp[k] += 1e-6
    A[:, j] = (derivatives(sp, c_tr, 0.0, ap, calm)[idx] - f0[idx]) / 1e-6
eig = np.linalg.eigvals(A)
dr = eig[eig.imag > 1e-6][0]
real = np.sort(eig[np.abs(eig.imag) <= 1e-6].real)
wn_lin, zeta_lin = abs(dr), -dr.real / abs(dr)
T_lin = 2 * np.pi / dr.imag

print("=" * 72)
print(f"Линейная модель бокового движения (трим Va={cfg.Va0:.0f} м/с, FPV-самолёт)")
print("=" * 72)
print(f"  Апериодика крена : λ = {real[0]:+.3f} 1/с   τ = {-1/real[0]:.3f} с")
print(f"  Голландский шаг  : λ = {dr.real:+.3f} ± {dr.imag:.3f}j   "
      f"ωn = {wn_lin:.2f} рад/с  ζ = {zeta_lin:.3f}  T = {T_lin:.3f} с")
sp_txt = "устойчива" if real[1] < 0 else f"неустойчива, T2 = {np.log(2)/real[1]:.1f} с"
print(f"  Спиральная мода  : λ = {real[1]:+.4f} 1/с   ({sp_txt})")
print(f"  Боковой трим (момент винта): δa = {deg(da_tr):+.3f}°,  δr = {deg(dr_tr):+.3f}°")

# ------------------------------------------------------------------
# 1. Импульс элеронами
# ------------------------------------------------------------------
def ctl_aileron(t, s, Va, al):
    return c_tr + [0.0, 0.0, D5 if t < 1.0 else 0.0, 0.0]

log1 = run(ctl_aileron, ap, WindParams(), cfg, state0=s0)
table(log1, [0, 0.25, 0.5, 1.0, 1.5, 2, 5, 10, 20, 30], "1. Импульс элеронами +5° на 1 с")
i1 = int(round(1.0 / cfg.dt))
p_ss = -ap.Croll_da * D5 / ap.Croll_p * 2 * cfg.Va0 / ap.b   # установившаяся p по Croll_p·p̂ + Croll_da·δa = 0
print(f"  p в конце импульса: {deg(log1.state[i1, P]):.1f} °/с  "
      f"(оценка одностепенной модели крена −Croll_da·δa/Croll_p·2Va/b = {deg(p_ss):.1f} °/с)")
print(f"  Ожидание: φ растёт только пока отклонены элероны, затем держится/медленно")
print(f"  меняется по спиральной моде; ψ нарастает в сторону крена (разворот вправо).")

# ------------------------------------------------------------------
# 2. Дублет рулём направления — голландский шаг
# ------------------------------------------------------------------
def ctl_rudder(t, s, Va, al):
    dr_ = D5 if t < 0.3 else (-D5 if t < 0.6 else 0.0)
    return c_tr + [0.0, 0.0, 0.0, dr_]

cfg2 = SimConfig(Va0=20.0, h0=300.0, dt=0.002, t_end=6.0)
log2 = run(ctl_rudder, ap, WindParams(), cfg2, state0=s0)
t2, b2 = log2.t, log2.beta
mask = t2 > 0.8
tb, bb = t2[mask], b2[mask]
# экстремумы β после окончания дублета
ext = [i for i in range(1, len(bb) - 1)
       if (bb[i] - bb[i-1]) * (bb[i+1] - bb[i]) < 0 and abs(bb[i]) > 1e-4]
print("\n--- 2. Дублет руля направления ±5° по 0.3 с ---")
if len(ext) >= 3:
    T_sim = 2 * np.mean(np.diff(tb[ext[:4]]))
    dec = np.log(abs(bb[ext[0]]) / abs(bb[ext[2]]))         # логарифм. декремент за период
    zeta_sim = dec / np.sqrt(4 * np.pi**2 + dec**2)
    print(f"  Экстремумы β: " + ", ".join(f"{tb[i]:.2f}с:{deg(bb[i]):+.2f}°" for i in ext[:5]))
    print(f"  Период  : моделирование T = {T_sim:.3f} с   линейная теория T = {T_lin:.3f} с")
    print(f"  Затухание: моделирование ζ = {zeta_sim:.3f}    линейная теория ζ = {zeta_lin:.3f}")
else:
    print("  недостаточно экстремумов β для оценки")
table(log2, [0, 0.15, 0.3, 0.45, 0.6, 1.0, 2.0, 4.0, 6.0], "2. Дублет руля направления (таблица)")

# ------------------------------------------------------------------
# 3. Кинематика разворота: ψ̇ = g·tg φ / Va при β ≈ 0
# ------------------------------------------------------------------
print("\n--- 3. Связь крена и угловой скорости курса (опыт 1) ---")
print(f"{'t, с':>6} {'φ, °':>7} {'β, °':>7} {'ψ̇ модели, °/с':>15} {'g·tgφ/Va, °/с':>15}")
for tt in [5, 10, 20, 30]:
    i = min(int(round(tt / cfg.dt)), len(log1.t) - 1)
    s = log1.state[i]
    psi_dot = derivatives(s, log1.controls[i], log1.t[i], ap, calm)[PSI]
    print(f"{log1.t[i]:6.1f} {deg(s[PHI]):7.2f} {deg(log1.beta[i]):7.2f} "
          f"{deg(psi_dot):15.2f} {deg(ap.g*np.tan(s[PHI])/log1.Va[i]):15.2f}")
print("  Ожидание: значения близки (разворот почти координированный, β мало).")

# ------------------------------------------------------------------
# 4. Ступенька бокового ветра
# ------------------------------------------------------------------
VW = 5.0
def ctl_trim(t, s, Va, al):
    return c_tr

wp4 = WindParams(Vw_cross=VW)
log4 = run(ctl_trim, ap, wp4, cfg, state0=s0)
table(log4, [0, 0.2, 0.5, 1, 2, 5, 10, 20, 30], f"4. Боковой ветер {VW:.0f} м/с на восток с t=0")
print(f"  Ожидание: в начальный момент β = −arctg({VW:.0f}/{cfg.Va0:.0f}) = {deg(-np.arctan(VW/cfg.Va0)):.1f}°;")
print(f"  флюгерный эффект разворачивает нос навстречу ветру (ψ < 0, влево),")
print(f"  β → 0, ЛА сносится ветром на восток (y растёт).")

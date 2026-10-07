"""
Оценка аэродинамических производных FPV-самолёта по геометрии (РЕШ-20).

Типовой FPV-самолёт классической схемы класса 1.8 м (Volantex Ranger 2000,
Skywalker 1900): высокоплан с прямоугольным крылом, хвостовое оперение
с рулём высоты и рулём направления, тянущий винт.
ГЕОМЕТРИЯ И МАССА — ОЦЕНКА [АВТОР: заменить замерами изделия].

Формулы — учебные инженерные оценки:
  [N]  Nelson R.C. Flight Stability and Automatic Control, 2nd ed., гл. 2–3, 5;
  [R]  Raymer D.P. Aircraft Design: A Conceptual Approach, гл. 12, 16;
  [Ro] Roskam J. Airplane Design, Part V (радиусы инерции);
  [G]  Глауэрт — теория тонкого профиля (эффективность закрылка/руля).
ВМГ: мотор T-Motor AT2814 900KV (паспорт), винт APC 11×7E — аппроксимация
продувки UIUC Propeller Database, vol. 1 (Brandt & Selig).

Запуск: python checks/estimate_fpv_aero.py — печатает производные и сверяет
с sim/config.py (AircraftParams по умолчанию).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

D2R = np.pi / 180.0

# ---------------------------------------------------------------------------
# Геометрия и масса (оценка)
# ---------------------------------------------------------------------------
m     = 2.0          # кг, полётная масса с FPV-оборудованием и АКБ 4S
rho   = 1.225        # кг/м³ — для расчётов производных (в симуляторе своя ρ)
g     = 9.81

# Крыло: прямоугольное, профиль типа Clark-Y
b, c  = 1.8, 0.25            # м
S     = b * c                # 0.45 м²
AR    = b**2 / S             # 7.2
lam   = 1.0                  # сужение
Gamma = 3.0 * D2R            # поперечное V
i_w   = 0.0 * D2R            # установочный угол крыла к строительной оси
alpha0L_2d = -3.5 * D2R      # угол нулевой подъёмной силы профиля (Clark-Y)
cm_ac_w    = -0.08           # момент профиля относительно фокуса (Clark-Y)
cl_max_2d  = 1.35            # максимум cl профиля при Re ≈ 2.5·10⁵
h_ac_w     = 0.25            # фокус крыла, доли САХ от носка

# Горизонтальное оперение
S_h, b_h = 0.080, 0.53       # м², м  (хорда ≈ 0.15 м)
AR_h     = b_h**2 / S_h      # 3.5
l_h      = 0.75              # м, плечо от ЦТ до фокуса ГО
cf_e     = 0.40              # хорда руля высоты / хорда ГО

# Вертикальное оперение
S_v, h_v = 0.040, 0.25       # м², м  (хорда ≈ 0.16 м)
AR_v     = h_v**2 / S_v      # 1.56
l_v      = 0.75              # м, плечо от ЦТ до фокуса ВО
z_v      = 0.10              # м, фокус ВО выше ЦТ
cf_r     = 0.40              # хорда руля направления / хорда ВО

# Фюзеляж
L_f, d_f = 1.20, 0.10        # м, длина и диаметр
z_w      = -0.05             # м, крыло выше оси фюзеляжа (высокоплан), > 0 — ниже

# Элероны: от 50 % до 95 % полуразмаха, 25 % хорды
y1, y2   = 0.50 * b / 2, 0.95 * b / 2
cf_a     = 0.25

eta_t    = 0.90              # скоростное торможение у ГО [N]
SM       = 0.10              # запас статической устойчивости, доли САХ
V_cruise = 16.0              # м/с — балансировка δe = 0 (выбор установки ГО)

# ---------------------------------------------------------------------------
# Вспомогательные формулы
# ---------------------------------------------------------------------------

def lift_slope(A, eta=0.95):
    """Наклон CL(α) крыла конечного размаха, Гельмбольд/DATCOM [R гл. 12], Λ = 0."""
    return 2 * np.pi * A / (2 + np.sqrt(4 + A**2 / eta**2))


def tau_flap(cf):
    """Эффективность руля (dα/dδ) по теории тонкого профиля [G]: верхняя оценка."""
    th = np.arccos(2 * cf - 1)
    return 1 - (th - np.sin(th)) / np.pi


# ---------------------------------------------------------------------------
# Продольный канал
# ---------------------------------------------------------------------------
a_w  = lift_slope(AR)
a_h  = lift_slope(AR_h)
deps = 2 * a_w / (np.pi * AR)                 # dε/dα [N гл. 2]
V_h  = S_h * l_h / (S * c)                    # коэффициент статического момента ГО
tau_e = tau_flap(cf_e)

CLa  = a_w + eta_t * (S_h / S) * a_h * (1 - deps)
Cma  = -SM * CLa                              # определение запаса устойчивости
CL_cr = 2 * m * g / (rho * V_cruise**2 * S)   # CL балансировки на крейсерской скорости

# Установка ГО i_t выбирается так, чтобы на V_cruise балансировка шла при δe = 0.
# CL0 и Cm0 линейны по i_t [N гл. 2]:
#   CL0 = a_w(i_w − α0L) + η(S_h/S)a_h(i_t − ε0)
#   Cm0 = cm_ac + a_w(i_w − α0L)(h_cg − h_ac) + η V_h a_h (ε0 − i_t)
h_np = h_ac_w + eta_t * V_h * (a_h / CLa) * (1 - deps)   # нейтральная центровка (без фюзеляжа)
h_cg = h_np - SM
CL0_w = a_w * (i_w - alpha0L_2d)
eps0  = 2 * CL0_w / (np.pi * AR)


def cl0_cm0(i_t):
    CL0 = CL0_w + eta_t * (S_h / S) * a_h * (i_t - eps0)
    Cm0 = cm_ac_w + CL0_w * (h_cg - h_ac_w) + eta_t * V_h * a_h * (eps0 - i_t)
    return CL0, Cm0


def cm_at_cruise(i_t):
    CL0, Cm0 = cl0_cm0(i_t)
    alpha_cr = (CL_cr - CL0) / CLa
    return Cm0 + Cma * alpha_cr


lo, hi = -10 * D2R, 10 * D2R
for _ in range(60):
    mid = 0.5 * (lo + hi)
    lo, hi = (mid, hi) if cm_at_cruise(mid) > 0 else (lo, mid)
i_t = 0.5 * (lo + hi)
CL0, Cm0 = cl0_cm0(i_t)

CLq  = 2 * eta_t * V_h * a_h                          # [N гл. 3]
Cmq  = -2 * eta_t * V_h * a_h * (l_h / c) * 1.1       # +10 % на фюзеляж/крыло [N]
CLde = eta_t * (S_h / S) * a_h * tau_e
Cmde = -eta_t * V_h * a_h * tau_e

# Сопротивление: эквивалентное трение [R гл. 12], малые Re + FPV-надстройки
S_wet = (2.0 * S * 1.024 + 2.0 * S_h + 2.0 * S_v + np.pi * d_f * L_f * 0.8)
C_fe  = 0.0080        # [R гл. 12] 0.0055 для лёгких самолётов; ×1.45 на малые Re
dCD_fpv = 0.008       # камера, антенны, шасси/лыжа — оценка
CDp   = C_fe * S_wet / S + dCD_fpv
e_osw = 1.78 * (1 - 0.045 * AR**0.68) - 0.64       # [R гл. 12]

# Срыв: CL_max самолёта ≈ 0.9·cl_max профиля [R гл. 12]
CL_max_target = 0.9 * cl_max_2d

# ---------------------------------------------------------------------------
# Боковой канал
# ---------------------------------------------------------------------------
a_v = lift_slope(1.55 * AR_v)                 # эффективное удлинение ВО ×1.55 [R гл. 12]
eta_v = (0.724 + 3.06 * (S_v / S) / 2 + 0.4 * z_w / d_f + 0.009 * AR)  # η_v(1+dσ/dβ) [N гл. 3]
V_v = S_v * l_v / (S * b)
tau_r = tau_flap(cf_r)
tau_a = tau_flap(cf_a)
Vol_f = np.pi / 4 * d_f**2 * L_f * 0.75

CY_beta = -eta_v * a_v * S_v / S
Cn_beta = eta_v * a_v * V_v - 1.3 * Vol_f / (S * b)           # фюзеляж [R гл. 16]
Croll_beta = (-a_w * Gamma * (1 + 2 * lam) / (6 * (1 + lam))   # поперечное V [N]
              - eta_v * a_v * (S_v / S) * (z_v / b))           # киль выше ЦТ
Croll_p = -a_w * (1 + 3 * lam) / (12 * (1 + lam))             # [N, полосовая теория]
Cn_p    = -CL_cr / 8                                          # [N]
CY_p    = 2 * eta_v * a_v * (S_v / S) * (z_v / b)
Croll_r = (CL_cr * (1 + 3 * lam) / (6 * (1 + lam))
           + 2 * eta_v * a_v * (S_v / S) * (z_v * l_v) / b**2)
Cn_r    = -CDp / 4 - 2 * eta_v * a_v * V_v * (l_v / b)
CY_r    = 2 * eta_v * a_v * (S_v / S) * (l_v / b)

Croll_da = a_w * tau_a * c * (y2**2 - y1**2) / (S * b)        # полосовая теория [N]
Cn_da    = 2 * (-0.15) * CL_cr * Croll_da                     # K ≈ −0.15 [N гл. 5]
CY_dr    = eta_v * a_v * tau_r * S_v / S
Cn_dr    = -eta_v * V_v * a_v * tau_r
Croll_dr = (z_v / b) * CY_dr

# ---------------------------------------------------------------------------
# Инерция: безразмерные радиусы инерции [Ro]: R̄x = 0.25, R̄y = 0.38, R̄z = 0.39
# ---------------------------------------------------------------------------
Jx = m * (b * 0.25 / 2)**2
Jy = m * (L_f * 0.38 / 2)**2
Jz = m * ((b + L_f) / 2 * 0.39 / 2)**2

# ---------------------------------------------------------------------------
# ВМГ: APC 11×7E — аппроксимация CT(J), CQ(J) = CP/(2π) по UIUC (≈6000 об/мин,
# J = 0…0.81, 43 точки); невязка |ΔCT| ≤ 0.0066, |ΔCQ| ≤ 0.0005
# ---------------------------------------------------------------------------
PROP = dict(D_prop=11 * 0.0254,
            C_T2=-0.15819, C_T1=-0.018756, C_T0=0.10922,
            C_Q2=-0.020775, C_Q1=0.0094015, C_Q0=0.0068499)
MOTOR = dict(KV_rpm=900.0, R_motor=0.082, i0=1.2, V_max=4 * 3.7)   # AT2814 900KV, 4S

EST = dict(mass=m, S=S, b=b, c=c, Jx=Jx, Jy=Jy, Jz=Jz, Jxz=0.0,
           CL0=CL0, CLa=CLa, CLq=CLq, CLde=CLde,
           CDp=CDp, e_oswald=e_osw,
           Cm0=Cm0, Cma=Cma, Cmq=Cmq, Cmde=Cmde,
           CY_beta=CY_beta, CY_p=CY_p, CY_r=CY_r, CY_dr=CY_dr,
           Croll_beta=Croll_beta, Croll_p=Croll_p, Croll_r=Croll_r,
           Croll_da=Croll_da, Croll_dr=Croll_dr,
           Cn_beta=Cn_beta, Cn_p=Cn_p, Cn_r=Cn_r, Cn_da=Cn_da, Cn_dr=Cn_dr)


if __name__ == "__main__":
    from sim.config import AircraftParams
    ac = AircraftParams()

    print("=" * 72)
    print("Оценка производных FPV-самолёта по геометрии (checks/estimate_fpv_aero.py)")
    print("=" * 72)
    print(f"  AR = {AR:.2f}  a_w = {a_w:.3f}  a_h = {a_h:.3f}  a_v = {a_v:.3f} 1/рад")
    print(f"  dε/dα = {deps:.3f}  V_h = {V_h:.3f}  V_v = {V_v:.4f}  η_v(1+dσ/dβ) = {eta_v:.3f}")
    print(f"  τ_e = {tau_e:.3f}  τ_r = {tau_r:.3f}  τ_a = {tau_a:.3f}")
    print(f"  нейтральная центровка h_n = {h_np:.3f}·c,  ЦТ h_cg = {h_cg:.3f}·c  (SM = {SM})")
    print(f"  установка ГО i_t = {i_t / D2R:+.2f}°  (δe = 0 на {V_cruise} м/с, CL = {CL_cr:.3f})")
    print(f"  S_wet/S = {S_wet / S:.2f}   CL_max (цель) = {CL_max_target:.3f}")
    print()
    print(f"  {'параметр':<12} {'оценка':>10} {'config':>10}")
    bad = []
    for k, v in EST.items():
        cv = getattr(ac, k)
        ok = np.isclose(v, cv, rtol=0.01, atol=1e-4)
        bad += [] if ok else [k]
        print(f"  {k:<12} {v:10.4f} {cv:10.4f} {'' if ok else '  ≠'}")
    for k, v in {**PROP, **MOTOR}.items():
        cv = getattr(ac, k)
        ok = np.isclose(v, cv, rtol=0.01)
        bad += [] if ok else [k]
        print(f"  {k:<12} {v:10.4f} {cv:10.4f} {'' if ok else '  ≠'}")
    print()
    print("  config совпадает с оценкой" if not bad else f"  РАСХОЖДЕНИЯ: {bad}")

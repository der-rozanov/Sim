# -*- coding: utf-8 -*-
"""
Быстрая проверка всех модулей симулятора.
Запуск: python check.py
"""

import numpy as np
import sys

PASS = "[OK]"
FAIL = "[!!]"

def section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

def check(name, condition, got=""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if got:
        msg += f"  =>  {got}"
    print(msg)
    return condition

all_ok = True

# ------------------------------------------------------------------
section("1. config.py")
# ------------------------------------------------------------------
try:
    from config import AircraftParams, WindParams, SensorParams, SimConfig, default_params
    ap, wp, sp, cfg = default_params()
    all_ok &= check("импорт",          True)
    all_ok &= check("mass = 13.5 кг",  ap.mass == 13.5,  f"mass={ap.mass}")
    all_ok &= check("Jy   = 1.135",    ap.Jy == 1.135,   f"Jy={ap.Jy}")
    all_ok &= check("CL0  = 0.28",     ap.CL0 == 0.28,   f"CL0={ap.CL0}")
    all_ok &= check("dt   = 0.01 с",   cfg.dt == 0.01,   f"dt={cfg.dt}")
except Exception as e:
    check("импорт", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("2. state.py")
# ------------------------------------------------------------------
try:
    from state import (initial_state, air_velocity, kinematic_gamma,
                       total_energy, U, W, Q, THETA, X, H, N_STATES)
    s = initial_state(cfg)
    all_ok &= check("N_STATES = 6",  N_STATES == 6)
    all_ok &= check("shape (6,)",    s.shape == (6,),    f"shape={s.shape}")
    all_ok &= check("Va0 в u",       s[U] == cfg.Va0,   f"u={s[U]}")
    all_ok &= check("h0 в H",        s[H] == cfg.h0,    f"h={s[H]}")

    # Нулевой ветер — Va = u, alpha = 0
    Va, alpha = air_velocity(s, (0.0, 0.0))
    all_ok &= check("Va без ветра",  abs(Va - cfg.Va0) < 1e-9, f"Va={Va:.4f}")
    all_ok &= check("alpha=0 без ветра", abs(alpha) < 1e-9,    f"alpha={np.degrees(alpha):.4f} deg")

    # Попутный ветер 5 м/с (Vwx>0) -> Va уменьшается
    Va_hw, alpha_hw = air_velocity(s, (5.0, 0.0))
    all_ok &= check("Va < Va0 pri poputnom vetre",
                    Va_hw < cfg.Va0, f"Va={Va_hw:.2f} m/s (ожидалось <{cfg.Va0})")

    # Восходящий ветер 5 м/с при theta=0 -> alpha > 0
    Va_up, alpha_up = air_velocity(s, (0.0, 5.0))
    all_ok &= check("alpha > 0 при восходящем ветре",
                    alpha_up > 0, f"alpha={np.degrees(alpha_up):.2f} deg")

    # Энергия
    Ek, Ep, Et = total_energy(s, ap)
    all_ok &= check("E_kin > 0",  Ek > 0, f"Ek={Ek:.1f} J")
    all_ok &= check("E_pot > 0",  Ep > 0, f"Ep={Ep:.1f} J")
    all_ok &= check("E_total = Ek+Ep", abs(Et - Ek - Ep) < 1e-9)

except Exception as e:
    check("импорт state", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("3. aero.py")
# ------------------------------------------------------------------
try:
    from aero import coef_CL, coef_CD, coef_Cm, aero_forces_moments

    # CL растёт с alpha в доступном диапазоне
    CLs = [coef_CL(np.radians(a), 0, 0, 30.0, ap) for a in [0, 5, 10, 20, 25]]
    all_ok &= check("CL(0) = CL0",      abs(CLs[0] - ap.CL0) < 1e-9, f"CL0={CLs[0]:.3f}")
    all_ok &= check("CL растёт 0->25 deg", CLs[4] > CLs[0],
                    f"CL(25)={CLs[4]:.3f} > CL(0)={CLs[0]:.3f}")

    # Срыв: CL на 30° меньше чем на 25°
    CL_25 = coef_CL(np.radians(25), 0, 0, 30.0, ap)
    CL_30 = coef_CL(np.radians(30), 0, 0, 30.0, ap)
    all_ok &= check("CL падает после срыва (25->30 deg)",
                    CL_30 < CL_25, f"CL(25)={CL_25:.3f}  CL(30)={CL_30:.3f}")

    # CD > 0 всегда
    CDs = [coef_CD(np.radians(a), ap) for a in [-10, 0, 10, 20]]
    all_ok &= check("CD > 0 для всех alpha", all(c > 0 for c in CDs),
                    f"min={min(CDs):.4f}")

    # Cma < 0 (продольная устойчивость)
    all_ok &= check("Cma < 0",  ap.Cma < 0, f"Cma={ap.Cma}")

    # Подъёмная сила на скорости 30 м/с ≈ mg при trim alpha
    CL_trim = 2 * ap.mass * ap.g / (ap.rho * 30.0**2 * ap.S)
    alpha_trim = (CL_trim - ap.CL0) / ap.CLa
    fx, fz, M = aero_forces_moments(30.0, alpha_trim, 0, 0, ap)
    L_actual = -fz * np.cos(alpha_trim) + fx * np.sin(alpha_trim)
    all_ok &= check("L = m*g на балансировочной скорости",
                    abs(L_actual - ap.mass * ap.g) < 1.0,
                    f"L={L_actual:.2f}  mg={ap.mass*ap.g:.2f}")

except Exception as e:
    check("импорт aero", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("4. wind.py")
# ------------------------------------------------------------------
try:
    from wind import wind

    # Нулевой ветер
    vx, vh = wind(100.0, 0.0, wp)
    all_ok &= check("нулевой ветер при Vw_const=0", vx == 0.0 and vh == 0.0,
                    f"Vwx={vx}, Vwh={vh}")

    # Постоянный ветер
    from config import WindParams
    wp2 = WindParams(Vw_const=5.0)
    vx2, _ = wind(100.0, 0.0, wp2)
    all_ok &= check("постоянный ветер 5 м/с", vx2 == 5.0, f"Vwx={vx2}")

    # Сдвиг ветра: ниже слоя — ноль, внутри — линейно, выше — полный
    wp3 = WindParams(Vw_const=0.0, h_shear_lo=50.0, h_shear_hi=100.0, dV_shear=10.0)
    vx_lo, _  = wind(30.0,  0.0, wp3)
    vx_mid, _ = wind(75.0,  0.0, wp3)
    vx_hi, _  = wind(110.0, 0.0, wp3)
    all_ok &= check("сдвиг: ниже слоя = 0",     abs(vx_lo)  < 1e-9,  f"{vx_lo:.2f}")
    all_ok &= check("сдвиг: середина = 5 м/с",  abs(vx_mid - 5.0) < 1e-9, f"{vx_mid:.2f}")
    all_ok &= check("сдвиг: выше слоя = 10 м/с", abs(vx_hi - 10.0) < 1e-9, f"{vx_hi:.2f}")

    # Порыв работает только в нужный момент
    wp4 = WindParams(gust_amp=8.0, gust_t0=5.0, gust_dur=2.0)
    vx_before, _ = wind(100.0, 4.9, wp4)
    vx_during, _ = wind(100.0, 6.0, wp4)
    vx_after,  _ = wind(100.0, 7.1, wp4)
    all_ok &= check("порыв: до = 0",       abs(vx_before) < 1e-9, f"{vx_before:.1f}")
    all_ok &= check("порыв: во время = 8", abs(vx_during - 8.0) < 1e-9, f"{vx_during:.1f}")
    all_ok &= check("порыв: после = 0",    abs(vx_after)  < 1e-9, f"{vx_after:.1f}")

except Exception as e:
    check("импорт wind", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("5. dynamics.py")
# ------------------------------------------------------------------
try:
    from dynamics import derivatives, thrust
    from wind import wind as wind_fn

    wind_call = lambda h, t: wind_fn(h, t, wp)
    s0 = initial_state(cfg)

    # derivatives возвращает вектор нужной длины
    ds = derivatives(s0, np.array([0.0, 0.0]), 0.0, ap, wind_call)
    all_ok &= check("derivatives: shape (6,)", ds.shape == (6,), f"shape={ds.shape}")

    # Тяга: нулевой газ -> тяга может быть < 0 (торможение),
    #       газ=1 -> тяга > 0
    T0 = thrust(0.0, 30.0, ap)
    T1 = thrust(1.0, 30.0, ap)
    all_ok &= check("thrust(1) > thrust(0)", T1 > T0, f"T0={T0:.1f} N  T1={T1:.1f} N")
    all_ok &= check("thrust(1) > 0",        T1 > 0,  f"T1={T1:.1f} N")

    # С нулевым ускорением: состояние меняется (ЛА не висит без тяги)
    all_ok &= check("derivatives != 0 при нулевом управлении",
                    np.any(ds != 0), f"max|ds|={np.max(np.abs(ds)):.4f}")

except Exception as e:
    check("импорт dynamics", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("6. integrators.py")
# ------------------------------------------------------------------
try:
    from integrators import step_rk4, step_euler

    s0 = initial_state(cfg)
    controls = np.array([0.0, 0.4])   # небольшая тяга
    wind_call = lambda h, t: wind_fn(h, t, wp)

    s_rk4   = step_rk4  (s0, controls, cfg.dt, 0.0, ap, wind_call)
    s_euler = step_euler(s0, controls, cfg.dt, 0.0, ap, wind_call)

    all_ok &= check("RK4:   состояние изменилось",   not np.allclose(s_rk4,   s0))
    all_ok &= check("Euler: состояние изменилось",   not np.allclose(s_euler, s0))

    # RK4 и Euler близки на малом шаге, но не равны
    diff = np.max(np.abs(s_rk4 - s_euler))
    all_ok &= check("RK4 != Euler (разные порядки)", diff > 1e-12, f"|diff|={diff:.2e}")

    # 5 секунд симуляции без краша
    s = initial_state(cfg)
    try:
        t = 0.0
        for _ in range(500):
            s = step_rk4(s, controls, cfg.dt, t, ap, wind_call)
            t += cfg.dt
        all_ok &= check("5 секунд RK4 без краша", True,
                        f"h={s[H]:.1f} m  Va={np.sqrt(s[U]**2+s[W]**2):.1f} m/s")
    except Exception as e:
        check("5 секунд RK4 без краша", False, str(e)); all_ok = False

except Exception as e:
    check("импорт integrators", False, str(e)); all_ok = False

# ------------------------------------------------------------------
print("\n" + "="*50)
if all_ok:
    print("  ALL CHECKS PASSED")
else:
    print("  FAILURES FOUND -- see [!!] above")
print("="*50)

sys.exit(0 if all_ok else 1)

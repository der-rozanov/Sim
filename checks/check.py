# -*- coding: utf-8 -*-
"""
Быстрая проверка всех модулей симулятора.
Запуск: python checks/check.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
section("1. sim/config.py")
# ------------------------------------------------------------------
try:
    from sim.config import AircraftParams, WindParams, SensorParams, SimConfig, default_params
    ap, wp, sp, cfg = default_params()
    all_ok &= check("импорт",          True)
    all_ok &= check("mass = 2 кг",     ap.mass == 2.0,   f"mass={ap.mass}")
    all_ok &= check("Jy   = 0.104",    ap.Jy == 0.1040,  f"Jy={ap.Jy}")
    all_ok &= check("Jx*Jz > Jxz^2 (тензор инерции положительно определён)",
                    ap.Jx * ap.Jz > ap.Jxz**2)
    all_ok &= check("CL0  = 0.2636",   ap.CL0 == 0.2636, f"CL0={ap.CL0}")
    all_ok &= check("dt   = 0.01 с",   cfg.dt == 0.01,   f"dt={cfg.dt}")
except Exception as e:
    check("импорт", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("2. sim/state.py")
# ------------------------------------------------------------------
try:
    from sim.state import (initial_state, air_velocity, kinematic_gamma,
                           total_energy, U, W, Q, THETA, X, H, N_STATES)
    s = initial_state(cfg)
    all_ok &= check("N_STATES = 12", N_STATES == 12)
    all_ok &= check("shape (12,)",   s.shape == (12,),   f"shape={s.shape}")
    all_ok &= check("Va0 в u",       s[U] == cfg.Va0,   f"u={s[U]}")
    all_ok &= check("h0 в H",        s[H] == cfg.h0,    f"h={s[H]}")

    # Нулевой ветер — Va = u, alpha = 0
    Va, alpha = air_velocity(s, (0.0, 0.0))
    all_ok &= check("Va без ветра",  abs(Va - cfg.Va0) < 1e-9, f"Va={Va:.4f}")
    all_ok &= check("alpha=0 без ветра", abs(alpha) < 1e-9,    f"alpha={np.degrees(alpha):.4f} deg")

    # Попутный ветер 5 м/с (Vwx>0) -> Va уменьшается
    Va_hw, alpha_hw = air_velocity(s, (5.0, 0.0))
    all_ok &= check("Va < Va0 при попутном ветре",
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
section("3. sim/aero.py")
# ------------------------------------------------------------------
try:
    from sim.aero import coef_CL, coef_CD, coef_Cm, aero_forces_moments

    # CL растёт с alpha до критического УА (максимум CL)
    a_cr = ap.alpha_crit
    CLs = [coef_CL(a, 0, 0, 30.0, ap) for a in [0.0, a_cr]]
    all_ok &= check("CL(0) ~ CL0",      abs(CLs[0] - ap.CL0) < 1e-4, f"CL0={CLs[0]:.6f}")
    all_ok &= check("CL растёт 0 -> alpha_crit", CLs[1] > CLs[0],
                    f"CL({np.degrees(a_cr):.0f})={CLs[1]:.3f} > CL(0)={CLs[0]:.3f}")

    # Срыв: за alpha_crit CL падает; alpha_crit — максимум CL (±0.5°)
    CL_post = coef_CL(a_cr + np.radians(5), 0, 0, 30.0, ap)
    all_ok &= check("CL падает после срыва (alpha_crit -> +5 deg)",
                    CL_post < CLs[1], f"CL={CLs[1]:.3f} -> {CL_post:.3f}")
    a_grid = np.radians(np.linspace(0, 30, 3001))
    a_max = a_grid[np.argmax([coef_CL(a, 0, 0, 30.0, ap) for a in a_grid])]
    all_ok &= check("alpha_crit = УА максимума CL", abs(a_max - a_cr) < np.radians(0.5),
                    f"max CL при {np.degrees(a_max):.2f}°, alpha_crit={np.degrees(a_cr):.2f}°")

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
section("4. sim/wind.py")
# ------------------------------------------------------------------
try:
    from sim.wind import wind
    from sim.config import WindParams

    # Нулевой ветер
    vx, vh, vy = wind(100.0, 0.0, wp)
    all_ok &= check("нулевой ветер при Vw_const=0", vx == 0.0 and vh == 0.0 and vy == 0.0,
                    f"Vwx={vx}, Vwh={vh}, Vwy={vy}")

    # Боковой ветер — третья компонента, продольные не трогает
    vx_c, vh_c, vy_c = wind(100.0, 0.0, WindParams(Vw_cross=4.0))
    all_ok &= check("боковой ветер 4 м/с -> Vwy", vy_c == 4.0 and vx_c == 0.0,
                    f"Vwy={vy_c}")

    # Постоянный ветер
    wp2 = WindParams(Vw_const=5.0)
    vx2 = wind(100.0, 0.0, wp2)[0]
    all_ok &= check("постоянный ветер 5 м/с", vx2 == 5.0, f"Vwx={vx2}")

    # Сдвиг ветра: ниже слоя — ноль, внутри — линейно, выше — полный
    wp3 = WindParams(Vw_const=0.0, h_shear_lo=50.0, h_shear_hi=100.0, dV_shear=10.0)
    vx_lo  = wind(30.0,  0.0, wp3)[0]
    vx_mid = wind(75.0,  0.0, wp3)[0]
    vx_hi  = wind(110.0, 0.0, wp3)[0]
    all_ok &= check("сдвиг: ниже слоя = 0",     abs(vx_lo)  < 1e-9,  f"{vx_lo:.2f}")
    all_ok &= check("сдвиг: середина = 5 м/с",  abs(vx_mid - 5.0) < 1e-9, f"{vx_mid:.2f}")
    all_ok &= check("сдвиг: выше слоя = 10 м/с", abs(vx_hi - 10.0) < 1e-9, f"{vx_hi:.2f}")

    # Порыв работает только в нужный момент
    wp4 = WindParams(gust_amp=8.0, gust_t0=5.0, gust_dur=2.0)
    vx_before = wind(100.0, 4.9, wp4)[0]
    vx_during = wind(100.0, 6.0, wp4)[0]
    vx_after  = wind(100.0, 7.1, wp4)[0]
    all_ok &= check("порыв: до = 0",       abs(vx_before) < 1e-9, f"{vx_before:.1f}")
    all_ok &= check("порыв: во время = 8", abs(vx_during - 8.0) < 1e-9, f"{vx_during:.1f}")
    all_ok &= check("порыв: после = 0",    abs(vx_after)  < 1e-9, f"{vx_after:.1f}")

except Exception as e:
    check("импорт wind", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("5. sim/dynamics.py")
# ------------------------------------------------------------------
try:
    from sim.dynamics import derivatives, thrust
    from sim.wind import wind as wind_fn

    wind_call = lambda h, t: wind_fn(h, t, wp)
    s0 = initial_state(cfg)

    # derivatives возвращает вектор нужной длины
    ds = derivatives(s0, np.array([0.0, 0.0]), 0.0, ap, wind_call)
    all_ok &= check("derivatives: shape (12,)", ds.shape == (12,), f"shape={ds.shape}")

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
section("6. sim/integrators.py")
# ------------------------------------------------------------------
try:
    from sim.integrators import step_rk4, step_euler

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
section("7. Модель 6DOF: боковой канал")
# ------------------------------------------------------------------
try:
    from dataclasses import replace
    from sim.state import (V, P, R, PHI, PSI, Y, air_data, rotation_body_to_earth)
    from sim.dynamics import derivatives, thrust, propeller, _gammas
    from sim.aero import aero_forces_moments
    from sim.integrators import step_rk4
    from runner import compute_trim, trim_state

    LAT = [V, P, R, PHI, PSI]
    calm = lambda h, t: (0.0, 0.0, 0.0)
    a_tr, de_tr, thr_tr = compute_trim(ap, cfg.Va0)
    s_tr = trim_state(ap, cfg)
    c_tr = np.array([de_tr, thr_tr, 0.0, 0.0])

    # 7.1 Матрица поворота ортогональна
    Rm = rotation_body_to_earth(0.3, -0.2, 1.1)
    all_ok &= check("R·R^T = I, det R = 1",
                    np.allclose(Rm @ Rm.T, np.eye(3)) and abs(np.linalg.det(Rm) - 1) < 1e-12)

    # 7.2 Симметричный полёт (в т.ч. с продольным и вертикальным ветром):
    #     боковой канал возбуждает только реактивный момент винта −Q:
    #     p_dot = −Г3·Q, r_dot = −Г4·Q, остальные производные строго нулевые
    wind_long = lambda h, t: (-4.0, 1.5, 0.0)
    s_test = s_tr.copy(); s_test[Q] = 0.1; s_test[THETA] += 0.1
    ds = derivatives(s_test, c_tr, 0.0, ap, wind_long)
    Q_pr = propeller(thr_tr, air_data(s_test, wind_long(0.0, 0.0))[0], ap)[1]
    G = _gammas(ap)
    all_ok &= check("симметричный полёт: d(v,phi,psi)/dt = 0",
                    np.all(ds[[V, PHI, PSI]] == 0.0))
    all_ok &= check("симметричный полёт: p_dot, r_dot — только от момента винта",
                    np.isclose(ds[P], -G[2] * Q_pr) and np.isclose(ds[R], -G[3] * Q_pr),
                    f"Q={Q_pr:.3f} Н·м  p_dot={ds[P]:.3f}  r_dot={ds[R]:.4f}")

    # 7.3 Вырождение в продольные уравнения (docs/physics.md, раздел 3.5)
    Va_t, al_t, _ = air_data(s_test, wind_long(0.0, 0.0))
    fx_a, fz_a, M_a = aero_forces_moments(Va_t, al_t, s_test[Q], de_tr, ap)
    th, u_, w_, q_ = s_test[THETA], s_test[U], s_test[W], s_test[Q]
    mg = ap.mass * ap.g
    ref = np.array([
        (fx_a + thrust(thr_tr, Va_t, ap) - mg * np.sin(th)) / ap.mass - q_ * w_,
        (fz_a + mg * np.cos(th)) / ap.mass + q_ * u_,
        M_a / ap.Jy,
        q_,
        u_ * np.cos(th) + w_ * np.sin(th),
        u_ * np.sin(th) - w_ * np.cos(th),
    ])
    err = np.max(np.abs(ds[[U, W, Q, THETA, X, H]] - ref))
    all_ok &= check("6DOF -> продольные уравнения при v=p=r=phi=psi=0", err < 1e-9,
                    f"max|err|={err:.1e}")

    # 7.4 Инвариантность к курсу: полёт на восток (psi=90°) — та же продольная динамика
    s_e = s_tr.copy(); s_e[PSI] = np.pi / 2
    d_n = derivatives(s_tr, c_tr, 0.0, ap, calm)
    d_e = derivatives(s_e,  c_tr, 0.0, ap, calm)
    all_ok &= check("psi=90°: производные u,w,q,theta,h не меняются",
                    np.allclose(d_n[[U, W, Q, THETA, H]], d_e[[U, W, Q, THETA, H]], atol=1e-12))
    all_ok &= check("psi=90°: dy/dt = Va, dx/dt = 0",
                    abs(d_e[Y] - cfg.Va0) < 1e-9 and abs(d_e[X]) < 1e-9,
                    f"dx={d_e[X]:.1e}  dy={d_e[Y]:.3f}")

    # 7.5 Знак УС: ветер на восток при psi=0 -> поток слева -> beta < 0
    _, _, b_cw = air_data(s_tr, (0.0, 0.0, 5.0))
    all_ok &= check("боковой ветер слева -> beta < 0", b_cw < 0,
                    f"beta={np.degrees(b_cw):.2f} deg")
    # Скольжение v>0 (поток справа): крен влево (p_dot<0), нос вправо (r_dot>0)
    s_b = s_tr.copy(); s_b[V] = 2.0
    d_b = derivatives(s_b, c_tr, 0.0, ap, calm)
    all_ok &= check("beta>0: поперечная устойчивость, p_dot<0", d_b[P] < 0, f"p_dot={d_b[P]:.3f}")
    all_ok &= check("beta>0: флюгерная устойчивость, r_dot>0",  d_b[R] > 0, f"r_dot={d_b[R]:.3f}")

    # 7.6 Знаки рулей
    d_a = derivatives(s_tr, c_tr + [0, 0, np.radians(5), 0], 0.0, ap, calm)
    d_r = derivatives(s_tr, c_tr + [0, 0, 0, np.radians(5)], 0.0, ap, calm)
    all_ok &= check("delta_a>0 -> крен вправо (p_dot>0)", d_a[P] > 0, f"p_dot={d_a[P]:.3f}")
    all_ok &= check("delta_r>0 -> нос влево (r_dot<0)",   d_r[R] < 0, f"r_dot={d_r[R]:.3f}")

    # 7.7 Связь крен-рыскание через Jxz (B&M упр. 3.4)
    G_0  = _gammas(replace(ap, Jxz=0.0))
    G_xz = _gammas(replace(ap, Jxz=0.05))
    all_ok &= check("Jxz=0: момент крена не даёт r_dot (Г4=0)", G_0[3] == 0.0)
    all_ok &= check("Jxz≠0: момент крена даёт r_dot (Г4≠0)",   G_xz[3] != 0.0,
                    f"Г4={G_xz[3]:.4f}")

    # 7.8 Прогон 20 с из трима: момент винта парирует САУ по крену (РЕШ-19)
    from control.controllers import with_roll_hold
    from sim.config import SensorParams
    from runner import run
    log_rh = run(with_roll_hold(lambda t_, s_, V_, a_: c_tr[:2], ap, SensorParams(), cfg.dt),
                 ap, WindParams(), replace(cfg, t_end=20.0), state0=s_tr)
    phi_max = np.degrees(np.max(np.abs(log_rh.state[:, PHI])))
    all_ok &= check("20 с трима с САУ по крену: |phi| < 1°", phi_max < 1.0,
                    f"max|phi|={phi_max:.2f}°  δa={np.degrees(log_rh.controls[-1, 2]):.2f}°")

    # 7.9 Собственные движения бокового канала (линеаризация в триме)
    eps = 1e-6
    idx = [V, P, R, PHI]
    f0 = derivatives(s_tr, c_tr, 0.0, ap, calm)
    A = np.zeros((4, 4))
    for j, k in enumerate(idx):
        s_p = s_tr.copy(); s_p[k] += eps
        A[:, j] = (derivatives(s_p, c_tr, 0.0, ap, calm)[idx] - f0[idx]) / eps
    eig = np.linalg.eigvals(A)
    cplx = eig[np.abs(eig.imag) > 1e-6]
    real = np.sort(eig[np.abs(eig.imag) <= 1e-6].real)
    print("        собственные значения [v,p,r,phi]: "
          + ", ".join(f"{e.real:+.3f}{e.imag:+.3f}j" for e in eig))
    all_ok &= check("голландский шаг: комплексная пара", len(cplx) == 2)
    if len(cplx) == 2 and len(real) == 2:
        wn = abs(cplx[0]); zeta = -cplx[0].real / wn
        all_ok &= check("голландский шаг затухает", cplx[0].real < 0,
                        f"wn={wn:.2f} рад/с  zeta={zeta:.2f}  T={2*np.pi/abs(cplx[0].imag):.2f} с")
        all_ok &= check("апериодика крена: быстрая и устойчивая", real[0] < -1.0,
                        f"lambda={real[0]:.2f} 1/с  tau={-1/real[0]:.2f} с")
        spiral = real[1]
        print(f"        спиральная мода: lambda={spiral:+.4f} 1/с  "
              + ("(устойчива)" if spiral < 0 else f"(неустойчива, T2={np.log(2)/spiral:.1f} с)"))

except Exception as e:
    import traceback; traceback.print_exc()
    check("6DOF", False, str(e)); all_ok = False

# ------------------------------------------------------------------
section("8. Боковая САУ (control/controllers.py: LateralController)")
# ------------------------------------------------------------------
try:
    from control.controllers import LateralController, LateralControlParams, wrap_angle
    from control.sensors import measure_gps_course

    all_ok &= check("wrap_angle(350°) = −10°",
                    abs(np.degrees(wrap_angle(np.radians(350))) + 10) < 1e-9)
    all_ok &= check("GPS-курс: скорость на восток -> chi = 90°",
                    abs(np.degrees(measure_gps_course(0.0, 30.0, 0.0, None)) - 90) < 1e-9)

    level = {'chi': 0.0, 'phi': 0.0, 'p': 0.0, 'beta': 0.0}
    lc = LateralController(ap, LateralControlParams())
    lc.set_course(np.radians(20))
    da, dr = lc.step(level, cfg.dt)
    all_ok &= check("курс правее -> phi_ref > 0, delta_a > 0 (крен вправо)",
                    lc.phi_ref > 0 and da > 0,
                    f"phi_ref={np.degrees(lc.phi_ref):.1f}°  δa={np.degrees(da):.1f}°")

    lc = LateralController(ap, LateralControlParams())
    lc.set_course(np.radians(-170))           # из 170° в −170°: кратчайший путь вправо
    lc.step({**level, 'chi': np.radians(170)}, cfg.dt)
    all_ok &= check("170° -> −170°: разворот по кратчайшему пути (вправо)",
                    lc.phi_ref > 0, f"phi_ref={np.degrees(lc.phi_ref):.1f}°")

    lc = LateralController(ap, LateralControlParams())
    _, dr = lc.step({**level, 'beta': np.radians(3)}, cfg.dt)
    all_ok &= check("beta > 0 -> delta_r < 0 (нос вправо, на поток)", dr < 0,
                    f"δr={np.degrees(dr):.2f}°")
    lc = LateralController(ap, LateralControlParams(beta_hold=False))
    _, dr = lc.step({**level, 'beta': np.radians(3)}, cfg.dt)
    all_ok &= check("beta_hold=False -> delta_r = 0", dr == 0.0)

    lc = LateralController(ap, LateralControlParams())
    lc.set_course(np.pi / 2)
    lc.step(level, cfg.dt)
    all_ok &= check("уставка крена ограничена ±phi_ref_max",
                    abs(lc.phi_ref) <= LateralControlParams().phi_ref_max + 1e-12,
                    f"phi_ref={np.degrees(lc.phi_ref):.1f}°")

except Exception as e:
    import traceback; traceback.print_exc()
    check("LateralController", False, str(e)); all_ok = False

# ------------------------------------------------------------------
print("\n" + "="*50)
if all_ok:
    print("  ALL CHECKS PASSED")
else:
    print("  FAILURES FOUND -- see [!!] above")
print("="*50)

sys.exit(0 if all_ok else 1)

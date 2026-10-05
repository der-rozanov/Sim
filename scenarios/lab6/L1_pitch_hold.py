# -*- coding: utf-8 -*-
"""
L1. Удержание угла тангажа (Лекция 6, слайды 18–20, 9)
=======================================================

Контур тангажа по Beard & McLain: пропорциональный по θ + обратная связь по q,
БЕЗ интегратора:

    δe = δe* + k_pθ·(θᶜ − θ) − k_dθ·q

Объект (короткопериодическое приближение, Va = const):

    θ(s)/δe(s) = a_θ3 / (s² + a_θ1·s + a_θ2)

Замкнутый контур:  θ/θᶜ = k_pθ·a_θ3 / (s² + (a_θ1 + a_θ3·k_dθ)·s + (a_θ2 + a_θ3·k_pθ))

ЧТО СДЕЛАТЬ
-----------
  TODO 1. pitch_tf_coeffs() — a_θ1, a_θ2, a_θ3 из параметров ЛА.
  TODO 2. pitch_gains()     — k_pθ, k_dθ по заданным ω_nθ, ζ_θ (слайд 19).
  TODO 3. pitch_control()   — закон управления (формула выше).
Запустите файл: внизу печатается самопроверка [✓]/[✗]. Сценарий запускается
и до выполнения TODO — тогда руль просто стоит на балансировочном значении.

ПОДСКАЗКИ (чтобы не застрять)
-----------------------------
  * Уравнение моментов: Jy·q̇ = ½ρVa²·S·c·Cm,
    Cm = Cm0 + Cma·α + Cmq·(c/(2Va))·q + Cmde·δe.
    Сопоставьте с q̇ = −a_θ1·q − a_θ2·α + a_θ3·δe  (при α ≈ θ).
  * Все параметры — в aircraft: rho, S, c, Jy, Cma, Cmq, Cmde.
  * Знаки: Cmde < 0 (δe > 0 — нос ВНИЗ), поэтому a_θ3 < 0 и k_pθ, k_dθ
    получаются ОТРИЦАТЕЛЬНЫМИ. Это правильно, не «исправляйте» знак.
  * Все углы в коде — в радианах.

ИССЛЕДОВАНИЕ (ответы — в отчёт)
-------------------------------
  1. Таблица a_θ при Va = 15/25/35 м/с печатается автоматически. Как a_θ3
     зависит от Va? Как это связано с gain scheduling ∝ (Va_ref/Va)² в
     PitchController проекта (control/controllers.py)?
  2. EXPERIMENT = "base". Сравните θ(t) с теорией. Где совпадает, где нет?
     Теория обещает θ → K_θDC·Δθ. Почему в симуляции θ уходит ближе к Δθ?
     (Посмотрите на график α и γ = θ − α.)
  3. EXPERIMENT = "omega". При каком ω_nθ руль высоты уходит в упор?
     Что происходит при ω_nθ < √a_θ2? Почему линейная теория (полюса
     s² + 2ζω s + ω²) при этом обещает устойчивость, а ЛА расходится?
  4. Почему B&M не ставят интегратор в контур тангажа (слайд 9)? Чем этот
     контур отличается от каскадного ПИД θ→q→δe в PitchController проекта?

Запуск:  python scenarios/lab6/L1_pitch_hold.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.dirname(_here)]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from lab6_common import (AircraftParams, WindParams, SimConfig, get_variant, trim, print_trim,
                         make_sensors, numeric_pitch_coeffs, lti2_step, check, check_close,
                         check_summary, check_flight_safe, save_fig)
from runner import run
from sim.state import THETA, H

# ------------------------------------------------------------------
# Настройки
# ------------------------------------------------------------------
VARIANT      = 1          # ваш номер варианта (1–6)
EXPERIMENT   = "base"     # "base" — ступенька по θ;  "omega" — перебор ω_nθ
SENSOR_NOISE = True       # шум гироскопа/ИНС


# ==================================================================
# TODO 1. Коэффициенты передаточной функции θ(s)/δe(s)
# ==================================================================
def pitch_tf_coeffs(aircraft, Va):
    """Вернуть (a_theta1, a_theta2, a_theta3) при воздушной скорости Va."""
    # ваш код здесь
    return None


# ==================================================================
# TODO 2. Коэффициенты регулятора по ω_nθ, ζ_θ (слайд 19)
# ==================================================================
def pitch_gains(a1, a2, a3, wn, zeta):
    """Вернуть (k_ptheta, k_dtheta)."""
    # ваш код здесь
    return None


# ==================================================================
# TODO 3. Закон управления
# ==================================================================
def pitch_control(theta_c, theta, q, delta_e_trim, kp, kd):
    """Вернуть δe, рад (насыщение руля добавляется снаружи)."""
    # ваш код здесь
    return None


# ==================================================================
# Ниже править не нужно
# ==================================================================

def simulate(aircraft, tr, v, gains, t_end):
    """Ступенька θᶜ = θ* + Δθ в t = 1 с; газ фиксирован на балансировочном."""
    meas = make_sensors(SENSOR_NOISE)
    dth = np.radians(v["dtheta"])

    def controls_fn(t, state, Va, alpha):
        m = meas(state, Va)
        theta_c = tr["theta"] + (dth if t >= T_STEP else 0.0)
        if gains is None:
            de = tr["delta_e"]
        else:
            de = pitch_control(theta_c, m["theta"], m["q"], tr["delta_e"], *gains)
            if de is None:
                de = tr["delta_e"]
        de = np.clip(de, aircraft.delta_e_min, aircraft.delta_e_max)
        return np.array([de, tr["throttle"]])

    return run(controls_fn, aircraft, WindParams(), SimConfig(dt=0.01, t_end=t_end),
               state0=tr["state"])


def theory_theta(t, coeffs, gains, dtheta_deg):
    a1, a2, a3 = coeffs
    kp, kd = gains
    return lti2_step(t, 0.0, kp * a3, a1 + a3 * kd, a2 + a3 * kp, amp=dtheta_deg, t_step=T_STEP)


T_STEP = 1.0
aircraft = AircraftParams()
v = get_variant(VARIANT)
tr = trim(aircraft, v["Va"])
print_trim(tr)

coeffs = pitch_tf_coeffs(aircraft, v["Va"])
gains = pitch_gains(*coeffs, v["wn_theta"], v["zeta_theta"]) if coeffs is not None else None
if gains is None:
    print("\n!!! TODO 1–2 не выполнены: руль стоит на балансировке (разомкнутый контур).\n")
else:
    a1, a2, a3 = coeffs
    kp, kd = gains
    K_dc = kp * a3 / (a2 + kp * a3)
    print(f"a_θ1={a1:.4f}  a_θ2={a2:.4f}  a_θ3={a3:.4f}")
    print(f"k_pθ={kp:.4f}  k_dθ={kd:.4f}  K_θDC={K_dc:.3f}")
    print("\nЗависимость от скорости (вопрос 1):")
    for Va_i in (15.0, 25.0, 35.0):
        b = pitch_tf_coeffs(aircraft, Va_i)
        print(f"  Va={Va_i:4.0f} м/с:  a_θ1={b[0]:7.3f}  a_θ2={b[1]:7.3f}  a_θ3={b[2]:8.3f}")

# ------------------------------------------------------------------
if EXPERIMENT == "base":
    T_END = 15.0
    log = simulate(aircraft, tr, v, gains, T_END)
    t = log.t
    th = np.degrees(log.state[:, THETA] - tr["theta"])
    al = np.degrees(log.alpha - tr["alpha"])
    gam = th - al

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, np.where(t >= T_STEP, v["dtheta"], 0.0), "k--", label="θᶜ − θ*")
    ax[0].plot(t, th, label="θ − θ*  (симуляция)")
    if gains is not None:
        ax[0].plot(t, theory_theta(t, coeffs, gains, v["dtheta"]), ":", lw=2, label="теория 2-го порядка")
        ax[0].axhline(K_dc * v["dtheta"], color="gray", lw=0.8, label="K_θDC·Δθ")
    ax[0].set_ylabel("град"); ax[0].legend(); ax[0].set_title(f"L1: ступенька по тангажу, вариант {VARIANT}")
    ax[1].plot(t, al, label="α − α*"); ax[1].plot(t, gam, label="γ = θ − α")
    ax[1].set_ylabel("град"); ax[1].legend()
    ax[2].plot(t, np.degrees(log.controls[:, 0]), label="δe")
    ax[2].axhline(np.degrees(aircraft.delta_e_min), color="r", lw=0.8, ls="--", label="упор руля")
    ax[2].set_ylabel("град"); ax[2].legend()
    ax[3].plot(t, log.Va, label="Va, м/с"); ax[3].plot(t, log.state[:, H] - tr["state"][H], label="Δh, м")
    ax[3].set_xlabel("t, с"); ax[3].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L1_base_v{VARIANT}")

    if gains is not None:
        i_pk = np.argmax(th)
        y_th = theory_theta(t, coeffs, gains, v["dtheta"])
        print(f"\n  пик θ: симуляция {th[i_pk]:.2f}° в t={t[i_pk] - T_STEP:.2f} с; "
              f"теория {y_th.max():.2f}° в t={t[np.argmax(y_th)] - T_STEP:.2f} с")
        print(f"  θ в конце: {th[-1]:.2f}°   (Δθ = {v['dtheta']}°, K_θDC·Δθ = {K_dc * v['dtheta']:.2f}°)")

    # ---------------- самопроверка ----------------
    print("\nСамопроверка L1:")
    an = numeric_pitch_coeffs(aircraft, tr)
    if coeffs is None:
        check("TODO 1", False, "не реализовано")
    else:
        for name, val, ref in zip(("a_θ1", "a_θ2", "a_θ3"), coeffs, an):
            check_close(name, val, ref)
    if gains is None:
        check("TODO 2", False, "не реализовано")
    else:
        wn, z = v["wn_theta"], v["zeta_theta"]
        check_close("a_θ1 + a_θ3·k_dθ = 2ζω", an[0] + an[2] * kd, 2 * z * wn)
        check_close("a_θ2 + a_θ3·k_pθ = ω²", an[1] + an[2] * kp, wn**2)
        de0 = pitch_control(0.1, 0.1, 0.0, -0.05, kp, kd)
        if de0 is None:
            check("TODO 3", False, "не реализовано")
        else:
            eps = 1e-3
            check_close("δe = δe* при θᶜ = θ, q = 0", de0, -0.05, 1e-6)
            check_close("∂δe/∂θᶜ = k_pθ", (pitch_control(0.1 + eps, 0.1, 0.0, -0.05, kp, kd) - de0) / eps, kp, 1e-3)
            check_close("∂δe/∂q = −k_dθ", (pitch_control(0.1, 0.1, eps, -0.05, kp, kd) - de0) / eps, -kd, 1e-3)
            t_front = 2.0 / v["wn_theta"]          # фронт переходного процесса
            m = (t >= T_STEP) & (t <= T_STEP + t_front)
            err = np.max(np.abs(th[m] - y_th[m]))
            check(f"на фронте (первые {t_front:.2f} с) отклик совпадает с теорией",
                  err < 0.15 * abs(v["dtheta"]), f"макс. расхождение {err:.2f}°")
            check_flight_safe(log, aircraft, T_END)
    check_summary()

# ------------------------------------------------------------------
elif EXPERIMENT == "omega":
    if gains is None:
        sys.exit("Сначала выполните TODO 1–3.")
    T_END = 8.0
    a1, a2, a3 = coeffs
    wn0 = v["wn_theta"]
    omegas = [0.8 * np.sqrt(a2), wn0, 2 * wn0, 4 * wn0]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    print(f"\n√a_θ2 = {np.sqrt(a2):.2f} рад/с")
    print(f"{'ω_nθ':>6} {'k_pθ':>8} {'K_θDC':>7} {'θ_пик, °':>9} {'θ_конец, °':>11} {'δe min/max, °':>16}  упор?")
    for wn in omegas:
        g = pitch_gains(a1, a2, a3, wn, v["zeta_theta"])
        log = simulate(aircraft, tr, v, g, T_END)
        th = np.degrees(log.state[:, THETA] - tr["theta"])
        de = np.degrees(log.controls[:, 0])
        sat = np.any(np.isclose(log.controls[:, 0], aircraft.delta_e_min)) or \
              np.any(np.isclose(log.controls[:, 0], aircraft.delta_e_max))
        Kdc = g[0] * a3 / (a2 + g[0] * a3)
        print(f"{wn:6.2f} {g[0]:8.3f} {Kdc:7.3f} {th.max():9.2f} {th[-1]:11.2f} {de.min():7.1f}/{de.max():6.1f}   {'ДА' if sat else 'нет'}")
        ax[0].plot(log.t, th, label=f"ω_nθ = {wn:.1f}")
        ax[1].plot(log.t, de, label=f"ω_nθ = {wn:.1f}")
    ax[0].axhline(v["dtheta"], color="k", ls="--", lw=0.8)
    ax[0].set_ylim(-3 * abs(v["dtheta"]), 3 * abs(v["dtheta"]))
    ax[0].set_ylabel("θ − θ*, град"); ax[0].legend(); ax[0].set_title("L1: влияние ω_nθ")
    ax[1].axhline(np.degrees(aircraft.delta_e_min), color="r", ls="--", lw=0.8)
    ax[1].axhline(np.degrees(aircraft.delta_e_max), color="r", ls="--", lw=0.8)
    ax[1].set_ylabel("δe, град"); ax[1].set_xlabel("t, с"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L1_omega_v{VARIANT}")

plt.show()

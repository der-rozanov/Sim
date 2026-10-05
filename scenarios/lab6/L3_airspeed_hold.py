# -*- coding: utf-8 -*-
"""
L3. Удержание воздушной скорости через газ (Лекция 6, слайды 23–24)
==================================================================

Объект (линеаризация около трима, α и γ — константы):

    V̄a(s)/δ̄t(s) = a_V2 / (s + a_V1)

Регулятор — ПИ с балансировочной тягой δt* как прямой связью (слайд 24):

    δt = δt* + k_pV·(Vaᶜ − Va) + k_iV·∫(Vaᶜ − Va) dt,     δt ∈ [0, 1]

Замкнутый контур:  Va/Vaᶜ = a_V2(k_pV·s + k_iV) / (s² + (a_V1 + a_V2·k_pV)·s + a_V2·k_iV)

Контуры тангажа и высоты здесь уже готовы (как в L1, L2): ЛА держит высоту,
вы управляете только газом.

ЧТО СДЕЛАТЬ
-----------
  TODO 1. speed_tf_coeffs() — a_V1, a_V2 из параметров ЛА и балансировки.
  TODO 2. speed_gains()     — k_pV, k_iV по ω_nV, ζ_V (слайд 24).
  TODO 3. speed_control()   — закон управления с насыщением [0, 1] и анти-виндапом.

ПОДСКАЗКИ
---------
  * Вдоль скорости:  m·V̇a = T − D − m·g·sin γ.
    Сопротивление:  D = ½ρVa²·S·CD(α*),   CD(α*) = coef_CD(alpha_trim, aircraft)
                    (уже импортирована, модель CD квадратичная).
    Тяга (модель винта проекта):  T = ½ρ·S_prop·C_prop·((k_motor·δt)² − Va²).
    Продифференцируйте правую часть по Va и по δt в точке трима (α, γ = const).
  * Параметры: aircraft.rho, S, mass, S_prop, C_prop, k_motor.
  * Интегратор — в mem["I"]; анти-виндап — как в L2 (не копить при насыщении).

ИССЛЕДОВАНИЕ (ответы — в отчёт)
-------------------------------
  1. a_V1 состоит из двух слагаемых. Какое больше? Каков физический смысл
     каждого?
  2. EXPERIMENT = "base". Сравните Va(t) с теорией. Посмотрите на график тяги:
     что происходит с T при разгоне? Как ведёт себя высота при смене скорости
     и кто её возвращает?
  3. EXPERIMENT = "omega". Как меняется перерегулирование с ростом ω_nV в
     симуляции и в теории? Почему при больших ω_nV симуляция расходится с
     теорией сильнее, хотя δt ещё не упёрся в 1? (Подсказка: sim/dynamics.py,
     функция thrust.) Почему анти-виндап этого не замечает?
  4. EXPERIMENT = "feedforward". Слайд 24: «если δt* известно неточно,
     интегратор это скомпенсирует». Проверьте для δt*, занижённого на 30%,
     и для δt* = 0. Чем платим за неточную прямую связь?
  5. EXPERIMENT = "compare". Сравните свой регулятор с SpeedController проекта
     (control/controllers.py). Почему у одного время установления в разы
     больше? В чём преимущество синтеза по ω_n, ζ перед ручным подбором?

Запуск:  python scenarios/lab6/L3_airspeed_hold.py
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
                         make_sensors, numeric_speed_coeffs, lti2_step, step_metrics, check,
                         check_close, check_summary, check_flight_safe, save_fig,
                         PitchLoopReady, AltitudeLoopReady)
from runner import run
from sim.aero import coef_CD
from sim.dynamics import thrust
from sim.state import H
from control.controllers import SpeedController, SpeedControlParams

# ------------------------------------------------------------------
# Настройки
# ------------------------------------------------------------------
VARIANT      = 1          # ваш номер варианта (1–6)
EXPERIMENT   = "base"     # "base" | "omega" | "feedforward" | "compare"
SENSOR_NOISE = True


# ==================================================================
# TODO 1. Коэффициенты передаточной функции Va(s)/δt(s)
# ==================================================================
def speed_tf_coeffs(aircraft, Va, alpha_trim, delta_t_trim):
    """Вернуть (a_V1, a_V2)."""
    # ваш код здесь
    return None


# ==================================================================
# TODO 2. Коэффициенты ПИ-регулятора скорости (слайд 24)
# ==================================================================
def speed_gains(aV1, aV2, wn, zeta):
    """Вернуть (k_pV, k_iV)."""
    # ваш код здесь
    return None


# ==================================================================
# TODO 3. Закон управления газом
# ==================================================================
def speed_control(Va_c, Va, mem, dt, kp, ki, delta_t_ff):
    """Вернуть δt ∈ [0, 1]. mem["I"] — интегратор, delta_t_ff — прямая связь δt*."""
    # ваш код здесь
    return None


# ==================================================================
# Ниже править не нужно
# ==================================================================
T_STEP = 2.0
aircraft = AircraftParams()
v = get_variant(VARIANT)
tr = trim(aircraft, v["Va"])
print_trim(tr)
h0 = tr["state"][H]
Va_c1 = v["Va"] + v["dVa"]

pitch = PitchLoopReady(aircraft, tr, v["wn_theta"], v["zeta_theta"])


def simulate(gains, t_end, ff_scale=1.0, controller="student"):
    """controller: "student" — ваш ПИ;  "project" — SpeedController проекта."""
    meas = make_sensors(SENSOR_NOISE)
    alt = AltitudeLoopReady(tr, v["wn_theta"], pitch.K_dc, v["W_h"], v["zeta_h"])
    proj = SpeedController(aircraft, SpeedControlParams())
    proj.set_trim_throttle(tr["throttle"])
    mem = {"I": 0.0}

    def controls_fn(t, state, Va, alpha):
        m = meas(state, Va)
        Va_c = Va_c1 if t >= T_STEP else v["Va"]
        theta_c = alt(h0, m["h"], 0.01)
        if controller == "project":
            proj.set_Va_ref(Va_c)
            dt_cmd = proj.step(m["Va"], 0.01)
        else:
            dt_cmd = None
            if gains is not None:
                dt_cmd = speed_control(Va_c, m["Va"], mem, 0.01, *gains, ff_scale * tr["throttle"])
            if dt_cmd is None:
                dt_cmd = tr["throttle"]
        return np.array([pitch(theta_c, m["theta"], m["q"]), dt_cmd])

    return run(controls_fn, aircraft, WindParams(), SimConfig(dt=0.01, t_end=t_end), state0=tr["state"])


def theory_Va(t, coeffs, gains):
    aV1, aV2 = coeffs
    kp, ki = gains
    return v["Va"] + lti2_step(t, aV2 * kp, aV2 * ki, aV1 + aV2 * kp, aV2 * ki, amp=v["dVa"], t_step=T_STEP)


def thrust_log(log):
    return np.array([thrust(c[1], Va, aircraft) for c, Va in zip(log.controls, log.Va)])


def fmt(m):
    ts = "∞" if not np.isfinite(m["t_settle"]) else f"{m['t_settle']:.1f}"
    return f"перерег. {m['overshoot']:5.1f}%   t_уст(5%) {ts:>5} с   e_уст {m['e_ss']:+.2f} м/с"


def todo_done():
    return gains is not None and speed_control(v["Va"], v["Va"], {"I": 0.0}, 0.01, *gains, 0.4) is not None


coeffs = speed_tf_coeffs(aircraft, v["Va"], tr["alpha"], tr["throttle"])
gains = speed_gains(*coeffs, v["wn_V"], v["zeta_V"]) if coeffs is not None else None
if gains is None:
    print("\n!!! TODO 1–2 не выполнены: газ стоит на балансировке.\n")
else:
    print(f"a_V1={coeffs[0]:.4f}  a_V2={coeffs[1]:.4f}   k_pV={gains[0]:.5f}  k_iV={gains[1]:.5f}")

# ------------------------------------------------------------------
if EXPERIMENT == "base":
    T_END = 40.0
    log = simulate(gains, T_END)
    t = log.t
    T = thrust_log(log)

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, np.where(t >= T_STEP, Va_c1, v["Va"]), "k--", label="Vaᶜ")
    ax[0].plot(t, log.Va, label="Va (симуляция)")
    if gains is not None:
        ax[0].plot(t, theory_Va(t, coeffs, gains), ":", lw=2, label="теория")
    ax[0].set_ylabel("м/с"); ax[0].legend(); ax[0].set_title(f"L3: ступенька по скорости, вариант {VARIANT}")
    ax[1].plot(t, log.controls[:, 1], label="δt"); ax[1].set_ylabel("газ"); ax[1].legend()
    ax[2].plot(t, T, label="тяга T"); ax[2].axhline(aircraft.T_max, color="r", ls="--", lw=0.8, label="T_max")
    ax[2].set_ylabel("Н"); ax[2].legend()
    ax[3].plot(t, log.state[:, H] - h0, label="Δh"); ax[3].set_ylabel("м"); ax[3].set_xlabel("t, с"); ax[3].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L3_base_v{VARIANT}")

    print("\nСамопроверка L3:")
    an = numeric_speed_coeffs(aircraft, tr)
    if coeffs is None:
        check("TODO 1", False, "не реализовано")
    else:
        check_close("a_V1", coeffs[0], an[0])
        check_close("a_V2", coeffs[1], an[1])
    if gains is None:
        check("TODO 2", False, "не реализовано")
    else:
        wn, z = v["wn_V"], v["zeta_V"]
        check_close("a_V1 + a_V2·k_pV = 2ζω", coeffs[0] + coeffs[1] * gains[0], 2 * z * wn, 1e-3)
        check_close("a_V2·k_iV = ω²", coeffs[1] * gains[1], wn**2, 1e-3)
        if not todo_done():
            check("TODO 3", False, "не реализовано")
        else:
            kp, ki = gains
            check_close("δt = δt* при Vaᶜ = Va", speed_control(25.0, 25.0, {"I": 0.0}, 0.01, kp, ki, 0.4), 0.4, 1e-6)
            d = speed_control(25.1, 25.0, {"I": 0.0}, 0.0, kp, ki, 0.4) - 0.4
            check_close("∂δt/∂e = k_pV", d / 0.1, kp, 1e-3)
            check("насыщение δt ∈ [0, 1]",
                  speed_control(1e4, 0.0, {"I": 0.0}, 0.0, kp, ki, 0.4) == 1.0 and
                  speed_control(0.0, 1e4, {"I": 0.0}, 0.0, kp, ki, 0.4) == 0.0)
            m = step_metrics(t, log.Va, v["Va"], Va_c1, T_STEP)
            mt = step_metrics(t, theory_Va(t, coeffs, gains), v["Va"], Va_c1, T_STEP)
            print(f"  симуляция: {fmt(m)}\n  теория:    {fmt(mt)}")
            check("установившаяся ошибка < 0.1 м/с", abs(m["e_ss"]) < 0.1, f"{m['e_ss']:+.2f} м/с")
            check("перерегулирование < 25%", m["overshoot"] < 25.0, f"{m['overshoot']:.1f}%")
            dh = np.abs(log.state[:, H] - h0).max()
            check("высота удержана (|Δh| < 2 м)", dh < 2.0, f"max |Δh| = {dh:.2f} м")
            check_flight_safe(log, aircraft, T_END)
    check_summary()

# ------------------------------------------------------------------
elif EXPERIMENT == "omega":
    if not todo_done():
        sys.exit("Сначала выполните TODO 1–3.")
    T_END = 40.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    print(f"\n{'ω_nV':>5}  {'перерег. сим/теор, %':>21}  {'t_уст, с':>8}  {'δt max':>6}  {'T max, Н':>8}  {'|Δh| max, м':>11}")
    for wn in (0.3, v["wn_V"], 2.0, 5.0):
        g = speed_gains(*coeffs, wn, v["zeta_V"])
        log = simulate(g, T_END)
        T = thrust_log(log)
        m = step_metrics(log.t, log.Va, v["Va"], Va_c1, T_STEP)
        mt = step_metrics(log.t, theory_Va(log.t, coeffs, g), v["Va"], Va_c1, T_STEP)
        ts = "∞" if not np.isfinite(m["t_settle"]) else f"{m['t_settle']:.1f}"
        print(f"{wn:5.1f}  {m['overshoot']:9.1f} / {mt['overshoot']:9.1f}  {ts:>8}  {log.controls[:, 1].max():6.2f}"
              f"  {T.max():8.1f}  {np.abs(log.state[:, H] - h0).max():11.2f}")
        ax[0].plot(log.t, log.Va, label=f"ω_nV = {wn}")
        ax[1].plot(log.t, log.controls[:, 1], label=f"ω_nV = {wn}")
        ax[2].plot(log.t, T, label=f"ω_nV = {wn}")
    ax[0].plot(log.t, np.where(log.t >= T_STEP, Va_c1, v["Va"]), "k--", lw=0.8)
    ax[0].set_ylabel("Va, м/с"); ax[0].legend(); ax[0].set_title("L3: влияние ω_nV")
    ax[1].set_ylabel("δt"); ax[1].legend()
    ax[2].axhline(aircraft.T_max, color="r", ls="--", lw=0.8)
    ax[2].set_ylabel("T, Н"); ax[2].set_xlabel("t, с"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L3_omega_v{VARIANT}")

# ------------------------------------------------------------------
elif EXPERIMENT == "feedforward":
    if not todo_done():
        sys.exit("Сначала выполните TODO 1–3.")
    T_END = 60.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    print()
    for name, k in (("δt* точное", 1.0), ("δt* занижено на 30%", 0.7), ("δt* = 0", 0.0)):
        log = simulate(gains, T_END, ff_scale=k)
        m = step_metrics(log.t, log.Va, v["Va"], Va_c1, T_STEP)
        print(f"  {name:22s} Va min {log.Va.min():5.1f} м/с   |Δh| max {np.abs(log.state[:, H] - h0).max():5.2f} м   "
              f"после ступеньки: e_уст {m['e_ss']:+.2f} м/с")
        ax[0].plot(log.t, log.Va, label=name)
        ax[1].plot(log.t, log.controls[:, 1], label=name)
        ax[2].plot(log.t, log.state[:, H] - h0, label=name)
    ax[0].set_ylabel("Va, м/с"); ax[0].legend(); ax[0].set_title("L3: роль прямой связи δt*")
    ax[1].set_ylabel("δt"); ax[1].legend()
    ax[2].set_ylabel("Δh, м"); ax[2].set_xlabel("t, с"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L3_feedforward_v{VARIANT}")

# ------------------------------------------------------------------
elif EXPERIMENT == "compare":
    if not todo_done():
        sys.exit("Сначала выполните TODO 1–3.")
    T_END = 60.0
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    print()
    for name, ctrl in (("ваш ПИ (синтез по ω_n, ζ)", "student"), ("SpeedController проекта", "project")):
        log = simulate(gains, T_END, controller=ctrl)
        m = step_metrics(log.t, log.Va, v["Va"], Va_c1, T_STEP)
        print(f"  {name:28s} {fmt(m)}")
        ax[0].plot(log.t, log.Va, label=name)
        ax[1].plot(log.t, log.controls[:, 1], label=name)
    ax[0].plot(log.t, np.where(log.t >= T_STEP, Va_c1, v["Va"]), "k--", lw=0.8)
    ax[0].set_ylabel("Va, м/с"); ax[0].legend(); ax[0].set_title("L3: сравнение с регулятором проекта")
    ax[1].set_ylabel("δt"); ax[1].set_xlabel("t, с"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L3_compare_v{VARIANT}")

plt.show()

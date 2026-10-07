# -*- coding: utf-8 -*-
"""
L2. Удержание высоты через тангаж (Лекция 6, слайды 17, 20–22, 31)
=================================================================

Последовательное замыкание контуров: внутренний контур тангажа (из L1, здесь
он уже готов) считаем звеном θ ≈ K_θDC·θᶜ, а высота — интегратор ḣ ≈ Va·θ.
Регулятор высоты — ПИ:

    θᶜ = θ* + k_ph·e + k_ih·∫e dt,     e = sat(hᶜ, h ± h_hold) − h
    θᶜ ограничивается ±θ_max, интегратор защищён от насыщения (анти-виндап).

Замкнутый контур (слайд 21–22):
    h/hᶜ = (K_θDC·Va·k_ph·s + K_θDC·Va·k_ih) / (s² + K_θDC·Va·k_ph·s + K_θDC·Va·k_ih)
    ω_nh = ω_nθ / W_h  — разнос полос пропускания.

Газ держит ГОТОВЫЙ регулятор скорости проекта (SpeedController), чтобы при
наборе не терять скорость. Свой регулятор скорости вы сделаете в L3.

ЧТО СДЕЛАТЬ
-----------
  TODO 1. altitude_gains()   — k_ph, k_ih по ω_nθ, W_h, ζ_h, K_θDC, Va (слайд 22).
  TODO 2. altitude_control() — закон управления (формула выше) с зоной h_hold,
          ограничением ±θ_max и анти-виндапом.

ПОДСКАЗКИ
---------
  * Интегратор хранится в словаре mem["I"] (начальное значение 0).
    Простейшее интегрирование: mem["I"] += e*dt.
  * Анти-виндап (флаг anti_windup): если θᶜ упёрлось в ограничение — не
    накапливать интеграл на этом шаге (условное интегрирование). Можно и
    вариант слайда 31 — оба подходят.
  * Зона удержания: np.clip(h_c, h - h_hold, h + h_hold). h_hold = np.inf —
    зоны нет.
  * θ* (theta_trim) — балансировочный тангаж; при e = 0, I = 0 должно быть θᶜ = θ*.

ИССЛЕДОВАНИЕ (ответы — в отчёт)
-------------------------------
  1. EXPERIMENT = "base". Сравните h(t) с теорией. Перерегулирование в
     симуляции больше теоретического — почему? Что в модели ḣ ≈ Va·θ не учтено?
     (Вспомните график α и γ из L1.)
  2. EXPERIMENT = "W_h". Заполните таблицу для W_h = 2, 5, 10, 20.
     Почему при малом W_h процесс хуже, а при большом — медленнее?
     Какой W_h вы бы выбрали и почему?
  3. EXPERIMENT = "windup" (ступенька 50 м). Сравните три случая:
     без анти-виндапа и без зоны / без анти-виндапа с зоной / с анти-виндапом.
     Объясните, откуда берётся перерегулирование и как его убирает каждая мера.
  4. В L1 вы видели, что на медленных процессах θ → θᶜ (а не K_θDC·θᶜ).
     Как это влияет на выбор k_ph, k_ih? Почему в контуре высоты интегратор
     нужен, а в контуре тангажа — нет?

Запуск:  python scenarios/lab6/L2_altitude_hold.py
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
                         make_sensors, lti2_step, step_metrics, check, check_close,
                         check_summary, check_flight_safe, save_fig, PitchLoopReady)
from runner import run
from sim.config import SensorParams
from control.controllers import with_roll_hold   # САУ по крену (РЕШ-19)
from sim.state import THETA, H
from control.controllers import SpeedController, SpeedControlParams

# ------------------------------------------------------------------
# Настройки
# ------------------------------------------------------------------
VARIANT      = 1          # ваш номер варианта (1–6)
EXPERIMENT   = "base"     # "base" | "W_h" | "windup"
SENSOR_NOISE = True

H_HOLD    = 15.0               # м, зона удержания высоты (слайд 17)
THETA_MAX = np.radians(20.0)   # ограничение команды тангажа


# ==================================================================
# TODO 1. Коэффициенты ПИ-регулятора высоты (слайд 22)
# ==================================================================
def altitude_gains(wn_theta, W_h, zeta_h, K_dc, Va):
    """Вернуть (k_ph, k_ih)."""
    # >>> SOLUTION
    wn_h = wn_theta / W_h
    kp = 2.0 * zeta_h * wn_h / (K_dc * Va)
    ki = wn_h**2 / (K_dc * Va)
    return kp, ki
    # <<< SOLUTION


# ==================================================================
# TODO 2. Закон управления высотой
# ==================================================================
def altitude_control(h_c, h, mem, dt, kp, ki, theta_trim, h_hold, theta_max, anti_windup):
    """Вернуть θᶜ, рад. mem["I"] — состояние интегратора."""
    # >>> SOLUTION
    e = np.clip(h_c, h - h_hold, h + h_hold) - h
    mem["I"] += e * dt
    theta_c = theta_trim + kp * e + ki * mem["I"]
    theta_c_sat = np.clip(theta_c, -theta_max, theta_max)
    if anti_windup and theta_c_sat != theta_c:
        mem["I"] -= e * dt            # условное интегрирование
    return theta_c_sat
    # <<< SOLUTION


# ==================================================================
# Ниже править не нужно
# ==================================================================
T_STEP = 2.0
aircraft = AircraftParams()
v = get_variant(VARIANT)
tr = trim(aircraft, v["Va"])
print_trim(tr)
h0 = tr["state"][H]

pitch = PitchLoopReady(aircraft, tr, v["wn_theta"], v["zeta_theta"])
print(f"Контур тангажа (готовый): ω_nθ={v['wn_theta']}, ζ_θ={v['zeta_theta']}, K_θDC={pitch.K_dc:.3f}")


def simulate(gains, dh, t_end, h_hold=H_HOLD, anti_windup=True):
    meas = make_sensors(SENSOR_NOISE)
    speed = SpeedController(aircraft, SpeedControlParams())
    speed.set_Va_ref(v["Va"])
    speed.set_trim_throttle(tr["throttle"])
    mem = {"I": 0.0}
    theta_c_buf = []

    def controls_fn(t, state, Va, alpha):
        m = meas(state, Va)
        h_c = h0 + (dh if t >= T_STEP else 0.0)
        theta_c = None
        if gains is not None:
            theta_c = altitude_control(h_c, m["h"], mem, 0.01, *gains, tr["theta"],
                                       h_hold, THETA_MAX, anti_windup)
        if theta_c is None:
            theta_c = tr["theta"]
        theta_c_buf.append(theta_c)
        return np.array([pitch(theta_c, m["theta"], m["q"]), speed.step(m["Va"], 0.01)])

    log = run(with_roll_hold(controls_fn, aircraft, SensorParams(), 0.01), aircraft, WindParams(), SimConfig(dt=0.01, t_end=t_end), state0=tr["state"])
    return log, np.array(theta_c_buf[:len(log.t)])


def theory_h(t, gains, dh):
    kp, ki = gains
    b1, b0 = pitch.K_dc * v["Va"] * kp, pitch.K_dc * v["Va"] * ki
    return h0 + lti2_step(t, b1, b0, b1, b0, amp=dh, t_step=T_STEP)


def fmt(m):
    ts = "∞" if not np.isfinite(m["t_settle"]) else f"{m['t_settle']:.1f}"
    return f"перерег. {m['overshoot']:5.1f}%   t_уст(5%) {ts:>5} с   e_уст {m['e_ss']:+.2f} м"


gains = altitude_gains(v["wn_theta"], v["W_h"], v["zeta_h"], pitch.K_dc, v["Va"])
if gains is None:
    print("\n!!! TODO 1 не выполнен: θᶜ = θ* (контур высоты разомкнут).\n")
else:
    print(f"k_ph={gains[0]:.5f}  k_ih={gains[1]:.5f}   (ω_nh = {v['wn_theta'] / v['W_h']:.3f} рад/с)")

# ------------------------------------------------------------------
if EXPERIMENT == "base":
    T_END = 60.0
    log, theta_c = simulate(gains, v["dh"], T_END)
    t, h = log.t, log.state[:, H]

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(t, h0 + np.where(t >= T_STEP, v["dh"], 0.0), "k--", label="hᶜ")
    ax[0].plot(t, h, label="h (симуляция)")
    if gains is not None:
        ax[0].plot(t, theory_h(t, gains, v["dh"]), ":", lw=2, label="теория")
    ax[0].set_ylabel("h, м"); ax[0].legend(); ax[0].set_title(f"L2: ступенька по высоте, вариант {VARIANT}")
    ax[1].plot(t, np.degrees(theta_c), "--", label="θᶜ"); ax[1].plot(t, np.degrees(log.state[:, THETA]), label="θ")
    ax[1].plot(t, np.degrees(log.alpha), label="α"); ax[1].set_ylabel("град"); ax[1].legend()
    ax[2].plot(t, log.Va, label="Va"); ax[2].set_ylabel("м/с"); ax[2].legend()
    ax[3].plot(t, log.controls[:, 1], label="газ δt"); ax[3].set_xlabel("t, с"); ax[3].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L2_base_v{VARIANT}")

    print("\nСамопроверка L2:")
    if gains is None:
        check("TODO 1", False, "не реализовано")
    else:
        m = step_metrics(t, h, h0, h0 + v["dh"], T_STEP)
        mt = step_metrics(t, theory_h(t, gains, v["dh"]), h0, h0 + v["dh"], T_STEP)
        print(f"  симуляция: {fmt(m)}\n  теория:    {fmt(mt)}")
        wn_h = v["wn_theta"] / v["W_h"]
        KV = pitch.K_dc * v["Va"]
        check_close("K_θDC·Va·k_ph = 2ζ_h·ω_nh", KV * gains[0], 2 * v["zeta_h"] * wn_h)
        check_close("K_θDC·Va·k_ih = ω_nh²", KV * gains[1], wn_h**2)

        mem = {"I": 0.0}
        th0 = altitude_control(h0, h0, mem, 0.01, *gains, tr["theta"], H_HOLD, THETA_MAX, True)
        if th0 is None:
            check("TODO 2", False, "не реализовано")
        else:
            check_close("θᶜ = θ* при hᶜ = h", th0, tr["theta"], 1e-6)
            mem = {"I": 0.0}
            d = altitude_control(h0 + 0.1, h0, mem, 0.0, *gains, tr["theta"], H_HOLD, THETA_MAX, True) - tr["theta"]
            check_close("∂θᶜ/∂e = k_ph", d / 0.1, gains[0], 1e-3)
            mem = {"I": 0.0}
            d = altitude_control(h0 + 1000, h0, mem, 0.0, *gains, tr["theta"], 2.0, THETA_MAX, True) - tr["theta"]
            check_close("зона удержания ограничивает ошибку", d, gains[0] * 2.0, 1e-3)
            mem = {"I": 0.0}
            th_big = altitude_control(h0 + 1000, h0, mem, 0.0, *gains, tr["theta"], np.inf, THETA_MAX, True)
            check_close("ограничение θᶜ ≤ θ_max", th_big, THETA_MAX, 1e-6)

            check("установившаяся ошибка < 0.5 м", abs(m["e_ss"]) < 0.5, f"{m['e_ss']:+.2f} м")
            check("перерегулирование < 40%", m["overshoot"] < 40.0, f"{m['overshoot']:.1f}%")
            check_flight_safe(log, aircraft, T_END)
            log_w, _ = simulate(gains, 50.0, 60.0, anti_windup=True)
            mw = step_metrics(log_w.t, log_w.state[:, H], h0, h0 + 50.0, T_STEP)
            check("анти-виндап: ступенька 50 м без большого перерегулирования",
                  mw["overshoot"] < 15.0, f"перерег. {mw['overshoot']:.1f}%")
    check_summary()

# ------------------------------------------------------------------
elif EXPERIMENT == "W_h":
    if gains is None or altitude_control(h0, h0, {"I": 0.0}, 0.01, *gains, tr["theta"],
                                         H_HOLD, THETA_MAX, True) is None:
        sys.exit("Сначала выполните TODO 1–2.")
    T_END = 60.0
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    print(f"\n{'W_h':>4} {'ω_nh':>6}   метрики симуляции" + " " * 34 + "α_max")
    for W in (2.0, 5.0, 10.0, 20.0):
        g = altitude_gains(v["wn_theta"], W, v["zeta_h"], pitch.K_dc, v["Va"])
        log, theta_c = simulate(g, v["dh"], T_END)
        m = step_metrics(log.t, log.state[:, H], h0, h0 + v["dh"], T_STEP)
        print(f"{W:4.0f} {v['wn_theta'] / W:6.3f}   {fmt(m)}   {np.degrees(log.alpha.max()):5.1f}°")
        ax[0].plot(log.t, log.state[:, H], label=f"W_h = {W:.0f}")
        ax[1].plot(log.t, np.degrees(log.state[:, THETA]), label=f"W_h = {W:.0f}")
    ax[0].plot(log.t, h0 + np.where(log.t >= T_STEP, v["dh"], 0.0), "k--", lw=0.8)
    ax[0].set_ylabel("h, м"); ax[0].legend(); ax[0].set_title("L2: влияние разноса полос W_h")
    ax[1].set_ylabel("θ, град"); ax[1].set_xlabel("t, с"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L2_Wh_v{VARIANT}")

# ------------------------------------------------------------------
elif EXPERIMENT == "windup":
    if gains is None or altitude_control(h0, h0, {"I": 0.0}, 0.01, *gains, tr["theta"],
                                         H_HOLD, THETA_MAX, True) is None:
        sys.exit("Сначала выполните TODO 1–2.")
    T_END, DH = 90.0, 50.0
    cases = [("без анти-виндапа, без зоны", np.inf, False),
             ("без анти-виндапа, зона h_hold", H_HOLD, False),
             ("анти-виндап + зона h_hold", H_HOLD, True)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    print()
    for name, hh, aw in cases:
        log, theta_c = simulate(gains, DH, T_END, h_hold=hh, anti_windup=aw)
        m = step_metrics(log.t, log.state[:, H], h0, h0 + DH, T_STEP)
        print(f"  {name:32s} {fmt(m)}")
        ax[0].plot(log.t, log.state[:, H], label=name)
        ax[1].plot(log.t, np.degrees(theta_c), label=name)
    ax[0].plot(log.t, h0 + np.where(log.t >= T_STEP, DH, 0.0), "k--", lw=0.8)
    ax[0].set_ylabel("h, м"); ax[0].legend(); ax[0].set_title(f"L2: ступенька {DH:.0f} м — насыщение θᶜ")
    ax[1].set_ylabel("θᶜ, град"); ax[1].set_xlabel("t, с"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"L2_windup_v{VARIANT}")

plt.show()

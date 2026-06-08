# -*- coding: utf-8 -*-
"""
GameScenario — Ручное и автоматическое управление в реальном времени.

Режим РУЧНОЙ (по умолчанию):
  W / S  — руль высоты  ±0.3°  (W=нос вниз, S=нос вверх)
  X / Z  — тяга  ±5%

Режим АВТОПИЛОТ (клавиша A):
  W / S  — уставка тангажа θ_ref  ±1°  (диапазон −20…+20°)
  X / Z  — уставка скорости Va_ref ±1 м/с  (диапазон 25…50 м/с)
  Автопилот управляет рулём и тягой через ПИД-контуры.

Режим УДЕРЖАНИЕ ВЫСОТЫ (клавиша H, только при включённом АП):
  W / S  — уставка высоты h_ref  ±10 м
  X / Z  — уставка скорости Va_ref ±1 м/с  (как в АП)
  Высота → тангаж вычисляется автоматически (KH-закон).

Общие клавиши:
  A — переключить АП
  H — удержание высоты (только при АП)
  R — сброс к балансировочным условиям

Запуск:  python scenarios/GameScenario.py
"""

import sys
import os
import itertools
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.integrators import step_rk4
from sim.wind import wind as _wind
from sim.state import air_velocity, U, W, Q, THETA, H, X, N_STATES
from runner import compute_trim, trim_state
from control.controllers import (PitchController, PitchControlParams,
                                  SpeedController, SpeedControlParams)

plt.rcParams["keymap.save"] = ["ctrl+s"]
plt.rcParams["keymap.quit"] = ["ctrl+q"]
plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams(Vw_const=0.0)
cfg         = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=9999.0)

DE_STEP    = np.radians(0.3)
DE_MAX     = np.radians(30.0)
DE_MIN     = np.radians(-30.0)
THR_STEP   = 0.05

THETA_STEP = np.radians(1.0)
THETA_MAX  = np.radians(20.0)
THETA_MIN  = np.radians(-20.0)
VA_STEP    = 1.0
VA_MAX     = 50.0
VA_MIN     = 25.0

H_STEP = 10.0           # шаг высоты в режиме удержания, м
H_MAX  = 500.0
H_MIN  = 10.0
KH     = 0.006          # коэффициент h→θ_ref, рад/м (из ТЗ)

SIM_STEPS_PER_FRAME = 2       # dt=0.01, fps=50  →  1× реальное время
ANIM_FPS   = 50
WINDOW_SEC = 30.0
HISTORY    = 9000

# ------------------------------------------------------------------
# Балансировка
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

print(f"Балансировка:  alpha={np.degrees(alpha_trim):.2f}°  "
      f"de={np.degrees(de_trim):.2f}°  thr={thr_trim:.3f}")
print()
print("РУЧНОЙ режим:       W/S — δe ±0.3°      X/Z — тяга ±5%")
print("АВТОПИЛОТ (A):      W/S — θ_ref ±1°     X/Z — Va_ref ±1 м/с")
print("УДЕРЖание выс (H):  W/S — h_ref ±10 м   X/Z — Va_ref ±1 м/с")
print("R — сброс")
print()

ALPHA_WARN_DEG = np.degrees(aircraft.alpha_warning)
ALPHA_CRIT_DEG = np.degrees(aircraft.alpha_crit)

# ------------------------------------------------------------------
# ПИД-контроллеры (для режима автопилота)
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
ap_pitch    = PitchController(aircraft, ctrl_params)
ap_pitch.set_trim_throttle(thr_trim)
ap_pitch.reset({"theta": s0[THETA], "q": 0.0, "h": cfg.h0})

spd_params = SpeedControlParams()
ap_speed   = SpeedController(aircraft, spd_params)
ap_speed.set_trim_throttle(thr_trim)
ap_speed.set_Va_ref(cfg.Va0)
ap_speed.reset()

wind_call = lambda h, t: _wind(h, t, wind_params)

# ------------------------------------------------------------------
# Изменяемое состояние
# ------------------------------------------------------------------
sim = {
    "state":     s0.copy(),
    "delta_e":   de_trim,
    "throttle":  thr_trim,
    "t":         0.0,
    "crashed":   False,
    "autopilot": False,
    "theta_ref": alpha_trim,
    "Va_ref":    cfg.Va0,
    "h_hold":    False,
    "h_ref":     cfg.h0,
}

t_buf     = deque(maxlen=HISTORY)
h_buf     = deque(maxlen=HISTORY)
x_buf     = deque(maxlen=HISTORY)
Va_buf    = deque(maxlen=HISTORY)
al_buf    = deque(maxlen=HISTORY)
theta_buf = deque(maxlen=HISTORY)
de_buf    = deque(maxlen=HISTORY)
thr_buf   = deque(maxlen=HISTORY)

# ------------------------------------------------------------------
# Компоновка фигуры  (6 правых субплотов, как в С8)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 13))
title_text = fig.suptitle(
    "РУЧНОЙ  │  W/S — δe  │  X/Z — тяга  │  A — автопилот  │  R — сброс",
    fontsize=10, fontweight="bold",
)

gs = gridspec.GridSpec(6, 2, figure=fig,
                       width_ratios=[1.5, 1],
                       hspace=0.68, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_de   = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_Va   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[5, 1], sharex=ax_h)

right_axes = (ax_h, ax_th, ax_al, ax_de, ax_Va, ax_thr)

# ---- Траектория ----
ax_traj.set_xlabel("x, м", fontsize=9)
ax_traj.set_ylabel("h, м", fontsize=9)
ax_traj.set_title("Траектория", fontsize=9)
ax_traj.grid(True, ls="--", alpha=0.5)
ax_traj.set_xlim(-20.0, 200.0)
ax_traj.set_ylim(0.0, 250.0)
ax_traj.axhline(0.0, color="brown", lw=1.5, alpha=0.6, label="земля")
ax_traj.legend(fontsize=8, loc="lower left")

traj_line,   = ax_traj.plot([], [], "b-",  lw=1.4, alpha=0.55)
traj_marker, = ax_traj.plot([], [], "b^",  ms=13,  zorder=6, markeredgecolor="navy")

info_box = ax_traj.text(
    0.02, 0.97, "",
    transform=ax_traj.transAxes, fontsize=8.5,
    ha="left", va="top", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.90),
)

HINT_MANUAL = "W — δe ↓    S — δe ↑\nX — тяга ↑   Z — тяга ↓\nA — АП  R — сброс"
HINT_AUTO   = "W — θ ↓     S — θ ↑\nX — Va ↑    Z — Va ↓\nA — ручной  H — выс  R — сброс"
HINT_HHOLD  = "W — h ↓     S — h ↑\nX — Va ↑    Z — Va ↓\nA — ручной  H — θ-реж  R — сброс"

ctrl_hint = ax_traj.text(
    0.98, 0.97, HINT_MANUAL,
    transform=ax_traj.transAxes, fontsize=8,
    ha="right", va="top", family="monospace", color="dimgray",
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="lightyellow", edgecolor="goldenrod", alpha=0.85),
)

# ---- Правые субплоты ----
for ax in right_axes:
    ax.grid(True, ls="--", alpha=0.5)
    ax.tick_params(labelsize=8)
for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
right_axes[-1].set_xlabel("Время, с", fontsize=9)

ax_h.set_ylabel("h, м",         fontsize=9)
ax_th.set_ylabel("θ, °",        fontsize=9)
ax_al.set_ylabel("УА α, °",     fontsize=9)
ax_de.set_ylabel("δe, °",       fontsize=9)
ax_Va.set_ylabel("Va, м/с",     fontsize=9)
ax_thr.set_ylabel("Тяга, о.е.", fontsize=9)

# Ориентиры
ax_al.axhline(ALPHA_WARN_DEG, color="orange", lw=1.0, ls="--", alpha=0.8,
              label=f"warn {ALPHA_WARN_DEG:.0f}°")
ax_al.axhline(ALPHA_CRIT_DEG, color="red",    lw=1.0, ls="--", alpha=0.8,
              label=f"crit {ALPHA_CRIT_DEG:.0f}°")
ax_al.legend(fontsize=7, loc="upper right")

ax_th.axhline(0.0, color="gray", lw=0.8, ls=":")
ax_th.axhline( np.degrees(THETA_MAX), color="steelblue", lw=0.7, ls=":", alpha=0.5)
ax_th.axhline( np.degrees(THETA_MIN), color="steelblue", lw=0.7, ls=":", alpha=0.5)

ax_de.axhline(np.degrees(de_trim), color="black", lw=0.8, ls=":", alpha=0.5,
              label=f"trim {np.degrees(de_trim):.1f}°")
ax_de.legend(fontsize=7, loc="upper right")

ax_Va.axhline(cfg.Va0, color="black", lw=0.8, ls=":", alpha=0.5,
              label=f"init {cfg.Va0:.0f}")
ax_Va.legend(fontsize=7, loc="upper right")

ax_thr.axhline(thr_trim, color="black", lw=0.8, ls=":", alpha=0.5,
               label=f"trim {thr_trim:.2f}")
ax_thr.set_ylim(-0.03, 1.05)
ax_thr.legend(fontsize=7, loc="upper right")

# Пунктирные линии уставок автопилота
ln_h_ref,     = ax_h.plot([], [],  color="crimson",  lw=1.4, ls="--", alpha=0.0, label="h_ref")
ln_theta_ref, = ax_th.plot([], [], color="navy",     lw=1.4, ls="--", alpha=0.0, label="θ_ref")
ln_Va_ref,    = ax_Va.plot([], [], color="tomato",   lw=1.4, ls="--", alpha=0.0, label="Va_ref")
ax_h.legend(fontsize=7, loc="upper right")
ax_th.legend(fontsize=7, loc="upper right")
ax_Va.legend(fontsize=7, loc="upper right")

# Живые линии
ln_h,   = ax_h.plot([], [],   color="royalblue",   lw=1.8)
ln_th,  = ax_th.plot([], [],  color="seagreen",     lw=1.8)
ln_al,  = ax_al.plot([], [],  color="darkorange",   lw=1.8)
ln_de,  = ax_de.plot([], [],  color="saddlebrown",  lw=1.8)
ln_Va,  = ax_Va.plot([], [],  color="steelblue",    lw=1.8)
ln_thr, = ax_thr.plot([], [], color="darkgreen",    lw=1.8)

# ------------------------------------------------------------------
# Обработчик клавиш
# ------------------------------------------------------------------
def _reset():
    sim["state"]     = s0.copy()
    sim["delta_e"]   = de_trim
    sim["throttle"]  = thr_trim
    sim["t"]         = 0.0
    sim["crashed"]   = False
    sim["autopilot"] = False
    sim["theta_ref"] = alpha_trim
    sim["Va_ref"]    = cfg.Va0
    sim["h_hold"]    = False
    sim["h_ref"]     = cfg.h0
    ap_pitch.reset({"theta": s0[THETA], "q": 0.0, "h": cfg.h0})
    ap_pitch.set_pitch_setpoint(alpha_trim)
    ap_speed.set_Va_ref(cfg.Va0)
    ap_speed.reset()
    for buf in (t_buf, h_buf, x_buf, Va_buf, al_buf, theta_buf, de_buf, thr_buf):
        buf.clear()


def on_key(event):
    if event.key is None:
        return
    key = event.key.lower()

    if key == "r":
        _reset()
        return

    if key == "a":
        sim["autopilot"] = not sim["autopilot"]
        sim["h_hold"]    = False          # сброс подрежима при выходе из АП
        if sim["autopilot"]:
            state = sim["state"]
            w_vec = wind_call(state[H], sim["t"])
            Va, _ = air_velocity(state, w_vec)
            sim["theta_ref"] = np.clip(state[THETA], THETA_MIN, THETA_MAX)
            sim["Va_ref"]    = np.clip(Va, VA_MIN, VA_MAX)
            ap_pitch.reset({"theta": state[THETA], "q": state[Q], "h": state[H]})
            ap_pitch.set_pitch_setpoint(sim["theta_ref"])
            ap_speed.set_Va_ref(sim["Va_ref"])
            ap_speed.reset()
        return

    if key == "h" and sim["autopilot"]:
        sim["h_hold"] = not sim["h_hold"]
        if sim["h_hold"]:
            # Захват текущей высоты как уставки
            sim["h_ref"] = sim["state"][H]
            # theta_ref сразу вычислим через KH (нулевая ошибка → alpha_trim)
            sim["theta_ref"] = alpha_trim
            ap_pitch.reset({"theta": sim["state"][THETA],
                             "q":     sim["state"][Q],
                             "h":     sim["state"][H]})
            ap_pitch.set_pitch_setpoint(sim["theta_ref"])
        return

    if sim["autopilot"]:
        if sim["h_hold"]:
            # W/S управляют высотой; X/Z — скоростью
            if key == "s":
                sim["h_ref"] = np.clip(sim["h_ref"] + H_STEP, H_MIN, H_MAX)
            elif key == "w":
                sim["h_ref"] = np.clip(sim["h_ref"] - H_STEP, H_MIN, H_MAX)
            elif key == "x":
                sim["Va_ref"] = np.clip(sim["Va_ref"] + VA_STEP, VA_MIN, VA_MAX)
                ap_speed.set_Va_ref(sim["Va_ref"])
            elif key == "z":
                sim["Va_ref"] = np.clip(sim["Va_ref"] - VA_STEP, VA_MIN, VA_MAX)
                ap_speed.set_Va_ref(sim["Va_ref"])
        else:
            if key == "s":
                sim["theta_ref"] = np.clip(sim["theta_ref"] + THETA_STEP, THETA_MIN, THETA_MAX)
                ap_pitch.set_pitch_setpoint(sim["theta_ref"])
            elif key == "w":
                sim["theta_ref"] = np.clip(sim["theta_ref"] - THETA_STEP, THETA_MIN, THETA_MAX)
                ap_pitch.set_pitch_setpoint(sim["theta_ref"])
            elif key == "x":
                sim["Va_ref"] = np.clip(sim["Va_ref"] + VA_STEP, VA_MIN, VA_MAX)
                ap_speed.set_Va_ref(sim["Va_ref"])
            elif key == "z":
                sim["Va_ref"] = np.clip(sim["Va_ref"] - VA_STEP, VA_MIN, VA_MAX)
                ap_speed.set_Va_ref(sim["Va_ref"])
    else:
        if key == "w":
            sim["delta_e"] = np.clip(sim["delta_e"] + DE_STEP, DE_MIN, DE_MAX)
        elif key == "s":
            sim["delta_e"] = np.clip(sim["delta_e"] - DE_STEP, DE_MIN, DE_MAX)
        elif key == "x":
            sim["throttle"] = min(1.0, sim["throttle"] + THR_STEP)
        elif key == "z":
            sim["throttle"] = max(0.0, sim["throttle"] - THR_STEP)


fig.canvas.mpl_connect("key_press_event", on_key)

# ------------------------------------------------------------------
# Анимация
# ------------------------------------------------------------------
_artists = (traj_line, traj_marker, info_box, ctrl_hint,
            ln_h, ln_th, ln_al, ln_de, ln_Va, ln_thr,
            ln_h_ref, ln_theta_ref, ln_Va_ref)


def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_al, ln_de, ln_Va, ln_thr,
               ln_h_ref, ln_theta_ref, ln_Va_ref):
        ln.set_data([], [])
    return _artists


def update(_frame):
    ap = sim["autopilot"]

    if not sim["crashed"]:
        for _ in range(SIM_STEPS_PER_FRAME):
            state = sim["state"]
            t     = sim["t"]
            h     = state[H]

            if h < 0.1:
                sim["crashed"] = True
                info_box.set_text("!! СТОЛКНОВЕНИЕ С ЗЕМЛЁЙ !!\n\nНажмите R для сброса")
                break

            w_vec     = wind_call(h, t)
            Va, alpha = air_velocity(state, w_vec)

            if ap:
                if sim["h_hold"]:
                    # KH-закон: высота → тангаж
                    h_err = sim["h_ref"] - h
                    theta_ref_calc = np.clip(
                        alpha_trim + KH * h_err, THETA_MIN, THETA_MAX
                    )
                    sim["theta_ref"] = theta_ref_calc
                    ap_pitch.set_pitch_setpoint(theta_ref_calc)
                meas = {"q": state[Q], "theta": state[THETA], "h": h, "Va": Va}
                ctrl_out     = ap_pitch.step(t, meas, cfg.dt)
                delta_e      = ctrl_out[0]
                throttle     = ap_speed.step(Va, cfg.dt)
                sim["delta_e"]  = delta_e
                sim["throttle"] = throttle
            else:
                delta_e  = sim["delta_e"]
                throttle = sim["throttle"]

            controls = np.array([delta_e, throttle])

            t_buf.append(t)
            h_buf.append(h)
            x_buf.append(state[X])
            Va_buf.append(Va)
            al_buf.append(np.degrees(alpha))
            theta_buf.append(np.degrees(state[THETA]))
            de_buf.append(np.degrees(delta_e))
            thr_buf.append(throttle)

            sim["state"] = step_rk4(state, controls, cfg.dt, t, aircraft, wind_call)
            sim["t"]     = t + cfg.dt

    if len(t_buf) < 2:
        return _artists

    t_arr     = np.array(t_buf)
    h_arr     = np.array(h_buf)
    x_arr     = np.array(x_buf)
    Va_arr    = np.array(Va_buf)
    al_arr    = np.array(al_buf)
    theta_arr = np.array(theta_buf)
    de_arr    = np.array(de_buf)
    thr_arr   = np.array(thr_buf)

    t_cur = t_arr[-1]
    t_lo  = max(0.0, t_cur - WINDOW_SEC)
    mask  = t_arr >= t_lo
    ts_w  = t_arr[mask]

    # ---- Траектория ----
    traj_line.set_data(x_arr, h_arr)
    traj_marker.set_data([x_arr[-1]], [h_arr[-1]])
    px = max((x_arr.max() - x_arr.min()) * 0.08, 30.0)
    ph = max((h_arr.max() - h_arr.min()) * 0.25, 40.0)
    ax_traj.set_xlim(x_arr.min() - px, x_arr.max() + px)
    ax_traj.set_ylim(max(0.0, h_arr.min() - ph), h_arr.max() + ph)

    # ---- Info box ----
    alpha_cur = al_arr[-1]
    if not sim["crashed"]:
        flag = ("  !!!! СРЫВ !!!!" if alpha_cur > ALPHA_CRIT_DEG
                else "  ! ВНИМАНИЕ"  if alpha_cur > ALPHA_WARN_DEG
                else "")
        if ap:
            if sim["h_hold"]:
                mode_label = "[АП + ВЫСОТА]"
                h_line = f"h      = {h_arr[-1]:6.1f} м  →{sim['h_ref']:.0f}\n"
                extra  = f"θ_ref  = {np.degrees(sim['theta_ref']):+5.1f}° (авто)\n"
            else:
                mode_label = "[АВТОПИЛОТ]"
                h_line = f"h      = {h_arr[-1]:6.1f} м\n"
                extra  = f"θ_ref  = {np.degrees(sim['theta_ref']):+5.1f}°\n"
            info_box.set_text(
                f"{mode_label}\n"
                f"t      = {t_cur:6.1f} с\n"
                + h_line +
                f"Va     = {Va_arr[-1]:5.1f} м/с\n"
                f"Va_ref = {sim['Va_ref']:5.1f} м/с\n"
                + extra +
                f"α      = {alpha_cur:+5.2f}°{flag}"
            )
        else:
            info_box.set_text(
                f"[РУЧНОЙ]\n"
                f"t      = {t_cur:6.1f} с\n"
                f"h      = {h_arr[-1]:6.1f} м\n"
                f"Va     = {Va_arr[-1]:5.1f} м/с\n"
                f"δe     = {de_arr[-1]:+5.1f}°\n"
                f"тяга   = {thr_arr[-1]:.3f}\n"
                f"α      = {alpha_cur:+5.2f}°{flag}"
            )

    # ---- Подсказка и заголовок ----
    if ap and sim["h_hold"]:
        ctrl_hint.set_text(HINT_HHOLD)
        ctrl_hint.get_bbox_patch().set(facecolor="lightcyan", edgecolor="steelblue")
        title_text.set_text(
            "АП + ВЫСОТА  │  W/S — h_ref  │  X/Z — Va_ref  │  H — θ-режим  │  A — ручной  │  R — сброс"
        )
    elif ap:
        ctrl_hint.set_text(HINT_AUTO)
        ctrl_hint.get_bbox_patch().set(facecolor="lightgreen", edgecolor="green")
        title_text.set_text(
            "АВТОПИЛОТ  │  W/S — θ_ref  │  X/Z — Va_ref  │  H — выс  │  A — ручной  │  R — сброс"
        )
    else:
        ctrl_hint.set_text(HINT_MANUAL)
        ctrl_hint.get_bbox_patch().set(facecolor="lightyellow", edgecolor="goldenrod")
        title_text.set_text(
            "РУЧНОЙ  │  W/S — δe  │  X/Z — тяга  │  A — автопилот  │  R — сброс"
        )

    # ---- Правые субплоты ----
    ln_h.set_data(ts_w, h_arr[mask])
    ln_th.set_data(ts_w, theta_arr[mask])
    ln_al.set_data(ts_w, al_arr[mask])
    ln_de.set_data(ts_w, de_arr[mask])
    ln_Va.set_data(ts_w, Va_arr[mask])
    ln_thr.set_data(ts_w, thr_arr[mask])

    # Линии уставок
    if len(ts_w) > 0:
        t0, t1 = ts_w[0], ts_w[-1]
        # h_ref — только в h_hold
        if ap and sim["h_hold"]:
            ln_h_ref.set_data([t0, t1], [sim["h_ref"]] * 2)
            ln_h_ref.set_alpha(0.8)
        else:
            ln_h_ref.set_alpha(0.0)
        # θ_ref — только в АП без h_hold
        if ap and not sim["h_hold"]:
            ln_theta_ref.set_data([t0, t1], [np.degrees(sim["theta_ref"])] * 2)
            ln_theta_ref.set_alpha(0.8)
        else:
            ln_theta_ref.set_alpha(0.0)
        # Va_ref — в любом режиме АП
        if ap:
            ln_Va_ref.set_data([t0, t1], [sim["Va_ref"]] * 2)
            ln_Va_ref.set_alpha(0.8)
        else:
            ln_Va_ref.set_alpha(0.0)

    # Авторасширение осей Y
    for ax, arr in zip(
        (ax_h, ax_th, ax_al, ax_de, ax_Va),
        (h_arr[mask], theta_arr[mask], al_arr[mask], de_arr[mask], Va_arr[mask]),
    ):
        if len(arr) < 2:
            continue
        lo, hi = arr.min(), arr.max()
        pad = max((hi - lo) * 0.20, 1.5)
        ax.set_ylim(lo - pad, hi + pad)

    for ax in right_axes:
        ax.set_xlim(t_lo, t_lo + WINDOW_SEC)

    return _artists


anim = FuncAnimation(
    fig, update,
    frames=itertools.count(),
    init_func=init,
    interval=1000 // ANIM_FPS,
    blit=False,
    cache_frame_data=False,
)

plt.subplots_adjust(top=0.94, left=0.07, right=0.97, bottom=0.06)
plt.show()

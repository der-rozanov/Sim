# -*- coding: utf-8 -*-
"""
С5: Контроль высоты — набор и снижение по уставке.

Структура управления:
  h_ref → [P_h] → theta_ref → [каскадный ПИД theta+q] → delta_e
  throttle = trim (фиксированный)

Сценарий:
   0 .. T_CLIMB   с → h_ref = H_TRIM  (горизонтальный трим)
  T_CLIMB .. T_DESCEND с → h_ref = H_HIGH  (набор высоты)
  T_DESCEND .. конец   с → h_ref = H_TRIM  (снижение обратно)

Запуск:  python scenarios/s5_altitude_control.py
         python scenarios/s5_altitude_control.py results/s5.gif
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import PitchController, PitchControlParams
from control.sensors import measure_gyro, measure_altitude, measure_airspeed
from sim.state import THETA, Q, H, X
from flight_logger import FlightLogger

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg         = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=60.0)

H_TRIM    = 100.0   # начальная высота, м
H_HIGH    = 200.0   # уставка набора, м
T_CLIMB   = 10.0    # с, команда набора
T_DESCEND = 40.0    # с, команда снижения

# P-коэффициент внешнего контура: ошибка высоты → поправка тангажа
# 50 м ошибки → theta_ref = alpha_trim + 50*KH (ограничено ±15°)
KH = 0.006          # рад/м

ANIM_SPEED = 2.0
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

logger = FlightLogger(
    scenario="С5: Контроль высоты",
    description=f"Набор h={H_HIGH:.0f} м, Va={cfg.Va0:.0f} м/с",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_CLIMB,   "label": "набор",    "color": "orange"},
        {"t": T_DESCEND, "label": "снижение", "color": "dodgerblue"},
    ],
)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")

# ------------------------------------------------------------------
# Внутренний каскадный ПИД тангажа
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': H_TRIM})

rng = np.random.default_rng(seed=42)
h_ref_buf     = []
theta_ref_buf = []

# ------------------------------------------------------------------
# Функция управления
# ------------------------------------------------------------------
def controls_fn(t, state, Va, alpha):
    # Уставка высоты по расписанию
    if t < T_CLIMB:
        h_ref = H_TRIM
    elif t < T_DESCEND:
        h_ref = H_HIGH
    else:
        h_ref = H_TRIM

    # Внешний контур: P по высоте → theta_ref
    h_meas    = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    h_err     = h_ref - h_meas
    theta_ref = np.clip(alpha_trim + KH * h_err,
                        np.radians(-15.0), np.radians(15.0))

    controller.set_pitch_setpoint(theta_ref)
    h_ref_buf.append(h_ref)
    theta_ref_buf.append(theta_ref)

    # Внутренний контур: каскадный ПИД
    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    return controller.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С5  Контроль высоты")
logger.save(log, h_ref=h_ref_buf, theta_ref=theta_ref_buf)

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n          = len(log.t)
t_all      = log.t
t_end      = t_all[-1]
h_all      = log.state[:, H]
x_all      = log.state[:, X]
h_ref_all  = np.array(h_ref_buf[:n])
theta_all  = np.degrees(log.state[:, THETA])
theta_ref  = np.degrees(np.array(theta_ref_buf[:n]))
q_all      = np.degrees(log.state[:, Q])
de_all     = np.degrees(log.controls[:, 0])
thr_all    = log.controls[:, 1]
Va_all     = log.Va
alpha_all  = np.degrees(log.alpha)

# ------------------------------------------------------------------
# Прореживание кадров
# ------------------------------------------------------------------
dt_sim   = t_all[1] - t_all[0]
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 13))
fig.suptitle(
    f"С5: Контроль высоты — трим → набор {H_HIGH:.0f} м → возврат {H_TRIM:.0f} м  (анимация)\n"
    f"(трим 0–{T_CLIMB:.0f} с,  набор {T_CLIMB:.0f}–{T_DESCEND:.0f} с,"
    f"  снижение {T_DESCEND:.0f}–{cfg.t_end:.0f} с)",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(6, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.68, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_q    = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_de   = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_Va   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[5, 1], sharex=ax_h)

# ------------------------------------------------------------------
# Левый subplot: траектория
# ------------------------------------------------------------------
_px = max((x_all.max() - x_all.min()) * 0.06, 10.0)
_ph = max((h_all.max() - h_all.min()) * 0.30, 20.0)

ax_traj.set_xlim(x_all.min() - _px, x_all.max() + _px)
ax_traj.set_ylim(h_all.min() - _ph, h_all.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.set_xlabel("Горизонтальная дальность x, м", fontsize=9)
ax_traj.set_ylabel("Высота h, м", fontsize=9)
ax_traj.set_title("Траектория в вертикальной плоскости", fontsize=9)
ax_traj.grid(True, linestyle="--", alpha=0.5)

# Уставки высоты — горизонтальные ориентиры
ax_traj.axhline(H_TRIM, color="steelblue", lw=1.1, ls="--", alpha=0.6,
                label=f"h={H_TRIM:.0f} м")
ax_traj.axhline(H_HIGH, color="orange",    lw=1.1, ls="--", alpha=0.6,
                label=f"h={H_HIGH:.0f} м")
ax_traj.legend(fontsize=8, loc="upper left")
ax_traj.plot(x_all, h_all, color="lightsteelblue", lw=1.2, alpha=0.4, zorder=1)
ax_traj.plot(x_all[0],  h_all[0],  "go", ms=8, zorder=5, label="старт")
ax_traj.plot(x_all[-1], h_all[-1], "rs", ms=8, zorder=5, label="финиш")

traj_line,   = ax_traj.plot([], [], "b-", lw=2.0, zorder=3)
traj_marker, = ax_traj.plot([], [], "b^", ms=11,  zorder=6, markeredgecolor="navy")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes,
    fontsize=8.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="gray", alpha=0.85),
    family="monospace"
)

# ------------------------------------------------------------------
# Общие настройки правых субплотов
# ------------------------------------------------------------------
right_axes = (ax_h, ax_th, ax_q, ax_de, ax_Va, ax_thr)
step_events = [
    (T_CLIMB,   "orange",     "набор"),
    (T_DESCEND, "dodgerblue", "снижение"),
]

for ax in right_axes:
    ax.set_xlim(0.0, t_end)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    for t_ev, col, lbl in step_events:
        ax.axvline(t_ev, color=col, lw=0.9, ls=":", alpha=0.7,
                   label=f"{lbl} t={t_ev:.0f}с" if ax is ax_h else None)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_thr.set_xlabel("Время, с", fontsize=9)

# ---- Высота ----
ax_h.set_ylabel("Высота h, м", fontsize=9)
_h_pad = max((h_all.max() - h_all.min()) * 0.25, 8.0)
ax_h.set_ylim(h_all.min() - _h_pad, h_all.max() + _h_pad)
ax_h.plot(t_all, h_ref_all, color="black", lw=1.1, ls="--", alpha=0.55, label="h_ref")
ax_h.plot(t_all, h_all, color="lightsteelblue", lw=1.0, alpha=0.4)
ax_h.legend(fontsize=7, loc="upper right")

ln_h, = ax_h.plot([], [], color="royalblue", lw=1.8)
pt_h, = ax_h.plot([], [], "o", color="royalblue", ms=5, zorder=5)

# ---- Тангаж ----
ax_th.set_ylabel("Угол тангажа θ, °", fontsize=9)
_th_lo = min(theta_all.min(), theta_ref.min()) - 2.0
_th_hi = max(theta_all.max(), theta_ref.max()) + 2.0
ax_th.set_ylim(_th_lo, _th_hi)
ax_th.axhline(0, color="gray", lw=0.8, ls=":")
ax_th.plot(t_all, theta_all, color="lightseagreen", lw=1.0, alpha=0.4)
ax_th.plot(t_all, theta_ref, color="black",         lw=1.1, ls="--", alpha=0.55,
           label="θ_ref")
ax_th.legend(fontsize=7, loc="upper right")

ln_th, = ax_th.plot([], [], color="seagreen", lw=1.8)
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- Угловая скорость ----
ax_q.set_ylabel("Угловая скорость q, °/с", fontsize=9)
_q_abs = max(abs(q_all).max() * 1.3, 2.0)
ax_q.set_ylim(-_q_abs, _q_abs)
ax_q.axhline(0, color="gray", lw=0.8, ls=":")
ax_q.plot(t_all, q_all, color="thistle", lw=1.0, alpha=0.45)

ln_q, = ax_q.plot([], [], color="mediumpurple", lw=1.8)
pt_q, = ax_q.plot([], [], "o", color="mediumpurple", ms=5, zorder=5)

# ---- Руль высоты ----
ax_de.set_ylabel("Руль высоты δe, °", fontsize=9)
_de_pad = max(abs(de_all).max() * 0.18, 1.5)
ax_de.set_ylim(de_all.min() - _de_pad, de_all.max() + _de_pad)
ax_de.axhline(0, color="gray", lw=0.8, ls=":")
ax_de.plot(t_all, de_all, color="burlywood", lw=1.0, alpha=0.5)

ln_de, = ax_de.plot([], [], color="saddlebrown", lw=1.8)
pt_de, = ax_de.plot([], [], "o", color="saddlebrown", ms=5, zorder=5)

# ---- Воздушная скорость ----
ax_Va.set_ylabel("Воздушная скорость Va, м/с", fontsize=9)
_Va_pad = max((Va_all.max() - Va_all.min()) * 0.25, 0.5)
ax_Va.set_ylim(Va_all.min() - _Va_pad, Va_all.max() + _Va_pad)
ax_Va.plot(t_all, Va_all, color="lightsteelblue", lw=1.0, alpha=0.45)

ln_Va, = ax_Va.plot([], [], color="steelblue", lw=1.8)
pt_Va, = ax_Va.plot([], [], "o", color="steelblue", ms=5, zorder=5)

# ---- Газ (тяга) ----
ax_thr.set_ylabel("Газ (тяга), о.е.", fontsize=9)
ax_thr.set_ylim(-0.05, 1.05)
ax_thr.plot(t_all, thr_all, color="lightgreen", lw=1.0, alpha=0.5)

ln_thr, = ax_thr.plot([], [], color="darkgreen", lw=1.8)
pt_thr, = ax_thr.plot([], [], "o", color="darkgreen", ms=5, zorder=5)

# Вертикальный курсор времени
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65)
          for ax in right_axes]

# ------------------------------------------------------------------
# init / update
# ------------------------------------------------------------------
_all_artists = (
    traj_line, traj_marker, info_box,
    ln_h,   pt_h,
    ln_th,  pt_th,
    ln_q,   pt_q,
    ln_de,  pt_de,
    ln_Va,  pt_Va,
    ln_thr, pt_thr,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_q, ln_de, ln_Va, ln_thr):
        ln.set_data([], [])
    for pt in (pt_h, pt_th, pt_q, pt_de, pt_Va, pt_thr):
        pt.set_data([], [])
    for vl in vlines:
        vl.set_xdata([0])
    return _all_artists

def update(fn):
    i     = idx_list[fn]
    t_cur = t_all[i]
    ts    = t_all[:i+1]

    traj_line.set_data(x_all[:i+1], h_all[:i+1])
    traj_marker.set_data([x_all[i]], [h_all[i]])

    info_box.set_text(
        f"t   = {t_cur:5.1f} с\n"
        f"h   = {h_all[i]:6.1f} м  ref={h_ref_all[i]:.0f} м\n"
        f"Va  = {Va_all[i]:5.1f} м/с\n"
        f"θ   = {theta_all[i]:+5.2f}°  ref={theta_ref[i]:+.1f}°\n"
        f"α   = {alpha_all[i]:+5.2f}°\n"
        f"δe  = {de_all[i]:+5.1f}°"
    )

    ln_h.set_data(ts, h_all[:i+1])
    pt_h.set_data([t_cur], [h_all[i]])
    ln_th.set_data(ts, theta_all[:i+1])
    pt_th.set_data([t_cur], [theta_all[i]])
    ln_q.set_data(ts, q_all[:i+1])
    pt_q.set_data([t_cur], [q_all[i]])
    ln_de.set_data(ts, de_all[:i+1])
    pt_de.set_data([t_cur], [de_all[i]])
    ln_Va.set_data(ts, Va_all[:i+1])
    pt_Va.set_data([t_cur], [Va_all[i]])
    ln_thr.set_data(ts, thr_all[:i+1])
    pt_thr.set_data([t_cur], [thr_all[i]])

    for vl in vlines:
        vl.set_xdata([t_cur])

    return _all_artists

anim = FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    interval=1000.0 / ANIM_FPS,
    blit=True,
)

# ------------------------------------------------------------------
# Сохранение / показ
# ------------------------------------------------------------------
save_path = sys.argv[1] if len(sys.argv) > 1 else None

if save_path is not None:
    if save_path.endswith(".gif"):
        try:
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=ANIM_FPS))
            print(f"Сохранено: {save_path}")
        except Exception as e:
            print(f"Не удалось сохранить GIF (нужен Pillow): {e}")
    else:
        try:
            anim.save(save_path, fps=ANIM_FPS, extra_args=["-vcodec", "libx264"])
            print(f"Сохранено: {save_path}")
        except Exception as e:
            print(f"Не удалось сохранить MP4 (нужен ffmpeg): {e}")

plt.show()

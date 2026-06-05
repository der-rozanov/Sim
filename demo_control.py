# -*- coding: utf-8 -*-
"""
demo_control.py -- анимированная демонстрация каскадного ПИД-контура тангажа.

Сценарий: два скачка уставки тангажа:
  0..20 с  → theta_ref = +5°
  20..40 с → theta_ref = -5°

Запуск:  python demo_control.py
         python demo_control.py out.gif    -- сохранить анимацию
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control import PitchController, PitchControlParams
from sensors import measure_gyro, measure_altitude, measure_airspeed
from state import THETA, Q, H, X

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg         = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=40.0)

THETA_POS = np.radians(5.0)
THETA_NEG = np.radians(2.0)
T_SWITCH  = 20.0   # с

ANIM_SPEED = 5.0   # множитель скорости воспроизведения
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")

# ------------------------------------------------------------------
# Контроллер
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': cfg.h0})

rng = np.random.default_rng(seed=7)
theta_ref_buf = []

def controls_fn(t, state, Va, alpha):
    theta_ref = THETA_POS if t < T_SWITCH else THETA_NEG
    controller.set_pitch_setpoint(theta_ref)
    theta_ref_buf.append(theta_ref)

    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    return controller.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="Pitch control demo  (+5° / -5°)")

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n         = len(log.t)
t_all     = log.t
t_end     = t_all[-1]
theta_all = np.degrees(log.state[:, THETA])
theta_ref = np.degrees(np.array(theta_ref_buf[:n]))
q_all     = np.degrees(log.state[:, Q])
de_all    = np.degrees(log.controls[:, 0])
thr_all   = log.controls[:, 1]
h_all     = log.state[:, H]
x_all     = log.state[:, X]
Va_all    = log.Va

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
fig = plt.figure(figsize=(14, 9))
fig.suptitle("Контур тангажа: скачок уставки  +5° → −5°  (анимация)",
             fontsize=12, fontweight="bold")

gs = gridspec.GridSpec(4, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.58, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])       # траектория — вся левая колонка
ax_th   = fig.add_subplot(gs[0, 1])       # тангаж
ax_q    = fig.add_subplot(gs[1, 1], sharex=ax_th)   # угловая скорость
ax_de   = fig.add_subplot(gs[2, 1], sharex=ax_th)   # руль высоты (управление)
ax_thr  = fig.add_subplot(gs[3, 1], sharex=ax_th)   # тяга (управление)

# ------------------------------------------------------------------
# Левый subplot: траектория
# ------------------------------------------------------------------
_px = max((x_all.max() - x_all.min()) * 0.06, 10.0)
_ph = max((h_all.max() - h_all.min()) * 0.18, 15.0)

ax_traj.set_xlim(x_all.min() - _px, x_all.max() + _px)
ax_traj.set_ylim(h_all.min() - _ph, h_all.max() + _ph)
ax_traj.set_xlabel("Горизонтальная дальность x, м", fontsize=9)
ax_traj.set_ylabel("Высота h, м", fontsize=9)
ax_traj.set_title("Траектория в вертикальной плоскости", fontsize=9)
ax_traj.grid(True, linestyle="--", alpha=0.5)

ax_traj.plot(x_all, h_all, color="lightsteelblue", lw=1.2, alpha=0.4, zorder=1)
ax_traj.plot(x_all[0],  h_all[0],  "go", ms=8, zorder=5, label="старт")
ax_traj.plot(x_all[-1], h_all[-1], "rs", ms=8, zorder=5, label="финиш")
ax_traj.legend(fontsize=8, loc="upper left")

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
right_axes = (ax_th, ax_q, ax_de, ax_thr)
for ax in right_axes:
    ax.set_xlim(0.0, t_end)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.axvline(T_SWITCH, color="red", lw=0.9, ls=":", alpha=0.7,
               label=f"t={T_SWITCH:.0f}с" if ax is ax_th else None)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_thr.set_xlabel("Время, с", fontsize=9)

# ---- Тангаж ----
ax_th.set_ylabel("Угол тангажа θ, °", fontsize=9)
_th_lo = min(theta_all.min(), theta_ref.min()) - 2.0
_th_hi = max(theta_all.max(), theta_ref.max()) + 2.0
ax_th.set_ylim(_th_lo, _th_hi)
ax_th.axhline(0, color="gray", lw=0.8, ls=":")
ax_th.plot(t_all, theta_all, color="lightseagreen", lw=1.0, alpha=0.4)
ax_th.plot(t_all, theta_ref, color="black", lw=1.1, ls="--", alpha=0.55,
           label="θ_ref")
ax_th.legend(fontsize=7, loc="upper right")

ln_th, = ax_th.plot([], [], color="seagreen",    lw=1.8, label="θ")
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- Угловая скорость ----
ax_q.set_ylabel("Угловая скорость q, °/с", fontsize=9)
_q_abs = max(abs(q_all).max() * 1.3, 2.0)
ax_q.set_ylim(-_q_abs, _q_abs)
ax_q.axhline(0, color="gray", lw=0.8, ls=":")
ax_q.plot(t_all, q_all, color="thistle", lw=1.0, alpha=0.45)

ln_q, = ax_q.plot([], [], color="mediumpurple", lw=1.8)
pt_q, = ax_q.plot([], [], "o", color="mediumpurple", ms=5, zorder=5)

# ---- Руль высоты (управление) ----
ax_de.set_ylabel("Руль высоты δe, °", fontsize=9)
_de_pad = max(abs(de_all).max() * 0.18, 1.5)
ax_de.set_ylim(de_all.min() - _de_pad, de_all.max() + _de_pad)
ax_de.axhline(0, color="gray", lw=0.8, ls=":")
ax_de.plot(t_all, de_all, color="burlywood", lw=1.0, alpha=0.5)

ln_de, = ax_de.plot([], [], color="saddlebrown", lw=1.8)
pt_de, = ax_de.plot([], [], "o", color="saddlebrown", ms=5, zorder=5)

# ---- Тяга (управление) ----
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
    ln_th, pt_th,
    ln_q,  pt_q,
    ln_de, pt_de,
    ln_thr, pt_thr,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_th, ln_q, ln_de, ln_thr):
        ln.set_data([], [])
    for pt in (pt_th, pt_q, pt_de, pt_thr):
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
        f"Va  = {Va_all[i]:5.1f} м/с\n"
        f"θ   = {theta_all[i]:+5.2f}°  ref={theta_ref[i]:+.1f}°\n"
        f"q   = {q_all[i]:+5.2f} °/с\n"
        f"δe  = {de_all[i]:+5.1f}°\n"
        f"газ = {thr_all[i]:.3f}"
    )

    ln_th.set_data(ts, theta_all[:i+1])
    pt_th.set_data([t_cur], [theta_all[i]])
    ln_q.set_data(ts, q_all[:i+1])
    pt_q.set_data([t_cur], [q_all[i]])
    ln_de.set_data(ts, de_all[:i+1])
    pt_de.set_data([t_cur], [de_all[i]])
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

plt.tight_layout()
plt.show()

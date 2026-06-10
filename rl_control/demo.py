# -*- coding: utf-8 -*-
"""
RL-демо: Q-learning агент стабилизирует тангаж.

Требует:  rl_control/q_table.npy  (сначала запустить train.py)
Запуск:   python rl_control/demo.py
          python rl_control/demo.py results/rl_demo.gif   -- сохранить
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config      import AircraftParams, WindParams, SimConfig
from sim.state       import THETA, Q, H, X
from runner          import run, compute_trim, trim_state
from rl_control.agent import QLearningAgent

SAVE_DIR = os.path.dirname(__file__)

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры сценария
# ------------------------------------------------------------------
ANIM_SPEED = 4.0
ANIM_FPS   = 25

# Ступенчатые изменения уставки тангажа
_SCHEDULE = [
    (0.0,  0.0),
    (5.0,  8.0),
    (20.0, -5.0),
    (35.0,  0.0),
]

def theta_ref_schedule(t: float) -> float:
    """Уставка тангажа в радианах (ступенчатая)."""
    deg = _SCHEDULE[0][1]
    for t_start, val in _SCHEDULE:
        if t >= t_start:
            deg = val
    return np.radians(deg)

# ------------------------------------------------------------------
# Загрузка агента и балансировка
# ------------------------------------------------------------------
qt_path = os.path.join(SAVE_DIR, "q_table.npy")
if not os.path.exists(qt_path):
    print("q_table.npy не найдена. Сначала запустите:  python rl_control/train.py")
    sys.exit(1)

aircraft = AircraftParams()
wind     = WindParams()
cfg      = SimConfig(Va0=30.0, h0=100.0, dt=0.01, t_end=50.0)

alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

de_trim_path = os.path.join(SAVE_DIR, "de_trim.npy")
de_trim_saved = float(np.load(de_trim_path)[0]) if os.path.exists(de_trim_path) else de_trim
agent = QLearningAgent(de_trim=de_trim_saved)
agent.load(qt_path)

print(f"Трим:  alpha={np.degrees(alpha_trim):.2f}  "
      f"delta_e={np.degrees(de_trim):.2f}  throttle={thr_trim:.3f}")

# ------------------------------------------------------------------
# controls_fn
# ------------------------------------------------------------------
theta_ref_log = []

def controls_fn(t, state, _Va, _alpha):
    ref   = theta_ref_schedule(t)
    de    = agent.get_delta_e(ref - state[THETA], state[Q])
    theta_ref_log.append(ref)
    return np.array([de, thr_trim])

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind, cfg, state0=s0)
print(f"Завершено: {log.t[-1]:.1f} с   h_final={log.state[-1, H]:.1f} м")

# ------------------------------------------------------------------
# Массивы для анимации
# ------------------------------------------------------------------
n          = len(log.t)
t_arr      = log.t
t_end_act  = t_arr[-1]
h_arr      = log.state[:, H]
x_arr      = log.state[:, X]
theta_arr  = np.degrees(log.state[:, THETA])
theta_ref  = np.degrees(np.array(theta_ref_log[:n]))
q_arr      = np.degrees(log.state[:, Q])
alpha_arr  = np.degrees(log.alpha)
de_arr     = np.degrees(log.controls[:, 0])
Va_arr     = log.Va

warn_deg  = np.degrees(aircraft.alpha_warning)
crit_deg  = np.degrees(aircraft.alpha_crit)
stall_deg = np.degrees(aircraft.alpha_stall)

# Прореживание кадров
dt_sim   = t_arr[1] - t_arr[0]
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 11))
fig.suptitle(
    "Q-Learning: стабилизация тангажа  |  "
    "уставки: 0deg -> 8deg -> -5deg -> 0deg",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(5, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.65, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_q    = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_de   = fig.add_subplot(gs[4, 1], sharex=ax_h)

# ------------------------------------------------------------------
# Левый subplot: траектория
# ------------------------------------------------------------------
_px = max((x_arr.max() - x_arr.min()) * 0.06, 10.0)
_ph = max((h_arr.max() - h_arr.min()) * 0.30, 20.0)

ax_traj.set_xlim(x_arr.min() - _px, x_arr.max() + _px)
ax_traj.set_ylim(h_arr.min() - _ph, h_arr.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.set_xlabel("Горизонтальная дальность x, м", fontsize=9)
ax_traj.set_ylabel("Высота h, м", fontsize=9)
ax_traj.set_title("Траектория в вертикальной плоскости", fontsize=9)
ax_traj.grid(True, linestyle="--", alpha=0.5)

ax_traj.plot(x_arr, h_arr, color="lightsteelblue", lw=1.2, alpha=0.4, zorder=1)
ax_traj.plot(x_arr[0],  h_arr[0],  "go", ms=8, zorder=5, label="старт")
ax_traj.plot(x_arr[-1], h_arr[-1], "rs", ms=8, zorder=5, label="финиш")
ax_traj.legend(fontsize=8, loc="upper left")

traj_line,   = ax_traj.plot([], [], "b-",  lw=2.0, zorder=3)
traj_marker, = ax_traj.plot([], [], "b^",  ms=11,  zorder=6,
                             markeredgecolor="navy")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes,
    fontsize=8.5, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="gray", alpha=0.85),
    family="monospace"
)

# ------------------------------------------------------------------
# Правые субплоты: общие настройки
# ------------------------------------------------------------------
right_axes = (ax_h, ax_th, ax_al, ax_q, ax_de)

_step_times  = [t for t, _ in _SCHEDULE[1:]]  # моменты смены уставки
_step_colors = ["steelblue", "darkorange", "seagreen"]

for ax in right_axes:
    ax.set_xlim(0.0, t_end_act)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    for t_ev, col in zip(_step_times, _step_colors):
        ax.axvline(t_ev, color=col, lw=1.0, ls=":", alpha=0.75)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_de.set_xlabel("Время, с", fontsize=9)

# ---- Высота ----
ax_h.set_ylabel("Высота h, м", fontsize=9)
_h_pad = max((h_arr.max() - h_arr.min()) * 0.25, 8.0)
ax_h.set_ylim(h_arr.min() - _h_pad, h_arr.max() + _h_pad)
ax_h.axhline(cfg.h0, color="gray", lw=0.9, ls="--", alpha=0.5,
             label=f"h0={cfg.h0:.0f} м")
ax_h.plot(t_arr, h_arr, color="lightsteelblue", lw=1.0, alpha=0.4)
ax_h.legend(fontsize=7, loc="upper left")

ln_h, = ax_h.plot([], [], color="royalblue", lw=1.8)
pt_h, = ax_h.plot([], [], "o", color="royalblue", ms=5, zorder=5)

# ---- Тангаж ----
ax_th.set_ylabel("Тангаж theta, °", fontsize=9)
_th_lo = min(theta_arr.min(), theta_ref.min()) - 2.0
_th_hi = max(theta_arr.max(), theta_ref.max()) + 2.0
ax_th.set_ylim(_th_lo, _th_hi)
ax_th.axhline(0, color="gray", lw=0.8, ls=":")
ax_th.plot(t_arr, theta_arr, color="lightseagreen", lw=1.0, alpha=0.4)
ax_th.step(t_arr, theta_ref, color="black", lw=1.2, ls="--", alpha=0.6,
           where="post", label="theta_ref")
ax_th.legend(fontsize=7, loc="upper right")

ln_th, = ax_th.plot([], [], color="seagreen", lw=1.8)
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- Угол атаки ----
ax_al.set_ylabel("УА alpha, °", fontsize=9)
_al_lo = min(alpha_arr.min() - 2.0, -1.0)
_al_hi = max(alpha_arr.max() + 3.0, warn_deg + 5.0)
ax_al.set_ylim(_al_lo, _al_hi)

ax_al.axhspan(warn_deg, crit_deg,  color="orange", alpha=0.12, zorder=0)
ax_al.axhspan(crit_deg, _al_hi+10, color="red",    alpha=0.08, zorder=0)
ax_al.axhline(warn_deg,  color="orange",  lw=1.1, ls="--", alpha=0.85,
              label=f"пред. {warn_deg:.0f}°")
ax_al.axhline(crit_deg,  color="red",     lw=1.1, ls="--", alpha=0.85,
              label=f"крит. {crit_deg:.0f}°")
ax_al.axhline(0, color="gray", lw=0.7, ls=":")
ax_al.plot(t_arr, alpha_arr, color="lightsalmon", lw=1.0, alpha=0.45)
ax_al.legend(fontsize=7, loc="upper right")

ln_al, = ax_al.plot([], [], color="crimson", lw=1.8)
pt_al, = ax_al.plot([], [], "o", color="crimson", ms=5, zorder=5)

# ---- Угловая скорость ----
ax_q.set_ylabel("q, °/с", fontsize=9)
_q_abs = max(abs(q_arr).max() * 1.3, 2.0)
ax_q.set_ylim(-_q_abs, _q_abs)
ax_q.axhline(0, color="gray", lw=0.8, ls=":")
ax_q.plot(t_arr, q_arr, color="thistle", lw=1.0, alpha=0.45)

ln_q, = ax_q.plot([], [], color="mediumpurple", lw=1.8)
pt_q, = ax_q.plot([], [], "o", color="mediumpurple", ms=5, zorder=5)

# ---- Руль высоты (ступенчатый — видна дискретность RL) ----
ax_de.set_ylabel("delta_e, °", fontsize=9)
_de_pad = max(abs(de_arr).max() * 0.18, 1.5)
ax_de.set_ylim(de_arr.min() - _de_pad, de_arr.max() + _de_pad)
ax_de.axhline(0, color="gray", lw=0.8, ls=":")
ax_de.axhline(np.degrees(de_trim), color="gray", lw=0.8, ls="--", alpha=0.5,
              label=f"de_trim={np.degrees(de_trim):.1f}°")
ax_de.step(t_arr, de_arr, color="burlywood", lw=1.0, alpha=0.5, where="post")
ax_de.legend(fontsize=7)

ln_de, = ax_de.step([], [], color="saddlebrown", lw=1.8, where="post")
pt_de, = ax_de.plot([], [], "o", color="saddlebrown", ms=5, zorder=5)

# Вертикальный курсор
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65)
          for ax in right_axes]

# ------------------------------------------------------------------
# Анимация
# ------------------------------------------------------------------
_all_artists = (
    traj_line, traj_marker, info_box,
    ln_h,  pt_h,
    ln_th, pt_th,
    ln_al, pt_al,
    ln_q,  pt_q,
    ln_de, pt_de,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_al, ln_q, ln_de):
        ln.set_data([], [])
    for pt in (pt_h, pt_th, pt_al, pt_q, pt_de):
        pt.set_data([], [])
    for vl in vlines:
        vl.set_xdata([0])
    return _all_artists

def update(fn):
    i     = idx_list[fn]
    t_cur = t_arr[i]
    ts    = t_arr[:i+1]

    traj_line.set_data(x_arr[:i+1], h_arr[:i+1])
    traj_marker.set_data([x_arr[i]], [h_arr[i]])

    a_cur = alpha_arr[i]
    if a_cur >= stall_deg:
        warn_str = "  !! СРЫВ !!"
    elif a_cur >= crit_deg:
        warn_str = "  ! критич. !"
    elif a_cur >= warn_deg:
        warn_str = "  предупрежд."
    else:
        warn_str = ""

    info_box.set_text(
        f"t    = {t_cur:5.1f} с\n"
        f"h    = {h_arr[i]:6.1f} м\n"
        f"Va   = {Va_arr[i]:5.1f} м/с\n"
        f"theta= {theta_arr[i]:+5.2f}  ref={theta_ref[i]:+.1f}\n"
        f"alpha= {a_cur:+5.2f}{warn_str}\n"
        f"q    = {q_arr[i]:+5.2f} /с\n"
        f"de   = {de_arr[i]:+5.1f}"
    )

    ln_h.set_data(ts,  h_arr[:i+1])
    pt_h.set_data([t_cur], [h_arr[i]])
    ln_th.set_data(ts, theta_arr[:i+1])
    pt_th.set_data([t_cur], [theta_arr[i]])
    ln_al.set_data(ts, alpha_arr[:i+1])
    pt_al.set_data([t_cur], [alpha_arr[i]])
    ln_q.set_data(ts,  q_arr[:i+1])
    pt_q.set_data([t_cur], [q_arr[i]])
    ln_de.set_data(ts, de_arr[:i+1])
    pt_de.set_data([t_cur], [de_arr[i]])

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
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
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

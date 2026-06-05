# -*- coding: utf-8 -*-
"""
С2: Кабрирование — выход на установившийся тангаж.

Сценарий:
  0 .. T_TRIM с   → горизонтальный полёт (трим)
  T_TRIM .. конец → уставка тангажа THETA_CLIMB (5°)

Тяга фиксирована на тримовом значении. Контроллер тангажа —
каскадный ПИД (theta → q → delta_e).

На графике выделены:
  - Переходный процесс по theta (уставка vs факт)
  - Угол атаки alpha с зонами предупреждения / критического УА
  - Высота (ЛА набирает высоту при положительном тангаже)

Запуск:  python scenarios/s2_pitch_up.py
         python scenarios/s2_pitch_up.py results/s2.gif   -- сохранить анимацию
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
cfg = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=40.0)

T_TRIM       = 10.0             # с, длительность тримового участка
T_RETURN     = 30.0             # с, момент возврата на тримовый тангаж
THETA_CLIMB  = np.radians(5.0) # рад, уставка кабрирования

ANIM_SPEED = 4.0
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

print(f"Трим:  alpha={np.degrees(alpha_trim):.2f}°  "
      f"delta_e={np.degrees(de_trim):.2f}°  throttle={thr_trim:.3f}")
print(f"Уставка кабрирования: theta_ref = {np.degrees(THETA_CLIMB):.1f}°")
print(f"Возврат на трим:      t = {T_RETURN:.0f} с")

# ------------------------------------------------------------------
# Контроллер
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.reset({"theta": s0[THETA], "q": 0.0, "h": cfg.h0})

rng            = np.random.default_rng(seed=42)
theta_ref_buf  = []

def controls_fn(t, state, Va, alpha):
    # Три фазы: трим → кабрирование → возврат на трим
    if t < T_TRIM:
        theta_ref = alpha_trim
    elif t < T_RETURN:
        theta_ref = THETA_CLIMB
    else:
        theta_ref = alpha_trim
    controller.set_pitch_setpoint(theta_ref)
    theta_ref_buf.append(theta_ref)

    q_meas     = measure_gyro(state[Q],     sp.gyro_bias,    sp.gyro_noise,    rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    h_meas     = measure_altitude(state[H], sp.baro_bias,    sp.baro_noise,    rng)
    Va_meas    = measure_airspeed(Va,       sp.airspeed_bias, sp.airspeed_noise, rng)

    meas = {"q": q_meas, "theta": theta_meas, "h": h_meas, "Va": Va_meas}
    return controller.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С2  Кабрирование 5°")

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n          = len(log.t)
t_arr      = log.t
t_end_act  = t_arr[-1]
h_arr      = log.state[:, H]
x_arr      = log.state[:, X]
theta_arr  = np.degrees(log.state[:, THETA])
theta_ref  = np.degrees(np.array(theta_ref_buf[:n]))
q_arr      = np.degrees(log.state[:, Q])
alpha_arr  = np.degrees(log.alpha)
de_arr     = np.degrees(log.controls[:, 0])
thr_arr    = log.controls[:, 1]
Va_arr     = log.Va

warn_deg  = np.degrees(aircraft.alpha_warning)
crit_deg  = np.degrees(aircraft.alpha_crit)
stall_deg = np.degrees(aircraft.alpha_stall)

# Числовые итоги переходного процесса
# t_reach: первый момент, когда theta впервые входит в ±0.5° от уставки
# (фугоидная составляющая не мешает определению быстрого переходного процесса)
theta_target = np.degrees(THETA_CLIMB)
band_reach   = 0.5    # °, полоса первого достижения

t_reach = None
for i in range(n):
    if t_arr[i] < T_TRIM:
        continue
    if abs(theta_arr[i] - theta_target) <= band_reach:
        t_reach = t_arr[i]
        break

t_reach_str = f"{t_reach - T_TRIM:.2f} с" if t_reach else "—"
alpha_max_after = np.max(alpha_arr[t_arr >= T_TRIM])
dh_total = h_arr[-1] - h_arr[0]

print(f"\n  Переходный процесс:")
print(f"    Время первого достижения ±0.5°: {t_reach_str} от момента команды")
print(f"    Макс. alpha после команды: {alpha_max_after:.2f}°")
print(f"    Набор высоты за прогон:   {dh_total:.1f} м")

# ------------------------------------------------------------------
# Прореживание кадров анимации
# ------------------------------------------------------------------
dt_sim   = t_arr[1] - t_arr[0]
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12))
fig.suptitle(
    f"С2: Кабрирование — трим → θ_ref = {np.degrees(THETA_CLIMB):.0f}° → возврат на трим\n"
    f"(трим 0–{T_TRIM:.0f} с,  кабрирование {T_TRIM:.0f}–{T_RETURN:.0f} с,"
    f"  возврат {T_RETURN:.0f}–{cfg.t_end:.0f} с)",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(6, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.68, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_q    = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_de   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[5, 1], sharex=ax_h)

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

traj_line,   = ax_traj.plot([], [], "b-", lw=2.0, zorder=3)
traj_marker, = ax_traj.plot([], [], "b^", ms=11,  zorder=6,
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
# Общие настройки правых субплотов
# ------------------------------------------------------------------
right_axes = (ax_h, ax_th, ax_al, ax_q, ax_de, ax_thr)
_events = [
    (T_TRIM,   "steelblue", f"кабрирование t={T_TRIM:.0f}с"),
    (T_RETURN, "darkorange", f"возврат t={T_RETURN:.0f}с"),
]
for ax in right_axes:
    ax.set_xlim(0.0, t_end_act)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    for t_ev, col, lbl in _events:
        ax.axvline(t_ev, color=col, lw=1.0, ls=":", alpha=0.75,
                   label=lbl if ax is ax_th else None)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_thr.set_xlabel("Время, с", fontsize=9)

# ---- Высота ----
ax_h.set_ylabel("Высота h, м", fontsize=9)
_h_pad = max((h_arr.max() - h_arr.min()) * 0.25, 8.0)
ax_h.set_ylim(h_arr.min() - _h_pad, h_arr.max() + _h_pad)
ax_h.axhline(cfg.h0, color="gray", lw=0.9, ls="--", alpha=0.5,
             label=f"h₀={cfg.h0:.0f} м")
ax_h.plot(t_arr, h_arr, color="lightsteelblue", lw=1.0, alpha=0.4)
ax_h.legend(fontsize=7, loc="upper left")

ln_h, = ax_h.plot([], [], color="royalblue", lw=1.8)
pt_h, = ax_h.plot([], [], "o", color="royalblue", ms=5, zorder=5)

# ---- Тангаж ----
ax_th.set_ylabel("Угол тангажа θ, °", fontsize=9)
_th_lo = min(theta_arr.min(), theta_ref.min()) - 2.0
_th_hi = max(theta_arr.max(), theta_ref.max()) + 2.0
ax_th.set_ylim(_th_lo, _th_hi)
ax_th.axhline(0, color="gray", lw=0.8, ls=":")
ax_th.plot(t_arr, theta_arr, color="lightseagreen", lw=1.0, alpha=0.4)
ax_th.plot(t_arr, theta_ref, color="black", lw=1.1, ls="--", alpha=0.55,
           label="θ_ref")
ax_th.legend(fontsize=7, loc="upper right")

ln_th, = ax_th.plot([], [], color="seagreen", lw=1.8)
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- Угол атаки (ключевой!) ----
ax_al.set_ylabel("Угол атаки α, °", fontsize=9)
_al_lo = min(alpha_arr.min() - 2.0, -1.0)
_al_hi = max(alpha_arr.max() + 3.0, warn_deg + 5.0)
ax_al.set_ylim(_al_lo, _al_hi)

# Цветные зоны опасности
ax_al.axhspan(warn_deg, crit_deg,  color="orange", alpha=0.12, zorder=0)
ax_al.axhspan(crit_deg, _al_hi+10, color="red",    alpha=0.08, zorder=0)

ax_al.axhline(warn_deg,  color="orange",  lw=1.1, ls="--", alpha=0.85,
              label=f"пред. {warn_deg:.0f}°")
ax_al.axhline(crit_deg,  color="red",     lw=1.1, ls="--", alpha=0.85,
              label=f"крит. {crit_deg:.0f}°")
ax_al.axhline(stall_deg, color="darkred", lw=1.2, ls="-",  alpha=0.7,
              label=f"срыв {stall_deg:.0f}°")
ax_al.axhline(0, color="gray", lw=0.7, ls=":")
ax_al.plot(t_arr, alpha_arr, color="lightsalmon", lw=1.0, alpha=0.45)
ax_al.legend(fontsize=7, loc="upper right")

ln_al, = ax_al.plot([], [], color="crimson", lw=1.8)
pt_al, = ax_al.plot([], [], "o", color="crimson", ms=5, zorder=5)

# ---- Угловая скорость ----
ax_q.set_ylabel("Угловая скорость q, °/с", fontsize=9)
_q_abs = max(abs(q_arr).max() * 1.3, 2.0)
ax_q.set_ylim(-_q_abs, _q_abs)
ax_q.axhline(0, color="gray", lw=0.8, ls=":")
ax_q.plot(t_arr, q_arr, color="thistle", lw=1.0, alpha=0.45)

ln_q, = ax_q.plot([], [], color="mediumpurple", lw=1.8)
pt_q, = ax_q.plot([], [], "o", color="mediumpurple", ms=5, zorder=5)

# ---- Руль высоты ----
ax_de.set_ylabel("Руль высоты δe, °", fontsize=9)
_de_pad = max(abs(de_arr).max() * 0.18, 1.5)
ax_de.set_ylim(de_arr.min() - _de_pad, de_arr.max() + _de_pad)
ax_de.axhline(0, color="gray", lw=0.8, ls=":")
ax_de.axhline(np.degrees(de_trim), color="gray", lw=0.8, ls="--", alpha=0.5,
              label=f"δe_трим={np.degrees(de_trim):.1f}°")
ax_de.plot(t_arr, de_arr, color="burlywood", lw=1.0, alpha=0.5)
ax_de.legend(fontsize=7)

ln_de, = ax_de.plot([], [], color="saddlebrown", lw=1.8)
pt_de, = ax_de.plot([], [], "o", color="saddlebrown", ms=5, zorder=5)

# ---- Газ ----
ax_thr.set_ylabel("Газ (тяга), о.е.", fontsize=9)
ax_thr.set_ylim(-0.05, 1.05)
ax_thr.axhline(thr_trim, color="gray", lw=0.8, ls="--", alpha=0.5,
               label=f"газ_трим={thr_trim:.3f}")
ax_thr.plot(t_arr, thr_arr, color="lightgreen", lw=1.0, alpha=0.5)
ax_thr.legend(fontsize=7)

ln_thr, = ax_thr.plot([], [], color="darkgreen", lw=1.8)
pt_thr, = ax_thr.plot([], [], "o", color="darkgreen", ms=5, zorder=5)

# Вертикальный курсор времени
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65)
          for ax in right_axes]

# ------------------------------------------------------------------
# init / update анимации
# ------------------------------------------------------------------
_all_artists = (
    traj_line, traj_marker, info_box,
    ln_h,  pt_h,
    ln_th, pt_th,
    ln_al, pt_al,
    ln_q,  pt_q,
    ln_de, pt_de,
    ln_thr, pt_thr,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_al, ln_q, ln_de, ln_thr):
        ln.set_data([], [])
    for pt in (pt_h, pt_th, pt_al, pt_q, pt_de, pt_thr):
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

    # Метка опасности по alpha
    a_cur = alpha_arr[i]
    if a_cur >= stall_deg:
        warn_str = "  !! СРЫВ !!"
    elif a_cur >= crit_deg:
        warn_str = "  ! критич. !"
    elif a_cur >= warn_deg:
        warn_str = "  предупрежд."
    else:
        warn_str = ""

    # Фаза
    if t_cur < T_TRIM:
        phase_str = "трим"
    elif t_cur < T_RETURN:
        phase_str = "кабрирование"
    else:
        phase_str = "возврат"

    info_box.set_text(
        f"t   = {t_cur:5.1f} с  [{phase_str}]\n"
        f"h   = {h_arr[i]:6.1f} м\n"
        f"Va  = {Va_arr[i]:5.1f} м/с\n"
        f"θ   = {theta_arr[i]:+5.2f}°  ref={theta_ref[i]:+.1f}°\n"
        f"α   = {a_cur:+5.2f}°{warn_str}\n"
        f"q   = {q_arr[i]:+5.2f} °/с\n"
        f"δe  = {de_arr[i]:+5.1f}°"
    )

    ln_h.set_data(ts, h_arr[:i+1])
    pt_h.set_data([t_cur], [h_arr[i]])
    ln_th.set_data(ts, theta_arr[:i+1])
    pt_th.set_data([t_cur], [theta_arr[i]])
    ln_al.set_data(ts, alpha_arr[:i+1])
    pt_al.set_data([t_cur], [alpha_arr[i]])
    ln_q.set_data(ts, q_arr[:i+1])
    pt_q.set_data([t_cur], [q_arr[i]])
    ln_de.set_data(ts, de_arr[:i+1])
    pt_de.set_data([t_cur], [de_arr[i]])
    ln_thr.set_data(ts, thr_arr[:i+1])
    pt_thr.set_data([t_cur], [thr_arr[i]])

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

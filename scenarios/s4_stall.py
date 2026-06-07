# -*- coding: utf-8 -*-
"""
С4: Управляемый срыв и вывод — индикация закритического УА.

Сценарий:
  0 .. T_SWITCH  с → тримовый полёт
  T_SWITCH .. T_RECOVER с → тангаж 20°, газ=0 → alpha растёт → срыв
  T_RECOVER .. конец  → нос вниз −5°, газ=трим → вывод из срыва

Ключевой элемент: сигнал индикатора УА (0=НОРМ / 1=ПРЕД / 2=КРИТ / 3=СРЫВ)
вычисляется как функция alpha и отображается отдельным субплотом.
Фон info_box меняет цвет вместе с уровнем тревоги.

Запуск:  python scenarios/s4_stall.py
         python scenarios/s4_stall.py results/s4.gif
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
cfg         = SimConfig(Va0=30.0, h0=300.0, theta0=0.0, dt=0.01, t_end=55.0)

THETA_POS     = np.radians(1.9)    # тримовый полёт (фаза 1)
THETA_NEG     = np.radians(20.0)   # тангаж + газ=0 → срыв (фаза 2)
THETA_RECOVER = np.radians(-5.0)   # нос вниз + газ → вывод из срыва (фаза 3)
T_SWITCH      = 8.0                # с, фаза 1 → 2
T_RECOVER     = 25.0               # с, фаза 2 → 3

ANIM_SPEED = 5.0   # множитель скорости воспроизведения
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

logger = FlightLogger(
    scenario="С4: Управляемый срыв",
    description="Трим → тангаж 20°/газ=0 → вывод −5°",
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_SWITCH,  "label": "срыв",  "color": "red"},
        {"t": T_RECOVER, "label": "вывод", "color": "dodgerblue"},
    ],
)

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
    if t < T_SWITCH:
        # Фаза 1: тримовый полёт
        controller.set_trim_throttle(thr_trim)
        theta_ref = THETA_POS
    elif t < T_RECOVER:
        # Фаза 2: газ в ноль + агрессивный тангаж → срыв
        # Пропеллер тормозит → Va падает → alpha растёт до срыва
        controller.set_trim_throttle(0.0)
        theta_ref = THETA_NEG
    else:
        # Фаза 3: нос вниз + газ → вывод из срыва
        # Va растёт при снижении → CL восстанавливается → выход из срыва
        controller.set_trim_throttle(thr_trim)
        theta_ref = THETA_RECOVER

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
logger.save(log, theta_ref=theta_ref_buf)

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n         = len(log.t)
t_all     = log.t
t_end     = t_all[-1]
theta_all = np.degrees(log.state[:, THETA])
theta_ref = np.degrees(np.array(theta_ref_buf[:n]))
q_all     = np.degrees(log.state[:, Q])
alpha_all = np.degrees(log.alpha)
de_all    = np.degrees(log.controls[:, 0])
thr_all   = log.controls[:, 1]
h_all     = log.state[:, H]
x_all     = log.state[:, X]
Va_all    = log.Va

# ------------------------------------------------------------------
# Индикатор УА: 0=НОРМ, 1=ПРЕДУПРЕЖДЕНИЕ, 2=КРИТИЧЕСКИЙ, 3=СРЫВ
# ------------------------------------------------------------------
_warn_deg  = np.degrees(aircraft.alpha_warning)
_crit_deg  = np.degrees(aircraft.alpha_crit)
_stall_deg = np.degrees(aircraft.alpha_stall)

def _alert_level(a_deg):
    if a_deg >= _stall_deg:
        return 3
    if a_deg >= _crit_deg:
        return 2
    if a_deg >= _warn_deg:
        return 1
    return 0

alert_all = np.array([_alert_level(a) for a in alpha_all])

# Числовые итоги
t_above_warn  = np.sum(alert_all >= 1) * cfg.dt
t_above_crit  = np.sum(alert_all >= 2) * cfg.dt
t_above_stall = np.sum(alert_all >= 3) * cfg.dt
alpha_max     = alpha_all.max()

print(f"\n  Индикатор УА:")
print(f"    alpha_max            = {alpha_max:.2f} deg")
print(f"    t >= ПРЕД ({_warn_deg:.0f} deg) = {t_above_warn:.2f} s")
print(f"    t >= КРИТ ({_crit_deg:.0f} deg) = {t_above_crit:.2f} s")
print(f"    t >= СРЫВ ({_stall_deg:.0f} deg) = {t_above_stall:.2f} s")

# Цвета фона info_box по уровню тревоги
_ALERT_BG = ["#e8f5e9", "#fff9c4", "#ffe0b2", "#ffcdd2"]  # зел/жёлт/оранж/красн

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
    "С4: Управляемый срыв — трим → тангаж 20°/газ=0 → вывод −5°\n"
    "Индикатор УА: НОРМ / ПРЕД / КРИТ / СРЫВ  (анимация)",
    fontsize=12, fontweight="bold"
)

gs = gridspec.GridSpec(6, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.65, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])                          # траектория — вся левая колонка
ax_th   = fig.add_subplot(gs[0, 1])                          # тангаж
ax_q    = fig.add_subplot(gs[1, 1], sharex=ax_th)            # угловая скорость
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_th)            # угол атаки
ax_de   = fig.add_subplot(gs[3, 1], sharex=ax_th)            # руль высоты
ax_thr  = fig.add_subplot(gs[4, 1], sharex=ax_th)            # тяга
ax_ind  = fig.add_subplot(gs[5, 1], sharex=ax_th)            # индикатор УА

# ------------------------------------------------------------------
# Левый subplot: траектория
# ------------------------------------------------------------------
_px = max((x_all.max() - x_all.min()) * 0.06, 10.0)
_ph = max((h_all.max() - h_all.min()) * 0.18, 15.0)

ax_traj.set_xlim(x_all.min() - _px, x_all.max() + _px)
ax_traj.set_ylim(h_all.min() - _ph, h_all.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")
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
right_axes = (ax_th, ax_q, ax_al, ax_de, ax_thr, ax_ind)
for ax in right_axes:
    ax.set_xlim(0.0, t_end)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.axvline(T_SWITCH,  color="red",       lw=0.9, ls=":", alpha=0.7,
               label=f"срыв t={T_SWITCH:.0f}с"   if ax is ax_th else None)
    ax.axvline(T_RECOVER, color="dodgerblue", lw=0.9, ls=":", alpha=0.7,
               label=f"вывод t={T_RECOVER:.0f}с" if ax is ax_th else None)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_ind.set_xlabel("Время, с", fontsize=9)

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

# ---- Угол атаки ----
ax_al.set_ylabel("Угол атаки α, °", fontsize=9)
_al_lo = min(alpha_all.min() - 2.0, -3.0)
_al_hi = max(alpha_all.max() + 3.0, _stall_deg + 5.0)
ax_al.set_ylim(_al_lo, _al_hi)
ax_al.axhline(0, color="gray", lw=0.8, ls=":")
# Цветные зоны опасности (статичные фоны)
ax_al.axhspan(_warn_deg, _crit_deg,  color="orange", alpha=0.12, zorder=0)
ax_al.axhspan(_crit_deg, _al_hi + 5, color="red",    alpha=0.10, zorder=0)
# Пороговые линии
ax_al.axhline(_warn_deg,  color="orange",    lw=1.1, ls="--", alpha=0.9,
              label=f"пред. {_warn_deg:.0f}°")
ax_al.axhline(_crit_deg,  color="red",       lw=1.1, ls="--", alpha=0.9,
              label=f"крит. {_crit_deg:.0f}°")
ax_al.axhline(_stall_deg, color="darkred",   lw=1.3, ls="-",  alpha=0.7,
              label=f"срыв {_stall_deg:.0f}°")
ax_al.legend(fontsize=7, loc="upper left")
ax_al.plot(t_all, alpha_all, color="lightsalmon", lw=1.0, alpha=0.45)

ln_al, = ax_al.plot([], [], color="crimson", lw=1.8)
pt_al, = ax_al.plot([], [], "o", color="crimson", ms=5, zorder=5)

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

# ---- Индикатор УА ----
ax_ind.set_ylabel("Индикатор УА", fontsize=9)
ax_ind.set_ylim(-0.5, 3.5)
ax_ind.set_yticks([0, 1, 2, 3])
ax_ind.set_yticklabels(["НОРМ", "ПРЕД", "КРИТ", "СРЫВ"], fontsize=8)
# Статичные цветные зоны
ax_ind.axhspan(-0.5, 0.5, color="green",  alpha=0.12, zorder=0)
ax_ind.axhspan( 0.5, 1.5, color="yellow", alpha=0.18, zorder=0)
ax_ind.axhspan( 1.5, 2.5, color="orange", alpha=0.18, zorder=0)
ax_ind.axhspan( 2.5, 3.5, color="red",    alpha=0.15, zorder=0)
# Фоновая "тень" полного прогона
ax_ind.step(t_all, alert_all, color="lightgray", lw=1.0, alpha=0.4,
            where="post", zorder=1)

ln_ind, = ax_ind.step([], [], color="black", lw=2.0, where="post", zorder=3)
pt_ind, = ax_ind.plot([], [], "o", ms=7, zorder=5)

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
    ln_al, pt_al,
    ln_de, pt_de,
    ln_thr, pt_thr,
    ln_ind, pt_ind,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_th, ln_q, ln_al, ln_de, ln_thr, ln_ind):
        ln.set_data([], [])
    for pt in (pt_th, pt_q, pt_al, pt_de, pt_thr, pt_ind):
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

    a_cur   = alpha_all[i]
    lv_cur  = alert_all[i]

    _warn_labels = ["", "  предупрежд.", "  ! критич. !", "  !! СРЫВ !!"]
    _ind_colors  = ["green", "goldenrod", "darkorange", "crimson"]
    warn_str = _warn_labels[lv_cur]

    # Фон info_box меняется по уровню тревоги
    info_box.get_bbox_patch().set_facecolor(_ALERT_BG[lv_cur])

    _ind_labels = ["НОРМ", "ПРЕД", "КРИТ", "СРЫВ"]
    info_box.set_text(
        f"t   = {t_cur:5.1f} с\n"
        f"Va  = {Va_all[i]:5.1f} м/с\n"
        f"θ   = {theta_all[i]:+5.2f}°  ref={theta_ref[i]:+.1f}°\n"
        f"α   = {a_cur:+5.2f}°{warn_str}\n"
        f"q   = {q_all[i]:+5.2f} °/с\n"
        f"δe  = {de_all[i]:+5.1f}°\n"
        f"газ = {thr_all[i]:.3f}\n"
        f"ИНД = {_ind_labels[lv_cur]}"
    )

    ln_th.set_data(ts, theta_all[:i+1])
    pt_th.set_data([t_cur], [theta_all[i]])
    ln_q.set_data(ts, q_all[:i+1])
    pt_q.set_data([t_cur], [q_all[i]])
    ln_al.set_data(ts, alpha_all[:i+1])
    pt_al.set_data([t_cur], [alpha_all[i]])
    ln_de.set_data(ts, de_all[:i+1])
    pt_de.set_data([t_cur], [de_all[i]])
    ln_thr.set_data(ts, thr_all[:i+1])
    pt_thr.set_data([t_cur], [thr_all[i]])

    # Индикатор УА (ступенчатая линия + цветная точка)
    ln_ind.set_data(ts, alert_all[:i+1])
    pt_ind.set_data([t_cur], [lv_cur])
    pt_ind.set_color(_ind_colors[lv_cur])

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

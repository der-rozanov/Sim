# -*- coding: utf-8 -*-
"""
С8-б: Контроль высоты + удержание скорости + метрика работы мотора.

Отличие от С8: вместо нормированного прокси  throttle * Va  используется
физически корректная работа мотора:

  T(t)    = 0.5·ρ·S_prop·C_prop·((k_motor·δt)² − Va²)   [Н]
  P_mot(t) = T(t) · Va(t)                                  [Вт]
  W_mot(t) = ∫₀ᵗ P_mot dτ                                  [Дж]  — нарастающий итог

Физический смысл:
  В горизонтальном установившемся полёте T = D, поэтому
  P_mot = D · Va = D_min · Va_opt при оптимальном УА.
  Чем ближе режим к оптимуму L/D, тем меньше W_mot за полёт.

Запуск:
    python scenarios/s8_b_motor_work.py
    python scenarios/s8_b_motor_work.py results/s8b.gif
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
from sim.dynamics import thrust as motor_thrust
from runner import run, compute_trim, trim_state, print_summary
from control.controllers import PitchController, PitchControlParams, SpeedController, SpeedControlParams
from control.sensors import (measure_gyro, measure_altitude, measure_airspeed,
                              measure_angle_of_attack, measure_gps_velocity_earth)
from control.estimators import estimate_alpha_indirect
from sim.state import THETA, Q, H, X, U, W
from flight_logger import FlightLogger

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры сценария
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams(Vw_const=5.0)
sp          = SensorParams()

VA_REF = 25.0   # целевая скорость полёта (точка замера)
cfg    = SimConfig(Va0=VA_REF, h0=100.0, theta0=0.0, dt=0.01, t_end=60.0)

H_TRIM = 100.0
H_HIGH = 150.0
VW        = wind_params.Vw_const
T_CLIMB   = 10.0
T_DESCEND = 40.0
KH        = 0.006

ANIM_SPEED = 2.0
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка на целевой скорости
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, VA_REF)
s0 = trim_state(aircraft, cfg)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  throttle={thr_trim:.3f}")
print(f"Ветер: Vwx={VW:+.1f} м/с")

# Тяга в тримовом режиме — для справки
T_trim = motor_thrust(thr_trim, cfg.Va0, aircraft)
P_trim = T_trim * cfg.Va0
print(f"Trim:  T={T_trim:.2f} Н  P_mot={P_trim:.1f} Вт")

# ------------------------------------------------------------------
# Регуляторы
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=30.0, gain_scheduling=True)  # 30.0 = скорость настройки ПИД
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': H_TRIM})

spd_params = SpeedControlParams()
spd_ctrl   = SpeedController(aircraft, spd_params)
spd_ctrl.set_trim_throttle(thr_trim)
spd_ctrl.set_Va_ref(VA_REF)
spd_ctrl.reset()

rng = np.random.default_rng(seed=42)

h_ref_buf     = []
theta_ref_buf = []

# ------------------------------------------------------------------
# Функция управления
# ------------------------------------------------------------------
def controls_fn(t, state, Va, alpha):
    if t < T_CLIMB:
        h_ref = H_TRIM
    elif t < T_DESCEND:
        h_ref = H_HIGH
    else:
        h_ref = H_TRIM

    h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    h_err     = h_ref - h_meas
    theta_ref = np.clip(alpha_trim + KH * h_err,
                        np.radians(-45.0), np.radians(45.0))
    controller.set_pitch_setpoint(theta_ref)
    h_ref_buf.append(h_ref)
    theta_ref_buf.append(theta_ref)

    meas     = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    ctrl_out = controller.step(t, meas, cfg.dt)
    delta_e  = ctrl_out[0]
    throttle = spd_ctrl.step(Va_meas, cfg.dt)

    return np.array([delta_e, throttle])

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
logger = FlightLogger(
    scenario="С8б: Контроль высоты + Va + работа мотора",
    description=f"Набор h={H_HIGH:.0f} м, ветер Vwx={VW:+.0f} м/с, Va_ref={VA_REF:.0f} м/с",
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

log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С8б  Работа мотора")

try:
    logger.save(log, h_ref=h_ref_buf, theta_ref=theta_ref_buf)
except OSError as e:
    print(f"[logger.save пропущен: {e}]")

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n         = len(log.t)
t_all     = log.t
t_end     = t_all[-1]
dt_sim    = t_all[1] - t_all[0]
h_all     = log.state[:, H]
x_all     = log.state[:, X]
h_ref_all = np.array(h_ref_buf[:n])
theta_all = np.degrees(log.state[:, THETA])
theta_ref = np.degrees(np.array(theta_ref_buf[:n]))
de_all    = np.degrees(log.controls[:, 0])
thr_all   = log.controls[:, 1]
Va_all    = log.Va
alpha_all = np.degrees(log.alpha)

# ------------------------------------------------------------------
# Метрика работы мотора (физически корректная)
# ------------------------------------------------------------------
T_all   = np.array([motor_thrust(float(thr), float(Va), aircraft)
                    for thr, Va in zip(thr_all, Va_all)])   # Н
P_all   = T_all * Va_all                                    # Вт
W_cum   = np.cumsum(P_all) * dt_sim                         # Дж, нарастающий итог
W_total = W_cum[-1]                                         # Дж, итого

# Прокси из С8 для сравнения
P_proxy = thr_all * Va_all
W_proxy = np.cumsum(P_proxy) * dt_sim

# ------------------------------------------------------------------
# Статистика
# ------------------------------------------------------------------
mask_climb   = (t_all >= T_CLIMB)   & (t_all < T_DESCEND)
mask_descend = (t_all >= T_DESCEND)

print(f"\n{'='*52}")
print(f"  Работа мотора  W = ∫T·Va dt")
print(f"  Полная работа:          W_total = {W_total:9.1f} Дж")
print(f"  Средняя мощность:       P_mean  = {P_all.mean():9.2f} Вт")
print(f"  Пиковая мощность:       P_max   = {P_all.max():9.2f} Вт  (набор)")
print(f"  Пик тяги:               T_max   = {T_all.max():9.2f} Н")
print(f"  Тримовая мощность:      P_trim  = {P_trim:9.2f} Вт")
print(f"  Работа фазы набора:     W_climb = {np.sum(P_all[mask_climb])*dt_sim:9.1f} Дж")
print(f"  Работа фазы снижения:   W_desc  = {np.sum(P_all[mask_descend])*dt_sim:9.1f} Дж")
print(f"{'='*52}")
print(f"  Прокси С8 (throttle·Va): W_proxy = {W_proxy[-1]:9.1f}  (о.е.·м — др. единицы, не сравнивать)")
print(f"{'='*52}")

# ------------------------------------------------------------------
# Прореживание кадров
# ------------------------------------------------------------------
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры  (7 правых субплотов)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 15))
fig.suptitle(
    f"С8б: Контроль высоты + Va={VA_REF:.0f} м/с  |  Ветер Vwx={VW:+.0f} м/с\n"
    f"Работа мотора  W = ∫T·Va dt  |  W_итог = {W_total:.0f} Дж  "
    f"(P_trim={P_trim:.1f} Вт)",
    fontsize=11, fontweight="bold"
)

gs = gridspec.GridSpec(7, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.72, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_de   = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_Va   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_P    = fig.add_subplot(gs[5, 1], sharex=ax_h)   # мощность мотора
ax_W    = fig.add_subplot(gs[6, 1], sharex=ax_h)   # накопленная работа

# ------------------------------------------------------------------
# Левый: траектория
# ------------------------------------------------------------------
_px = max((x_all.max() - x_all.min()) * 0.06, 10.0)
_ph = max((h_all.max() - h_all.min()) * 0.30, 20.0)
ax_traj.set_xlim(x_all.min() - _px, x_all.max() + _px)
ax_traj.set_ylim(h_all.min() - _ph, h_all.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.set_xlabel("x, м", fontsize=9)
ax_traj.set_ylabel("h, м", fontsize=9)
ax_traj.set_title("Траектория", fontsize=9)
ax_traj.grid(True, ls="--", alpha=0.5)
ax_traj.axhline(H_TRIM, color="steelblue", lw=1.1, ls="--", alpha=0.6, label=f"h={H_TRIM:.0f}")
ax_traj.axhline(H_HIGH, color="orange",    lw=1.1, ls="--", alpha=0.6, label=f"h={H_HIGH:.0f}")
ax_traj.legend(fontsize=8, loc="upper left")
ax_traj.plot(x_all, h_all, color="lightsteelblue", lw=1.2, alpha=0.4)
ax_traj.plot(x_all[0],  h_all[0],  "go", ms=8, zorder=5)
ax_traj.plot(x_all[-1], h_all[-1], "rs", ms=8, zorder=5)

traj_line,   = ax_traj.plot([], [], "b-",  lw=2.0, zorder=3)
traj_marker, = ax_traj.plot([], [], "b^",  ms=11,  zorder=6, markeredgecolor="navy")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes, fontsize=8.0,
    ha="right", va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.88),
)

# ------------------------------------------------------------------
# Общие настройки правых субплотов
# ------------------------------------------------------------------
right_axes = (ax_h, ax_th, ax_al, ax_de, ax_Va, ax_P, ax_W)
step_events = [(T_CLIMB, "orange", "набор"), (T_DESCEND, "dodgerblue", "снижение")]

for ax in right_axes:
    ax.set_xlim(0.0, t_end)
    ax.grid(True, ls="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    for t_ev, col, lbl in step_events:
        ax.axvline(t_ev, color=col, lw=0.9, ls=":", alpha=0.7,
                   label=f"{lbl} t={t_ev:.0f}с" if ax is ax_h else None)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_W.set_xlabel("Время, с", fontsize=9)

# ---- Высота ----
ax_h.set_ylabel("h, м", fontsize=9)
_h_pad = max((h_all.max() - h_all.min()) * 0.25, 8.0)
ax_h.set_ylim(h_all.min() - _h_pad, h_all.max() + _h_pad)
ax_h.plot(t_all, h_ref_all, color="black", lw=1.1, ls="--", alpha=0.55, label="h_ref")
ax_h.plot(t_all, h_all, color="lightsteelblue", lw=1.0, alpha=0.4)
ax_h.legend(fontsize=7, loc="upper right")
ln_h, = ax_h.plot([], [], color="royalblue", lw=1.8)
pt_h, = ax_h.plot([], [], "o", color="royalblue", ms=5, zorder=5)

# ---- Тангаж ----
ax_th.set_ylabel("θ, °", fontsize=9)
_th_lo = min(theta_all.min(), theta_ref.min()) - 2.0
_th_hi = max(theta_all.max(), theta_ref.max()) + 2.0
ax_th.set_ylim(_th_lo, _th_hi)
ax_th.axhline(0, color="gray", lw=0.8, ls=":")
ax_th.plot(t_all, theta_all, color="lightseagreen", lw=1.0, alpha=0.4)
ax_th.plot(t_all, theta_ref, color="black",         lw=1.1, ls="--", alpha=0.55, label="θ_ref")
ax_th.legend(fontsize=7, loc="upper right")
ln_th, = ax_th.plot([], [], color="seagreen", lw=1.8)
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- УА ----
ax_al.set_ylabel("α, °", fontsize=9)
_al_pad = max(abs(alpha_all).max() * 0.25, 0.5)
ax_al.set_ylim(alpha_all.min() - _al_pad, alpha_all.max() + _al_pad)
ax_al.axhline(np.degrees(alpha_trim), color="black", lw=0.9, ls="--",
              alpha=0.5, label=f"trim={np.degrees(alpha_trim):.1f}°")
ax_al.plot(t_all, alpha_all, color="plum", lw=1.0, alpha=0.4)
ax_al.legend(fontsize=7, loc="upper right")
ln_al, = ax_al.plot([], [], color="purple", lw=1.8)
pt_al, = ax_al.plot([], [], "o", color="purple", ms=5, zorder=5)

# ---- Руль высоты ----
ax_de.set_ylabel("δe, °", fontsize=9)
_de_pad = max(abs(de_all).max() * 0.18, 1.5)
ax_de.set_ylim(de_all.min() - _de_pad, de_all.max() + _de_pad)
ax_de.axhline(0, color="gray", lw=0.8, ls=":")
ax_de.plot(t_all, de_all, color="burlywood", lw=1.0, alpha=0.5)
ln_de, = ax_de.plot([], [], color="saddlebrown", lw=1.8)
pt_de, = ax_de.plot([], [], "o", color="saddlebrown", ms=5, zorder=5)

# ---- Воздушная скорость ----
ax_Va.set_ylabel("Va, м/с", fontsize=9)
ax_Va.set_ylim(min(Va_all.min(), VA_REF) - 1.0, max(Va_all.max(), VA_REF) + 1.0)
ax_Va.axhline(VA_REF, color="black", lw=1.1, ls="--", alpha=0.55, label=f"ref={VA_REF:.0f}")
ax_Va.plot(t_all, Va_all, color="lightsteelblue", lw=1.0, alpha=0.45)
ax_Va.legend(fontsize=7, loc="upper right")
ln_Va, = ax_Va.plot([], [], color="steelblue", lw=1.8)
pt_Va, = ax_Va.plot([], [], "o", color="steelblue", ms=5, zorder=5)

# ---- Мощность мотора ----
ax_P.set_ylabel("P_мот, Вт", fontsize=9)
_P_lo = min(P_all.min() * 1.1, -5.0)   # авторотация может давать P < 0
_P_hi = P_all.max() * 1.1
ax_P.set_ylim(_P_lo, _P_hi)
ax_P.axhline(P_trim, color="black", lw=0.9, ls="--", alpha=0.45,
             label=f"trim={P_trim:.1f} Вт")
ax_P.axhline(0, color="gray", lw=0.7, ls=":")
ax_P.plot(t_all, P_all, color="lightsalmon", lw=1.0, alpha=0.5)
ax_P.legend(fontsize=7, loc="upper right")
ln_P, = ax_P.plot([], [], color="orangered", lw=1.8)
pt_P, = ax_P.plot([], [], "o", color="orangered", ms=5, zorder=5)

# ---- Накопленная работа ----
ax_W.set_ylabel("W, Дж", fontsize=9)
ax_W.set_ylim(0, W_cum.max() * 1.08)
ax_W.plot(t_all, W_cum, color="lightgreen", lw=1.0, alpha=0.4)
ax_W.set_title(f"Работа мотора  W_итог={W_total:.0f} Дж", fontsize=8, pad=2)
ln_W, = ax_W.plot([], [], color="darkgreen", lw=1.8)
pt_W, = ax_W.plot([], [], "o", color="darkgreen", ms=5, zorder=5)

# Курсор времени
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65) for ax in right_axes]

# ------------------------------------------------------------------
# init / update
# ------------------------------------------------------------------
_all_artists = (
    traj_line, traj_marker, info_box,
    ln_h, pt_h, ln_th, pt_th,
    ln_al, pt_al, ln_de, pt_de,
    ln_Va, pt_Va, ln_P, pt_P, ln_W, pt_W,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_al, ln_de, ln_Va, ln_P, ln_W):
        ln.set_data([], [])
    for pt in (pt_h, pt_th, pt_al, pt_de, pt_Va, pt_P, pt_W):
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
        f"t      = {t_cur:5.1f} с\n"
        f"h      = {h_all[i]:6.1f} м  ref={h_ref_all[i]:.0f}\n"
        f"Va     = {Va_all[i]:5.1f} м/с\n"
        f"α      = {alpha_all[i]:+5.2f}°\n"
        f"T      = {T_all[i]:5.1f} Н\n"
        f"P_мот  = {P_all[i]:6.1f} Вт\n"
        f"W      = {W_cum[i]:7.0f} Дж"
    )

    ln_h.set_data(ts, h_all[:i+1]);      pt_h.set_data([t_cur], [h_all[i]])
    ln_th.set_data(ts, theta_all[:i+1]); pt_th.set_data([t_cur], [theta_all[i]])
    ln_al.set_data(ts, alpha_all[:i+1]); pt_al.set_data([t_cur], [alpha_all[i]])
    ln_de.set_data(ts, de_all[:i+1]);    pt_de.set_data([t_cur], [de_all[i]])
    ln_Va.set_data(ts, Va_all[:i+1]);    pt_Va.set_data([t_cur], [Va_all[i]])
    ln_P.set_data(ts, P_all[:i+1]);      pt_P.set_data([t_cur], [P_all[i]])
    ln_W.set_data(ts, W_cum[:i+1]);      pt_W.set_data([t_cur], [W_cum[i]])

    for vl in vlines:
        vl.set_xdata([t_cur])
    return _all_artists

anim = FuncAnimation(fig, update, frames=n_frames,
                     init_func=init, interval=1000.0/ANIM_FPS, blit=True)

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
            print(f"Ошибка GIF: {e}")
    else:
        try:
            anim.save(save_path, fps=ANIM_FPS, extra_args=["-vcodec", "libx264"])
            print(f"Сохранено: {save_path}")
        except Exception as e:
            print(f"Ошибка MP4: {e}")

plt.show()

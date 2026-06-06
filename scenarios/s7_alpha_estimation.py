# -*- coding: utf-8 -*-
"""
С7: Контроль высоты + удержание скорости + сравнение методов оценки УА.

Структура управления — идентична С6:
  h_ref → [P_h] → theta_ref → [каскадный ПИД theta+q] → delta_e
  Va_ref → [ПИД Va] → throttle

Добавлено: попутный ветер 5 м/с, чтобы показать расхождение методов оценки УА:
  • Истинный УА — из симулятора (эталон)
  • Зонд (прямое измерение): alpha_probe = alpha_true + шум_зонда
  • Косвенная оценка (ИНС+GPS): alpha_est = theta_мес − gamma_gps
    При попутном ветре Vwx > 0: gamma_gps < gamma_air →  alpha_est > alpha_true
    (систематическое завышение во время набора, занижение во время снижения)

Запуск:  python scenarios/s7_alpha_estimation.py
         python scenarios/s7_alpha_estimation.py results/s7.gif
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

from config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state, print_summary
from control import PitchController, PitchControlParams, SpeedController, SpeedControlParams
from sensors import (measure_gyro, measure_altitude, measure_airspeed,
                     measure_angle_of_attack, measure_gps_velocity_earth)
from estimators import estimate_alpha_indirect
from state import THETA, Q, H, X, U, W

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры сценария
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams(Vw_const=5.0)   # попутный ветер 5 м/с
sp          = SensorParams()
cfg         = SimConfig(Va0=30.0, h0=100.0, theta0=0.0, dt=0.01, t_end=60.0)

H_TRIM    = 100.0   # начальная высота, м
H_HIGH    = 200.0   # уставка набора, м
VA_REF    = 40.0    # уставка воздушной скорости, м/с
VW        = wind_params.Vw_const   # для подписей
T_CLIMB   = 10.0    # с, команда набора
T_DESCEND = 40.0    # с, команда снижения

KH = 0.006          # рад/м — P-коэффициент контура высоты

ANIM_SPEED = 2.0
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка и начальное состояние
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

print(f"Trim:  alpha={np.degrees(alpha_trim):.2f} deg  "
      f"delta_e={np.degrees(de_trim):.2f} deg  "
      f"throttle={thr_trim:.3f}")
print(f"Ветер: Vwx={VW:+.1f} м/с (попутный)")

# ------------------------------------------------------------------
# Регуляторы (идентичны С6)
# ------------------------------------------------------------------
ctrl_params = PitchControlParams(Va_ref=cfg.Va0)
controller  = PitchController(aircraft, ctrl_params)
controller.set_trim_throttle(thr_trim)
controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': H_TRIM})

spd_params  = SpeedControlParams()
spd_ctrl    = SpeedController(aircraft, spd_params)
spd_ctrl.set_trim_throttle(thr_trim)
spd_ctrl.set_Va_ref(VA_REF)
spd_ctrl.reset()

rng = np.random.default_rng(seed=42)

# Буферы для логирования
h_ref_buf      = []
theta_ref_buf  = []
alpha_probe_buf = []
alpha_est_buf  = []

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

    # Измерения базовых датчиков
    h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    q_meas     = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    theta_meas = state[THETA] + rng.normal(0.0, sp.gyro_noise)
    Va_meas    = measure_airspeed(Va, sp.airspeed_bias, sp.airspeed_noise, rng)

    # --- Оценка УА (два метода) ---
    # 1) Зонд: прямое измерение
    alpha_probe = measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng)

    # 2) Косвенная оценка: alpha = theta - gamma_gps
    Vx_gps, Vh_gps = measure_gps_velocity_earth(
        state[U], state[W], state[THETA],
        bias=0.0, noise_std=sp.gps_vel_noise, rng=rng,
    )
    alpha_est = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)

    alpha_probe_buf.append(alpha_probe)
    alpha_est_buf.append(alpha_est)

    # Контур высоты → theta_ref
    h_err     = h_ref - h_meas
    theta_ref = np.clip(alpha_trim + KH * h_err,
                        np.radians(-15.0), np.radians(15.0))

    controller.set_pitch_setpoint(theta_ref)
    h_ref_buf.append(h_ref)
    theta_ref_buf.append(theta_ref)

    # Контур тангажа → delta_e
    meas     = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
    ctrl_out = controller.step(t, meas, cfg.dt)
    delta_e  = ctrl_out[0]

    # Контур скорости → throttle
    throttle = spd_ctrl.step(Va_meas, cfg.dt)

    return np.array([delta_e, throttle])

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С7  Оценка УА (зонд vs косвенная)")

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n  = len(log.t)
t_all       = log.t
t_end       = t_all[-1]
h_all       = log.state[:, H]
x_all       = log.state[:, X]
h_ref_all   = np.array(h_ref_buf[:n])
theta_all   = np.degrees(log.state[:, THETA])
theta_ref   = np.degrees(np.array(theta_ref_buf[:n]))
de_all      = np.degrees(log.controls[:, 0])
thr_all     = log.controls[:, 1]
Va_all      = log.Va
alpha_true  = np.degrees(log.alpha)
alpha_probe = np.degrees(np.array(alpha_probe_buf[:n]))
alpha_est   = np.degrees(np.array(alpha_est_buf[:n]))

# ------------------------------------------------------------------
# Прореживание кадров
# ------------------------------------------------------------------
dt_sim   = t_all[1] - t_all[0]
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры  (7 правых субплотов)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 15))
fig.suptitle(
    f"С7: Контроль высоты + удержание Va={VA_REF:.0f} м/с  |  Ветер: Vwx={VW:+.0f} м/с  (анимация)\n"
    f"Сравнение оценок УА: зонд (прямое) vs ИНС+GPS (косвенное)\n"
    f"(трим 0–{T_CLIMB:.0f} с,  набор {T_CLIMB:.0f}–{T_DESCEND:.0f} с,"
    f"  снижение {T_DESCEND:.0f}–{cfg.t_end:.0f} с)",
    fontsize=11, fontweight="bold"
)

gs = gridspec.GridSpec(7, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.72, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_th   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_al   = fig.add_subplot(gs[2, 1], sharex=ax_h)   # УА: главный новый график
ax_de   = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_Va   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[5, 1], sharex=ax_h)
ax_err  = fig.add_subplot(gs[6, 1], sharex=ax_h)   # ошибка оценки

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

ax_traj.axhline(H_TRIM, color="steelblue", lw=1.1, ls="--", alpha=0.6,
                label=f"h={H_TRIM:.0f} м")
ax_traj.axhline(H_HIGH, color="orange",    lw=1.1, ls="--", alpha=0.6,
                label=f"h={H_HIGH:.0f} м")
ax_traj.legend(fontsize=8, loc="upper left")
ax_traj.plot(x_all, h_all, color="lightsteelblue", lw=1.2, alpha=0.4, zorder=1)
ax_traj.plot(x_all[0],  h_all[0],  "go", ms=8, zorder=5)
ax_traj.plot(x_all[-1], h_all[-1], "rs", ms=8, zorder=5)

traj_line,   = ax_traj.plot([], [], "b-", lw=2.0, zorder=3)
traj_marker, = ax_traj.plot([], [], "b^", ms=11,  zorder=6, markeredgecolor="navy")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes,
    fontsize=8.0, ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="gray", alpha=0.88),
    family="monospace"
)

# ------------------------------------------------------------------
# Общие настройки правых субплотов
# ------------------------------------------------------------------
right_axes = (ax_h, ax_th, ax_al, ax_de, ax_Va, ax_thr, ax_err)
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
ax_err.set_xlabel("Время, с", fontsize=9)

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
ax_th.set_ylabel("θ, °", fontsize=9)
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

# ---- Угол атаки: ключевой subplot ----
ax_al.set_ylabel("УА α, °", fontsize=9)
_al_vals = np.concatenate([alpha_true, alpha_probe, alpha_est])
_al_pad  = max((_al_vals.max() - _al_vals.min()) * 0.2, 0.5)
ax_al.set_ylim(_al_vals.min() - _al_pad, _al_vals.max() + _al_pad)
ax_al.axhline(0, color="gray", lw=0.8, ls=":")

# Полные временные ряды (фон, серый)
ax_al.plot(t_all, alpha_true,  color="silver",    lw=1.2, alpha=0.6, zorder=1, label="истинный")
ax_al.plot(t_all, alpha_probe, color="steelblue", lw=0.9, alpha=0.35, zorder=2)
ax_al.plot(t_all, alpha_est,   color="tomato",    lw=0.9, alpha=0.35, zorder=2)
ax_al.legend(fontsize=7, loc="upper right")

ln_al_true,  = ax_al.plot([], [], color="gray",      lw=1.5, zorder=3)
ln_al_probe, = ax_al.plot([], [], color="steelblue", lw=1.5, zorder=4, label="зонд")
ln_al_est,   = ax_al.plot([], [], color="tomato",    lw=1.5, ls="--", zorder=4, label="ИНС+GPS")
pt_al,       = ax_al.plot([], [], "o", color="gray", ms=5,   zorder=5)
ax_al.legend(fontsize=7, loc="upper right")

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
_Va_lo = min(Va_all.min(), VA_REF) - 1.0
_Va_hi = max(Va_all.max(), VA_REF) + 1.0
ax_Va.set_ylim(_Va_lo, _Va_hi)
ax_Va.axhline(VA_REF, color="black", lw=1.1, ls="--", alpha=0.55,
              label=f"Va_ref={VA_REF:.0f}")
ax_Va.plot(t_all, Va_all, color="lightsteelblue", lw=1.0, alpha=0.45)
ax_Va.legend(fontsize=7, loc="upper right")

ln_Va, = ax_Va.plot([], [], color="steelblue", lw=1.8)
pt_Va, = ax_Va.plot([], [], "o", color="steelblue", ms=5, zorder=5)

# ---- Газ (тяга) ----
ax_thr.set_ylabel("Тяга, о.е.", fontsize=9)
ax_thr.set_ylim(-0.05, 1.05)
ax_thr.axhline(thr_trim, color="black", lw=1.0, ls="--", alpha=0.4,
               label=f"trim={thr_trim:.2f}")
ax_thr.plot(t_all, thr_all, color="lightgreen", lw=1.0, alpha=0.5)
ax_thr.legend(fontsize=7, loc="upper right")

ln_thr, = ax_thr.plot([], [], color="darkgreen", lw=1.8)
pt_thr, = ax_thr.plot([], [], "o", color="darkgreen", ms=5, zorder=5)

# ---- Ошибка оценки УА ----
err_probe = alpha_probe - alpha_true   # зонд vs истинный
err_est   = alpha_est   - alpha_true   # косвенная vs истинный

ax_err.set_ylabel("Δα, °", fontsize=9)
_err_vals = np.concatenate([err_probe, err_est])
_err_abs  = max(abs(_err_vals).max() * 1.25, 0.3)
ax_err.set_ylim(-_err_abs, _err_abs)
ax_err.axhline(0, color="gray", lw=1.0, ls="-")
ax_err.plot(t_all, err_probe, color="steelblue", lw=0.9, alpha=0.4)
ax_err.plot(t_all, err_est,   color="tomato",    lw=0.9, alpha=0.4)
ax_err.set_title("Ошибка оценки (зонд − синий, ИНС+GPS − красный)",
                 fontsize=7.5, pad=2)

ln_err_probe, = ax_err.plot([], [], color="steelblue", lw=1.5, label="зонд")
ln_err_est,   = ax_err.plot([], [], color="tomato",    lw=1.5, ls="--", label="ИНС+GPS")
pt_err,       = ax_err.plot([], [], "o", color="tomato", ms=5, zorder=5)
ax_err.legend(fontsize=7, loc="upper right")

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
    ln_al_true, ln_al_probe, ln_al_est, pt_al,
    ln_de,  pt_de,
    ln_Va,  pt_Va,
    ln_thr, pt_thr,
    ln_err_probe, ln_err_est, pt_err,
    *vlines,
)

def init():
    traj_line.set_data([], [])
    traj_marker.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h, ln_th, ln_al_true, ln_al_probe, ln_al_est,
               ln_de, ln_Va, ln_thr, ln_err_probe, ln_err_est):
        ln.set_data([], [])
    for pt in (pt_h, pt_th, pt_al, pt_de, pt_Va, pt_thr, pt_err):
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
        f"t    = {t_cur:5.1f} с\n"
        f"h    = {h_all[i]:6.1f} м  ref={h_ref_all[i]:.0f} м\n"
        f"Va   = {Va_all[i]:5.1f} м/с  ref={VA_REF:.0f}\n"
        f"θ    = {theta_all[i]:+5.2f}°\n"
        f"α_ист  = {alpha_true[i]:+5.2f}°\n"
        f"α_зонд = {alpha_probe[i]:+5.2f}°  Δ={alpha_probe[i]-alpha_true[i]:+.2f}°\n"
        f"α_оц  = {alpha_est[i]:+5.2f}°  Δ={alpha_est[i]-alpha_true[i]:+.2f}°"
    )

    ln_h.set_data(ts, h_all[:i+1]);         pt_h.set_data([t_cur], [h_all[i]])
    ln_th.set_data(ts, theta_all[:i+1]);    pt_th.set_data([t_cur], [theta_all[i]])

    ln_al_true.set_data(ts, alpha_true[:i+1])
    ln_al_probe.set_data(ts, alpha_probe[:i+1])
    ln_al_est.set_data(ts, alpha_est[:i+1])
    pt_al.set_data([t_cur], [alpha_true[i]])

    ln_de.set_data(ts, de_all[:i+1]);       pt_de.set_data([t_cur], [de_all[i]])
    ln_Va.set_data(ts, Va_all[:i+1]);       pt_Va.set_data([t_cur], [Va_all[i]])
    ln_thr.set_data(ts, thr_all[:i+1]);     pt_thr.set_data([t_cur], [thr_all[i]])

    ln_err_probe.set_data(ts, err_probe[:i+1])
    ln_err_est.set_data(ts, err_est[:i+1])
    pt_err.set_data([t_cur], [err_est[i]])

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
# Вывод статистики ошибок
# ------------------------------------------------------------------
# Только фаза набора/снижения (без тримовых участков)
mask_maneuver = (t_all >= T_CLIMB) & (t_all <= t_all[-1])
print(f"Ошибки оценки УА (фаза манёвра {T_CLIMB:.0f}–{t_all[-1]:.0f} с):")
print(f"  Зонд:    СКО = {err_probe[mask_maneuver].std():.3f}°  |  "
      f"max |Δ| = {abs(err_probe[mask_maneuver]).max():.3f}°  |  "
      f"смещение = {err_probe[mask_maneuver].mean():+.3f}°")
print(f"  ИНС+GPS: СКО = {err_est[mask_maneuver].std():.3f}°  |  "
      f"max |Δ| = {abs(err_est[mask_maneuver]).max():.3f}°  |  "
      f"смещение = {err_est[mask_maneuver].mean():+.3f}°")

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

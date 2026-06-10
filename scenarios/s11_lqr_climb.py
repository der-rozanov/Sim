# -*- coding: utf-8 -*-
"""
С11: LQR — набор высоты → горизонтальный полёт → снижение.

Три фазы:
  0        – T_CRUISE  с — набор:       h_ref = H_CRUISE (150 м)
  T_CRUISE – T_DESCEND с — горизонт:   h_ref = H_CRUISE
  T_DESCEND – T_END    с — снижение:   h_ref = H_INIT   (100 м)

LQR-регулятор: численная линеаризация + DARE (scipy).
Измерения: гироскоп q, барометр h, СВС Va, ИНС theta.
u, w реконструируются из Va и theta (приближение без зонда УА).

Запуск:  python scenarios/s11_lqr_climb.py
         python scenarios/s11_lqr_climb.py results/s11.gif   -- сохранить анимацию
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.config import AircraftParams, WindParams, SensorParams, SimConfig
from sim.state import U, W, Q, THETA, H, X
from sim.wind import wind as _wind_base
from control.lqr import LQRController, LQRParams
from control.sensors import measure_gyro, measure_altitude, measure_airspeed
from runner import run, compute_trim, trim_state, print_summary
from flight_logger import FlightLogger

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
aircraft    = AircraftParams()
wind_params = WindParams()
sp          = SensorParams()
cfg = SimConfig(dt=0.01, t_end=145.0, Va0=30.0, h0=100.0)

VA_REF    = 30.0
H_INIT    = 100.0
H_CRUISE  = 150.0
T_CRUISE  = 50.0    # с, начало горизонтального полёта
T_DESCEND = 100.0   # с, начало снижения

ANIM_SPEED = 8.0    # ускорение анимации (×реальное время)
ANIM_FPS   = 25

# ------------------------------------------------------------------
# Балансировка
# ------------------------------------------------------------------
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, VA_REF)
s0            = trim_state(aircraft, cfg)
controls_trim = np.array([de_trim, thr_trim])

def wind_fn(h, t):
    return _wind_base(h, t, wind_params)

# ------------------------------------------------------------------
# LQR-регулятор
# ------------------------------------------------------------------
lqr_params = LQRParams(
    Q_diag=[1.0, 1.0, 10.0, 100.0],
    R_diag=[10.0, 1.0],
    h_Kp=0.05,
    h_ref=H_INIT,
    Va_ref=VA_REF,
)
lqr = LQRController(aircraft, s0, controls_trim, wind_fn, cfg.dt, lqr_params)
lqr.print_gains()

print(f"Трим: alpha={np.degrees(alpha_trim):.2f}°  "
      f"delta_e={np.degrees(de_trim):.2f}°  thr={thr_trim:.3f}")

# ------------------------------------------------------------------
# Лётный журнал
# ------------------------------------------------------------------
logger = FlightLogger(
    scenario="С11: LQR набор/горизонт/снижение",
    description=(f"LQR h_ref: {H_INIT:.0f}→{H_CRUISE:.0f}→{H_INIT:.0f} м, "
                 f"Va={VA_REF:.0f} м/с"),
    aircraft=aircraft,
    wind_params=wind_params,
    cfg=cfg,
    sp=sp,
    trim=(alpha_trim, de_trim, thr_trim),
    events=[
        {"t": T_CRUISE,  "label": "набор→горизонт",    "color": "steelblue"},
        {"t": T_DESCEND, "label": "горизонт→снижение",  "color": "darkorange"},
    ],
)

# ------------------------------------------------------------------
# Закон управления
# ------------------------------------------------------------------
rng           = np.random.default_rng(42)
h_ref_buf     = []
theta_ref_buf = []   # эффективная уставка тангажа из внешнего P-контура LQR

# Экспоненциальный фильтр барометра: h_filt = α·h_filt + (1−α)·h_meas
# α=0.9 при dt=0.01 с → постоянная времени τ = dt/(1-α) ≈ 0.1 с
H_FILT_ALPHA = 0.9
h_filt       = [s0[H]]   # список — мутабельный контейнер для замыкания

def controls_fn(t, state, Va, alpha):  # noqa: ARG001 — alpha required by runner interface
    h_ref = H_CRUISE if t < T_DESCEND else H_INIT
    lqr.set_altitude_ref(h_ref)
    h_ref_buf.append(h_ref)

    q_meas     = measure_gyro(state[Q],     sp.gyro_bias,     sp.gyro_noise,     rng)
    h_meas     = measure_altitude(state[H], sp.baro_bias,     sp.baro_noise,     rng)
    Va_meas    = measure_airspeed(Va,       sp.airspeed_bias, sp.airspeed_noise, rng)
    theta_meas = state[THETA]

    # Фильтрация высоты перед подачей во внешний P-контур
    h_filt[0] = H_FILT_ALPHA * h_filt[0] + (1.0 - H_FILT_ALPHA) * h_meas

    # Сохраняем эффективную уставку тангажа (зеркало внешнего контура LQR)
    theta_ref_eff = lqr.x_trim[3] + lqr.params.h_Kp * (h_ref - h_filt[0])
    theta_ref_eff = np.clip(theta_ref_eff, np.radians(-25.0), np.radians(25.0))
    theta_ref_buf.append(theta_ref_eff)

    u_meas = Va_meas * np.cos(theta_meas)
    w_meas = Va_meas * np.sin(theta_meas)

    meas = {
        "u": u_meas, "w": w_meas,
        "q": q_meas, "theta": theta_meas,
        "h": h_filt[0], "Va": Va_meas,
    }
    return lqr.step(t, meas, cfg.dt)

# ------------------------------------------------------------------
# Прогон
# ------------------------------------------------------------------
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
print_summary(log, aircraft, label="С11  LQR набор/горизонт/снижение")
logger.save(log, h_ref=h_ref_buf, theta_ref=theta_ref_buf)

# ------------------------------------------------------------------
# Массивы данных
# ------------------------------------------------------------------
n         = len(log.t)
t_arr     = log.t
t_end_act = t_arr[-1]
h_arr     = log.state[:, H]
x_arr     = log.state[:, X]
theta_arr = np.degrees(log.state[:, THETA])
theta_ref = np.degrees(np.array(theta_ref_buf[:n]))
h_ref_arr = np.array(h_ref_buf[:n])
q_arr     = np.degrees(log.state[:, Q])
alpha_arr = np.degrees(log.alpha)
de_arr    = np.degrees(log.controls[:, 0])
thr_arr   = log.controls[:, 1]
Va_arr    = log.Va

warn_deg  = np.degrees(aircraft.alpha_warning)
crit_deg  = np.degrees(aircraft.alpha_crit)
stall_deg = np.degrees(aircraft.alpha_stall)

# Метрики точности
h_err = h_arr - h_ref_arr
mask_climb   = t_arr <  T_CRUISE
mask_cruise  = (t_arr >= T_CRUISE) & (t_arr < T_DESCEND)
mask_descent = t_arr >= T_DESCEND

print("\n  СКО удержания высоты по фазам:")
print(f"    Набор    (0–{T_CRUISE:.0f} с)           : {np.sqrt(np.mean(h_err[mask_climb]**2)):.2f} м")
print(f"    Горизонт ({T_CRUISE:.0f}–{T_DESCEND:.0f} с) : {np.sqrt(np.mean(h_err[mask_cruise]**2)):.2f} м")
print(f"    Снижение ({T_DESCEND:.0f}–{cfg.t_end:.0f} с) : {np.sqrt(np.mean(h_err[mask_descent]**2)):.2f} м")

# ------------------------------------------------------------------
# Прореживание кадров
# ------------------------------------------------------------------
dt_sim   = t_arr[1] - t_arr[0]
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

# ------------------------------------------------------------------
# Компоновка фигуры (идентична s2/s3)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12))
fig.suptitle(
    f"С11: LQR — набор h={H_INIT:.0f}→{H_CRUISE:.0f} м → горизонтальный полёт"
    f" → снижение h={H_INIT:.0f} м\n"
    f"(набор 0–{T_CRUISE:.0f} с,  горизонт {T_CRUISE:.0f}–{T_DESCEND:.0f} с,"
    f"  снижение {T_DESCEND:.0f}–{cfg.t_end:.0f} с,  Va_ref={VA_REF:.0f} м/с)",
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

ax_traj.axhline(H_CRUISE, color="seagreen",  lw=1.3, ls=":", alpha=0.85,
                label=f"h_ref={H_CRUISE:.0f} м")
ax_traj.axhline(H_INIT,   color="royalblue", lw=1.0, ls=":", alpha=0.65,
                label=f"h_ref={H_INIT:.0f} м")
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
    (T_CRUISE,  "steelblue",  f"набор→горизонт t={T_CRUISE:.0f}с"),
    (T_DESCEND, "darkorange", f"горизонт→снижение t={T_DESCEND:.0f}с"),
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
ax_h.plot(t_arr, h_ref_arr, color="black",          lw=1.1, ls="--", alpha=0.55,
          label="h_ref")
ax_h.plot(t_arr, h_arr,     color="lightsteelblue", lw=1.0, alpha=0.4)
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
ax_th.plot(t_arr, theta_ref, color="black",         lw=1.1, ls="--", alpha=0.55,
           label="θ_ref (LQR)")
ax_th.legend(fontsize=7, loc="upper right")

ln_th, = ax_th.plot([], [], color="seagreen", lw=1.8)
pt_th, = ax_th.plot([], [], "o", color="seagreen", ms=5, zorder=5)

# ---- Угол атаки ----
ax_al.set_ylabel("Угол атаки α, °", fontsize=9)
_al_lo = min(alpha_arr.min() - 2.0, -1.0)
_al_hi = max(alpha_arr.max() + 3.0, warn_deg + 5.0)
ax_al.set_ylim(_al_lo, _al_hi)

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

    a_cur = alpha_arr[i]
    if a_cur >= stall_deg:
        warn_str = "  !! СРЫВ !!"
    elif a_cur >= crit_deg:
        warn_str = "  ! критич. !"
    elif a_cur >= warn_deg:
        warn_str = "  предупрежд."
    else:
        warn_str = ""

    if t_cur < T_CRUISE:
        phase_str = "набор"
    elif t_cur < T_DESCEND:
        phase_str = "горизонтальный"
    else:
        phase_str = "снижение"

    info_box.set_text(
        f"t   = {t_cur:5.1f} с  [{phase_str}]\n"
        f"h   = {h_arr[i]:6.1f} м  ref={h_ref_arr[i]:.0f} м\n"
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

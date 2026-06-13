# -*- coding: utf-8 -*-
"""
С10-Б: Парный прогон АУА — зонд vs косвенная оценка при попутном порыве.

Оба прогона имеют АУА включённый. Различается ТОЛЬКО источник УА для АУА:
  Прогон А (зонд):       alpha_probe = alpha_true + bias + шум_зонда
  Прогон Б (ИНС+GPS):    alpha_est   = theta_мес - gamma_gps,
                          gamma_gps   = arctan(Vh_earth / Vx_earth)

Физика ошибки при попутном порыве (Vwx > 0, ЛА снижается):
  GPS: Vx_earth = Vx_air + Vwx  >>  Vx_air
  → |gamma_gps| < |gamma_air| → alpha_est < alpha_true
  → АУА видит заниженный УА → задержка перехвата → ЛА входит в срыв.

Запуск:  python scenarios/s10_b_probe_vs_estimate.py
         python scenarios/s10_b_probe_vs_estimate.py results/s10b.gif
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
from sim.state  import THETA, Q, H, U, W, X
from runner import run, compute_trim, trim_state
from control.controllers import PitchController, PitchControlParams
from control.sensors import (measure_gyro, measure_altitude,
                              measure_airspeed, measure_angle_of_attack,
                              measure_gps_velocity_earth)
from control.estimators import estimate_alpha_indirect
from control.aua import AngleOfAttackProtector, AUAParams, AUAState

plt.rcParams["font.family"] = "DejaVu Sans"

# ══════════════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЦЕНАРИЯ
# ══════════════════════════════════════════════════════════════════════════════
aircraft = AircraftParams()
sp       = SensorParams()
cfg      = SimConfig(Va0=30.0, h0=200.0, dt=0.01, t_end=80.0)

H_REF    = cfg.h0
KH       = 0.06
GUST_VWX = 15.0
T_GUST   = 10.0
GUST_DUR = 30.0

wind_params = WindParams(gust_amp=GUST_VWX, gust_t0=T_GUST, gust_dur=GUST_DUR)

C_PROBE = "royalblue"
C_EST   = "tomato"

ANIM_SPEED = 6.0
ANIM_FPS   = 25

# ══════════════════════════════════════════════════════════════════════════════
#  БАЛАНСИРОВКА
# ══════════════════════════════════════════════════════════════════════════════
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

_warn_deg  = np.degrees(aircraft.alpha_warning)
_crit_deg  = np.degrees(aircraft.alpha_crit)
_stall_deg = np.degrees(aircraft.alpha_stall)

print(f"{'═'*62}")
print(f"С10-Б: Зонд vs косвенная оценка — что видит АУА при порыве")
print(f"Трим: alpha={np.degrees(alpha_trim):.1f}°  thr={thr_trim:.3f}")
print(f"Порыв: Vwx={GUST_VWX} м/с, t=[{T_GUST}, {T_GUST+GUST_DUR}] с")
print(f"Пороги: ПРЕД {_warn_deg:.0f}°  КРИТ {_crit_deg:.0f}°  СРЫВ {_stall_deg:.0f}°")
print(f"{'═'*62}")


# ══════════════════════════════════════════════════════════════════════════════
#  ФАБРИКА ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════
def _make_run(use_probe: bool):
    cp = PitchControlParams(Va_ref=cfg.Va0, gain_scheduling=True, h_Kp=0.0)
    controller = PitchController(aircraft, cp)
    controller.set_trim_throttle(thr_trim)
    controller.reset({"theta": s0[THETA], "q": 0.0, "h": H_REF})

    ap = AUAParams(
        enabled=True,
        alpha_warn=aircraft.alpha_warning,
        alpha_crit=aircraft.alpha_crit,
        alpha_exit=np.radians(8.0),
        theta_recovery=np.radians(-7.0),
        throttle_recovery=1.0,
        theta_warn_delta=np.radians(-3.0),
    )
    aua = AngleOfAttackProtector(aircraft, ap)

    # rng_common (seed=42): стандартные датчики — одинаковы в обоих прогонах
    # rng_probe  (seed=99): шум зонда          — только прогон А
    # rng_gps    (seed=77): шум GPS скорости   — только прогон Б (отдельный,
    #                        чтобы rng_common давал ту же последовательность)
    rng_common = np.random.default_rng(seed=42)
    rng_probe  = np.random.default_rng(seed=99)
    rng_gps    = np.random.default_rng(seed=77)

    aua_states    = []
    throttle_log  = []
    alpha_src_log = []

    def controls_fn(t, state, Va, alpha_true):
        q_meas     = measure_gyro(state[Q],     sp.gyro_bias,     sp.gyro_noise,     rng_common)
        h_meas     = measure_altitude(state[H], sp.baro_bias,     sp.baro_noise,     rng_common)
        theta_meas = state[THETA] + rng_common.normal(0.0, sp.gyro_noise)
        Va_meas    = measure_airspeed(Va,        sp.airspeed_bias, sp.airspeed_noise, rng_common)

        gust_active = T_GUST <= t <= T_GUST + GUST_DUR
        base_thr = 0.0 if gust_active else thr_trim

        h_err         = H_REF - h_meas
        theta_mission = np.clip(alpha_trim + KH * h_err,
                                np.radians(-20.0), np.radians(20.0))

        if use_probe:
            alpha_src = measure_angle_of_attack(
                alpha_true, sp.probe_bias, sp.probe_noise, rng_probe)
        else:
            Vx_gps, Vh_gps = measure_gps_velocity_earth(
                state[U], state[W], state[THETA],
                0.0, sp.gps_vel_noise, rng_gps)
            alpha_src = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)

        aua_out = aua.step(alpha_src, theta_mission, base_thr, cfg.dt)
        controller.set_pitch_setpoint(aua_out.theta_ref)
        controller.set_trim_throttle(aua_out.thr_trim)

        meas = {"q": q_meas, "theta": theta_meas, "h": h_meas, "Va": Va_meas}
        ctrl = controller.step(t, meas, cfg.dt)
        if aua_out.force_throttle is not None:
            ctrl[1] = aua_out.force_throttle

        aua_states.append(int(aua_out.state))
        throttle_log.append(ctrl[1])
        alpha_src_log.append(np.degrees(alpha_src))
        return ctrl

    return controls_fn, aua_states, throttle_log, alpha_src_log


# ══════════════════════════════════════════════════════════════════════════════
#  ПРОГОНЫ
# ══════════════════════════════════════════════════════════════════════════════
print("Запуск прогона А (зонд + АУА) ...")
fn_a, aua_st_a, thr_a_log, asrc_a = _make_run(use_probe=True)
log_a = run(fn_a, aircraft, wind_params, cfg, state0=s0)

print("Запуск прогона Б (оценка ИНС+GPS + АУА) ...")
fn_b, aua_st_b, thr_b_log, asrc_b = _make_run(use_probe=False)
log_b = run(fn_b, aircraft, wind_params, cfg, state0=s0)


# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════
n_a, n_b = len(log_a.t), len(log_b.t)

alpha_max_a = np.degrees(log_a.alpha.max())
alpha_max_b = np.degrees(log_b.alpha.max())
dh_a        = H_REF - log_a.state[:n_a, H].min()
dh_b        = H_REF - log_b.state[:n_b, H].min()
t_crit_a    = np.sum(log_a.alpha > aircraft.alpha_crit)  * cfg.dt
t_crit_b    = np.sum(log_b.alpha > aircraft.alpha_crit)  * cfg.dt
t_stall_a   = np.sum(log_a.alpha > aircraft.alpha_stall) * cfg.dt
t_stall_b   = np.sum(log_b.alpha > aircraft.alpha_stall) * cfg.dt

crit_idx_a = next((i for i, s in enumerate(aua_st_a[:n_a])
                   if s >= int(AUAState.CRITICAL)), None)
crit_idx_b = next((i for i, s in enumerate(aua_st_b[:n_b])
                   if s >= int(AUAState.CRITICAL)), None)
t_first_crit_a = log_a.t[crit_idx_a] if crit_idx_a is not None else None
t_first_crit_b = log_b.t[crit_idx_b] if crit_idx_b is not None else None

def _fmt(v, unit="с"):
    return f"{v:.1f}{unit}" if v is not None else "нет"

print(f"\n  Прогон А (зонд + АУА):")
print(f"    alpha_max        = {alpha_max_a:.1f}°")
print(f"    t > alpha_stall  = {t_stall_a:.1f} с")
print(f"    dh_min           = {dh_a:.1f} м")
print(f"    1й КРИТ АУА      = {_fmt(t_first_crit_a)}")
print(f"\n  Прогон Б (оценка ИНС+GPS + АУА):")
print(f"    alpha_max        = {alpha_max_b:.1f}°")
print(f"    t > alpha_stall  = {t_stall_b:.1f} с")
print(f"    dh_min           = {dh_b:.1f} м")
print(f"    1й КРИТ АУА      = {_fmt(t_first_crit_b)}")
if t_first_crit_a is not None and t_first_crit_b is not None:
    print(f"\n  Задержка перехвата: {t_first_crit_b - t_first_crit_a:+.1f} с")


# ══════════════════════════════════════════════════════════════════════════════
#  МАССИВЫ ДЛЯ АНИМАЦИИ
# ══════════════════════════════════════════════════════════════════════════════
t_a = log_a.t;  t_b = log_b.t
t_end_anim = max(t_a[-1], t_b[-1])
dt_sim = cfg.dt

h_a_arr   = log_a.state[:n_a, H]
h_b_arr   = log_b.state[:n_b, H]
x_a_arr   = log_a.state[:n_a, X]
x_b_arr   = log_b.state[:n_b, X]
Va_a_arr  = log_a.Va[:n_a]
Va_b_arr  = log_b.Va[:n_b]
al_a_deg  = np.degrees(log_a.alpha)
al_b_deg  = np.degrees(log_b.alpha)
thr_a_arr = np.array(thr_a_log[:n_a])
thr_b_arr = np.array(thr_b_log[:n_b])
aua_a_arr = np.array(aua_st_a[:n_a])
aua_b_arr = np.array(aua_st_b[:n_b])
src_a_arr = np.array(asrc_a[:n_a])
src_b_arr = np.array(asrc_b[:n_b])

_al_hi = max(alpha_max_a, alpha_max_b) + 4.0
_AUA_NAMES = {0: "НОРМ", 1: "ПРЕД", 2: "КРИТ", 3: "ВОССТ"}

# Прореживание кадров
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
n_total  = max(n_a, n_b)
idx_list = list(range(0, n_total, stride))
n_frames = len(idx_list)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА  (один файл = одна фигура, как s8)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 12))
fig.suptitle(
    f"С10-Б: Зонд (синий) vs ИНС+GPS (красный) — что видит АУА при попутном порыве\n"
    f"Vwx={GUST_VWX} м/с, t=[{T_GUST:.0f}, {T_GUST+GUST_DUR:.0f}] с  |  АУА включён в обоих",
    fontsize=11, fontweight="bold"
)

gs = gridspec.GridSpec(5, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.65, wspace=0.42)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_al   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_src  = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_aua  = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[4, 1], sharex=ax_h)


# ══════════════════════════════════════════════════════════════════════════════
#  ЛЕВЫЙ SUBPLOT: ТРАЕКТОРИЯ
# ══════════════════════════════════════════════════════════════════════════════
_all_x = np.concatenate([x_a_arr, x_b_arr])
_all_h = np.concatenate([h_a_arr, h_b_arr])
_px = max((_all_x.max() - _all_x.min()) * 0.06, 10.0)
_ph = max((_all_h.max() - _all_h.min()) * 0.30, 20.0)

ax_traj.set_xlim(_all_x.min() - _px, _all_x.max() + _px)
ax_traj.set_ylim(_all_h.min() - _ph, _all_h.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")   # 1 м по x = 1 м по h
ax_traj.set_xlabel("x, м", fontsize=9)
ax_traj.set_ylabel("h, м", fontsize=9)
ax_traj.set_title("Траектория в вертикальной плоскости", fontsize=9)
ax_traj.grid(True, ls="--", alpha=0.5)
ax_traj.axhline(H_REF, color="gray", lw=1.1, ls="--", alpha=0.6,
                label=f"h_ref={H_REF:.0f} м")

# Полные траектории (бледные — опорные)
ax_traj.plot(x_a_arr, h_a_arr, color=C_PROBE, lw=1.2, alpha=0.25, zorder=1)
ax_traj.plot(x_b_arr, h_b_arr, color=C_EST,   lw=1.2, alpha=0.25, zorder=1)

# Вертикальные линии по x-позиции начала и конца порыва (как на правых субплотах)
_ig0   = int(T_GUST / cfg.dt)
_ig1_a = min(int((T_GUST + GUST_DUR) / cfg.dt), n_a - 1)
_x_gust_start = x_a_arr[_ig0]
_x_gust_end   = x_a_arr[_ig1_a]
ax_traj.axvline(_x_gust_start, color="slateblue", lw=1.3, ls="--", alpha=0.7,
                label=f"Порыв Vwx={GUST_VWX:.0f} м/с")
ax_traj.axvline(_x_gust_end,   color="slateblue", lw=1.1, ls="--", alpha=0.5)

ax_traj.plot(x_a_arr[0],  h_a_arr[0],  "go", ms=8, zorder=5, label="Старт")
ax_traj.plot(x_a_arr[-1], h_a_arr[-1], color=C_PROBE,
             marker="s", ms=8, zorder=5, ls="", label=f"Финиш А (зонд, dh={dh_a:.0f}м)")
ax_traj.plot(x_b_arr[-1], h_b_arr[-1], color=C_EST,
             marker="^", ms=9, zorder=5, ls="", label=f"Финиш Б (ИНС+GPS, dh={dh_b:.0f}м)")
ax_traj.legend(fontsize=8, loc="upper left")

# Живые линии и маркеры (стиль s8: «b^»)
traj_line_a, = ax_traj.plot([], [], color=C_PROBE, lw=2.0, zorder=3)
traj_mark_a, = ax_traj.plot([], [], "^", color=C_PROBE,
                             ms=11, zorder=6, markeredgecolor="navy")
traj_line_b, = ax_traj.plot([], [], color=C_EST, lw=2.0, zorder=3)
traj_mark_b, = ax_traj.plot([], [], "^", color=C_EST,
                             ms=11, zorder=6, markeredgecolor="darkred")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes, fontsize=8.0,
    ha="right", va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="gray", alpha=0.88),
)


# ══════════════════════════════════════════════════════════════════════════════
#  ПРАВЫЕ SUBPLOTS
# ══════════════════════════════════════════════════════════════════════════════
right_axes = (ax_h, ax_al, ax_src, ax_aua, ax_thr)

_gust_lbl = f"Порыв Vwx={GUST_VWX:.0f} м/с"
for ax in right_axes:
    ax.set_xlim(0.0, t_end_anim)
    ax.grid(True, ls="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    # Зона порыва: закрашенная полоса + границы
    ax.axvspan(T_GUST, T_GUST + GUST_DUR,
               color="slateblue", alpha=0.10, zorder=0,
               label=_gust_lbl if ax is ax_h else None)
    ax.axvline(T_GUST,            color="slateblue", lw=1.3, ls="--", alpha=0.7)
    ax.axvline(T_GUST + GUST_DUR, color="slateblue", lw=1.1, ls="--", alpha=0.5)

# Подпись зоны порыва над первым субплотом
_t_gust_mid = (T_GUST + T_GUST + GUST_DUR) / 2.0
ax_h.text(_t_gust_mid / t_end_anim, 0.97,
          f"← порыв {GUST_VWX:.0f} м/с →",
          transform=ax_h.transAxes, ha="center", va="top",
          fontsize=7.5, color="slateblue", fontweight="bold")

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_thr.set_xlabel("Время, с", fontsize=9)

# Курсор времени
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65) for ax in right_axes]

# ── Высота ────────────────────────────────────────────────────────────────────
ax_h.set_ylabel("h, м", fontsize=9)
_h_pad = max((_all_h.max() - _all_h.min()) * 0.25, 8.0)
ax_h.set_ylim(_all_h.min() - _h_pad, _all_h.max() + _h_pad)
ax_h.axhline(H_REF, color="gray", lw=1.1, ls="--", alpha=0.55, label=f"h_ref={H_REF:.0f}")
ax_h.plot(t_a, h_a_arr, color=C_PROBE, lw=1.0, alpha=0.3)
ax_h.plot(t_b, h_b_arr, color=C_EST,   lw=1.0, alpha=0.3)
ax_h.legend(fontsize=7, loc="lower left")
ln_h_a, = ax_h.plot([], [], color=C_PROBE, lw=1.8, label="Зонд")
ln_h_b, = ax_h.plot([], [], color=C_EST,   lw=1.8, ls="--", label="ИНС+GPS")
pt_h_a, = ax_h.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_h_b, = ax_h.plot([], [], "o", color=C_EST,   ms=5, zorder=5)

# ── Истинный УА ───────────────────────────────────────────────────────────────
ax_al.set_ylabel("Истинный УА α, °", fontsize=9)
ax_al.set_ylim(-5.0, _al_hi)
ax_al.axhspan(_warn_deg, _crit_deg,  color="orange",  alpha=0.12, zorder=0)
ax_al.axhspan(_crit_deg, _stall_deg, color="red",     alpha=0.10, zorder=0)
ax_al.axhspan(_stall_deg, _al_hi+5,  color="darkred", alpha=0.07, zorder=0)
ax_al.axhline(_warn_deg,  color="orange",  lw=0.9, ls="--",
              label=f"ПРЕД {_warn_deg:.0f}°", alpha=0.8)
ax_al.axhline(_crit_deg,  color="red",     lw=0.9, ls="--",
              label=f"КРИТ {_crit_deg:.0f}°", alpha=0.8)
ax_al.axhline(_stall_deg, color="darkred", lw=1.1, ls="-",
              label=f"СРЫВ {_stall_deg:.0f}°", alpha=0.8)
ax_al.plot(t_a, al_a_deg, color=C_PROBE, lw=1.0, alpha=0.3)
ax_al.plot(t_b, al_b_deg, color=C_EST,   lw=1.0, alpha=0.3)
ax_al.legend(fontsize=7, loc="upper left", ncol=2)
ln_al_a, = ax_al.plot([], [], color=C_PROBE, lw=1.8, label="Зонд")
ln_al_b, = ax_al.plot([], [], color=C_EST,   lw=1.8, ls="--", label="ИНС+GPS")
pt_al,   = ax_al.plot([], [], "o", color="gray", ms=4, zorder=5)

# ── Вход в АУА (что видит защита) ─────────────────────────────────────────────
# Заголовок вынесен над субплотом — не перекрывает данные
ax_src.set_title(
    "При Vwx↑: Vx_gps↑ → gamma_gps↓ → alpha_ест < alpha_true → АУА срабатывает позже",
    fontsize=7, color=C_EST, pad=3,
)
ax_src.set_ylabel("Вход АУА, °", fontsize=9)
ax_src.set_ylim(-5.0, _al_hi)
ax_src.axhline(_warn_deg, color="orange", lw=0.9, ls="--", alpha=0.6)
ax_src.axhline(_crit_deg, color="red",    lw=0.9, ls="--", alpha=0.6)
ax_src.plot(t_a, al_a_deg,  color=C_PROBE, lw=1.0, ls="--", alpha=0.2)  # alpha_true А (фон)
ax_src.plot(t_a, src_a_arr, color=C_PROBE, lw=1.0, alpha=0.3)
ax_src.plot(t_b, al_b_deg,  color=C_EST,   lw=1.0, ls="--", alpha=0.2)  # alpha_true Б (фон)
ax_src.plot(t_b, src_b_arr, color=C_EST,   lw=1.0, alpha=0.3)
ln_src_a, = ax_src.plot([], [], color=C_PROBE, lw=1.8, label="alpha_зонд (А)")
ln_src_b, = ax_src.plot([], [], color=C_EST,   lw=1.8, ls="--", label="alpha_ест (Б)")
pt_src,   = ax_src.plot([], [], "o", color="gray", ms=4, zorder=5)
ax_src.legend(fontsize=7, loc="upper left")

# ── Состояние АУА ─────────────────────────────────────────────────────────────
ax_aua.set_ylabel("АУА состояние", fontsize=9)
ax_aua.set_ylim(-0.5, 3.5)
ax_aua.set_yticks([0, 1, 2, 3])
ax_aua.set_yticklabels(["НОРМ", "ПРЕД", "КРИТ", "ВОССТ"], fontsize=7)
for col, y_lo, y_hi in [("#e8f5e9", -0.5, 0.5), ("#fff9c4", 0.5, 1.5),
                          ("#ffe0b2", 1.5, 2.5), ("#e3f2fd", 2.5, 3.5)]:
    ax_aua.axhspan(y_lo, y_hi, color=col, alpha=0.5, zorder=0)
ax_aua.step(t_a, aua_a_arr,        color=C_PROBE, lw=1.0, where="post", alpha=0.3)
ax_aua.step(t_b, aua_b_arr + 0.12, color=C_EST,   lw=1.0, where="post", alpha=0.3)
ln_aua_a, = ax_aua.plot([], [], color=C_PROBE, lw=2.2,
                         drawstyle="steps-post", label="Зонд")
ln_aua_b, = ax_aua.plot([], [], color=C_EST,   lw=2.2,
                         drawstyle="steps-post", ls="--", label="ИНС+GPS")
ax_aua.legend(fontsize=7, loc="upper right")

# ── Тяга ──────────────────────────────────────────────────────────────────────
ax_thr.set_ylabel("Тяга δt, о.е.", fontsize=9)
ax_thr.set_ylim(-0.05, 1.05)
ax_thr.axhline(1.0, color="green", lw=0.8, ls=":", alpha=0.5)
ax_thr.axhline(0.0, color="gray",  lw=0.8, ls=":")
ax_thr.plot(t_a, thr_a_arr, color=C_PROBE, lw=1.0, alpha=0.3)
ax_thr.plot(t_b, thr_b_arr, color=C_EST,   lw=1.0, alpha=0.3)
ax_thr.legend(fontsize=7, loc="upper right")
ln_thr_a, = ax_thr.plot([], [], color=C_PROBE, lw=1.8, label="Зонд")
ln_thr_b, = ax_thr.plot([], [], color=C_EST,   lw=1.8, ls="--", label="ИНС+GPS")
pt_thr_a, = ax_thr.plot([], [], "o", color=C_PROBE, ms=4, zorder=5)
pt_thr_b, = ax_thr.plot([], [], "o", color=C_EST,   ms=4, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
#  ИТОГОВАЯ ТАБЛИЦА (вне графика)
# ══════════════════════════════════════════════════════════════════════════════
delay_str = ""
if t_first_crit_a is not None and t_first_crit_b is not None:
    delay_str = f"\n{'Δt КРИТ':>14}{t_first_crit_b - t_first_crit_a:>+10.1f}с"
summary = (
    f"{'':>14}{'Зонд (А)':>12}{'ИНС+GPS (Б)':>12}\n"
    f"{'alpha_max':>14}{alpha_max_a:>11.1f}°{alpha_max_b:>11.1f}°\n"
    f"{'t>alpha_crit':>14}{t_crit_a:>10.1f}с{t_crit_b:>10.1f}с\n"
    f"{'t>alpha_stall':>14}{t_stall_a:>10.1f}с{t_stall_b:>10.1f}с\n"
    f"{'dh_min':>14}{dh_a:>11.1f}м{dh_b:>11.1f}м\n"
    f"{'1й КРИТ АУА':>14}{_fmt(t_first_crit_a):>12}{_fmt(t_first_crit_b):>12}"
    + delay_str
)
fig.text(0.98, 0.005, summary, ha="right", va="bottom", fontsize=8,
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                   edgecolor="gray", alpha=0.9))


# ══════════════════════════════════════════════════════════════════════════════
#  АНИМАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
_all_artists = (
    traj_line_a, traj_mark_a,
    traj_line_b, traj_mark_b,
    info_box,
    ln_h_a, ln_h_b, pt_h_a, pt_h_b,
    ln_al_a, ln_al_b, pt_al,
    ln_src_a, ln_src_b, pt_src,
    ln_aua_a, ln_aua_b,
    ln_thr_a, ln_thr_b, pt_thr_a, pt_thr_b,
    *vlines,
)


def init():
    traj_line_a.set_data([], []);  traj_mark_a.set_data([], [])
    traj_line_b.set_data([], []);  traj_mark_b.set_data([], [])
    info_box.set_text("")
    for ln in (ln_h_a, ln_h_b, ln_al_a, ln_al_b,
               ln_src_a, ln_src_b, ln_aua_a, ln_aua_b,
               ln_thr_a, ln_thr_b):
        ln.set_data([], [])
    for pt in (pt_h_a, pt_h_b, pt_al, pt_src, pt_thr_a, pt_thr_b):
        pt.set_data([], [])
    for vl in vlines:
        vl.set_xdata([0])
    return _all_artists


def update(fn):
    raw = idx_list[fn]
    i_a = min(raw, n_a - 1)
    i_b = min(raw, n_b - 1)
    t_cur = t_a[i_a]
    ts_a  = t_a[:i_a + 1]
    ts_b  = t_b[:i_b + 1]

    # Траектории + маркеры
    traj_line_a.set_data(x_a_arr[:i_a + 1], h_a_arr[:i_a + 1])
    traj_mark_a.set_data([x_a_arr[i_a]], [h_a_arr[i_a]])
    traj_line_b.set_data(x_b_arr[:i_b + 1], h_b_arr[:i_b + 1])
    traj_mark_b.set_data([x_b_arr[i_b]], [h_b_arr[i_b]])

    # Инфо-блок
    info_box.set_text(
        f"t      = {t_cur:5.1f} с\n"
        f"── Зонд (А) ──────────\n"
        f"h      = {h_a_arr[i_a]:6.1f} м\n"
        f"Va     = {Va_a_arr[i_a]:5.1f} м/с\n"
        f"α_ист  = {al_a_deg[i_a]:5.1f}°\n"
        f"α_вход = {src_a_arr[i_a]:5.1f}°\n"
        f"АУА    = {_AUA_NAMES.get(aua_a_arr[i_a], '?')}\n"
        f"── ИНС+GPS (Б) ───────\n"
        f"h      = {h_b_arr[i_b]:6.1f} м\n"
        f"Va     = {Va_b_arr[i_b]:5.1f} м/с\n"
        f"α_ист  = {al_b_deg[i_b]:5.1f}°\n"
        f"α_ест  = {src_b_arr[i_b]:5.1f}°\n"
        f"АУА    = {_AUA_NAMES.get(aua_b_arr[i_b], '?')}"
    )

    # Высота
    ln_h_a.set_data(ts_a, h_a_arr[:i_a + 1])
    ln_h_b.set_data(ts_b, h_b_arr[:i_b + 1])
    pt_h_a.set_data([t_cur], [h_a_arr[i_a]])
    pt_h_b.set_data([t_b[i_b]], [h_b_arr[i_b]])

    # Истинный УА
    ln_al_a.set_data(ts_a, al_a_deg[:i_a + 1])
    ln_al_b.set_data(ts_b, al_b_deg[:i_b + 1])
    pt_al.set_data([t_cur], [al_a_deg[i_a]])

    # Вход АУА
    ln_src_a.set_data(ts_a, src_a_arr[:i_a + 1])
    ln_src_b.set_data(ts_b, src_b_arr[:i_b + 1])
    pt_src.set_data([t_cur], [src_a_arr[i_a]])

    # Состояние АУА
    ln_aua_a.set_data(ts_a, aua_a_arr[:i_a + 1])
    ln_aua_b.set_data(ts_b, aua_b_arr[:i_b + 1] + 0.12)

    # Тяга
    ln_thr_a.set_data(ts_a, thr_a_arr[:i_a + 1])
    ln_thr_b.set_data(ts_b, thr_b_arr[:i_b + 1])
    pt_thr_a.set_data([t_cur], [thr_a_arr[i_a]])
    pt_thr_b.set_data([t_b[i_b]], [thr_b_arr[i_b]])

    for vl in vlines:
        vl.set_xdata([t_cur])

    return _all_artists


anim = FuncAnimation(fig, update, frames=n_frames,
                     init_func=init, interval=1000.0 / ANIM_FPS, blit=True)

# ══════════════════════════════════════════════════════════════════════════════
#  СОХРАНЕНИЕ / ПОКАЗ
# ══════════════════════════════════════════════════════════════════════════════
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

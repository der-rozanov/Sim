# -*- coding: utf-8 -*-
"""
С10: АУА (Автомат Углов Атаки) vs. попутный порыв ветра.

Физический механизм:
  ЛА летит в режиме контроля высоты (P-контур h → theta).
  Попутный порыв Vwx=15 м/с снижает воздушную скорость:
    Va_eff = Va - Vwx ≈ 30 − 15 = 15 м/с ≈ Va_stall
  При этом лобовой поток на винт падает — пропеллер теряет тягу
  (модель Beard&McLain: T = 0.5·ρ·S·((k·δt)²−Va²); при попутном
  ветре Va_eff мала, но двигатель настроен на Va=30 → устанавливаем
  throttle=0 на время порыва для демонстрации предельного случая).

  Контур высоты видит снижение → theta_ref растёт до 20°.

  БЕЗ АУА: theta=20°, газ=0 → пропеллер тормозит (T<0),
           alpha растёт, срыв; ЛА теряет высоту.

  С АУА:   alpha ≥ alpha_crit → theta=−7°, газ=100%.
           Набегающий поток разгоняет Va — вывод из срыва за 4–6 с.

Запуск:  python scenarios/s10_aua_gust_protection.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state
from control.controllers import PitchController, PitchControlParams
from control.sensors import (measure_gyro, measure_altitude,
                              measure_airspeed, measure_angle_of_attack)
from control.aua import AngleOfAttackProtector, AUAParams, AUAOutput, AUAState
from sim.state import THETA, Q, H

plt.rcParams["font.family"] = "DejaVu Sans"

# ══════════════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЦЕНАРИЯ
# ══════════════════════════════════════════════════════════════════════════════
aircraft = AircraftParams()   # стандартный ЛА, Cma=-0.38
sp       = SensorParams()
cfg      = SimConfig(Va0=30.0, h0=200.0, dt=0.01, t_end=80.0)

H_REF = cfg.h0          # м, уставка высоты
KH    = 0.06            # рад/м, усиление P-контура по высоте

# Попутный порыв: снижает Va и убирает лобовой поток на винт
GUST_VWX = 15.0         # м/с, амплитуда
T_GUST   = 10.0         # с, начало
GUST_DUR = 30.0         # с, длительность (долго → ЛА без АУА теряет высоту)

wind_params = WindParams(
    gust_amp = GUST_VWX,
    gust_t0  = T_GUST,
    gust_dur = GUST_DUR,
)

C_AUA  = "royalblue"
C_NOUA = "tomato"

# ══════════════════════════════════════════════════════════════════════════════
#  БАЛАНСИРОВКА
# ══════════════════════════════════════════════════════════════════════════════
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

_warn_deg  = np.degrees(aircraft.alpha_warning)
_crit_deg  = np.degrees(aircraft.alpha_crit)
_stall_deg = np.degrees(aircraft.alpha_stall)

print(f"{'═'*60}")
print(f"С10: АУА против попутного порыва (Vwx={GUST_VWX} м/с)")
print(f"Трим: alpha={np.degrees(alpha_trim):.1f}°  thr={thr_trim:.3f}")
print(f"Порыв: t=[{T_GUST}, {T_GUST+GUST_DUR}] с  throttle→0 на время порыва")
print(f"Пороги: ПРЕД {_warn_deg:.0f}°  КРИТ {_crit_deg:.0f}°  СРЫВ {_stall_deg:.0f}°")
print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  ФАБРИКА ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════
def _make_run(aua_enabled: bool):
    # h_Kp=0: тягой по высоте не управляем — только theta.
    # Это создаёт порочный круг при срыве: снижение → theta растёт → alpha → ещё больше снижение
    cp = PitchControlParams(Va_ref=cfg.Va0, gain_scheduling=True, h_Kp=0.0)
    controller = PitchController(aircraft, cp)
    controller.set_trim_throttle(thr_trim)
    controller.reset({"theta": s0[THETA], "q": 0.0, "h": H_REF})

    ap = AUAParams(
        enabled           = aua_enabled,
        alpha_warn        = aircraft.alpha_warning,
        alpha_crit        = aircraft.alpha_crit,
        alpha_exit        = np.radians(8.0),
        theta_recovery    = np.radians(-7.0),
        throttle_recovery = 1.0,
        theta_warn_delta  = np.radians(-3.0),
    )
    aua = AngleOfAttackProtector(aircraft, ap)

    rng_common = np.random.default_rng(seed=42)
    rng_probe  = np.random.default_rng(seed=99)

    aua_states    = []
    theta_ref_log = []
    throttle_log  = []

    def controls_fn(t, state, Va, alpha_true):
        q_meas     = measure_gyro(state[Q],     sp.gyro_bias,     sp.gyro_noise,     rng_common)
        h_meas     = measure_altitude(state[H], sp.baro_bias,     sp.baro_noise,     rng_common)
        theta_meas = state[THETA] + rng_common.normal(0.0, sp.gyro_noise)
        Va_meas    = measure_airspeed(Va,        sp.airspeed_bias, sp.airspeed_noise, rng_common)

        # Во время порыва — газ в ноль (лобовой поток на винт пропал)
        gust_active = T_GUST <= t <= T_GUST + GUST_DUR
        base_thr = 0.0 if gust_active else thr_trim

        # P-контур по высоте: снижение → theta растёт (без АУА → порочный круг)
        h_err         = H_REF - h_meas
        theta_mission = np.clip(alpha_trim + KH * h_err,
                                np.radians(-20.0), np.radians(20.0))

        if aua_enabled:
            alpha_probe = measure_angle_of_attack(
                alpha_true, sp.probe_bias, sp.probe_noise, rng_probe)
            aua_out = aua.step(alpha_probe, theta_mission, base_thr, cfg.dt)
        else:
            aua_out = AUAOutput(theta_mission, base_thr, None, AUAState.NORMAL)

        controller.set_pitch_setpoint(aua_out.theta_ref)
        controller.set_trim_throttle(aua_out.thr_trim)

        meas = {"q": q_meas, "theta": theta_meas, "h": h_meas, "Va": Va_meas}
        ctrl = controller.step(t, meas, cfg.dt)

        if aua_out.force_throttle is not None:
            ctrl[1] = aua_out.force_throttle

        aua_states.append(int(aua_out.state))
        theta_ref_log.append(np.degrees(aua_out.theta_ref))
        throttle_log.append(ctrl[1])
        return ctrl

    return controls_fn, aua_states, theta_ref_log, throttle_log


# ══════════════════════════════════════════════════════════════════════════════
#  ПРОГОНЫ
# ══════════════════════════════════════════════════════════════════════════════
print("Запуск прогона А (с АУА) ...")
fn_a, aua_st_a, tref_a, thr_a_log = _make_run(aua_enabled=True)
log_a = run(fn_a, aircraft, wind_params, cfg, state0=s0)

print("Запуск прогона Б (без АУА) ...")
fn_b, aua_st_b, tref_b, thr_b_log = _make_run(aua_enabled=False)
log_b = run(fn_b, aircraft, wind_params, cfg, state0=s0)

# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════
n_a, n_b = len(log_a.t), len(log_b.t)
alpha_max_a = np.degrees(log_a.alpha.max())
alpha_max_b = np.degrees(log_b.alpha.max())
dh_a = H_REF - log_a.state[:n_a, H].min()
dh_b = H_REF - log_b.state[:n_b, H].min()
t_crit_a = np.sum(log_a.alpha > aircraft.alpha_crit) * cfg.dt
t_crit_b = np.sum(log_b.alpha > aircraft.alpha_crit) * cfg.dt
aua_s    = np.sum(np.array(aua_st_a[:n_a]) >= int(AUAState.CRITICAL)) * cfg.dt

print(f"\n  Прогон А (с АУА):")
print(f"    alpha_max       = {alpha_max_a:.1f}°")
print(f"    t > alpha_crit  = {t_crit_a:.1f} с")
print(f"    dh_min          = {dh_a:.1f} м")
print(f"    АУА перехват    = {aua_s:.1f} с")
print(f"\n  Прогон Б (без АУА):")
print(f"    alpha_max       = {alpha_max_b:.1f}°")
print(f"    t > alpha_crit  = {t_crit_b:.1f} с")
print(f"    dh_min          = {dh_b:.1f} м")


# ══════════════════════════════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
t_full   = np.arange(0, cfg.t_end, cfg.dt)
vwx_arr  = np.where((t_full >= T_GUST) & (t_full <= T_GUST + GUST_DUR), GUST_VWX, 0.0)

t_a = log_a.t;  t_b = log_b.t
alpha_a_deg = np.degrees(log_a.alpha)
alpha_b_deg = np.degrees(log_b.alpha)
h_a = log_a.state[:n_a, H]
h_b = log_b.state[:n_b, H]
thr_a = np.array(thr_a_log[:n_a])
thr_b = np.array(thr_b_log[:n_b])
aua_arr  = np.array(aua_st_a[:n_a])
tref_arr = np.array(tref_a[:n_a])

fig = plt.figure(figsize=(12, 14))
fig.suptitle(
    "С10: АУА против попутного порыва\n"
    f"Vwx={GUST_VWX} м/с, t=[{T_GUST:.0f}, {T_GUST+GUST_DUR:.0f}] с  |  "
    f"КН высоты: theta_ref = alpha_trim + {KH}·h_err",
    fontsize=11, fontweight="bold"
)
gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.52)
ax_h   = fig.add_subplot(gs[0])
ax_al  = fig.add_subplot(gs[1], sharex=ax_h)
ax_thr = fig.add_subplot(gs[2], sharex=ax_h)
ax_aua = fig.add_subplot(gs[3], sharex=ax_h)
ax_wnd = fig.add_subplot(gs[4], sharex=ax_h)

for ax in [ax_h, ax_al, ax_thr, ax_aua, ax_wnd]:
    ax.set_xlim(0.0, cfg.t_end)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    ax.axvline(T_GUST,            color="dimgray", lw=1.2, ls=":", alpha=0.7)
    ax.axvline(T_GUST + GUST_DUR, color="dimgray", lw=1.0, ls=":", alpha=0.5)

for ax in [ax_h, ax_al, ax_thr, ax_aua]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_wnd.set_xlabel("Время, с", fontsize=9)

# ── Высота ───────────────────────────────────────────────────────────────────
ax_h.set_ylabel("Высота h, м", fontsize=9)
ax_h.axhline(H_REF, color="gray", lw=1.0, ls="--", alpha=0.5, label=f"h_ref={H_REF:.0f} м")
ax_h.plot(t_a, h_a, color=C_AUA,  lw=2.0, label=f"С АУА  (dh={dh_a:.0f} м)")
ax_h.plot(t_b, h_b, color=C_NOUA, lw=2.0, label=f"Без АУА (dh={dh_b:.0f} м)")
ax_h.legend(fontsize=8, loc="lower left")

# ── УА ───────────────────────────────────────────────────────────────────────
ax_al.set_ylabel("Угол атаки α, °", fontsize=9)
_al_hi = max(alpha_max_a, alpha_max_b) + 5.0
ax_al.axhspan(_warn_deg,  _crit_deg, color="orange", alpha=0.12, zorder=0)
ax_al.axhspan(_crit_deg, _al_hi + 5, color="red",    alpha=0.10, zorder=0)
ax_al.axhline(_warn_deg,  color="orange",  lw=1.1, ls="--", label=f"ПРЕД {_warn_deg:.0f}°")
ax_al.axhline(_crit_deg,  color="red",     lw=1.1, ls="--", label=f"КРИТ {_crit_deg:.0f}°")
ax_al.axhline(_stall_deg, color="darkred", lw=1.3, ls="-",  label=f"СРЫВ {_stall_deg:.0f}°")
ax_al.plot(t_a, alpha_a_deg, color=C_AUA,  lw=2.0, label=f"С АУА  (t_crit={t_crit_a:.0f}с)")
ax_al.plot(t_b, alpha_b_deg, color=C_NOUA, lw=2.0, label=f"Без АУА (t_crit={t_crit_b:.0f}с)")
ax_al.set_ylim(-5.0, _al_hi)
ax_al.legend(fontsize=7, loc="upper left", ncol=2)

# ── Тяга ─────────────────────────────────────────────────────────────────────
ax_thr.set_ylabel("Тяга δt, о.е.", fontsize=9)
ax_thr.set_ylim(-0.05, 1.05)
ax_thr.axhline(1.0, color="green",  lw=0.8, ls=":", alpha=0.5)
ax_thr.axhline(0.0, color="gray",   lw=0.8, ls=":")
ax_thr.axvspan(T_GUST, T_GUST + GUST_DUR, color="wheat", alpha=0.3, label="газ=0 (порыв)")
ax_thr.plot(t_a, thr_a, color=C_AUA,  lw=2.0, label="С АУА")
ax_thr.plot(t_b, thr_b, color=C_NOUA, lw=2.0, alpha=0.8, label="Без АУА")
ax_thr.legend(fontsize=8)

# ── Состояние АУА ────────────────────────────────────────────────────────────
ax_aua.set_ylabel("Состояние АУА\n(Прогон А)", fontsize=9)
ax_aua.set_ylim(-0.5, 3.5)
ax_aua.set_yticks([0, 1, 2, 3])
ax_aua.set_yticklabels(["НОРМ", "ПРЕД", "КРИТ", "ВОССТ"], fontsize=8)
for col, y_lo, y_hi in [("#e8f5e9", -0.5, 0.5), ("#fff9c4", 0.5, 1.5),
                          ("#ffe0b2", 1.5, 2.5), ("#e3f2fd", 2.5, 3.5)]:
    ax_aua.axhspan(y_lo, y_hi, color=col, alpha=0.5, zorder=0)
ax_aua.step(t_a, aua_arr, color=C_AUA, lw=2.0, where="post", zorder=3)

# ── Ветер ─────────────────────────────────────────────────────────────────────
ax_wnd.set_ylabel("Ветер Vwx, м/с\n(попутный)", fontsize=9)
ax_wnd.fill_between(t_full, 0, vwx_arr, color="slateblue", alpha=0.35)
ax_wnd.plot(t_full, vwx_arr, color="slateblue", lw=1.5)
ax_wnd.set_ylim(-1.0, GUST_VWX * 1.4)
ax_wnd.axhline(0, color="gray", lw=0.8, ls=":")

# ── Итоговая таблица ─────────────────────────────────────────────────────────
summary = (
    f"{'':>14}{'С АУА':>12}{'Без АУА':>12}\n"
    f"{'alpha_max':>14}{alpha_max_a:>11.1f}°{alpha_max_b:>11.1f}°\n"
    f"{'t>alpha_crit':>14}{t_crit_a:>10.1f}с{t_crit_b:>10.1f}с\n"
    f"{'dh_min':>14}{dh_a:>11.1f}м{dh_b:>11.1f}м"
)
fig.text(0.98, 0.01, summary, ha="right", va="bottom", fontsize=8.5,
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                   edgecolor="gray", alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 1])

save_path = sys.argv[1] if len(sys.argv) > 1 else None
if save_path is not None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Сохранено: {save_path}")

plt.show()

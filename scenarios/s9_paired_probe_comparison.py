# -*- coding: utf-8 -*-
"""
С9: Парный прогон «с зондом vs без зонда» — кульминационный результат.

Два независимых прогона идентичны во всём, кроме источника УА:
  «С зондом»:  alpha_src = measure_angle_of_attack()  — прямое измерение
  «Без зонда»: alpha_src = estimate_alpha_indirect()  — ИНС+GPS (theta − gamma_gps)

Альтитудный контроллер в обоих прогонах:
    theta_ref = alpha_src + KH * (h_ref − h_meas)

Честность сравнения:
  Общие датчики (гироскоп, барометр, СВС, GPS) — rng_common с одним seed.
  Порядок и число вызовов rng_common в обоих прогонах ОДИНАКОВЫ.
  Шум зонда — rng_probe с отдельным seed (только в прогоне «с зондом»).

Метрики:
  J_E    = ∫ throttle dt            [о.е.·с]  — суммарные затраты энергии
  J_h    = RMS(h − h_ref)           [м]        — точность отработки высоты
  J_safe = max(alpha − alpha_warn,0)[°]        — выход на предкрит. УА
  J_eff  = ∫ Va dt / J_E            [м/о.е.]  — ключевая интегральная характеристика:
           «дальность на единицу израсходованной энергии двигателя»

Запуск:  python scenarios/s9_paired_probe_comparison.py
         python scenarios/s9_paired_probe_comparison.py results/s9.gif
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
from control.controllers import (PitchController, PitchControlParams,
                                  SpeedController, SpeedControlParams)
from control.sensors import (measure_gyro, measure_altitude, measure_airspeed,
                              measure_angle_of_attack, measure_gps_velocity_earth)
from control.estimators import estimate_alpha_indirect
from sim.state import THETA, Q, H, X, U, W
from flight_logger import FlightLogger

plt.rcParams["font.family"] = "DejaVu Sans"

# ══════════════════════════════════════════════════════════════════════════════
#  ПРЕСЕТЫ УСЛОВИЙ
#  Один активен; остальные закомментированы.
#  Для смены: раскомментируй нужный блок, закомментируй активный.
# ══════════════════════════════════════════════════════════════════════════════

# ─── ПРЕСЕТ 1 (АКТИВЕН): Постоянный встречный ветер ─────────────────────────
# Главный демонстрационный сценарий.
# Механизм ошибки при встречном ветре (Vwx < 0) и снижении:
#   Vx_earth = Va·cos(γ) + Vwx  →  Vx_earth < Vx_air
#   gamma_gps = atan(Vh / Vx_earth) → КРУЧЕ, чем реальный gamma_air
#   alpha_est = theta − gamma_gps  → ПЕРЕОЦЕНКА alpha (alpha_est > alpha_true)
# Результат в контуре: theta_ref_est > theta_ref_probe → ЛА «без зонда» медленнее
# снижается, за то же время накапливает большую ошибку высоты и тратит больше тяги.
# Уставка задаётся РАМПОЙ (не ступенькой) — h_err мал на всём снижении, клип
# theta_ref = ±20° не включается, и разница alpha_src всегда видна.
PRESET_NAME = "Встречный ветер, рамп-снижение"
aircraft    = AircraftParams()
wind_params = WindParams(Vw_const=-5.0)   # м/с, встречный ветер (минус = встречный)
sp  = SensorParams()
cfg = SimConfig(Va0=30.0, h0=150.0, dt=0.01, t_end=90.0)

# ─── ПРЕСЕТ 2 (закомментирован): Встречный сдвиг ветра по высоте ─────────────
# То же, что Пресет 1, но встречный ветер УСИЛИВАЕТСЯ с высотой.
# При снижении через слой ошибка alpha_est убывает вместе с ветром →
# наглядно показывает связь «сила ветра — точность оценки УА».
# PRESET_NAME = "Встречный сдвиг ветра"
# wind_params = WindParams(
#     Vw_const   = -2.0,
#     h_shear_lo = 100.0,
#     h_shear_hi = 150.0,
#     dV_shear   = -6.0,   # → на H_HIGH=150м: Vwx = −8 м/с
# )
# cfg = SimConfig(Va0=30.0, h0=150.0, dt=0.01, t_end=90.0)

# ─── ПРЕСЕТ 3 (закомментирован): Резкий порыв в середине снижения ────────────
# Прямоугольный порыв 7 м/с на 15 с — ступенчатый сюрприз.
# До/после порыва обе CAУ работают одинаково; в момент порыва без-зонд теряет высоту.
# PRESET_NAME = "Порыв при снижении"
# wind_params = WindParams(gust_amp=-7.0, gust_t0=35.0, gust_dur=15.0)
# cfg = SimConfig(Va0=30.0, h0=150.0, dt=0.01, t_end=90.0)

# ─── ПРЕСЕТ 4 (закомментирован): Штиль — контрольный прогон ─────────────────
# Без ветра: alpha_est ≈ alpha_true → обе CAУ должны дать J_E ≈ J_E, J_h ≈ J_h.
# Разница — только шум зонда. Используй для проверки честности сравнения.
# PRESET_NAME = "Штиль (контрольный)"
# wind_params = WindParams()
# cfg = SimConfig(Va0=30.0, h0=150.0, dt=0.01, t_end=90.0)

# ══════════════════════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЦЕНАРИЯ (общие для всех прогонов)
# ══════════════════════════════════════════════════════════════════════════════
H_HIGH    = 150.0   # м, стартовая высота
H_LOW     = 100.0   # м, целевая высота
VA_REF    = 30.0    # м/с

# Расписание уставки высоты:
#   0 … T_HOLD_HI  — удержание H_HIGH      (уровень, ветер, alpha_est ≈ alpha_true)
#   T_HOLD_HI … T_RAMP_END — ЛИНЕЙНЫЙ РАМП H_HIGH → H_LOW  (ошибка alpha_est видна!)
#   T_RAMP_END … t_end — удержание H_LOW   (уровень, оба метода сходятся)
#
# Рамп (а не ступенька) — ключевое условие: h_err всегда мал, клип theta_ref
# не включается и разница alpha_src (0.6° vs 0.6°+bias) постоянно присутствует в контуре.
T_HOLD_HI  = 10.0   # с
T_RAMP_END = 70.0   # с  (рамп длится 60 с, скорость снижения ≈ 0.83 м/с)

KH        = 0.006   # рад/м, усиление контура высоты
THETA_CLIP = 0.349  # рад = 20°, клип theta_ref (расширен vs стандартных 15°,
                    # чтобы клип не маскировал разницу alpha_src при малом h_err)

ANIM_SPEED = 2.0    # кратность ускорения анимации
ANIM_FPS   = 25

C_PROBE = "royalblue"   # цвет «с зондом» (единообразно во всех субплотах)
C_EST   = "tomato"      # цвет «без зонда»

# ══════════════════════════════════════════════════════════════════════════════
#  БАЛАНСИРОВКА
# ══════════════════════════════════════════════════════════════════════════════
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, cfg.Va0)
s0 = trim_state(aircraft, cfg)

Vw_high = wind_params.Vw_const + wind_params.dV_shear   # ветер на H_HIGH
Vw_low  = wind_params.Vw_const                           # ветер на H_LOW

print(f"{'═'*64}")
print(f"С9  Парный прогон  |  Пресет: {PRESET_NAME}")
print(f"Трим: α={np.degrees(alpha_trim):.2f}°  δe={np.degrees(de_trim):.2f}°  thr={thr_trim:.3f}")
print(f"Ветер: Vwx@{H_HIGH:.0f}м = {Vw_high:+.1f} м/с  "
      f"Vwx@{H_LOW:.0f}м = {Vw_low:+.1f} м/с")
print(f"{'═'*64}")


def _h_ref_schedule(t: float) -> float:
    """Уставка высоты: удержание H_HIGH → линейный рамп → удержание H_LOW."""
    if t < T_HOLD_HI:
        return H_HIGH
    if t < T_RAMP_END:
        frac = (t - T_HOLD_HI) / (T_RAMP_END - T_HOLD_HI)
        return H_HIGH - frac * (H_HIGH - H_LOW)
    return H_LOW


# ══════════════════════════════════════════════════════════════════════════════
#  ФАБРИКА РЕГУЛЯТОРА
#  Возвращает (controls_fn, data_buf).
#  alpha_mode = "probe" | "est"
#  Единственное различие двух прогонов — строка выбора alpha_src.
# ══════════════════════════════════════════════════════════════════════════════
def _make_controller(alpha_mode: str):
    ctrl = PitchController(aircraft, PitchControlParams(Va_ref=cfg.Va0))
    ctrl.set_trim_throttle(thr_trim)
    ctrl.reset({"theta": s0[THETA], "q": 0.0, "h": H_HIGH})

    spd = SpeedController(aircraft, SpeedControlParams())
    spd.set_trim_throttle(thr_trim)
    spd.set_Va_ref(VA_REF)
    spd.reset()

    # Общий шум: одинаковый seed → одна и та же реализация для обоих прогонов
    rng_common = np.random.default_rng(seed=42)
    # Шум зонда: независимый seed — используется только в режиме "probe"
    rng_probe  = np.random.default_rng(seed=99)

    buf = {"h_ref": [], "alpha_src": [], "theta_ref": []}

    def controls_fn(t, state, Va, alpha):
        h_ref = _h_ref_schedule(t)

        # ── Общие датчики ──────────────────────────────────────────────────────
        # ВАЖНО: порядок и число вызовов rng_common одинаковы в обоих режимах.
        # Это гарантирует идентичность шума общих датчиков → честное сравнение.
        h_meas     = measure_altitude(state[H], sp.baro_bias, sp.baro_noise,   rng_common)
        q_meas     = measure_gyro(state[Q],    sp.gyro_bias, sp.gyro_noise,    rng_common)
        theta_meas = state[THETA] + rng_common.normal(0.0, sp.gyro_noise)
        Va_meas    = measure_airspeed(Va,      sp.airspeed_bias, sp.airspeed_noise, rng_common)
        Vx_gps, Vh_gps = measure_gps_velocity_earth(
            state[U], state[W], state[THETA], 0.0, sp.gps_vel_noise, rng_common)

        # ── Источник УА (единственное различие двух прогонов) ─────────────────
        if alpha_mode == "probe":
            alpha_src = measure_angle_of_attack(
                alpha, sp.probe_bias, sp.probe_noise, rng_probe)
        else:  # "est"
            alpha_src = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)

        # ── Контур высоты: theta_ref = alpha_src + KH·h_err ──────────────────
        # Физический смысл: theta = alpha + gamma, поэтому gamma = KH·h_err.
        # При ошибке alpha_src (ветровой bias) theta_ref смещается → h не удерживается.
        h_err     = h_ref - h_meas
        theta_ref = np.clip(alpha_src + KH * h_err, -THETA_CLIP, THETA_CLIP)
        ctrl.set_pitch_setpoint(theta_ref)

        meas     = {"q": q_meas, "theta": theta_meas, "h": h_meas, "Va": Va_meas}
        delta_e  = ctrl.step(t, meas, cfg.dt)[0]
        throttle = spd.step(Va_meas, cfg.dt)

        buf["h_ref"].append(h_ref)
        buf["alpha_src"].append(float(alpha_src))
        buf["theta_ref"].append(float(theta_ref))

        return np.array([delta_e, throttle])

    return controls_fn, buf


# ══════════════════════════════════════════════════════════════════════════════
#  ДВА ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════
fn_probe, buf_probe = _make_controller("probe")
fn_est,   buf_est   = _make_controller("est")

print("Прогон 1/2: с зондом…",  flush=True)
log_probe = run(fn_probe, aircraft, wind_params, cfg, state0=s0)
print_summary(log_probe, aircraft, label="С9 | С ЗОНДОМ")

print("Прогон 2/2: без зонда…", flush=True)
log_est = run(fn_est, aircraft, wind_params, cfg, state0=s0)
print_summary(log_est, aircraft, label="С9 | БЕЗ ЗОНДА")

# ══════════════════════════════════════════════════════════════════════════════
#  ПОДГОТОВКА МАССИВОВ
# ══════════════════════════════════════════════════════════════════════════════
n      = min(len(log_probe.t), len(log_est.t))
t_arr  = log_probe.t[:n]
dt_sim = float(t_arr[1] - t_arr[0])

h_p   = log_probe.state[:n, H];    h_e   = log_est.state[:n, H]
x_p   = log_probe.state[:n, X];    x_e   = log_est.state[:n, X]
thr_p = log_probe.controls[:n, 1]; thr_e = log_est.controls[:n, 1]
de_p  = log_probe.controls[:n, 0]; de_e  = log_est.controls[:n, 0]
Va_p  = log_probe.Va[:n];          Va_e  = log_est.Va[:n]
al_p  = log_probe.alpha[:n];       al_e  = log_est.alpha[:n]
th_p  = np.degrees(log_probe.state[:n, THETA])
th_e  = np.degrees(log_est.state[:n, THETA])

h_ref_arr  = np.array(buf_probe["h_ref"][:n])   # расписание одинаково
asrc_p_deg = np.degrees(np.array(buf_probe["alpha_src"][:n]))
asrc_e_deg = np.degrees(np.array(buf_est["alpha_src"][:n]))
tref_p_deg = np.degrees(np.array(buf_probe["theta_ref"][:n]))
tref_e_deg = np.degrees(np.array(buf_est["theta_ref"][:n]))
al_p_deg   = np.degrees(al_p)
al_e_deg   = np.degrees(al_e)

h_err_p = h_p - h_ref_arr
h_err_e = h_e - h_ref_arr

E_cum_p = np.cumsum(thr_p) * dt_sim
E_cum_e = np.cumsum(thr_e) * dt_sim

# ══════════════════════════════════════════════════════════════════════════════
#  МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════
# Фаза удержания H_HIGH: t < T_HOLD_HI
mask_hold    = t_arr <  T_HOLD_HI
# Фаза рампа + финальное удержание: t ≥ T_HOLD_HI
mask_descent = t_arr >= T_HOLD_HI

# J_E: интеграл дросселя за весь полёт и по фазам [о.е.·с]
J_E_p      = float(np.trapz(thr_p, t_arr))
J_E_e      = float(np.trapz(thr_e, t_arr))
J_E_p_hold = float(np.trapz(thr_p[mask_hold], t_arr[mask_hold]))
J_E_e_hold = float(np.trapz(thr_e[mask_hold], t_arr[mask_hold]))
J_E_p_desc = float(np.trapz(thr_p[mask_descent], t_arr[mask_descent]))
J_E_e_desc = float(np.trapz(thr_e[mask_descent], t_arr[mask_descent]))

# J_h: СКО ошибки высоты по фазе снижения [м]
J_h_p = float(np.sqrt(np.mean(h_err_p[mask_descent]**2)))
J_h_e = float(np.sqrt(np.mean(h_err_e[mask_descent]**2)))

# J_safe: максимальный выход на предкритический УА [°]
J_safe_p = float(np.degrees(np.max(np.maximum(0.0, al_p - aircraft.alpha_warning))))
J_safe_e = float(np.degrees(np.max(np.maximum(0.0, al_e - aircraft.alpha_warning))))

# J_eff: дальность на единицу энергии [м / (о.е.·с)]  ← ключевая интегральная характеристика
# Физический смысл: сколько метров ЛА пролетает на одну единицу дроссельного ресурса.
# При Va ≈ Va_ref (SpeedController удерживает скорость) это прямой КПД маршрута.
# Формула: J_eff = ∫Va dt / ∫throttle dt = пройденный путь / затраченная энергия
J_eff_p = float(np.trapz(Va_p, t_arr)) / J_E_p
J_eff_e = float(np.trapz(Va_e, t_arr)) / J_E_e

# ── Таблица метрик ─────────────────────────────────────────────────────────
def _delta_str(probe, est, lower_better=True):
    if abs(est) < 1e-9:
        return "     —"
    d = 100.0 * (est - probe) / abs(est)
    if not lower_better:
        d = -d
    return f"{d:+5.1f}%"

print(f"\n{'═'*70}")
print(f"МЕТРИКИ  |  Пресет: {PRESET_NAME}")
print(f"{'─'*70}")
print(f"{'Метрика':<34}{'С зондом':>12}{'Без зонда':>12}{'Δ лучше':>10}")
print(f"{'─'*70}")
print(f"{'J_E  [∫thr dt] весь полёт, о.е.·с':<34}"
      f"{J_E_p:>12.2f}{J_E_e:>12.2f}"
      f"{_delta_str(J_E_p, J_E_e):>10}")
print(f"{'  из них: фаза удержания':<34}"
      f"{J_E_p_hold:>12.2f}{J_E_e_hold:>12.2f}"
      f"{_delta_str(J_E_p_hold, J_E_e_hold):>10}")
print(f"{'  из них: фаза снижения':<34}"
      f"{J_E_p_desc:>12.2f}{J_E_e_desc:>12.2f}"
      f"{_delta_str(J_E_p_desc, J_E_e_desc):>10}")
print(f"{'J_h  [СКО h-h_ref, снижение], м':<34}"
      f"{J_h_p:>12.3f}{J_h_e:>12.3f}"
      f"{_delta_str(J_h_p, J_h_e):>10}")
print(f"{'J_safe [max α-αwarn, °]':<34}"
      f"{J_safe_p:>12.3f}{J_safe_e:>12.3f}"
      f"{'':>10}")
print(f"{'J_eff  [∫Va dt / J_E, м/о.е.·с]':<34}"
      f"{J_eff_p:>12.1f}{J_eff_e:>12.1f}"
      f"{_delta_str(J_eff_p, J_eff_e, lower_better=False):>10}")
print(f"{'  ↑ ключевая интегральная хар-ка'}")
print(f"{'═'*70}")

# Смещение оценки УА — поясняет механизм разницы в J_h
alpha_bias_probe = float(np.mean(asrc_p_deg[mask_descent] - al_p_deg[mask_descent]))
alpha_bias_est   = float(np.mean(asrc_e_deg[mask_descent] - al_e_deg[mask_descent]))
print("\nМеханизм: смещение alpha_src vs alpha_true (фаза рампа)")
print(f"  Зонд    : {alpha_bias_probe:+.3f}°  (≈ 0 + шум зонда — ожидаемо)")
print(f"  ИНС+GPS : {alpha_bias_est:+.3f}°  (ветровая погрешность: gamma_gps ≠ gamma_air)")
print(f"  Разность: {alpha_bias_est - alpha_bias_probe:+.3f}°  →  theta_ref_est смещён")
print(f"  Следствие: ЛА без зонда медленнее следует за рампой → J_h хуже на "
      f"{100*(J_h_e - J_h_p)/J_h_e:.1f}%")
if abs(J_E_p - J_E_e) / max(J_E_p, 1e-9) < 0.005:
    print("  J_E совпадает (SpeedController удерживает Va=const → тяга ≈ trim): "
          "точность достигнута БЕЗ дополнительных затрат энергии.")

# ══════════════════════════════════════════════════════════════════════════════
#  АНИМАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
stride   = max(1, int(round(ANIM_SPEED / ANIM_FPS / dt_sim)))
idx_list = list(range(0, n, stride))
n_frames = len(idx_list)

t_end = float(t_arr[-1])

# ── Компоновка фигуры ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 15))
fig.suptitle(
    f"С9: Парный прогон «с зондом vs без зонда»  |  {PRESET_NAME}\n"
    f"Vw@{H_HIGH:.0f}м={Vw_high:+.1f}м/с → Vw@{H_LOW:.0f}м={Vw_low:+.1f}м/с  |  "
    f"J_E: зонд={J_E_p:.1f}  без={J_E_e:.1f}  "
    f"J_eff: зонд={J_eff_p:.0f}  без={J_eff_e:.0f} м/о.е.",
    fontsize=10, fontweight="bold",
)

gs = gridspec.GridSpec(6, 2, figure=fig,
                       width_ratios=[1.4, 1],
                       hspace=0.75, wspace=0.45)

ax_traj = fig.add_subplot(gs[:, 0])
ax_h    = fig.add_subplot(gs[0, 1])
ax_al   = fig.add_subplot(gs[1, 1], sharex=ax_h)
ax_thr  = fig.add_subplot(gs[2, 1], sharex=ax_h)
ax_E    = fig.add_subplot(gs[3, 1], sharex=ax_h)
ax_he   = fig.add_subplot(gs[4, 1], sharex=ax_h)
ax_Va   = fig.add_subplot(gs[5, 1], sharex=ax_h)

right_axes = (ax_h, ax_al, ax_thr, ax_E, ax_he, ax_Va)

# ── Левый субплот: траектория ─────────────────────────────────────────────
x_all = np.concatenate([x_p, x_e])
h_all = np.concatenate([h_p, h_e])
_px = max((x_all.max() - x_all.min()) * 0.06, 10.0)
_ph = max((h_all.max() - h_all.min()) * 0.30, 20.0)
ax_traj.set_xlim(x_all.min() - _px, x_all.max() + _px)
ax_traj.set_ylim(h_all.min() - _ph, h_all.max() + _ph)
ax_traj.set_aspect("equal", adjustable="datalim")
ax_traj.set_xlabel("x, м", fontsize=9)
ax_traj.set_ylabel("h, м", fontsize=9)
ax_traj.set_title("Траектория", fontsize=9)
ax_traj.grid(True, ls="--", alpha=0.5)

# Уставки высоты
ax_traj.axhline(H_HIGH, color="gray", lw=1.0, ls="--", alpha=0.55,
                label=f"h_ref: {H_HIGH:.0f}/{H_LOW:.0f}м")
ax_traj.axhline(H_LOW,  color="gray", lw=1.0, ls="--", alpha=0.55)

# Слой сдвига ветра (если есть)
if wind_params.dV_shear != 0.0:
    ax_traj.axhspan(wind_params.h_shear_lo, wind_params.h_shear_hi,
                    color="lightyellow", alpha=0.55, zorder=0, label="слой сдвига ветра")

# Полные пути (фоновые линии)
ax_traj.plot(x_p, h_p, color=C_PROBE, lw=1.0, alpha=0.25)
ax_traj.plot(x_e, h_e, color=C_EST,   lw=1.0, alpha=0.25)
ax_traj.plot(x_p[0], h_p[0], "o", color="green",  ms=9, zorder=6)
ax_traj.plot(x_p[-1], h_p[-1], "s", color="black", ms=8, zorder=6)

ax_traj.legend(fontsize=8, loc="upper left")

traj_ln_p, = ax_traj.plot([], [], color=C_PROBE, lw=2.0, label="с зондом",   zorder=4)
traj_ln_e, = ax_traj.plot([], [], color=C_EST,   lw=2.0, label="без зонда",  zorder=4,
                           ls="--")
traj_mk_p, = ax_traj.plot([], [], "^", color=C_PROBE, ms=11, zorder=7,
                           markeredgecolor="navy")
traj_mk_e, = ax_traj.plot([], [], "v", color=C_EST,   ms=11, zorder=7,
                           markeredgecolor="darkred")
ax_traj.legend(fontsize=8, loc="upper left")

info_box = ax_traj.text(
    0.98, 0.04, "",
    transform=ax_traj.transAxes, fontsize=7.5,
    ha="right", va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.90),
)

# Метрики — статичная сводка в левом нижнем углу
metrics_summary = (
    f"J_E   зонд={J_E_p:.1f}  без={J_E_e:.1f}\n"
    f"J_h   зонд={J_h_p:.2f}м без={J_h_e:.2f}м\n"
    f"J_eff зонд={J_eff_p:.0f} без={J_eff_e:.0f} м/о.е."
)
ax_traj.text(
    0.02, 0.04, metrics_summary,
    transform=ax_traj.transAxes, fontsize=7.5,
    ha="left", va="bottom", family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
              edgecolor="goldenrod", alpha=0.90),
)

# ── Правые субплоты: общая настройка ─────────────────────────────────────
for ax in right_axes:
    ax.set_xlim(0.0, t_end)
    ax.grid(True, ls="--", alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.axvline(T_HOLD_HI,  color="darkorange", lw=0.9, ls=":", alpha=0.8)
    ax.axvline(T_RAMP_END, color="steelblue",  lw=0.9, ls=":", alpha=0.8)

for ax in right_axes[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_Va.set_xlabel("Время, с", fontsize=9)

# Метки событий (только на верхнем субплоте)
ax_h.axvline(T_HOLD_HI,  color="darkorange", lw=0.9, ls=":", alpha=0.8,
             label=f"рамп t={T_HOLD_HI:.0f}с")
ax_h.axvline(T_RAMP_END, color="steelblue",  lw=0.9, ls=":", alpha=0.8,
             label=f"удерж. t={T_RAMP_END:.0f}с")

# ── h ────────────────────────────────────────────────────────────────────
ax_h.set_ylabel("h, м", fontsize=9)
_hpad = max((h_all.max() - h_all.min()) * 0.20, 5.0)
ax_h.set_ylim(h_all.min() - _hpad, h_all.max() + _hpad)
ax_h.plot(t_arr, h_ref_arr, "k--", lw=1.1, alpha=0.55, label="h_ref")
ax_h.plot(t_arr, h_p, color=C_PROBE, lw=1.0, alpha=0.3)
ax_h.plot(t_arr, h_e, color=C_EST,   lw=1.0, alpha=0.3)
ax_h.legend(fontsize=7, loc="upper right")
ln_h_p, = ax_h.plot([], [], color=C_PROBE, lw=1.8, label="зонд")
ln_h_e, = ax_h.plot([], [], color=C_EST,   lw=1.8, ls="--", label="без зонда")
pt_h_p, = ax_h.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_h_e, = ax_h.plot([], [], "o", color=C_EST,   ms=5, zorder=5)
ax_h.legend(fontsize=7, loc="upper right")

# ── alpha_src (что видит каждый контроллер) ──────────────────────────────
ax_al.set_ylabel("alpha_src, °", fontsize=9)
_al_all = np.concatenate([asrc_p_deg, asrc_e_deg, al_p_deg])
_alpad  = max((_al_all.max() - _al_all.min()) * 0.20, 0.5)
ax_al.set_ylim(_al_all.min() - _alpad, _al_all.max() + _alpad)
ax_al.axhline(np.degrees(alpha_trim), color="gray", lw=0.8, ls=":")
ax_al.axhline(np.degrees(aircraft.alpha_warning), color="orangered",
              lw=0.7, ls=":", alpha=0.6, label=f"αwarn")
ax_al.plot(t_arr, al_p_deg,   "silver", lw=1.1, alpha=0.55, label="α истинный")
ax_al.plot(t_arr, asrc_p_deg, color=C_PROBE, lw=0.9, alpha=0.3)
ax_al.plot(t_arr, asrc_e_deg, color=C_EST,   lw=0.9, alpha=0.3)
ax_al.legend(fontsize=7, loc="upper right")
ln_al_true, = ax_al.plot([], [], "silver", lw=1.5)
ln_al_p,    = ax_al.plot([], [], color=C_PROBE, lw=1.8, label="зонд")
ln_al_e,    = ax_al.plot([], [], color=C_EST,   lw=1.8, ls="--", label="ИНС+GPS")
pt_al,      = ax_al.plot([], [], "o", color="gray", ms=5, zorder=5)
ax_al.legend(fontsize=7, loc="upper right")

# ── Тяга ─────────────────────────────────────────────────────────────────
ax_thr.set_ylabel("Тяга, о.е.", fontsize=9)
_thr_all = np.concatenate([thr_p, thr_e])
ax_thr.set_ylim(max(0.0, _thr_all.min() - 0.05), min(1.05, _thr_all.max() + 0.05))
ax_thr.axhline(thr_trim, color="black", lw=1.0, ls="--", alpha=0.4,
               label=f"thr_trim={thr_trim:.2f}")
ax_thr.plot(t_arr, thr_p, color=C_PROBE, lw=1.0, alpha=0.3)
ax_thr.plot(t_arr, thr_e, color=C_EST,   lw=1.0, alpha=0.3)
ax_thr.legend(fontsize=7, loc="upper right")
ln_thr_p, = ax_thr.plot([], [], color=C_PROBE, lw=1.8)
ln_thr_e, = ax_thr.plot([], [], color=C_EST,   lw=1.8, ls="--")
pt_thr_p, = ax_thr.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_thr_e, = ax_thr.plot([], [], "o", color=C_EST,   ms=5, zorder=5)

# ── Накопленная энергия (J_E нарастающий итог) ───────────────────────────
ax_E.set_ylabel("J_E нараст., о.е.", fontsize=9)
ax_E.set_ylim(0, max(E_cum_p.max(), E_cum_e.max()) * 1.08)
ax_E.plot(t_arr, E_cum_p, color=C_PROBE, lw=1.0, alpha=0.3)
ax_E.plot(t_arr, E_cum_e, color=C_EST,   lw=1.0, alpha=0.3)
ax_E.set_title(
    f"∫thr dt  зонд={J_E_p:.1f}  без={J_E_e:.1f}  "
    f"Δ={_delta_str(J_E_p, J_E_e).strip()}",
    fontsize=8, pad=2)
ln_E_p, = ax_E.plot([], [], color=C_PROBE, lw=1.8, label="зонд")
ln_E_e, = ax_E.plot([], [], color=C_EST,   lw=1.8, ls="--", label="без зонда")
pt_E_p, = ax_E.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_E_e, = ax_E.plot([], [], "o", color=C_EST,   ms=5, zorder=5)
ax_E.legend(fontsize=7, loc="upper left")

# ── Ошибка высоты ─────────────────────────────────────────────────────────
ax_he.set_ylabel("Δh = h−h_ref, м", fontsize=9)
_he_all = np.concatenate([h_err_p, h_err_e])
_hepad  = max(abs(_he_all).max() * 0.20, 0.5)
ax_he.set_ylim(_he_all.min() - _hepad, _he_all.max() + _hepad)
ax_he.axhline(0, color="gray", lw=0.8, ls=":")
ax_he.plot(t_arr, h_err_p, color=C_PROBE, lw=1.0, alpha=0.3)
ax_he.plot(t_arr, h_err_e, color=C_EST,   lw=1.0, alpha=0.3)
ax_he.set_title(
    f"J_h снижение:  зонд={J_h_p:.2f}м  без={J_h_e:.2f}м  "
    f"Δ={_delta_str(J_h_p, J_h_e).strip()}",
    fontsize=8, pad=2)
ln_he_p, = ax_he.plot([], [], color=C_PROBE, lw=1.8)
ln_he_e, = ax_he.plot([], [], color=C_EST,   lw=1.8, ls="--")
pt_he_p, = ax_he.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_he_e, = ax_he.plot([], [], "o", color=C_EST,   ms=5, zorder=5)

# ── Воздушная скорость ────────────────────────────────────────────────────
ax_Va.set_ylabel("Va, м/с", fontsize=9)
_Va_all = np.concatenate([Va_p, Va_e])
ax_Va.set_ylim(min(_Va_all.min(), VA_REF) - 1.0, max(_Va_all.max(), VA_REF) + 1.0)
ax_Va.axhline(VA_REF, color="black", lw=1.1, ls="--", alpha=0.55,
              label=f"ref={VA_REF:.0f}")
ax_Va.plot(t_arr, Va_p, color=C_PROBE, lw=1.0, alpha=0.3)
ax_Va.plot(t_arr, Va_e, color=C_EST,   lw=1.0, alpha=0.3)
ax_Va.legend(fontsize=7, loc="upper right")
ln_Va_p, = ax_Va.plot([], [], color=C_PROBE, lw=1.8)
ln_Va_e, = ax_Va.plot([], [], color=C_EST,   lw=1.8, ls="--")
pt_Va_p, = ax_Va.plot([], [], "o", color=C_PROBE, ms=5, zorder=5)
pt_Va_e, = ax_Va.plot([], [], "o", color=C_EST,   ms=5, zorder=5)

# Курсоры времени
vlines = [ax.axvline(0, color="gray", lw=0.8, ls=":", alpha=0.65) for ax in right_axes]

# ══════════════════════════════════════════════════════════════════════════════
#  АНИМАЦИЯ: init / update
# ══════════════════════════════════════════════════════════════════════════════
_all_artists = (
    traj_ln_p, traj_ln_e, traj_mk_p, traj_mk_e, info_box,
    ln_h_p, ln_h_e, pt_h_p, pt_h_e,
    ln_al_true, ln_al_p, ln_al_e, pt_al,
    ln_thr_p, ln_thr_e, pt_thr_p, pt_thr_e,
    ln_E_p, ln_E_e, pt_E_p, pt_E_e,
    ln_he_p, ln_he_e, pt_he_p, pt_he_e,
    ln_Va_p, ln_Va_e, pt_Va_p, pt_Va_e,
    *vlines,
)


def init():
    for ln in (traj_ln_p, traj_ln_e,
               ln_h_p, ln_h_e, ln_al_true, ln_al_p, ln_al_e,
               ln_thr_p, ln_thr_e, ln_E_p, ln_E_e,
               ln_he_p, ln_he_e, ln_Va_p, ln_Va_e):
        ln.set_data([], [])
    for pt in (traj_mk_p, traj_mk_e,
               pt_h_p, pt_h_e, pt_al,
               pt_thr_p, pt_thr_e, pt_E_p, pt_E_e,
               pt_he_p, pt_he_e, pt_Va_p, pt_Va_e):
        pt.set_data([], [])
    info_box.set_text("")
    for vl in vlines:
        vl.set_xdata([0])
    return _all_artists


def update(fn):
    i   = idx_list[fn]
    t_c = t_arr[i]
    ts  = t_arr[:i+1]

    # Траектория
    traj_ln_p.set_data(x_p[:i+1], h_p[:i+1])
    traj_ln_e.set_data(x_e[:i+1], h_e[:i+1])
    traj_mk_p.set_data([x_p[i]], [h_p[i]])
    traj_mk_e.set_data([x_e[i]], [h_e[i]])

    # Инфобокс
    info_box.set_text(
        f"t       = {t_c:5.1f} с\n"
        f"───── зонд ─────\n"
        f"h       = {h_p[i]:6.1f} м  ref={h_ref_arr[i]:.0f}\n"
        f"Va      = {Va_p[i]:5.1f} м/с\n"
        f"αsrc    = {asrc_p_deg[i]:+5.2f}°  thr={thr_p[i]:.3f}\n"
        f"J_E     = {E_cum_p[i]:6.1f}\n"
        f"───── без зонда ─\n"
        f"h       = {h_e[i]:6.1f} м\n"
        f"Va      = {Va_e[i]:5.1f} м/с\n"
        f"αsrc    = {asrc_e_deg[i]:+5.2f}°  thr={thr_e[i]:.3f}\n"
        f"J_E     = {E_cum_e[i]:6.1f}"
    )

    # Субплоты
    ln_h_p.set_data(ts, h_p[:i+1]);       pt_h_p.set_data([t_c], [h_p[i]])
    ln_h_e.set_data(ts, h_e[:i+1]);       pt_h_e.set_data([t_c], [h_e[i]])

    ln_al_true.set_data(ts, al_p_deg[:i+1])
    ln_al_p.set_data(ts, asrc_p_deg[:i+1]); pt_al.set_data([t_c], [al_p_deg[i]])
    ln_al_e.set_data(ts, asrc_e_deg[:i+1])

    ln_thr_p.set_data(ts, thr_p[:i+1]);   pt_thr_p.set_data([t_c], [thr_p[i]])
    ln_thr_e.set_data(ts, thr_e[:i+1]);   pt_thr_e.set_data([t_c], [thr_e[i]])

    ln_E_p.set_data(ts, E_cum_p[:i+1]);   pt_E_p.set_data([t_c], [E_cum_p[i]])
    ln_E_e.set_data(ts, E_cum_e[:i+1]);   pt_E_e.set_data([t_c], [E_cum_e[i]])

    ln_he_p.set_data(ts, h_err_p[:i+1]);  pt_he_p.set_data([t_c], [h_err_p[i]])
    ln_he_e.set_data(ts, h_err_e[:i+1]);  pt_he_e.set_data([t_c], [h_err_e[i]])

    ln_Va_p.set_data(ts, Va_p[:i+1]);     pt_Va_p.set_data([t_c], [Va_p[i]])
    ln_Va_e.set_data(ts, Va_e[:i+1]);     pt_Va_e.set_data([t_c], [Va_e[i]])

    for vl in vlines:
        vl.set_xdata([t_c])

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
        except Exception as exc:
            print(f"Ошибка GIF: {exc}")
    else:
        try:
            anim.save(save_path, fps=ANIM_FPS, extra_args=["-vcodec", "libx264"])
            print(f"Сохранено: {save_path}")
        except Exception as exc:
            print(f"Ошибка MP4: {exc}")

plt.show()

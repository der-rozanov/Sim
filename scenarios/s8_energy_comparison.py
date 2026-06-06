# -*- coding: utf-8 -*-
"""
С8: Сравнение энергопотребления — три скорости × три режима оценки УА.

Три рабочие точки на аэродинамической поляре:
  Va_LOW  = 20 м/с  — alpha≈12°, индуктивное сопр. доминирует (у предупр. УА)
  Va_MID  = 27 м/с  — alpha≈4°,  умеренный крейс
  Va_HIGH = 35 м/с  — alpha≈0.5°, паразитное сопр. доминирует

Три режима оценки УА (одинаковые ПИД, ветер, шум общих датчиков):
  normal   — нет обратной связи по alpha (как С6)
  probe    — прямое измерение зондом → точная α-ОС
  indirect — ИНС+GPS → задержанная/смещённая α-ОС при ветре

Закон управления (одинаков для probe и indirect):
  theta_ref = alpha_trim + KH * h_err + K_ALPHA * (alpha_ref - alpha_meas)
  При точном alpha_meas (probe): быстрое гашение α-возмущений от турбулентности.
  При ошибочном alpha_meas (indirect): задержанная/неверная коррекция → хуже.
  При mode='normal': K_ALPHA=0 → нет α-ОС вообще.

Метрика: E = ∫ throttle(t) * Va(t) dt  (нормированная мощность двигателя).

Запуск:
  python scenarios/s8_energy_comparison.py
  python scenarios/s8_energy_comparison.py results/s8.png
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import run, compute_trim, trim_state
from control import (PitchController, PitchControlParams,
                     SpeedController, SpeedControlParams)
from sensors import (measure_gyro, measure_altitude, measure_airspeed,
                     measure_angle_of_attack)
from estimators import estimate_alpha_indirect
from state import THETA, Q, H, U, W

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Общие параметры
# ------------------------------------------------------------------
aircraft = AircraftParams()
sp       = SensorParams()

WIND = WindParams(
    Vw_const  = 7.0,    # попутный ветер, м/с
    turb_amp  = 4.5,    # вертикальная турбулентность, м/с
    turb_freq = 0.08,   # Гц (период ~12 с, быстрее → сложнее компенсировать)
)

H_REF   = 100.0   # удерживаемая высота, м
KH      = 0.006   # рад/м — P-контур высоты
K_ALPHA = 1.0     # рад/рад — усиление α-ОС: при точном зонде гасит срыв
T_END   = 90.0    # с, длительность каждого прогона
DT      = 0.01    # с

# Три рабочих точки
SPEEDS = [
    dict(Va=20.0, label='Va=20 м/с\n(α≈12°, инд. сопр.)',  color='tomato'),
    dict(Va=27.0, label='Va=27 м/с\n(α≈4°, крейс)',         color='seagreen'),
    dict(Va=35.0, label='Va=35 м/с\n(α≈0.5°, парз. сопр.)', color='steelblue'),
]

MODES = ['normal', 'probe', 'indirect']
MODE_LABELS = {
    'normal':   'Обычный\n(без α-ОС)',
    'probe':    'С зондом\n(прямое α)',
    'indirect': 'Без зонда\n(ИНС+GPS)',
}
MODE_COLORS = {'normal': 'gray', 'probe': 'seagreen', 'indirect': 'tomato'}

# ------------------------------------------------------------------
# Функция одного прогона
# ------------------------------------------------------------------
def run_one(Va_ref: float, mode: str, seed_shared: int = 42) -> dict:
    """
    Запустить один прогон при заданной Va_ref и режиме оценки УА.
    Возвращает словарь с массивами t, throttle, Va, alpha, h, E_cum.
    """
    cfg = SimConfig(Va0=Va_ref, h0=H_REF, theta0=0.0, dt=DT, t_end=T_END)

    alpha_trim, _de_trim, thr_trim = compute_trim(aircraft, Va_ref)
    s0 = trim_state(aircraft, cfg)
    # Поправка: trim_state задаёт u,w как компоненты воздушной скорости (без ветра).
    # При Vwx≠0 нужны земные скорости: u_earth = Va*cos(α) + Vwx*cos(θ), etc.
    ct, st = np.cos(alpha_trim), np.sin(alpha_trim)
    s0[U] += WIND.Vw_const * ct
    s0[W] += WIND.Vw_const * st

    N = int(round(T_END / DT)) + 20
    rng_sh = np.random.default_rng(seed_shared)
    rng_pr = np.random.default_rng(seed_shared + 1000)

    noise_q     = rng_sh.normal(0, sp.gyro_noise,      N)
    noise_theta = rng_sh.normal(0, sp.gyro_noise,      N)
    noise_baro  = rng_sh.normal(0, sp.baro_noise,      N)
    noise_Va    = rng_sh.normal(0, sp.airspeed_noise,  N)
    noise_gps_x = rng_sh.normal(0, sp.gps_vel_noise,   N)
    noise_gps_h = rng_sh.normal(0, sp.gps_vel_noise,   N)
    noise_probe = rng_pr.normal(0, sp.probe_noise,      N)

    ctrl_params = PitchControlParams(Va_ref=Va_ref)
    controller  = PitchController(aircraft, ctrl_params)
    controller.set_trim_throttle(thr_trim)
    controller.reset({'theta': s0[THETA], 'q': 0.0, 'h': H_REF})

    spd_params = SpeedControlParams()
    spd_ctrl   = SpeedController(aircraft, spd_params)
    spd_ctrl.set_trim_throttle(thr_trim)
    spd_ctrl.set_Va_ref(Va_ref)
    spd_ctrl.reset()

    step = [0]

    def controls_fn(t, state, Va, alpha):
        i = step[0]
        step[0] += 1

        q_meas     = state[Q]     + sp.gyro_bias  + noise_q[i]
        theta_meas = state[THETA] +                 noise_theta[i]
        h_meas     = state[H]     + sp.baro_bias  + noise_baro[i]
        Va_meas    = max(1.0, Va  + sp.airspeed_bias + noise_Va[i])

        # Оценка УА
        if mode == 'normal':
            alpha_meas = None
        elif mode == 'probe':
            alpha_meas = alpha + sp.probe_bias + noise_probe[i]
        else:  # indirect
            ct, st   = np.cos(state[THETA]), np.sin(state[THETA])
            Vx = state[U] * ct - state[W] * st + noise_gps_x[i]
            Vh = state[U] * st - state[W] * ct + noise_gps_h[i]
            alpha_meas = estimate_alpha_indirect(theta_meas, Vx, Vh)

        # Закон управления
        h_err      = H_REF - h_meas
        theta_base = alpha_trim + KH * h_err

        if mode == 'normal':
            theta_ref = theta_base
        else:
            alpha_err = alpha_trim - alpha_meas          # > 0 когда α занижен
            theta_ref = theta_base + K_ALPHA * alpha_err

        theta_ref = np.clip(theta_ref, np.radians(-15.0), np.radians(20.0))
        controller.set_pitch_setpoint(theta_ref)

        meas     = {'q': q_meas, 'theta': theta_meas, 'h': h_meas, 'Va': Va_meas}
        delta_e  = controller.step(t, meas, DT)[0]
        throttle = spd_ctrl.step(Va_meas, DT)

        return np.array([delta_e, throttle])

    log = run(controls_fn, aircraft, WIND, cfg, state0=s0)
    dt  = log.t[1] - log.t[0]
    P   = log.controls[:, 1] * log.Va
    E   = np.cumsum(P) * dt

    return dict(
        t       = log.t,
        throttle= log.controls[:, 1],
        Va      = log.Va,
        alpha   = log.alpha,
        h       = log.state[:, H],
        E_cum   = E,
        E_total = E[-1],
        alpha_trim = alpha_trim,
        thr_trim   = thr_trim,
    )


# ------------------------------------------------------------------
# 9 прогонов
# ------------------------------------------------------------------
print("Запуск 9 прогонов (3 скорости × 3 режима)...\n")
results = {}   # results[Va][mode]

for sp_cfg in SPEEDS:
    Va = sp_cfg['Va']
    results[Va] = {}
    alpha_tr, _, thr_tr = compute_trim(aircraft, Va)
    CL = aircraft.CL0 + aircraft.CLa * alpha_tr
    AR = aircraft.b**2 / aircraft.S
    CD = aircraft.CDp + CL**2 / (np.pi * aircraft.e_oswald * AR)
    P_trim = 0.5 * aircraft.rho * Va**3 * aircraft.S * CD
    print(f"Va={Va:.0f} м/с: alpha_trim={np.degrees(alpha_tr):+.1f}°  "
          f"CL={CL:.3f}  CD={CD:.4f}  CL/CD={CL/CD:.1f}  P≈{P_trim:.0f} Вт")
    for mode in MODES:
        res = run_one(Va, mode, seed_shared=42)
        results[Va][mode] = res
        print(f"  [{mode:8s}]  E={res['E_total']:7.1f}  "
              f"h_fin={res['h'][-1]:.1f} м  "
              f"Va_min={res['Va'].min():.1f} м/с")
    # Разница probe vs indirect
    E_p = results[Va]['probe']['E_total']
    E_i = results[Va]['indirect']['E_total']
    print(f"  → без зонда vs зонд: {100*(E_i-E_p)/E_p:+.2f}%\n")

# ------------------------------------------------------------------
# Поляра (аналитически)
# ------------------------------------------------------------------
alpha_range = np.linspace(-3, 25, 300)
alpha_rad   = np.radians(alpha_range)
AR = aircraft.b**2 / aircraft.S
CL_lin = aircraft.CL0 + aircraft.CLa * alpha_rad
# Sigmoid-blending (упрощённо: клипируем где формула ломается)
CL_arr = np.clip(CL_lin, -0.5, 2.0)
CD_arr = aircraft.CDp + CL_arr**2 / (np.pi * aircraft.e_oswald * AR)
ratio  = CL_arr / CD_arr

# ------------------------------------------------------------------
# Графики
# ------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    "С8: Сравнение энергопотребления — три режима × три рабочих точки\n"
    f"Ветер Vwx={WIND.Vw_const:+.0f} м/с  +  турбулентность "
    f"{WIND.turb_amp:.0f} м/с @ {WIND.turb_freq:.2f} Гц",
    fontsize=11, fontweight="bold"
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)
ax_polar = fig.add_subplot(gs[0, 0])
ax_ratio = fig.add_subplot(gs[1, 0])
ax_bars  = [fig.add_subplot(gs[0, i+1]) for i in range(2)]
ax_ts    = [fig.add_subplot(gs[1, i+1]) for i in range(2)]

# ---- Аэродинамическая поляра ----
ax_polar.plot(CD_arr, CL_arr, 'k-', lw=1.5, label='CL(CD)')
ax_polar.axhline(0, color='gray', lw=0.7, ls=':')
for sp_cfg in SPEEDS:
    Va = sp_cfg['Va']
    res = results[Va]['normal']
    at  = res['alpha_trim']
    CL_ = aircraft.CL0 + aircraft.CLa * at
    CD_ = aircraft.CDp + CL_**2 / (np.pi * aircraft.e_oswald * AR)
    ax_polar.plot(CD_, CL_, 'o', ms=10, color=sp_cfg['color'],
                  label=f"Va={Va:.0f}")
    ax_polar.annotate(f"Va={Va:.0f}\nα={np.degrees(at):.1f}°",
                      xy=(CD_, CL_), xytext=(CD_+0.002, CL_+0.03),
                      fontsize=7.5, color=sp_cfg['color'])
# Линия от нуля к точке max CL/CD (приблизительно)
idx_opt = np.argmax(ratio)
ax_polar.plot([0, CD_arr[idx_opt]], [0, CL_arr[idx_opt]],
              'b--', lw=0.9, alpha=0.6, label='касательная max CL/CD')
ax_polar.set_xlabel("CD", fontsize=9)
ax_polar.set_ylabel("CL", fontsize=9)
ax_polar.set_title("Аэродинамическая поляра", fontsize=9)
ax_polar.legend(fontsize=7, loc='upper left')
ax_polar.grid(True, ls='--', alpha=0.4)
ax_polar.set_xlim(left=0)

# ---- CL/CD vs alpha ----
ax_ratio.plot(alpha_range, ratio, 'k-', lw=1.5)
for sp_cfg in SPEEDS:
    Va   = sp_cfg['Va']
    at   = results[Va]['normal']['alpha_trim']
    CL_  = aircraft.CL0 + aircraft.CLa * at
    CD_  = aircraft.CDp + CL_**2 / (np.pi * aircraft.e_oswald * AR)
    ax_ratio.axvline(np.degrees(at), color=sp_cfg['color'],
                     lw=1.3, ls='--', alpha=0.8, label=f"Va={Va:.0f}")
ax_ratio.axvline(np.degrees(aircraft.alpha_warning), color='orange',
                 lw=1.0, ls=':', alpha=0.8, label='α_warn')
ax_ratio.axvline(np.degrees(aircraft.alpha_crit),    color='red',
                 lw=1.0, ls=':', alpha=0.8, label='α_crit')
ax_ratio.set_xlabel("α, °", fontsize=9)
ax_ratio.set_ylabel("CL/CD", fontsize=9)
ax_ratio.set_title("Аэродинамическое качество", fontsize=9)
ax_ratio.legend(fontsize=7)
ax_ratio.grid(True, ls='--', alpha=0.4)
ax_ratio.set_xlim(-3, 25)

# ---- Бар-диаграммы энергии (два правых верхних subplot) ----
Va_list = [sp['Va'] for sp in SPEEDS]
mode_order = ['normal', 'probe', 'indirect']
x_pos = np.arange(len(mode_order))
width = 0.22

for ax_idx, (sp_cfg, Va) in enumerate(zip(SPEEDS, Va_list)):
    # Первые два Va на первом subplot, третий на втором
    # Перекомпоновка: все три Va на ax_bars[0], TS для Va_LOW на ax_bars[1]
    pass

# Перерисую как одну большую группированную бар-диаграмму
ax_bars[0].set_visible(False)
ax_bars[1].set_visible(False)

# Уберём старые ax_bars и создадим один широкий (займём обе ячейки)
ax_energy = fig.add_subplot(gs[0, 1:])

n_speeds = len(SPEEDS)
n_modes  = len(MODES)
group_w  = 0.7
bar_w    = group_w / n_modes
x_groups = np.arange(n_speeds)

for m_idx, mode in enumerate(mode_order):
    x_off = (m_idx - n_modes/2 + 0.5) * bar_w
    energies = [results[Va][mode]['E_total'] for Va in Va_list]
    bars = ax_energy.bar(x_groups + x_off, energies,
                         width=bar_w * 0.92,
                         color=MODE_COLORS[mode],
                         alpha=0.80,
                         edgecolor='gray', linewidth=0.7,
                         label=MODE_LABELS[mode].replace('\n', ' '))
    for bar, val, Va in zip(bars, energies, Va_list):
        E_ref = results[Va]['normal']['E_total']
        pct   = 100 * (val - E_ref) / E_ref
        lbl   = f"{val:.0f}" if abs(pct) < 0.05 else f"{val:.0f}\n({pct:+.1f}%)"
        ax_energy.text(bar.get_x() + bar.get_width()/2,
                       val + max(energies)*0.01,
                       lbl, ha='center', va='bottom',
                       fontsize=7)

ax_energy.set_xticks(x_groups)
ax_energy.set_xticklabels([sp['label'] for sp in SPEEDS], fontsize=8)
ax_energy.set_ylabel("E итоговая, о.е.", fontsize=9)
ax_energy.set_title("Суммарное энергопотребление ∫ throttle·Va dt — три рабочих точки",
                     fontsize=9)
ax_energy.legend(fontsize=8, loc='upper left')
ax_energy.grid(axis='y', ls='--', alpha=0.4)

# ---- Временные ряды (нижние два правых): тяга и E(t) для Va_LOW ----
Va_show = SPEEDS[0]['Va']   # Va=20 — самый интересный случай
res_show = {mode: results[Va_show][mode] for mode in MODES}

ax_thr_ts = ax_ts[0]
ax_E_ts   = ax_ts[1]

for mode in MODES:
    r   = res_show[mode]
    c   = MODE_COLORS[mode]
    lbl = MODE_LABELS[mode].replace('\n', ' ')
    ls  = '-' if mode == 'normal' else ('--' if mode == 'probe' else ':')
    ax_thr_ts.plot(r['t'], r['throttle'], color=c, lw=1.4, ls=ls, label=lbl)
    ax_E_ts.plot(  r['t'], r['E_cum'],    color=c, lw=1.8, ls=ls,
                   label=f"{lbl}  E={r['E_total']:.0f}")

ax_thr_ts.set_ylabel("Тяга, о.е.", fontsize=9)
ax_thr_ts.set_xlabel("Время, с", fontsize=9)
ax_thr_ts.set_title(f"Va={Va_show:.0f} м/с: тяга во времени", fontsize=9)
ax_thr_ts.legend(fontsize=7, loc='upper right')
ax_thr_ts.grid(True, ls='--', alpha=0.45)
ax_thr_ts.axhline(res_show['normal']['thr_trim'], color='black',
                  lw=0.8, ls='--', alpha=0.4, label='trim')

ax_E_ts.set_ylabel("E нарастающий, о.е.", fontsize=9)
ax_E_ts.set_xlabel("Время, с", fontsize=9)
ax_E_ts.set_title(f"Va={Va_show:.0f} м/с: накопленная энергия", fontsize=9)
ax_E_ts.legend(fontsize=7, loc='upper left')
ax_E_ts.grid(True, ls='--', alpha=0.45)

# ------------------------------------------------------------------
# Сохранение / показ
# ------------------------------------------------------------------
save_path = sys.argv[1] if len(sys.argv) > 1 else None
if save_path:
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {save_path}")

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

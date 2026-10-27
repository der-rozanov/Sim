# -*- coding: utf-8 -*-
"""
Переходные процессы и характеристики САУ для docs/control.md и docs/SENSORS.md.

    python docs/figures/sau_plots.py              # всё: числа в консоль + docs/img/*.png
    python docs/figures/sau_plots.py steps route  # выборочно (имена — ключи FIGS)

Всё считается на полной нелинейной модели 6DOF (sim/) с САУ control/ по умолчанию
(FPV-самолёт, расчёт control/tuning.py на 16 м/с); линейная теория — передаточные
функции из того же расчёта. Числа в таблицах control.md — из вывода этого скрипта.
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from sim.state import Q, H, P, R, V, X, Y, air_data, earth_velocity, euler, with_euler
from sim.wind import wind as wind_at
from runner import run, compute_trim, trim_state
from control.controllers import (PitchController, PitchControlParams, SpeedController,
                                 SpeedControlParams, LateralController, LateralControlParams,
                                 RollHold, AltitudeHold, AltitudeHoldParams)
from control.tuning import design_sau
from control.sau import SAU, SAUParams, TruthSensors, NoisySensors, Mode
from control.aua import AngleOfAttackProtector, AUAParams, AUAState
from control.navigation import WaypointNavigator, NavParams

OUT = os.path.join(ROOT, "docs", "img")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
                     "axes.grid": True, "grid.alpha": 0.35, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = ("#3b6fb6", "#d68910", "#3c8d4a", "#c0392b", "#7b4a9a",
                                          "#888888")
deg, rad = np.degrees, np.radians

ac = AircraftParams()
VA, H0, DT = 16.0, 200.0, 0.01
D = design_sau(ac, VA)
K, GN = D.coeffs, D.gains
TRIM = compute_trim(ac, VA)
ALPHA_T, DE_T, THR_T = TRIM
S0 = trim_state(ac, SimConfig(Va0=VA, h0=H0))
NOISE0 = SensorParams(gyro_noise=0.0, ins_angle_noise=0.0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("сохранено docs/img/" + name)


def cfg(t_end, Va=VA):
    return SimConfig(Va0=Va, h0=H0, dt=DT, t_end=t_end)


# ---------------------------------------------------------------------------
# Линейная теория и метрики (как в checks/check_sau.py)
# ---------------------------------------------------------------------------
def tf_step(num, den, t, amp=1.0):
    """Переходная функция num/den (каноническая форма, Эйлер с мелким шагом)."""
    num, den = np.atleast_1d(num) / den[0], np.atleast_1d(den) / den[0]
    n = len(den) - 1
    num = np.r_[np.zeros(n + 1 - len(num)), num]
    A = np.zeros((n, n)); A[:-1, 1:] = np.eye(n - 1); A[-1] = -den[:0:-1]
    C = num[:0:-1] - num[0] * den[:0:-1]
    x, y, h, ti = np.zeros(n), np.empty(len(t)), 1e-4, 0.0
    for i, tt in enumerate(t):
        while ti < tt - 1e-12:
            x = x + h * (A @ x + np.r_[np.zeros(n - 1), 1.0] * amp); ti += h
        y[i] = C @ x + num[0] * amp
    return y


def metrics(t, y, y0, y1, tol=0.05):
    """Перерегулирование, % и время установления в трубку tol от ступеньки."""
    step = y1 - y0
    over = max(0.0, (np.max((y - y0) / step) - 1.0) * 100)
    out = np.where(np.abs(y - y1) > tol * abs(step))[0]
    return over, (t[out[-1]] if len(out) else 0.0)


def course(log):
    return np.array([np.arctan2(*earth_velocity(s)[1::-1]) for s in log.state])


# ---------------------------------------------------------------------------
# Прогоны
# ---------------------------------------------------------------------------
def closed_loop(theta_ref=None, h_ref=None, Va_ref=VA, chi_ref=None, state0=None, t_end=20.0):
    """Регуляторы напрямую (как checks/check_sau.py): тангаж, ПИ высоты, ПИ скорости, крен/курс."""
    f = lambda v, t: v(t) if callable(v) else v
    pitch = PitchController(ac, PitchControlParams())
    pitch.set_trim_elevator(DE_T); pitch.set_trim_throttle(THR_T)
    speed = SpeedController(ac, SpeedControlParams())
    speed.set_trim_throttle(THR_T)
    lat = LateralController(ac, LateralControlParams(beta_hold=True))
    hold = RollHold(ac, NOISE0, np.random.default_rng(0))
    alt = AltitudeHold(AltitudeHoldParams(), ALPHA_T)

    def fn(t, s, Va, al):
        speed.set_Va_ref(f(Va_ref, t))
        pitch.theta_ref = alt.step(f(h_ref, t), s[H], DT) if h_ref is not None else theta_ref
        de = pitch.step(t, {"theta": euler(s)[1], "q": s[Q], "h": s[H], "Va": Va}, DT)[0]
        dt_ = speed.step(Va, DT)
        if chi_ref is None:
            da, dr = hold.step(euler(s)[0], s[P], Va, DT), 0.0
        else:
            Vx, Vy, _ = earth_velocity(s)
            lat.set_course(f(chi_ref, t))
            da, dr = lat.step({"chi": np.arctan2(Vy, Vx), "Vg": np.hypot(Vx, Vy), "phi": euler(s)[0],
                               "p": s[P], "Va": Va, "beta": np.arcsin(np.clip(s[V] / Va, -1, 1))}, DT)
        return np.array([de, dt_, da, dr])

    return run(fn, ac, WindParams(), cfg(t_end), state0=S0.copy() if state0 is None else state0)


def fly(t_end, lon="tecs", setup=None, events=None, wind=WindParams(), sensors=None, prot=None,
        nav=None, params=None, sau_cls=SAU, Va=VA, mode=Mode.ALTITUDE):
    """Полная САУ control.sau.SAU; rec — уставки и внутренние сигналы на каждом шаге."""
    trim = compute_trim(ac, Va)
    s0 = trim_state(ac, cfg(t_end, Va))
    sau = sau_cls(ac, trim, sensors or TruthSensors(), params or SAUParams(lon=lon), nav=nav,
                  prot=prot)
    sau.engage(theta=trim[0], h=H0, Va=Va, chi=0.0)
    sau.set_mode(mode)
    if setup:
        setup(sau)
    rec = {k: [] for k in ("theta_cmd", "theta_ref", "phi_ref", "chi_ref", "h_ref", "prot", "est",
                           "e_py", "e_true", "nav_mode")}

    def fn(t, s, Va_, al):
        if events:
            events(t, sau, s)
        b = air_data(s, wind_at(s[H], t, wind))[2]
        c = sau.step(t, s, Va_, al, b, DT)
        rec["theta_cmd"].append(sau.theta_cmd); rec["theta_ref"].append(sau.theta_ref)
        rec["phi_ref"].append(sau.lat.phi_ref); rec["chi_ref"].append(sau.lat.chi_ref)
        rec["h_ref"].append(sau.refs.h)
        rec["prot"].append(int(sau.prot_out.state) if sau.prot_out is not None else 0)
        rec["est"].append(dict(sau.est))
        rec["e_py"].append(nav.e_py if nav is not None else 0.0)
        rec["e_true"].append(nav_error(nav, s[X], s[Y]) if nav is not None else 0.0)
        rec["nav_mode"].append(nav.mode if nav is not None else "")
        return c

    log = run(fn, ac, wind, cfg(t_end, Va), state0=s0)
    rec = {k: (np.array(v) if k not in ("est", "nav_mode") else v) for k, v in rec.items()}
    return log, rec


def nav_error(nav, pn, pe):
    """Истинное отклонение от участка навигатора (по истинному, а не GPS-положению)."""
    if nav.mode == "line":
        r = nav.prev
        return -np.sin(nav.chi_q) * (pn - r[0]) + np.cos(nav.chi_q) * (pe - r[1])
    if nav.mode in ("fillet", "orbit"):
        cn, ce, rho, _ = nav.circle
        return np.hypot(pn - cn, pe - ce) - rho
    return 0.0


def legend_below(fig, axs, ncol=4, y=-0.02):
    h, l = axs.flat[0].get_legend_handles_labels() if hasattr(axs, "flat") else axs.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=9.5)


# ---------------------------------------------------------------------------
# 1. Ступеньки внутренних контуров: тангаж, крен, курс
# ---------------------------------------------------------------------------
def fig_steps():
    a1, a2, tau = K["a_phi1"], K["a_phi2"], GN["tau_phi"]
    FFk = GN["FF_p"] + GN["kp_p"]
    num_phi = [a2 * FFk / tau, a2 * GN["ki_p"] / tau]
    den_phi = [1, a1 + a2 * GN["kp_p"], a2 * (GN["ki_p"] + FFk / tau), a2 * GN["ki_p"] / tau]

    L1 = closed_loop(theta_ref=ALPHA_T + rad(3.0), t_end=2.0)
    th = deg(L1.theta - ALPHA_T)
    lin_th = tf_step([GN["kp_theta"] * K["a_theta3"]],
                     [1, K["a_theta1"] + GN["kd_theta"] * K["a_theta3"],
                      K["a_theta2"] + GN["kp_theta"] * K["a_theta3"]], L1.t, 3.0) / GN["K_theta_DC"]
    s_r = with_euler(S0, rad(20.0), euler(S0)[1], 0.0)
    L2 = closed_loop(theta_ref=ALPHA_T, h_ref=H0, state0=s_r, t_end=2.0)
    phi = deg(L2.phi)
    lin_phi = 20.0 - tf_step(num_phi, den_phi, L2.t, 20.0)
    L3 = closed_loop(h_ref=H0, chi_ref=rad(20.0), t_end=8.0)
    chi = deg(course(L3))
    Kn = GN["K_chi"] * np.array(num_phi)
    lin_chi = tf_step(Kn, np.polyadd(np.r_[den_phi, 0.0], Kn), L3.t, 20.0)

    print("\n[steps] контур: перерегулирование мод/теор %, t_уст 5 % мод/теор, с")
    for nm, t, y, ln, y0, y1 in (("тангаж +3°", L1.t, th, lin_th, 0, 3), ("крен 20→0", L2.t, phi, lin_phi, 20, 0),
                                 ("курс +20°", L3.t, chi, lin_chi, 0, 20)):
        om, tm = metrics(t, y, y0, y1); ol, tl = metrics(t, ln, y0, y1)
        print(f"  {nm:<12} {om:5.1f} / {ol:5.1f}   {tm:5.2f} / {tl:5.2f}")
    print(f"  макс. δe {deg(np.abs(L1.controls[:, 0] - DE_T)).max():.1f}°, δa (крен) "
          f"{deg(np.abs(L2.controls[:, 2])).max():.1f}°, δa (курс) {deg(np.abs(L3.controls[:, 2])).max():.1f}°, "
          f"φ (курс) {deg(np.abs(L3.phi)).max():.1f}°")

    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4), sharex="col",
                            gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, (t, y, ln, y1, ttl, yl) in enumerate((
            (L1.t, th, lin_th, 3.0, "Тангаж: θ_ref +3°", "Δθ, °"),
            (L2.t, phi, lin_phi, 0.0, "Крен: из 20° к 0", "φ, °"),
            (L3.t, chi, lin_chi, 20.0, "Курс: χ_ref +20°", "χ, °"))):
        ax = axs[0, j]
        ax.axhline(y1, color=GRAY, lw=1, ls=":")
        ax.plot(t, y, color=BLUE, lw=2, label="полная модель 6DOF")
        ax.plot(t, ln, color=ORANGE, lw=2, ls="--", label="линейная теория (tuning.py)")
        ax.set(title=ttl, ylabel=yl)
    axs[0, 1].plot(L2.t, deg(L2.state[:, P]) / 10, color=GREEN, lw=1.4, label="p / 10, °/с")
    axs[0, 2].plot(L3.t, deg(L3.phi), color=GREEN, lw=1.4, label="крен φ, °")
    axs[0, 1].text(1.0, 0.03, "зелёная — p/10, °/с", transform=axs[0, 1].transAxes, ha="right",
                   color=GREEN, fontsize=9)
    axs[0, 2].text(1.0, 0.03, "зелёная — крен φ, °", transform=axs[0, 2].transAxes, ha="right",
                   color=GREEN, fontsize=9)
    for j, (L, i, nm, base) in enumerate(((L1, 0, "δe − δe*, °", DE_T), (L2, 2, "δa, °", 0.0),
                                          (L3, 2, "δa, °", 0.0))):
        ax = axs[1, j]
        ax.plot(L.t, deg(L.controls_cmd[:, i] - base), color=PURPLE, lw=1.3, label="команда САУ")
        ax.plot(L.t, deg(L.controls[:, i] - base), color="#333333", lw=1.8, label="положение руля (после привода)")
        ax.set(ylabel=nm, xlabel="t, с")
    axs[0, 0].text(0.98, 0.35, "теория B&M (α ≡ θ) даёт\nK = 0.61; кривая приведена\nк K = 1 — "
                   "сравнивается\nтолько форма", transform=axs[0, 0].transAxes, ha="right", fontsize=8.5,
                   color="#555")
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=9.5)
    axs[1, 1].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sau_steps.png")


# ---------------------------------------------------------------------------
# 2. Ступеньки высоты и скорости: TECS и раздельные контуры против теории
# ---------------------------------------------------------------------------
def fig_outer_steps():
    tc = D.tecs
    res = {}
    for lon in ("tecs", "split"):
        Lh, _ = fly(15.0, lon, setup=lambda s: setattr(s.refs, "h", H0 + 5.0))
        Lv, _ = fly(15.0, lon, setup=lambda s: setattr(s.refs, "Va", VA + 2.0))
        res[lon] = (Lh, Lv)
    t = res["tecs"][0].t
    th_tecs = tf_step([tc.K_h], [1, tc.K_h], t, 5.0)
    tv_tecs = tf_step([tc.K_V], [1, tc.K_V], t, 2.0)
    th_split = tf_step([GN["kp_h"] * VA, GN["ki_h"] * VA], [1, GN["kp_h"] * VA, GN["ki_h"] * VA], t, 5.0)
    tv_split = tf_step([GN["kp_V"] * K["a_V2"], GN["ki_V"] * K["a_V2"]],
                       [1, K["a_V1"] + K["a_V2"] * GN["kp_V"], K["a_V2"] * GN["ki_V"]], t, 2.0)
    print("\n[outer] ступенька: перерегулирование мод/теор %, t_уст 5 % мод/теор, с; перекрёстная связь")
    for lon, (Lh, Lv), (thh, tvv) in (("TECS", res["tecs"], (th_tecs, tv_tecs)),
                                      ("раздельные", res["split"], (th_split, tv_split))):
        for nm, L, y, ln, a in (("h +5 м", Lh, Lh.state[:, H] - H0, thh, 5.0),
                                ("Va +2 м/с", Lv, Lv.Va - VA, tvv, 2.0)):
            om, tm = metrics(L.t, y, 0, a); ol, tl = metrics(L.t, ln, 0, a)
            cross = (f"|ΔVa| ≤ {np.abs(L.Va - VA).max():.2f} м/с" if "h" in nm
                     else f"|Δh| ≤ {np.abs(L.state[:, H] - H0).max():.2f} м")
            print(f"  {lon:<11} {nm:<10} {om:5.1f} / {ol:5.1f}   {tm:5.2f} / {tl:5.2f}   {cross}")

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 6.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.6, 1]))
    for j, (k, a, nm) in enumerate(((0, 5.0, "Высота: h_ref +5 м"), (1, 2.0, "Скорость: Va_ref 16 → 18 м/с"))):
        ax = axs[0, j]
        ax.axhline(a, color=GRAY, lw=1, ls=":")
        for lon, col, lab, ln in (("tecs", BLUE, "TECS", (th_tecs, tv_tecs)[k]),
                                  ("split", ORANGE, "раздельные контуры", (th_split, tv_split)[k])):
            L = res[lon][k]
            y = L.state[:, H] - H0 if k == 0 else L.Va - VA
            ax.plot(L.t, y, color=col, lw=2, label=lab + " — модель")
            ax.plot(t, ln, color=col, lw=1.4, ls="--", label=lab + " — теория")
        ax.set(title=nm, ylabel="Δh, м" if k == 0 else "ΔVa, м/с")
        ax2 = axs[1, j]
        for lon, col in (("tecs", BLUE), ("split", ORANGE)):
            L = res[lon][k]
            y = L.Va - VA if k == 0 else L.state[:, H] - H0
            ax2.plot(L.t, y, color=col, lw=1.8)
        ax2.set(ylabel="ΔVa, м/с" if k == 0 else "Δh, м", xlabel="t, с",
                title="перекрёстная связь: " + ("скорость при наборе" if k == 0 else "высота при разгоне"),)
        ax2.title.set_fontsize(9.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=9.5)
    fig.tight_layout()
    save(fig, "sau_outer_steps.png")


# ---------------------------------------------------------------------------
# 3. TECS против раздельных контуров на больших манёврах
# ---------------------------------------------------------------------------
def fig_tecs_vs_split():
    man = (("h_ref +50 м", 40.0, lambda s: setattr(s.refs, "h", H0 + 50.0)),
           ("h_ref −50 м", 40.0, lambda s: setattr(s.refs, "h", H0 - 50.0)),
           ("Va_ref 16 → 11 м/с, h = const", 30.0, lambda s: setattr(s.refs, "Va", 11.0)))
    runs = {(m[0], lon): fly(m[1], lon, setup=m[2])[0] for m in man for lon in ("tecs", "split")}
    print("\n[tecs_vs_split] манёвр / контуры: t_уст h ±1 м, Va min…max, α max, доля газа в 0, Δh max")
    for nm, _, _ in man:
        for lon in ("split", "tecs"):
            L = runs[(nm, lon)]
            dh = L.state[:, H] - H0
            target = 50.0 if "+50" in nm else (-50.0 if "−50" in nm else 0.0)
            out = np.where(np.abs(dh - target) > 1.0)[0]
            ts = L.t[out[-1]] if len(out) else 0.0
            print(f"  {nm:<28} {lon:<6} t_уст {ts:5.1f} с  Va {L.Va.min():5.2f}…{L.Va.max():5.2f}  "
                  f"α max {deg(L.alpha.max()):4.1f}°  газ=0 {100 * np.mean(L.controls[:, 1] <= 1e-3):4.0f} %  "
                  f"|Δh| max {np.abs(dh).max():5.2f}")
    # шум датчиков в установившемся режиме
    for lon in ("split", "tecs"):
        L, _ = fly(30.0, lon, sensors=NoisySensors(SensorParams(), np.random.default_rng(1)))
        m = L.t > 5
        print(f"  шум, {lon:<6}: σ газа {np.std(L.controls[m, 1]):.3f}, σ δe {deg(np.std(L.controls[m, 0])):.2f}°")

    rows = (("h, м", lambda L: L.state[:, H]), ("Va, м/с", lambda L: L.Va),
            ("газ δt", lambda L: L.controls[:, 1]), ("α, °", lambda L: deg(L.alpha)))
    fig, axs = plt.subplots(4, 3, figsize=(14, 9.5), sharex="col")
    for j, (nm, _, _) in enumerate(man):
        for i, (yl, f) in enumerate(rows):
            ax = axs[i, j]
            for lon, col, lab in (("split", ORANGE, "раздельные контуры"), ("tecs", BLUE, "TECS")):
                L = runs[(nm, lon)]
                ax.plot(L.t, f(L), color=col, lw=1.9, label=lab)
            if j == 0:
                ax.set_ylabel(yl)
        axs[0, j].set_title(nm)
        axs[-1, j].set_xlabel("t, с")
    axs[1, 2].axhline(1.25 * K["Va_stall"], color=RED, lw=1, ls=":")
    axs[1, 2].text(29.5, 1.25 * K["Va_stall"] + 0.15, "Va_min = 1.25·Va_stall", color=RED, ha="right",
                   fontsize=8.5)
    axs[3, 2].axhline(deg(ac.alpha_warning), color=RED, lw=1, ls=":")
    axs[3, 2].text(29.5, deg(ac.alpha_warning) + 0.2, "α_пред 9°", color=RED, ha="right", fontsize=8.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.tight_layout()
    save(fig, "sau_tecs_vs_split.png")


# ---------------------------------------------------------------------------
# 4. Курс: ступеньки 10 / 30 / 90°
# ---------------------------------------------------------------------------
def fig_course():
    steps = (10.0, 30.0, 90.0)
    cols = (GREEN, BLUE, ORANGE)
    runs = [fly(12.0, setup=lambda s, a=a: setattr(s.refs, "chi", rad(a))) for a in steps]
    print("\n[course] ступенька: перерегулирование %, t_уст 5 %, φ max, δa max, p max, r min, β max, |Δh|")
    for a, (L, _) in zip(steps, runs):
        chi = deg(course(L))
        om, tm = metrics(L.t, chi, 0, a)
        print(f"  {a:4.0f}°: {om:4.1f} %  {tm:4.2f} с  φ {deg(np.abs(L.phi)).max():4.1f}°  "
              f"δa {deg(np.abs(L.controls[:, 2])).max():4.1f}°  p {deg(np.abs(L.state[:, P])).max():5.1f} °/с  "
              f"r min {deg(L.state[:, R].min()):5.1f} °/с  β {deg(np.abs(L.beta)).max():3.1f}°  "
              f"|Δh| {np.abs(L.state[:, H] - H0).max():.2f} м")
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 6.6), sharex=True)
    for a, c, (L, rec) in zip(steps, cols, runs):
        axs[0, 0].plot(L.t, deg(course(L)) / a * 100, color=c, lw=2, label=f"ступенька {a:.0f}°")
        axs[0, 1].plot(L.t, deg(L.phi), color=c, lw=2)
        axs[0, 1].plot(L.t, deg(rec["phi_ref"]), color=c, lw=1, ls="--")
        axs[1, 0].plot(L.t, deg(L.state[:, P]), color=c, lw=2)
        axs[1, 1].plot(L.t, deg(L.controls[:, 2]), color=c, lw=2)
    axs[0, 0].axhline(100, color=GRAY, lw=1, ls=":")
    axs[0, 0].set(title="Курс χ, % от ступеньки")
    axs[0, 1].set(title="Крен φ, ° (пунктир — уставка φ_ref)")
    axs[0, 1].axhline(30, color=RED, lw=1, ls=":")
    axs[1, 0].set(title="Скорость крена p, °/с", xlabel="t, с")
    axs[1, 0].axhline(120, color=RED, lw=1, ls=":")
    axs[1, 0].text(11.8, 123, "p_max", color=RED, ha="right", fontsize=8.5)
    axs[1, 1].set(title="Элероны δa, °", xlabel="t, с")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.tight_layout()
    save(fig, "sau_course.png")


# ---------------------------------------------------------------------------
# 5. Координация разворота: ограничение скорости φ_ref и микс δa → δr
# ---------------------------------------------------------------------------
def fig_coord():
    base = LateralControlParams()
    var = (("каскад крена без средств", 1e3, 0.0, GRAY),
           ("+ ограничение скорости φ_ref", base.phi_ref_rate, 0.0, ORANGE),
           ("+ микс δa → δr", 1e3, base.da_dr_mix, GREEN),
           ("оба (по умолчанию)", base.phi_ref_rate, base.da_dr_mix, BLUE))
    runs = []
    print("\n[coord] курс +30°: r min, max |ṙ|, β max, t_уст 5 %")
    for nm, rate, mix, c in var:
        pr = SAUParams(lateral=LateralControlParams(phi_ref_rate=rate, da_dr_mix=mix))
        L, _ = fly(6.0, params=pr, setup=lambda s: setattr(s.refs, "chi", rad(30.0)))
        r = deg(L.state[:, R])
        print(f"  {nm:<30} r min {r.min():6.1f} °/с  |ṙ| {np.abs(np.diff(r)).max() / DT:5.0f} °/с²  "
              f"β max {deg(np.abs(L.beta)).max():4.1f}°  t_уст {metrics(L.t, deg(course(L)), 0, 30)[1]:4.2f} с")
        runs.append((nm, c, L))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for nm, c, L in runs:
        lw = 2.2 if c == BLUE else 1.6
        axs[0].plot(L.t, deg(L.state[:, R]), color=c, lw=lw, label=nm)
        axs[1].plot(L.t, deg(L.beta), color=c, lw=lw)
        axs[2].plot(L.t, deg(L.controls[:, 2]), color=c, lw=lw)
    axs[0].set(title="Рыскание r, °/с (< 0 — нос против разворота)", xlim=(0, 3), xlabel="t, с")
    axs[1].set(title="Скольжение β, °", xlabel="t, с")
    axs[2].set(title="Элероны δa, °", xlabel="t, с")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=9.5)
    fig.tight_layout()
    save(fig, "sau_coord.png")


# ---------------------------------------------------------------------------
# 6. Защита по α: режим «Срыв» стенда (газ 0, θ_ref = +15°)
# ---------------------------------------------------------------------------
class StallSAU(SAU):
    """Как BenchSAU стенда: в режиме «Срыв» внешний продольный контур отключён."""
    stall = False

    def _outer(self, est, dt):
        if self.stall:
            return rad(15.0), 0.0
        return super()._outer(est, dt)


def fig_aoa():
    warn, crit = ac.alpha_warning, ac.alpha_crit
    aua = AUAParams(enabled=True, alpha_warn=warn, alpha_crit=crit, alpha_exit=warn - (crit - warn))
    T0 = 2.0

    def ev(t, sau, s):
        if t >= T0 and not getattr(sau, "_done", False) and not sau.stall:
            sau.stall = True
        if sau.stall and sau.prot_out is not None and sau.prot_out.state == AUAState.CRITICAL:
            sau.stall, sau._done = False, True             # защита перехватила — выход из «Срыва»
            sau.refs.h = s[H]
            sau.lon.reset()

    L, rec = fly(25.0, sau_cls=StallSAU, prot=AngleOfAttackProtector(ac, aua), events=ev)

    def ev0(t, sau, s):
        sau.stall = t >= T0
    L0, _ = fly(25.0, sau_cls=StallSAU, events=ev0)

    st = rec["prot"]
    tc = L.t[np.argmax(st == int(AUAState.CRITICAL))]
    print(f"\n[aoa] защита: КРИТ на {tc:.2f} с, α max {deg(L.alpha.max()):.1f}°, Va min {L.Va.min():.1f} м/с, "
          f"Δh {L.state[:, H].min() - H0:+.1f}…{L.state[:, H].max() - H0:+.1f} м; без защиты α max "
          f"{deg(L0.alpha.max()):.1f}°, Va min {L0.Va.min():.1f}, Δh за 25 с {L0.state[-1, H] - H0:.0f} м, ḣ {np.gradient(L0.state[-500:, H], DT).mean():.1f} м/с")

    from control.aua import AUA_COLORS, AUA_LABELS
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.6, 1, 1, 1]))
    for ax in axs:
        i = 0
        while i < len(st):
            j = i
            while j < len(st) and st[j] == st[i]:
                j += 1
            if st[i] != 0:
                ax.axvspan(L.t[i], L.t[min(j, len(L.t) - 1)], color=AUA_COLORS[AUAState(st[i])], alpha=0.13,
                           lw=0)
            i = j
    ax = axs[0]
    ax.plot(L0.t, deg(L0.alpha), color=GRAY, lw=1.6, ls="--", label="без защиты")
    ax.plot(L.t, deg(L.alpha), color=BLUE, lw=2.2, label="с защитой по α")
    for v, nm in ((crit, "α_крит 12.2°"), (warn, "α_пред 9°"), (aua.alpha_exit, "α_выход 5.8°")):
        ax.axhline(deg(v), color=RED, lw=1, ls=":")
        ax.text(24.8, deg(v) + 0.2, nm, color=RED, fontsize=8.5, ha="right")
    ax.set(ylabel="α, °", title="Режим «Срыв» с 2 с: газ 0, θ_ref = +15°; фон — состояние автомата защиты")
    ax.set_ylim(-2, 24)
    ax.annotate(f"без защиты α → {deg(L0.alpha[-1]):.0f}°:\nглубокий срыв, парашютирование",
                xy=(5.9, 23.5), xytext=(7.0, 17), fontsize=9, color="#555",
                arrowprops=dict(arrowstyle="-|>", color="#555"))
    for s_ in (1, 2, 3):
        ax.fill_between([], [], color=AUA_COLORS[AUAState(s_)], alpha=0.3, label=AUA_LABELS[AUAState(s_)])
    ax.legend(loc="upper right", ncol=5, fontsize=9, bbox_to_anchor=(1.0, 0.93))
    axs[1].plot(L0.t, L0.Va, color=GRAY, lw=1.6, ls="--")
    axs[1].plot(L.t, L.Va, color=BLUE, lw=2)
    axs[1].set(ylabel="Va, м/с")
    axs[2].plot(L.t, deg(rec["theta_cmd"]), color=ORANGE, lw=1.4, ls="--", label="θ_cmd (до защиты)")
    axs[2].plot(L.t, deg(rec["theta_ref"]), color=PURPLE, lw=1.6, label="θ_ref (после защиты)")
    axs[2].plot(L.t, deg(L.theta), color=BLUE, lw=2, label="θ")
    axs[2].set(ylabel="θ, °")
    axs[2].legend(loc="upper right", ncol=3, fontsize=9)
    axs[3].plot(L.t, L.state[:, H] - H0, color=BLUE, lw=2)
    axs[3].plot(L0.t, L0.state[:, H] - H0, color=GRAY, lw=1.6, ls="--")
    axs[3].set(ylabel="Δh, м", xlabel="t, с")
    fig.tight_layout()
    save(fig, "sau_aoa_demo.png")


# ---------------------------------------------------------------------------
# 7. Навигация: векторные поля и полёт по маршруту
# ---------------------------------------------------------------------------
def fig_nav_field():
    pr = NavParams()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    # прямая вдоль севера через начало координат
    n, e = np.meshgrid(np.linspace(-50, 450, 19), np.linspace(-300, 300, 21), indexing="ij")
    e_py = e
    chi = 0.0 - pr.chi_inf * (2 / np.pi) * np.arctan(pr.k_path * e_py)
    ax = axs[0]
    ax.quiver(e, n, np.sin(chi), np.cos(chi), color=BLUE, alpha=0.75, scale=30, width=0.004)
    ax.plot([0, 0], [-50, 450], color=RED, lw=2.5, label="заданная прямая")
    ax.set(title="Прямая: χ_ref = χ_q − χ∞·(2/π)·atan(k_path·e_py)", xlabel="восток E, м",
           ylabel="север N, м", aspect="equal")
    ax.title.set_fontsize(10)
    ax.text(-290, -35, f"χ∞ = 60°, k_path = {pr.k_path} 1/м", fontsize=9, color="#444")
    # окружность
    rho = pr.R_orbit
    n, e = np.meshgrid(np.linspace(-420, 420, 23), np.linspace(-420, 420, 23), indexing="ij")
    d = np.hypot(n, e)
    vph = np.arctan2(e, n)
    chi = vph + 1.0 * (np.pi / 2 + np.arctan(pr.k_orbit * (d - rho) / rho))
    ax = axs[1]
    ax.quiver(e, n, np.sin(chi), np.cos(chi), color=BLUE, alpha=0.75, scale=30, width=0.004)
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rho * np.sin(a), rho * np.cos(a), color=RED, lw=2.5)
    ax.set(title="Окружность (по часовой): χ_ref = ϑ + λ·(π/2 + atan(k_orbit·(d − ρ)/ρ))",
           xlabel="восток E, м", ylabel="север N, м", aspect="equal")
    ax.title.set_fontsize(10)
    # скругление угла
    ax = axs[2]
    w0, w1, w2 = np.array([0.0, 0.0]), np.array([600.0, 0.0]), np.array([600.0, 600.0])   # (N, E)
    q0 = (w1 - w0) / 600; q1 = (w2 - w1) / 600
    R = pr.R_fillet
    varrho = np.arccos(-q0 @ q1)
    dd = R / np.tan(varrho / 2)
    z1, z2 = w1 - dd * q0, w1 + dd * q1
    c = w1 - R / np.sin(varrho / 2) * (q0 - q1) / np.linalg.norm(q0 - q1)
    P_ = lambda v: (v[1], v[0])
    ax.plot(*zip(P_(w0), P_(w1), P_(w2)), color=GRAY, lw=1.4, ls="--", marker="o", ms=7)
    th = np.linspace(np.arctan2(*(z1 - c)[::-1]), np.arctan2(*(z2 - c)[::-1]), 60)
    arc = c[:, None] + R * np.array([np.cos(th), np.sin(th)])
    ax.plot([w0[1], z1[1]], [w0[0], z1[0]], color=RED, lw=2.5, label="путь ЛА")
    ax.plot(arc[1], arc[0], color=RED, lw=2.5)
    ax.plot([z2[1], w2[1]], [z2[0], w2[0]], color=RED, lw=2.5)
    for p, nm, off in ((w0, "$w_{i-1}$", (15, -30)), (w1, "$w_i$", (15, 10)), (w2, "$w_{i+1}$", (-30, 25)),
                       (z1, "z1", (-60, 10)), (z2, "z2", (15, -40)), (c, "c", (-35, 10))):
        ax.plot(p[1], p[0], "o", color="k", ms=5)
        ax.text(p[1] + off[0], p[0] + off[1], nm, fontsize=10)
    ax.plot([c[1], arc[1][30]], [c[0], arc[0][30]], color=GREEN, lw=1.3)
    ax.text((c[1] + arc[1][30]) / 2 + 10, (c[0] + arc[0][30]) / 2, "R", color=GREEN, fontsize=11)
    for z, q in ((z1, q0), (z2, q1)):                     # полуплоскости переключения
        nrm = np.array([-q[1], q[0]])
        a_, b_ = z - 130 * nrm, z + 130 * nrm
        ax.plot([a_[1], b_[1]], [a_[0], b_[0]], color=ORANGE, lw=1.4, ls=":")
    ax.text(330, 80, "граница полуплоскости:\nпересёк — следующий\nучасток", fontsize=8.5, color=ORANGE)
    ax.set(title=f"Скругление угла: d = R/tg(ϱ/2), R = {R:.0f} м", xlabel="восток E, м",
           ylabel="север N, м", aspect="equal", xlim=(-80, 700), ylim=(-80, 700))
    ax.title.set_fontsize(10)
    fig.tight_layout()
    save(fig, "sau_nav_field.png")


def fig_route():
    WPS = [(1000, 0, 100), (1000, 1000, 130), (0, 1000, 130), (600, 300, 100)]
    out = []
    print("\n[route] маршрут s10 (шум датчиков): |e_py| max на прямых (без 10 с после входа), |d − R| на дугах,"
          " |h − h_ref| в конце участков")
    for wp, nm in ((WindParams(), "штиль"), (WindParams(Vw_cross=5.0), "ветер 5 м/с на восток")):
        nav = WaypointNavigator(NavParams())

        def setup(s):
            s.fly_route(WPS, start=(0.0, 0.0, 100.0))
        global H0
        h_keep, H0 = H0, 100.0
        L, rec = fly(340.0, setup=setup, nav=nav, wind=wp, mode=Mode.ALTITUDE,
                     sensors=NoisySensors(SensorParams(), np.random.default_rng(7), gps_pos=True))
        H0 = h_keep
        modes = np.array(rec["nav_mode"])
        line = modes == "line"
        start = np.r_[False, line[1:] & ~line[:-1]]
        t_in = np.maximum.accumulate(np.where(start | (np.arange(len(line)) == 0), L.t, 0))
        m = line & (L.t - t_in > 10.0)
        arcm = (modes == "fillet")
        orb = modes == "orbit"
        t_orb = L.t[np.argmax(orb)]
        orb = orb & (L.t > t_orb + 40.0)              # выход на окружность из её центра — отброшен
        rec["e_plot"] = np.where((modes == "orbit") & ~orb, np.nan, rec["e_true"])
        dh = np.abs(L.state[:, H] - rec["h_ref"])
        e = rec["e_true"]
        print(f"  {nm:<24} истинное: прямые {np.abs(e[m]).max():4.1f} м (по GPS {np.abs(rec['e_py'][m]).max():4.1f})   "
              f"дуги {np.abs(e[arcm]).max():4.1f} м   кружение {np.abs(e[orb]).max():4.1f} м (СКО {np.std(e[orb]):.1f})   "
              f"|h − h_ref| (конец) {dh[-1]:.2f} м")
        out.append((nm, L, rec))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, (nm, L, rec), c in zip(axs[:2], out, (BLUE, ORANGE)):
        ax.plot(L.state[:, Y], L.state[:, X], color=c, lw=2, label="траектория")
        w = np.array(WPS)
        ax.plot(np.r_[0, w[:, 1]], np.r_[0, w[:, 0]], color=GRAY, lw=1.2, ls="--", marker="o", ms=6,
                label="точки маршрута")
        for k, p in enumerate(WPS):
            ax.text(p[1] + 30, p[0] + 30, f"{k + 1}", fontsize=10, fontweight="bold")
        ax.plot(0, 0, "s", color="k", ms=6)
        ax.text(30, -60, "старт", fontsize=9)
        if "ветер" in nm:
            ax.annotate("", xy=(1350, 500), xytext=(1150, 500),
                        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
            ax.text(1150, 540, "ветер 5 м/с", color=GREEN, fontsize=9)
        ax.set(title=nm, xlabel="восток E, м", ylabel="север N, м", aspect="equal",
               xlim=(-250, 1450), ylim=(-150, 1300))
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.33, 0.02), fontsize=9.5)
    ax = axs[2]
    for (nm, L, rec), c in zip(out, (BLUE, ORANGE)):
        ax.plot(L.t, rec["e_plot"], color=c, lw=1.2, label=nm, alpha=0.85)
    ax.set(title="Истинное отклонение от пути: e_py на прямых, d − R на дугах и кружении\n"
           "(выход на окружность кружения из её центра не показан)", xlabel="t, с", ylabel="м", ylim=(-1.5, 1.5))
    ax.title.set_fontsize(10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "sau_route.png")


# ---------------------------------------------------------------------------
# 8. Полосы контуров: разнос частот
# ---------------------------------------------------------------------------
def fig_bandwidth():
    from model_plots import linearize
    eig = linearize(VA)
    lon, lat = eig["lon"], eig["lat"]
    cl = sorted(lon[lon.imag > 1e-6], key=abs)
    dr = lat[lat.imag > 1e-6][0]
    roll = -np.sort(lat[np.abs(lat.imag) <= 1e-6].real)[0]
    w_beta = np.sqrt(GN["ki_beta"] * K["a_beta2"])
    nav_bw = VA * NavParams().chi_inf * 2 / np.pi * NavParams().k_path
    tc = D.tecs
    rows = [
        ("ЛА: продольное", [(abs(cl[-1]), "короткопериод."), (abs(cl[0]), "фугоида")], PURPLE),
        ("ЛА: боковое", [(roll, "крен"), (abs(dr), "голл. шаг")], PURPLE),
        ("Тангаж", [(D.specs.wn_theta, "θ (ωn_θ)")], BLUE),
        ("TECS", [(D.specs.tecs_w_T, "Ė, Ḃ (ω_T, ω_B)"), (tc.K_h, "K_h, K_V")], BLUE),
        ("Раздельные", [(D.specs.wn_V, "Va (ωn_V)"), (GN["w_h"], "h (ω_h)")], ORANGE),
        ("Крен", [(D.specs.lam_p, "p (λ_p)"), (1 / GN["tau_phi"], "φ (1/τ_φ)")], GREEN),
        ("Курс, β", [(1 / (2 * GN["tau_phi"]), "χ (двойной полюс)"), (w_beta, "β")], GREEN),
        ("Навигация", [(nav_bw, "выход на линию")], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, (nm, pts, c) in enumerate(rows):
        y = len(rows) - 1 - i
        xs = [p[0] for p in pts]
        ax.plot([min(xs), max(xs)], [y, y], color=c, lw=1.5, alpha=0.5)
        for k, (x, lab) in enumerate(pts):
            ax.plot(x, y, "o", color=c, ms=9, mec="white", mew=1.5, zorder=3)
            ax.text(x, y + 0.22, f"{lab}\n{x:.2g}", ha="center", va="bottom", fontsize=8.3, color="#333")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xscale("log")
    ax.set(xlim=(0.05, 200), ylim=(-0.6, len(rows) + 0.2),
           xlabel="частота (полоса, ωn или полюс), рад/с — логарифмическая шкала")
    ax.grid(axis="y", alpha=0)
    ax.set_title("Частоты собственных движений ЛА и полосы контуров САУ (FPV, 16 м/с)", fontsize=11)
    fig.tight_layout()
    save(fig, "sau_bandwidth.png")
    print(f"\n[bandwidth] кп {abs(cl[-1]):.1f}, фугоида {abs(cl[0]):.2f}, крен {roll:.1f}, голл. шаг {abs(dr):.1f}, "
          f"β {w_beta:.2f}, навигация {nav_bw:.3f} рад/с")


# ---------------------------------------------------------------------------
# 9. Датчики: измерения против истины
# ---------------------------------------------------------------------------
def fig_sensors():
    L, rec = fly(10.0, sensors=NoisySensors(SensorParams(), np.random.default_rng(3), gps_pos=True,
                                            alpha_probe=True))
    est = rec["est"]
    g = lambda k: np.array([e[k] for e in est])
    Vx = np.array([earth_velocity(s)[0] for s in L.state])
    Vy = np.array([earth_velocity(s)[1] for s in L.state])
    Vh = np.array([earth_velocity(s)[2] for s in L.state])
    panels = (("h, м — барометр (σ 0.5 м)", L.state[:, H], g("h"), 1),
              ("Va, м/с — ПВД (σ 0.2 м/с)", L.Va, g("Va"), 1),
              ("$\\dot{h}$, м/с — GPS (σ 0.1 м/с)", Vh, g("hdot"), 1),
              ("θ, ° — ИНС (σ 0.11°)", deg(L.theta), deg(g("theta")), 1),
              ("α, ° — зонд (σ 0.57°)", deg(L.alpha), deg(g("alpha")), 1),
              ("β, ° — зонд (σ 0.57°)", deg(L.beta), deg(g("beta")), 1),
              ("χ, ° — GPS по скорости (σ_V 0.1 м/с)", deg(np.arctan2(Vy, Vx)), deg(g("chi")), 1),
              ("q, °/с — гироскоп (σ 0.11 °/с)", deg(L.state[:, Q]), deg(g("q")), 1),
              ("pn, м — GPS (σ 1 м), отн. линейного хода", L.state[:, X] - VA * L.t, g("pn") - VA * L.t, 1))
    fig, axs = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for ax, (ttl, tru, mea, _) in zip(axs.flat, panels):
        ax.plot(L.t, mea, ".", color=ORANGE, ms=2.2, alpha=0.7, label="измерение (100 Гц)")
        ax.plot(L.t, tru, color=BLUE, lw=2, label="истинное значение")
        ax.set_title(ttl, fontsize=9.5)
    for ax in axs[-1]:
        ax.set_xlabel("t, с")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10, markerscale=5)
    fig.tight_layout()
    save(fig, "sensors_noise.png")
    print("\n[sensors] СКО (изм − истина): " + ", ".join(
        f"{t.split(',')[0]} {np.std(m - tr):.3g}" for t, tr, m, _ in panels))


FIGS = {"steps": fig_steps, "outer": fig_outer_steps, "tecs": fig_tecs_vs_split, "course": fig_course,
        "coord": fig_coord, "aoa": fig_aoa, "navfield": fig_nav_field, "route": fig_route,
        "bandwidth": fig_bandwidth, "sensors": fig_sensors}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(FIGS):
        FIGS[n]()

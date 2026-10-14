# -*- coding: utf-8 -*-
"""
Графики и таблицы модели ЛА для docs/physics.md и docs/aerodynamics.md.

    python docs/figures/model_plots.py            # всё: таблицы в консоль + docs/img/*.png
    python docs/figures/model_plots.py tables     # только таблицы

Всё считается по коду модели (sim/), для AircraftParams() — текущего ЛА по умолчанию.
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim.config import AircraftParams, SimConfig
from sim.state import U, W, Q, THETA, V, P, R, PHI, DT
from sim.aero import coef_CL, coef_CD, coef_Cm, _sigmoid
from sim.dynamics import derivatives, propeller
from sim.actuators import actuator_init, actuator_step
from runner import compute_trim, trim_state

OUT = os.path.join(ROOT, "docs", "img")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.35, "axes.spines.top": False,
                     "axes.spines.right": False})
BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = ("#3b6fb6", "#c0392b", "#3c8d4a", "#d68910",
                                          "#7b4a9a", "#888888")
ac = AircraftParams()
deg = np.degrees
rad = np.radians
CALM = lambda h, t: (0.0, 0.0, 0.0)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("сохранено docs/img/" + name)


# ---------------------------------------------------------------------------
# Расчёты
# ---------------------------------------------------------------------------
def cl_max():
    a = rad(np.linspace(0, 20, 4001))
    cl = np.array([coef_CL(x, 0, 0, 20.0, ac) for x in a])
    i = int(np.argmax(cl))
    return cl[i], deg(a[i])


def stall_speed():
    clm, _ = cl_max()
    return np.sqrt(2 * ac.mass * ac.g / (ac.rho * ac.S * clm))


def trim_point(Va):
    """Балансировка и мощности горизонтального полёта."""
    a, de, dt = compute_trim(ac, Va)
    T, _ = propeller(dt, Va, ac)
    D = 0.5 * ac.rho * Va**2 * ac.S * coef_CD(a, ac, delta_e=de)
    return dict(Va=Va, alpha=deg(a), de=deg(de), dt=dt, T=T, D=D, P=T * Va,
                LD=ac.mass * ac.g / D)


def max_level_speed():
    lo, hi = 15.0, 45.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        _, _, dt = compute_trim(ac, mid)
        lo, hi = (mid, hi) if dt < 0.999 else (lo, mid)
    return lo


def linearize(Va):
    """Собственные значения продольного [u, w, q, θ] и бокового [v, p, r, φ] движения."""
    cfg = SimConfig(Va0=Va, h0=100.0)
    a, de, dt = compute_trim(ac, Va)
    s0 = trim_state(ac, cfg)
    c0 = np.array([de, dt, 0.0, 0.0])
    # боковой трим: δa, δr против реактивного момента винта (ṗ = ṙ = 0)
    d0 = derivatives(s0, c0, 0.0, ac, CALM)[[P, R]]
    B = np.column_stack([(derivatives(s0, c0 + e, 0.0, ac, CALM)[[P, R]] - d0) / 1e-6
                         for e in (np.array([0, 0, 1e-6, 0]), np.array([0, 0, 0, 1e-6]))])
    c0[2:] = np.linalg.solve(B, -d0)
    out = {}
    for name, idx in (("lon", [U, W, Q, THETA]), ("lat", [V, P, R, PHI])):
        f0 = derivatives(s0, c0, 0.0, ac, CALM)[idx]
        A = np.zeros((4, 4))
        for j, k in enumerate(idx):
            sp = s0.copy()
            sp[k] += 1e-6
            A[:, j] = (derivatives(sp, c0, 0.0, ac, CALM)[idx] - f0) / 1e-6
        out[name] = np.linalg.eigvals(A)
    return out


def describe_modes(eig):
    """Корни → (имя, текст) по типам движения."""
    lon, lat = eig["lon"], eig["lat"]
    cl = sorted(lon[lon.imag > 1e-6], key=abs)            # фугоида — медленная пара
    res = []
    for name, e in (("фугоида", cl[0]), ("короткопериодическое", cl[-1])):
        wn = abs(e)
        res.append((name, f"λ = {e.real:+.3f} ± {e.imag:.3f}j, ωn = {wn:.2f} рад/с, "
                          f"ζ = {-e.real / wn:.2f}, T = {2 * np.pi / e.imag:.2f} с"))
    dr = lat[lat.imag > 1e-6][0]
    real = np.sort(lat[np.abs(lat.imag) <= 1e-6].real)
    res.append(("крен (апериодика)", f"λ = {real[0]:+.2f} 1/с, τ = {-1 / real[0]:.3f} с"))
    wn = abs(dr)
    res.append(("голландский шаг", f"λ = {dr.real:+.3f} ± {dr.imag:.3f}j, ωn = {wn:.2f} рад/с, "
                                   f"ζ = {-dr.real / wn:.2f}, T = {2 * np.pi / dr.imag:.2f} с"))
    sp = real[1]
    res.append(("спиральная", f"λ = {sp:+.4f} 1/с, " +
                ("устойчива" if sp < 0 else f"неустойчива, время удвоения {np.log(2) / sp:.0f} с")))
    return res


def tables():
    clm, a_clm = cl_max()
    print(f"CL_max = {clm:.3f} при α = {a_clm:.1f}°;  Va_stall = {stall_speed():.2f} м/с;  "
          f"Va_max горизонта = {max_level_speed():.1f} м/с")
    T0, _ = propeller(1.0, 0.0, ac)
    print(f"статическая тяга {T0:.1f} Н, T/W = {T0 / (ac.mass * ac.g):.2f}")
    print("\nα, °  σ      CL      CD      Cm      K=CL/CD")
    for a in (-4, 0, 2, 4, 6, 8, 10, 12, 12.2, 14, 15, 16, 18, 20, 30, 45, 90):
        x = rad(a)
        cl, cd, cm = coef_CL(x, 0, 0, 20, ac), coef_CD(x, ac), coef_Cm(x, 0, 0, 20, ac)
        print(f"{a:5.1f} {_sigmoid(x, ac):5.2f} {cl:7.3f} {cd:7.3f} {cm:+7.3f} {cl / cd:7.1f}")
    print("\nVa    α°      δe°     δt     T, Н   T·Va, Вт  K")
    for Va in (9, 10, 12, 14, 16, 18, 20, 24, 28):
        p = trim_point(Va)
        print(f"{Va:4.0f} {p['alpha']:6.2f} {p['de']:7.2f} {p['dt']:6.3f} {p['T']:6.2f} "
              f"{p['P']:7.1f} {p['LD']:6.1f}")
    print("\nТяга T, Н: δt \\ Va")
    vas = (0, 5, 10, 16, 20, 25, 30)
    print("δt   " + "".join(f"{v:7.0f}" for v in vas))
    for dt in (0.25, 0.5, 0.75, 1.0):
        print(f"{dt:4.2f} " + "".join(f"{propeller(dt, v, ac)[0]:7.2f}" for v in vas))
    for Va in (16.0, 22.0):
        print(f"\nСобственные движения, Va = {Va:.0f} м/с")
        for name, txt in describe_modes(linearize(Va)):
            print(f"  {name:22s} {txt}")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
def _alpha_marks(ax, label=True):
    for a, c, name in ((deg(ac.alpha_warning), ORANGE, "предупр."),
                       (deg(ac.alpha_crit), RED, "крит."),
                       (deg(ac.alpha_stall), "#7b241c", "α₀ срыва")):
        ax.axvline(a, color=c, ls="--", lw=1.0, alpha=0.8, label=f"{name} {a:.1f}°" if label else None)


def fig_aero_coeffs():
    a = np.linspace(-10, 35, 901)
    x = rad(a)
    sig = np.array([_sigmoid(v, ac) for v in x])
    cl = np.array([coef_CL(v, 0, 0, 20, ac) for v in x])
    cl_lin = ac.CL0 + ac.CLa * x
    cl_flat = 2 * np.sign(x) * np.sin(x)**2 * np.cos(x)
    cd = np.array([coef_CD(v, ac) for v in x])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    ax.fill_between(a, 0, 1.6, where=sig > 0.5, color=RED, alpha=0.06, lw=0)
    ax.plot(a, cl_lin, color=GRAY, ls="--", lw=1.2, label="линейная  CL0 + CLα·α")
    ax.plot(a, cl_flat, color=GRAY, ls=":", lw=1.4, label="плоская пластина")
    ax.plot(a, cl, color=BLUE, lw=2.4, label="модель (смешение по σ)")
    _alpha_marks(ax)
    ax.set_ylim(-0.4, 1.6)
    ax.set(title="Подъёмная сила CL(α)", xlabel="α, °", ylabel="CL")
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[1]
    ax.plot(a, cd, color=RED, lw=2.4, label="модель")
    ax.plot(a, ac.CDp + cl_lin**2 / (np.pi * ac.e_oswald * ac.b**2 / ac.S), color=GRAY, ls="--",
            lw=1.2, label="поляра без срыва")
    _alpha_marks(ax, label=False)
    ax.set_ylim(0, 0.6)
    ax.set(title="Сопротивление CD(α)", xlabel="α, °", ylabel="CD")
    ax.legend(fontsize=8, loc="upper left")
    ax = axs[2]
    for de, c in ((-10, GREEN), (0, PURPLE), (10, ORANGE)):
        cm = np.array([coef_Cm(v, 0, rad(de), 20, ac) for v in x])
        ax.plot(a, cm, color=c, lw=2.2, label=f"δe = {de:+d}°")
    ax.axhline(0, color="k", lw=0.8)
    _alpha_marks(ax, label=False)
    ax.set(title="Момент тангажа Cm(α)", xlabel="α, °", ylabel="Cm")
    ax.legend(fontsize=8)
    ax.text(0.03, 0.04, "Cm(α) убывает → ЛА статически устойчив\nδe > 0 (руль вниз) — момент на пикирование",
            transform=ax.transAxes, fontsize=8, color="#444")
    fig.tight_layout()
    save(fig, "aero_coeffs.png")


def fig_polar():
    a = np.linspace(-4, 16, 401)
    x = rad(a)
    cl = np.array([coef_CL(v, 0, 0, 20, ac) for v in x])
    cd = np.array([coef_CD(v, ac) for v in x])
    k = cl / cd
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    ax.plot(cd, cl, color=BLUE, lw=2.4)
    for av in (0, 4, 8, 12.2, 15):
        i = int(np.argmin(abs(a - av)))
        ax.plot(cd[i], cl[i], "o", color=BLUE)
        ax.annotate(f"α = {av:g}°", (cd[i], cl[i]), xytext=(8, -4), textcoords="offset points",
                    fontsize=8.5)
    i = int(np.argmax(k))
    ax.plot([0, cd[i] * 1.6], [0, cl[i] * 1.6], color=GREEN, ls="--", lw=1.2)
    ax.annotate(f"касательная из начала:\nK_max = {k[i]:.1f} при α = {a[i]:.1f}°", (cd[i], cl[i]),
                xytext=(25, -40), textcoords="offset points", fontsize=8.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set(title="Поляра CL(CD)", xlabel="CD", ylabel="CL", xlim=(0, 0.14))
    ax = axs[1]
    ax.plot(a, k, color=GREEN, lw=2.4)
    ax.axvline(a[i], color=GREEN, ls="--", lw=1)
    _alpha_marks(ax)
    ax.set(title="Аэродинамическое качество K = CL / CD", xlabel="α, °", ylabel="K")
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    save(fig, "polar.png")


def fig_envelope():
    vs = np.linspace(8.0, 32.0, 121)
    pts = [trim_point(v) for v in vs]
    Tav = [propeller(1.0, v, ac)[0] for v in vs]
    Vst, Vmax = stall_speed(), max_level_speed()
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.4))
    ax = axs[0, 0]
    ax.plot(vs, [p["D"] for p in pts], color=RED, lw=2.4, label="потребная тяга = сопротивление D")
    ax.plot(vs, Tav, color=BLUE, lw=2.4, label="располагаемая тяга (полный газ)")
    ax.fill_between(vs, [p["D"] for p in pts], Tav, where=np.array(Tav) > [p["D"] for p in pts],
                    color=BLUE, alpha=0.08, label="избыток тяги — набор высоты")
    ax.axvline(Vst, color=RED, ls="--", lw=1)
    ax.axvline(Vmax, color=BLUE, ls="--", lw=1)
    ax.text(Vst + 0.3, 12.5, f"сваливание\n{Vst:.1f} м/с", fontsize=8.5, color=RED)
    ax.text(Vmax - 0.3, 12.5, f"Va_max\n{Vmax:.1f} м/с", fontsize=8.5, color=BLUE, ha="right")
    j = int(np.argmin([p["D"] for p in pts]))
    ax.plot(vs[j], pts[j]["D"], "o", color=RED)
    ax.annotate(f"минимум сопротивления: {vs[j]:.1f} м/с,\nK = {pts[j]['LD']:.1f} — наибольшая дальность",
                (vs[j], pts[j]["D"]), xytext=(15, 28), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set(title="Тяга горизонтального полёта", xlabel="Va, м/с", ylabel="Н", ylim=(0, 24))
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[0, 1]
    ax.plot(vs, [p["P"] for p in pts], color=PURPLE, lw=2.4)
    p16 = trim_point(16.0)
    ax.plot(16, p16["P"], "o", color=PURPLE)
    ax.annotate(f"крейсер: {p16['P']:.0f} Вт", (16, p16["P"]), xytext=(-80, 25),
                textcoords="offset points", fontsize=8.5)
    ax.text(8.6, 195, "мощность падает до самого сваливания:\nеё минимум лежит ниже скорости сваливания",
            fontsize=8.5, color="#555")
    ax.axvline(16, color=GRAY, ls=":", lw=1)
    ax.set(title="Потребная мощность T·Va", xlabel="Va, м/с", ylabel="Вт", ylim=(0, 260))
    ax = axs[1, 0]
    ax.plot(vs, [p["alpha"] for p in pts], color=BLUE, lw=2.2, label="α, °")
    ax.plot(vs, [p["de"] for p in pts], color=ORANGE, lw=2.2, label="δe, °")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(16, color=GRAY, ls=":", lw=1)
    ax.set(title="Балансировочные угол атаки и руль высоты", xlabel="Va, м/с", ylabel="°")
    ax.legend(fontsize=8.5)
    ax = axs[1, 1]
    ax.plot(vs, [p["dt"] for p in pts], color=GREEN, lw=2.4)
    ax.axvline(16, color=GRAY, ls=":", lw=1)
    ax.text(16.3, 0.15, "крейсер 16 м/с", fontsize=8.5, color="#555")
    ax.set(title="Балансировочный газ δt", xlabel="Va, м/с", ylabel="δt", ylim=(0, 1.05))
    fig.tight_layout()
    save(fig, "envelope.png")


def fig_propeller():
    vs = np.linspace(0, 32, 161)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for dt, c in ((0.25, "#9ab3d8"), (0.5, "#6b8fc7"), (0.75, "#3b6fb6"), (1.0, "#1b3f7a")):
        T = np.array([propeller(dt, v, ac)[0] for v in vs])
        axs[0].plot(vs, T, color=c, lw=2.2, label=f"δt = {dt:g}")
        axs[1].plot(vs, np.maximum(T, 0) * vs, color=c, lw=2.2, label=f"δt = {dt:g}")
    axs[0].axhline(0, color="k", lw=0.7)
    axs[0].set(title="Тяга винта T(Va)", xlabel="Va, м/с", ylabel="T, Н")
    axs[0].text(0.5, -3.3, "T < 0: винт тормозит (малый газ на скорости)", fontsize=8.5, color="#555")
    axs[1].set(title="Полезная мощность винта T·Va", xlabel="Va, м/с", ylabel="Вт")
    for ax in axs:
        ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "propeller.png")


def fig_modes():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    for Va, mk in ((16.0, "o"), (22.0, "s")):
        e = linearize(Va)
        for ax, key in ((axs[0], "lon"), (axs[1], "lat")):
            ax.plot(e[key].real, e[key].imag, mk, ms=8, mfc="none", mew=1.8,
                    color=BLUE if Va == 16 else ORANGE, label=f"Va = {Va:.0f} м/с")
    for ax, title in ((axs[0], "Продольное движение [u, w, q, θ]"),
                      (axs[1], "Боковое движение [v, p, r, φ]")):
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(title=title, xlabel="Re λ, 1/с", ylabel="Im λ, рад/с")
        ax.legend(fontsize=8.5, loc="upper left")
    axs[0].annotate("короткопериодическое", (-13.4, 8.8), xytext=(-12, 13), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#555"))
    axs[0].annotate("фугоида (у нуля)", (-0.2, 0.47), xytext=(-9, 3.5), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#555"))
    axs[1].annotate("крен (апериодика)", (-55.8, 0), xytext=(-60, 4), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#555"))
    axs[1].annotate("голландский шаг", (-1.9, 7.2), xytext=(-40, 9), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#555"))
    axs[1].annotate("спиральная (λ > 0, медленно)", (0.02, 0), xytext=(-40, -6), fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color="#555"))
    axs[0].set_ylim(-16, 16)
    axs[1].set_ylim(-12, 12)
    fig.tight_layout()
    save(fig, "modes.png")


def fig_actuator():
    dt, t = 0.01, np.arange(0, 0.35, 0.001)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0))
    for ax, step in ((axs[0], 2.0), (axs[1], 20.0)):
        pos = actuator_init([0, 0, 0, 0], ac)
        ys, ts = [], []
        for k in range(int(0.35 / dt) + 1):
            ys.append(deg(pos[0]))
            ts.append(k * dt)
            pos = actuator_step(pos, [rad(step) if k * dt >= 0.05 else 0, 0, 0, 0], dt, ac)
        ax.step(ts, [step if x >= 0.05 else 0 for x in ts], where="post", color="k", ls="--",
                lw=1.2, label="команда САУ δ_cmd")
        ax.plot(ts, ys, "-o", ms=3, color=BLUE, lw=2, label="положение руля δ")
        ax.set(xlabel="t, с", ylabel="δe, °")
        ax.legend(fontsize=8.5, loc="lower right")
    axs[0].set_title(f"Малая команда 2°: апериодическое звено, τ = {ac.servo_tau * 1000:.0f} мс")
    axs[1].set_title(f"Большая команда 20°: упор в скорость {deg(ac.servo_rate):.0f} °/с")
    fig.tight_layout()
    save(fig, "actuator.png")


def fig_gear():
    wh, hd = np.array(ac.gear_wheels), np.array(ac.gear_hard)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]                                   # вид сбоку: x вперёд → вправо, z вниз → вниз
    ax.plot(hd[:, 0], -hd[:, 2], "s", color=RED, ms=7, label="точки конструкции")
    ax.plot(wh[:, 0], -wh[:, 2], "o", color=BLUE, ms=11, label="колёса")
    ax.plot(0, 0, "+", color="k", ms=14, mew=2, label="центр масс")
    ax.axhline(-wh[1, 2], color=GREEN, lw=1.2, ls="--", label="земля на стоянке")
    ax.set(title="Вид сбоку (нос вправо)", xlabel="x, м", ylabel="−z (вверх), м")
    ax.set_aspect("equal")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    ax = axs[1]                                   # вид сверху: y вправо, x вверх
    ax.plot(hd[:, 1], hd[:, 0], "s", color=RED, ms=7)
    ax.plot(wh[:, 1], wh[:, 0], "o", color=BLUE, ms=11)
    ax.plot(0, 0, "+", color="k", ms=14, mew=2)
    ax.set(title="Вид сверху (нос вверх)", xlabel="y (правое крыло), м", ylabel="x, м")
    ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "gear.png")


FIGS = [fig_aero_coeffs, fig_polar, fig_envelope, fig_propeller, fig_modes, fig_actuator, fig_gear]

if __name__ == "__main__":
    tables()
    if "tables" not in sys.argv[1:]:
        for f in FIGS:
            f()

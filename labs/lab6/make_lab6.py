# -*- coding: utf-8 -*-
"""
Лабораторная 6 (лекция 6 «Синтез САУ»): схемы-заготовки для BlockLab3D, иллюстрации
и эталонные числа для преподавателя. Код проекта не меняется — здесь только безоконный
аналог BlockLab3D: штатная САУ (TECS, TruthSensors) + схема из блоков; уставки и импульсы
— как кнопки редактора.

    python labs/lab6/make_lab6.py          # schemes/*.json, img/*.png, печать чисел
"""
import os, sys, math, json, copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import H, THETA, PHI, X, Y, air_data, earth_velocity
from control.controllers import PitchControlParams, wrap_angle
from control.blocks import Diagram, sau_signals, templates, _blk, _w
from control.sau import SAU, SAUParams, TruthSensors, Mode
from runner import run, compute_trim, trim_state

AC = AircraftParams()
DT = 0.01
KICKS = {"de": (0, math.radians(3.0), 1.0), "da": (2, math.radians(5.0), 0.5),
         "dr": (3, math.radians(5.0), 1.0)}
TPL = templates(AC, 16.0)
C1, C2, C3, INK, GRID = "#1f6fd1", "#e08a00", "#c2337a", "#14233a", "#d5dde8"


# --- безоконный BlockLab3D ----------------------------------------------------------
def fly(scheme, events, t_end=30.0, mode="alt", h0=200.0):
    cfg = SimConfig(Va0=16.0, h0=h0, theta0=0.0, dt=DT, t_end=t_end)
    trim = compute_trim(AC, 16.0)
    sau = SAU(AC, trim, TruthSensors(), SAUParams(pitch=PitchControlParams(Va_ref=16.0)))
    sau.engage(theta=trim[0], h=h0, Va=16.0, chi=0.0)
    sau.set_mode(Mode.ALTITUDE if mode == "alt" else Mode.PITCH)
    dia = Diagram(scheme) if scheme else None
    ev, st = sorted(events), {"i": 0, "kick": None, "until": -1.0}
    rec = {"t": [], "theta_ref": [], "h_ref": [], "chi_ref": [], "vals": []}

    def fn(t, s, Va, alpha):
        while st["i"] < len(ev) and t >= ev[st["i"]][0] - DT / 2:
            _, kind, val = ev[st["i"]]
            st["i"] += 1
            if kind == "theta":
                if sau.mode is not Mode.PITCH:
                    sau.set_mode(Mode.PITCH)
                sau.refs.theta += math.radians(val)
            elif kind == "h":
                if sau.mode is Mode.PITCH:
                    sau.refs.h = float(s[H])
                sau.refs.h += val
                sau.set_mode(Mode.ALTITUDE)
            elif kind == "va":
                sau.refs.Va += val
            elif kind == "chi":
                sau.refs.chi = wrap_angle(sau.refs.chi + math.radians(val))
            elif kind == "kick":
                st["kick"], st["until"] = val, t + KICKS[val][2]
        _, _, beta = air_data(s, (0.0, 0.0, 0.0))
        c = sau.step(t, s, Va, alpha, beta, DT).copy()
        if dia is not None:
            out = dia.step(sau_signals(sau, s, alpha, beta), t, DT)
            for idx, v in dia.commands(out).items():
                c[idx] = v
            rec["vals"].append(dict(dia.values))
        if t < st["until"]:
            idx, d, _ = KICKS[st["kick"]]
            c[idx] += d
        rec["t"].append(t)
        rec["theta_ref"].append(math.degrees(sau.theta_ref))
        rec["h_ref"].append(sau.refs.h)
        rec["chi_ref"].append(math.degrees(sau.refs.chi))
        return c

    log = run(fn, AC, WindParams(), cfg, state0=trim_state(AC, cfg))
    for k in ("t", "theta_ref", "h_ref", "chi_ref"):
        rec[k] = np.array(rec[k])
    log.rec = rec
    log.chi = np.degrees(np.unwrap([math.atan2(v[1], v[0]) for v in map(earth_velocity, log.state)]))
    return log


def metrics(t, y, y0, y1, t0, tube=0.05):
    """σ, %; время установления в трубке tube·|Δ|, с; остаточная ошибка."""
    m = t >= t0
    tt, yy, d = t[m], y[m], y1 - y0
    over = max(0.0, (np.max((yy - y0) * np.sign(d)) - abs(d)) / abs(d) * 100)
    out = np.abs(yy - y1) > tube * abs(d)
    ts = (tt[np.where(out)[0][-1]] - t0) if out.any() else 0.0
    return over, ts, yy[-1] - y1


# --- схемы ------------------------------------------------------------------------
def pitch_p(Kp):
    """Задание 1, часть А: только П по θ."""
    b = [_blk("in_theta_ref", "Input", 20, 30, signal="theta_ref"),
         _blk("in_theta", "Input", 20, 110, signal="theta"),
         _blk("in_de_trim", "Input", 20, 190, signal="de_trim"),
         _blk("s", "Sum", 200, 30, signs="+-"), _blk("k", "P", 290, 38, Kp=Kp),
         _blk("neg", "Gain", 450, 38, K=-1.0), _blk("s2", "Sum", 640, 40, signs="++"),
         _blk("sat", "Sat", 720, 48, min=-25, max=25), _blk("out", "Output", 880, 48, act="de")]
    w = [_w("in_theta_ref", "s", 0), _w("in_theta", "s", 1), _w("s", "k"), _w("k", "neg"),
         _w("neg", "s2", 0), _w("in_de_trim", "s2", 1), _w("s2", "sat"), _w("sat", "out")]
    return {"blocks": b, "wires": w}


def pitch_cascade(Kth, Kq):
    """Задание 1, часть Б: θ → q_ref → δe."""
    b = [_blk("in_theta_ref", "Input", 20, 30, signal="theta_ref"),
         _blk("in_theta", "Input", 20, 110, signal="theta"),
         _blk("in_q", "Input", 20, 190, signal="q"),
         _blk("in_de_trim", "Input", 20, 270, signal="de_trim"),
         _blk("s", "Sum", 200, 30, signs="+-"), _blk("kth", "P", 280, 38, Kp=Kth),
         _blk("sq", "Sum", 440, 40, signs="+-"), _blk("kq", "P", 520, 48, Kp=Kq),
         _blk("neg", "Gain", 680, 48, K=-1.0), _blk("s2", "Sum", 860, 50, signs="++"),
         _blk("sat", "Sat", 940, 58, min=-25, max=25), _blk("out", "Output", 1100, 58, act="de")]
    w = [_w("in_theta_ref", "s", 0), _w("in_theta", "s", 1), _w("s", "kth"), _w("kth", "sq", 0),
         _w("in_q", "sq", 1), _w("sq", "kq"), _w("kq", "neg"), _w("neg", "s2", 0),
         _w("in_de_trim", "s2", 1), _w("s2", "sat"), _w("sat", "out")]
    return {"blocks": b, "wires": w}


def altitude_trap():
    """Задание 2: «Штатная САУ: высота + тангаж», но у ПИ высоты нет пределов (y_max, I_max)."""
    s = copy.deepcopy(TPL["Штатная САУ: высота + тангаж"])
    for b in s["blocks"]:
        if b["id"] == "k_h":
            b["p"].update(y_max=0.0, i_max=0.0)
    return s


def altitude_var(**p):
    s = copy.deepcopy(TPL["Штатная САУ: высота + тангаж"])
    for b in s["blocks"]:
        if b["id"] == "k_h":
            b["p"].update(p)
    return s


def course_trap():
    """Задание 3: штатный боковой канал, но ошибка курса — обычным сумматором."""
    s = copy.deepcopy(TPL["Штатная САУ: курс + крен + скольжение"])
    for b in s["blocks"]:
        if b["id"] == "e_chi":
            b["type"], b["p"] = "Sum", {"signs": "+-"}
    return s


def course_var(Kchi=None, phimax=None):
    s = copy.deepcopy(TPL["Штатная САУ: курс + крен + скольжение"])
    for b in s["blocks"]:
        if Kchi is not None and b["id"] == "k_chi":
            b["p"]["K"] = Kchi
        if phimax is not None and b["id"] == "sat_phi":
            b["p"].update(min=-phimax, max=phimax)
    return s


# --- оформление графиков --------------------------------------------------------------
def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#8a97a8")
    ax.tick_params(colors=INK, labelsize=9)


plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.titleweight": "bold",
                     "axes.titlesize": 11, "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(HERE, "img", name), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(os.path.join(HERE, "schemes"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    for name, sch in [("lab6_2_vysota_oshibka.json", altitude_trap()),
                      ("lab6_3_kurs_oshibka.json", course_trap())]:
        with open(os.path.join(HERE, "schemes", name), "w", encoding="utf-8") as f:
            json.dump(sch, f, ensure_ascii=False, indent=1)

    # ---------- Задание 1 ----------
    print("== Задание 1: тангаж, ступенька θ_ref +5° ==")
    step = [(2.0, "theta", 5.0)]

    def th_m(L):
        th = np.degrees(L.state[:, THETA])
        return metrics(L.t, th, th[150], th[150] + 5.0, 2.0)

    print("А. П по θ:")
    for Kp in (0.3, 0.6, 1.5, 3.0, 3.5, 4.0):
        o, ts, _ = th_m(fly(pitch_p(Kp), step, 12, "pitch"))
        print(f"   Kp = {Kp:4.1f}: σ = {o:5.0f} %, t_уст = {ts:5.2f} с")
    print("Б. каскад, Kθ·Kq = 3:")
    for Kq in (0.02, 0.0381, 0.08, 0.15):
        o, ts, _ = th_m(fly(pitch_cascade(3 / Kq, Kq), step, 12, "pitch"))
        print(f"   Kq = {Kq:6.4f}, Kθ = {3 / Kq:6.1f}: σ = {o:5.0f} %, t_уст = {ts:5.2f} с")
    print("В. Kq = 0.15, рост Kθ:")
    for Kth in (10, 16.2, 30, 45, 60):
        o, ts, _ = th_m(fly(pitch_cascade(Kth, 0.15), step, 12, "pitch"))
        print(f"   Kθ = {Kth:5.1f}: σ = {o:5.0f} %, t_уст = {ts:5.2f} с")
    o, ts, _ = th_m(fly(TPL["Штатная САУ: тангаж"], step, 12, "pitch"))
    print(f"   штатная (Kθ = 16.2, Kq = 0.0381): σ = {o:.0f} %, t_уст = {ts:.2f} с")

    # рис. 1: как снимать σ и t_уст (П, Kp = 1.5)
    L = fly(pitch_p(1.5), step, 8, "pitch")
    th, t = np.degrees(L.state[:, THETA]), L.t
    y0 = th[150]
    o, ts, _ = metrics(t, th, y0, y0 + 5, 2.0)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axhspan(y0 + 5 * 0.95, y0 + 5 * 1.05, color=C1, alpha=0.10, lw=0)
    ax.plot(t, L.rec["theta_ref"], "--", color=INK, lw=1.4, label="θ_ref")
    ax.plot(t, th, color=C1, lw=2.4, label="θ (П-регулятор, Kp = 1.5)")
    i_max = np.argmax(th)
    ax.annotate(f"σ = {o:.0f} %", (t[i_max], th[i_max]), xytext=(t[i_max] + 0.8, th[i_max] + 0.2),
                color=C1, fontweight="bold", arrowprops=dict(arrowstyle="->", color=C1))
    ax.annotate("", (2.0, y0 + 2.3), (2.0 + ts, y0 + 2.3), arrowprops=dict(arrowstyle="<->", color=C3))
    ax.text(2.0 + ts + 0.1, y0 + 2.3, f"t_уст ≈ {ts:.1f} с", color=C3, va="center", fontweight="bold")
    ax.text(4.4, y0 + 5 * 1.05 + 0.1, "трубка ±5 %", color=C1, fontsize=9)
    ax.set_xlim(1.5, 5)
    style(ax, "тангаж, °")
    ax.set_xlabel("t, с", color=INK)
    ax.legend(loc="lower right")
    ax.set_title("Показатели качества переходного процесса: ступенчатое воздействие θ_ref +5°", color=INK, loc="left")
    save(fig, "fig1_metrics.png")

    # ---------- Задание 2 ----------
    print("\n== Задание 2: высота, ступенька h_ref +10 и +50 м ==")
    for name, sch in [("исходная (без ограничений)", altitude_trap()),
                      ("I_max = 15°", altitude_var(y_max=0.0)),
                      ("y_max = 15° (условное интегрирование)", altitude_var()),
                      ("без И (Ki = 0)", altitude_var(Ki=0.0))]:
        for dh in (10, 50):
            L = fly(sch, [(2.0, "h", dh)], 60)
            o, ts, e = metrics(L.t, L.state[:, H], 200, 200 + dh, 2.0)
            print(f"   {name:40s} Δh = {dh:2d} м: σ = {o:5.1f} %, h_max = {L.state[:, H].max():6.1f} м, "
                  f"t_уст = {ts:5.1f} с")
    L = fly(altitude_var(Ki=0.0), [(2.0, "va", -4.0)], 60)
    print(f"   без И, Va_ref 16 → 12 м/с: h − h_ref в конце = {L.state[-1, H] - 200:+.2f} м")
    L = fly(altitude_var(), [(2.0, "va", -4.0)], 60)
    print(f"   штатная, Va_ref 16 → 12 м/с: h − h_ref в конце = {L.state[-1, H] - 200:+.2f} м")

    # рис. 2: переходный процесс в схеме с внесённой ошибкой
    L = fly(altitude_trap(), [(2.0, "h", 50.0)], 60)
    Lg = fly(altitude_var(), [(2.0, "h", 50.0)], 60)
    integ = np.array([v.get("k_h", np.nan) for v in L.rec["vals"]])
    th_sch = np.array([v.get("sat_th", np.nan) for v in L.rec["vals"]])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1]})
    a1.plot(L.t, L.rec["h_ref"], "--", color=INK, lw=1.4, label="h_ref")
    a1.plot(L.t, L.state[:, H], color=C2, lw=2.6, label="h — схема с внесённой ошибкой")
    hm = np.argmax(L.state[:, H])
    a1.annotate(f"h_max = {L.state[hm, H]:.0f} м", (L.t[hm], L.state[hm, H]),
                xytext=(L.t[hm] + 6, L.state[hm, H] - 6), color=C2, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C2))
    style(a1, "высота, м")
    a1.legend(loc="lower right")
    a1.set_title("Задание 2: переходный процесс по высоте, ступенчатое воздействие h_ref +50 м", color=INK, loc="left")
    a2.axhspan(-15, 15, color=C2, alpha=0.10, lw=0)
    a2.plot(L.t, integ, color=C3, lw=2.2, label="выход ПИ высоты (до насыщения)")
    a2.plot(L.t, th_sch, color=C2, lw=1.6, label="θ_ref (после насыщения ±15°)")
    style(a2, "°")
    a2.set_xlabel("t, с", color=INK)
    a2.legend(loc="upper right", fontsize=9)
    save(fig, "fig2_windup.png")

    # ---------- Задание 3 ----------
    print("\n== Задание 3: курс ==")
    ev = [(2.0 + 6 * k, "chi", 30.0) for k in range(7)]
    Ltr, Lok = fly(course_trap(), ev, 60), fly(TPL["Штатная САУ: курс + крен + скольжение"], ev, 60)
    print(f"   с ошибкой: χ в конце {Ltr.chi[-1]:.0f}°, минимум χ {Ltr.chi.min():.0f}°;"
          f" штатная: χ в конце {Lok.chi[-1]:.0f}°")
    for K in (0.3, 1.0, 3.0, 10.0):
        for pm in (30.0, 80.0):
            L = fly(course_var(K, pm), [(2.0, "chi", 30.0)] * 3, 25)
            o, ts, _ = metrics(L.t, L.chi, 0, 90, 2.0)
            print(f"   Kχ = {K:4.1f} 1/с, φ_max = {pm:2.0f}°: σ = {o:5.1f} %, t_уст = {ts:5.1f} с, "
                  f"|φ|max = {np.degrees(np.abs(L.state[:, PHI])).max():3.0f}°, "
                  f"потеря h = {200 - L.state[:, H].min():.1f} м")

    # рис. 3: вид сверху
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for L, col, lab, lw in [(Lok, "#7d8ba0", "штатная схема", 1.8), (Ltr, C3, "схема с внесённой ошибкой", 2.8)]:
        ax.plot(L.state[:, Y], L.state[:, X], color=col, lw=lw, label=lab)
        ax.plot(L.state[-1, Y], L.state[-1, X], "o", color=col, ms=7)
    i7 = np.argmin(np.abs(Ltr.t - 38.0))
    ax.annotate("χ_ref: 180° → 210° (седьмое воздействие)\nразворот на 330° влево вместо 30° вправо",
                (Ltr.state[i7, Y], Ltr.state[i7, X]),
                xytext=(Ltr.state[i7, Y] - 420, Ltr.state[i7, X] - 150), color=C3, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C3))
    ax.plot(0, 0, "s", color=INK, ms=7)
    ax.text(10, 15, "старт, χ = 0° (на север)", color=INK, fontsize=9)
    ax.set_aspect("equal")
    style(ax, "север, м")
    ax.set_xlabel("восток, м", color=INK)
    ax.legend(loc="lower left")
    ax.set_title("Задание 3: траектории в горизонтальной плоскости, семь воздействий χ_ref +30°", color=INK, loc="left")
    save(fig, "fig3_course_top.png")
    print("\nготово:", os.path.join(HERE, "img"))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Структурные схемы САУ для docs/control.md и docs/SENSORS.md (docs/img/*.png).

    python docs/figures/sau_diagrams.py            # все схемы
    python docs/figures/sau_diagrams.py pitch      # одна (имя функции без fig_)

Стиль и палитра — общие с diagrams.py: синие блоки — звенья САУ, жёлтые —
ограничения, фиолетовые — объект управления, зелёные — датчики. Числа на схемах —
расчёт control/tuning.py для FPV-самолёта на 16 м/с.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from diagrams import canvas, box, arrow, poly, right, left, top, bottom, save, C, TXT

GRAY_T = "#555555"
plt.rcParams["mathtext.fontset"] = "dejavusans"


# ---------------------------------------------------------------------------
# Элементы структурных схем
# ---------------------------------------------------------------------------
def summ(ax, x, y, r=0.2, signs=None):
    """Сумматор: круг с крестом; signs — {сторона: знак}, сторона из 'l', 'b', 't', 'r'."""
    ax.add_patch(Circle((x, y), r, fc="white", ec=TXT, lw=1.4, zorder=3))
    d = r * 0.707
    ax.plot([x - d, x + d], [y - d, y + d], color=TXT, lw=0.9, zorder=4)
    ax.plot([x - d, x + d], [y + d, y - d], color=TXT, lw=0.9, zorder=4)
    off = {"l": (-r - 0.13, r * 0.75), "b": (-r * 0.85, -r - 0.14), "t": (r * 0.85, r + 0.14),
           "r": (r + 0.13, r * 0.75)}
    for side, s in (signs or {}).items():
        dx, dy = off[side]
        ax.text(x + dx, y + dy, s, ha="center", va="center", fontsize=12 if s == "−" else 11,
                fontweight="bold", color="#c0392b" if s == "−" else TXT, zorder=4)
    return (x - r, y - r, 2 * r, 2 * r)


def sat(ax, x, y, w, h, label, fs=8.5):
    """Звено насыщения: значок характеристики и подпись предела."""
    b = box(ax, x, y, w, h, "", None, "lim")
    cx, cy = x + w / 2, y + h * 0.63
    s = min(w, h) * 0.22
    ax.plot([cx - 1.6 * s, cx - 0.6 * s, cx + 0.6 * s, cx + 1.6 * s],
            [cy - 0.6 * s, cy - 0.6 * s, cy + 0.6 * s, cy + 0.6 * s], color=C["lim"][1], lw=1.6,
            zorder=4)
    ax.text(cx, y + h * 0.24, label, ha="center", va="center", fontsize=fs, color=TXT, zorder=4)
    return b


def tf(ax, x, y, w, h, text, sub=None, role="ctrl", fs=11, sfs=8.0):
    """Звено с передаточной функцией / формулой."""
    return box(ax, x, y, w, h, text, sub, role, fs=fs, sfs=sfs)


def sig(ax, x, y, text, ha="center", va="bottom", fs=9, color="#333333", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=5, **kw)


def dot(ax, x, y):
    ax.plot(x, y, "o", color=TXT, ms=4.5, zorder=5)


def frame(ax, x, y, w, h, title, color="#8a8a8a", fc="none", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec=color, lw=1.3, ls=(0, (5, 3)), zorder=0))
    ax.text(x + 0.15, y + h - 0.12, title, ha="left", va="top", fontsize=fs, color=color,
            fontweight="bold", zorder=1)


def hline(ax, x0, x1, y, text=None, tx=None, head=True, fs=9):
    arrow(ax, (x0, y), (x1, y), style="-|>" if head else "-")
    if text:
        sig(ax, (x0 + x1) / 2 if tx is None else tx, y + 0.06, text, fs=fs)


def note(ax, x, y, text, fs=9, color=GRAY_T, ha="left", va="top"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.45)


def feedback(ax, pl, y, n_in, n_out, x_in, x_out, y_in, y_out, l_in, l_out):
    """Две обратные связи от объекта без пересечений: внутренняя (скорость) — с нижнего
    выхода к ближнему сумматору, внешняя (угол) — с верхнего выхода к дальнему."""
    xr = pl[0] + pl[2]
    poly(ax, [(xr, y - 0.3), (xr + 0.25, y - 0.3), (xr + 0.25, y_in), (x_in, y_in), (x_in, y - 0.2)])
    poly(ax, [(xr, y + 0.3), (xr + 0.5, y + 0.3), (xr + 0.5, y_out), (x_out, y_out), (x_out, y - 0.2)])
    sig(ax, xr + 0.08, y - 0.27, n_in, ha="left", va="top", fs=10)
    sig(ax, xr + 0.08, y + 0.35, n_out, ha="left", va="bottom", fs=10)
    sig(ax, (x_in + xr) / 2, y_in + 0.05, l_in, fs=9)
    sig(ax, (x_out + xr) / 2, y_out + 0.05, l_out, fs=9)


# ---------------------------------------------------------------------------
def fig_sau_structure():
    """Вся САУ: control.sau.SAU в контуре с объектом управления."""
    fig, ax = canvas(17.4, 8.9)
    sp = box(ax, 0.2, 4.05, 2.1, 1.55, "Уставки", "оператор, стенд,\nсценарий:\nh, Va, χ, θ, режим",
             "data", sfs=8.5)
    frame(ax, 2.65, 0.25, 10.75, 8.4, "SAU — control/sau.py", C["ctrl"][1])
    nav = box(ax, 3.0, 6.55, 2.7, 1.3, "Навигатор  nav", "только в режиме ROUTE:\nмаршрут → χ_ref, h_ref, κ, $\\dot{h}$_ff",
              "ctrl", sfs=8.2)
    refs = box(ax, 3.0, 4.05, 2.7, 1.55, "Уставки  refs", "h, Va, χ, κ, $\\dot{h}$\n(в режиме PITCH — θ)", "data",
               sfs=8.5)
    arrow(ax, right(sp), left(refs))
    poly(ax, [top(sp), (1.25, 7.5), (3.0, 7.5)])
    sig(ax, 1.35, 7.56, "маршрут", ha="left", fs=8.5)
    arrow(ax, bottom(nav), top(refs), "ROUTE", toff=(0.35, -0.05))

    frame(ax, 6.25, 2.0, 6.85, 6.15, "Контуры  _control()", "#7f7f7f", fc="#fafafa", fs=9.5)
    lon = box(ax, 6.55, 5.95, 2.05, 1.45, "lon", "TECS (по умолч.)\nили раздельные\nh, Va → θ_cmd, δt",
              "ctrl", sfs=8.0)
    prot = box(ax, 9.0, 6.1, 1.6, 1.15, "prot", "защита по α\n(может не быть)", "lim", sfs=8.0)
    pit = box(ax, 11.0, 5.95, 1.85, 1.45, "pitch", "тангаж\nθ → q → δe", "ctrl", sfs=8.5)
    lat = box(ax, 6.55, 2.65, 6.3, 2.05, "lat — LateralController",
              "курс:  χ_ref, κ → φ_ref        крен (lat.roll):  φ_ref → p_ref → δa\n"
              "скольжение:  β → δr,  + микс элеронов в РН  δr += K·δa", "ctrl", sfs=8.6)
    arrow(ax, right(lon, 0.1), left(prot, 0.1), "θ_cmd", toff=(0, 0.04))
    arrow(ax, right(prot, 0.1), left(pit, 0.1), "θ_ref", toff=(0, 0.04))
    poly(ax, [right(refs, 0.35), (6.1, 5.18), (6.1, 6.68), (6.55, 6.68)])
    sig(ax, 6.05, 6.75, "h, Va, θ", ha="right")
    poly(ax, [right(refs, -0.35), (6.1, 4.48), (6.1, 3.68), (6.55, 3.68)])
    sig(ax, 6.05, 3.38, "χ, κ", ha="right", va="top")

    # выходы → приводы → ЛА
    act = box(ax, 13.7, 2.65, 1.3, 4.75, "", None, "plant")
    ax.text(14.35, 5.0, "Приводы рулей\nsim/actuators.py", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color=TXT, linespacing=1.4)
    pl = box(ax, 15.45, 2.65, 1.75, 4.75, "", None, "plant")
    ax.text(16.32, 5.0, "ЛА 6DOF\nderivatives() + ветер, RK4", rotation=90, ha="center",
            va="center", fontsize=10, fontweight="bold", color=TXT, linespacing=1.4)
    hline(ax, 12.85, 13.7, 6.68, "δe")
    poly(ax, [(7.6, 5.95), (7.6, 5.3), (13.7, 5.3)])
    sig(ax, 13.25, 5.36, "δt")
    hline(ax, 12.85, 13.7, 4.1, "δa")
    hline(ax, 12.85, 13.7, 3.2, "δr")
    for y in (6.68, 5.3, 4.1, 3.2):
        arrow(ax, (15.0, y), (15.45, y))
    sig(ax, 14.35, 2.35, "положения\nрулей", va="top", fs=8.5)

    # обратная связь: состояние → датчики → оценка
    sen = box(ax, 10.6, 0.55, 2.3, 1.05, "sensors", "Truth | Noisy", "sens", sfs=8.5)
    est = box(ax, 7.4, 0.55, 2.4, 1.05, "estimator", "нет: est = meas", "gray", sfs=8.5)
    poly(ax, [bottom(pl), (16.32, 1.07), (12.9, 1.07)])
    sig(ax, 14.75, 1.13, "истинное состояние, Va, α, β")
    arrow(ax, left(sen), right(est), "meas", toff=(0, 0.04))
    arrow(ax, (8.6, 1.6), (8.6, 2.0))
    sig(ax, 8.72, 1.66, "est: h, $\\dot{h}$, Va, θ, q, φ, p, χ, Vg, β, [α, pn, pe]", ha="left", fs=8.5)
    poly(ax, [left(est), (2.85, 1.07), (2.85, 6.9), (3.0, 6.9)])
    sig(ax, 2.97, 2.2, "pn, pe, χ", ha="left", fs=8.5, rotation=90)
    note(ax, 0.2, 3.7, "Режимы  Mode:\nPITCH — θ и курс\nALTITUDE — h и курс\nROUTE — маршрут\n\n"
         "Скорость Va\nдержится всегда", fs=8.8)
    save(fig, "sau_structure.png")


def fig_pitch():
    """Каскад тангажа PitchController: θ → q_ref → δe."""
    fig, ax = canvas(16.4, 5.4)
    y = 3.4
    sig(ax, 0.25, y + 0.08, "θ_ref", ha="left", fs=10)
    s1 = summ(ax, 1.35, y, signs={"l": "+", "b": "−"})
    arrow(ax, (0.25, y), (1.15, y))
    k1 = tf(ax, 1.9, y - 0.45, 1.25, 0.9, "Kθ", "16.2 1/с")
    arrow(ax, (1.55, y), left(k1))
    s_q = sat(ax, 3.55, y - 0.45, 1.3, 0.9, "±q_max")
    arrow(ax, right(k1), left(s_q))
    s2 = summ(ax, 5.75, y, signs={"l": "+", "b": "−"})
    arrow(ax, right(s_q), (5.55, y), "q_ref", tpos=0.4, toff=(0, 0.06))
    k2 = tf(ax, 6.3, y - 0.45, 1.3, 0.9, "−Kq", "0.038 с")
    arrow(ax, (5.95, y), left(k2))
    gs = tf(ax, 8.0, y - 0.55, 2.2, 1.1, "× (Va₀/Va)²", "огр. 0.25…9\nVa₀ = 16 м/с", sfs=8)
    arrow(ax, right(k2), left(gs))
    s3 = summ(ax, 10.85, y, signs={"l": "+", "t": "+"})
    arrow(ax, right(gs), (10.65, y))
    s_e = sat(ax, 11.45, y - 0.45, 1.3, 0.9, "±25°")
    arrow(ax, (11.05, y), left(s_e))
    pl = box(ax, 13.6, y - 0.75, 1.9, 1.5, "Привод\n+ ЛА", None, "plant", fs=10)
    arrow(ax, right(s_e), left(pl), "δe", toff=(0, 0.06))
    # упреждения и масштаб
    arrow(ax, (10.85, 4.75), (10.85, y + 0.2))
    sig(ax, 10.85, 4.8, "δe*  (балансировка)", fs=9)
    arrow(ax, (9.1, 4.75), top(gs))
    sig(ax, 9.1, 4.8, "Va (ПВД)", fs=9)
    # обратные связи
    feedback(ax, pl, y, "q", "θ", 5.75, 1.35, 1.95, 1.3, "q — гироскоп", "θ — ИНС")
    note(ax, 0.25, 0.95, "Без насыщений каскад тождествен ПД-закону B&M:  δe = δe* + kp_θ·(θ_ref − θ) − kd_θ·q,   "
         "Kθ = kp_θ/kd_θ,  Kq = −kd_θ.\nПредел q_max = Kθ·40° = 648 °/с — не срабатывает при |θ_ref − θ| ≤ 40°; "
         "уставку ограничивают внешние контуры (±15°) и set_pitch_setpoint (±30°).", fs=8.8)
    save(fig, "sau_pitch.png")


def fig_split():
    """Раздельные контуры высоты и скорости (SAUParams.lon = 'split')."""
    fig, ax = canvas(14.2, 5.2)
    for y, (r, fb, pi, piv, ff, lim, out, fbl) in zip(
            (3.85, 1.55),
            (("h_ref", "h", "ПИ высоты", "kp_h + ki_h/s\n0.094 рад/м, 0.016 рад/(м·с)", "α*", "±15°",
              "θ_cmd", "h — барометр"),
             ("Va_ref", "Va", "ПИ скорости", "kp_V + ki_V/s\n0.106 с/м, 0.087 1/м", "δt*", "0…1", "δt",
              "Va — ПВД"))):
        sig(ax, 0.25, y + 0.08, r, ha="left", fs=10)
        summ(ax, 1.55, y, signs={"l": "+", "b": "−"})
        arrow(ax, (0.25, y), (1.35, y))
        b = tf(ax, 2.2, y - 0.6, 3.2, 1.2, pi, piv, sfs=8.2, fs=10.5)
        arrow(ax, (1.75, y), left(b))
        summ(ax, 6.15, y, signs={"l": "+", "t": "+"})
        arrow(ax, right(b), (5.95, y))
        arrow(ax, (6.15, y + 0.85), (6.15, y + 0.2))
        sig(ax, 6.15, y + 0.9, ff + "  (балансировка)", fs=9)
        s = sat(ax, 6.85, y - 0.45, 1.3, 0.9, lim)
        arrow(ax, (6.35, y), left(s))
        arrow(ax, right(s), (9.4, y), out, toff=(0, 0.06))
        arrow(ax, (1.55, y - 0.85), (1.55, y - 0.2))
        sig(ax, 1.7, y - 0.85, fbl, ha="left", va="bottom", fs=8.8)
    box(ax, 9.75, 1.0, 4.2, 3.6, "Анти-виндап", "интеграл не копится,\nпока выход в упоре\n"
        "и ошибка тянет дальше\n(PID.unwind)\n\nПредел интеграла скорости\n1/ki — весь диапазон газа",
        "gray", fs=10, sfs=8.6)
    note(ax, 0.25, 0.55, "Классы AltitudeHold + SpeedController, обёртка AltSpeedHold (control/sau.py). "
         "Тангаж — от высоты, газ — от скорости: каналы связаны только через ЛА.", fs=8.8)
    save(fig, "sau_split.png")


def fig_tecs():
    """TECS: внешние контуры → требуемые производные энергии → газ и тангаж."""
    fig, ax = canvas(19.4, 9.0)
    yA, yB = 7.35, 5.45
    # --- внешние контуры: высота и скорость
    frame(ax, 0.15, 4.45, 8.1, 4.4, "Внешние контуры: что нужно", "#7f7f7f")
    sig(ax, 0.3, yA + 0.08, "h_ref", ha="left", fs=10)
    summ(ax, 1.35, yA, signs={"l": "+", "b": "−"})
    arrow(ax, (0.3, yA), (1.15, yA))
    kh = tf(ax, 1.85, yA - 0.4, 0.95, 0.8, "K_h", "0.5 1/с", sfs=7.8)
    arrow(ax, (1.55, yA), left(kh))
    summ(ax, 3.3, yA, signs={"l": "+", "t": "+"})
    arrow(ax, right(kh), (3.1, yA))
    arrow(ax, (3.3, yA + 0.85), (3.3, yA + 0.2))
    sig(ax, 3.3, yA + 0.9, "$\\dot{h}$_ff (уклон маршрута)", fs=8.5)
    s = sat(ax, 3.9, yA - 0.5, 2.2, 1.0, "−Va·γ_sink … Va·γ_climb", fs=8)
    arrow(ax, (3.5, yA), left(s))
    dv = tf(ax, 6.55, yA - 0.4, 0.9, 0.8, "÷ Va")
    arrow(ax, right(s), left(dv), "$\\dot{h}$_ref", toff=(0, 0.53))
    arrow(ax, (1.35, yA - 0.8), (1.35, yA - 0.2))
    sig(ax, 1.47, yA - 0.8, "h (баро)", ha="left", fs=8.5)

    sig(ax, 0.3, yB + 0.08, "V_ref", ha="left", fs=10)
    summ(ax, 1.35, yB, signs={"l": "+", "b": "−"})
    arrow(ax, (0.3, yB), (1.15, yB))
    kv = tf(ax, 1.85, yB - 0.4, 0.95, 0.8, "K_V", "0.5 1/с", sfs=7.8)
    arrow(ax, (1.55, yB), left(kv))
    s = sat(ax, 3.9, yB - 0.5, 2.2, 1.0, "±acc_max = 5.4 м/с²", fs=8)
    arrow(ax, right(kv), left(s))
    arrow(ax, (1.35, yB - 0.8), (1.35, yB - 0.2))
    sig(ax, 1.47, yB - 0.8, "Va (ПВД)", ha="left", fs=8.5)

    # --- требуемые производные энергии
    E = box(ax, 8.65, 4.65, 2.9, 3.95, "Производные энергии\n(требуемые)", "Ė_ref = γ_ref + $\\dot{V}$_ref/g\n\n"
            "Ḃ_ref = w·γ_ref − (2 − w)·$\\dot{V}$_ref/g\n\nγ_ff = (Ḃ_ref + (2 − w)·Ė_ref) / 2\n\nw = 1 — высота\nи скорость поровну",
            "ctrl", sfs=8.3)
    arrow(ax, right(dv), (8.65, yA), "γ_ref", toff=(0, 0.06))
    arrow(ax, (6.1, yB), (8.65, yB), "$\\dot{V}$_ref", toff=(0, 0.06))

    # --- внутренние контуры: газ и тангаж
    frame(ax, 11.95, 4.45, 7.3, 4.4, "Внутренние контуры: как", "#7f7f7f")
    yT, yP = yA, yB
    for yy, nm, ki, kiv, ff, lim, extra, out, fbn, sx in (
            (yT, "газ", "ki_T / s", "1.70", "δt*(V_ref) + K_E·Ė_ref", "0…1", True, "δt", "Ė", 12.85),
            (yP, "тангаж", "ki_B / s", "1.0", "α*(Va) + γ_ff", "±15°", False, "θ_cmd", "Ḃ", 12.35)):
        summ(ax, sx, yy, signs={"l": "+", "b": "−"})
        arrow(ax, (11.55, yy), (sx - 0.2, yy), ("Ė_ref" if extra else "Ḃ_ref"), tpos=0.35,
              toff=(0, 0.06))
        k = tf(ax, 13.45, yy - 0.4, 1.25, 0.8, ki, kiv, sfs=7.8)
        arrow(ax, (sx + 0.2, yy), left(k))
        summ(ax, 15.25, yy, signs={"l": "+", "t": "+"})
        arrow(ax, right(k), (15.05, yy))
        arrow(ax, (15.25, yy + 0.8), (15.25, yy + 0.2))
        sig(ax, 15.25, yy + 0.85, ff, fs=8.5)
        s = sat(ax, 15.85, yy - 0.42, 1.05, 0.84, lim, fs=8)
        arrow(ax, (15.45, yy), left(s))
        if extra:
            pr_ = box(ax, 17.25, yy - 0.5, 1.8, 1.0, "защита Va", "δt += k·(1 − δt)", "lim", fs=9,
                      sfs=7.8)
            arrow(ax, right(s), left(pr_))
            arrow(ax, right(pr_), (19.35, yy))
            sig(ax, 19.3, yy + 0.08, out, ha="right", fs=10)
        else:
            arrow(ax, right(s), (19.35, yy))
            sig(ax, 19.3, yy + 0.08, out, ha="right", fs=10)
    # Ė_ref в упреждение газа — пометка
    # --- измеренные производные энергии
    M = box(ax, 11.95, 1.15, 5.0, 2.25, "Производные энергии (измеренные)",
            "γ = $\\dot{h}$ / Va        $\\dot{V}$ = d/dt [ Va / (τ_f·s + 1) ],  τ_f = 0.15 с\n\n"
            "Ė = γ + $\\dot{V}$/g        Ḃ = w·γ − (2 − w)·$\\dot{V}$/g", "sens", sfs=8.6)
    arrow(ax, (9.8, 2.6), (11.95, 2.6), "$\\dot{h}$ (GPS)", toff=(0, 0.06))
    arrow(ax, (9.8, 1.85), (11.95, 1.85), "Va (ПВД)", toff=(0, 0.06))
    arrow(ax, (12.85, 3.4), (12.85, yT - 0.2))
    arrow(ax, (12.35, 3.4), (12.35, yP - 0.2))
    sig(ax, 12.97, 3.55, "Ė", ha="left", fs=10)
    sig(ax, 12.23, 3.55, "Ḃ", ha="right", fs=10)
    note(ax, 0.2, 3.85,
         "Газ меняет полную энергию:  Ė = (T − D)/(m·g)  — от тангажа не зависит.\n"
         "Тангаж перераспределяет её между высотой и скоростью:  Ḃ = 2γ − (2 − w)·Ė,  ∂Ḃ/∂θ ≈ 2.\n\n"
         "Защита от потери скорости:  при Va < Va_min = 1.25·Va_stall = 9.4 м/с\n"
         "k растёт от 0 до 1 на 1.5 м/с:  w → 0 (тангаж держит только скорость),\n"
         "набор запрещён, газ → 1.\n\n"
         "Режим PITCH: γ_ref = γ, θ_cmd = θ_ref оператора — газ держит скорость.\n"
         "Анти-виндап обоих интегралов — условное интегрирование.", fs=9)
    save(fig, "sau_tecs.png")


def fig_roll():
    """Крен по схеме PX4/ArduPilot: φ → p_ref → FF + ПИ → δa."""
    fig, ax = canvas(16.4, 5.6)
    y = 3.0
    sig(ax, 0.25, y + 0.08, "φ_ref", ha="left", fs=10)
    summ(ax, 1.3, y, signs={"l": "+", "b": "−"})
    arrow(ax, (0.25, y), (1.1, y))
    k = tf(ax, 1.85, y - 0.45, 1.2, 0.9, "1/τ_φ", "τ_φ = 0.25 с", sfs=7.8)
    arrow(ax, (1.5, y), left(k))
    s = sat(ax, 3.45, y - 0.45, 1.3, 0.9, "±p_max 120°/с", fs=7.8)
    arrow(ax, right(k), left(s))
    arrow(ax, right(s), (5.75, y), "p_ref", tpos=0.38, toff=(0, 0.06))
    dot(ax, 5.25, y)
    summ(ax, 5.95, y, signs={"l": "+", "b": "−"})
    pi = tf(ax, 6.55, y - 0.55, 2.3, 1.1, "kp_p + ki_p/s", "0.032 с,  0.128", sfs=8)
    arrow(ax, (6.15, y), left(pi))
    sc = tf(ax, 9.25, y - 0.45, 1.1, 0.9, "× s²")
    arrow(ax, right(pi), left(sc))
    ff = tf(ax, 7.0, 4.15, 2.0, 0.9, "s · FF", "FF = 0.094 с", sfs=7.8)
    poly(ax, [(5.25, y), (5.25, 4.6), (7.0, 4.6)])
    summ(ax, 11.0, y, signs={"l": "+", "t": "+"})
    poly(ax, [right(ff), (11.0, 4.6), (11.0, y + 0.2)])
    arrow(ax, right(sc), (10.8, y))
    sig(ax, 10.1, 4.67, "δa_ff — упреждение", fs=8.5)
    sd = sat(ax, 11.6, y - 0.45, 1.25, 0.9, "±δa_max")
    arrow(ax, (11.2, y), left(sd))
    pl = box(ax, 13.6, y - 0.75, 1.8, 1.5, "Привод\n+ ЛА", None, "plant", fs=10)
    arrow(ax, right(sd), left(pl), "δa", toff=(0, 0.06))
    feedback(ax, pl, y, "p", "φ", 5.95, 1.3, 1.75, 1.15, "p — гироскоп", "φ — ИНС")
    note(ax, 0.25, 5.45, "Масштаб по скорости  s = Va₀/Va ∈ [0.5, 2],  Va₀ = 16 м/с,  Va — ПВД:  "
         "a_φ1 ∝ Va, a_φ2 ∝ Va²  ⇒  FF ∝ 1/Va,  kp, ki ∝ 1/Va²", fs=8.8)
    note(ax, 0.25, 0.8, "Упреждение FF = a_φ1/a_φ2 даёт p = p_ref без участия обратной связи; ПИ только поправляет: "
         "П — против возмущений, И — против реактивного момента винта.\n"
         "Анти-виндап: интеграл не копится, пока элероны в упоре. Аналоги: PX4 FW_R_TC, FW_R_RMAX, FW_RR_FF/P/I; "
         "ArduPilot RLL2SRV_TCONST, RLL_RATE_FF/P/I.", fs=8.6)
    save(fig, "sau_roll.png")


def fig_lateral():
    """Боковая САУ: курс → уставка крена; скольжение и микс → РН."""
    fig, ax = canvas(17.2, 6.6)
    y1, y2 = 4.55, 1.95
    sig(ax, 0.25, y1 + 0.08, "χ_ref", ha="left", fs=10)
    summ(ax, 1.3, y1, signs={"l": "+", "b": "−"})
    arrow(ax, (0.25, y1), (1.1, y1))
    sig(ax, 1.3, y1 + 0.3, "wrap", fs=8)
    kc = tf(ax, 1.85, y1 - 0.4, 1.0, 0.8, "K_χ", "1.0 1/с", sfs=7.8)
    arrow(ax, (1.5, y1), left(kc))
    summ(ax, 3.4, y1, signs={"l": "+", "t": "+"})
    arrow(ax, right(kc), (3.2, y1))
    arrow(ax, (3.4, y1 + 0.95), (3.4, y1 + 0.2))
    sig(ax, 3.4, y1 + 1.0, "Vg·κ (кривизна маршрута)", fs=8.5)
    at = tf(ax, 4.4, y1 - 0.5, 2.2, 1.0, "atan(Vg·$\\dot{\\chi}$_ref / g)", None, fs=9.5)
    arrow(ax, (3.6, y1), left(at), "$\\dot{\\chi}$_ref", toff=(0, 0.06))
    s1 = sat(ax, 7.0, y1 - 0.45, 1.2, 0.9, "±30°")
    arrow(ax, right(at), left(s1))
    rl = box(ax, 8.6, y1 - 0.5, 1.65, 1.0, "огр. скорости", "90 °/с", "lim", fs=9, sfs=8)
    arrow(ax, right(s1), left(rl))
    roll = box(ax, 10.95, y1 - 0.6, 2.0, 1.2, "Контур крена", "φ_ref → p_ref → δa\n(схема выше)", "ctrl",
               fs=10, sfs=8)
    arrow(ax, right(rl), left(roll), "φ_ref", toff=(0, 0.06))
    arrow(ax, right(roll), (17.0, y1))
    sig(ax, 16.95, y1 + 0.08, "δa", ha="right", fs=10)
    arrow(ax, (1.3, y1 - 0.85), (1.3, y1 - 0.2))
    sig(ax, 1.42, y1 - 0.85, "χ — GPS", ha="left", fs=8.5)
    # скольжение и микс
    sig(ax, 8.3, y2 + 0.08, "β (зонд УС)", ha="left", fs=10)
    pb = tf(ax, 10.6, y2 - 0.55, 2.35, 1.1, "kp_β + ki_β/s", "0.5,  0.55 1/с", sfs=8)
    arrow(ax, (8.3, y2), left(pb))
    summ(ax, 13.9, y2, signs={"l": "−", "t": "+"})
    arrow(ax, right(pb), (13.7, y2))
    dot(ax, 13.9, y1)
    mix = tf(ax, 13.3, 2.95, 1.2, 0.8, "K_mix", "−0.654", sfs=7.8)
    arrow(ax, (13.9, y1), top(mix))
    arrow(ax, bottom(mix), (13.9, y2 + 0.2))
    s2 = sat(ax, 14.5, y2 - 0.45, 1.2, 0.9, "±25°")
    arrow(ax, (14.1, y2), left(s2))
    arrow(ax, right(s2), (17.0, y2))
    sig(ax, 16.95, y2 + 0.08, "δr", ha="right", fs=10)
    note(ax, 0.25, 3.15,
         "При установившемся крене\n"
         "$\\dot{\\chi}$ = (g/Vg)·tg φ = $\\dot{\\chi}$_ref:\n"
         "тангенс и арктангенс сокращаются.\n\n"
         "Интеграла по курсу нет: на прямой\n"
         "φ = 0 при любом χ — статической\n"
         "ошибки нет, ветер не мешает.", fs=8.8)
    note(ax, 8.3, 1.05, "δr > 0 — нос влево; чтобы убрать β > 0 (поток справа), нос доворачивают вправо: знак «−».\n"
         "K_mix = −N_δa/N_δr гасит момент рыскания от элеронов сразу, не дожидаясь роста β.  "
         "beta_hold = False: δr = 0.", fs=8.5)
    save(fig, "sau_lateral.png")


def fig_aoa():
    """Автомат защиты по углу атаки (AngleOfAttackProtector)."""
    fig, ax = canvas(14.4, 6.6)
    cols = {"N": ("#e5f4e7", "#3c8d4a"), "W": ("#fff3d6", "#c08a17"), "C": ("#fde3d8", "#d35400"),
            "R": ("#e3edf9", "#3b6fb6")}

    def st(x, y, key, name, out):
        fc, ec = cols[key]
        ax.add_patch(FancyBboxPatch((x, y), 3.2, 1.55, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc=fc, ec=ec, lw=2, zorder=2))
        ax.text(x + 1.6, y + 1.08, name, ha="center", va="center", fontsize=12, fontweight="bold",
                color=ec, zorder=3)
        ax.text(x + 1.6, y + 0.48, out, ha="center", va="center", fontsize=8.6, color=TXT, zorder=3,
                linespacing=1.3)
        return (x, y, 3.2, 1.55)

    N = st(0.4, 2.55, "N", "НОРМ", "θ_ref = θ_cmd\nгаз — от TECS")
    W = st(5.6, 4.6, "W", "ПРЕД", "θ_ref = θ_cmd − 3°\nгаз — от TECS")
    Cr = st(10.8, 2.55, "C", "КРИТ", "θ_ref = −7°\nгаз = 1")
    R = st(5.6, 0.5, "R", "ВОССТ", "θ_ref = −7°\nгаз = 1")
    lab = dict(fs=8.8, toff=(0, 0.08))
    arrow(ax, (3.0, 4.1), (5.6, 5.55), "α ≥ α_пред", rad=-0.15, **lab)
    arrow(ax, (5.6, 4.95), (3.5, 4.1), rad=-0.15)
    sig(ax, 4.75, 4.15, "α < α_пред", fs=8.8)
    arrow(ax, (8.8, 5.6), (11.6, 4.1), "α ≥ α_крит", rad=-0.15, **lab)
    arrow(ax, right(N), left(Cr), "α ≥ α_крит", **lab)
    arrow(ax, (12.4, 2.55), (8.8, 1.15), rad=-0.15)
    sig(ax, 11.4, 1.35, "α < α_крит", fs=8.8)
    arrow(ax, (8.8, 1.65), (11.4, 2.55), rad=-0.15)
    sig(ax, 9.55, 2.25, "α ≥ α_крит\n(рецидив)", fs=8.5)
    arrow(ax, (5.6, 1.2), (2.0, 2.55), rad=-0.15)
    sig(ax, 3.1, 1.25, "α < α_выход", fs=8.8)
    note(ax, 0.4, 6.45, "Пороги (стенд, default_aua):  α_пред = 9°,  α_крит = 12.2° (максимум CL),  "
         "α_выход = α_пред − (α_крит − α_пред) = 5.8°", fs=9.2, color=TXT)
    note(ax, 0.4, 0.25 + 0.25, "", fs=1)
    save(fig, "sau_aoa.png")


def fig_sensor_model():
    """Модель псевдодатчика и путь измерений от физики до САУ."""
    fig, ax = canvas(15.6, 5.4)
    # модель одного датчика
    y = 4.1
    sig(ax, 0.3, y + 0.08, "x — истинное значение", ha="left", fs=9.5)
    summ(ax, 3.4, y, signs={"l": "+", "t": "+", "b": "+"})
    arrow(ax, (0.3, y), (3.2, y))
    arrow(ax, (3.4, y + 0.95), (3.4, y + 0.2))
    sig(ax, 3.4, y + 1.0, "b — смещение (bias)", fs=9)
    arrow(ax, (3.4, y - 0.95), (3.4, y - 0.2))
    sig(ax, 3.4, y - 1.0, "N(0, σ²) — белый шум, rng", fs=9, va="top")
    arrow(ax, (3.6, y), (6.1, y))
    sig(ax, 4.85, y + 0.08, "x_изм = x + b + n", fs=9.5)
    note(ax, 0.3, 2.55, "Каждый датчик — чистая функция\nmeasure_*(x, b, σ, rng),\n"
         "control/sensors.py", fs=9)
    ax.plot([6.6, 6.6], [0.3, 5.2], color="#cccccc", lw=1)
    # путь данных
    ph = box(ax, 7.0, 3.25, 2.3, 1.5, "Физика", "runner.run()\nистинное состояние", "plant", sfs=8.3)
    se = box(ax, 10.0, 3.25, 2.3, 1.5, "sensors", "measure(t, state,\nVa, α, β) → meas", "sens", sfs=8.3)
    es = box(ax, 13.0, 3.25, 2.3, 1.5, "estimator", "нет: est = meas\n(фильтр — в планах)", "gray",
             sfs=8.3)
    arrow(ax, right(ph), left(se))
    arrow(ax, right(se), left(es))
    lg = box(ax, 7.0, 0.6, 2.3, 1.4, "Log", "только истинные\nзначения", "data", sfs=8.3)
    arrow(ax, bottom(ph), top(lg))
    ct = box(ax, 13.0, 0.6, 2.3, 1.4, "Контуры САУ", "видят только est", "ctrl", sfs=8.3)
    arrow(ax, bottom(es), top(ct))
    note(ax, 9.7, 1.95, "Измерения в Log не пишутся;\nсценарий может собрать их\nв буферы и передать\n"
         "в FlightLogger.save()", fs=8.6)
    save(fig, "sensor_model.png")


FIGS = {k[4:]: v for k, v in dict(globals()).items() if k.startswith("fig_")}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()

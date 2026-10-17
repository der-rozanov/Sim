# -*- coding: utf-8 -*-
"""
Схемы для документации (docs/img/*.png) — рисуются matplotlib, единым стилем.

    python docs/figures/diagrams.py            # все схемы
    python docs/figures/diagrams.py keys       # одна (имя функции без fig_)

Каждая fig_*() строит одну картинку. Координаты — в «клетках» холста (ось x 0…W,
y 0…H, y вверх), размер картинки в дюймах = W/k × H/k.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "img")
plt.rcParams["font.family"] = "DejaVu Sans"

# Палитра (заливка, обводка) по ролям блоков
C = {
    "prog":  ("#e3edf9", "#3b6fb6"),   # программы
    "data":  ("#fff3d6", "#c08a17"),   # файлы, данные
    "view":  ("#e5f4e7", "#3c8d4a"),   # просмотр, вывод
    "ctrl":  ("#e3edf9", "#3b6fb6"),   # звенья САУ
    "plant": ("#f1e6f7", "#7b4a9a"),   # объект управления
    "sens":  ("#e5f4e7", "#3c8d4a"),   # датчики
    "lim":   ("#fff3d6", "#c08a17"),   # ограничения, защита
    "gray":  ("#f2f2f2", "#8a8a8a"),
    "key":   ("#ffffff", "#5a5a5a"),
}
TXT = "#222222"
ARR = "#4a4a4a"


def canvas(W, H, k=1.0):
    fig = plt.figure(figsize=(W / k, H / k))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, role="prog", fs=11, sfs=8.5, r=0.12, lw=1.4):
    """Блок с заголовком и (необязательно) подписью; (x, y) — левый нижний угол."""
    fc, ec = C[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=TXT, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sfs,
                color="#444444", zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=TXT, zorder=3, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, text=None, color=ARR, lw=1.5, style="-|>", rad=0.0, tpos=0.5,
          toff=(0, 0.12), fs=8.5, ls="-", z=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z,
                                 shrinkA=0, shrinkB=0))
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]
        y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, color="#333333", zorder=3)


def poly(ax, pts, color=ARR, lw=1.5, head=True):
    """Ломаная со стрелкой на конце (линии связи блок-схем)."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, zorder=1,
            solid_capstyle="round")
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def right(b, dy=0.0):
    x, y, w, h = b
    return (x + w, y + h / 2 + dy)


def left(b, dy=0.0):
    x, y, w, h = b
    return (x, y + h / 2 + dy)


def top(b, dx=0.0):
    x, y, w, h = b
    return (x + w / 2 + dx, y + h)


def bottom(b, dx=0.0):
    x, y, w, h = b
    return (x + w / 2 + dx, y)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("сохранено", os.path.normpath(path))


# ---------------------------------------------------------------------------
def fig_overview():
    """Из чего состоит симулятор: программы → .flightlog → просмотр."""
    fig, ax = canvas(13.0, 6.2)
    ax.text(0.3, 5.85, "Программы", fontsize=12, fontweight="bold", color=C["prog"][1])
    progs = [
        ("3D-тренажёр", "GameScenario3D.py\nручной полёт и САУ, маршрут"),
        ("Стенд САУ", "Bench3D.py\nконтуры с ползунками"),
        ("Конструктор САУ", "BlockLab3D.py\nсвоя схема из блоков"),
        ("Сценарии s1–s10", "scenarios/s*.py\nрасчёт без окон 3D"),
    ]
    bs = [box(ax, 0.3, 4.4 - i * 1.35, 3.6, 1.05, t, s, "prog") for i, (t, s) in enumerate(progs)]
    core = box(ax, 4.9, 1.75, 2.6, 2.6, "Ядро",
               "модель полёта 6DOF — sim/\nСАУ — control/\nцикл run() — runner.py", "plant", sfs=9)
    for b in bs:
        poly(ax, [right(b), (4.4, right(b)[1]), (4.4, 3.05), (4.9, 3.05)], head=False)
    arrow(ax, (4.4, 3.05), (4.9, 3.05))
    log = box(ax, 8.6, 2.45, 2.0, 1.2, ".flightlog", "results/\nзапись полёта", "data")
    arrow(ax, right(core), left(log))
    ax.text(8.05, 3.2, "запись", ha="center", fontsize=8.5, color="#333333")
    ax.text(10.95, 5.85, "Просмотр", fontsize=12, fontweight="bold", color=C["view"][1])
    v2 = box(ax, 10.95, 3.75, 1.85, 1.6, "Графики", "viz/viewer.py\nанимация\nи панели", "view")
    v3 = box(ax, 10.95, 0.95, 1.85, 1.6, "3D-повтор", "viz/viewer3d.py\nс картой\nи камерами", "view")
    arrow(ax, right(log, 0.2), left(v2), rad=-0.15)
    arrow(ax, right(log, -0.2), left(v3), rad=0.15)
    ax.text(6.2, 0.55, "Карты, модели ЛА, окна схем и карты — viz/", ha="center",
            fontsize=9, color="#555555")
    save(fig, "overview.png")


def _key(ax, x, y, k, label, role="key", w=0.9, h=0.9):
    fc, ec = {"key": ("#ffffff", "#5a5a5a"), "hot": ("#e3edf9", "#3b6fb6"),
              "sau": ("#e5f4e7", "#3c8d4a"), "gen": ("#f2f2f2", "#8a8a8a")}[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h * 0.62, k, ha="center", va="center", fontsize=13,
            fontweight="bold", color=TXT)
    ax.text(x + w / 2, y + h * 0.22, label, ha="center", va="center", fontsize=6.6,
            color="#333333")


def fig_keys():
    """Клавиши 3D-тренажёра: ручной режим и САУ."""
    fig, ax = canvas(13.0, 5.6)
    for col, (title, keys, role) in enumerate([
        ("Ручной режим — рули", {
            "Q": "РН влево", "W": "нос вниз", "E": "РН вправо",
            "A": "крен влево", "S": "нос вверх", "D": "крен вправо",
            "Z": "газ −5 %", "X": "газ +5 %"}, "hot"),
        ("С САУ (P) — уставки", {
            "Q": "", "W": "θ −1° / h −10", "E": "",
            "A": "курс −10°", "S": "θ +1° / h +10", "D": "курс +10°",
            "Z": "Va −1 м/с", "X": "Va +1 м/с"}, "sau")]):
        x0 = 0.4 + col * 6.5
        ax.text(x0, 5.15, title, fontsize=13, fontweight="bold",
                color=C["prog" if role == "hot" else "sens"][1])
        rows = [("QWE", 0.0), ("ASD", 0.25), ("ZX", 0.55)]
        for r, (row, shift) in enumerate(rows):
            for i, k in enumerate(row):
                lab = keys.get(k, "")
                _key(ax, x0 + shift + i * 1.0, 3.75 - r * 1.0, k, lab,
                     role if lab else "gen")
        # спец-клавиши справа
        extra = ([("↑", "трим: нос ↓"), ("↓", "трим: нос ↑")] if role == "hot"
                 else [("H", "держать h"), ("N", "маршрут")])
        for j, (k, lab) in enumerate(extra):
            _key(ax, x0 + 3.6 + j * 1.0, 2.75, k, lab, role)
        _key(ax, x0 + 3.6, 3.75, "P", "САУ вкл/выкл", "sau", w=1.9)
    ax.text(0.4, 0.95, "Общие:", fontsize=11, fontweight="bold", color=TXT)
    for j, (k, lab) in enumerate([("␣", "пауза"), ("R", "сброс"), ("1–4", "камера"),
                                  ("M", "масштаб ЛА"), ("T", "турбулентн."), ("Esc", "выход")]):
        _key(ax, 1.9 + j * 1.25, 0.45, k, lab, "gen", w=1.1)
    ax.text(9.6, 0.85, "Рули отклоняются, пока клавиша нажата,\nи сами возвращаются к балансировке.",
            fontsize=9, color="#444444", va="center")
    save(fig, "keys.png")


def fig_sau_simple():
    """Упрощённая структура САУ (control/sau.py)."""
    fig, ax = canvas(13.6, 6.0)
    sens = box(ax, 0.3, 2.35, 1.7, 1.3, "Датчики", "h, Va, углы,\nGPS, β", "sens")
    nav = box(ax, 2.6, 4.25, 2.2, 1.15, "Навигатор", "маршрут → χ_ref, h_ref", "ctrl")
    tecs = box(ax, 5.5, 3.05, 2.4, 1.3, "TECS", "h_ref, Va_ref →\nθ_cmd, газ δt", "ctrl")
    prot = box(ax, 8.5, 3.15, 1.75, 1.1, "Защита по α", None, "lim", fs=10)
    pitch = box(ax, 10.85, 3.05, 1.6, 1.3, "Тангаж", "θ → q → δe", "ctrl")
    lat = box(ax, 5.5, 0.9, 2.4, 1.3, "Курс и крен", "χ → φ → p → δa\nβ → δr", "ctrl")
    act = box(ax, 12.85, 1.6, 0.6, 3.2, "", None, "lim")
    ax.text(13.15, 3.2, "приводы рулей", rotation=90, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=TXT)
    # уставки пользователя
    ax.text(0.3, 5.55, "Уставки: h_ref, Va_ref, χ_ref (клавиши, стенд, сценарий)", fontsize=10,
            color="#333333")
    poly(ax, [right(sens), (2.3, 3.0), (2.3, 3.7), (5.5, 3.7)])
    poly(ax, [(2.3, 3.0), (2.3, 1.55), (5.5, 1.55)])
    poly(ax, [(2.3, 3.7), (2.3, 4.82), (2.6, 4.82)])
    poly(ax, [right(nav), (5.2, 4.82), (5.2, 3.95), (5.5, 3.95)])
    ax.text(5.1, 4.02, "h_ref", ha="right", fontsize=8.5, color="#333333")
    ax.text(5.1, 1.98, "χ_ref", ha="right", fontsize=8.5, color="#333333")
    ax.text(2.2, 3.12, "измерения", ha="right", fontsize=8.5, color="#333333")
    poly(ax, [(5.2, 4.82), (5.2, 4.82), (5.2, 2.3), (5.2, 1.9), (5.5, 1.9)])
    arrow(ax, right(tecs, 0.0), left(prot), "θ_cmd", toff=(0, 0.05))
    arrow(ax, right(prot), left(pitch), "θ_ref", toff=(0, 0.05))
    poly(ax, [right(pitch), (12.85, 3.7)])
    poly(ax, [right(tecs, -0.45), (8.2, 3.25), (8.2, 2.6), (12.85, 2.6)])
    ax.text(10.5, 2.68, "газ δt", fontsize=8.5, color="#333333")
    poly(ax, [right(lat), (12.85, 1.55)])
    ax.text(10.5, 1.63, "δa, δr", fontsize=8.5, color="#333333")
    ax.text(0.3, 0.35, "Режимы: PITCH — уставка тангажа; ALTITUDE — высоты; ROUTE — курс и высота от навигатора. "
            "Скорость держится всегда.", fontsize=9, color="#555555")
    save(fig, "sau_simple.png")


def fig_own_scenario():
    """Как устроен сценарий: шесть шагов."""
    fig, ax = canvas(13.6, 2.6)
    steps = [("1. Параметры", "ЛА, ветер,\nшаг, длительность"),
             ("2. Балансировка", "compute_trim()\ntrim_state()"),
             ("3. Управление", "controls_fn(t, …)\nили SAU"),
             ("4. Прогон", "run() → Log"),
             ("5. Лог", "FlightLogger.save()\n+ уставки, события"),
             ("6. Просмотр", "show_flightlog()")]
    roles = ["prog", "prog", "ctrl", "plant", "data", "view"]
    w, gap = 1.95, 0.32
    prev = None
    for i, ((t, s), role) in enumerate(zip(steps, roles)):
        b = box(ax, 0.25 + i * (w + gap), 0.55, w, 1.5, t, s, role, fs=10, sfs=8.5)
        if prev:
            arrow(ax, right(prev), left(b))
        prev = b
    save(fig, "own_scenario.png")


# ---------------------------------------------------------------------------
# Геометрия: системы координат, углы, силы
# ---------------------------------------------------------------------------
import numpy as np                                                    # noqa: E402
from matplotlib.patches import Polygon, Arc                           # noqa: E402

# Силуэт ЛА сбоку (нос → +x), длина корпуса 2, как в 3D-модели по смыслу
_SIDE = [
    ([[-0.60, 0.09], [-0.80, 0.46], [-0.97, 0.46], [-1.0, 0.06]], "#9fb8e0", "#3b6fb6"),   # киль
    ([[0.98, 0.0], [0.80, 0.11], [0.35, 0.15], [-0.30, 0.10], [-1.0, 0.05], [-1.0, -0.01],
      [-0.30, -0.07], [0.35, -0.11], [0.80, -0.10]], "#c9d8ef", "#3b6fb6"),                 # фюзеляж
    ([[0.30, 0.19], [-0.18, 0.19], [-0.20, 0.15], [0.30, 0.15]], "#7f9fd6", "#3b6fb6"),     # крыло
    ([[-0.70, 0.03], [-1.0, 0.03], [-1.0, 0.0], [-0.70, 0.0]], "#7f9fd6", "#3b6fb6"),       # ГО
    ([[1.0, 0.22], [1.03, 0.22], [1.03, -0.22], [1.0, -0.22]], "#555555", "#555555"),       # винт
]


def _rot(pts, ang, s=1.0, c=(0, 0)):
    a = np.radians(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(pts) * s) @ R.T + np.asarray(c)


def _plane_side(ax, c, ang, s):
    for pts, fc, ec in _SIDE:
        ax.add_patch(Polygon(_rot(pts, ang, s, c), closed=True, fc=fc, ec=ec, lw=1.2, zorder=3))


def _vec(ax, c, ang, L, color, text=None, lw=2.2, toff=(0, 0), fs=11, ha="left", z=1):
    a = np.radians(ang)
    q = (c[0] + L * np.cos(a), c[1] + L * np.sin(a))
    arrow(ax, c, q, color=color, lw=lw, z=z)
    if text:
        ax.text(q[0] + toff[0], q[1] + toff[1], text, color=color, fontsize=fs, ha=ha,
                va="center", fontweight="bold")
    return q


def _arc(ax, c, r, a1, a2, color, text, tr=1.18, fs=11):
    lo, hi = min(a1, a2), max(a1, a2)
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=lo, theta2=hi, color=color, lw=1.6, zorder=4))
    m = np.radians((lo + hi) / 2)
    ax.text(c[0] + tr * r * np.cos(m), c[1] + tr * r * np.sin(m), text, color=color,
            fontsize=fs, ha="center", va="center", fontweight="bold")


def _titles(fig, axs, titles, y=0.97, fs=12.5):
    """Заголовки панелей на одной высоте (у панелей с равным масштабом осей разная высота)."""
    fig.canvas.draw()
    for ax, t in zip(axs, titles):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, y, t, ha="center", va="top", fontsize=fs, fontweight="bold")


def fig_frames():
    """Системы координат и углы: вид сбоку (θ, α, γ), сверху (ψ, β, χ, ветер), с хвоста (φ)."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.2),
                            gridspec_kw=dict(width_ratios=[1.3, 1.05, 0.95]))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.08)
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")
    RED_ = "#c0392b"
    # --- сбоку: θ = 22°, γ = 8° → α = 14°
    ax = axs[0]
    ax.set_xlim(-3.0, 4.3)
    ax.set_ylim(-2.7, 3.0)
    th, ga = 22.0, 8.0
    c = (0.0, 0.0)
    ax.plot([-3.0, 4.2], [0, 0], color="#888", lw=1, ls="--")
    ax.text(4.2, -0.2, "горизонт", ha="right", fontsize=9, color="#666")
    _plane_side(ax, c, th, 1.1)
    _vec(ax, c, th, 3.9, C["prog"][1], "x_b", toff=(0.1, 0.05), z=5)
    _vec(ax, c, th - 90, 1.9, C["prog"][1], "z_b", toff=(0.12, 0), z=5)
    _vec(ax, c, ga, 3.9, C["data"][1], "Va", toff=(0.1, 0), z=5)
    _arc(ax, c, 3.35, 0, th, C["prog"][1], "θ", tr=1.07)
    _arc(ax, c, 2.15, 0, ga, C["data"][1], "γ", tr=1.13)
    _arc(ax, c, 2.75, ga, th, RED_, "α", tr=1.09)
    _vec(ax, (-2.6, -2.4), 90, 0.9, "#555", "h (вверх)", lw=1.4, fs=9, toff=(0.1, 0.05))
    _vec(ax, (-2.6, -2.4), 0, 1.0, "#555", "x (на север)", lw=1.4, fs=9, toff=(0.08, 0))
    ax.text(-2.9, 2.85, "θ — тангаж: нос над горизонтом\nγ — наклон траектории: скорость над горизонтом\n"
            "α — угол атаки: нос над вектором скорости\nбез ветра  θ = γ + α", fontsize=9.2,
            va="top", color="#333", linespacing=1.4)
    # --- сверху: север вверх, восток вправо; ψ = 35°, β = 12°, ветер на восток
    ax = axs[1]
    ax.set_xlim(-1.3, 4.0)
    ax.set_ylim(-1.2, 4.1)
    c = (0.0, 0.0)
    az = lambda a: 90 - a                                 # азимут (от севера по часовой) → угол рисунка
    psi, beta = 35.0, 12.0
    ax.plot([0, 0], [0, 3.9], color="#888", lw=1, ls="--")
    ax.text(0.06, 3.9, "север", fontsize=9, color="#666", va="top")
    nose = _vec(ax, c, az(psi), 2.1, C["prog"][1], z=5)
    ax.text(nose[0] - 0.12, nose[1] + 0.05, "нос", color=C["prog"][1], fontsize=10.5,
            fontweight="bold", ha="right")
    va = _vec(ax, c, az(psi + beta), 2.9, C["data"][1], z=5)
    ax.text(va[0] - 0.1, va[1] + 0.12, "Va", color=C["data"][1], fontsize=11, fontweight="bold",
            ha="right")
    W = np.array([1.1, 0.0])
    vg = np.array(va) + W
    arrow(ax, va, tuple(vg), color=C["view"][1], lw=2.2, z=5)
    ax.text(va[0] + 0.55, va[1] + 0.14, "ветер", color=C["view"][1], fontsize=10, fontweight="bold",
            ha="center")
    arrow(ax, c, tuple(vg), color="#222", lw=2.4, z=5)
    ax.text(vg[0] + 0.08, vg[1] - 0.25, "Vg — земная\nскорость", fontsize=9.5, fontweight="bold",
            va="top")
    chi = 90 - np.degrees(np.arctan2(vg[1], vg[0]))
    _arc(ax, c, 0.9, az(psi), 90, C["prog"][1], "ψ", tr=1.25)
    _arc(ax, c, 1.55, az(psi + beta), az(psi), RED_, "β", tr=1.15)
    _arc(ax, c, 2.35, az(chi), 90, "#222", "χ", tr=1.07)
    ax.text(-1.25, -0.6, "ψ — курс (куда смотрит нос)\nβ — скольжение (поток сбоку)\n"
            "χ — путевой угол (куда летим\n      над землёй)", fontsize=9.2, va="top", color="#333",
            linespacing=1.35)
    # --- с хвоста: крен φ > 0 — правое крыло вниз (на рисунке справа)
    ax = axs[2]
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.2, 2.4)
    phi = -25.0
    wing = _rot([[-1.9, -0.035], [1.9, -0.035], [1.9, 0.035], [-1.9, 0.035]], phi)
    ax.add_patch(Polygon(wing, closed=True, fc="#9fb8e0", ec="#3b6fb6", lw=1.2, zorder=3))
    fin = _rot([[-0.05, 0.15], [0.05, 0.15], [0.05, 0.95], [-0.05, 0.95]], phi)
    ax.add_patch(Polygon(fin, closed=True, fc="#9fb8e0", ec="#3b6fb6", lw=1.2, zorder=3))
    ax.add_patch(plt.Circle((0, 0), 0.22, fc="#c9d8ef", ec="#3b6fb6", lw=1.2, zorder=4))
    ax.plot([-2.2, 2.2], [0, 0], color="#888", lw=1, ls="--")
    _vec(ax, (0, 0), phi, 2.1, C["prog"][1], z=5)
    ax.text(1.75, -1.25, "y_b\n(правое крыло)", color=C["prog"][1], fontsize=9.5,
            fontweight="bold", ha="center", va="top")
    _vec(ax, (0, 0), phi - 90, 1.6, C["prog"][1], "z_b", toff=(0.12, 0), z=5)
    _arc(ax, (0, 0), 1.35, phi, 0, RED_, "φ", tr=1.25)
    _titles(fig, axs, ["Вид сбоку: тангаж, наклон траектории, угол атаки",
                       "Вид сверху: курс, скольжение, путевой угол",
                       "Вид с хвоста: крен φ > 0 — правое крыло вниз"])
    save(fig, "frames.png")


def fig_forces():
    """Силы и момент тангажа в вертикальной плоскости."""
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-4.4, 4.6)
    ax.set_ylim(-3.1, 3.2)
    th, ga = 12.0, 3.0
    c = (0.0, 0.0)
    ax.plot([-4.3, 4.5], [0, 0], color="#888", lw=1, ls="--")
    s = 2.0
    _plane_side(ax, c, th, s)
    nose = tuple(_rot([[1.0, 0.0]], th, s)[0])
    tail = tuple(_rot([[-1.0, 0.0]], th, s)[0])
    _vec(ax, nose, ga, 1.6, C["data"][1], "Va", toff=(0.1, 0), z=6, lw=1.6)
    _vec(ax, c, ga + 90, 2.6, C["view"][1], "L — подъёмная сила, ⊥ Va", toff=(0.12, 0.1), fs=10.5, z=6)
    _vec(ax, tail, ga + 180, 1.4, "#c0392b", None, z=6)
    ax.text(tail[0] - 1.45, tail[1] - 0.4, "D — сопротивление,\n∥ −Va", color="#c0392b",
            fontsize=10.5, fontweight="bold", ha="left", va="top")
    _vec(ax, nose, th, 1.3, "#7b4a9a", None, z=6, lw=2.6)
    ax.text(nose[0] + 0.25, nose[1] + 0.75, "T — тяга винта,\nвдоль оси x_b", color="#7b4a9a",
            fontsize=10.5, fontweight="bold")
    _vec(ax, c, -90, 2.5, "#444", "mg — вес", toff=(0.12, 0.1), fs=10.5, z=6)
    ax.add_patch(Arc(c, 1.3, 1.3, theta1=200, theta2=320, color=C["lim"][1], lw=2.2, zorder=6))
    a2 = np.radians(320)
    arrow(ax, (0.65 * np.cos(a2 - 0.15), 0.65 * np.sin(a2 - 0.15)),
          (0.65 * np.cos(a2), 0.65 * np.sin(a2)), color=C["lim"][1], lw=2.2, z=6)
    ax.text(0.85, -0.95, "M — момент тангажа\n(> 0 — нос вверх)", color=C["lim"][1], fontsize=10.5,
            fontweight="bold")
    ax.plot(*c, "o", color="k", ms=5, zorder=7)
    ax.text(-0.15, 0.22, "ЦМ", fontsize=9, ha="right", zorder=7)
    fig.text(0.5, 0.96, "Силы в вертикальной плоскости", ha="center", va="top", fontsize=13,
             fontweight="bold")
    save(fig, "forces.png")


FIGS = {k[4:]: v for k, v in dict(globals()).items() if k.startswith("fig_")}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()

# -*- coding: utf-8 -*-
"""
HUD 3D-окна: символика в стиле OSD (Walksnail / ArduPilot) — белые линии и
подписи с мягкой тёмной обводкой, без плашек, только необходимое (макет — решение автора 2026-10-15,
подписи по-английски).

Состав (уровень 2 — полный):
  сверху       — строка активных режимов САУ; курсовая лента: ψ (галочка), χ (ромб),
                 χ_ref (скоба);
  центр        — продольная ось ЛА («W»), маркер вектора воздушной скорости (кружок
                 с крылышками) и линии тангажа; камеры 1–2. Положения — проекция
                 реальных направлений (нос, вектор Va, горизонт) через объектив
                 камеры: разнос «ось — вектор скорости» по вертикали равен α, по
                 горизонтали — β в любой камере;
  слева        — Va, газ, α, β, φ, χ;
  справа       — лента высоты: h, уставка h_ref, вертикальная скорость Vh;
  справа внизу — круг ветра (север вверху): куда дует ветер и курс ЛА;
  рамка УА     — только при α ≥ alpha_warning (WARN / CRIT / STALL);
  снизу        — строка события (авария, событие лога).
Уровни (F1): 0 — пусто, 1 — режимы + предупреждения, 2 — всё.

Только отображение: данные — HudData от владельца (проигрыватель лога, игра).
Координаты camera.ui: y ∈ [−0.5, 0.5], x ∈ [−A/2, A/2], A — отношение сторон окна.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from panda3d.core import (Point3, Point2, Geom, GeomLines, GeomVertexData, GeomVertexFormat,
                          GeomEnums)
from ursina import Entity, Mesh, Text, Color, Vec3, camera, window, application

WHITE = Color(1.0, 1.0, 1.0, 0.96)          # символика и числа
LABEL = Color(0.78, 1.0, 0.86, 1)           # мелкие подписи величин (как «AIR SPD» в OSD)
HALO  = Color(0.0, 0.0, 0.0, 0.42)           # мягкая тёмная обводка линий
HUD_FONT = "bahnschrift.ttf"                 # широкий шрифт без засечек (кириллица, греческий)
CYAN  = Color(0.35, 0.85, 1.0, 1)
ALERT = [None,
         ("WARN", Color(1.0, 0.72, 0.15, 1)),
         ("CRIT", Color(1.0, 0.45, 0.15, 1)),
         ("STALL", Color(1.0, 0.25, 0.25, 1))]
FAR = 5000.0                  # м — точка «на бесконечности» для проекции направлений
LADDER_Y = (-0.40, 0.33)      # зона линий тангажа по вертикали (выше — курсовая лента)


@dataclass
class HudData:
    """Что показывает HUD. Углы — в градусах, ψ и χ — [0, 360)."""
    t: float
    Va: float                 # воздушная скорость, м/с
    h: float                  # высота, м
    Vh: float                 # вертикальная скорость, м/с (вверх +)
    alpha: float
    beta: float
    phi: float
    psi: float
    chi: float                # путевой угол
    thr: float                # газ δt, 0…1
    wind_n: float = 0.0       # ветер (куда дует) на север / восток, м/с
    wind_e: float = 0.0
    alert: int = 0            # 0 норма, 1 предупреждение, 2 критический УА, 3 срыв
    modes: str = ""           # строка режимов сверху
    status: str = ""          # слева сверху: PAUSED, ON GROUND, ×2 …
    event: str = ""           # строка под рамкой УА (авария, событие лога)
    Va_ref: Optional[float] = None
    h_ref: Optional[float] = None
    chi_ref: Optional[float] = None
    nose: Optional[np.ndarray] = None      # направление продольной оси в СК Ursina
    vel: Optional[np.ndarray] = None       # направление вектора воздушной скорости


def _show(e, on):
    """Включить/выключить элемент только при смене состояния (сеттер enabled Ursina дорогой)."""
    if e.enabled != on:
        e.enabled = on


def _geom_lines(V, S):
    """Отрезки одним примитивом GeomLines: V — вершины (n, 3), S — индексы концов парами."""
    vd = GeomVertexData("hud", GeomVertexFormat.get_v3(), Geom.UH_dynamic)
    vd.unclean_set_num_rows(len(V))
    memoryview(vd.modify_array(0)).cast("B")[:] = np.asarray(V, np.float32).tobytes()
    prim = GeomLines(Geom.UH_dynamic)
    prim.set_index_type(GeomEnums.NT_uint32)
    arr = prim.modify_vertices()
    arr.unclean_set_num_rows(len(S))
    memoryview(arr).cast("B")[:] = np.asarray(S, np.uint32).tobytes()
    g = Geom(vd)
    g.add_primitive(prim)
    return g


class _Lines:
    """Набор ломаных в camera.ui одним объектом (+ тёмная обводка под ним).

    Геометрия — сразу Panda3D (_geom_lines), один примитив на весь набор. Mesh Ursina
    для линий так не умеет: mode="line" — непрерывная ломаная (GeomLinestrips), а отрезки
    парами [(0, 1), (1, 2), …] дают по примитиву на отрезок — ≈ 330 примитивов на кадр,
    75 % времени HUD (замер viz/perf.py, 2026-10-16). Entity + Mesh остаются ради толщины,
    цвета и места в camera.ui."""

    def __init__(self, thickness=2.0, color=WHITE):
        mk = lambda th, c, z: Entity(parent=camera.ui, z=z, color=c, model=Mesh(
            vertices=[(0, 0, 0), (0, 0, 0)], mode="line", thickness=th))
        self.bg = mk(thickness + 3.5, HALO, 0.01)
        self.fg = mk(thickness, color, 0.0)
        self._last = None

    def set(self, polylines):
        V, S = [], []
        for pl in polylines:
            pl = list(pl)
            k = len(V)
            V += [(float(x), float(y), 0.0) for x, y in pl]
            for i in range(k, k + len(pl) - 1):
                S += (i, i + 1)
        if not S:
            V, S = [(0, 0, 0), (0, 0, 0)], [0, 1]
        if (V, S) == self._last:                  # без изменений — геометрию не пересобирать
            return
        self._last = (V, S)
        g = _geom_lines(V, S)
        for e in (self.bg, self.fg):
            node = e.model.geomNode
            node.remove_all_geoms()
            node.add_geom(g)

    @property
    def enabled(self):
        return self.fg.enabled

    @enabled.setter
    def enabled(self, on):
        _show(self.fg, on)
        _show(self.bg, on)

    @property
    def color(self):
        return self.fg.color

    @color.setter
    def color(self, c):
        self.fg.color = c


def _f1(x):
    """Число с одним знаком после запятой без «−0.0»."""
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def _circle(cx, cy, r, n=24):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))


def _dir_u(az_deg, el_deg):
    """Направление по азимуту (от севера) и углу места → СК Ursina (X — восток, Y — вверх, Z — север)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])


def _project(d):
    """Направление d (СК Ursina) → точка camera.ui или None (позади камеры)."""
    p = camera.world_position + Vec3(*(FAR * np.asarray(d, float)))
    pc = application.base.cam.get_relative_point(application.base.render, Point3(p.x, p.y, p.z))
    if pc.dot(camera.lens.get_view_vector()) <= 1.0:      # позади камеры (ось взгляда — из объектива)
        return None
    p2 = Point2()
    camera.lens.project(pc, p2)               # вне кадра — тоже заполняет (за пределами ±1)
    return np.array([p2.x * window.aspect_ratio / 2, p2.y / 2])


class Hud:

    ROWS = ("AIR SPD", "THR", "AOA α", "SLIP β", "BANK φ", "TRK χ")

    def __init__(self, font: str, help_text: str):
        """font — моноширинный шрифт подсказки по клавишам (в нём есть стрелки ←/→)."""
        self.level = 2
        self._wind_A = None                       # отношение сторон, для которого построена шкала ветра
        T = lambda **kw: Text(parent=camera.ui, font=HUD_FONT, text="", **{"color": WHITE, **kw})
        # уровень ≥ 1
        self.modes = T(origin=(0, 0), scale=1.15)
        self.alert_box = _Lines(2.2)
        self.alert_txt = T(origin=(0, 0), scale=1.4)
        self.event_txt = T(origin=(0, 0), scale=1.1)
        # уровень 2
        self.hdg = _Lines(2.0)
        self.hdg_lbl = [T(origin=(0, 0), scale=0.9) for _ in range(8)]
        self.hdg_val = T(origin=(0, 0), scale=1.3)
        self.row_lbl = [T(origin=(-0.5, 0), scale=0.85, color=LABEL) for _ in self.ROWS]
        self.row_val = [T(origin=(0.5, 0), scale=1.3) for _ in self.ROWS]
        self.row_unit = [T(origin=(-0.5, 0), scale=0.75) for _ in self.ROWS]
        self.alt = _Lines(2.0)
        self.alt_val = T(origin=(0.5, 0), scale=1.3)
        self.alt_unit = T(origin=(-0.5, 0), scale=0.75)
        self.vh_val = T(origin=(-0.5, 0), scale=1.0)
        self.sym = _Lines(2.6)                               # ось, вектор скорости
        self.ladder = _Lines(2.0)
        self.ladder_lbl = [T(origin=(0, 0), scale=0.85) for _ in range(8)]
        self.wind = _Lines(1.6)                              # шкала круга ветра (неподвижна)
        self.wind_hdg = _Lines(1.6)                          # риска курса ЛА
        self.wind_arrow = _Lines(2.6, CYAN)
        self.wind_txt = T(origin=(0, 0), scale=0.9)
        self.wind_n = T(origin=(0, 0), scale=0.8, color=LABEL)
        self.status = T(origin=(-0.5, 0.5), scale=0.95)
        self.help_text = help_text + "  F1 HUD"
        self.help = Text(parent=camera.ui, font=font, scale=0.7, origin=(-0.5, -0.5),
                         color=Color(1, 1, 1, 0.75), text="")
        # мягкая обводка — свойство шрифта (Panda3D DynamicTextFont): цвет, ширина, размытие края
        f = self.modes.font
        f.setOutline((0, 0, 0, 0.85), 0.42, 0.5)
        f.clear()
        self._lvl1 = [self.modes, self.alert_box, self.alert_txt, self.event_txt]
        self._lvl2 = [self.hdg, *self.hdg_lbl, self.hdg_val, *self.row_lbl, *self.row_val,
                      *self.row_unit, self.alt, self.alt_val, self.alt_unit, self.vh_val, self.sym,
                      self.ladder, *self.ladder_lbl, self.wind, self.wind_hdg, self.wind_arrow, self.wind_txt,
                      self.wind_n, self.status, self.help]

    def cycle(self):
        self.level = (self.level + 1) % 3

    # --- текст: меняется только при изменении (пересборка глифов дорогая) -------
    def _set(self, t, s, pos=None, color=None):
        if pos is not None:
            t.position = (float(pos[0]), float(pos[1]), 0)
        if color is not None and t.color != color:
            t.color = color
        if t.text != s:
            t.text = s
            if t is self.help:                    # моноширинный шрифт подсказки — без обводки, с тенью
                for tn in t.text_nodes:
                    tn.node().setShadow(0.06, 0.06)
                    tn.node().setShadowColor(0, 0, 0, 0.8)

    # --- кадр ----------------------------------------------------------------------
    def update(self, d: HudData, cam_mode: int, cam_name: str):
        for e in self._lvl1:
            _show(e, self.level >= 1)
        for e in self._lvl2:
            _show(e, self.level >= 2)
        if self.level == 0:
            return
        A = window.aspect_ratio
        self._set(self.modes, d.modes, (0, 0.475))
        self._alert(d)
        self._set(self.event_txt, d.event, (0, -0.41), Color(1, 0.85, 0.4, 1))
        if self.level < 2:
            return
        self._heading(d)
        self._left_column(d, A)
        self._altitude(d, A)
        self._wind(d, A)
        self._symbology(d, cam_mode in (1, 2))
        self._set(self.status, f"t {d.t:.1f} s   {d.status}", (-A / 2 + 0.02, 0.49))
        self._set(self.help, f"{self.help_text}   cam: {cam_name}", (-A / 2 + 0.02, -0.495))

    def _alert(self, d):
        lvl = int(np.clip(d.alert, 0, 3))
        on = lvl > 0
        self.alert_box.enabled = on
        _show(self.alert_txt, on)
        if not on:
            return
        word, c = ALERT[lvl]
        self._set(self.alert_txt, f"AOA {d.alpha:4.1f}°  {word}", (0, 0.285), c)
        w, h, y = 0.2, 0.026, 0.285
        self.alert_box.set([[(-w, y - h), (w, y - h), (w, y + h), (-w, y + h), (-w, y - h)]])
        self.alert_box.color = c

    def _heading(self, d):
        y, half, k = 0.415, 0.26, 0.26 / 30.0           # ±30° на ленте
        segs = []
        lbl = iter(self.hdg_lbl)
        for a in range(int(d.psi) - 35, int(d.psi) + 36):
            if a % 5:
                continue
            x = (a - d.psi) * k
            if abs(x) > half:
                continue
            segs.append([(x, y), (x, y - (0.022 if a % 10 == 0 else 0.011))])
            if a % 10 == 0:
                t = next(lbl, None)
                if t is not None:
                    a360 = a % 360
                    s = {0: "N", 90: "E", 180: "S", 270: "W"}.get(a360, f"{a360 // 10:02d}")
                    self._set(t, s, (x, y + 0.018))
                    _show(t, True)
        for t in lbl:
            _show(t, False)
        segs.append([(-0.012, y - 0.05), (0, y - 0.032), (0.012, y - 0.05)])          # ψ
        xc = np.clip(((d.chi - d.psi + 180) % 360 - 180) * k, -half, half)             # χ
        segs.append([(xc, y - 0.026), (xc + 0.008, y - 0.036), (xc, y - 0.046),
                     (xc - 0.008, y - 0.036), (xc, y - 0.026)])
        if d.chi_ref is not None:                                                      # χ_ref
            xr = np.clip(((d.chi_ref - d.psi + 180) % 360 - 180) * k, -half, half)
            segs.append([(xr - 0.012, y + 0.008), (xr - 0.012, y), (xr + 0.012, y), (xr + 0.012, y + 0.008)])
        self.hdg.set(segs)
        self._set(self.hdg_val, f"{d.psi % 360:03.0f}°", (0, y - 0.075))

    def _left_column(self, d, A):
        """Столбец «подпись — число — единица», как в OSD."""
        vals = [(_f1(d.Va), "m/s" + (f"  ({d.Va_ref:.1f})" if d.Va_ref is not None else "")),
                (f"{100 * d.thr:.0f}", "%"), (_f1(d.alpha) + "°", ""), (_f1(d.beta) + "°", ""),
                (_f1(d.phi) + "°", ""), (f"{d.chi % 360:.0f}°", "")]
        x0, y = -A / 2 + 0.09 * A, 0.13
        for lbl, val, unit, name, (v, u) in zip(self.row_lbl, self.row_val, self.row_unit,
                                                self.ROWS, vals):
            self._set(lbl, name, (x0, y))
            self._set(val, v, (x0 + 0.21, y))
            self._set(unit, u, (x0 + 0.215, y - 0.004))
            y -= 0.05

    def _altitude(self, d, A):
        x, yc, half, k = A / 2 - 0.17 * A, 0.04, 0.15, 0.004      # ±37 м на ленте
        segs = [[(x, yc - half), (x, yc + half)]]
        for hh in range(int(d.h) - 40, int(d.h) + 41):
            if hh % 5:
                continue
            y = yc + (hh - d.h) * k
            if abs(y - yc) <= half:
                segs.append([(x, y), (x + (0.025 if hh % 10 == 0 else 0.012), y)])
        segs.append([(x - 0.004, yc), (x - 0.018, yc + 0.008), (x - 0.018, yc - 0.008), (x - 0.004, yc)])
        if d.h_ref is not None:
            yr = yc + np.clip((d.h_ref - d.h) * k, -half, half)
            segs.append([(x + 0.003, yr), (x + 0.018, yr + 0.009), (x + 0.018, yr - 0.009), (x + 0.003, yr)])
        yv = yc + np.clip(d.Vh, -6, 6) * 0.02                     # вертикальная скорость
        segs.append([(x + 0.04, yv), (x + 0.058, yv + 0.01), (x + 0.058, yv - 0.01), (x + 0.04, yv)])
        self.alt.set(segs)
        self._set(self.alt_val, f"{d.h:.0f}", (x - 0.05, yc))
        self._set(self.alt_unit, "m", (x - 0.046, yc - 0.004))
        vh = _f1(d.Vh)
        self._set(self.vh_val, ("" if vh[0] == "-" or vh == "0.0" else "+") + vh + " m/s", (x + 0.065, yv))

    def _wind(self, d, A):
        cx, cy, R = A / 2 - 0.1, -0.30, 0.065
        if self._wind_A != A:                                     # шкала — только при смене окна
            self._wind_A = A
            segs = [_circle(cx, cy, R), _circle(cx, cy, 0.6 * R, 18)]
            for a in range(0, 360, 30):
                t = np.radians(a)
                r2 = R - (0.012 if a % 90 else 0.02)
                segs.append([(cx + R * np.sin(t), cy + R * np.cos(t)), (cx + r2 * np.sin(t), cy + r2 * np.cos(t))])
            self.wind.set(segs)
        ps = np.radians(d.psi)                                    # курс ЛА — короткая риска
        self.wind_hdg.set([[(cx, cy), (cx + 0.45 * R * np.sin(ps), cy + 0.45 * R * np.cos(ps))]])
        V = float(np.hypot(d.wind_n, d.wind_e))
        if V > 0.2:
            ux, uy = d.wind_e / V, d.wind_n / V                   # куда дует, север — вверх
            p0, p1 = (cx - 0.8 * R * ux, cy - 0.8 * R * uy), (cx + 0.8 * R * ux, cy + 0.8 * R * uy)
            hx, hy = -uy, ux
            head = [(p1[0] - 0.018 * ux + 0.01 * hx, p1[1] - 0.018 * uy + 0.01 * hy), p1,
                    (p1[0] - 0.018 * ux - 0.01 * hx, p1[1] - 0.018 * uy - 0.01 * hy)]
            self.wind_arrow.set([[p0, p1], head])
        else:
            self.wind_arrow.set([])
        self._set(self.wind_n, "N", (cx, cy + R + 0.016))
        self._set(self.wind_txt, f"WIND {V:.1f} m/s", (cx, cy - R - 0.022))

    def _symbology(self, d, on):
        self.sym.enabled = self.ladder.enabled = on
        if not on or d.nose is None:
            for t in self.ladder_lbl:
                _show(t, False)
            return
        segs = []
        b = _project(d.nose)                                     # продольная ось — «W»
        if b is not None:
            x, y = b
            segs.append([(x - 0.045, y), (x - 0.022, y), (x - 0.011, y - 0.014), (x, y),
                         (x + 0.011, y - 0.014), (x + 0.022, y), (x + 0.045, y)])
        v = _project(d.vel) if d.vel is not None else None      # вектор воздушной скорости
        if v is not None:
            x, y = v
            r = 0.012
            segs += [_circle(x, y, r, 16), [(x - r, y), (x - 0.038, y)], [(x + r, y), (x + 0.038, y)],
                     [(x, y + r), (x, y + 0.027)]]
        self.sym.set(segs)
        # линии тангажа: горизонт и ±5…20° по курсу ψ (связаны с землёй)
        lad = []
        lbl = iter(self.ladder_lbl)
        for el in range(-20, 25, 5):
            gap, hw = (2.5, 14.0) if el == 0 else (2.5, 7.0)
            for s in (-1, 1):
                p_in, p_out = _project(_dir_u(d.psi + s * gap, el)), _project(_dir_u(d.psi + s * hw, el))
                if p_in is None or p_out is None or not all(
                        LADDER_Y[0] < p[1] < LADDER_Y[1] for p in (p_in, p_out)):
                    continue
                if el < 0:                                      # ниже горизонта — штрихи
                    for f0 in (0.0, 0.4, 0.8):
                        lad.append([p_in + f0 * (p_out - p_in), p_in + (f0 + 0.22) * (p_out - p_in)])
                else:
                    lad.append([p_in, p_out])
                if el:
                    tip = _project(_dir_u(d.psi + s * hw, el - np.sign(el) * 1.0))
                    if tip is not None:
                        lad.append([p_out, tip])
                    if el % 10 == 0:
                        t = next(lbl, None)
                        if t is not None:
                            u = (p_out - p_in) / (np.linalg.norm(p_out - p_in) + 1e-9)
                            self._set(t, f"{abs(el)}", p_out + 0.022 * u)
                            _show(t, True)
        for t in lbl:
            _show(t, False)
        self.ladder.set(lad)

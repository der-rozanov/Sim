# -*- coding: utf-8 -*-
"""
3D-проигрыватель полётных логов (.flightlog) на движке Ursina (Panda3D).

Запуск:
    python viz/viewer3d.py                                  # последний лог из results/
    python viz/viewer3d.py results/<файл>.flightlog
    python viz/viewer3d.py <файл> --speed 2 --cam 3         # скорость, камера на старте

Управление:
    Пробел      — пауза / продолжение
    ← / →       — перемотка −5 / +5 с          R — в начало
    ↑ / ↓       — скорость проигрывания ×2 / ÷2
    1 2 3 4     — камера: за хвостом / из кабины / облёт мышью / обзор сверху
    правая кнопка мыши + движение, колесо — облёт и зум (камера 3)
    M           — увеличить силуэт ×10 (для обзора)    T — показать весь путь
    клик по полосе времени внизу — переход к моменту
    Esc         — выход

Только отображение: читает .flightlog (flight_logger.load_log), физику не знает.

Системы координат
-----------------
Симулятор: земная x — север, y — восток, h — вверх; связанная СК — x вперёд,
y на правое крыло, z вниз; углы φ, θ, ψ (порядок ZYX).
Ursina (левая СК, y вверх): X — восток, Y — вверх, Z — север.
Отображение вектора (N, E, D) -> (E, −D, N) сохраняет «право/лево» и
направление разворота на экране.
"""

import sys
import os
import glob
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flight_logger import load_log

from ursina import (Ursina, Entity, Mesh, Text, Sky, Vec3, Color, camera, scene,
                    window, mouse, held_keys, application, time as utime,
                    DirectionalLight, AmbientLight)
from ursina.shaders.lit_with_shadows_shader import lit_with_shadows_shader
from ursina.shaders.unlit_shader import unlit_shader
from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader

ALERT_LABEL = ["НОРМ", "ПРЕД", "КРИТ", "СРЫВ"]
ALERT_CLR   = [(0.55, 0.95, 0.55), (1.0, 0.9, 0.3), (1.0, 0.6, 0.2), (1.0, 0.3, 0.3)]
SUN_DIR     = np.array([0.45, 0.75, 0.35]) / np.linalg.norm([0.45, 0.75, 0.35])
FOG_CLR     = (0.72, 0.80, 0.88)


# ===========================================================================
# Подготовка данных (чистый numpy)
# ===========================================================================

def ned_to_u(n, e, d):
    """Земной вектор (N, E, D) -> координаты Ursina (X=E, Y=−D, Z=N)."""
    return np.stack([e, -d, n], axis=-1)


def body_axes(phi, theta, psi):
    """
    Оси связанной СК в земной СК (NED) по углам φ, θ, ψ (ZYX).
    Возвращает (x_b, z_b) — столбцы 1 и 3 матрицы R_b→n, формы (N, 3).
    """
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    x_b = np.stack([ct * cp, ct * sp, -st], axis=-1)
    z_b = np.stack([cf * st * cp + sf * sp, cf * st * sp - sf * cp, cf * ct], axis=-1)
    return x_b, z_b


class Track:
    """Массивы лога, приведённые к сцене Ursina."""

    def __init__(self, data: dict):
        d = data
        self.meta = d["meta"]
        self.t = d["t"]
        n = len(self.t)
        z = np.zeros(n)
        get = lambda k: d[k] if k in d else z

        self.pos = ned_to_u(d["x"], get("y"), -d["h"])           # (N, 3)
        x_b, z_b = body_axes(get("phi"), d["theta"], get("psi"))
        self.nose = ned_to_u(x_b[:, 0], x_b[:, 1], x_b[:, 2])
        self.up   = ned_to_u(-z_b[:, 0], -z_b[:, 1], -z_b[:, 2])

        deg = np.degrees
        self.Va, self.h = d["Va"], d["h"]
        self.alpha = deg(d["alpha_true"])
        self.beta  = deg(get("beta_true"))
        self.phi, self.theta = deg(get("phi")), deg(d["theta"])
        self.psi   = (deg(get("psi")) + 360.0) % 360.0
        self.de, self.da, self.dr = d["delta_e"], get("delta_a"), get("delta_r")
        self.thr   = d["throttle"]
        self.wind  = np.stack([d["wind_x"], get("wind_y"), d["wind_h"]], axis=-1)
        self.alert = d["alert"].astype(int) if "alert" in d else z.astype(int)
        # путевой угол χ по земной скорости (разность координат)
        vx, vy = np.gradient(d["x"], self.t), np.gradient(get("y"), self.t)
        self.chi = (deg(np.arctan2(vy, vx)) + 360.0) % 360.0
        self.events = self.meta.get("events", [])
        self.b = float(self.meta.get("aircraft", {}).get("b", 2.0))
        self.c = float(self.meta.get("aircraft", {}).get("c", 0.2))

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def position(self, t: float) -> np.ndarray:
        """Положение с линейной интерполяцией (плавность при 60 кадрах/с)."""
        return np.array([np.interp(t, self.t, self.pos[:, k]) for k in range(3)])


# ===========================================================================
# Подстилающая поверхность (процедурная, детерминированная)
# ===========================================================================

class Terrain:
    """
    Рельеф вокруг траектории: поля, холмы ±12 м, лес, ВПП в точке старта,
    гряда гор по краю карты (ориентир горизонта). Освещение «запечено» в цвет.
    """

    def __init__(self, track: Track, seed: int = 7):
        p = track.pos
        self.cx = 0.5 * (p[:, 0].min() + p[:, 0].max())
        self.cz = 0.5 * (p[:, 2].min() + p[:, 2].max())
        span = max(np.ptp(p[:, 0]), np.ptp(p[:, 2]))
        self.half = 0.5 * span + 3000.0
        self.rng = np.random.default_rng(seed)
        # ВПП: в точке старта, вдоль начального курса
        self.rw_pos = p[0].copy()
        hd = track.nose[0].copy(); hd[1] = 0.0
        self.rw_dir = hd / (np.linalg.norm(hd) + 1e-9)
        self.rw_len, self.rw_w = 600.0, 30.0

    # --- высота поверхности -------------------------------------------
    def _runway_mask(self, X, Z):
        dx, dz = X - self.rw_pos[0], Z - self.rw_pos[2]
        along = dx * self.rw_dir[0] + dz * self.rw_dir[2]
        across = -dx * self.rw_dir[2] + dz * self.rw_dir[0]
        a = np.clip((np.abs(along) - 0.5 * self.rw_len) / 300.0, 0, 1)
        c = np.clip((np.abs(across) - 3 * self.rw_w) / 300.0, 0, 1)
        return np.maximum(a, c)                      # 0 на ВПП, 1 вдали

    def height(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        hills = (6.0 * np.sin(X / 180 + 1.3) * np.cos(Z / 230)
                 + 4.0 * np.sin((X + Z) / 97) + 2.5 * np.cos((X - 2 * Z) / 61))
        r = np.maximum(np.abs(X - self.cx), np.abs(Z - self.cz)) / self.half
        ang = np.arctan2(Z - self.cz, X - self.cx)
        ridge = np.clip((r - 0.62) / 0.33, 0, 1) ** 1.5
        mountains = ridge * (260 + 120 * np.sin(5 * ang) + 60 * np.sin(13 * ang + 1)
                             + 25 * np.sin(X / 70) * np.cos(Z / 85))
        return hills * self._runway_mask(X, Z) + mountains

    # --- сетка ----------------------------------------------------------
    def build(self) -> list:
        n = int(np.clip(2 * self.half / 40.0, 80, 220))     # клетка ≈ 40 м
        xs = np.linspace(self.cx - self.half, self.cx + self.half, n + 1)
        zs = np.linspace(self.cz - self.half, self.cz + self.half, n + 1)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        Y = self.height(X, Z)

        # каждый квадрат — свои 4 вершины (чёткие границы полей)
        v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], -1)
        v10 = np.stack([X[1:, :-1],  Y[1:, :-1],  Z[1:, :-1]], -1)
        v11 = np.stack([X[1:, 1:],   Y[1:, 1:],   Z[1:, 1:]], -1)
        v01 = np.stack([X[:-1, 1:],  Y[:-1, 1:],  Z[:-1, 1:]], -1)
        nrm = np.cross(v01 - v00, v10 - v00)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        nrm *= np.sign(nrm[..., 1:2])                       # нормаль вверх
        shade = 0.55 + 0.45 * np.clip(nrm @ SUN_DIR, 0, 1)

        # поля: участки 4×4 клетки, тип по случайному выбору
        palette = np.array([[0.42, 0.62, 0.28], [0.52, 0.68, 0.30], [0.70, 0.68, 0.38],
                            [0.36, 0.55, 0.25], [0.62, 0.55, 0.35], [0.20, 0.40, 0.18]])
        fi = (np.arange(n)[:, None] // 4) * 1000 + (np.arange(n)[None, :] // 4)
        _, inv = np.unique(fi, return_inverse=True)
        ftype = self.rng.integers(0, len(palette), inv.max() + 1)[inv.reshape(n, n)]
        col = palette[ftype] * (0.92 + 0.16 * self.rng.random((n, n, 1)))
        yc = 0.25 * (v00[..., 1] + v10[..., 1] + v11[..., 1] + v01[..., 1])
        rock = np.clip((yc - 40) / 60, 0, 1)[..., None]
        snow = np.clip((yc - 230) / 40, 0, 1)[..., None]
        col = col * (1 - rock) + np.array([0.50, 0.47, 0.43]) * rock
        col = col * (1 - snow) + np.array([0.95, 0.96, 0.98]) * snow
        col = col * shade[..., None]

        verts = np.stack([v00, v10, v11, v01], axis=2).reshape(-1, 3)
        cols = np.repeat(col.reshape(-1, 3), 4, axis=0)
        base = np.arange(n * n) * 4
        tris = np.stack([base, base + 1, base + 2, base, base + 2, base + 3], -1).ravel()
        ground = _mesh_entity(verts, tris, cols)

        forest = (ftype == len(palette) - 1)
        return [ground, self._build_runway(), self._build_trees(X, Z, forest)]

    def _build_runway(self):
        d, L, W = self.rw_dir, self.rw_len, self.rw_w
        side = np.array([d[2], 0, -d[0]])
        c = np.array([self.rw_pos[0], 0.3, self.rw_pos[2]])
        q = [c + s1 * 0.5 * L * d + s2 * 0.5 * W * side
             for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        verts, cols, tris = list(q), [(0.32, 0.32, 0.34)] * 4, [0, 1, 2, 0, 2, 3]
        # осевая разметка
        for k in np.arange(-0.45, 0.45, 0.06):
            a = c + k * L * d + np.array([0, 0.05, 0])
            m = [a + s1 * 7.5 * d + s2 * 0.6 * side
                 for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            i0 = len(verts)
            verts += m; cols += [(0.95, 0.95, 0.95)] * 4
            tris += [i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3]
        return _mesh_entity(np.array(verts), np.array(tris), np.array(cols))

    def _build_trees(self, X, Z, forest):
        cells = np.argwhere(forest)
        if len(cells) == 0:
            return Entity()
        pick = cells[self.rng.integers(0, len(cells), 2500)]
        dx = X[1, 0] - X[0, 0]
        px = X[pick[:, 0], pick[:, 1]] + self.rng.random(len(pick)) * dx
        pz = Z[pick[:, 0], pick[:, 1]] + self.rng.random(len(pick)) * dx
        keep = self._runway_mask(px, pz) > 0.5
        px, pz = px[keep], pz[keep]
        py = self.height(px, pz)
        hgt = 8 + 7 * self.rng.random(len(px))
        rad = 0.3 * hgt
        verts, cols = [], []
        base_c = np.array([0.13, 0.30, 0.12])
        for k in range(4):                                    # 4 грани пирамиды
            a0, a1 = k * np.pi / 2, (k + 1) * np.pi / 2
            p0 = np.stack([px + rad * np.cos(a0), py, pz + rad * np.sin(a0)], -1)
            p1 = np.stack([px + rad * np.cos(a1), py, pz + rad * np.sin(a1)], -1)
            tp = np.stack([px, py + hgt, pz], -1)
            mid = 0.5 * (a0 + a1)
            sh = 0.6 + 0.4 * max(0.0, np.cos(mid) * SUN_DIR[0] + np.sin(mid) * SUN_DIR[2])
            verts.append(np.stack([p0, p1, tp], 1))
            cols.append(np.broadcast_to(base_c * sh, (len(px), 3, 3)))
        verts = np.concatenate(verts).reshape(-1, 3)
        cols = np.concatenate(cols).reshape(-1, 3)
        return _mesh_entity(verts, np.arange(len(verts)), cols)


def _mesh_entity(verts, tris, cols, **kw):
    mesh = Mesh(vertices=[tuple(v) for v in verts.tolist()],
                triangles=tris.tolist(),
                colors=[Color(r, g, b, 1) for r, g, b in cols.tolist()],
                static=True)
    return Entity(model=mesh, shader=unlit_with_fog_shader, double_sided=True, **kw)


# ===========================================================================
# Силуэт ЛА
# ===========================================================================

class AircraftModel:
    """
    Силуэт ЛА из примитивов в масштабе размаха b (из метаданных лога).
    Локальные оси модели: +Z — нос, +X — правое крыло, +Y — верх.
    Рули отклоняются по логу (знаки как в sim/config.py):
      δe > 0 — задняя кромка руля высоты вниз;
      δa > 0 — правый элерон вверх, левый вниз (правый крен);
      δr > 0 — задняя кромка руля направления влево.
    """

    def __init__(self, b: float, c: float):
        self.root = Entity()
        self.body = Entity(parent=self.root)
        s = dict(parent=self.body, shader=lit_with_shadows_shader)
        L = 0.7 * b
        d = 0.075 * b
        c = max(c, 0.09 * b)                           # хорда не тоньше «штриха»
        white, grey = Color(0.95, 0.95, 0.97, 1), Color(0.6, 0.62, 0.66, 1)
        orange, red = Color(1.0, 0.55, 0.1, 1), Color(0.85, 0.12, 0.1, 1)

        Entity(model="sphere", color=white, scale=(d, d * 1.1, L), **s)          # фюзеляж
        Entity(model="sphere", color=Color(0.2, 0.3, 0.45, 1),
               scale=(0.6 * d, 0.5 * d, 0.25 * L), position=(0, 0.45 * d, 0.22 * L), **s)
        zw = 0.08 * L                                   # крыло
        Entity(model="cube", color=white, scale=(b, 0.03 * c + 0.01, 0.75 * c),
               position=(0, 0, zw + 0.125 * c), **s)
        for sgn in (-1, 1):
            Entity(model="cube", color=red, scale=(0.06 * b, 0.035 * c + 0.012, 0.75 * c),
                   position=(sgn * 0.48 * b, 0, zw + 0.125 * c), **s)
        self.ail = []
        for sgn in (-1, 1):                             # элероны: внешние 40 % полуразмаха
            pv = Entity(parent=self.body, position=(sgn * 0.37 * b, 0, zw - 0.25 * c))
            Entity(parent=pv, model="cube", color=orange, shader=lit_with_shadows_shader,
                   scale=(0.26 * b, 0.02 * c + 0.008, 0.25 * c), position=(0, 0, -0.125 * c))
            self.ail.append((sgn, pv))
        zt = -0.47 * L                                  # оперение
        ct = 0.8 * c
        Entity(model="cube", color=white, scale=(0.36 * b, 0.02, 0.6 * ct),
               position=(0, 0, zt + 0.3 * ct), **s)
        self.elev = Entity(parent=self.body, position=(0, 0, zt))
        Entity(parent=self.elev, model="cube", color=orange, shader=lit_with_shadows_shader,
               scale=(0.36 * b, 0.015, 0.4 * ct), position=(0, 0, -0.2 * ct))
        Entity(model="cube", color=white, scale=(0.02, 0.17 * b, 0.6 * ct),
               position=(0, 0.085 * b, zt + 0.3 * ct), **s)
        self.rud = Entity(parent=self.body, position=(0, 0, zt))
        Entity(parent=self.rud, model="cube", color=orange, shader=lit_with_shadows_shader,
               scale=(0.015, 0.17 * b, 0.4 * ct), position=(0, 0.085 * b, -0.2 * ct))
        self.prop = Entity(parent=self.body, position=(0, 0, 0.5 * L + 0.02))
        for a in (0, 90):
            Entity(parent=self.prop, model="cube", color=grey, shader=lit_with_shadows_shader,
                   scale=(0.32 * b, 0.03 * b, 0.01), rotation_z=a)

    def pose(self, pos, nose, up, scale: float):
        self.root.position = Vec3(*pos)
        self.root.scale = scale
        p = Vec3(*pos)
        self.root.lookAt(p + Vec3(*nose), Vec3(*up))   # Panda3D: точно, с креном

    def surfaces(self, de, da, dr, thr, dt):
        self.elev.rotation_x = -np.degrees(de)
        for sgn, pv in self.ail:
            pv.rotation_x = sgn * np.degrees(da)
        self.rud.rotation_y = np.degrees(dr)
        self.prop.rotation_z += 3000.0 * thr * dt


# ===========================================================================
# Проигрыватель
# ===========================================================================

def _line(points, color, thickness=2.0):
    pts = [tuple(p) for p in np.asarray(points).tolist()] or [(0, 0, 0), (0, 0, 0)]
    if len(pts) < 2:
        pts = pts * 2
    return Entity(model=Mesh(vertices=pts, mode="line", thickness=thickness),
                  color=color, shader=unlit_shader)


def _set_line(ent, points):
    pts = [tuple(p) for p in np.asarray(points).tolist()]
    if len(pts) < 2:
        pts = (pts or [(0, 0, 0)]) * 2
    ent.model.vertices = pts
    ent.model.generate()


class Player3D(Entity):

    CAM_NAMES = {1: "за хвостом", 2: "из кабины", 3: "облёт мышью", 4: "обзор сверху"}

    def __init__(self, track: Track, terrain: Terrain, speed=1.0, cam=1, shot=None):
        super().__init__()
        self.tr, self.ter = track, terrain
        self.t_play, self.speed, self.paused = float(track.t[0]), speed, False
        self.cam_mode, self.big, self.show_path = cam, False, False
        self.orbit_yaw, self.orbit_pitch, self.orbit_r = 30.0, 15.0, 12.0 * track.b
        self.shot = shot
        self._frames = 0

        self.ac = AircraftModel(track.b, track.c)
        step = max(1, int(round(0.1 / (track.t[1] - track.t[0]))))
        self._sub = np.arange(0, len(track.t), step)               # прорежение следа
        gpos = track.pos[self._sub].copy()
        gpos[:, 1] = terrain.height(gpos[:, 0], gpos[:, 2]) + 0.8
        self._gshadow = gpos
        self.path_all = _line(track.pos[self._sub], Color(1, 1, 1, 0.35), 1.5)
        self.path_all.enabled = False
        self.trail = _line([], Color(1.0, 0.25, 0.2, 1), 3.0)
        self.shadow = _line([], Color(0.1, 0.12, 0.1, 0.8), 2.0)
        self.drop = _line([], Color(1, 1, 1, 0.5), 1.0)

        font = _font()
        Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45), origin=(-0.5, 0.5),
               position=window.top_left + Vec3(0.01, -0.01, 0), scale=(0.47, 0.29))
        Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45),
               position=window.top + Vec3(0, -0.055, 0), scale=(0.42, 0.1))
        self.hud = Text(parent=camera.ui, position=window.top_left + Vec3(0.02, -0.02, 0),
                        scale=0.85, font=font, text="", line_height=1.1)
        self.alert_txt = Text(parent=camera.ui, position=window.top + Vec3(0, -0.03, 0),
                              origin=(0, 0), scale=1.6, font=font, text="")
        self.event_txt = Text(parent=camera.ui, position=window.top + Vec3(0, -0.08, 0),
                              origin=(0, 0), scale=1.1, font=font, text="",
                              color=Color(1, 0.85, 0.4, 1))
        self.help = Text(parent=camera.ui, position=window.bottom_left + Vec3(0.02, 0.075, 0),
                         scale=0.7, font=font, color=Color(1, 1, 1, 0.75),
                         text="Пробел пауза  ←/→ ±5 с  ↑/↓ скорость  R начало  "
                              "1-4 камера  M масштаб  T весь путь  Esc выход")
        self.bar_bg = Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45),
                             scale=(1.6, 0.018), position=(0, -0.47), collider="box")
        self.bar = Entity(parent=camera.ui, model="quad", color=Color(1, 0.45, 0.2, 0.9),
                          scale=(0.0, 0.018), position=(-0.8, -0.47), origin=(-0.5, 0))
        for ev in track.events:                              # метки событий на полосе
            fx = (ev["t"] - track.t[0]) / (track.t[-1] - track.t[0])
            Entity(parent=camera.ui, model="quad", color=Color(1, 0.85, 0.3, 1),
                   scale=(0.004, 0.03), position=(-0.8 + 1.6 * fx, -0.47))
        self._cam_pos = None
        self._set_fog()

    def _set_fog(self):
        # шейдер Ursina: доля тумана = расстояние / (end − start); в обзоре — реже
        k = 4.0 if self.cam_mode == 4 else 1.6
        scene.fog_density = (0.0, k * self.ter.half)

    # --- ввод -----------------------------------------------------------
    def input(self, key):
        t0, t1 = self.tr.t[0], self.tr.t[-1]
        if key == "space":
            self.paused = not self.paused
        elif key in ("right arrow", "right arrow hold"):
            self.t_play = min(self.t_play + 5.0, t1)
        elif key in ("left arrow", "left arrow hold"):
            self.t_play = max(self.t_play - 5.0, t0)
        elif key == "up arrow":
            self.speed = min(self.speed * 2, 32.0)
        elif key == "down arrow":
            self.speed = max(self.speed / 2, 0.125)
        elif key == "r":
            self.t_play = t0
        elif key in ("1", "2", "3", "4"):
            self.cam_mode, self._cam_pos = int(key), None
            self._set_fog()
        elif key == "m":
            self.big = not self.big
        elif key == "t":
            self.show_path = not self.show_path
        elif key == "scroll up":
            self.orbit_r = max(self.orbit_r * 0.85, 2.0 * self.tr.b)
        elif key == "scroll down":
            self.orbit_r = min(self.orbit_r * 1.18, 3000.0)
        elif key == "left mouse down" and mouse.hovered_entity == self.bar_bg:
            fx = np.clip(mouse.point.x + 0.5, 0, 1)
            self.t_play = t0 + fx * (t1 - t0)
        elif key == "escape":
            application.quit()

    # --- кадр -----------------------------------------------------------
    def update(self):
        tr, dt = self.tr, utime.dt
        if not self.paused:
            self.t_play = min(self.t_play + dt * self.speed, tr.t[-1])
        i = tr.index(self.t_play)
        pos = tr.position(self.t_play)

        scale = 25.0 if self.cam_mode == 4 else (10.0 if self.big else 1.0)
        self.ac.pose(pos, tr.nose[i], tr.up[i], scale)
        self.ac.surfaces(tr.de[i], tr.da[i], tr.dr[i], tr.thr[i], dt)

        j = int(np.searchsorted(self._sub, i))
        _set_line(self.trail, np.vstack([tr.pos[self._sub[:j]], pos]))
        _set_line(self.shadow, self._gshadow[:j + 1])
        self.path_all.enabled = self.show_path or self.cam_mode == 4
        self.ac.prop.enabled = self.cam_mode != 2           # винт закрывает обзор из кабины
        gy = self.ter.height(pos[0], pos[2])
        _set_line(self.drop, [pos, (pos[0], gy, pos[2])])

        self._camera(pos, tr.nose[i], tr.up[i], dt)
        self._hud(i)

        self._frames += 1
        if self.shot and self._frames == 40:
            from panda3d.core import Filename
            base = application.base
            base.win.saveScreenshot(Filename.fromOsSpecific(self.shot))
            application.quit()

    def _camera(self, pos, nose, up, dt):
        p = Vec3(*pos)
        flat = np.array([nose[0], 0.0, nose[2]])
        flat = flat / (np.linalg.norm(flat) + 1e-9)
        b = self.tr.b
        if self.cam_mode == 1:                                    # за хвостом
            target = pos - 4.5 * b * flat + np.array([0, 1.2 * b, 0])
            k = 1.0 - np.exp(-4.0 * dt)
            self._cam_pos = target if self._cam_pos is None else \
                self._cam_pos + k * (target - self._cam_pos)
            camera.position = Vec3(*self._cam_pos)
            camera.look_at(p + Vec3(*(flat * 4 * b)))
            camera.rotation_z = 0
        elif self.cam_mode == 2:                                  # из кабины
            camera.position = Vec3(*(pos + 0.16 * b * np.asarray(nose) + 0.07 * b * np.asarray(up)))
            camera.lookAt(camera.position + Vec3(*nose), Vec3(*up))
        elif self.cam_mode == 3:                                  # облёт мышью
            if mouse.right:
                self.orbit_yaw += mouse.velocity[0] * 200
                self.orbit_pitch = np.clip(self.orbit_pitch - mouse.velocity[1] * 200, -80, 85)
            yw, pt = np.radians(self.orbit_yaw), np.radians(self.orbit_pitch)
            off = self.orbit_r * np.array([np.sin(yw) * np.cos(pt), np.sin(pt), -np.cos(yw) * np.cos(pt)])
            camera.position = Vec3(*(pos + off))
            camera.look_at(p)
            camera.rotation_z = 0
        else:                                                     # обзор сверху
            P = self.tr.pos
            c = 0.5 * (P.min(0) + P.max(0))
            span = max(np.ptp(P[:, 0]), np.ptp(P[:, 2]), 300.0)
            camera.position = Vec3(c[0] - 0.1 * span, c[1] + 0.85 * span, c[2] - 0.6 * span)
            camera.look_at(Vec3(*c))
            camera.rotation_z = 0

    def _hud(self, i):
        tr = self.tr
        wN, wE, wH = tr.wind[i]
        speed = "ПАУЗА" if self.paused else f"×{self.speed:g}"
        self.hud.text = (
            f"{tr.meta.get('scenario', '')}\n"
            f"t   {tr.t[i]:7.2f} с   {speed}\n"
            f"Va  {tr.Va[i]:7.2f} м/с\n"
            f"h   {tr.h[i]:7.1f} м\n"
            f"α   {tr.alpha[i]:7.2f}°    β  {tr.beta[i]:6.2f}°\n"
            f"φ   {tr.phi[i]:7.2f}°    θ  {tr.theta[i]:6.2f}°\n"
            f"ψ   {tr.psi[i]:7.1f}°    χ  {tr.chi[i]:6.1f}°\n"
            f"δe  {np.degrees(tr.de[i]):7.2f}°    δa {np.degrees(tr.da[i]):6.2f}°\n"
            f"δr  {np.degrees(tr.dr[i]):7.2f}°    δt {tr.thr[i]:6.2f}\n"
            f"ветер С {wN:5.1f}  В {wE:5.1f}  верт {wH:5.1f} м/с\n"
            f"камера: {self.CAM_NAMES[self.cam_mode]}"
        )
        a = int(np.clip(tr.alert[i], 0, 3))
        self.alert_txt.text = f"УА {ALERT_LABEL[a]}"
        self.alert_txt.color = Color(*ALERT_CLR[a], 1)
        past = [e for e in tr.events if e["t"] <= tr.t[i]]
        self.event_txt.text = f"{past[-1]['label']}  (t={past[-1]['t']:g} с)" if past else ""
        fx = (tr.t[i] - tr.t[0]) / (tr.t[-1] - tr.t[0])
        self.bar.scale_x = 1.6 * fx


def _font():
    """Моноширинный шрифт с кириллицей (Consolas / DejaVu Sans Mono)."""
    from pathlib import Path
    for p in (r"C:\Windows\Fonts\consola.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            application.fonts_folder = Path(p).parent      # Ursina ищет шрифт по имени
            return Path(p).name
    return "VeraMono.ttf"


# ===========================================================================
# Запуск
# ===========================================================================

def play(path: str, speed: float = 1.0, cam: int = 1, shot: str = None,
         t_start: float = None):
    data = load_log(path)
    track = Track(data)
    app = Ursina(title=f"3D: {os.path.basename(path)}", size=(1600, 900),
                 borderless=False, development_mode=False)
    window.color = Color(*FOG_CLR, 1)
    camera.fov = 70
    camera.clip_plane_near = 0.1
    camera.clip_plane_far = 40000
    terrain = Terrain(track)
    terrain.build()
    sky = Sky()
    sky.scale = 4 * terrain.half
    scene.fog_color = Color(*FOG_CLR, 1)
    sun = DirectionalLight(shadows=False)
    sun.look_at(Vec3(*(-SUN_DIR)))
    AmbientLight(color=Color(0.55, 0.55, 0.6, 1))
    player = Player3D(track, terrain, speed=speed, cam=cam, shot=shot)
    if t_start is not None:
        player.t_play = float(t_start)
    window.fps_counter.enabled = True
    app.run()


def _latest_log() -> str:
    files = glob.glob(os.path.join(_ROOT, "results", "*.flightlog*"))
    if not files:
        sys.exit("В results/ нет .flightlog — сначала запустите сценарий.")
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="3D-проигрыватель .flightlog (Ursina)")
    ap.add_argument("path", nargs="?", help="файл .flightlog (по умолчанию — последний)")
    ap.add_argument("--speed", type=float, default=1.0, help="скорость проигрывания")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4], help="камера")
    ap.add_argument("--at", type=float, default=None, help="начать с момента t, с")
    ap.add_argument("--shot", default=None, help="сохранить скриншот и выйти (для отчётов)")
    a = ap.parse_args()
    play(a.path or _latest_log(), a.speed, a.cam, a.shot, a.at)


if __name__ == "__main__":
    main()

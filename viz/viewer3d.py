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
        self.phi = (deg(get("phi")) + 180.0) % 360.0 - 180.0     # крен копится при бочках
        self.theta = deg(d["theta"])
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
    Рельеф: поля, холмы ±12 м, лес, ВПП в точке старта. Освещение «запечено» в цвет.

    Два режима:
      for_track(track) — карта вокруг траектории лога + гряда гор по краю;
      endless(...)     — периодическая карта (квадрат period × period) для
                         свободного полёта: сетка 3×3 квадрата переставляется
                         под ЛА (follow), шов незаметен — высота, поля, лес и ВПП
                         повторяются с периодом period.
    Высота поверхности — height(X, Z) в координатах Ursina.
    """

    TILE = 80           # клеток на квадрат в периодическом режиме

    def __init__(self, cx, cz, half, start_pos, start_dir, period=None, seed=7):
        self.cx, self.cz, self.half, self.period = cx, cz, half, period
        self.rng = np.random.default_rng(seed)
        self.rw_pos = np.asarray(start_pos, float).copy()
        d = np.array([start_dir[0], 0.0, start_dir[2]])
        self.rw_dir = d / (np.linalg.norm(d) + 1e-9)
        self.rw_len, self.rw_w = 600.0, 30.0
        self.root = None

    @classmethod
    def for_track(cls, track: "Track"):
        p = track.pos
        cx = 0.5 * (p[:, 0].min() + p[:, 0].max())
        cz = 0.5 * (p[:, 2].min() + p[:, 2].max())
        half = 0.5 * max(np.ptp(p[:, 0]), np.ptp(p[:, 2])) + 3000.0
        return cls(cx, cz, half, p[0], track.nose[0])

    @classmethod
    def endless(cls, start_pos, start_dir, period=5000.0):
        return cls(start_pos[0], start_pos[2], 1.5 * period, start_pos, start_dir,
                   period=period)

    # --- высота поверхности -------------------------------------------
    def _wrap(self, d):
        if self.period is None:
            return d
        P = self.period
        return (d + 0.5 * P) % P - 0.5 * P

    def _runway_mask(self, X, Z):
        dx = self._wrap(X - self.rw_pos[0])
        dz = self._wrap(Z - self.rw_pos[2])
        along = dx * self.rw_dir[0] + dz * self.rw_dir[2]
        across = -dx * self.rw_dir[2] + dz * self.rw_dir[0]
        a = np.clip((np.abs(along) - 0.5 * self.rw_len) / 300.0, 0, 1)
        c = np.clip((np.abs(across) - 3 * self.rw_w) / 300.0, 0, 1)
        return np.maximum(a, c)                      # 0 на ВПП, 1 вдали

    def height(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        if self.period is not None:
            # те же холмы, но с целым числом волн на период
            w = 2 * np.pi / self.period
            hills = (6.0 * np.sin(4 * w * X + 1.3) * np.cos(3 * w * Z)
                     + 4.0 * np.sin(8 * w * (X + Z)) + 2.5 * np.cos(13 * w * (X - 2 * Z)))
            return hills * self._runway_mask(X, Z)
        hills = (6.0 * np.sin(X / 180 + 1.3) * np.cos(Z / 230)
                 + 4.0 * np.sin((X + Z) / 97) + 2.5 * np.cos((X - 2 * Z) / 61))
        r = np.maximum(np.abs(X - self.cx), np.abs(Z - self.cz)) / self.half
        ang = np.arctan2(Z - self.cz, X - self.cx)
        ridge = np.clip((r - 0.62) / 0.33, 0, 1) ** 1.5
        mountains = ridge * (260 + 120 * np.sin(5 * ang) + 60 * np.sin(13 * ang + 1)
                             + 25 * np.sin(X / 70) * np.cos(Z / 85))
        return hills * self._runway_mask(X, Z) + mountains

    # --- сетка ----------------------------------------------------------
    def build(self) -> Entity:
        if self.period is None:
            n = int(np.clip(2 * self.half / 40.0, 80, 220))     # клетка ≈ 40 м
            n_tile, tiles = n, 1
        else:
            n_tile, tiles = self.TILE, 3
            n = n_tile * tiles
        x0, z0 = self.cx - self.half, self.cz - self.half
        dx = 2 * self.half / n
        xs, zs = x0 + dx * np.arange(n + 1), z0 + dx * np.arange(n + 1)
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

        # поля: участки 4×4 клетки, тип случайный; в периодическом режиме —
        # раскраска одного квадрата, повторённая tiles×tiles раз
        palette = np.array([[0.42, 0.62, 0.28], [0.52, 0.68, 0.30], [0.70, 0.68, 0.38],
                            [0.36, 0.55, 0.25], [0.62, 0.55, 0.35], [0.20, 0.40, 0.18]])
        nf = -(-n_tile // 4)
        ftile = self.rng.integers(0, len(palette), (nf, nf))
        ftile = np.repeat(np.repeat(ftile, 4, 0), 4, 1)[:n_tile, :n_tile]
        btile = 0.92 + 0.16 * self.rng.random((n_tile, n_tile, 1))
        ftype = np.tile(ftile, (tiles, tiles))
        col = palette[ftype] * np.tile(btile, (tiles, tiles, 1))
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

        self.root = Entity()
        _mesh_entity(verts, tris, cols, parent=self.root)
        offs = [0.0] if self.period is None else [-self.period, 0.0, self.period]
        for ox in offs:
            for oz in offs:
                self._build_runway(ox, oz).parent = self.root
        forest = ftile == len(palette) - 1
        self._build_trees(x0, z0, dx, forest, tiles, n_tile).parent = self.root
        return self.root

    def follow(self, pos):
        """Периодический режим: сдвинуть карту на целое число периодов под ЛА."""
        if self.period is None or self.root is None:
            return
        P = self.period
        self.root.x = P * np.round((pos[0] - self.cx) / P)
        self.root.z = P * np.round((pos[2] - self.cz) / P)

    def _build_runway(self, ox=0.0, oz=0.0):
        d, L, W = self.rw_dir, self.rw_len, self.rw_w
        side = np.array([d[2], 0, -d[0]])
        c = np.array([self.rw_pos[0] + ox, 0.3, self.rw_pos[2] + oz])
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

    def _build_trees(self, x0, z0, dx, forest, tiles, n_tile):
        cells = np.argwhere(forest)
        if len(cells) == 0:
            return Entity()
        pick = cells[self.rng.integers(0, len(cells), 2500 if tiles == 1 else 1200)]
        px = x0 + (pick[:, 0] + self.rng.random(len(pick))) * dx
        pz = z0 + (pick[:, 1] + self.rng.random(len(pick))) * dx
        hgt0 = 8 + 7 * self.rng.random(len(px))
        if tiles > 1:                                          # копии в каждый квадрат
            P = n_tile * dx
            k = np.arange(tiles) * P
            px, pz = np.broadcast_arrays(px[:, None, None] + k[None, :, None],
                                         pz[:, None, None] + k[None, None, :])
            px, pz = px.ravel(), pz.ravel()
            hgt0 = np.repeat(hgt0, tiles * tiles)
        keep = self._runway_mask(px, pz) > 0.5
        px, pz, hgt = px[keep], pz[keep], hgt0[keep]
        py = self.height(px, pz)
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
# Сцена: силуэт, след, камеры, приборная панель (общая для проигрывателя и игры)
# ===========================================================================

def _line(points, color, thickness=2.0):
    ent = Entity(model=Mesh(vertices=[(0, 0, 0), (0, 0, 0)], mode="line",
                            thickness=thickness),
                 color=color, shader=unlit_shader)
    _set_line(ent, points)
    return ent


def _set_line(ent, points):
    pts = [tuple(p) for p in np.asarray(points, float).reshape(-1, 3).tolist()]
    if len(pts) < 2:
        pts = (pts or [(0, 0, 0)]) * 2
    ent.model.vertices = pts
    ent.model.generate()


def _font():
    """Моноширинный шрифт с кириллицей (Consolas / DejaVu Sans Mono)."""
    from pathlib import Path
    for p in (r"C:\Windows\Fonts\consola.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            application.fonts_folder = Path(p).parent      # Ursina ищет шрифт по имени
            return Path(p).name
    return "VeraMono.ttf"


class View3D:
    """
    Всё, что рисуется вокруг ЛА: силуэт, след и его тень, отвес до земли,
    4 камеры, приборная панель. Источник состояния — снаружи (лог или игра):
    владелец каждый кадр вызывает render() и set_hud().
    """

    CAM_NAMES = {1: "за хвостом", 2: "из кабины", 3: "облёт мышью", 4: "обзор сверху"}

    def __init__(self, b: float, c: float, terrain: Terrain, help_text: str,
                 cam: int = 1, hud_lines: int = 11):
        self.b, self.ter = b, terrain
        self.cam_mode, self.big = cam, False
        self.orbit_yaw, self.orbit_pitch, self.orbit_r = 30.0, 15.0, 12.0 * b
        self._cam_pos = None

        self.ac = AircraftModel(b, c)
        self.path_all = _line([], Color(1, 1, 1, 0.35), 1.5)
        self.path_all.enabled = False
        self.trail = _line([], Color(1.0, 0.25, 0.2, 1), 3.0)
        self.shadow = _line([], Color(0.1, 0.12, 0.1, 0.8), 2.0)
        self.drop = _line([], Color(1, 1, 1, 0.5), 1.0)

        font = _font()
        Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45), origin=(-0.5, 0.5),
               position=window.top_left + Vec3(0.01, -0.01, 0),
               scale=(0.47, 0.03 + 0.0235 * hud_lines))
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
                         scale=0.7, font=font, color=Color(1, 1, 1, 0.75), text=help_text)
        self.set_fog()

    def set_fog(self):
        # шейдер Ursina: доля тумана = расстояние / (end − start); в обзоре — реже
        k = 4.0 if self.cam_mode == 4 else 1.6
        if self.ter.period is not None:
            k = 3.0 if self.cam_mode == 4 else 0.67     # край сетки ≥ 1 периода от ЛА
        scene.fog_density = (0.0, k * self.ter.half)

    def handle_key(self, key) -> bool:
        """Клавиши камеры; True — клавиша обработана."""
        if key in ("1", "2", "3", "4"):
            self.cam_mode, self._cam_pos = int(key), None
            self.set_fog()
        elif key == "m":
            self.big = not self.big
        elif key == "scroll up":
            self.orbit_r = max(self.orbit_r * 0.85, 2.0 * self.b)
        elif key == "scroll down":
            self.orbit_r = min(self.orbit_r * 1.18, 3000.0)
        else:
            return False
        return True

    def render(self, pos, nose, up, de, da, dr, thr, trail, dt, overview_pts=None):
        """trail — точки следа (N, 3) в СК Ursina; overview_pts — что охватить в обзоре."""
        scale = 25.0 if self.cam_mode == 4 else (10.0 if self.big else 1.0)
        self.ac.pose(pos, nose, up, scale)
        self.ac.surfaces(de, da, dr, thr, dt)
        self.ac.prop.enabled = self.cam_mode != 2           # винт закрывает обзор из кабины

        trail = np.vstack([np.asarray(trail, float).reshape(-1, 3), pos])
        _set_line(self.trail, trail)
        g = trail.copy()
        g[:, 1] = self.ter.height(g[:, 0], g[:, 2]) + 0.8
        _set_line(self.shadow, g)
        gy = float(self.ter.height(pos[0], pos[2]))
        _set_line(self.drop, [pos, (pos[0], gy, pos[2])])
        self.ter.follow(pos)

        self._camera(pos, nose, up, dt, trail if overview_pts is None else overview_pts)

    def _camera(self, pos, nose, up, dt, overview_pts):
        p = Vec3(*pos)
        flat = np.array([nose[0], 0.0, nose[2]])
        flat = flat / (np.linalg.norm(flat) + 1e-9)
        b = self.b
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
            P = np.vstack([np.asarray(overview_pts, float).reshape(-1, 3), pos])
            c = 0.5 * (P.min(0) + P.max(0))
            span = max(np.ptp(P[:, 0]), np.ptp(P[:, 2]), 300.0)
            camera.position = Vec3(c[0] - 0.1 * span, c[1] + 0.85 * span, c[2] - 0.6 * span)
            camera.look_at(Vec3(*c))
            camera.rotation_z = 0

    def set_hud(self, text: str, alert: int, event: str = ""):
        a = int(np.clip(alert, 0, 3))
        self.hud.text = text + f"\nкамера: {self.CAM_NAMES[self.cam_mode]}"
        self.alert_txt.text = f"УА {ALERT_LABEL[a]}"
        self.alert_txt.color = Color(*ALERT_CLR[a], 1)
        self.event_txt.text = event


def make_app(title: str, terrain_factory):
    """Окно Ursina, небо, свет; terrain_factory() -> Terrain (строится после окна)."""
    app = Ursina(title=title, size=(1600, 900), borderless=False, development_mode=False)
    window.color = Color(*FOG_CLR, 1)
    camera.fov = 70
    camera.clip_plane_near = 0.1
    camera.clip_plane_far = 40000
    terrain = terrain_factory()
    terrain.build()
    sky = Sky()
    sky.scale = 4 * terrain.half
    scene.fog_color = Color(*FOG_CLR, 1)
    sun = DirectionalLight(shadows=False)
    sun.look_at(Vec3(*(-SUN_DIR)))
    AmbientLight(color=Color(0.55, 0.55, 0.6, 1))
    window.fps_counter.enabled = True
    return app, terrain


def screenshot_and_quit(path: str):
    from panda3d.core import Filename
    application.base.win.saveScreenshot(Filename.fromOsSpecific(path))
    application.quit()


# ===========================================================================
# Проигрыватель
# ===========================================================================

class Player3D(Entity):

    HELP = ("Пробел пауза  ←/→ ±5 с  ↑/↓ скорость  R начало  "
            "1-4 камера  M масштаб  T весь путь  Esc выход")

    def __init__(self, track: Track, terrain: Terrain, speed=1.0, cam=1, shot=None):
        super().__init__()
        self.tr = track
        self.t_play, self.speed, self.paused = float(track.t[0]), speed, False
        self.show_path = False
        self.shot, self._frames = shot, 0

        self.view = View3D(track.b, track.c, terrain, self.HELP, cam=cam)
        step = max(1, int(round(0.1 / (track.t[1] - track.t[0]))))
        self._sub = np.arange(0, len(track.t), step)               # прорежение следа
        _set_line(self.view.path_all, track.pos[self._sub])

        self.bar_bg = Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45),
                             scale=(1.6, 0.018), position=(0, -0.47), collider="box")
        self.bar = Entity(parent=camera.ui, model="quad", color=Color(1, 0.45, 0.2, 0.9),
                          scale=(0.0, 0.018), position=(-0.8, -0.47), origin=(-0.5, 0))
        for ev in track.events:                              # метки событий на полосе
            fx = (ev["t"] - track.t[0]) / (track.t[-1] - track.t[0])
            Entity(parent=camera.ui, model="quad", color=Color(1, 0.85, 0.3, 1),
                   scale=(0.004, 0.03), position=(-0.8 + 1.6 * fx, -0.47))

    def input(self, key):
        if self.view.handle_key(key):
            return
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
        elif key == "t":
            self.show_path = not self.show_path
        elif key == "left mouse down" and mouse.hovered_entity == self.bar_bg:
            fx = np.clip(mouse.point.x + 0.5, 0, 1)
            self.t_play = t0 + fx * (t1 - t0)
        elif key == "escape":
            application.quit()

    def update(self):
        tr, dt = self.tr, utime.dt
        if not self.paused:
            self.t_play = min(self.t_play + dt * self.speed, tr.t[-1])
        i = tr.index(self.t_play)
        pos = tr.position(self.t_play)
        j = int(np.searchsorted(self._sub, i))
        v = self.view
        v.path_all.enabled = self.show_path or v.cam_mode == 4
        v.render(pos, tr.nose[i], tr.up[i], tr.de[i], tr.da[i], tr.dr[i], tr.thr[i],
                 tr.pos[self._sub[:j]], dt, overview_pts=tr.pos[self._sub])
        self._hud(i)

        self._frames += 1
        if self.shot and self._frames == 40:
            screenshot_and_quit(self.shot)

    def _hud(self, i):
        tr = self.tr
        wN, wE, wH = tr.wind[i]
        speed = "ПАУЗА" if self.paused else f"×{self.speed:g}"
        text = (
            f"{tr.meta.get('scenario', '')}\n"
            f"t   {tr.t[i]:7.2f} с   {speed}\n"
            f"Va  {tr.Va[i]:7.2f} м/с\n"
            f"h   {tr.h[i]:7.1f} м\n"
            f"α   {tr.alpha[i]:7.2f}°    β  {tr.beta[i]:6.2f}°\n"
            f"φ   {tr.phi[i]:7.2f}°    θ  {tr.theta[i]:6.2f}°\n"
            f"ψ   {tr.psi[i]:7.1f}°    χ  {tr.chi[i]:6.1f}°\n"
            f"δe  {np.degrees(tr.de[i]):7.2f}°    δa {np.degrees(tr.da[i]):6.2f}°\n"
            f"δr  {np.degrees(tr.dr[i]):7.2f}°    δt {tr.thr[i]:6.2f}\n"
            f"ветер С {wN:5.1f}  В {wE:5.1f}  верт {wH:5.1f} м/с"
        )
        past = [e for e in tr.events if e["t"] <= tr.t[i]]
        ev = f"{past[-1]['label']}  (t={past[-1]['t']:g} с)" if past else ""
        self.view.set_hud(text, tr.alert[i], ev)
        fx = (tr.t[i] - tr.t[0]) / (tr.t[-1] - tr.t[0])
        self.bar.scale_x = 1.6 * fx


# ===========================================================================
# Запуск
# ===========================================================================

def play(path: str, speed: float = 1.0, cam: int = 1, shot: str = None,
         t_start: float = None):
    track = Track(load_log(path))
    app, terrain = make_app(f"3D: {os.path.basename(path)}",
                            lambda: Terrain.for_track(track))
    player = Player3D(track, terrain, speed=speed, cam=cam, shot=shot)
    if t_start is not None:
        player.t_play = float(t_start)
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

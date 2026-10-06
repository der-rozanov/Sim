# -*- coding: utf-8 -*-
"""
Мир 3D-тренажёра: карта + бесконечная равнина вокруг. Карты — make_world(имя):

  default — придуманная карта 10 × 10 км с ориентирами (DefaultMap);
  kainki  — аэродром Каинки: спутниковый снимок 4 × 4 км на реальном рельефе
            (KainkiMap; данные — viz/mapdata.py, локальный кэш).

Координаты Ursina (см. viewer3d.py): X — восток, Y — вверх, Z — север, м.
Старт всех сценариев — (0, h0, 0), над ВПП карты; курс старта игры — start_psi.

Карта default (X ∈ [−5, 5] км, Z ∈ [−2.5, 7.5] км):
    ВПП 800 × 30 м в начале координат вдоль Z, ангар и вышка к востоку;
    река с севера на юг (X ≈ −1.6 км), впадает в озеро на юго-западе;
    деревня с церковью на западном берегу, дорога от аэродрома с мостом;
    холм 150 м на северо-востоке (скальная вершина), леса на склонах,
    вдоль реки, на северо-западе, роща у деревни; остальное — поля.
    Линия E = 0 к северу свободна от высоких препятствий — по ней летят s1–s13.

Вокруг любой карты — равнина: квадрат plain_p × plain_p (поля, редкие деревья,
пологие волны ±2.5 м), размноженный по сетке вокруг ЛА (Panda3D instanceTo —
геометрия одна, перестройки при полёте нет). Квадраты, попадающие на карту,
скрыты (границы карты кратны plain_p). Края карты плавно (fade м) переходят
в равнину, шва по высоте нет.

Высота поверхности height(X, Z) — та же функция, по которой построена сетка
(используется для тени следа, отвеса и столкновения в игре).
"""

import json
import os

import numpy as np
from ursina import Entity, Mesh, Color, Texture
from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader

SUN_DIR = np.array([0.45, 0.75, 0.35]) / np.linalg.norm([0.45, 0.75, 0.35])

PLAIN = 2500.0                       # период равнины, м
MAP_X = (-5000.0, 5000.0)            # границы карты (кратны PLAIN)
MAP_Z = (-2500.0, 7500.0)
FADE  = 600.0                        # переход карта → равнина, м
CELL  = 40.0                         # клетка сетки карты, м

RUNWAY = dict(len=800.0, w=30.0)
HILLS  = [(3000.0, 5200.0, 150.0, 450.0),          # (X, Z, высота, σ)
          (3600.0, 4500.0, 80.0, 350.0)]
LAKE_Z, LAKE_R = -1400.0, 450.0
WATER_Y, BED_Y = -2.0, -5.0                        # уровень воды и дна
VILLAGE = (-3000.0, 2600.0)
ROAD = [(30, -420), (-300, -300), (-800, 150), (-1200, 800), (-1900, 1500),
        (-2500, 2150), (-3000, 2600)]
FORESTS = [(2300, 4300, 900, 1300), (-3500, 6000, 900, 900),     # (X, Z, rX, rZ)
           (-3600, 2000, 260, 260), (1200, 6600, 500, 350)]

PALETTE = dict(
    crops=np.array([[0.46, 0.64, 0.29], [0.55, 0.69, 0.31], [0.72, 0.69, 0.40],
                    [0.38, 0.57, 0.26], [0.63, 0.56, 0.36], [0.78, 0.74, 0.45]]),
    meadow=np.array([0.40, 0.60, 0.27]), forest=np.array([0.19, 0.36, 0.16]),
    rock=np.array([0.52, 0.49, 0.44]), bank=np.array([0.47, 0.42, 0.31]),
    tree=np.array([0.13, 0.30, 0.12]),
)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def river_x(Z):
    """Ось реки: X как функция Z (река течёт с севера на юг)."""
    Z = np.asarray(Z, float)
    return -1600.0 + 500.0 * np.sin(Z / 1100.0 + 0.5) + 220.0 * np.sin(Z / 430.0)


LAKE_X = float(river_x(LAKE_Z))


def _seg_dist(X, Z, pts):
    """Расстояние от точек (X, Z) до ломаной pts."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    d = np.full(np.broadcast(X, Z).shape, np.inf)
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        vx, vz = x1 - x0, z1 - z0
        t = np.clip(((X - x0) * vx + (Z - z0) * vz) / (vx * vx + vz * vz), 0, 1)
        d = np.minimum(d, np.hypot(X - x0 - t * vx, Z - z0 - t * vz))
    return d


class World:
    """
    База карты: равнина вокруг, плавный переход, перестановка квадратов равнины.
    Наследник задаёт map_x, map_z, plain_p, fade и реализует height(), _build_map().
    build() — один раз после создания окна; follow(pos) — каждый кадр.
    """

    name, title, attribution = "", "", ""
    start_psi = 0.0                    # курс старта игры, рад (вдоль ВПП)
    map_x, map_z = (-5000.0, 5000.0), (-5000.0, 5000.0)
    plain_p, fade = 2500.0, 600.0
    plain_tint = 1.0                   # яркость полей равнины

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.period = None             # совместимость с View3D
        self._tiles, self._tile_ij = [], None

    def plain(self, X, Z):
        w = 2 * np.pi / self.plain_p
        return (1.5 * np.sin(2 * w * X) * np.cos(3 * w * Z)
                + 1.0 * np.sin(5 * w * (X + Z)))

    def inside(self, X, Z):
        """Вес карты: 1 внутри, плавно → 0 на краю (fade)."""
        (x0, x1), (z0, z1) = self.map_x, self.map_z
        d = np.minimum.reduce([X - x0, x1 - X, Z - z0, z1 - Z])
        return _smooth(0.0, self.fade, d)

    def build(self):
        self.root = Entity()
        self._build_map().parent = self.root
        self._build_plain_tiles()
        return self.root

    def follow(self, pos):
        """Переставить квадраты равнины вокруг ЛА (только при смене квадрата)."""
        P, (x0, x1), (z0, z1) = self.plain_p, self.map_x, self.map_z
        i = int(np.floor((pos[0] - x0) / P))
        j = int(np.floor((pos[2] - z0) / P))
        if (i, j) == self._tile_ij:
            return
        self._tile_ij = (i, j)
        nx, nz = int(round((x1 - x0) / P)), int(round((z1 - z0) / P))
        R = self.POOL // 2
        for k, e in enumerate(self._tiles):
            ti, tj = i + k // self.POOL - R, j + k % self.POOL - R
            e.position = (x0 + ti * P, 0, z0 + tj * P)
            e.enabled = not (0 <= ti < nx and 0 <= tj < nz)


    # --- равнина ----------------------------------------------------------
    POOL = 7                                     # 7×7 квадратов вокруг ЛА

    def _build_plain_tiles(self):
        rng, P = np.random.default_rng(11), self.plain_p
        n = int(round(P / 50))                   # клетка 50 м
        s = np.linspace(0, P, n + 1)
        X, Z = np.meshgrid(s, s, indexing="ij")
        Y = self.plain(X, Z)                     # квадраты стоят в кратных P — высота совпадает
        nf = n // 5                              # поля 250 м
        crop = rng.integers(0, len(PALETTE["crops"]), (nf, nf))
        crop = np.repeat(np.repeat(crop, 5, 0), 5, 1)
        col = PALETTE["crops"][crop] * self.plain_tint * (0.94 + 0.12 * rng.random((n, n, 1)))
        tile = Entity()
        _grid_mesh(X, Y, Z, col).parent = tile
        # редкие деревья — полосами по границам полей (лесополосы) и одиночные
        k = 260
        tx = np.concatenate([rng.uniform(0, P, k), np.repeat(np.arange(1, nf) * 250.0, 25) + rng.normal(0, 3, (nf - 1) * 25)])
        tz = np.concatenate([rng.uniform(0, P, k), rng.uniform(0, P, (nf - 1) * 25)])
        _trees(tx, self.plain(tx, tz), tz, rng).parent = tile
        self._tiles = [Entity() for _ in range(self.POOL * self.POOL)]
        tile.parent = self._tiles[0]
        for e in self._tiles[1:]:
            tile.instanceTo(e)                   # одна геометрия на все квадраты
        self.follow(np.zeros(3))


class DefaultMap(World):
    """Придуманная карта 10 × 10 км (описание — в шапке модуля)."""

    name, title = "default", "учебная карта 10×10 км"
    map_x, map_z, plain_p, fade = MAP_X, MAP_Z, PLAIN, FADE

    @staticmethod
    def runway_mask(X, Z):
        """0 — площадка аэродрома (ровно), 1 — вдали."""
        a = _smooth(0, 250, np.abs(Z) - 0.5 * RUNWAY["len"] - 60)
        c = _smooth(0, 250, np.abs(X - 40) - 160)
        return np.maximum(a, c)

    @staticmethod
    def river_dist(X, Z):
        """Расстояние до реки (по горизонтали от оси, только севернее озера)."""
        d = np.abs(X - river_x(Z))
        return np.where(Z >= LAKE_Z, d, np.inf)

    @staticmethod
    def lake_dist(X, Z):
        return np.hypot(X - LAKE_X, Z - LAKE_Z) - LAKE_R

    def forest(self, X, Z):
        n = 0.15 * np.sin(X / 90.0) * np.cos(Z / 110.0) + 0.1 * np.sin((X - Z) / 57.0)
        f = np.zeros(np.broadcast(X, Z).shape, bool)
        for x0, z0, rx, rz in FORESTS:
            f |= ((X - x0) / rx) ** 2 + ((Z - z0) / rz) ** 2 < 1.0 + n
        dr = self.river_dist(X, Z)
        f |= (dr > 90) & (dr < 260 + 80 * n) & (Z > 3000)
        return f & (self.inside(X, Z) > 0.5)

    def height(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        w = self.inside(X, Z)
        m = self.runway_mask(X, Z)
        hills = sum(h * np.exp(-((X - x0) ** 2 + (Z - z0) ** 2) / (2 * s * s))
                    for x0, z0, h, s in HILLS)
        rolling = (5.0 * (1 + np.sin(X / 700.0 + 1.0) * np.cos(Z / 900.0))
                   + 1.5 * (1 + np.sin((X + 2 * Z) / 330.0)))           # 0…13 м
        H = self.plain(X, Z) * m + w * (hills + rolling * m)
        # русло реки и озеро: плоское дно BED_Y (120 м — шире двух клеток сетки),
        # берега — плавный склон ~160 м
        c = np.maximum(1 - _smooth(60, 220, self.river_dist(X, Z)),
                       1 - _smooth(0, 150, self.lake_dist(X, Z))) * w
        return H * (1 - c) + c * BED_Y

    # --- карта ------------------------------------------------------------
    def _build_map(self):
        root = Entity()
        nx = int((MAP_X[1] - MAP_X[0]) / CELL)
        nz = int((MAP_Z[1] - MAP_Z[0]) / CELL)
        xs = MAP_X[0] + CELL * np.arange(nx + 1)
        zs = MAP_Z[0] + CELL * np.arange(nz + 1)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        Y = self.height(X, Z)
        Xc, Zc = X[:-1, :-1] + CELL / 2, Z[:-1, :-1] + CELL / 2
        Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])

        # раскраска по смыслу
        fi = np.floor((Xc - MAP_X[0]) / 200).astype(int) * 1000 + np.floor((Zc - MAP_Z[0]) / 160).astype(int)
        _, inv = np.unique(fi, return_inverse=True)
        crop = self.rng.integers(0, len(PALETTE["crops"]), inv.max() + 1)[inv.reshape(fi.shape)]
        col = PALETTE["crops"][crop] * (0.94 + 0.12 * self.rng.random(Xc.shape + (1,)))
        meadow = ((self.river_dist(Xc, Zc) < 380) | (self.lake_dist(Xc, Zc) < 300)
                  | (np.hypot(Xc - VILLAGE[0], Zc - VILLAGE[1]) < 360)
                  | (self.runway_mask(Xc, Zc) < 0.5))
        col[meadow] = PALETTE["meadow"] * (0.95 + 0.1 * self.rng.random((meadow.sum(), 1)))
        col[self.forest(Xc, Zc)] = PALETTE["forest"]
        col = _blend(col, PALETTE["rock"], _smooth(85, 125, Yc))
        valley = (self.river_dist(Xc, Zc) < 230) | (self.lake_dist(Xc, Zc) < 160)
        col = _blend(col, PALETTE["bank"], _smooth(-0.5, -2.5, Yc) * valley)
        _grid_mesh(X, Y, Z, col).parent = root

        self._build_water().parent = root
        self._build_road().parent = root
        self._build_airfield().parent = root
        self._build_village().parent = root
        self._build_trees().parent = root
        return root

    def _build_water(self):
        Zs = np.arange(LAKE_Z, MAP_Z[1], 20.0)
        Xs = river_x(Zs)
        hw = 75.0                                # по краю дна — лишнее скрыто берегом
        L = np.stack([Xs - hw, np.full_like(Zs, WATER_Y), Zs], -1)
        Rr = np.stack([Xs + hw, np.full_like(Zs, WATER_Y), Zs], -1)
        v, t = _ribbon(L, Rr)
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([LAKE_X + (LAKE_R + 120) * np.cos(a), np.full_like(a, WATER_Y),
                         LAKE_Z + (LAKE_R + 120) * np.sin(a)], -1)
        c0 = len(v)
        v = np.vstack([v, [[LAKE_X, WATER_Y, LAKE_Z]], ring])
        k = np.arange(64)
        t = np.concatenate([t, np.stack([np.full(64, c0), c0 + 1 + k, c0 + 1 + (k + 1) % 64], -1).ravel()])
        return _mesh(v, t, np.tile([0.25, 0.45, 0.62], (len(v), 1)))

    def _build_road(self):
        pts = np.array(ROAD, float)
        seg = [np.linspace(p0, p1, max(2, int(np.hypot(*(p1 - p0)) / 10)), endpoint=False)
               for p0, p1 in zip(pts[:-1], pts[1:])]
        P = np.vstack(seg + [pts[-1:]])
        d = np.gradient(P, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], -1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        bridge = np.abs(P[:, 0] - river_x(P[:, 1])) < 75
        y = np.maximum(self.height(P[:, 0], P[:, 1]) + 0.4, np.where(bridge, 1.5, -99))
        L = np.stack([P[:, 0] - 4 * n[:, 0], y, P[:, 1] - 4 * n[:, 1]], -1)
        R = np.stack([P[:, 0] + 4 * n[:, 0], y, P[:, 1] + 4 * n[:, 1]], -1)
        v, t = _ribbon(L, R)
        c = np.where(np.repeat(bridge, 2)[:, None], [0.42, 0.30, 0.20], [0.36, 0.34, 0.31])
        return _mesh(v, t, c)

    def _build_airfield(self):
        L, W = RUNWAY["len"], RUNWAY["w"]
        parts = [_quad((0, 0.3, 0), W, L, (0.32, 0.32, 0.34)),
                 _quad((90, 0.25, -250), 120, 120, (0.42, 0.42, 0.44))]     # перрон
        for z in np.arange(-0.45 * L, 0.45 * L, 50):                       # разметка
            parts.append(_quad((0, 0.35, z), 1.2, 15, (0.95, 0.95, 0.95)))
        for x in (-W / 2 + 1, W / 2 - 1):
            parts.append(_quad((x, 0.35, 0), 0.8, L, (0.95, 0.95, 0.95)))
        parts.append(_house(110, -260, 40, 30, 10, 6, 0, (0.70, 0.74, 0.80), (0.45, 0.50, 0.58), 0))
        parts.append(_box(95, -150, 6, 6, 18, 0, (0.85, 0.85, 0.82), 0))     # вышка
        parts.append(_box(95, -150, 8, 8, 3, 0, (0.30, 0.40, 0.50), 18))
        return _merge(parts)

    def _build_village(self):
        rng, (cx, cz) = self.rng, VILLAGE
        g = np.arange(-300, 301, 46.0)
        GX, GZ = np.meshgrid(g, g)
        X = cx + GX.ravel() + rng.uniform(-12, 12, GX.size)
        Z = cz + GZ.ravel() + rng.uniform(-12, 12, GX.size)
        keep = ((np.hypot(X - cx, Z - cz) < 230 + 60 * rng.random(X.size))
                & (np.hypot(X - cx, Z - cz) > 45) & (_seg_dist(X, Z, ROAD) > 14)
                & (self.river_dist(X, Z) > 220))
        walls = [(0.93, 0.90, 0.82), (0.88, 0.84, 0.72), (0.95, 0.95, 0.93), (0.80, 0.72, 0.60)]
        roofs = [(0.62, 0.22, 0.15), (0.45, 0.25, 0.18), (0.35, 0.36, 0.40), (0.70, 0.35, 0.20)]
        parts = []
        for x, z in zip(X[keep], Z[keep]):
            y0 = float(self.height(x, z)) - 0.5
            parts.append(_house(x, z, rng.uniform(7, 11), rng.uniform(8, 13), rng.uniform(4, 6),
                                rng.uniform(2.5, 4), rng.uniform(0, np.pi),
                                walls[rng.integers(4)], roofs[rng.integers(4)], y0))
        y0 = float(self.height(cx, cz)) - 0.5                                # церковь
        parts.append(_house(cx, cz, 12, 24, 11, 6, 0.3, (0.96, 0.96, 0.94), (0.30, 0.45, 0.35), y0))
        tx, tz = cx + 13 * np.sin(0.3), cz + 13 * np.cos(0.3)
        parts.append(_box(tx, tz, 6, 6, 26, 0.3, (0.96, 0.96, 0.94), y0))
        parts.append(_spire(tx, tz, 4.5, 12, (0.85, 0.70, 0.25), y0 + 26))
        return _merge(parts)

    def _build_trees(self):
        rng = self.rng
        X = rng.uniform(*MAP_X, 120000)
        Z = rng.uniform(*MAP_Z, 120000)
        Y = self.height(X, Z)
        ok = ((Y > -0.5) & (self.runway_mask(X, Z) > 0.6) & (_seg_dist(X, Z, ROAD) > 10)
              & (np.hypot(X - VILLAGE[0], Z - VILLAGE[1]) > 330))
        f = ok & self.forest(X, Z)
        lone = ok & ~f & (rng.random(X.size) < 0.012)            # одиночные деревья в полях
        sel = np.flatnonzero(f)[:10000].tolist() + np.flatnonzero(lone).tolist()
        return _trees(X[sel], Y[sel], Z[sel], rng)



class KainkiMap(World):
    """
    Аэродром Каинки: спутниковый снимок 4 × 4 км, натянутый на реальный рельеф,
    + 3D-модель аэродрома поверх снимка.

    Снимок Esri старше аэродрома (на нём поле) — полосы, площадка и постройки
    дорисованы по скриншотам Яндекс.Карт от автора: масштаб и привязка — по двум
    перекрёсткам дорог, общим для обоих снимков (1.59 м/пикс, поворот < 1.5°),
    два независимых измерения сошлись в пределах 5–10 м. Точность ±10 м.

    Центр карты — стык полос «Т» (55.644685 с. ш., 48.507016 в. д.; ≈ 204 м В и
    82 м Ю от метки автора 55.645423, 48.503763). Полосы ≈ 10 м шириной:
    основная 96 м курсом 107.7°/287.7° (вдоль дороги), вторая 63 м курсом 198°.
    Высота — от превышения стыка (≈ 52 м над уровнем моря): h = 0 на полосах;
    на запад — река и пойма, на восток — посёлок Каинки и холмы до ≈ +130 м.
    Снимок и рельеф скачиваются при первом запуске (viz/mapdata.py) в
    viz/map_cache/kainki/ — в git не попадают (условия Esri).
    """

    name, title = "kainki", "аэродром Каинки (спутниковый снимок)"
    attribution = "Снимок: Esri World Imagery (© Esri, Maxar, Earthstar Geographics); рельеф: SRTM"
    LAT0, LON0, SIZE = 55.644685, 48.507016, 4000.0
    RUNWAY_PSI_DEG = 107.7            # основная полоса; старт игры — вдоль неё
    # Аэродром, м от центра (X — восток, Z — север); по скриншотам автора, ±10 м
    STRIPS = [((-59.4, 25.9), (32.2, -3.3), 10.0),       # основная («перекладина Т»)
              ((0.7, 6.7), (-18.8, -53.3), 10.0)]        # вторая («ножка Т»)
    HELIPAD = ((-242.0, 66.8), 25.0, 5.0)                # центр, R, ширина кольца
    BUILDINGS = [  # (X, Z, ширина, длина, высота, курс длинной стороны °, тип)
        (-175.5, 11.8, 7, 70, 4, 107.7, "flat"),         # длинное здание вдоль дороги
        (-194.7, 38.1, 15, 20, 6, 107.7, "flat"),        # белый ангар у площадки
        (-127.1, -8.7, 12, 24, 5, 107.7, "house"),       # домик у перрона
    ] + [(-330.0 + 15 * k * 0.952, 153.0 - 15 * k * 0.305, 10, 12, 5, 107.7, "hangar")
         for k in range(4)]                              # ангары у реки
    APRON = ((-140.5, -6.9), 35.0, 32.0)
    start_psi = np.radians(RUNWAY_PSI_DEG)
    map_x = map_z = (-SIZE / 2, SIZE / 2)
    plain_p, fade = 2000.0, 500.0
    plain_tint = 0.68                 # равнина темнее — ближе к тонам снимка
    CELL = 20.0                       # клетка сетки рельефа, м

    def __init__(self, seed: int = 7):
        super().__init__(seed)
        from viz.mapdata import prepare
        d = prepare(self.name, self.LAT0, self.LON0, self.SIZE)
        self.cache = d
        self.dem = np.load(os.path.join(d, "dem.npy")).astype(float)
        self.meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.z_rw = float(self._dem_at(0.0, 0.0))          # превышение ВПП, м

    def _dem_at(self, X, Z):
        """Билинейная выборка рельефа; X — восток, Z — север (м от центра)."""
        n, S = self.dem.shape[0], self.SIZE
        c = np.clip((np.asarray(X, float) + S / 2) / S * n - 0.5, 0, n - 1.001)
        r = np.clip((S / 2 - np.asarray(Z, float)) / S * n - 0.5, 0, n - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        D = self.dem
        top = D[r0, c0] * (1 - fc) + D[r0, c0 + 1] * fc
        bot = D[r0 + 1, c0] * (1 - fc) + D[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr

    def height(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        w = self.inside(X, Z)
        return (1 - w) * self.plain(X, Z) + w * (self._dem_at(X, Z) - self.z_rw)

    def _build_map(self):
        from PIL import Image
        S, h = self.SIZE, self.CELL
        n = int(S / h)
        s = -S / 2 + h * np.arange(n + 1)
        X, Z = np.meshgrid(s, s, indexing="ij")
        Y = self.height(X, Z)
        # мягкая отмывка рельефа поверх снимка (на снимке своя светотень)
        gx = np.gradient(Y, h, axis=0); gz = np.gradient(Y, h, axis=1)
        nrm = np.stack([-gx, np.ones_like(Y), -gz], -1)
        shade = 0.82 + 0.18 * np.clip(_shade(nrm) * 2 - 1.1, 0, 1)
        verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
        uvs = np.stack([(X + S / 2) / S, (Z + S / 2) / S], -1).reshape(-1, 2)
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        tris = np.stack([a, b, c, a, c, d], -1).ravel()
        cols = np.repeat(shade.reshape(-1, 1), 3, 1)
        m = Mesh(vertices=[tuple(v) for v in verts.tolist()], triangles=tris.tolist(),
                 uvs=[tuple(u) for u in uvs.tolist()],
                 colors=[Color(g, g, g, 1) for g in cols[:, 0].tolist()], static=True)
        tex = Texture(Image.open(os.path.join(self.cache, "ortho.jpg")))
        tex.filtering = "bilinear"        # вблизи — сглаживание (увеличение)
        tex.filtering = "mipmap"          # вдали — мипмапы (уменьшение), без ряби
        root = Entity()
        Entity(model=m, texture=tex, shader=unlit_with_fog_shader, double_sided=True, parent=root)
        self._build_airfield().parent = root
        return root

    def _on_ground(self, pts, dy=0.5):
        """Точки (X, Z) → (X, рельеф + dy, Z)."""
        pts = np.asarray(pts, float)
        return np.column_stack([pts[:, 0], self.height(pts[:, 0], pts[:, 1]) + dy, pts[:, 1]])

    def _build_airfield(self):
        asphalt, white = (0.40, 0.40, 0.42), (0.95, 0.95, 0.95)
        parts = []
        (ax, az), aw, al = self.APRON
        parts.append(_faces([self._on_ground(_rect((ax, az), aw, al, 107.7), 0.45)],
                            (0.47, 0.47, 0.48), (ax, -50, az)))
        for p0, p1, w in self.STRIPS:
            parts.append(_faces([self._on_ground(_strip(p0, p1, w), 0.5)], asphalt,
                                (p0[0], -50, p0[1])))
            p0, p1 = np.asarray(p0), np.asarray(p1)
            L = np.linalg.norm(p1 - p0)
            for t in np.arange(8, L - 8, 12) / L:             # осевая разметка
                c = p0 + t * (p1 - p0)
                u = (p1 - p0) / L
                parts.append(_faces([self._on_ground(_strip(c - 2.5 * u, c + 2.5 * u, 0.5), 0.55)],
                                    white, (c[0], -50, c[1])))
        (hx, hz), R, wr = self.HELIPAD                        # кольцевая площадка
        a = np.linspace(0, 2 * np.pi, 49)
        for k in range(48):
            q = [(hx + r * np.sin(t), hz + r * np.cos(t))
                 for r, t in ((R - wr, a[k]), (R, a[k]), (R, a[k + 1]), (R - wr, a[k + 1]))]
            parts.append(_faces([self._on_ground(q, 0.45)], (0.62, 0.60, 0.55), (hx, -50, hz)))
        for x, z, w, l, h, b, kind in self.BUILDINGS:
            y0 = float(self.height(x, z)) - 0.3
            yaw = np.radians(b)
            if kind == "house":
                parts.append(_house(x, z, w, l, h, 2.5, yaw, (0.92, 0.92, 0.90), (0.35, 0.40, 0.50), y0))
            elif kind == "hangar":
                parts.append(_house(x, z, w, l, h, 1.5, yaw, (0.55, 0.57, 0.60), (0.30, 0.32, 0.36), y0))
            else:
                parts.append(_box(x, z, w, l, h, yaw, (0.93, 0.94, 0.95), y0))
        return _merge(parts)


MAPS = {"default": DefaultMap, "kainki": KainkiMap}


def make_world(name: str = "default") -> World:
    if name not in MAPS:
        raise ValueError(f"неизвестная карта {name!r}; есть: {', '.join(MAPS)}")
    return MAPS[name]()


# ==========================================================================
# Геометрия (numpy → Mesh)
# ==========================================================================

def _blend(col, c2, t):
    t = np.asarray(t)[..., None]
    return col * (1 - t) + np.asarray(c2) * t


def _shade(normals):
    n = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    return 0.55 + 0.45 * np.clip(n @ SUN_DIR, 0, 1)


def _mesh(verts, tris, cols):
    m = Mesh(vertices=[tuple(v) for v in np.asarray(verts, float).tolist()],
             triangles=np.asarray(tris).ravel().tolist(),
             colors=[Color(r, g, b, 1) for r, g, b in np.asarray(cols, float).tolist()],
             static=True)
    return Entity(model=m, shader=unlit_with_fog_shader, double_sided=True)


def _grid_mesh(X, Y, Z, col):
    """Сетка: каждый квадрат — свои 4 вершины (чёткие границы), освещение по нормали."""
    v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], -1)
    v10 = np.stack([X[1:, :-1],  Y[1:, :-1],  Z[1:, :-1]], -1)
    v11 = np.stack([X[1:, 1:],   Y[1:, 1:],   Z[1:, 1:]], -1)
    v01 = np.stack([X[:-1, 1:],  Y[:-1, 1:],  Z[:-1, 1:]], -1)
    nrm = np.cross(v01 - v00, v10 - v00)
    nrm *= np.sign(nrm[..., 1:2])
    col = col * _shade(nrm)[..., None]
    verts = np.stack([v00, v10, v11, v01], axis=2).reshape(-1, 3)
    cols = np.repeat(col.reshape(-1, 3), 4, axis=0)
    b = np.arange(len(verts) // 4) * 4
    tris = np.stack([b, b + 1, b + 2, b, b + 2, b + 3], -1)
    return _mesh(verts, tris, cols)


def _ribbon(L, R):
    """Лента между ломаными L и R (N, 3) → (verts, tris)."""
    n = len(L)
    v = np.empty((2 * n, 3)); v[0::2], v[1::2] = L, R
    i = np.arange(n - 1) * 2
    t = np.stack([i, i + 1, i + 3, i, i + 3, i + 2], -1).ravel()
    return v, t


def _quad(c, sx, sz, color):
    x, y, z = c
    v = np.array([[x - sx / 2, y, z - sz / 2], [x + sx / 2, y, z - sz / 2],
                  [x + sx / 2, y, z + sz / 2], [x - sx / 2, y, z + sz / 2]])
    return v, np.array([0, 1, 2, 0, 2, 3]), np.tile(color, (4, 1))


def _strip(p0, p1, w):
    """Прямоугольник шириной w вдоль отрезка p0 → p1 (точки (X, Z)) → 4 угла."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]]) * w / 2
    return [p0 - n, p1 - n, p1 + n, p0 + n]


def _rect(c, w, l, bearing_deg):
    """Прямоугольник w × l с центром c, длинная сторона по курсу bearing → 4 угла."""
    b = np.radians(bearing_deg)
    u = np.array([np.sin(b), np.cos(b)]) * l / 2
    return _strip(np.asarray(c) - u, np.asarray(c) + u, w)


def _rot(px, pz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return px * c + pz * s, -px * s + pz * c


def _faces(polys, color, center):
    """
    Плоские многоугольники (списки вершин) → (verts, tris, cols).
    Освещение по нормали грани, развёрнутой наружу от center.
    """
    V, T, C = [], [], []
    for p in polys:
        p = np.asarray(p, float)
        i0 = sum(len(x) for x in V)
        V.append(p)
        T += [[i0, i0 + k, i0 + k + 1] for k in range(1, len(p) - 1)]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(nrm, p.mean(0) - np.asarray(center, float)) < 0:
            nrm = -nrm
        C.append(np.tile(np.asarray(color) * _shade(nrm), (len(p), 1)))
    return np.vstack(V), np.array(T).ravel(), np.vstack(C)


def _box(cx, cz, sx, sz, h, yaw, color, y0):
    """Коробка (без дна), основание на высоте y0."""
    corners = [(-sx / 2, -sz / 2), (sx / 2, -sz / 2), (sx / 2, sz / 2), (-sx / 2, sz / 2)]
    P = [(cx + a, cz + b) for a, b in (_rot(px, pz, yaw) for px, pz in corners)]
    lo = [(x, y0, z) for x, z in P]
    hi = [(x, y0 + h, z) for x, z in P]
    walls = [[lo[k], lo[(k + 1) % 4], hi[(k + 1) % 4], hi[k]] for k in range(4)]
    return _faces(walls + [hi], color, (cx, y0 + h / 2, cz))


def _house(cx, cz, sx, sz, h, roof_h, yaw, wall, roof, y0):
    """Дом: стены + двускатная крыша (конёк вдоль длинной стороны sz)."""
    v1, t1, c1 = _box(cx, cz, sx, sz, h, yaw, wall, y0)
    def P(px, pz, y):
        a, b = _rot(px, pz, yaw)
        return (cx + a, y, cz + b)
    e, top = 0.6, y0 + h
    A, B = P(-sx / 2 - e, -sz / 2 - e, top), P(sx / 2 + e, -sz / 2 - e, top)
    Cc, D = P(sx / 2 + e, sz / 2 + e, top), P(-sx / 2 - e, sz / 2 + e, top)
    R0, R1 = P(0, -sz / 2 - e, top + roof_h), P(0, sz / 2 + e, top + roof_h)
    mid = (cx, y0 + h / 2, cz)
    v2, t2, c2 = _faces([[A, R0, R1, D], [B, Cc, R1, R0]], roof, mid)
    G0, G1 = P(-sx / 2, -sz / 2, top), P(sx / 2, -sz / 2, top)
    G2, G3 = P(sx / 2, sz / 2, top), P(-sx / 2, sz / 2, top)
    v3, t3, c3 = _faces([[G0, G1, P(0, -sz / 2, top + roof_h)],
                         [G2, G3, P(0, sz / 2, top + roof_h)]], wall, mid)
    return _merge_arrays([(v1, t1, c1), (v2, t2, c2), (v3, t3, c3)])


def _spire(cx, cz, r, h, color, y0):
    base = [(cx + r * np.cos(a), y0, cz + r * np.sin(a)) for a in np.arange(4) * np.pi / 2 + np.pi / 4]
    tip = (cx, y0 + h, cz)
    return _faces([[base[k], base[(k + 1) % 4], tip] for k in range(4)], color, (cx, y0, cz))


def _trees(X, Y, Z, rng):
    """Ели-пирамиды (4 грани), освещение по граням; всё одним мешем."""
    n = len(X)
    if n == 0:
        return Entity()
    hgt = 8 + 8 * rng.random(n)
    rad = 0.3 * hgt
    verts, cols = [], []
    for k in range(4):
        a0, a1 = k * np.pi / 2, (k + 1) * np.pi / 2
        p0 = np.stack([X + rad * np.cos(a0), Y - 0.3, Z + rad * np.sin(a0)], -1)
        p1 = np.stack([X + rad * np.cos(a1), Y - 0.3, Z + rad * np.sin(a1)], -1)
        tp = np.stack([X, Y + hgt, Z], -1)
        mid = 0.5 * (a0 + a1)
        sh = 0.6 + 0.4 * max(0.0, np.cos(mid) * SUN_DIR[0] + np.sin(mid) * SUN_DIR[2])
        verts.append(np.stack([p0, p1, tp], 1))
        cols.append(np.broadcast_to(PALETTE["tree"] * sh * (0.85 + 0.3 * rng.random((n, 1, 1))), (n, 3, 3)))
    v = np.concatenate(verts).reshape(-1, 3)
    return _mesh(v, np.arange(len(v)), np.concatenate(cols).reshape(-1, 3))


def _merge_arrays(parts):
    V, T, C, off = [], [], [], 0
    for v, t, c in parts:
        V.append(v); T.append(np.asarray(t) + off); C.append(c); off += len(v)
    return np.vstack(V), np.concatenate(T), np.vstack(C)


def _merge(parts):
    return _mesh(*_merge_arrays(parts))

# -*- coding: utf-8 -*-
"""
Мир 3D-тренажёра: карта + бесконечная равнина вокруг. Карты — make_world(имя):

  default — придуманная карта 10 × 10 км с ориентирами (DefaultMap);
  valley, alps, marsh — процедурные карты 10 × 10 км (viz/maps3d.py);

Координаты Ursina (см. viewer3d.py): X — восток, Y — вверх, Z — север, м.
Старт всех сценариев — (0, h0, 0), над ВПП карты; курс старта игры — start_psi.

Карта default (X ∈ [−5, 5] км, Z ∈ [−2.5, 7.5] км):
    ВПП 800 × 30 м в начале координат вдоль Z, ангар и вышка к востоку;
    река с севера на юг (X ≈ −1.6 км), впадает в озеро на юго-западе;
    деревня с церковью на западном берегу, дорога от аэродрома с мостом;
    холм 150 м на северо-востоке (скальная вершина), леса на склонах,
    вдоль реки, на северо-западе, роща у деревни; остальное — поля.
    Хутора (дом, сарай, навес, сад, стога) и деревушка на востоке, полевые дороги,
    ветряная мельница у деревни; леса смешанные, рощи и лесополосы среди полей,
    по берегу реки — ивы, в полях — одиночные деревья.
    Линия E = 0 к северу свободна от высоких препятствий — по ней летят s1–s9, p7–p10.

Вокруг любой карты — равнина: квадрат plain_p × plain_p (поля, редкие деревья,
пологие волны ±2.5 м), размноженный по сетке вокруг ЛА (Panda3D instanceTo —
геометрия одна, перестройки при полёте нет). Квадраты, попадающие на карту,
скрыты (границы карты кратны plain_p). Края карты плавно (fade м) переходят
в равнину, шва по высоте нет.

Высота поверхности height(X, Z) — та же функция, по которой построена сетка
(используется для тени следа и отвеса).

Твёрдая земля для физики (шасси, удар) — surface(X, Z): ровно то, что нарисовано —
узлы сетки карты grid() и квадратов равнины, та же разбивка клеток на треугольники,
+ покрытия поверх рельефа (_cover: ВПП, перрон, дороги, мост, вода). Так у каждой
карты земля своя и совпадает с картинкой (РЕШ-25). is_water(X, Z) — касание воды.

Вид сверху top_image() / save_top_image() — фон карты окна станции (viz/station.py):
та же раскраска, что у сетки, север вверх, охват map_x × map_z; строится
после build().
"""

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
FARMS = [(1800.0, 1200.0, 0.3), (2600.0, -1500.0, 1.1),            # хутора: (X, Z, поворот, рад)
         (3600.0, 2600.0, -0.4), (-800.0, 4600.0, 0.7),
         (900.0, -1900.0, 0.0), (4200.0, -1200.0, 0.5),
         (-700.0, 6500.0, 1.4), (2600.0, 6900.0, -0.2)]
HAMLET = (3300.0, 700.0)                                            # хутор-деревушка
TRACKS = [[(160, -250), (1500, -250), (1500, 1200), (1800, 1200)],  # полевые дороги (грунт)
          [(1500, -250), (1500, -1500), (2600, -1500), (4200, -1200)],
          [(1500, -1500), (1500, -1900), (900, -1900)],
          [(1500, 1200), (1500, 2600), (3600, 2600)],
          [(1800, 1200), (3300, 700)],
          [(-800, 150), (-750, 1500), (-950, 3000), (-800, 4600), (-700, 6500)],
          [(3600, 2600), (3300, 4000), (2600, 6900)]]
MILL = (-2550.0, 3150.0)                                            # ветряная мельница


def _groves():
    """Рощи (колки) среди полей: (X, Z, радиус); не над линией E = 0 к северу,
    не на воде, не на аэродроме и деревне. Детерминированно (seed)."""
    rng = np.random.default_rng(5)
    out = []
    while len(out) < 32:
        x, z, r = rng.uniform(-4500, 4500), rng.uniform(-2000, 7000), rng.uniform(50, 140)
        if (abs(x) < 350 and z > -600) or abs(x - river_x(z)) < 450 or np.hypot(x - LAKE_X, z - LAKE_Z) < 800 \
                or np.hypot(x - VILLAGE[0], z - VILLAGE[1]) < 500 or np.hypot(x - HAMLET[0], z - HAMLET[1]) < 300 \
                or min(np.hypot(x - fx, z - fz) for fx, fz, _ in FARMS) < 200:
            continue
        out.append((x, z, r))
    return out


def _belts():
    """Лесополосы по границам полей: отрезки вдоль X = const (шаг полей 200 м)."""
    rng = np.random.default_rng(6)
    out = []
    for x in np.arange(-4400, 4401, 400.0):
        if abs(x) < 400:
            continue
        z0 = rng.uniform(-2000, 5000)
        out.append([(x, z0), (x, z0 + rng.uniform(600, 1800))])
    return out



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
GROVES = _groves()                   # после river_x и LAKE_X — они нужны для проверок
BELTS = _belts()


def noise_mix(X, Z):
    """Гладкое поле ≈ [−1, 1] с пятнами ~400 м — где лес лиственный."""
    return 0.6 * np.sin(X / 260.0 + 1.3) * np.cos(Z / 310.0) + 0.4 * np.sin((X + Z) / 170.0)


def _seg_dist(X, Z, pts, near=None):
    """Расстояние от точек (X, Z) до ломаной pts. near — нужно только сравнение d ≤ near:
    отрезки, в габарит которых (± near) не попала ни одна точка, пропускаются (там d — inf)."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    d = np.full(np.broadcast(X, Z).shape, np.inf)
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        if near is not None and (X.max() < min(x0, x1) - near or X.min() > max(x0, x1) + near
                                 or Z.max() < min(z0, z1) - near or Z.min() > max(z0, z1) + near):
            continue
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

    CELL = 40.0                        # клетка сетки рельефа карты, м

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.period = None             # совместимость с View3D
        self._tiles, self._tile_ij = [], None
        self._grid, self._plain_grid = None, None

    def grid(self):
        """Узлы сетки рельефа карты (X, Y, Z), индексы [i по X, j по Z] — по ним
        строится сетка карты и считается surface()."""
        if self._grid is None:
            (x0, x1), (z0, z1) = self.map_x, self.map_z
            h = self.CELL
            xs = x0 + h * np.arange(int(round((x1 - x0) / h)) + 1)
            zs = z0 + h * np.arange(int(round((z1 - z0) / h)) + 1)
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            self._grid = (X, self.height(X, Z), Z)
        return self._grid

    def plain_grid(self):
        """Узлы квадрата равнины (X, Y, Z) от его угла; клетка ≈ 50 м."""
        if self._plain_grid is None:
            P = self.plain_p
            s = np.linspace(0, P, int(round(P / 50)) + 1)
            X, Z = np.meshgrid(s, s, indexing="ij")
            self._plain_grid = (X, self.plain(X, Z), Z)
        return self._plain_grid

    # --- твёрдая земля (физика) ------------------------------------------------
    def surface(self, X, Z):
        """Высота твёрдой поверхности, м: сетка как нарисована + покрытия (_cover)."""
        X, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Z, float))
        return self._cover(X, Z, self._mesh_height(X, Z))

    def is_water(self, X, Z):
        """True — в точке вода (касание — авария)."""
        return np.zeros(np.broadcast(X, Z).shape, bool)

    def _cover(self, X, Z, h):
        """Покрытия поверх сетки рельефа (ВПП и т. п.); h — высота сетки."""
        return h

    def _mesh_height(self, X, Z):
        """Высота нарисованной сетки: карта внутри map_x × map_z, квадраты равнины
        вне (квадрат стоит в (x0 + i·P, z0 + j·P) — см. follow())."""
        (x0, x1), (z0, z1) = self.map_x, self.map_z
        _, Ym, _ = self.grid()
        Xp, Yp, _ = self.plain_grid()
        P, cp = self.plain_p, Xp[1, 0]
        on_map = (X >= x0) & (X <= x1) & (Z >= z0) & (Z <= z1)
        hm = _tri_interp(Ym, (X - x0) / self.CELL, (Z - z0) / self.CELL)
        if on_map.all():                                   # все точки на карте — равнина не нужна
            return hm
        hp = _tri_interp(Yp, ((X - x0) % P) / cp, ((Z - z0) % P) / cp)
        return np.where(on_map, hm, hp)

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

    # --- вид сверху (окно карты) ---------------------------------------------
    def top_image(self, px: int = 720):
        """Вид карты сверху, север вверх: PIL RGB шириной px (высота — по пропорциям карты)."""
        from PIL import Image, ImageDraw
        (x0, x1), (z0, z1) = self.map_x, self.map_z
        W, H = px, int(round(px * (z1 - z0) / (x1 - x0)))
        img = self._top_base().convert("RGB").resize((W, H), Image.LANCZOS)
        to_px = lambda pts: [((x - x0) / (x1 - x0) * W, (z1 - z) / (z1 - z0) * H) for x, z in pts]
        self._top_overlay(ImageDraw.Draw(img), to_px, W / (x1 - x0))
        return img

    def save_top_image(self, px: int = 720) -> str:
        """Сохранить вид сверху в viz/map_cache/top_<имя>.png (кэш, не в git); путь к файлу."""
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_cache")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"top_{self.name}.png")
        self.top_image(px).save(path)
        return path

    def _top_base(self):
        raise NotImplementedError

    def _top_overlay(self, draw, to_px, scale):
        """Дорисовать поверх фона (дороги, ВПП); scale — пикс/м."""

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
        X, Y, Z = self.plain_grid()              # клетка ≈ 50 м
        n = X.shape[0] - 1
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
    CELL = CELL

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
        for x0, z0, r in GROVES:
            f |= np.hypot(X - x0, Z - z0) < r * (1 + n)
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

    # --- твёрдая земля: покрытия как в _build_water / _road / _airfield -------
    def _wet(self, X, Z, h):
        """Вода видна, где сетка ниже WATER_Y под лентой реки или кругом озера."""
        river = (np.abs(X - river_x(Z)) < 75) & (Z >= LAKE_Z)
        lake = np.hypot(X - LAKE_X, Z - LAKE_Z) < LAKE_R + 120
        return (h < WATER_Y) & (river | lake)

    def _cover(self, X, Z, h):
        h = np.where(self._wet(X, Z, h), WATER_Y, h)
        L, W = RUNWAY["len"], RUNWAY["w"]
        h = np.where((np.abs(X) <= W / 2) & (np.abs(Z) <= L / 2), np.maximum(h, 0.3), h)
        h = np.where((np.abs(X - 90) <= 60) & (np.abs(Z + 250) <= 60), np.maximum(h, 0.25), h)
        for t in TRACKS:                                       # полевые дороги — +0.15 м
            h = np.where(_seg_dist(X, Z, t, near=4) <= 4, h + 0.15, h)
        road = _seg_dist(X, Z, ROAD, near=4) <= 4
        bridge = np.abs(X - river_x(Z)) < 75
        return np.where(road, np.maximum(h, np.where(bridge, 1.5, h + 0.4)), h)

    def is_water(self, X, Z):
        X, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Z, float))
        band = self._wet(X, Z, np.full(X.shape, -np.inf))     # лента реки / круг озера
        if not band.any():                                     # вдали от воды — без рельефа
            return band
        return self._wet(X, Z, self._mesh_height(X, Z)) & (self.surface(X, Z) <= WATER_Y)

    # --- карта ------------------------------------------------------------
    def _build_map(self):
        root = Entity()
        X, Y, Z = self.grid()
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
        top = col.copy()                                   # для вида сверху: вода — где дно
        top[(Yc < WATER_Y) & valley] = (0.25, 0.45, 0.62)
        self._top_col = top

        self._build_water().parent = root
        self._build_road().parent = root
        self._build_airfield().parent = root
        self._build_village().parent = root
        self._build_farms().parent = root
        self._build_trees().parent = root
        return root

    def _top_base(self):
        from PIL import Image
        a = np.clip(self._top_col.transpose(1, 0, 2)[::-1] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(a)                          # строки — с севера на юг

    def _top_overlay(self, draw, to_px, scale):
        draw.line(to_px(ROAD), fill=(92, 87, 79), width=max(1, round(8 * scale)))
        L, W = RUNWAY["len"], RUNWAY["w"]
        draw.polygon(to_px([(-W / 2, -L / 2), (W / 2, -L / 2), (W / 2, L / 2), (-W / 2, L / 2)]),
                     fill=(82, 82, 87))
        draw.polygon(to_px([(30, -310), (150, -310), (150, -190), (30, -190)]), fill=(107, 107, 112))
        cx, cz = VILLAGE
        draw.ellipse(to_px([(cx - 230, cz + 230), (cx + 230, cz - 230)]), outline=(150, 120, 100))
        for t in TRACKS:
            draw.line(to_px(t), fill=(150, 128, 92), width=max(1, round(5 * scale)))
        for fx, fz, _ in FARMS + [HAMLET + (0,)]:
            draw.rectangle(to_px([(fx - 25, fz + 25), (fx + 25, fz - 25)]), fill=(150, 72, 52))

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

    def _track_mesh(self, pts):
        pts = np.array(pts, float)
        seg = [np.linspace(p0, p1, max(2, int(np.hypot(*(p1 - p0)) / 10)), endpoint=False)
               for p0, p1 in zip(pts[:-1], pts[1:])]
        P = np.vstack(seg + [pts[-1:]])
        d = np.gradient(P, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], -1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        y = self._mesh_height(P[:, 0], P[:, 1]) + 0.2
        L = np.stack([P[:, 0] - 4 * n[:, 0], y, P[:, 1] - 4 * n[:, 1]], -1)
        R = np.stack([P[:, 0] + 4 * n[:, 0], y, P[:, 1] + 4 * n[:, 1]], -1)
        v, t = _ribbon(L, R)
        return v, t, np.tile([0.48, 0.40, 0.29], (len(v), 1))

    def _build_farms(self):
        """Хутора, мельница, стога на лугах у реки, полевые дороги."""
        rng = np.random.default_rng(23)
        parts = [self._track_mesh(t) for t in TRACKS]
        wall, roof = (0.93, 0.90, 0.82), (0.62, 0.22, 0.15)
        barn_w, barn_r = (0.55, 0.40, 0.28), (0.40, 0.36, 0.32)
        for fx, fz, a in FARMS:
            y = lambda x, z: float(self.height(x, z)) - 0.4
            def at(px, pz):                                  # точка в осях хутора
                return fx + px * np.cos(a) + pz * np.sin(a), fz - px * np.sin(a) + pz * np.cos(a)
            parts.append(_house(fx, fz, 9, 12, 4.5, 3.5, a, wall, roof, y(fx, fz)))
            bx, bz = at(28, 6)
            parts.append(_house(bx, bz, 12, 22, 5, 4, a, barn_w, barn_r, y(bx, bz)))
            sx, sz = at(6, 30)
            parts.append(_house(sx, sz, 8, 14, 3, 1.5, a + np.pi / 2, barn_w, barn_r, y(sx, sz)))
            for k in range(5):                               # стога
                hx, hz = at(rng.uniform(40, 120), rng.uniform(-60, 60))
                parts.append(_spire(hx, hz, 2.6, 4.5, (0.72, 0.62, 0.34), y(hx, hz) + 0.3))
        X = rng.uniform(-3000, 0, 3000)                      # стога на лугах вдоль реки
        Z = rng.uniform(LAKE_Z + 600, MAP_Z[1] - 800, 3000)
        d = self.river_dist(X, Z)
        ok = (d > 260) & (d < 380) & (np.hypot(X - VILLAGE[0], Z - VILLAGE[1]) > 400)
        for x, z in list(zip(X[ok], Z[ok]))[:60]:
            parts.append(_spire(x, z, 2.6, 4.5, (0.72, 0.62, 0.34), float(self.height(x, z)) - 0.1))
        parts += self._mill(*MILL)
        cx, cz = HAMLET                                      # деревушка: ~20 домов вокруг пруда-луга
        for k in range(22):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(40, 170)
            x, z = cx + r * np.cos(a), cz + r * np.sin(a)
            if any(_seg_dist(np.array([x]), np.array([z]), t)[0] < 12 for t in TRACKS):
                continue
            parts.append(_house(x, z, rng.uniform(7, 10), rng.uniform(8, 12), rng.uniform(4, 5.5),
                                rng.uniform(2.5, 4), a + np.pi / 2, wall, roof, float(self.height(x, z)) - 0.5))
        return _merge(parts)

    def _mill(self, cx, cz):
        """Ветряная мельница: башня-шатёр, крыша, четыре крыла (плоские лопасти на юг)."""
        y0 = float(self.height(cx, cz)) - 0.5
        wood, dark = (0.62, 0.50, 0.36), (0.30, 0.26, 0.24)
        parts = [_box(cx, cz, 8, 8, 12, 0.0, wood, y0), _spire(cx, cz, 6.5, 5, dark, y0 + 12)]
        hub = np.array([cx, y0 + 13, cz - 5.0])
        for k in range(4):
            a = np.pi / 4 + k * np.pi / 2
            u = np.array([np.cos(a), np.sin(a), 0.0])        # вдоль крыла
            w = np.array([-np.sin(a), np.cos(a), 0.0])       # поперёк
            p0, p1 = hub + 1.5 * u, hub + 11 * u
            quad = [p0 - 0.2 * w, p1 - 0.2 * w, p1 + 1.6 * w, p0 + 1.6 * w]
            parts.append(_faces([quad], (0.88, 0.85, 0.78), (cx, y0 + 13, cz)))
        return parts

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
        parts.append(_house(110, -330, 30, 24, 8, 5, 0, (0.66, 0.70, 0.74), (0.40, 0.46, 0.54), 0))  # ангар 2
        for z in (-200.0, -188.0):                                          # топливные баки
            parts.append(_box(150, z, 5, 9, 4, np.pi / 2, (0.90, 0.90, 0.88), 0))
        parts.append(_box(40, 330, 0.3, 0.3, 7, 0, (0.85, 0.85, 0.85), 0))   # ветроуказатель
        parts.append(_box(40, 332.5, 0.9, 5, 0.9, 0, (0.95, 0.45, 0.15), 6.2))
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
        from viz.maps3d import _crowns                           # лиственные — кроны-октаэдры
        rng = self.rng
        X = rng.uniform(*MAP_X, 220000)
        Z = rng.uniform(*MAP_Z, 220000)
        Y = self.height(X, Z)
        ok = ((Y > -0.5) & (self.runway_mask(X, Z) > 0.6) & (_seg_dist(X, Z, ROAD) > 10)
              & (np.hypot(X - VILLAGE[0], Z - VILLAGE[1]) > 330))
        for t in TRACKS:
            ok &= _seg_dist(X, Z, t, near=8) > 7
        for fx, fz, _ in FARMS:
            ok &= np.hypot(X - fx, Z - fz) > 45
        ok &= np.hypot(X - MILL[0], Z - MILL[1]) > 25
        ok &= np.hypot(X - HAMLET[0], Z - HAMLET[1]) > 190
        f = ok & self.forest(X, Z)
        mixed = noise_mix(X, Z) > 0.15                           # пятна лиственного леса
        con = f & ~mixed
        brd = f & mixed & (rng.random(X.size) < 0.8)
        dr = self.river_dist(X, Z)
        brd |= ok & (dr > 80) & (dr < 130) & (np.sin(Z / 140.0) + np.sin(Z / 53.0) > -0.6)  # ивы
        for fx, fz, a in FARMS:                                  # сады за домом
            gx, gz = fx - 40 * np.cos(a) + 20 * np.sin(a), fz + 40 * np.sin(a) + 20 * np.cos(a)
            brd |= ok & (np.hypot(X - gx, Z - gz) < 40) & (rng.random(X.size) < 0.6)
        for b in BELTS:                                          # лесополосы
            brd |= ok & (_seg_dist(X, Z, b, near=8) < 7) & (rng.random(X.size) < 0.9)
        lone = ok & ~f & (rng.random(X.size) < 0.012)            # одиночные деревья в полях
        brd |= lone & (rng.random(X.size) < 0.7)
        con |= lone & ~brd
        ic = np.flatnonzero(con)[:12000]
        ib = np.flatnonzero(brd & ~con)[:12000]
        root = Entity()
        _trees(X[ic], Y[ic], Z[ic], rng).parent = root
        _crowns(X[ib], Y[ib], Z[ib], rng).parent = root
        return root


MAPS = {"default": DefaultMap}


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


def _mesh(verts, tris, cols, uvs=None):
    m = Mesh(vertices=[tuple(v) for v in np.asarray(verts, float).tolist()],
             triangles=np.asarray(tris).ravel().tolist(),
             colors=[Color(r, g, b, 1) for r, g, b in np.asarray(cols, float).tolist()],
             uvs=None if uvs is None else [tuple(u) for u in np.asarray(uvs, float).tolist()],
             static=True)
    return Entity(model=m, shader=unlit_with_fog_shader, double_sided=True)


def _grid_mesh(X, Y, Z, col, uv_size=None):
    """
    Сетка: каждый квадрат — свои 4 вершины (чёткие границы), освещение по нормали.
    uv_size — размер карты, м: тогда UV вершин = (X, Z) / uv_size + 0.5 (для текстуры).
    """
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
    if uv_size is None:
        return _mesh(verts, tris, cols)
    uvs = verts[:, [0, 2]] / uv_size + 0.5
    return _mesh(verts, tris, cols, uvs=uvs)


def _tri_interp(Y, gx, gz):
    """
    Высота сетки Y[i, j] в точке с дробными индексами (gx, gz) — по треугольникам,
    как их рисуют _grid_mesh и _terrain: клетка делится диагональю (i, j)–(i+1, j+1).
    """
    i = np.clip(np.floor(gx).astype(int), 0, Y.shape[0] - 2)
    j = np.clip(np.floor(gz).astype(int), 0, Y.shape[1] - 2)
    fx, fz = np.clip(gx - i, 0, 1), np.clip(gz - j, 0, 1)
    y00, y10, y11, y01 = Y[i, j], Y[i + 1, j], Y[i + 1, j + 1], Y[i, j + 1]
    return np.where(fx >= fz, y00 + fx * (y10 - y00) + fz * (y11 - y10),
                    y00 + fz * (y01 - y00) + fx * (y11 - y01))


def _in_convex(X, Z, poly):
    """Точки (X, Z) внутри выпуклого многоугольника poly (обход в любую сторону)."""
    poly = np.asarray(poly, float)
    cr = [(b[0] - a[0]) * (Z - a[1]) - (b[1] - a[1]) * (X - a[0])
          for a, b in zip(poly, np.roll(poly, -1, 0))]
    return np.logical_and.reduce([c >= 0 for c in cr]) | np.logical_and.reduce([c <= 0 for c in cr])


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


# Процедурные карты 10 × 10 км (viz/maps3d.py): модуль сам добавляет их в MAPS.
# Импорт в конце файла — модулю карт нужны World и функции геометрии выше.
import viz.maps3d                   # noqa: E402,F401

# -*- coding: utf-8 -*-
"""
Процедурные карты 3D-тренажёра 10 × 10 км (вымышленная местность):

  valley — долина реки: меандрирующая река в корытообразной долине, холмы до ≈ 400 м,
           виноградники на южных склонах, деревни по берегам, замок на отроге,
           аэродром среди домов (по мотивам долин Дуная и Мозеля);
  alps   — горы: котловина с озером, городом и аэродромом, хребты до ≈ 1900 м
           со снегом и скалами, выход из котловины двумя долинами
           (по мотивам альпийских котловин с озером);
  marsh  — болото: восточная половина — поля и деревни, западная — пойменное
           болото: река на входе распадается на рукава, старицы, озерки,
           лесистые острова, гать к деревне на острове (по мотивам болот Полесья).

Общая схема — ProcMap: рельеф relief(X, Z) — аналитическая функция (шум + формы),
height() = relief с переходом в равнину на краю и выровненным аэродромом в начале
координат; вода — всё, что ниже WATER_Y (одна отметка на карту); раскраска — по
клеткам сетки (colors); дороги, деревни, деревья — списки, которые задаёт карта.
Твёрдая земля surface() — как у World: нарисованная сетка + покрытия (ВПП, дороги,
мосты, вода).

Шум — значения в узлах решётки (value noise) со сглаживающей интерполяцией;
fBm — сумма октав (каждая — вдвое мельче и вдвое слабее), ridged — «гребни»
1 − |шум| (острые хребты). Всё детерминировано (seed), карта при каждом запуске одна.

Модуль импортируется из конца viz/world3d.py (нужны World и функции геометрии).
"""

import numpy as np
from ursina import Entity

from viz.world3d import (World, PALETTE, SUN_DIR, _smooth, _seg_dist, _blend, _grid_mesh,
                         _mesh, _ribbon, _box, _house, _spire, _trees, _merge, _tri_interp)


# ==========================================================================
# Шум
# ==========================================================================

def _hash(ix, iz, seed):
    """Псевдослучайное число в узле решётки (ix, iz): [−1, 1]."""
    h = (ix * 374761393 + iz * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFF) / 32767.5 - 1.0


def noise(X, Z, scale, seed=0):
    """Шум значений с периодом решётки scale, м: гладкий, [−1, 1]."""
    a = 0.6 + 0.37 * seed                       # поворот решётки — без «клеток» по осям
    x = (X * np.cos(a) - Z * np.sin(a)) / scale + 1000.0
    z = (X * np.sin(a) + Z * np.cos(a)) / scale + 1000.0
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = x - ix, z - iz
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)            # сглаживание 5-го порядка
    uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    ix, iz = ix.astype(np.int64), iz.astype(np.int64)
    a00, a10 = _hash(ix, iz, seed), _hash(ix + 1, iz, seed)
    a01, a11 = _hash(ix, iz + 1, seed), _hash(ix + 1, iz + 1, seed)
    return a00 + (a10 - a00) * ux + (a01 - a00) * uz + (a00 - a10 - a01 + a11) * ux * uz


def fbm(X, Z, scale, octaves=5, seed=0, gain=0.5):
    """Сумма октав шума (fractal Brownian motion), нормирована: ≈ [−0.7, 0.7]."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    s, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        s = s + amp * noise(X, Z, scale / 2 ** k, seed * 31 + k)
        tot += amp
        amp *= gain
    return s / tot


def ridged(X, Z, scale, octaves=5, seed=0):
    """Гребни: октавы (1 − |шум|)², каждая следующая — слабее там, где предыдущая низкая. [0, 1]."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    s, amp, w, tot = 0.0, 1.0, 1.0, 0.0
    for k in range(octaves):
        n = (1.0 - np.abs(noise(X, Z, scale / 2 ** k, seed * 17 + k))) ** 2
        s = s + amp * n * w
        w = np.clip(n * 1.5, 0, 1)
        tot += amp
        amp *= 0.5
    return s / tot


def smin(a, b, k):
    """Гладкий минимум: стык скруглён там, где |a − b| < k, м; дальше — точно min(a, b).
    Не опускается ниже min(a, b) − 1 м (у дна долин не роет ям ниже уровня воды)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    s = b * (1 - h) + a * h - k * h * (1 - h)
    return np.maximum(s, np.minimum(a, b) - 1.0)


def _seg_along(X, Z, p0, p1):
    """Отрезок p0 → p1: (расстояние вдоль от p0, м; расстояние поперёк, м)."""
    (x0, z0), (x1, z1) = p0, p1
    L = np.hypot(x1 - x0, z1 - z0)
    ux, uz = (x1 - x0) / L, (z1 - z0) / L
    s = (X - x0) * ux + (Z - z0) * uz
    return s, np.abs(-(X - x0) * uz + (Z - z0) * ux), L


# ==========================================================================
# Деревья и постройки
# ==========================================================================

BROADLEAF = np.array([0.24, 0.42, 0.17])


def _crowns(X, Y, Z, rng, color=BROADLEAF, hmin=6.0, hmax=12.0):
    """Лиственные деревья: крона-октаэдр (8 граней), снизу почти до земли; одним мешем."""
    n = len(X)
    if n == 0:
        return Entity()
    h = hmin + (hmax - hmin) * rng.random(n)
    r = 0.35 * h
    y0, ym, y1 = Y + 0.8, Y + 0.8 + 0.45 * h, Y + h
    tips = [np.stack([X, y0, Z], -1), np.stack([X, y1, Z], -1)]
    ring = [np.stack([X + r * np.cos(a), ym, Z + r * np.sin(a)], -1) for a in np.arange(4) * np.pi / 2]
    tint = color * (0.8 + 0.4 * rng.random((n, 1)))
    V, C = [], []
    for k in range(4):
        a, b = ring[k], ring[(k + 1) % 4]
        mid = (k + 0.5) * np.pi / 2
        for tip, up in ((tips[1], 1.0), (tips[0], -1.0)):
            nrm = np.array([np.cos(mid), 0.7 * up, np.sin(mid)])
            sh = 0.55 + 0.45 * max(0.0, float(nrm @ SUN_DIR) / np.linalg.norm(nrm))
            V.append(np.stack([a, b, tip], 1))
            C.append(np.repeat((tint * sh)[:, None, :], 3, 1))
    v = np.concatenate(V).reshape(-1, 3)
    return _mesh(v, np.arange(len(v)), np.concatenate(C).reshape(-1, 3))


WALLS = dict(village=[(0.93, 0.90, 0.82), (0.88, 0.84, 0.72), (0.95, 0.95, 0.93), (0.85, 0.78, 0.62)],
             town=[(0.95, 0.94, 0.90), (0.92, 0.88, 0.78), (0.86, 0.86, 0.84), (0.93, 0.85, 0.70)],
             chalet=[(0.55, 0.38, 0.22), (0.62, 0.45, 0.28), (0.50, 0.34, 0.20)])
ROOFS = dict(village=[(0.62, 0.22, 0.15), (0.70, 0.35, 0.20), (0.45, 0.25, 0.18), (0.35, 0.36, 0.40)],
             town=[(0.30, 0.31, 0.34), (0.55, 0.22, 0.16), (0.38, 0.30, 0.26), (0.25, 0.27, 0.30)],
             chalet=[(0.28, 0.24, 0.22), (0.35, 0.30, 0.26), (0.45, 0.42, 0.40)])


# ==========================================================================
# База процедурной карты
# ==========================================================================

class ProcMap(World):
    """
    Процедурная карта 10 × 10 км, аэродром — в начале координат.
    Наследник задаёт relief(), colors(), а также ROADS, VILLAGES и маски деревьев.
    """

    map_x, map_z = (-5000.0, 5000.0), (-5000.0, 5000.0)
    plain_p, fade = 2500.0, 1500.0
    CELL = 40.0
    WATER_Y = -2.0                     # уровень воды, м
    RUNWAY = dict(len=800.0, w=30.0, bearing=0.0)   # курс ВПП, град; центр — (0, 0)
    APRON = (90.0, -250.0)             # центр перрона в осях ВПП (поперёк, вдоль), м
    SEED = 1
    TREES_MAX = 16000                  # хвойных; лиственных — столько же
    SHORE = 0.0                        # отступ суши и дна от уровня воды, м (см. shore)

    # --- что задаёт карта ---------------------------------------------------
    def relief(self, X, Z):
        """Рельеф без перехода в равнину и без аэродрома, м."""
        raise NotImplementedError

    def colors(self, X, Z, Y, slope, rng):
        """Цвета клеток (…, 3) по центрам клеток X, Z, высоте Y и уклону slope (tg)."""
        raise NotImplementedError

    def tree_masks(self, X, Z, Y, slope, rng):
        """(хвойные, лиственные) — маски для случайных точек."""
        f = np.zeros(X.shape, bool)
        return f, f

    def extras(self):
        """Дополнительные постройки: список (verts, tris, cols)."""
        return []

    ROADS = []          # [(точки [(X, Z)…], полуширина м, подъём над рельефом м)]
    VILLAGES = []       # [dict(c=(X, Z), r=м, step=м, style='village'|'town'|'chalet', church=bool)]

    @property
    def start_psi(self):
        return np.radians(self.RUNWAY["bearing"])

    # --- оси ВПП и аэродром -------------------------------------------------
    def _rw(self, X, Z):
        """Координаты в осях ВПП: (вдоль, поперёк), м."""
        b = np.radians(self.RUNWAY["bearing"])
        return X * np.sin(b) + Z * np.cos(b), X * np.cos(b) - Z * np.sin(b)

    def _from_rw(self, v, u):
        """Обратно: (поперёк, вдоль) → (X, Z)."""
        b = np.radians(self.RUNWAY["bearing"])
        return v * np.cos(b) + u * np.sin(b), -v * np.sin(b) + u * np.cos(b)

    def airfield_mask(self, X, Z):
        """0 — площадка аэродрома (ровно, h = 0), 1 — вдали."""
        u, v = self._rw(X, Z)
        a = _smooth(0, 250, np.abs(u) - 0.5 * self.RUNWAY["len"] - 60)
        c = _smooth(0, 250, np.abs(v - 40) - 160)
        return np.maximum(a, c)

    def on_airfield(self, X, Z, margin=0.0):
        u, v = self._rw(X, Z)
        return (np.abs(u) < 0.5 * self.RUNWAY["len"] + 80 + margin) & (np.abs(v - 40) < 170 + margin)

    # --- высота --------------------------------------------------------------
    def shore(self, d):
        """Высота над водой d → d ± SHORE: суша не ниже +SHORE, дно не выше −SHORE.
        Иначе почти плоская земля вровень с водой на дальних планах мерцает с ней (не
        хватает точности буфера глубины). Сетку образуют только узлы, поэтому скачок
        функции не виден: клетка, пересекающая воду, получает перепад ≥ 2·SHORE и идёт
        к воде под углом, а не параллельно ей. Берег остаётся на том же месте."""
        return d + self.SHORE * np.where(d >= 0, 1.0, -1.0)

    def height(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        w = self.inside(X, Z)
        H = w * self.relief(X, Z) + (1 - w) * self.plain(X, Z)
        if self.SHORE:                                   # после перехода в равнину — и там тоже
            H = self.WATER_Y + self.shore(H - self.WATER_Y)
        return H * self.airfield_mask(X, Z)

    def ground(self, X, Z):
        """Высота нарисованной сетки (быстро: интерполяция узлов) — для деревьев и домов."""
        X, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Z, float))
        return self._mesh_height(X, Z)

    def slope(self, X, Z):
        """Уклон рельефа (тангенс угла) по узлам сетки."""
        if getattr(self, "_slope", None) is None:
            _, Y, _ = self.grid()
            gx, gz = np.gradient(Y, self.CELL)
            self._slope = np.hypot(gx, gz)
        (x0, _), (z0, _) = self.map_x, self.map_z
        return _tri_interp(self._slope, (np.asarray(X) - x0) / self.CELL, (np.asarray(Z) - z0) / self.CELL)

    # --- твёрдая земля --------------------------------------------------------
    def _wet(self, X, Z, h):
        """Вода: сетка ниже уровня и не в переходе к равнине (там низины — суша)."""
        return (h < self.WATER_Y) & (self.inside(X, Z) > 0.3)

    def _cover(self, X, Z, h):
        h = np.where(self._wet(X, Z, h), self.WATER_Y, h)
        u, v = self._rw(X, Z)
        L, W = self.RUNWAY["len"], self.RUNWAY["w"]
        h = np.where((np.abs(v) <= W / 2) & (np.abs(u) <= L / 2), np.maximum(h, 0.3), h)
        av, au = self.APRON
        h = np.where((np.abs(v - av) <= 60) & (np.abs(u - au) <= 60), np.maximum(h, 0.25), h)
        for pts, hw, lift in self.ROADS:
            on = _seg_dist(X, Z, pts, near=hw) <= hw
            if on.any():
                h = np.where(on, np.maximum(h, self._road_y(h, lift)), h)
        return h

    def _road_y(self, h, lift):
        """Отметка полотна дороги: над рельефом на lift, над водой — мост."""
        return np.maximum(h + lift, np.where(h < self.WATER_Y + 1.0, self.WATER_Y + 2.5, -1e9))

    def is_water(self, X, Z):
        X, Z = np.broadcast_arrays(np.asarray(X, float), np.asarray(Z, float))
        hm = self._mesh_height(X, Z)
        return self._wet(X, Z, hm) & (self.surface(X, Z) <= self.WATER_Y)

    # --- построение -----------------------------------------------------------
    def _build_map(self):
        root = Entity()
        rng = np.random.default_rng(self.SEED)
        X, Y, Z = self.grid()
        c = self.CELL
        Xc, Zc = X[:-1, :-1] + c / 2, Z[:-1, :-1] + c / 2
        Yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        gx = 0.5 * (Y[1:, :-1] + Y[1:, 1:] - Y[:-1, :-1] - Y[:-1, 1:]) / c
        gz = 0.5 * (Y[:-1, 1:] + Y[1:, 1:] - Y[:-1, :-1] - Y[1:, :-1]) / c
        col = self.colors(Xc, Zc, Yc, np.hypot(gx, gz), rng)
        col = _mix(col, np.asarray(self.plain_color), 1 - _smooth(0.0, 0.6, self.inside(Xc, Zc)))
        _grid_mesh(X, Y, Z, col).parent = root
        Ymin = np.minimum.reduce([Y[:-1, :-1], Y[1:, :-1], Y[1:, 1:], Y[:-1, 1:]])
        wet = self._wet(Xc, Zc, Ymin)
        top = col.copy()
        top[self._wet(Xc, Zc, Yc)] = self.water_color
        self._top_col = top

        self._build_water(Xc, Zc, wet).parent = root
        self._build_roads().parent = root
        self._build_airfield().parent = root
        houses = self._build_villages(rng)
        extra = self.extras()
        for v, _, _ in extra:                                                # постройки — на вид сверху
            self._roofs.append((float(v[:, 0].mean()), float(v[:, 2].mean()), 8.0))
        parts = houses + extra
        if parts:
            _merge(parts).parent = root
        self._build_trees(rng).parent = root
        return root

    plain_color = (0.50, 0.64, 0.30)
    water_color = (0.25, 0.45, 0.62)

    def _build_water(self, Xc, Zc, wet):
        """Плоская вода на клетках, где хоть один узел ниже уровня (выше — закрывает берег)."""
        x, z = Xc[wet], Zc[wet]
        if x.size == 0:
            return Entity()
        h, y = self.CELL / 2, self.WATER_Y
        v = np.stack([np.stack([x - h, np.full_like(x, y), z - h], -1),
                      np.stack([x + h, np.full_like(x, y), z - h], -1),
                      np.stack([x + h, np.full_like(x, y), z + h], -1),
                      np.stack([x - h, np.full_like(x, y), z + h], -1)], 1).reshape(-1, 3)
        b = np.arange(x.size) * 4
        t = np.stack([b, b + 1, b + 2, b, b + 2, b + 3], -1)
        return _mesh(v, t, np.tile(self.water_color, (len(v), 1)))

    def _road_mesh(self, pts, hw, lift):
        pts = np.asarray(pts, float)
        seg = [np.linspace(p0, p1, max(2, int(np.hypot(*(p1 - p0)) / 10)), endpoint=False)
               for p0, p1 in zip(pts[:-1], pts[1:])]
        P = np.vstack(seg + [pts[-1:]])
        d = np.gradient(P, axis=0)
        n = np.stack([-d[:, 1], d[:, 0]], -1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        h = self.ground(P[:, 0], P[:, 1])
        y = self._road_y(h, lift) + 0.05
        bridge = h < self.WATER_Y + 1.0
        L = np.stack([P[:, 0] - hw * n[:, 0], y, P[:, 1] - hw * n[:, 1]], -1)
        R = np.stack([P[:, 0] + hw * n[:, 0], y, P[:, 1] + hw * n[:, 1]], -1)
        v, t = _ribbon(L, R)
        base = (0.36, 0.34, 0.31) if lift < 1.0 else (0.50, 0.44, 0.33)      # насыпь — грунтовая
        c = np.where(np.repeat(bridge, 2)[:, None], [0.42, 0.30, 0.20], base)
        return v, t, c

    def _build_roads(self):
        if not self.ROADS:
            return Entity()
        return _merge([self._road_mesh(*r) for r in self.ROADS])

    def _build_airfield(self):
        L, W = self.RUNWAY["len"], self.RUNWAY["w"]
        b = self.RUNWAY["bearing"]
        def P(v, u, y=0.3):
            x, z = self._from_rw(v, u)
            return (x, y, z)
        def quad(v, u, sv, su, y, colr):
            pts = [P(v - sv / 2, u - su / 2, y), P(v + sv / 2, u - su / 2, y),
                   P(v + sv / 2, u + su / 2, y), P(v - sv / 2, u + su / 2, y)]
            return np.array(pts), np.array([0, 1, 2, 0, 2, 3]), np.tile(colr, (4, 1))
        av, au = self.APRON
        parts = [quad(0, 0, W, L, 0.3, (0.32, 0.32, 0.34)),
                 quad(av, au, 120, 120, 0.25, (0.42, 0.42, 0.44))]
        for u in np.arange(-0.45 * L, 0.45 * L, 50):
            parts.append(quad(0, u, 1.2, 15, 0.35, (0.95, 0.95, 0.95)))
        for v in (-W / 2 + 1, W / 2 - 1):
            parts.append(quad(v, 0, 0.8, L, 0.35, (0.95, 0.95, 0.95)))
        hx, hz = self._from_rw(av + 20, au - 10)
        yaw = np.radians(b)
        parts.append(_house(hx, hz, 40, 30, 10, 6, yaw, (0.70, 0.74, 0.80), (0.45, 0.50, 0.58), 0))
        tx, tz = self._from_rw(av + 5, au + 100)
        parts.append(_box(tx, tz, 6, 6, 18, yaw, (0.85, 0.85, 0.82), 0))     # вышка
        parts.append(_box(tx, tz, 8, 8, 3, yaw, (0.30, 0.40, 0.50), 18))
        return _merge(parts)

    def _house_ok(self, X, Z):
        """Где можно ставить дом: сухо, полого, не на дороге и не на аэродроме."""
        ok = (self.ground(X, Z) > self.WATER_Y + 1.5) & (self.slope(X, Z) < 0.28)
        ok &= ~self.on_airfield(X, Z)
        for pts, hw, _ in self.ROADS:
            ok &= _seg_dist(X, Z, pts, near=hw + 10) > hw + 8
        return ok

    def _build_villages(self, rng):
        parts = []
        self._roofs = []                   # (X, Z, размер) — для вида сверху
        for v in self.VILLAGES:
            (cx, cz), r, s, style = v["c"], v["r"], v.get("step", 40.0), v.get("style", "village")
            g = np.arange(-r, r + 1, s)
            GX, GZ = np.meshgrid(g, g)
            a = v.get("yaw", 0.0)
            gx, gz = GX.ravel(), GZ.ravel()
            jit = 0.0 if style == "town" else 0.3 * s
            X = cx + gx * np.cos(a) - gz * np.sin(a) + rng.uniform(-jit, jit, gx.size)
            Z = cz + gx * np.sin(a) + gz * np.cos(a) + rng.uniform(-jit, jit, gx.size)
            d = np.hypot(X - cx, Z - cz)
            keep = (d < r * (0.75 + 0.25 * rng.random(X.size))) & self._house_ok(X, Z)
            if v.get("church"):
                keep &= d > 40
            walls, roofs = WALLS[style], ROOFS[style]
            for x, z in zip(X[keep], Z[keep]):
                if style == "town":
                    sx, sz, h, rh = rng.uniform(10, 15), rng.uniform(12, 20), rng.uniform(7, 13), rng.uniform(3, 5)
                    yaw = a + (np.pi / 2 if rng.random() < 0.5 else 0.0)
                elif style == "chalet":
                    sx, sz, h, rh = rng.uniform(8, 11), rng.uniform(10, 13), rng.uniform(5, 7), rng.uniform(2.5, 3.5)
                    yaw = rng.uniform(0, np.pi)
                else:
                    sx, sz, h, rh = rng.uniform(7, 10), rng.uniform(8, 12), rng.uniform(4, 5.5), rng.uniform(2.5, 4)
                    yaw = rng.uniform(0, np.pi)
                e = 0.5 * max(sx, sz)
                y0 = float(np.min(self.ground(np.array([x - e, x + e, x - e, x + e, x]),
                                              np.array([z - e, z - e, z + e, z + e, z])))) - 0.3
                parts.append(_house(x, z, sx, sz, h, rh, yaw, walls[rng.integers(len(walls))],
                                    roofs[rng.integers(len(roofs))], y0))
                self._roofs.append((x, z, max(sx, sz)))
            if v.get("church"):
                parts += self._church(cx, cz, a)
        return parts

    def _church(self, cx, cz, yaw):
        y0 = float(self.ground(cx, cz)) - 0.5
        tx, tz = cx + 13 * np.sin(yaw), cz + 13 * np.cos(yaw)
        return [_house(cx, cz, 12, 24, 11, 6, yaw, (0.96, 0.96, 0.94), (0.30, 0.45, 0.35), y0),
                _box(tx, tz, 6, 6, 26, yaw, (0.96, 0.96, 0.94), y0),
                _spire(tx, tz, 4.5, 12, (0.85, 0.70, 0.25), y0 + 26)]

    def _build_trees(self, rng):
        N = 300000
        X = rng.uniform(*self.map_x, N)
        Z = rng.uniform(*self.map_z, N)
        ok = self.inside(X, Z) > 0.4
        X, Z = X[ok], Z[ok]
        Y = self.ground(X, Z)
        sl = self.slope(X, Z)
        free = (Y > self.WATER_Y + 0.6) & ~self.on_airfield(X, Z, 60)
        for pts, hw, _ in self.ROADS:
            free &= _seg_dist(X, Z, pts, near=hw + 8) > hw + 6
        for v in self.VILLAGES:
            free &= np.hypot(X - v["c"][0], Z - v["c"][1]) > 0.8 * v["r"]
        con, brd = self.tree_masks(X, Z, Y, sl, rng)
        ic = np.flatnonzero(con & free)[:self.TREES_MAX]
        ib = np.flatnonzero(brd & free & ~con)[:self.TREES_MAX]
        root = Entity()
        _trees(X[ic], Y[ic], Z[ic], rng).parent = root
        _crowns(X[ib], Y[ib], Z[ib], rng, self.broadleaf).parent = root
        return root

    broadleaf = BROADLEAF

    # --- вид сверху -------------------------------------------------------------
    def _top_base(self):
        from PIL import Image
        a = np.clip(self._top_col.transpose(1, 0, 2)[::-1] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    def _top_overlay(self, draw, to_px, scale):
        for x, z, sz in getattr(self, "_roofs", []):
            (px, pz), = to_px([(x, z)])
            r = max(1.0, 0.5 * sz * scale)
            draw.rectangle([px - r, pz - r, px + r, pz + r], fill=(150, 72, 52))
        for pts, hw, lift in self.ROADS:
            draw.line(to_px(pts), fill=(92, 87, 79) if lift < 1 else (128, 112, 84),
                      width=max(1, round(2 * hw * scale)))
        L, W = self.RUNWAY["len"], self.RUNWAY["w"]
        corners = [self._from_rw(v, u) for v, u in ((-W / 2, -L / 2), (W / 2, -L / 2), (W / 2, L / 2), (-W / 2, L / 2))]
        draw.polygon(to_px(corners), fill=(82, 82, 87))
        av, au = self.APRON
        corners = [self._from_rw(av + a, au + b) for a, b in ((-60, -60), (60, -60), (60, 60), (-60, 60))]
        draw.polygon(to_px(corners), fill=(107, 107, 112))


def _patch_colors(X, Z, size_x, size_z, rng, yaw=0.3):
    """Пёстрые поля: лоскуты size_x × size_z м (повёрнутая сетка), у каждого — своя культура."""
    u = X * np.cos(yaw) - Z * np.sin(yaw)
    w = X * np.sin(yaw) + Z * np.cos(yaw)
    fi = np.floor(u / size_x).astype(np.int64) * 100003 + np.floor(w / size_z).astype(np.int64)
    _, inv = np.unique(fi, return_inverse=True)
    crops = PALETTE["crops"]
    return crops[rng.integers(0, len(crops), inv.max() + 1)[inv.reshape(np.shape(X))]]


def _mix(col, c2, t):
    """Смешать цвета: доля c2 — t, ограниченная [0, 1] (сумма масок не уводит цвет за палитру)."""
    return _blend(col, c2, np.clip(t, 0.0, 1.0))


def _jitter(col, rng, k=0.10):
    return col * (1 - k / 2 + k * rng.random(col.shape[:-1] + (1,)))


# ==========================================================================
# Долина реки
# ==========================================================================

def valley_axis(Z):
    """Ось долины: X(Z), м (плавные изгибы)."""
    return -1000.0 + 450.0 * np.sin(np.asarray(Z, float) / 1300.0 + 0.7)


def valley_river(Z):
    """Ось реки: меандры внутри долины."""
    Z = np.asarray(Z, float)
    return valley_axis(Z) + 180.0 * np.sin(Z / 520.0 + 1.3)


def _road_along(f, dx, z0, z1, step=200.0):
    zs = np.arange(z0, z1 + 1, step)
    return [(float(f(z) + dx), float(z)) for z in zs]


class ValleyMap(ProcMap):
    """Долина реки (описание — в шапке модуля)."""

    name, title = "valley", "долина реки 10×10 км"
    SEED = 3
    W_FLOOR = 1050.0                   # полуширина дна долины, м
    TRIBS = [((float(valley_axis(-2500.0)) + 1050.0 - 150.0, -2500.0), (1.0, -0.15), 3800.0),
             ((float(valley_axis(2800.0)) - 1050.0 + 150.0, 2800.0), (-1.0, 0.35), 3200.0)]
    # боковые долины: (устье на краю дна, направление, длина, м)

    def __init__(self, seed=7):
        super().__init__(seed)
        main = _road_along(valley_axis, 380.0, -5000, 5000)
        (me, de, Le), (mw, dw, Lw) = [(m, np.array(d) / np.hypot(*d), L) for m, d, L in self.TRIBS]
        east = [(float(valley_axis(me[1]) + 380.0), me[1])] +                [tuple(map(float, np.array(me) + de * s)) for s in np.arange(0, Le + 1, 400)]
        west = [(float(valley_axis(mw[1]) + 380.0), mw[1])] +                [tuple(map(float, np.array(mw) + dw * s)) for s in np.arange(0, Lw + 1, 400)]
        side = tuple(map(float, np.array(me) + de * 2100))
        apron = [(float(valley_axis(-250.0) + 380.0), -250.0), (150.0, -250.0)]
        self.ROADS = [(main, 4.0, 0.4), (west, 3.5, 0.4), (east, 3.5, 0.4), (apron, 3.5, 0.4)]
        self.VILLAGES = [
            dict(c=(420.0, 150.0), r=330.0, step=38.0, church=True),          # деревня у аэродрома
            dict(c=(float(valley_river(-3800) + 380), -3800.0), r=280.0),
            dict(c=(float(valley_river(-1600) - 360), -1600.0), r=230.0),
            dict(c=(float(valley_river(2800) - 420), 2900.0), r=300.0, church=True),
            dict(c=(float(valley_river(4300) + 380), 4300.0), r=260.0),
            dict(c=side, r=200.0),                                            # в боковой долине
        ]
        self._farms = None

    def relief(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        dv = np.abs(X - valley_axis(Z))
        floor = 2.5 + 1.2 * fbm(X, Z, 500, 3, self.SEED)
        Xw = X + 400 * fbm(X, Z, 3000, 2, self.SEED + 1)
        hills = (240 + 260 * fbm(Xw, Z, 2000, 5, self.SEED + 2)
                 + 170 * ridged(Xw, Z, 1300, 5, self.SEED + 3))              # ≈ 60…600 м
        hills = np.maximum(hills, 60.0)
        W = self.W_FLOOR + 200 * fbm(X, Z, 1500, 2, self.SEED + 4)
        side = floor + 0.45 * 40 * np.log1p(np.exp((dv - W) / 40))          # корыто: дно + склон 0.45
        H = smin(hills, side, 25.0)
        for (mx, mz), (ux, uz), L in self.TRIBS:                             # боковые долины
            p1 = (mx + ux * L / np.hypot(ux, uz), mz + uz * L / np.hypot(ux, uz))
            s, d, _ = _seg_along(X, Z, (mx, mz), p1)
            notch = 3.0 + 0.035 * np.clip(s, 0, None) + 0.30 * np.maximum(d - 90, 0)
            notch = np.where((s > -300) & (s < L), notch, 1e4)
            H = smin(H, notch, 20.0)
        dr = np.abs(X - valley_river(Z))                                     # русло: ширина 90…150 м
        n = noise(X, Z, 900, self.SEED + 8)
        c = 1 - _smooth(55 + 20 * n, 170 + 30 * n, dr)
        return H * (1 - c) + c * (-5.0)

    def farms(self):
        """Хутора на холмах: точки на пологих местах выше дна долины."""
        if self._farms is None:
            rng = np.random.default_rng(self.SEED + 9)
            X, Z = rng.uniform(-4500, 4500, 6000), rng.uniform(-4500, 4500, 6000)
            Y = self.ground(X, Z)
            ok = (Y > 60) & (self.slope(X, Z) < 0.10) & self._house_ok(X, Z)
            idx = np.flatnonzero(ok)
            pick = []
            for i in idx:                                                    # не ближе 500 м друг к другу
                if all(np.hypot(X[i] - X[j], Z[i] - Z[j]) > 500 for j in pick):
                    pick.append(i)
                if len(pick) >= 28:
                    break
            self._farms = np.stack([X[pick], Z[pick]], -1)
        return self._farms

    def colors(self, X, Z, Y, slope, rng):
        S = self.SEED
        col = _jitter(_patch_colors(X, Z, 210, 150, rng), rng)
        dv = np.abs(X - valley_axis(Z))
        dr = np.abs(X - valley_river(Z))
        fields = ((Y < 14) | (slope < 0.10) & (fbm(X, Z, 900, 3, S + 7) > 0.05)) & (dr > 330)
        col[~fields] = _jitter(np.broadcast_to(PALETTE["meadow"], col[~fields].shape).copy(), rng)
        # виноградники: южные склоны (рельеф растёт к северу), террасы — полосы по горизонталям
        n = fbm(X, Z, 700, 3, S + 5)
        gz = np.gradient(Y, axis=1) / self.CELL
        vine = (gz > 0.07) & (slope > 0.10) & (slope < 0.45) & (Y > 8) & (Y < 170) \
            & (dv < self.W_FLOOR + 800) & (n > -0.1)
        stripe = (np.floor(Y / 5.0) % 2 == 0)
        vcol = np.where(stripe[..., None], [0.42, 0.52, 0.22], [0.55, 0.50, 0.30])
        col[vine] = vcol[vine]
        # лес: крутые склоны и вершины; лиственный ниже, хвойный выше
        f = fbm(X, Z, 600, 4, S + 6)
        forest = ((slope > 0.30) | (Y > 190) & (f > -0.15) | (f > 0.30) & (Y > 30)) & ~vine
        dark = np.where((Y > 230)[..., None], PALETTE["forest"], [0.22, 0.40, 0.17])
        col[forest] = _jitter(dark[forest], rng, 0.08)
        # хутора — луг вокруг
        for fx, fz in self.farms():
            m = np.hypot(X - fx, Z - fz) < 110
            col[m] = PALETTE["meadow"] * 1.05
        col = _mix(col, PALETTE["rock"], _smooth(0.65, 0.9, slope))
        col = _mix(col, PALETTE["bank"], _smooth(0.0, -2.5, Y) * (dr < 260))
        return col

    def tree_masks(self, X, Z, Y, sl, rng):
        f = fbm(X, Z, 600, 4, self.SEED + 6)
        forest = (sl > 0.30) | (Y > 190) & (f > -0.15) | (f > 0.30) & (Y > 30)
        dr = np.abs(X - valley_river(Z))
        con = forest & (Y > 150) & (rng.random(X.size) < 0.55)
        brd = forest & (Y <= 150) & (rng.random(X.size) < 0.55)
        brd |= (dr > 75) & (dr < 140) & (noise(X, Z, 300, 77) > -0.2)       # ивы по берегам
        brd |= (rng.random(X.size) < 0.004) & (sl < 0.2)                    # одиночные деревья
        for fx, fz in self.farms():                                          # сады у хуторов
            brd |= (np.hypot(X - fx, Z - fz) < 90) & (rng.random(X.size) < 0.3)
        return con, brd

    def extras(self):
        rng = np.random.default_rng(self.SEED + 11)
        parts = []
        for fx, fz in self.farms():                                          # хутор: дом + сарай
            y0 = float(self.ground(fx, fz)) - 0.4
            yaw = rng.uniform(0, np.pi)
            parts.append(_house(fx, fz, 8, 11, 4.5, 3, yaw, WALLS["village"][rng.integers(4)],
                                ROOFS["village"][rng.integers(4)], y0))
            bx, bz = fx + 22 * np.cos(yaw), fz - 22 * np.sin(yaw)
            parts.append(_house(bx, bz, 10, 16, 4, 3.5, yaw, (0.55, 0.40, 0.28), (0.40, 0.36, 0.32),
                                float(self.ground(bx, bz)) - 0.6))
        parts += self._castle()
        return parts

    def _castle(self):
        """Замок на отроге: самая высокая точка западного склона у излучины."""
        g = np.linspace(-450, 450, 31)
        GX, GZ = np.meshgrid(g, g)
        X0, Z0 = -2600.0 + GX, 1600.0 + GZ
        Y = self.ground(X0, Z0)
        k = np.unravel_index(np.argmax(Y), Y.shape)
        cx, cz = float(X0[k]), float(Z0[k])
        y0 = float(np.min(self.ground(cx + np.array([-30, 30, -30, 30]), cz + np.array([-30, -30, 30, 30])))) - 1
        stone, roof = (0.78, 0.74, 0.66), (0.45, 0.30, 0.25)
        parts = [_box(cx, cz, 9, 9, 28, 0.2, stone, y0), _spire(cx, cz, 6.5, 9, roof, y0 + 28)]
        for a in np.arange(4) * np.pi / 2 + 0.2:                             # стены и угловые башни
            px, pz = cx + 32 * np.cos(a), cz + 32 * np.sin(a)
            qx, qz = cx + 32 * np.cos(a + np.pi / 2), cz + 32 * np.sin(a + np.pi / 2)
            mx, mz = (px + qx) / 2, (pz + qz) / 2
            L = np.hypot(qx - px, qz - pz)
            parts.append(_box(mx, mz, 2.5, L, 9, np.arctan2(qx - px, qz - pz), stone, y0))
            parts.append(_box(px, pz, 7, 7, 14, 0.2, stone, y0))
            parts.append(_spire(px, pz, 5, 6, roof, y0 + 14))
        return parts


# ==========================================================================
# Альпы
# ==========================================================================

class AlpsMap(ProcMap):
    """Горы с котловиной (описание — в шапке модуля)."""

    name, title = "alps", "горы 10×10 км"
    SEED = 5
    WATER_Y = -2.0
    BASIN = (300.0, 1100.0, 1600.0, 2000.0)          # эллипс котловины: центр X, Z, полуоси
    LAKE = (500.0, 2550.0, 1350.0, 560.0, 0.12)       # озеро: центр, полуоси, поворот (рад)
    CORRIDORS = [([(0.0, -300.0), (350.0, -2000.0), (-350.0, -3500.0), (100.0, -5000.0)], 480.0),
                 ([(-1100.0, 1700.0), (-2500.0, 2300.0), (-3800.0, 2000.0), (-5000.0, 2300.0)], 330.0)]
    TOWN = (900.0, 1250.0)
    TREES_MAX = 20000
    broadleaf = np.array([0.22, 0.40, 0.16])

    def __init__(self, seed=7):
        super().__init__(seed)
        tx, tz = self.TOWN
        south = self.CORRIDORS[0][0]
        west = self.CORRIDORS[1][0]
        lake_rd = self._lake_shore_road()
        self.ROADS = [([(150.0, -250.0), (450.0, -150.0), (tx - 200, tz - 350), (tx, tz)], 4.0, 0.4),
                      ([(0.0, -500.0)] + south[1:], 4.0, 0.4),
                      ([(tx - 300, tz + 150), (-500.0, 1500.0)] + west, 3.5, 0.4),
                      (lake_rd, 3.5, 0.4)]
        self.VILLAGES = [dict(c=self.TOWN, r=520.0, step=30.0, style="town", church=True, yaw=0.25),
                         dict(c=(-1900.0, 2150.0), r=200.0),
                         dict(c=(250.0, -2600.0), r=240.0, church=True)]
        self._chalets = None

    def _lake_q(self, X, Z):
        cx, cz, a, b, r = self.LAKE
        u = (X - cx) * np.cos(r) - (Z - cz) * np.sin(r)
        w = (X - cx) * np.sin(r) + (Z - cz) * np.cos(r)
        return np.sqrt((u / a) ** 2 + (w / b) ** 2)

    def _lake_shore_road(self):
        cx, cz, a, b, r = self.LAKE
        t = np.linspace(np.pi * 1.05, np.pi * 1.95, 14)
        u, w = (a + 120) * np.cos(t), (b + 110) * np.sin(t)
        X = cx + u * np.cos(r) + w * np.sin(r)
        Z = cz - u * np.sin(r) + w * np.cos(r)
        return [(float(x), float(z)) for x, z in zip(X, Z)]

    def basin_dist(self, X, Z):
        """Расстояние от дна котловины и долин-выходов, м (< 0 — внутри)."""
        cx, cz, a, b = self.BASIN
        q = np.sqrt(((X - cx) / a) ** 2 + ((Z - cz) / b) ** 2)
        d = (q - 1) * min(a, b)
        for pts, hw in self.CORRIDORS:
            d = np.minimum(d, _seg_dist(X, Z, pts) - hw)
        return d

    def relief(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        S = self.SEED
        db = self.basin_dist(X, Z) + 250 * fbm(X, Z, 900, 2, S)
        edge = np.minimum.reduce([X - self.map_x[0], self.map_x[1] - X, Z - self.map_z[0], self.map_z[1] - Z])
        Xw = X + 600 * fbm(X, Z, 2500, 3, S + 1)
        Zw = Z + 600 * fbm(X, Z, 2500, 3, S + 2)
        A = 2300 * _smooth(0, 2300, db) * (0.45 + 0.55 * _smooth(0, 2600, edge))
        M = A * (0.30 + 0.85 * ridged(Xw, Zw, 3200, 6, S + 3)) + 140 * _smooth(-100, 500, db) * (1 + fbm(X, Z, 700, 3, S + 4))
        floor = 2.0 + 2.0 * fbm(X, Z, 600, 3, S + 5) + 0.012 * np.clip(-Z - 1500, 0, None)
        floor += 0.03 * np.clip(-X - 1500, 0, None) * (Z > 1000)                 # западная долина — к перевалу
        H = floor + M
        c = 1 - _smooth(0.86, 1.0, self._lake_q(X, Z))                         # озеро
        return H * (1 - c) + c * (-14.0)

    def chalets(self):
        if self._chalets is None:
            rng = np.random.default_rng(self.SEED + 9)
            X, Z = rng.uniform(-4000, 4000, 20000), rng.uniform(-4000, 4000, 20000)
            Y = self.ground(X, Z)
            ok = (Y > 60) & (Y < 1000) & (self.slope(X, Z) < 0.32) & self._house_ok(X, Z)
            idx = np.flatnonzero(ok)[:90]
            self._chalets = np.stack([X[idx], Z[idx]], -1)
        return self._chalets

    def colors(self, X, Z, Y, slope, rng):
        S = self.SEED
        n = fbm(X, Z, 300, 3, S + 6)
        col = np.empty(X.shape + (3,))
        col[:] = PALETTE["meadow"]
        db = self.basin_dist(X, Z)
        fields = (db < -150) & (Y < 20)
        pc = _patch_colors(X, Z, 160, 110, rng, 0.25)
        col[fields] = pc[fields]
        col = _jitter(col, rng)
        forest = (Y > 25) & (Y < 1150 + 120 * n) & (slope < 0.95) & (fbm(X, Z, 500, 4, S + 7) > -0.25)
        col[forest] = _jitter(np.broadcast_to(PALETTE["forest"], col[forest].shape).copy(), rng, 0.08)
        alp = (Y > 1050 + 150 * n) & (slope < 0.7)
        col = _mix(col, [0.52, 0.62, 0.32], alp * _smooth(1050, 1250, Y + 150 * n))
        col = _mix(col, PALETTE["rock"] * 1.05, _smooth(0.75, 1.05, slope) + _smooth(1450, 1650, Y + 100 * n))
        snow = _smooth(1500, 1600, Y + 140 * n) * (1 - _smooth(1.1, 1.4, slope))
        col = _mix(col, [0.95, 0.96, 0.98], np.clip(snow, 0, 1))
        col = _mix(col, PALETTE["bank"], _smooth(0.0, -2.5, Y) * (self._lake_q(X, Z) < 1.15))
        for cx, cz in self.chalets():
            m = np.hypot(X - cx, Z - cz) < 70
            col[m] = [0.50, 0.66, 0.30]
        return col

    def tree_masks(self, X, Z, Y, sl, rng):
        S = self.SEED
        n = fbm(X, Z, 300, 3, S + 6)
        forest = (Y > 25) & (Y < 1150 + 120 * n) & (sl < 0.95) & (fbm(X, Z, 500, 4, S + 7) > -0.25)
        con = forest & (rng.random(X.size) < 0.6)
        brd = forest & (Y < 500) & (rng.random(X.size) < 0.25)
        db = self.basin_dist(X, Z)
        brd |= (db < -100) & (Y < 20) & (rng.random(X.size) < 0.004)
        return con, brd

    def extras(self):
        rng = np.random.default_rng(self.SEED + 11)
        parts = []
        for cx, cz in self.chalets():
            y0 = float(np.min(self.ground(cx + np.array([-7, 7, -7, 7]), cz + np.array([-7, -7, 7, 7])))) - 0.5
            parts.append(_house(cx, cz, rng.uniform(9, 12), rng.uniform(11, 14), rng.uniform(5, 7), 3,
                                rng.uniform(0, np.pi), WALLS["chalet"][rng.integers(3)],
                                ROOFS["chalet"][rng.integers(3)], y0))
        tx, tz = self.TOWN                                                     # отели у озера
        for k in range(7):
            a = np.pi * (1.2 + 0.09 * k)
            cx0, cz0, la, lb, r = self.LAKE
            hx = cx0 + (la + 200) * np.cos(a) * np.cos(r) + (lb + 180) * np.sin(a) * np.sin(r)
            hz = cz0 - (la + 200) * np.cos(a) * np.sin(r) + (lb + 180) * np.sin(a) * np.cos(r)
            if not self._house_ok(np.array([hx]), np.array([hz]))[0]:
                continue
            parts.append(_house(hx, hz, 16, 34, rng.uniform(14, 22), 5, a, (0.96, 0.95, 0.90),
                                (0.30, 0.31, 0.34), float(self.ground(hx, hz)) - 0.5))
        return parts


# ==========================================================================
# Болото
# ==========================================================================

def marsh_edge(Z):
    """Граница болота и равнины: X(Z), м (западнее — болото)."""
    Z = np.asarray(Z, float)
    return -1000.0 + 300.0 * np.sin(Z / 1100.0 + 0.4) + 120.0 * np.sin(Z / 400.0)


def marsh_river(Z):
    Z = np.asarray(Z, float)
    return -2600.0 + 320.0 * np.sin(Z / 650.0) + 140.0 * np.sin(Z / 240.0 + 1.0)


class MarshMap(ProcMap):
    """Болото и равнина (описание — в шапке модуля)."""

    name, title = "marsh", "болото и равнина 10×10 км"
    SEED = 8
    WATER_Y = -1.0
    SPLIT_Z = 1600.0                  # здесь река распадается на рукава
    ISLAND = (-3000.0, -2000.0)
    OXBOWS = [(-2150.0, 3300.0, 190.0, 0.4, 4.2), (-3300.0, 4300.0, 160.0, 2.0, 5.8),
              (-1900.0, -900.0, 220.0, 3.6, 7.3), (-3900.0, 400.0, 170.0, 0.0, 3.6)]   # центр, R, дуга
    broadleaf = np.array([0.38, 0.52, 0.24])           # берёза, ольха — светлее
    SHORE = 3.0

    def __init__(self, seed=7):
        super().__init__(seed)
        ix, iz = self.ISLAND
        gat = [(-60.0, -380.0), (-700.0, -700.0), (-1500.0, -1100.0), (-2300.0, -1600.0), (ix + 150, iz + 60)]
        main = [(1400.0 + 200 * np.sin(z / 1300), float(z)) for z in np.arange(-5000, 5001, 250)]
        main = [(float(x), z) for x, z in main]
        self.ROADS = [(gat, 3.0, 1.4),
                      ([(150.0, -250.0), (800.0, -300.0), (float(main[19][0]), -300.0)], 3.5, 0.4),
                      (main, 4.0, 0.4),
                      ([(float(main[26][0]), 1500.0), (2400.0, 1600.0), (3800.0, 2400.0), (5000.0, 2600.0)], 3.5, 0.4),
                      ([(float(main[10][0]), -2500.0), (2500.0, -2600.0), (3600.0, -3200.0)], 3.5, 0.4)]
        self.VILLAGES = [dict(c=(2400.0, 1600.0), r=320.0, church=True),
                         dict(c=(2500.0, -2650.0), r=280.0),
                         dict(c=(3900.0, 3700.0), r=230.0),
                         dict(c=(700.0, 3200.0), r=180.0),
                         dict(c=(ix, iz), r=170.0, step=36.0)]

    def channels(self, X, Z):
        """Расстояние до ближайшего русла (основное + рукава ниже SPLIT_Z), м; полуширина 30 м."""
        xr = marsh_river(Z)
        d = np.abs(X - xr)
        spread = np.clip(self.SPLIT_Z - Z, 0, None)
        for k, (sgn, rate, cap) in enumerate(((-1, 0.35, 1300.0), (1, 0.28, 800.0), (1, 0.55, 1300.0))):
            dx = sgn * np.minimum(rate * spread, cap)                        # рукав отходит от реки
            wig = 90 * np.sin(Z / (210 + 40 * k) + k)
            dk = np.abs(X - (xr + dx + wig * (spread > 0)))
            d = np.minimum(d, np.where(Z < self.SPLIT_Z, dk + 8, np.inf))
        return d

    def _oxbow_dist(self, X, Z):
        d = np.full(np.shape(X), np.inf)
        for cx, cz, R, a0, a1 in self.OXBOWS:
            a = np.mod(np.arctan2(Z - cz, X - cx) - a0, 2 * np.pi)
            on = a < (a1 - a0)
            d = np.minimum(d, np.where(on, np.abs(np.hypot(X - cx, Z - cz) - R), np.inf))
        return d

    def marsh_w(self, X, Z):
        """Вес болота: 1 — болото, 0 — равнина."""
        return 1 - _smooth(-280, 280, X - marsh_edge(Z))

    def relief(self, X, Z):
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        S = self.SEED
        plain = (3.5 + 4.0 * fbm(X, Z, 1600, 4, S) + 1.5 * fbm(X, Z, 350, 2, S + 1)
                 + 22.0 * np.exp(-((X - 3800) ** 2 + (Z + 600) ** 2) / (2 * 700 ** 2)))      # пологий холм
        isl = 3.0 * _smooth(0.22, 0.42, fbm(X, Z, 800, 3, S + 2))
        ix, iz = self.ISLAND
        isl = np.maximum(isl, 3.2 * (1 - _smooth(200, 420, np.hypot(X - ix, Z - iz))))
        bog = -0.75 + 0.9 * fbm(X, Z, 240, 4, S + 3) + isl
        m = self.marsh_w(X, Z)
        H = bog * m + plain * (1 - m)
        c = 1 - _smooth(25, 80, self.channels(X, Z))                         # русла и старицы
        c = np.maximum(c, 1 - _smooth(15, 55, self._oxbow_dist(X, Z)))
        return H * (1 - c) + c * (-3.0)

    def colors(self, X, Z, Y, slope, rng):
        S = self.SEED
        col = _patch_colors(X, Z, 240, 170, rng, 0.1)
        col = _jitter(col, rng)
        m = self.marsh_w(X, Z)
        f = fbm(X, Z, 700, 4, S + 4)
        meadow = (f < -0.2) | (m > 0.05)
        col[meadow] = _jitter(np.broadcast_to(PALETTE["meadow"], col[meadow].shape).copy(), rng)
        col[(f > 0.28) & (m < 0.3)] = PALETTE["forest"]
        hw = Y - self.WATER_Y                                                # высота над водой
        n = fbm(X, Z, 150, 3, S + 5)
        reed = np.array([0.62, 0.60, 0.38])
        sedge = np.array([0.50, 0.56, 0.30])
        moss = np.array([0.45, 0.52, 0.28])
        peat = np.array([0.40, 0.36, 0.25])
        sh = self.shore                                                      # пороги — до отступа
        bog = np.where((hw < sh(0.35 + 0.2 * n))[..., None], reed,
                       np.where((hw < sh(0.9 + 0.3 * n))[..., None], sedge, moss))
        bog = np.where(((n > 0.35) & (hw < sh(1.2)))[..., None], peat, bog)
        bog = np.where((hw > sh(2.2))[..., None], [0.24, 0.40, 0.18], bog)   # острова — лес
        bog = _jitter(bog, rng, 0.12)
        col = _mix(col, bog, _smooth(0.3, 0.7, m))
        return col

    def tree_masks(self, X, Z, Y, sl, rng):
        S = self.SEED
        m = self.marsh_w(X, Z)
        hw = Y - self.WATER_Y
        f = fbm(X, Z, 700, 4, S + 4)
        con = (f > 0.28) & (m < 0.3) & (rng.random(X.size) < 0.6)
        isl = hw > self.shore(2.2)
        con |= (m > 0.5) & isl & (rng.random(X.size) < 0.25)
        brd = (m > 0.5) & isl & (rng.random(X.size) < 0.35)
        brd |= (m > 0.5) & (hw > self.shore(0.6)) & (rng.random(X.size) < 0.012)        # редкие берёзы по болоту
        brd |= (m < 0.3) & (rng.random(X.size) < 0.003)
        for pts, hwid, lift in self.ROADS:                                   # обсадка дорог
            if lift < 1:
                d = _seg_dist(X, Z, pts, near=30)
                brd |= (d > 10) & (d < 16) & (rng.random(X.size) < 0.35) & (m < 0.3)
        return con, brd

    def extras(self):
        """Стога и сараи на лугах у болота."""
        rng = np.random.default_rng(self.SEED + 11)
        parts = []
        Z = rng.uniform(-4500, 4500, 3000)
        X = marsh_edge(Z) + rng.uniform(150, 900, Z.size)                   # луга вдоль края болота
        ok = self._house_ok(X, Z)
        for x, z in list(zip(X[ok], Z[ok]))[:120]:
            y0 = float(self.ground(x, z)) - 0.3
            parts.append(_spire(x, z, 2.6, 4.5, (0.70, 0.60, 0.32), y0))
        return parts


NEW_MAPS = {"valley": ValleyMap, "alps": AlpsMap, "marsh": MarshMap}

from viz.world3d import MAPS         # noqa: E402 — регистрация в общем списке карт
MAPS.update(NEW_MAPS)

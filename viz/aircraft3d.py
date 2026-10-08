# -*- coding: utf-8 -*-
"""
3D-модели ЛА для viz/viewer3d.py (Ursina).

Модели строятся процедурно из сечений (лофт) — файлов-ассетов нет, «вес» модели —
этот код (≈ 2 тыс. треугольников). Плоское low-poly затенение: у каждого
треугольника своя нормаль и цвет.

Реестр MODELS: имя типа ЛА (AircraftParams.name, sim/config.AIRCRAFT_TYPES) ->
класс модели. Неизвестное имя (старые логи, новые типы) -> DEFAULT_MODEL.

Локальные оси модели: +Z — нос, +X — правое крыло, +Y — верх.
Рули отклоняются по логу (знаки как в sim/config.py):
  δe > 0 — задняя кромка руля высоты вниз;
  δa > 0 — правый элерон вверх, левый вниз (правый крен);
  δr > 0 — задняя кромка руля направления влево.
"""

import numpy as np
from ursina import Entity, Mesh, Vec3, Color
from ursina.shaders.lit_with_shadows_shader import lit_with_shadows_shader


# ===========================================================================
# Построение сетки
# ===========================================================================

class MeshBuilder:
    """
    Набор треугольников с плоскими нормалями. Каждый лофт ориентируется целиком
    по знаку объёма (нормали наружу) — надёжно и для тонких профилей.
    """

    FRONT_CCW = False      # порядок обхода лицевой грани в Ursina (проверено скриншотом)

    def __init__(self):
        self.v, self.n, self.col = [], [], []

    def _emit(self, tris):
        for a, b, c, color in tris:
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue                               # вырожденный (обрезка профиля)
            self.v += [tuple(a), tuple(b), tuple(c)] if self.FRONT_CCW else \
                      [tuple(a), tuple(c), tuple(b)]
            self.n += [tuple(nrm / ln)] * 3
            self.col += [color] * 3

    def loft(self, sections, color_fn, caps=True):
        """
        Поверхность через замкнутые сечения одинаковой длины (список массивов (M, 3)).
        color_fn(i, j) — цвет полосы между сечениями i, i+1 и точками j, j+1.
        """
        S = [np.asarray(s, float) for s in sections]
        m = len(S[0])
        tris = []
        for i in range(len(S) - 1):
            for j in range(m):
                k, col = (j + 1) % m, color_fn(i, j)
                tris += [(S[i][j], S[i][k], S[i + 1][k], col),
                         (S[i][j], S[i + 1][k], S[i + 1][j], col)]
        if caps:
            c0, c1 = S[0].mean(0), S[-1].mean(0)
            for j in range(m):
                k = (j + 1) % m
                tris += [(c0, S[0][k], S[0][j], color_fn(0, j)),
                         (c1, S[-1][j], S[-1][k], color_fn(len(S) - 2, j))]
        ref = np.mean([t[0] for t in tris], axis=0)
        vol = sum(np.dot(np.cross(b - a, c - a), a - ref) for a, b, c, _ in tris)
        if vol < 0:
            tris = [(a, c, b, col) for a, b, c, col in tris]
        self._emit(tris)

    def entity(self, parent, **kw):
        mesh = Mesh(vertices=self.v, normals=self.n, colors=self.col)
        return Entity(parent=parent, model=mesh, shader=lit_with_shadows_shader, **kw)


def naca(t: float, camber: float = 0.0, n: int = 7):
    """
    Профиль NACA 4-digit: замкнутый контур (s, y) в долях хорды, s = 0 — носок.
    Верх — от задней кромки к носку, низ — от носка к задней кромке.
    """
    s = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))           # сгущение к носку
    yt = 5 * t * (0.2969 * np.sqrt(s) - 0.126 * s - 0.3516 * s**2
                  + 0.2843 * s**3 - 0.1036 * s**4)
    p = 0.4
    yc = np.where(s < p, camber / p**2 * (2 * p * s - s**2),
                  camber / (1 - p)**2 * (1 - 2 * p + 2 * p * s - s**2))
    up = np.stack([s, yc + yt], 1)[::-1]
    lo = np.stack([s, yc - yt], 1)[1:-1]
    return np.vstack([up, lo])


def surface_section(prof, s0, s1, le, chord, at, normal_axis):
    """
    Сечение несущей поверхности: часть профиля между долями хорды s0..s1
    (обрезка по линии шарнира — вертикальный срез).
    le — 3D-точка носка, хорда направлена по −Z; normal_axis — ось толщины
    (1 — Y для крыла и ГО, 0 — X для киля); at — координата сечения по размаху.
    """
    s = np.clip(prof[:, 0], s0, s1)
    sec = np.zeros((len(prof), 3))
    sec[:, 2] = le[2] - s * chord
    sec[:, normal_axis] = le[normal_axis] + prof[:, 1] * chord
    sec[:, 1 - normal_axis] = at
    return sec


# ===========================================================================
# Cessna 172 (low-poly)
# ===========================================================================

class CessnaModel:
    """
    Лёгкий одномоторный высокоплан с подкосами (по мотивам Cessna 172) —
    та же схема, что у флайт-модели FPV (высокоплан, классическое оперение).
    Строится в метрах прототипа (размах 11 м) и масштабируется до размаха b
    из параметров ЛА; это внешний вид, геометрия флайт-модели — в sim/config.py.
    """

    SPAN = 11.0
    EYE = (0.0, 0.55, 0.35)          # глаз пилота, м прототипа (камера «из кабины»)

    WHITE = Color(0.96, 0.96, 0.97, 1)
    GLASS = Color(0.16, 0.22, 0.30, 1)
    BLUE  = Color(0.10, 0.22, 0.52, 1)
    RED   = Color(0.80, 0.12, 0.10, 1)
    DARK  = Color(0.12, 0.12, 0.13, 1)
    METAL = Color(0.62, 0.64, 0.68, 1)

    # профиль фюзеляжа (правая половина снизу вверх): (доля полуширины, доля высоты)
    PROFILE = [(0, 0), (0.55, 0.02), (0.88, 0.12), (1.0, 0.30), (1.0, 0.42),
               (1.0, 0.50), (1.0, 0.60), (0.92, 0.80), (0.70, 0.95), (0.35, 1.0), (0, 1.0)]
    # станции фюзеляжа: z, полуширина, низ, верх (м прототипа)
    STATIONS = [(2.78, 0.30, -0.22, 0.24), (2.45, 0.46, -0.45, 0.33),
                (1.95, 0.52, -0.56, 0.38), (1.40, 0.55, -0.60, 0.42),
                (0.60, 0.56, -0.62, 0.80), (-0.25, 0.56, -0.62, 0.81),
                (-0.45, 0.56, -0.62, 0.81), (-1.30, 0.52, -0.56, 0.80),
                (-2.40, 0.36, -0.36, 0.60), (-4.00, 0.17, -0.06, 0.44),
                (-5.25, 0.06, 0.16, 0.38)]

    def __init__(self, b: float, c: float = None):
        self.root = Entity()
        self.body = Entity(parent=self.root, scale=b / self.SPAN)
        self.eye = np.array(self.EYE) * b / self.SPAN
        self._fuselage()
        self._wing()
        self._tail()
        self._gear()
        self._prop()

    # --- фюзеляж ---------------------------------------------------------
    def _fuselage(self):
        P = self.PROFILE
        ring = P + [(-u, v) for u, v in reversed(P[1:-1])]         # замкнутый контур
        nb = len(P) - 1                                             # полос на половине

        def band(j):                       # номер полосы на половине (симметрично)
            return j if j < nb else 2 * nb - 1 - j

        secs = [[(w * u, lo + (hi - lo) * v, z) for u, v in ring]
                for z, w, lo, hi in self.STATIONS]

        def color(i, j):
            z = 0.5 * (self.STATIONS[i][0] + self.STATIONS[i + 1][0])
            k = band(j)
            if k == 4:
                return self.BLUE                                    # полосы по борту
            if k == 3:
                return self.RED
            if 0.6 < z < 1.4 and k >= 6:
                return self.GLASS                                   # лобовое стекло
            if -1.3 < z < 0.6 and k == 6 and not -0.45 < z < -0.25:
                return self.GLASS                                   # боковые окна, стойка
            if -2.4 < z < -1.3 and k >= 6:
                return self.GLASS                                   # заднее окно
            return self.WHITE

        m = MeshBuilder()
        m.loft(secs, color)
        ang = np.linspace(0, 2 * np.pi, 10)[:-1]                    # кок винта
        m.loft([[(r * np.cos(a), r * np.sin(a), z) for a in ang]
                for z, r in ((2.78, 0.22), (2.95, 0.19), (3.10, 0.10), (3.18, 0.001))],
               lambda i, j: self.RED, caps=False)    # без торца — не виден из кабины
        self.fuselage = m.entity(self.body)

    # --- крыло, элероны, подкосы ----------------------------------------
    def _wing(self):
        prof = naca(0.12, camber=0.02)
        z_le, z_te, z_hinge = 0.40, -1.23, -0.88
        x_kink, x_ail0, x_ail1, x_tip = 2.5, 2.9, 5.0, 5.5
        y0, dih = 0.80, np.tan(np.radians(1.7))
        white = lambda i, j: self.WHITE

        def le(x):                         # сужение только по передней кромке
            return z_le - max(0.0, x - x_kink) / (x_tip - x_kink) * 0.5

        def sec(x, s0=0.0, s1=1.0):
            ax = abs(x)
            return surface_section(prof, s0, s1, (0, y0 + ax * dih, le(ax)),
                                   le(ax) - z_te, x, 1)

        def hinge_s(x):
            return (le(x) - z_hinge) / (le(x) - z_te)

        m = MeshBuilder()
        for sg in (-1, 1):
            m.loft([sec(sg * x) for x in (0.0, x_kink, x_ail0)], white)
            m.loft([sec(sg * x, 0, hinge_s(x)) for x in (x_ail0, x_ail1)], white)
            m.loft([sec(sg * x) for x in (x_ail1, x_tip - 0.15)], white)
            tip = sec(sg * x_tip)                                   # законцовка
            tip[:, 1] = tip[:, 1].mean() + 0.3 * (tip[:, 1] - tip[:, 1].mean())
            m.loft([sec(sg * (x_tip - 0.15)), tip], lambda i, j: self.RED)
            # подкос: от низа фюзеляжа к крылу, сечение — вытянутый ромб
            rhomb = np.array([(0, 0, 0.09), (0, 0.025, 0), (0, 0, -0.09), (0, -0.025, 0)])
            m.loft([np.array(p) + rhomb for p in ((sg * 0.50, -0.42, -0.15),
                                                  (sg * 2.60, y0 + 2.6 * dih - 0.07, -0.15))],
                   white)
        self.wing = m.entity(self.body)

        self.ail = []
        hinge = np.array([0, y0 + 0.5 * (x_ail0 + x_ail1) * dih, z_hinge])
        for sg in (-1, 1):                 # элерон: шарнир — прямая z = z_hinge
            ma = MeshBuilder()
            ma.loft([sec(sg * x, hinge_s(x), 1.0) - hinge
                     for x in (x_ail0 + 0.02, x_ail1 - 0.02)], white)
            pv = Entity(parent=self.body, position=tuple(hinge))
            ma.entity(pv)
            self.ail.append((sg, pv))

    # --- оперение ---------------------------------------------------------
    def _tail(self):
        prof = naca(0.09)
        white = lambda i, j: self.WHITE
        yt, z_h, z_te, z_rh = 0.30, -4.85, -5.40, -5.00

        def stab_le(x):                    # передняя кромка ГО
            return -3.95 - 0.45 * x / 1.72

        m = MeshBuilder()
        for sg in (-1, 1):                 # ГО до шарнира РВ
            m.loft([surface_section(prof, 0, (stab_le(x) - z_h) / (stab_le(x) - z_te),
                                    (0, yt, stab_le(x)), stab_le(x) - z_te, sg * x, 1)
                    for x in (0.0, 1.72)], white)
        # киль до шарнира РН (сильная стреловидность) и форкиль
        for fin in (((0.35, -3.55, z_rh), (1.95, -4.65, z_rh)),        # (y, носок, конец)
                    ((0.50, -2.40, -3.60), (0.62, -3.30, -3.60))):
            m.loft([surface_section(prof, 0, 1, (0, y, zl), zl - zt, y, 0)
                    for y, zl, zt in fin], white)
        self.tail = m.entity(self.body)

        # руль высоты: две половины с вырезом под РН, шарнир — ось X
        self.elev = Entity(parent=self.body, position=(0, yt, z_h))
        me = MeshBuilder()
        for sg in (-1, 1):
            me.loft([surface_section(prof, (stab_le(x) - z_h) / (stab_le(x) - z_te), 1,
                                     (0, yt, stab_le(x)), stab_le(x) - z_te, sg * x, 1)
                     - np.array([0, yt, z_h]) for x in (0.14, 1.70)], white)
        me.entity(self.elev)

        # руль направления: шарнир — вертикаль z = z_rh
        self.rud = Entity(parent=self.body, position=(0, 0, z_rh))
        mr = MeshBuilder()
        secs = []
        for y, depth in ((0.05, 0.40), (1.40, 0.45), (2.05, 0.30)):
            ch = depth / 0.4                       # РН — задние 40 % профиля
            secs.append(surface_section(prof, 0.6, 1, (0, y, 0.6 * ch), ch, y, 0))
        mr.loft(secs, lambda i, j: self.BLUE if i == 0 else self.RED)
        mr.entity(self.rud)

    # --- шасси ------------------------------------------------------------
    def _gear(self):
        m = MeshBuilder()

        def leg(a, b, w=0.06):
            sq = np.array([(w, 0, 0), (0, 0, w), (-w, 0, 0), (0, 0, -w)])
            m.loft([np.array(p) + sq for p in (a, b)], lambda i, j: self.METAL)

        def wheel(x, y, z, r=0.24, w=0.15):
            ang = np.linspace(0, 2 * np.pi, 13)[:-1]
            m.loft([[(xx, y + r * np.sin(a), z + r * np.cos(a)) for a in ang]
                    for xx in (x - w / 2, x + w / 2)], lambda i, j: self.DARK)
            # обтекатель колеса (каплевидный): dz, полуширина, полувысота
            pr = [(0.42, 0.02, 0.03), (0.30, 0.12, 0.18), (0.0, 0.13, 0.24),
                  (-0.40, 0.10, 0.17), (-0.70, 0.01, 0.03)]
            ang8 = np.linspace(0, 2 * np.pi, 9)[:-1]
            m.loft([[(x + wx * np.cos(a), y + 0.06 + hy * np.sin(a), z + dz) for a in ang8]
                    for dz, wx, hy in pr],
                   lambda i, j: self.RED if j in (0, 7) else self.WHITE)

        for sg in (-1, 1):
            leg((sg * 0.45, -0.58, -0.35), (sg * 1.25, -1.30, -0.35))
            wheel(sg * 1.25, -1.40, -0.35)
        leg((0, -0.40, 2.10), (0, -1.30, 2.15), 0.045)
        wheel(0, -1.42, 2.15, r=0.20, w=0.12)
        self.gear = m.entity(self.body)

    # --- винт -------------------------------------------------------------
    def _prop(self):
        self.prop = Entity(parent=self.body, position=(0, 0, 3.02))
        m = MeshBuilder()
        for sg in (-1, 1):                 # две лопасти с круткой
            secs = []
            for r, ch, tw in ((0.12, 0.10, 35), (0.55, 0.15, 22), (0.95, 0.08, 14)):
                t = np.radians(tw)
                d = 0.5 * ch * np.array([np.cos(t), 0, np.sin(t)])
                e = 0.012 * np.array([-np.sin(t), 0, np.cos(t)])
                secs.append([(sg * p[0], sg * r, p[2]) for p in (d, e, -d, -e)])
            m.loft(secs, lambda i, j: self.DARK)
        m.entity(self.prop)

    # --- интерфейс для View3D ---------------------------------------------
    def pose(self, pos, nose, up, scale: float):
        self.root.position = Vec3(*pos)
        self.root.scale = scale
        p = Vec3(*pos)
        self.root.lookAt(p + Vec3(*nose), Vec3(*up))   # Panda3D: точно, с креном

    def surfaces(self, de, da, dr, thr, dt):
        self.elev.rotation_x = -np.degrees(de)
        for sg, pv in self.ail:
            pv.rotation_x = sg * np.degrees(da)
        self.rud.rotation_y = np.degrees(dr)
        self.prop.rotation_z += 3000.0 * thr * dt


MODELS = {"fpv": CessnaModel}
DEFAULT_MODEL = CessnaModel


def make_model(name: str, b: float, c: float):
    """3D-модель для типа ЛА name (неизвестный тип — модель по умолчанию)."""
    return MODELS.get(name, DEFAULT_MODEL)(b, c)

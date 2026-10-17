"""
Турбулентность по модели Драйдена (B&M, разд. 4.4): DrydenTurbulence(params)(t) -> (ug, vg, wg).

Белый шум единичной интенсивности подаётся на формирующие фильтры с передаточными
функциями (Va — скорость, на которую рассчитаны фильтры; L — масштаб, σ — СКО порыва):

    H_u(s) = σu·√(2Va/Lu) · 1 / (s + Va/Lu)
    H_v(s) = σv·√(3Va/Lv) · (s + Va/(√3·Lv)) / (s + Va/Lv)²
    H_w(s) = σw·√(3Va/Lw) · (s + Va/(√3·Lw)) / (s + Va/Lw)²

Дисперсия выхода каждого фильтра равна σ² (проверка — checks/check_turbulence.py).

УПРОЩЕНИЯ (решение автора, docs/physics.md, разд. «Турбулентность»):
  1. Оси — земные, а не связанные: ug — на север, vg — на восток, wg — вверх.
     Тогда wind(h, t) не зависит от ориентации ЛА и интерфейс ветра не меняется.
     Продольный спектр H_u отдан северной составляющей, поперечный H_v — восточной.
  2. Фильтры рассчитаны на постоянную скорость turb_Va (крейсер 16 м/с), а не на
     текущую Va: поле порывов — заданная функция времени.
  3. Параметры L, σ — таблица 4.1 B&M для двух высот (50 и 600 м), без зависимости
     от высоты внутри полёта. Угловая турбулентность (p_g, q_g, r_g) не моделируется.

Ряд порывов строится на равномерной сетке шага turb_dt интегрированием фильтров
методом Эйлера — Маруямы (шум на шаге — N(0, 1)/√dt), между узлами — линейная
интерполяция. Сетка достраивается по мере роста t, поэтому одно и то же t всегда даёт
одно и то же значение (RK4 на подшагах и 3D-тренажёр видят одно поле).
t0 — момент включения: до него порыв нулевой, после — фильтры стартуют из нуля
(включение в 3D-тренажёре посреди полёта — без скачка ветра).
"""

import numpy as np

from .config import WindParams

# Таблица 4.1 B&M: (высота, м; Lu = Lv, м; Lw, м; σu = σv, м/с; σw, м/с)
DRYDEN_TABLE = {
    ("low",    "light"):    (50.0,  200.0, 50.0,  1.06, 0.7),
    ("low",    "moderate"): (50.0,  200.0, 50.0,  2.12, 1.4),
    ("medium", "light"):    (600.0, 533.0, 533.0, 1.5,  1.5),
    ("medium", "moderate"): (600.0, 533.0, 533.0, 3.0,  3.0),
}
TURB_LEVELS = ("off", "light", "moderate")


def dryden_params(params: WindParams) -> tuple:
    """(Lu, Lv, Lw, σu, σv, σw) для уровня и высоты из WindParams."""
    _, L_uv, Lw, s_uv, sw = DRYDEN_TABLE[(params.turb_alt, params.turb_level)]
    return L_uv, L_uv, Lw, s_uv, s_uv, sw


class DrydenTurbulence:
    """Порывы Драйдена (ug, vg, wg), м/с, в земной СК: север, восток, вверх."""

    CHUNK = 1000                       # узлов сетки за одно достраивание

    def __init__(self, params: WindParams, t0: float = 0.0):
        self.t0 = t0
        Lu, Lv, Lw, su, sv, sw = dryden_params(params)
        V, self.dt = params.turb_Va, params.turb_dt
        self.rng = np.random.default_rng(params.turb_seed)
        # Состояние фильтров x = [x_u, x_v1, x_v2, x_w1, x_w2]: x' = A·x + B·η, порыв = C·x.
        # Звено 1-го порядка: x_u' = −a·x_u + η,  ug = K·x_u.
        # Звено 2-го порядка в канонической форме: x1' = x2, x2' = −a²·x1 − 2a·x2 + η,
        # порыв = K·(a/√3·x1 + x2).
        au, av, aw = V / Lu, V / Lv, V / Lw
        Ku = su * np.sqrt(2.0 * V / Lu)
        Kv = sv * np.sqrt(3.0 * V / Lv)
        Kw = sw * np.sqrt(3.0 * V / Lw)
        A = np.zeros((5, 5))
        A[0, 0] = -au
        A[1, 2], A[2, 1], A[2, 2] = 1.0, -av**2, -2.0 * av
        A[3, 4], A[4, 3], A[4, 4] = 1.0, -aw**2, -2.0 * aw
        B = np.zeros((5, 3))
        B[0, 0], B[2, 1], B[4, 2] = 1.0, 1.0, 1.0
        self.C = np.zeros((3, 5))
        self.C[0, 0] = Ku
        self.C[1, 1], self.C[1, 2] = Kv * av / np.sqrt(3.0), Kv
        self.C[2, 3], self.C[2, 4] = Kw * aw / np.sqrt(3.0), Kw
        # Эйлер — Маруяма: x_{k+1} = (I + A·dt)·x_k + B·√dt·n_k,  n_k ~ N(0, 1)
        self.Ad = np.eye(5) + A * self.dt
        self.Bd = B * np.sqrt(self.dt)
        self.x = np.zeros(5)
        self.series = np.zeros((1, 3)) # узлы сетки: t_k = t0 + k·dt; в t0 порыв нулевой

    def _extend(self, n_min: int):
        """Достроить ряд до n_min узлов (с запасом CHUNK)."""
        n_new = max(n_min - len(self.series), 0) + self.CHUNK
        drive = self.rng.standard_normal((n_new, 3)) @ self.Bd.T
        xs = np.empty((n_new, 5))
        x, Ad = self.x, self.Ad
        for k in range(n_new):
            x = Ad @ x + drive[k]
            xs[k] = x
        self.x = x
        self.series = np.vstack([self.series, xs @ self.C.T])

    def __call__(self, t: float) -> tuple:
        """Порыв в момент t: (ug — север, vg — восток, wg — вверх), м/с."""
        s = max(t - self.t0, 0.0) / self.dt
        k = int(s)
        if k + 2 > len(self.series):
            self._extend(k + 2)
        f = s - k
        g = (1.0 - f) * self.series[k] + f * self.series[k + 1]
        return g[0], g[1], g[2]

"""
Навигация по точкам (B&M гл. 10–11): уставки курса и высоты для САУ.

Структура (поверх боковой САУ LateralController и контура высоты):

    маршрут w_1..w_N  (N, E, h)
      |   p = (N, E) — GPS
      v
    [менеджер маршрута]  — прямые участки, скругление углов дугой радиуса R_fillet
      |   прямая w_{i−1} → w_i  или окружность (центр c, радиус ρ, направление λ)
      v
    [следование по прямой / окружности] -> chi_ref, h_ref
      |
      v
    LateralController (χ → φ → δa),  контур высоты (h → θ → δe)

Следование по прямой (B&M 10.1, векторное поле), начало r, направление q:
    chi_q   = atan2(q_E, q_N)
    e_py    = −sin(chi_q)·(p_N − r_N) + cos(chi_q)·(p_E − r_E)   — боковое отклонение
                                                                    (> 0 — правее линии)
    chi_ref = chi_q − chi_inf·(2/π)·atan(k_path·e_py)
  Далеко от линии ЛА подходит к ней под углом chi_inf, вблизи — плавно ложится
  на линию (масштаб перехода ~ 1/k_path метров).

Следование по окружности (B&M 10.2), λ = +1 — по часовой, −1 — против:
    d = |p − c|,  varphi = atan2(p_E − c_E, p_N − c_N)
    chi_ref = varphi + λ·(π/2 + atan(k_orbit·(d − ρ)/ρ))

Менеджер маршрута со скруглением (B&M 11.2, алгоритм 6). Угол у w_i:
    q0 = орт(w_i − w_{i−1}),  q1 = орт(w_{i+1} − w_i),  ϱ = arccos(−q0·q1)
    прямая w_{i−1} → w_i до полуплоскости через z1 = w_i − (R/tg(ϱ/2))·q0 (нормаль q0);
    дуга: c = w_i − (R/sin(ϱ/2))·орт(q0 − q1),  λ = sign(q0_N·q1_E − q0_E·q1_N),
          до полуплоскости через z2 = w_i + (R/tg(ϱ/2))·q1 (нормаль q1);
    далее — прямая w_i → w_{i+1}.
  ЛА не пролетает саму точку (срезает угол по дуге), зато не проскакивает поворот.
  Последняя точка (без замыкания) — прямая до полуплоскости через w_N, затем
  кружение радиусом R_orbit вокруг w_N.

Высота: на прямой — линейно по ходу участка от h_{i−1} к h_i, на дуге — h_i,
на окружности — h последней точки.
"""

import numpy as np
from dataclasses import dataclass

from .controllers import wrap_angle


@dataclass
class NavParams:
    """Параметры навигации. Подобраны моделированием (s14) для Va = 30 м/с, φ_max = 30°
    (минимальный радиус координированного разворота Va²/(g·tg φ) ≈ 160 м)."""
    chi_inf: float = np.radians(60.0)   # угол подхода к линии издалека
    k_path: float = 0.01                # 1/м: переход на линию на масштабе ~100 м
    k_orbit: float = 2.0                # крутизна поля на окружности
    R_fillet: float = 200.0             # радиус скругления углов маршрута, м (0 — без)
    R_orbit: float = 200.0              # радиус кружения у последней точки, м
    orbit_cw: bool = True               # направление кружения: по часовой
    loop: bool = False                  # замкнутый маршрут: после последней — к первой


def _unit(v):
    return v / max(np.hypot(*v), 1e-9)


class WaypointNavigator:
    """
    Менеджер маршрута + следование по прямой/окружности.

    Режимы mode: "idle"   — маршрута нет (step возвращает None);
                 "line"   — прямая prev → wps[idx];
                 "fillet" — дуга скругления у wps[idx];
                 "orbit"  — кружение вокруг последней точки.
    """

    def __init__(self, params: NavParams = None):
        self.params = params or NavParams()
        self.wps = np.zeros((0, 3))
        self.mode = "idle"
        self.idx = 0
        self.lapped = False
        self.prev = np.zeros(3)
        self.circle = None       # (c_N, c_E, ρ, λ) текущей дуги/окружности
        self.e_py = 0.0          # отклонение: от прямой или от окружности (d − ρ), м
        self.chi_q = 0.0         # направление участка (касательной к окружности), рад
        self.chi_ref = 0.0
        self.h_ref = 0.0

    def set_route(self, waypoints, start):
        """
        Новый маршрут с первой точки.

        Args:
            waypoints: [(N, E, h), ...] — точки, м
            start: (N, E, h) — где ЛА сейчас: начало участка к первой точке
        """
        self.wps = np.asarray(waypoints, float).reshape(-1, 3)
        self.prev = np.asarray(start, float)
        self.idx = 0
        self.lapped = False      # замкнутый маршрут пройден хотя бы раз (prev — последняя точка)
        self.mode = "line" if len(self.wps) else "idle"

    def update_route(self, waypoints):
        """
        Правка маршрута в полёте: цель — точка с тем же номером, участок к ней —
        от предыдущей точки нового маршрута (на первом участке — от старта).
        Кружение остаётся кружением — вокруг новой последней точки.
        """
        new = np.asarray(waypoints, float).reshape(-1, 3)
        if self.mode == "idle" or not len(new):
            self.wps, self.mode = new, ("idle" if not len(new) else self.mode)
            return
        if self.mode == "orbit" and not (self.params.loop and len(new) >= 2):
            self.wps = new
            _, _, rho, lam = self.circle
            self.circle = (new[-1, 0], new[-1, 1], rho, lam)
            return
        lapped = self.lapped and self.params.loop and len(new) >= 2
        start = self.prev if (self.idx == 0 and not lapped) or self.mode == "orbit" else None
        self.wps = new
        self.idx = 0 if self.mode == "orbit" else min(self.idx, len(new) - 1)
        if start is None:
            start = new[self.idx - 1]                      # idx = 0 на втором круге — последняя
        self.prev = np.asarray(start, float).copy()
        self.mode = "line"

    def clear(self):
        self.set_route([], np.zeros(3))

    @property
    def target(self):
        """Текущая точка (N, E, h) или None."""
        return None if self.mode == "idle" else self.wps[self.idx]

    def _next(self):
        """Точка после текущей или None (конец незамкнутого маршрута)."""
        if self.idx + 1 < len(self.wps):
            return self.wps[self.idx + 1]
        return self.wps[0] if (self.params.loop and len(self.wps) >= 2) else None

    def _advance(self):
        """Текущая точка пройдена: следующий участок или кружение."""
        self.prev = self.wps[self.idx].copy()
        if self._next() is None:
            c = self.wps[-1]
            lam = 1.0 if self.params.orbit_cw else -1.0
            self.mode, self.circle = "orbit", (c[0], c[1], self.params.R_orbit, lam)
            return
        self.idx = (self.idx + 1) % len(self.wps)
        self.lapped |= self.idx == 0
        self.mode = "line"

    def _corner(self):
        """Скругление у wps[idx]: (z1, z2, q0, q1, (c_N, c_E, R, λ)) или None, если угла нет."""
        w, nxt, R = self.wps[self.idx][:2], self._next(), self.params.R_fillet
        if nxt is None or R <= 0:
            return None
        q0, q1 = _unit(w - self.prev[:2]), _unit(nxt[:2] - w)
        cos_turn = float(np.clip(q0 @ q1, -1.0, 1.0))
        if cos_turn > np.cos(np.radians(2.0)):             # почти прямо — без дуги
            return None
        rho = max(np.arccos(-cos_turn), np.radians(5.0))   # ϱ: внутренний угол у точки
        # Вынос точек касания d = R/tg(ϱ/2) — не больше половины соседних участков
        # (острый угол / короткий участок): тогда дуга меньшего радиуса, ЛА её не
        # выдерживает по крену и проскакивает — это видно на карте и в e_py.
        L = min(np.hypot(*(w - self.prev[:2])), np.hypot(*(nxt[:2] - w)))
        d = min(R / np.tan(rho / 2), 0.5 * L)
        R = max(d * np.tan(rho / 2), 1.0)
        c = w - R / np.sin(rho / 2) * _unit(q0 - q1)
        lam = np.sign(q0[0] * q1[1] - q0[1] * q1[0]) or 1.0   # разворот на 180° — вправо
        return w - d * q0, w + d * q1, q0, q1, (c[0], c[1], R, lam)

    def step(self, pn: float, pe: float, chi: float):
        """
        Уставки на шаг.

        Args:
            pn, pe: положение ЛА (GPS), м; chi — путевой угол, рад (закон его не
                    использует — рассогласование приводит LateralController)
        Returns:
            (chi_ref, h_ref) в рад и м, или None, если маршрута нет
        """
        if self.mode == "idle":
            return None
        p = np.array([pn, pe])

        # --- переключения (не больше одного за шаг) ---
        if self.mode == "line":
            corner = self._corner()
            w = self.wps[self.idx][:2]
            if corner is None:
                if (p - w) @ _unit(w - self.prev[:2]) >= 0.0:
                    self._advance()
            elif (p - corner[0]) @ corner[2] >= 0.0:       # z1 пересечена — на дугу
                self.mode, self.circle = "fillet", corner[4]
                self._z2, self._q1 = corner[1], corner[3]
        elif self.mode == "fillet":
            if (p - self._z2) @ self._q1 >= 0.0:           # z2 пересечена — на прямую
                self._advance()

        # --- закон следования ---
        pr = self.params
        if self.mode == "line":
            a, b = self.prev, self.wps[self.idx]
            ab = b[:2] - a[:2]
            L = max(np.hypot(*ab), 1e-6)
            q = ab / L
            chi_q = np.arctan2(q[1], q[0])
            r = p - a[:2]
            self.e_py = -np.sin(chi_q) * r[0] + np.cos(chi_q) * r[1]
            self.chi_q = chi_q
            chi_ref = chi_q - pr.chi_inf * (2.0 / np.pi) * np.arctan(pr.k_path * self.e_py)
            s = np.clip(r @ q / L, 0.0, 1.0)                  # доля пройденного участка
            self.h_ref = a[2] + s * (b[2] - a[2])
        else:                                                 # дуга или кружение
            cn, ce, rho, lam = self.circle
            d = np.hypot(pn - cn, pe - ce)
            varphi = np.arctan2(pe - ce, pn - cn)
            self.chi_q = wrap_angle(varphi + lam * np.pi / 2)
            chi_ref = varphi + lam * (np.pi / 2 + np.arctan(pr.k_orbit * (d - rho) / rho))
            self.e_py = d - rho
            self.h_ref = (self.wps[-1] if self.mode == "orbit" else self.wps[self.idx])[2]
        self.chi_ref = wrap_angle(chi_ref)
        return self.chi_ref, self.h_ref

"""
LQR-регулятор продольного канала.

Численная линеаризация dynamics.derivatives() вокруг точки трима.
Оптимальное усиление K: дискретное уравнение Риккати (DARE) через scipy.

Структура:
    h_ref ──[P · Kh]──> theta_ref (ограничен ±25°)
                             |
    x_ref = [Va·cos θ_ref, Va·sin θ_ref, 0, θ_ref]
                             |
    x_meas ──────[–K]──────> Δu ──[+ u_trim]──> [delta_e, throttle]

Состояние LQR (4 компонента): [Δu, Δw, Δq, Δθ]
Управление (2 компонента):    [Δδe, Δthrottle]
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from dataclasses import dataclass, field
from typing import List

from sim.config import AircraftParams
from sim.state import U, W, Q, THETA
from sim.dynamics import derivatives as _derivatives


_IDX4 = [U, W, Q, THETA]   # индексы 4 состояний в 6-мерном векторе состояния
_N = 4                      # размерность состояния LQR
_M = 2                      # размерность вектора управления


@dataclass
class LQRParams:
    """Параметры LQR-регулятора."""

    # Диагональ матрицы Q (штраф за отклонение состояния): [Δu, Δw, Δq, Δθ]
    Q_diag: List[float] = field(default_factory=lambda: [1.0, 1.0, 10.0, 20.0])

    # Диагональ матрицы R (штраф за управление): [Δδe, Δthrottle]
    R_diag: List[float] = field(default_factory=lambda: [150.0, 1.0])

    # Внешний P-контур высоты: theta_ref = theta_trim + h_Kp · (h_ref − h)
    h_Kp: float = 0.05   # рад/м

    # Уставки
    h_ref:  float = 100.0   # м
    Va_ref: float = 30.0    # м/с


class LQRController:
    """
    LQR-регулятор тангажа + скорости с P-контуром высоты.

    Инициализация (выполняется один раз):
      1. Численная линеаризация derivatives() → A_c (4×4), B_c (4×2)
      2. Дискретизация Эйлера: A_d = I + A_c·dt, B_d = B_c·dt
      3. DARE (scipy.linalg.solve_discrete_are) → P → K (2×4)

    Закон управления на каждом шаге:
      θ_ref  = θ_trim + Kh·(h_ref − h_meas)    [внешний P-контур, ±25°]
      x_ref  = [Va·cos θ_ref, Va·sin θ_ref, 0, θ_ref]
      Δu     = −K·(x_meas − x_ref)
      u_cmd  = u_trim + Δu                      [насыщение по физ. ограничениям]

    Интерфейс совместим с PitchController: step(t, meas, dt) → [delta_e, throttle].
    """

    def __init__(self,
                 aircraft: AircraftParams,
                 state_trim: np.ndarray,
                 controls_trim: np.ndarray,
                 wind_fn,
                 dt: float,
                 params: LQRParams):
        """
        Args:
            aircraft:      параметры ЛА
            state_trim:    6-мерный вектор состояния в точке трима
            controls_trim: [delta_e_trim, throttle_trim]
            wind_fn:       wind(h, t) → (Vwx, Vwh)
            dt:            шаг интегрирования, с
            params:        параметры LQR
        """
        self.aircraft = aircraft
        self.params   = params

        # Точка линеаризации
        self.x_trim = state_trim[_IDX4].copy()   # 4-вектор
        self.u_trim = controls_trim.copy()        # [de_trim, thr_trim]

        # Линеаризация
        A_c, B_c = _linearize(state_trim, controls_trim, aircraft, wind_fn)

        # Дискретизация Эйлера (точности достаточно при dt = 0.01 с)
        A_d = np.eye(_N) + A_c * dt
        B_d = B_c * dt

        # DARE → оптимальное усиление
        Q_mat = np.diag(params.Q_diag)
        R_mat = np.diag(params.R_diag)
        P = solve_discrete_are(A_d, B_d, Q_mat, R_mat)
        self.K = np.linalg.inv(R_mat + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

        self.h_ref  = params.h_ref
        self.Va_ref = params.Va_ref

        # Для диагностики
        self.A_c = A_c
        self.B_c = B_c

    # ------------------------------------------------------------------
    # Управление уставками
    # ------------------------------------------------------------------

    def set_altitude_ref(self, h_ref: float):
        """Установить уставку высоты, м."""
        self.h_ref = h_ref

    def set_Va_ref(self, Va_ref: float):
        """Установить уставку воздушной скорости, м/с."""
        self.Va_ref = max(Va_ref, 1.0)

    # ------------------------------------------------------------------
    # Основной шаг
    # ------------------------------------------------------------------

    def step(self, t: float, meas: dict, dt: float) -> np.ndarray:
        """
        Один шаг регулятора.

        Args:
            t:    текущее время, с
            meas: словарь измерений:
                    'u'     — скорость вдоль x_body, м/с
                    'w'     — скорость вдоль z_body, м/с
                    'q'     — угловая скорость тангажа, рад/с
                    'theta' — угол тангажа, рад
                    'h'     — высота, м
                    'Va'    — воздушная скорость, м/с (не используется внутри,
                              но можно передавать для совместимости)
            dt:   шаг, с (не используется в расчёте, но сохраняет интерфейс)

        Returns:
            np.array([delta_e, throttle])
        """
        x_meas = np.array([
            meas.get('u',     self.x_trim[0]),
            meas.get('w',     self.x_trim[1]),
            meas.get('q',     0.0),
            meas.get('theta', self.x_trim[3]),
        ])
        h_meas = meas.get('h', self.h_ref)

        # Внешний P-контур: ошибка высоты → доп. угол тангажа
        theta_ref = self.x_trim[3] + self.params.h_Kp * (self.h_ref - h_meas)
        theta_ref = np.clip(theta_ref, np.radians(-25.0), np.radians(25.0))

        # Уставка вектора состояния
        x_ref = np.array([
            self.Va_ref * np.cos(theta_ref),
            self.Va_ref * np.sin(theta_ref),
            0.0,
            theta_ref,
        ])

        # LQR: корректирующее управление
        delta_u = -self.K @ (x_meas - x_ref)

        # Команды = трим + коррекция (с насыщением)
        u_cmd    = self.u_trim + delta_u
        delta_e  = np.clip(u_cmd[0], self.aircraft.delta_e_min,  self.aircraft.delta_e_max)
        throttle = np.clip(u_cmd[1], self.aircraft.throttle_min, self.aircraft.throttle_max)

        return np.array([delta_e, throttle])

    # ------------------------------------------------------------------
    # Диагностика
    # ------------------------------------------------------------------

    def print_gains(self):
        """Вывести матрицу K и линеаризованные матрицы A_c, B_c."""
        labels = ['du', 'dw', 'dq', 'dth']
        print("LQR gain K  (rows: delta_e, throttle):")
        for i, name in enumerate(['delta_e ', 'throttle']):
            vals = "  ".join(f"{labels[j]}={self.K[i, j]:+.4f}" for j in range(_N))
            print(f"  {name}: {vals}")

        print("\nLinearized A_c (4x4):")
        row_labels = col_labels = ['u', 'w', 'q', 'th']
        header = "        " + "  ".join(f"{c:>8}" for c in col_labels)
        print(header)
        for i, rl in enumerate(row_labels):
            row = f"  d{rl}/dt:  " + "  ".join(f"{self.A_c[i, j]:+8.4f}" for j in range(_N))
            print(row)

        print("\nLinearized B_c (4x2):")
        print("           de       thr")
        for i, rl in enumerate(row_labels):
            print(f"  d{rl}/dt:  {self.B_c[i, 0]:+8.4f}  {self.B_c[i, 1]:+8.4f}")
        print()


# ---------------------------------------------------------------------------
# Численная линеаризация
# ---------------------------------------------------------------------------

def _linearize(state_trim: np.ndarray,
               controls_trim: np.ndarray,
               aircraft: AircraftParams,
               wind_fn) -> tuple:
    """
    Якобианы A_c = ∂f/∂x и B_c = ∂f/∂u в точке трима (конечные разности).

    Возвращает A_c (4×4) и B_c (4×2) — только для 4 состояний [u, w, q, θ].
    """
    eps_s = 1e-5   # приращение по состоянию
    eps_c = 1e-5   # приращение по управлению
    t0    = 0.0    # момент линеаризации (ветер при t=0)

    def f4(s, c):
        d = _derivatives(s, c, t0, aircraft, wind_fn)
        return d[_IDX4]

    f0 = f4(state_trim, controls_trim)

    A = np.zeros((_N, _N))
    for j, state_idx in enumerate(_IDX4):
        sp = state_trim.copy()
        sp[state_idx] += eps_s
        A[:, j] = (f4(sp, controls_trim) - f0) / eps_s

    B = np.zeros((_N, _M))
    for j in range(_M):
        cp = controls_trim.copy()
        cp[j] += eps_c
        B[:, j] = (f4(state_trim, cp) - f0) / eps_c

    return A, B

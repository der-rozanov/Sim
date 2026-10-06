"""
Вектор состояния ЛА (полная модель 6DOF).

Состав (12 компонент):
  [u, w, q, theta, x, h,  v, p, r, phi, psi, y]
   └── продольный ─────┘  └──── боковой ──────┘
  u     — скорость вдоль оси x_body (вперёд), м/с
  w     — скорость вдоль оси z_body (вниз), м/с
  q     — угловая скорость тангажа, рад/с
  theta — угол тангажа, рад
  x     — координата на север (при psi=0 — вперёд по полёту), м
  h     — высота (вверх положительно), м
  v     — скорость вдоль оси y_body (вправо), м/с
  p     — угловая скорость крена, рад/с
  r     — угловая скорость рыскания, рад/с
  phi   — угол крена (> 0 — правое крыло вниз), рад
  psi   — угол рыскания (> 0 — нос вправо от севера), рад
  y     — координата на восток, м

Индексы 0..5 совпадают с прежним продольным вектором: боковой канал добавлен
расширением, а не переписыванием.

Системы координат:
  Земная: x — север, y — восток, h — вверх (NED с h = −z)
  Связанная: x_body — к носу, y_body — на правое крыло, z_body — вниз

Ветер в земной СК: (Vwx, Vwh, Vwy) — север, вверх, восток. Боковая составляющая
идёт последней, поэтому пара (Vwx, Vwh) из продольного кода тоже принимается.

Управление: [delta_e, throttle, delta_a, delta_r] — индексы DE, DT, DA, DR.
"""

import numpy as np
from .config import AircraftParams, SimConfig

# Именованные индексы — никаких «магических» чисел в остальном коде
# Продольный канал
U     = 0
W     = 1
Q     = 2
THETA = 3
X     = 4
H     = 5
# Боковой канал
V     = 6
P     = 7
R     = 8
PHI   = 9
PSI   = 10
Y     = 11

N_STATES = 12

# Индексы вектора управления
DE = 0   # руль высоты, рад
DT = 1   # газ, 0..1
DA = 2   # элероны, рад
DR = 3   # руль направления, рад

N_CONTROLS = 4


def full_controls(controls) -> np.ndarray:
    """
    Привести управление к полному виду [δe, δt, δa, δr].
    Продольные сценарии отдают только [δe, δt] — тогда δa = δr = 0.
    """
    c = np.zeros(N_CONTROLS)
    c[:len(controls)] = controls
    return c


def initial_state(cfg: SimConfig) -> np.ndarray:
    """
    Начальный вектор состояния: прямолинейный полёт на север, без крена.
    Вся начальная скорость — вдоль x_body (alpha = 0).
    """
    state = np.zeros(N_STATES)
    state[U]     = cfg.Va0
    state[H]     = cfg.h0
    state[THETA] = cfg.theta0
    return state


def rotation_body_to_earth(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Матрица перехода из связанной СК в земную NED (последовательность psi→theta→phi).
    Столбцы — орты осей x_b, y_b, z_b в земной СК.
    """
    cf, sf = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi),   np.sin(psi)
    return np.array([
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st,     sf * ct,                cf * ct               ],
    ])


def _rotation(state: np.ndarray) -> np.ndarray:
    return rotation_body_to_earth(state[PHI], state[THETA], state[PSI])


def earth_velocity(state: np.ndarray) -> tuple:
    """
    Земная (путевая) скорость ЛА.

    Возвращает: (Vx, Vy, Vh) — на север, на восток, вверх, м/с
    """
    vn, ve, vd = _rotation(state) @ state[[U, V, W]]
    return vn, ve, -vd


def wind_body(state: np.ndarray, wind_earth) -> np.ndarray:
    """
    Ветер в связанной СК: (u_w, v_w, w_w) = Rᵀ · (Vwx, Vwy, −Vwh).

    wind_earth = (Vwx, Vwh[, Vwy]) — север, вверх, восток; Vwy по умолчанию 0.
    """
    Vwx, Vwh = wind_earth[0], wind_earth[1]
    Vwy = wind_earth[2] if len(wind_earth) > 2 else 0.0
    return _rotation(state).T @ np.array([Vwx, Vwy, -Vwh])


def air_data(state: np.ndarray, wind_earth) -> tuple:
    """
    Воздушная скорость и аэродинамические углы с учётом ветра.

    Возвращает: (Va, alpha, beta)
      Va    — м/с, воздушная скорость
      alpha — рад, УА (> 0 когда нос выше вектора набегающего потока)
      beta  — рад, УС (> 0 когда набегающий поток приходит справа)
    """
    u_w, v_w, w_w = wind_body(state, wind_earth)

    # Скорость ЛА относительно воздуха
    ur = state[U] - u_w
    vr = state[V] - v_w
    wr = state[W] - w_w

    Va    = np.sqrt(ur**2 + vr**2 + wr**2)
    alpha = np.arctan2(wr, ur)
    beta  = np.arcsin(np.clip(vr / Va, -1.0, 1.0)) if Va > 0 else 0.0
    return Va, alpha, beta


def air_velocity(state: np.ndarray, wind_earth) -> tuple:
    """
    Воздушная скорость Va и УА alpha с учётом ветра (интерфейс продольного канала).

    wind_earth = (Vwx, Vwh[, Vwy]):
      Vwx > 0 — ветер на север (при psi=0 — попутный), м/с
      Vwh > 0 — восходящий вертикальный ветер, м/с
      Vwy > 0 — ветер на восток, м/с (необязателен)

    Возвращает: (Va, alpha). УС — см. air_data().
    """
    Va, alpha, _ = air_data(state, wind_earth)
    return Va, alpha


def flight_path_angle(state: np.ndarray, wind_earth) -> float:
    """
    Угол наклона траектории gamma = theta - alpha (по воздушной скорости).
    Точен только в полёте без крена и скольжения.
    """
    _, alpha = air_velocity(state, wind_earth)
    return state[THETA] - alpha


def kinematic_gamma(state: np.ndarray) -> float:
    """
    Угол наклона траектории по земной скорости (GPS-измерение).
    gamma_gps = arctan(Vh / √(Vx² + Vy²))

    Отличается от истинного gamma при наличии ветра —
    это и есть «слепота» косвенной оценки УА.
    """
    Vx, Vy, Vh = earth_velocity(state)
    return np.arctan2(Vh, np.hypot(Vx, Vy))


def total_energy(state: np.ndarray, params: AircraftParams,
                 wind_earth: tuple = (0.0, 0.0)) -> tuple:
    """
    Полная механическая энергия системы.
    Используется для верификации интегратора и графика расхождения.

    Возвращает: (E_kin, E_pot, E_total) в Дж
    """
    Va, _ = air_velocity(state, wind_earth)
    E_kin = 0.5 * params.mass * Va**2
    E_pot = params.mass * params.g * state[H]
    return E_kin, E_pot, E_kin + E_pot

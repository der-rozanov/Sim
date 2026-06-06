"""
Вектор состояния продольного канала.

Состав (6 компонент):
  [u, w, q, theta, x, h]
  u     — скорость вдоль оси x_body (вперёд), м/с
  w     — скорость вдоль оси z_body (вниз), м/с
  q     — угловая скорость тангажа, рад/с
  theta — угол тангажа, рад
  x     — горизонтальная координата (инерциальная), м
  h     — высота (инерциальная, вверх положительно), м

Системы координат:
  Земная: x_earth — горизонталь вперёд, h_earth — вверх
  Связанная: x_body — к носу, z_body — вниз (правая СК через y_body — вправо)
"""

import numpy as np
from config import AircraftParams, SimConfig

# Именованные индексы — никаких «магических» чисел в остальном коде
U     = 0
W     = 1
Q     = 2
THETA = 3
X     = 4
H     = 5

N_STATES = 6


def initial_state(cfg: SimConfig) -> np.ndarray:
    """
    Начальный вектор состояния: прямолинейный полёт, нулевой тангаж.
    Вся начальная скорость — вдоль x_body (alpha = 0).
    """
    state = np.zeros(N_STATES)
    state[U]     = cfg.Va0
    state[H]     = cfg.h0
    state[THETA] = cfg.theta0
    return state


def air_velocity(state: np.ndarray, wind_earth: tuple) -> tuple:
    """
    Воздушная скорость Va и УА alpha с учётом ветра.

    wind_earth = (Vwx, Vwh):
      Vwx > 0 — попутный горизонтальный ветер (в направлении полёта), м/с
      Vwh > 0 — восходящий вертикальный ветер (термик/порыв вверх), м/с

    Возвращает: (Va, alpha)
      Va    — м/с, воздушная скорость
      alpha — рад, угол атаки (> 0 когда нос выше вектора набегающего потока)
    """
    u, w = state[U], state[W]
    theta = state[THETA]
    Vwx, Vwh = wind_earth

    ct, st = np.cos(theta), np.sin(theta)

    # Ветер в связанной СК (проекция на оси тела)
    # u_w_body = Vwx·cos θ + Vwh·sin θ
    # w_w_body = Vwx·sin θ − Vwh·cos θ  (ось z_body направлена вниз)
    u_wind =  Vwx * ct + Vwh * st
    w_wind =  Vwx * st - Vwh * ct

    # Скорость ЛА относительно воздуха
    ur = u - u_wind
    wr = w - w_wind

    Va    = np.sqrt(ur**2 + wr**2)
    alpha = np.arctan2(wr, ur)

    return Va, alpha


def flight_path_angle(state: np.ndarray, wind_earth: tuple) -> float:
    """
    Угол наклона траектории gamma = theta - alpha (по воздушной скорости).
    Используется в косвенной оценке УА (estimators.py).
    """
    _, alpha = air_velocity(state, wind_earth)
    return state[THETA] - alpha


def kinematic_gamma(state: np.ndarray) -> float:
    """
    Угол наклона траектории по земной скорости (GPS-измерение).
    gamma_gps = arctan(ḣ / ẋ) ≈ arctan(u·sin θ − w·cos θ) / (u·cos θ + w·sin θ))

    Отличается от истинного gamma при наличии ветра —
    это и есть «слепота» косвенной оценки УА.
    """
    u, w = state[U], state[W]
    theta = state[THETA]
    ct, st = np.cos(theta), np.sin(theta)

    Vx_earth =  u * ct - w * st   # горизонтальная составляющая земной скорости
    Vh_earth =  u * st - w * ct   # вертикальная составляющая (вверх)

    return np.arctan2(Vh_earth, Vx_earth)


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

"""
Функция производных продольного канала.

derivatives(state, controls, t, params, wind_fn) -> dstate

Чистая функция: только вход -> выход, без побочных эффектов.
Это прямая запись системы ОДУ для приложения Б диссертации.

Уравнения движения (Beard & McLain гл. 3–4, продольный канал):

  Поступательное движение (связанная СК):
    u_dot = (fx_aero + fx_thrust + fx_gravity) / m  −  q·w
    w_dot = (fz_aero + fz_gravity) / m              +  q·u

  Вращательное движение:
    q_dot = M_aero / Jy

  Кинематика:
    theta_dot = q
    x_dot     = u·cos θ − w·sin θ
    h_dot     = u·sin θ + w·cos θ   (знак: h растёт вверх)

Силы гравитации в связанной СК:
    fx_grav = −m·g·sin θ
    fz_grav = +m·g·cos θ

Тяга: только вдоль x_body (тянущий винт, вдоль оси фюзеляжа).
    fx_thrust = k_motor · δt² · ρ · S_prop · (k_motor·δt − Va)

Примечание: δt ∈ [0, 1] — относительная тяга (газ).
"""

import numpy as np
from .config import AircraftParams
from .state import U, W, Q, THETA, X, H, N_STATES
from .aero import aero_forces_moments
from .state import air_velocity


def thrust(throttle: float, Va: float, params: AircraftParams) -> float:
    """
    Тяговое усилие винта, Н.
    Модель Beard & McLain (упрощённая):
      T = 0.5 · ρ · S_prop · C_prop · ((k_motor·δt)² − Va²)
    Зажим сверху T_max (физический предел мотора).
    Авторотация (T < 0 при Va > k_motor·δt) сохраняется.
    """
    Vmotor = params.k_motor * throttle
    T = 0.5 * params.rho * params.S_prop * params.C_prop * (Vmotor**2 - Va**2)
    return min(T, params.T_max)


def derivatives(state: np.ndarray,
                controls: np.ndarray,
                t: float,
                params: AircraftParams,
                wind_fn) -> np.ndarray:
    """
    Производная вектора состояния.

    controls = [delta_e, throttle]
      delta_e  — отклонение руля высоты, рад
      throttle — газ, безразмерный [0, 1]

    wind_fn(h, t) -> (Vwx, Vwh) — функция ветра

    Возвращает dstate той же размерности, что и state.
    """
    u, w, q, theta = state[U], state[W], state[Q], state[THETA]
    delta_e, throttle = controls[0], controls[1]

    # Ветер и воздушные углы
    h = state[H]
    wind_vec = wind_fn(h, t)
    Va, alpha = air_velocity(state, wind_vec)

    # Аэродинамические силы и момент
    fx_a, fz_a, M_pitch = aero_forces_moments(Va, alpha, q, delta_e, params)

    # Тяга (вдоль x_body)
    fx_t = thrust(throttle, Va, params)

    # Гравитация в связанной СК
    fx_g = -params.mass * params.g * np.sin(theta)
    fz_g =  params.mass * params.g * np.cos(theta)

    # Суммарные силы
    fx = fx_a + fx_t + fx_g
    fz = fz_a + fz_g

    # Производные состояния
    dstate = np.zeros(N_STATES)

    # Поступательное движение (уравнения Ньютона в связанной СК)
    dstate[U] = fx / params.mass - q * w
    dstate[W] = fz / params.mass + q * u

    # Вращательное движение
    dstate[Q] = M_pitch / params.Jy

    # Кинематика
    dstate[THETA] = q
    dstate[X] = u * np.cos(theta) - w * np.sin(theta)
    dstate[H] = u * np.sin(theta) - w * np.cos(theta)

    return dstate

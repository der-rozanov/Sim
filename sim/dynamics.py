"""
Функция производных ЛА — полная нелинейная модель 6DOF.

derivatives(state, controls, t, params, wind_fn) -> dstate

Чистая функция: только вход -> выход, без побочных эффектов.
Это прямая запись системы ОДУ для приложения Б диссертации
(вывод и обозначения — docs/physics.md, разделы 3–4).

Уравнения движения (Beard & McLain ур. 3.14–3.17):

  Поступательное движение (связанная СК):
    u_dot = r·v − q·w + fx / m
    v_dot = p·w − r·u + fy / m
    w_dot = q·u − p·v + fz / m

  Вращательное движение (Γ-коэффициенты — см. _gammas):
    p_dot = Γ1·p·q − Γ2·q·r + Γ3·L_roll + Γ4·N_yaw
    q_dot = Γ5·p·r − Γ6·(p² − r²) + M_pitch / Jy
    r_dot = Γ7·p·q − Γ1·q·r + Γ4·L_roll + Γ8·N_yaw

  Кинематика углов Эйлера:
    phi_dot   = p + (q·sin φ + r·cos φ)·tan θ
    theta_dot = q·cos φ − r·sin φ
    psi_dot   = (q·sin φ + r·cos φ) / cos θ

  Кинематика положения: (x_dot, y_dot, −h_dot) = R_b→e · (u, v, w)

Силы тяжести в связанной СК:
    fx_grav = −m·g·sin θ
    fy_grav = +m·g·cos θ·sin φ
    fz_grav = +m·g·cos θ·cos φ

Тяга: только вдоль x_body (тянущий винт, вдоль оси фюзеляжа).

При v = p = r = phi = psi = 0 и delta_a = delta_r = 0 уравнения в точности
совпадают с прежней продольной моделью (docs/physics.md, раздел 3.5).
"""

import numpy as np
from .config import AircraftParams
from .state import (U, W, Q, THETA, X, H, V, P, R, PHI, PSI, Y, N_STATES,
                    DE, DT, DA, DR, full_controls, air_data, earth_velocity)
from .aero import aero_forces_moments, aero_lateral


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


def _gammas(params: AircraftParams) -> tuple:
    """Γ-коэффициенты уравнений вращения (B&M ур. 3.13)."""
    Jx, Jy, Jz, Jxz = params.Jx, params.Jy, params.Jz, params.Jxz
    G  = Jx * Jz - Jxz**2
    G1 = Jxz * (Jx - Jy + Jz) / G
    G2 = (Jz * (Jz - Jy) + Jxz**2) / G
    G3 = Jz / G
    G4 = Jxz / G
    G5 = (Jz - Jx) / Jy
    G6 = Jxz / Jy
    G7 = ((Jx - Jy) * Jx + Jxz**2) / G
    G8 = Jx / G
    return G1, G2, G3, G4, G5, G6, G7, G8


def derivatives(state: np.ndarray,
                controls: np.ndarray,
                t: float,
                params: AircraftParams,
                wind_fn) -> np.ndarray:
    """
    Производная вектора состояния.

    controls = [delta_e, throttle, delta_a, delta_r]
      delta_e  — руль высоты, рад
      throttle — газ, безразмерный [0, 1]
      delta_a  — элероны, рад      (необязателен, по умолчанию 0)
      delta_r  — руль направления, рад (необязателен, по умолчанию 0)

    wind_fn(h, t) -> (Vwx, Vwh[, Vwy]) — функция ветра

    Возвращает dstate той же размерности, что и state.
    """
    u, v, w = state[U], state[V], state[W]
    p, q, r = state[P], state[Q], state[R]
    phi, theta = state[PHI], state[THETA]
    c = full_controls(controls)

    # Ветер и аэродинамические углы
    wind_vec = wind_fn(state[H], t)
    Va, alpha, beta = air_data(state, wind_vec)

    # Аэродинамические силы и моменты
    fx_a, fz_a, M_pitch = aero_forces_moments(Va, alpha, q, c[DE], params)
    fy_a, L_roll, N_yaw = aero_lateral(Va, beta, p, r, c[DA], c[DR], params)

    # Тяга (вдоль x_body)
    fx_t = thrust(c[DT], Va, params)

    # Сила тяжести в связанной СК
    mg = params.mass * params.g
    sf, cf = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    fx_g = -mg * st
    fy_g =  mg * ct * sf
    fz_g =  mg * ct * cf

    # Суммарные силы
    fx = fx_a + fx_t + fx_g
    fy = fy_a + fy_g
    fz = fz_a + fz_g

    G1, G2, G3, G4, G5, G6, G7, G8 = _gammas(params)
    m = params.mass

    dstate = np.zeros(N_STATES)

    # Поступательное движение (уравнения Ньютона в связанной СК)
    dstate[U] = r * v - q * w + fx / m
    dstate[V] = p * w - r * u + fy / m
    dstate[W] = q * u - p * v + fz / m

    # Вращательное движение (уравнения Эйлера)
    dstate[P] = G1 * p * q - G2 * q * r + G3 * L_roll + G4 * N_yaw
    dstate[Q] = G5 * p * r - G6 * (p**2 - r**2) + M_pitch / params.Jy
    dstate[R] = G7 * p * q - G1 * q * r + G4 * L_roll + G8 * N_yaw

    # Кинематика углов Эйлера
    dstate[PHI]   = p + (q * sf + r * cf) * np.tan(theta)
    dstate[THETA] = q * cf - r * sf
    dstate[PSI]   = (q * sf + r * cf) / ct

    # Кинематика положения
    dstate[X], dstate[Y], dstate[H] = earth_velocity(state)

    return dstate

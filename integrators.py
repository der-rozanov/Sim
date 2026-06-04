"""
Численные интеграторы.

Оба взаимозаменяемы — одинаковая сигнатура:
  step(state, controls, dt, t, params, wind_fn) -> state_new

Управление controls заморожено внутри шага (не пересчитывается на k2,k3,k4).
Это намеренно: соответствует дискретной природе реальной САУ,
которая обновляет команды раз в такт.
"""

import numpy as np
from dynamics import derivatives


def step_euler(state: np.ndarray,
               controls: np.ndarray,
               dt: float,
               t: float,
               params,
               wind_fn) -> np.ndarray:
    """Метод Эйлера первого порядка. Используется для сравнения/диагностики."""
    return state + dt * derivatives(state, controls, t, params, wind_fn)


def step_rk4(state: np.ndarray,
             controls: np.ndarray,
             dt: float,
             t: float,
             params,
             wind_fn) -> np.ndarray:
    """
    Метод Рунге-Кутта 4-го порядка.
    Основной интегратор симулятора.

    controls одинаковы для k1..k4 — управление заморожено на шаг.
    """
    k1 = derivatives(state,              controls, t,            params, wind_fn)
    k2 = derivatives(state + dt/2 * k1, controls, t + dt/2,     params, wind_fn)
    k3 = derivatives(state + dt/2 * k2, controls, t + dt/2,     params, wind_fn)
    k4 = derivatives(state + dt   * k3, controls, t + dt,       params, wind_fn)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

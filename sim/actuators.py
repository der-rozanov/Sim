"""
Приводы рулей (сервоприводы) — часть объекта управления (РЕШ-28).

    actuator_step(pos, cmd, dt, params) -> pos_new

Команда САУ (или лётчика) δ_cmd отрабатывается рулём не мгновенно. Модель
каждого руля (δe, δa, δr) — апериодическое звено с ограничением скорости
перекладки и упорами:

    δ_cmd* = clip(δ_cmd, δ_min, δ_max)                 — упоры руля
    δ̇      = clip((δ_cmd* − δ) / τ, −δ̇_max, +δ̇_max)   — инерция + предел скорости

τ = servo_tau — постоянная времени контура положения сервопривода;
δ̇_max = servo_rate — паспортная скорость под нагрузкой (например 0.2 с / 60°).
Малая отработка — линейное звено 1/(τs + 1); большая — перекладка с постоянной
скоростью δ̇_max (насыщение по скорости).

Дискретизация на такт dt (команда постоянна на такте):
    Δδ = clip((δ_cmd* − δ)·(1 − e^(−dt/τ)), −δ̇_max·dt, +δ̇_max·dt)
Без предела скорости это точное решение звена на такте; при τ → 0 — мгновенная
отработка (прежнее поведение). Внутри шага RK4 положение руля заморожено,
как и управление (архитектурное правило 4).

Положения рулей — отдельный вектор той же раскладки, что управление
[δe, δt, δa, δr] (индексы DE, DT, DA, DR). Газ проходит без задержки
(динамика ESC + винта — не моделируется).
"""

import numpy as np
from .config import AircraftParams
from .state import DE, DT, DA, DR, full_controls

SURFACES = (DE, DA, DR)


def surface_limits(params: AircraftParams) -> tuple:
    """Упоры рулей (lo, hi) в порядке SURFACES = (δe, δa, δr), рад."""
    lo = np.array([params.delta_e_min, -params.delta_a_max, -params.delta_r_max])
    hi = np.array([params.delta_e_max, params.delta_a_max, params.delta_r_max])
    return lo, hi


def actuator_init(cmd, params: AircraftParams) -> np.ndarray:
    """Начальное положение рулей — отработанная команда (без переходного процесса)."""
    pos = full_controls(cmd).astype(float)
    idx = list(SURFACES)
    pos[idx] = np.clip(pos[idx], *surface_limits(params))
    return pos


def actuator_step(pos: np.ndarray, cmd, dt: float,
                  params: AircraftParams) -> np.ndarray:
    """Положение рулей через такт dt при команде cmd. Чистая функция."""
    cmd = full_controls(cmd)
    idx = list(SURFACES)
    target = np.clip(cmd[idx], *surface_limits(params))

    k = 1.0 - np.exp(-dt / params.servo_tau) if params.servo_tau > 0.0 else 1.0
    d_max = params.servo_rate * dt
    step = np.clip(k * (target - pos[idx]), -d_max, d_max)

    pos_new = pos.copy()
    pos_new[idx] = pos[idx] + step
    pos_new[DT] = cmd[DT]
    return pos_new

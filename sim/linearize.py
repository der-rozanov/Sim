"""
Численная линеаризация модели 6DOF в переменных механики полёта.

Вектор состояния хранит ориентацию кватернионом (sim/state.py), а линейные модели
продольного и бокового движения записывают в углах Эйлера: [u, w, q, θ], [v, p, r, φ].
Здесь переменная линеаризации — имя из LIN_VARS:
  u, w, q, h, v, p, r, x, y — компоненты вектора состояния;
  phi, theta, psi           — углы Эйлера: возмущение — with_euler(),
                              производная — euler_rates() (B&M ур. 3.3).

    A[i, j] = ∂(ẋ_i)/∂x_j,   B[i, k] = ∂(ẋ_i)/∂δ_k      — разности вперёд, шаг eps

Точка линеаризации — не у θ = ±90° (там углы Эйлера вырождаются).
"""

import numpy as np
from .state import U, W, Q, X, H, V, P, R, Y, euler, with_euler, euler_rates, full_controls
from .dynamics import derivatives

_BODY  = {"u": U, "w": W, "q": Q, "x": X, "h": H, "v": V, "p": P, "r": R, "y": Y}
_EULER = {"phi": 0, "theta": 1, "psi": 2}
LIN_VARS = tuple(_BODY) + tuple(_EULER)


def _perturb(state, name, d):
    if name in _BODY:
        s = state.copy()
        s[_BODY[name]] += d
        return s
    ang = list(euler(state))
    ang[_EULER[name]] += d
    return with_euler(state, *ang)


def _rates(state, dstate, names):
    er = euler_rates(state) if any(n in _EULER for n in names) else None
    return np.array([dstate[_BODY[n]] if n in _BODY else er[_EULER[n]] for n in names])


def jacobian(state, controls, names, params, wind_fn, t=0.0, eps=1e-6):
    """
    Матрицы линейной модели ẋ = A·x + B·δ в точке (state, controls).

    names — переменные состояния, например ("v", "p", "r", "phi");
    B — по всем четырём рулям [δe, δt, δa, δr] (столбцы — индексы DE, DT, DA, DR).
    Возвращает (A, B): (n, n) и (n, 4).
    """
    c0 = full_controls(controls)
    f0 = _rates(state, derivatives(state, c0, t, params, wind_fn), names)
    A = np.zeros((len(names), len(names)))
    for j, name in enumerate(names):
        s = _perturb(state, name, eps)
        A[:, j] = (_rates(s, derivatives(s, c0, t, params, wind_fn), names) - f0) / eps
    B = np.zeros((len(names), len(c0)))
    for k in range(len(c0)):
        c = c0.copy()
        c[k] += eps
        B[:, k] = (_rates(state, derivatives(state, c, t, params, wind_fn), names) - f0) / eps
    return A, B

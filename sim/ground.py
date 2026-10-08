"""
Контакт ЛА с землёй: колёса шасси и точки конструкции (docs/physics.md 4.6, РЕШ-25).

ground_fn(N, E) -> h_g — высота твёрдой поверхности (м, вверх) в точках с
координатами север N, восток E (массивы). Её задаёт мир (карта в viz/world3d.py
— World.surface); физика о картах не знает.

contacts()       — положение, заглубление, скорости точек касания (чистая функция);
ground_forces()  — силы и моменты земли в связанной СК, слагаемое в derivatives();
near_ground()    — ЛА у земли? Вызов поверхности дорогой (~0.2 мс): прогон
                   передаёт ground_fn в шаг, только когда near_ground() = True.

Каждая точка — пружина + демпфер по нормали к земле, трение — сухое,
регуляризованное (sat). Колёса катятся вдоль своей оси (носовое поворачивается
рулём направления), конструкция скользит.
"""

import numpy as np
from .config import AircraftParams
from .state import (U, V, W, P, Q, R, X, Y, H, PHI, THETA, PSI, DR, full_controls,
                    rotation_body_to_earth)

GRAD_STEP = 0.25     # м, шаг разностной производной поверхности (нормаль)


def contact_points(params: AircraftParams):
    """Точки касания (K, 3) в связанной СК и маска колёс (K,)."""
    wh = np.asarray(params.gear_wheels, float).reshape(-1, 3)
    hd = np.asarray(params.gear_hard, float).reshape(-1, 3)
    return np.vstack([wh, hd]), np.arange(len(wh) + len(hd)) < len(wh)


def near_ground(state: np.ndarray, params: AircraftParams, ground_fn) -> bool:
    """ЛА ближе к земле, чем 2·(размер ЛА) + 1 м (с запасом на уклоны до 45°)."""
    reach = np.linalg.norm(contact_points(params)[0], axis=1).max()
    h_g = float(np.ravel(ground_fn(np.array([state[X]]), np.array([state[Y]])))[0])
    return state[H] - h_g < 2 * reach + 1


def contacts(state: np.ndarray, params: AircraftParams, ground_fn):
    """
    Состояние точек касания.

    Возвращает dict массивов по точкам (K — колёса, затем конструкция):
      r     (K, 3) — точки в связанной СК;  wheel (K,) — маска колёс
      pos   (K, 3) — положение, NED;        vel   (K, 3) — скорость точки, NED
      n     (K, 3) — нормаль к земле (вверх, NED)
      depth (K,)   — заглубление по нормали, м (> 0 — касание)
      vn    (K,)   — скорость по нормали, м/с (< 0 — к земле)
    """
    r, wheel = contact_points(params)
    Rm = rotation_body_to_earth(state[PHI], state[THETA], state[PSI])
    pos = np.array([state[X], state[Y], -state[H]]) + r @ Rm.T
    omega = state[[P, Q, R]]
    vel = (state[[U, V, W]] + np.cross(omega, r)) @ Rm.T

    # высота земли и её уклоны в точках — одним вызовом (центр, ±N, ±E)
    N, E, d = pos[:, 0], pos[:, 1], GRAD_STEP
    hg = ground_fn(np.concatenate([N, N + d, N - d, N, N]),
                   np.concatenate([E, E, E, E + d, E - d])).reshape(5, -1)
    dN = (hg[1] - hg[2]) / (2 * d)
    dE = (hg[3] - hg[4]) / (2 * d)
    norm = np.sqrt(1.0 + dN**2 + dE**2)
    n = np.stack([-dN, -dE, -np.ones_like(dN)], 1) / norm[:, None]
    depth = (hg[0] + pos[:, 2]) / norm            # h_g − h_точки, по нормали
    vn = np.einsum("ij,ij->i", vel, n)
    return dict(r=r, wheel=wheel, pos=pos, vel=vel, n=n, depth=depth, vn=vn)


def ground_forces(state: np.ndarray, controls, params: AircraftParams, ground_fn):
    """Сила (fx, fy, fz) и момент (L, M, N) земли в связанной СК."""
    zero = np.zeros(3)
    c = contacts(state, params, ground_fn)
    if not (c["depth"] > 0).any():
        return zero, zero
    on = c["depth"] > 0
    r, n, wheel = c["r"][on], c["n"][on], c["wheel"][on]
    vn, vel = c["vn"][on], c["vel"][on]
    pr = params
    sat = lambda v: np.clip(v / pr.gear_v_eps, -1.0, 1.0)

    Nf = np.maximum(0.0, pr.gear_k * c["depth"][on] - pr.gear_c * vn)    # реакция
    vt = vel - vn[:, None] * n                                           # касательная скорость

    # колёса: направление качения e — ось x_b, повёрнутая на угол колеса (носовое — от δr)
    Rm = rotation_body_to_earth(state[PHI], state[THETA], state[PSI])
    steer = np.zeros(len(c["wheel"]))
    steer[0] = -pr.gear_steer * full_controls(controls)[DR]              # δr > 0 — нос влево
    steer = steer[on]
    e = np.stack([np.cos(steer), np.sin(steer), np.zeros_like(steer)], 1) @ Rm.T
    e -= np.einsum("ij,ij->i", e, n)[:, None] * n                        # в плоскость земли
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    s = np.cross(n, e)
    v_roll = np.clip(np.einsum("ij,ij->i", vt, e) / pr.gear_v_eps_roll, -1.0, 1.0)
    F_wheel = -(pr.gear_mu_roll * v_roll[:, None] * e
                + pr.gear_mu_side * sat(np.einsum("ij,ij->i", vt, s))[:, None] * s)
    # конструкция: скольжение против касательной скорости
    vabs = np.linalg.norm(vt, axis=1)
    F_slide = -pr.gear_mu_slide * (sat(vabs) / (vabs + 1e-12))[:, None] * vt
    Ft = np.where(wheel[:, None], F_wheel, F_slide)

    F = (Nf[:, None] * (n + Ft)) @ Rm                                    # NED → связанная
    return F.sum(0), np.cross(r, F).sum(0)

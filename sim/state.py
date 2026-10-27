"""
Вектор состояния ЛА (полная модель 6DOF).

Состав (13 компонент):
  [u, w, q, x, h,  v, p, r, y,  e0, e1, e2, e3]
   └ продольный ┘  └ боковой ┘  └ ориентация ┘
  u     — скорость вдоль оси x_body (вперёд), м/с
  w     — скорость вдоль оси z_body (вниз), м/с
  q     — угловая скорость тангажа, рад/с
  x     — координата на север, м
  h     — высота (вверх положительно), м
  v     — скорость вдоль оси y_body (вправо), м/с
  p     — угловая скорость крена, рад/с
  r     — угловая скорость рыскания, рад/с
  y     — координата на восток, м
  e0..e3 — кватернион ориентации (поворот земной СК NED в связанную), |e| = 1;
           e0 — скалярная часть (B&M, приложение B)

Ориентация хранится кватернионом, а не углами Эйлера (решение автора 2026-10-27):
кинематика углов Эйлера вырождается при θ = ±90° (деление на cos θ в φ̇, ψ̇), а
кватерниона — нет; вертикали, петли и бочки проходятся без особенностей.
Углы Эйлера (φ, θ, ψ) — производные величины: euler(state). Диапазоны:
φ, ψ ∈ (−π, π], θ ∈ [−π/2, π/2].
  phi   — угол крена (> 0 — правое крыло вниз)
  theta — угол тангажа (> 0 — нос вверх)
  psi   — угол рыскания (> 0 — нос вправо от севера)

Связь с углами Эйлера (последовательность ψ → θ → φ, B&M ур. B.1):
  e0 = cψ/2·cθ/2·cφ/2 + sψ/2·sθ/2·sφ/2
  e1 = cψ/2·cθ/2·sφ/2 − sψ/2·sθ/2·cφ/2
  e2 = cψ/2·sθ/2·cφ/2 + sψ/2·cθ/2·sφ/2
  e3 = sψ/2·cθ/2·cφ/2 − cψ/2·sθ/2·sφ/2

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
U  = 0
W  = 1
Q  = 2
X  = 3
H  = 4
# Боковой канал
V  = 5
P  = 6
R  = 7
Y  = 8
# Ориентация — кватернион
E0 = 9
E1 = 10
E2 = 11
E3 = 12
QUAT = [E0, E1, E2, E3]

N_STATES = 13

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


# ---------------------------------------------------------------------------
# Ориентация: кватернион ↔ углы Эйлера ↔ матрица поворота
# ---------------------------------------------------------------------------

def quat_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    """Кватернион [e0, e1, e2, e3] по углам Эйлера (B&M ур. B.1)."""
    cf, sf = np.cos(phi / 2),   np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cp, sp = np.cos(psi / 2),   np.sin(psi / 2)
    return np.array([cp * ct * cf + sp * st * sf,
                     cp * ct * sf - sp * st * cf,
                     cp * st * cf + sp * ct * sf,
                     sp * ct * cf - cp * st * sf])


def euler(state: np.ndarray) -> tuple:
    """
    Углы Эйлера (φ, θ, ψ) по кватерниону состояния (B&M ур. B.2), рад.

    state — один вектор (13,) или массив векторов (N, 13): тогда φ, θ, ψ — (N,).
    Кватернион нормируется (|e| после интегрирования может отличаться от 1 на ~1e-12).
    У θ = ±90° углы φ и ψ по отдельности не определены (определена их сумма или
    разность) — это свойство углов Эйлера, не модели.
    """
    e0, e1, e2, e3 = (state[..., k] for k in QUAT)
    n2 = e0**2 + e1**2 + e2**2 + e3**2
    phi   = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2 + e3**2 - e1**2 - e2**2)
    theta = np.arcsin(np.clip(2.0 * (e0 * e2 - e1 * e3) / n2, -1.0, 1.0))
    psi   = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2 + e1**2 - e2**2 - e3**2)
    return phi, theta, psi


def with_euler(state: np.ndarray, phi: float, theta: float, psi: float) -> np.ndarray:
    """Копия вектора состояния с ориентацией, заданной углами Эйлера."""
    s = np.array(state, dtype=float)
    s[QUAT] = quat_from_euler(phi, theta, psi)
    return s


def euler_rates(state: np.ndarray) -> np.ndarray:
    """
    Производные углов Эйлера по угловым скоростям (B&M ур. 3.3), рад/с:
      φ̇ = p + (q·sin φ + r·cos φ)·tan θ
      θ̇ = q·cos φ − r·sin φ
      ψ̇ = (q·sin φ + r·cos φ) / cos θ
    В интегрировании не участвует — для линеаризации и проверок
    (особенность при θ = ±90°).
    """
    phi, theta, _ = euler(state)
    p, q, r = state[P], state[Q], state[R]
    sf, cf = np.sin(phi), np.cos(phi)
    return np.array([p + (q * sf + r * cf) * np.tan(theta),
                     q * cf - r * sf,
                     (q * sf + r * cf) / np.cos(theta)])


def initial_state(cfg: SimConfig) -> np.ndarray:
    """
    Начальный вектор состояния: прямолинейный полёт на север, без крена.
    Вся начальная скорость — вдоль x_body (alpha = 0).
    """
    state = np.zeros(N_STATES)
    state[U]    = cfg.Va0
    state[H]    = cfg.h0
    state[QUAT] = quat_from_euler(0.0, cfg.theta0, 0.0)
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


def rotation_quat(e) -> np.ndarray:
    """
    Матрица перехода из связанной СК в земную NED по кватерниону (B&M ур. B.4,
    транспонированная). Кватернион нормируется.
    """
    e0, e1, e2, e3 = np.asarray(e, dtype=float) / np.linalg.norm(e)
    return np.array([
        [e0**2 + e1**2 - e2**2 - e3**2, 2 * (e1 * e2 - e0 * e3),       2 * (e1 * e3 + e0 * e2)],
        [2 * (e1 * e2 + e0 * e3),       e0**2 - e1**2 + e2**2 - e3**2, 2 * (e2 * e3 - e0 * e1)],
        [2 * (e1 * e3 - e0 * e2),       2 * (e2 * e3 + e0 * e1),       e0**2 - e1**2 - e2**2 + e3**2],
    ])


def rotation(state: np.ndarray) -> np.ndarray:
    """Матрица перехода связанная → земная NED для вектора состояния."""
    return rotation_quat(state[QUAT])


def earth_velocity(state: np.ndarray) -> tuple:
    """
    Земная (путевая) скорость ЛА.

    Возвращает: (Vx, Vy, Vh) — на север, на восток, вверх, м/с
    """
    vn, ve, vd = rotation(state) @ state[[U, V, W]]
    return vn, ve, -vd


def wind_body(state: np.ndarray, wind_earth) -> np.ndarray:
    """
    Ветер в связанной СК: (u_w, v_w, w_w) = Rᵀ · (Vwx, Vwy, −Vwh).

    wind_earth = (Vwx, Vwh[, Vwy]) — север, вверх, восток; Vwy по умолчанию 0.
    """
    Vwx, Vwh = wind_earth[0], wind_earth[1]
    Vwy = wind_earth[2] if len(wind_earth) > 2 else 0.0
    return rotation(state).T @ np.array([Vwx, Vwy, -Vwh])


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

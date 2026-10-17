"""
Псевдодатчики (датчики) с шумом и смещением.

Каждый датчик — чистая функция:
    meas = measure_*(true_value, bias, noise_std, rng)

Шум генерируется независимо для каждого датчика и прогона (разные seed'ы).
Источник параметров: B&M прил. H (по умолчанию) или авторские данные.
"""

import numpy as np


def measure_gyro(q_true, bias, noise_std, rng):
    """
    Гироскоп (угловая скорость по одной оси: p, q или r).

    Args:
        q_true: истинная угловая скорость, рад/с
        bias: смещение датчика, рад/с
        noise_std: среднее квадратичное отклонение шума, рад/с
        rng: numpy.random.Generator

    Returns:
        q_meas: измеренная угловая скорость, рад/с
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return q_true + bias + noise


def measure_altitude(h_true, bias, noise_std, rng):
    """
    Барометр (высота).

    Args:
        h_true: истинная высота, м
        bias: смещение датчика, м
        noise_std: СКО шума, м
        rng: numpy.random.Generator

    Returns:
        h_meas: измеренная высота, м
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return h_true + bias + noise


def measure_airspeed(Va_true, bias, noise_std, rng):
    """
    СВС — датчик воздушной скорости.

    Args:
        Va_true: истинная воздушная скорость, м/с
        bias: смещение, м/с
        noise_std: СКО шума, м/с
        rng: numpy.random.Generator

    Returns:
        Va_meas: измеренная воздушная скорость, м/с (≥ 0)
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return max(0.0, Va_true + bias + noise)


def measure_angle_of_attack(alpha_true, bias, noise_std, rng):
    """
    Зонд УА (прямое измерение угла атаки).

    Args:
        alpha_true: истинный УА, рад
        bias: смещение, рад
        noise_std: СКО шума, рад
        rng: numpy.random.Generator

    Returns:
        alpha_meas: измеренный УА, рад
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return alpha_true + bias + noise


def measure_gps_velocity_earth(u_body, w_body, theta, bias, noise_std, rng):
    """
    GPS: земная скорость (горизонтальная и вертикальная).
    Преобразование из связанной СК в земную и добавление шума.
    Продольная версия (полёт без крена и скольжения): Vx = u·cos θ + w·sin θ.
    Знак исправлен 2026-10-06 (было −w·sin θ) — согласовано с dynamics и
    state.kinematic_gamma, см. docs/physics.md раздел 3.5.

    Args:
        u_body: продольная скорость в связанной СК, м/с
        w_body: нормальная скорость в связанной СК, м/с
        theta: угол тангажа, рад
        bias: смещение (кортеж из 2 элементов или 0.0)
        noise_std: СКО шума, м/с
        rng: numpy.random.Generator

    Returns:
        (Vx_earth, Vh_earth): компоненты земной скорости, м/с
            Vx_earth > 0 — движение вперёд
            Vh_earth > 0 — восходящее движение
    """
    ct, st = np.cos(theta), np.sin(theta)

    # Преобразование в земную СК (определение см. state.py)
    Vx_earth_true =  u_body * ct + w_body * st
    Vh_earth_true =  u_body * st - w_body * ct

    bias_x = bias if isinstance(bias, (int, float)) else bias[0] if len(bias) > 0 else 0.0
    bias_h = bias if isinstance(bias, (int, float)) else bias[1] if len(bias) > 1 else 0.0

    if noise_std > 0:
        noise_x = rng.normal(0.0, noise_std)
        noise_h = rng.normal(0.0, noise_std)
    else:
        noise_x = noise_h = 0.0

    return (Vx_earth_true + bias_x + noise_x,
            Vh_earth_true + bias_h + noise_h)


# ---------------------------------------------------------------------------
# Боковой канал
# ---------------------------------------------------------------------------

def measure_attitude(angle_true, bias, noise_std, rng):
    """
    ИНС: угол ориентации (крен phi, тангаж theta или рыскание psi).

    Args:
        angle_true: истинный угол, рад
        bias: смещение, рад
        noise_std: СКО шума, рад
        rng: numpy.random.Generator

    Returns:
        angle_meas: измеренный угол, рад
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return angle_true + bias + noise


def measure_sideslip(beta_true, bias, noise_std, rng):
    """
    Зонд УС (прямое измерение угла скольжения).

    Тот же зонд, что измеряет УА: оба аэродинамических угла.

    Args:
        beta_true: истинный УС, рад
        bias: смещение, рад
        noise_std: СКО шума, рад
        rng: numpy.random.Generator

    Returns:
        beta_meas: измеренный УС, рад
    """
    noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    return beta_true + bias + noise


def measure_gps_course(Vx_true, Vy_true, noise_std, rng):
    """
    GPS: путевой угол chi = arctan(Vy / Vx) по земной скорости.

    Шум добавляется к составляющим скорости (как у приёмника), затем
    вычисляется угол. При ветре путевой угол chi ≠ курсу psi (угол сноса).

    Args:
        Vx_true: земная скорость на север, м/с
        Vy_true: земная скорость на восток, м/с
        noise_std: СКО шума скорости, м/с
        rng: numpy.random.Generator

    Returns:
        chi_meas: путевой угол, рад (0 — север, > 0 — к востоку)
    """
    if noise_std > 0:
        Vx_true = Vx_true + rng.normal(0.0, noise_std)
        Vy_true = Vy_true + rng.normal(0.0, noise_std)
    return np.arctan2(Vy_true, Vx_true)


# ---------------------------------------------------------------------------
# Звенья реалистичного набора датчиков (RealisticSensors, control/sau.py)
# ---------------------------------------------------------------------------

class GaussMarkov:
    """
    Марковский процесс 1-го порядка (процесс Гаусса — Маркова) — медленный дрейф
    смещения датчика:

        b' = −b/τ + w,   b_{k+1} = e^{−T/τ}·b_k + σ·√(1 − e^{−2T/τ})·n_k,  n_k ~ N(0, 1)

    σ — установившееся СКО, τ — время корреляции (за τ корреляция падает до 1/e).
    Дискретизация точная при любом шаге T. Старт — из установившегося распределения.
    """

    def __init__(self, sigma, tau, rng, size=1):
        self.sigma, self.tau, self.rng = sigma, tau, rng
        self.b = sigma * rng.standard_normal(size)

    def step(self, T):
        if T > 0 and self.sigma > 0:
            a = np.exp(-T / self.tau)
            self.b = a * self.b + self.sigma * np.sqrt(1.0 - a * a) * self.rng.standard_normal(self.b.shape)
        return self.b


class SampledChannel:
    """
    Канал датчика с частотой отсчётов и запаздыванием: отсчёт берётся раз в
    period с; на выход идёт последний отсчёт, взятый не позже t − delay; между
    отсчётами выход удерживается (экстраполятор нулевого порядка).
    До первого отсчёта старше delay — выдаётся самый первый отсчёт.
    """

    def __init__(self, period, delay=0.0):
        self.period, self.delay = period, delay
        self.reset()

    def reset(self):
        self.buf, self.t_next = [], None

    def update(self, t, sample_fn):
        """sample_fn() -> значение отсчёта; вызывается только в моменты отсчёта."""
        eps = 1e-9
        if self.t_next is None or t >= self.t_next - eps:
            self.buf.append((t, sample_fn()))
            self.t_next = (t if self.t_next is None else max(self.t_next, t - self.period)) + self.period
        while len(self.buf) > 1 and self.buf[1][0] <= t - self.delay + eps:
            self.buf.pop(0)
        return self.buf[0][1]

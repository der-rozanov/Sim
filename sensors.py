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
    Гироскоп (угловая скорость тангажа).

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


def measure_gps_position(x_true, h_true, bias, noise_std, rng):
    """
    GPS: горизонтальная координата и высота (независимые шумы).

    Args:
        x_true: истинная горизонтальная координата, м
        h_true: истинная высота, м
        bias: смещение (кортеж из 2 элементов или 0.0)
        noise_std: СКО шума, м
        rng: numpy.random.Generator

    Returns:
        (x_meas, h_meas): измеренные координаты, м
    """
    bias_x = bias if isinstance(bias, (int, float)) else bias[0] if len(bias) > 0 else 0.0
    bias_h = bias if isinstance(bias, (int, float)) else bias[1] if len(bias) > 1 else 0.0

    if noise_std > 0:
        noise_x = rng.normal(0.0, noise_std)
        noise_h = rng.normal(0.0, noise_std)
    else:
        noise_x = noise_h = 0.0

    return (x_true + bias_x + noise_x, h_true + bias_h + noise_h)


def measure_gps_velocity_earth(u_body, w_body, theta, bias, noise_std, rng):
    """
    GPS: земная скорость (горизонтальная и вертикальная).
    Преобразование из связанной СК в земную и добавление шума.

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
    Vx_earth_true =  u_body * ct - w_body * st
    Vh_earth_true =  u_body * st + w_body * ct

    bias_x = bias if isinstance(bias, (int, float)) else bias[0] if len(bias) > 0 else 0.0
    bias_h = bias if isinstance(bias, (int, float)) else bias[1] if len(bias) > 1 else 0.0

    if noise_std > 0:
        noise_x = rng.normal(0.0, noise_std)
        noise_h = rng.normal(0.0, noise_std)
    else:
        noise_x = noise_h = 0.0

    return (Vx_earth_true + bias_x + noise_x,
            Vh_earth_true + bias_h + noise_h)


def measure_accelerometer(u_dot, w_dot, theta, g, bias, noise_std, rng):
    """
    Акселерометр (линейные ускорения в связанной СК, с гравитацией).

    Сумма инерционного ускорения (u_dot, w_dot) и гравитационного эффекта.
    В связанной СК при крене=0:
        a_x_body = u_dot − g·sin(theta)
        a_z_body = w_dot + g·cos(theta)  (z_body — вниз)

    Args:
        u_dot: истинное ускорение u, м/с²
        w_dot: истинное ускорение w, м/с²
        theta: угол тангажа, рад
        g: ускорение свободного падения, м/с²
        bias: смещение (кортеж из 2 элементов или 0.0)
        noise_std: СКО шума, м/с²
        rng: numpy.random.Generator

    Returns:
        (a_x, a_z): измеренные ускорения в связанной СК, м/с²
    """
    st, ct = np.sin(theta), np.cos(theta)

    a_x_true = u_dot - g * st
    a_z_true = w_dot + g * ct

    bias_x = bias if isinstance(bias, (int, float)) else bias[0] if len(bias) > 0 else 0.0
    bias_z = bias if isinstance(bias, (int, float)) else bias[1] if len(bias) > 1 else 0.0

    if noise_std > 0:
        noise_x = rng.normal(0.0, noise_std)
        noise_z = rng.normal(0.0, noise_std)
    else:
        noise_x = noise_z = 0.0

    return (a_x_true + bias_x + noise_x,
            a_z_true + bias_z + noise_z)

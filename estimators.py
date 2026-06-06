"""
Оценщики угла атаки (УА).

Два подхода:
1. Прямое измерение (зонд): sensors.measure_angle_of_attack()
       alpha_probe = alpha_true + bias + noise_probe
       Измеряет воздушный угол напрямую — результат не зависит от ветра.

2. Косвенная оценка (ИНС + GPS):
       alpha_est = theta_meas − gamma_gps
       gamma_gps = arctan(Vh_gps / Vx_gps)  — угол по ЗЕМНОЙ скорости

   Источник ошибки при ветре:
       gamma_air = arctan(Vh_air / Vx_air)  ≠  gamma_gps, если Vwind ≠ 0.
       При горизонтальном ветре Vwx (попутный > 0):
           Vx_earth = Vx_air + Vwx  →  gamma_gps < gamma_air (при подъёме)
           delta_alpha ≈ +Vwx · sin(gamma_air) / Va   (систематическое завышение)
"""

import numpy as np


def estimate_alpha_indirect(theta_meas: float,
                             Vx_gps: float,
                             Vh_gps: float) -> float:
    """
    Косвенная оценка УА по данным ИНС (тангаж) и GPS (земная скорость).

    Формула: alpha_est = theta_meas − gamma_gps
             gamma_gps = arctan(Vh_gps / Vx_gps)

    Точна при нулевом ветре. При ненулевом горизонтальном ветре
    вносит систематическую ошибку, пропорциональную вертикальной скорости:
        delta_alpha ≈ Vwx · Vh_air / Va²   (малые углы, горизонтальный ветер)

    Args:
        theta_meas: тангаж (рад), от ИНС/гироскопа
        Vx_gps:     горизонтальная земная скорость (м/с), от GPS (вперёд > 0)
        Vh_gps:     вертикальная земная скорость (м/с), от GPS (вверх > 0)

    Returns:
        alpha_est: оцененный УА, рад
    """
    # Защита: не делить на почти-нуль (ЛА почти не летит горизонтально)
    Vx_safe = Vx_gps if abs(Vx_gps) > 0.5 else (0.5 if Vx_gps >= 0 else -0.5)
    gamma_gps = np.arctan2(Vh_gps, Vx_safe)
    return theta_meas - gamma_gps

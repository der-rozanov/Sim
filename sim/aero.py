"""
Аэродинамическая модель ЛА.

Коэффициенты и силы по Beard & McLain гл. 4.
Продольный канал: модель CL — нелинейная (sigmoid), корректно воспроизводит срыв.
Боковой канал: линейная модель CY, Croll, Cn по β, p̂, r̂, δa, δr.
"""

import numpy as np
from .config import AircraftParams


# ---------------------------------------------------------------------------
# Коэффициенты
# ---------------------------------------------------------------------------

def _sigmoid(alpha: float, params: AircraftParams) -> float:
    """
    Сглаживающая функция σ(α) ∈ [0, 1].
    σ → 0 при малых α (линейная аэродинамика),
    σ → 1 при α >> alpha_stall (режим плоской пластины / срыв).
    """
    M  = params.M_sigmoid
    a0 = params.alpha_stall
    e_pos = np.exp(-M * (alpha - a0))
    e_neg = np.exp( M * (alpha + a0))
    return (1.0 + e_pos + e_neg) / ((1.0 + e_pos) * (1.0 + e_neg))


def coef_CL(alpha: float, q: float, delta_e: float,
            Va: float, params: AircraftParams) -> float:
    """
    Коэффициент подъёмной силы CL (нелинейная модель).

    Смешивает линейный CL и CL плоской пластины через σ(α):
      CL_linear     = CL0 + CLa·α
      CL_flat_plate = 2·sign(α)·sin²(α)·cos(α)
      CL_base       = (1−σ)·CL_linear + σ·CL_flat_plate

    Добавки от q и delta_e — линейные (квазистационарное приближение).
    """
    sigma     = _sigmoid(alpha, params)
    CL_linear = params.CL0 + params.CLa * alpha
    CL_flat   = 2.0 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha)

    Va_safe = max(Va, 1.0)   # защита от деления на ноль при Va → 0
    q_hat   = params.c * q / (2.0 * Va_safe)

    return ((1.0 - sigma) * CL_linear + sigma * CL_flat
            + params.CLq  * q_hat
            + params.CLde * delta_e)


def coef_CD(alpha: float, params: AircraftParams) -> float:
    """
    Коэффициент лобового сопротивления CD (квадратичная модель).

    CD = CDp + CL_linear² / (π·e·AR)
      CDp  — вредное (вязкостное) сопротивление
      AR   — удлинение крыла b²/S
    """
    AR        = params.b**2 / params.S
    CL_linear = params.CL0 + params.CLa * alpha
    return params.CDp + CL_linear**2 / (np.pi * params.e_oswald * AR)


def coef_Cm(alpha: float, q: float, delta_e: float,
            Va: float, params: AircraftParams) -> float:
    """
    Коэффициент момента тангажа Cm (линейная модель).

    Cma < 0 — продольная статическая устойчивость (ЛА само стремится
    вернуться к балансировочному УА при возмущении).
    """
    Va_safe = max(Va, 1.0)
    q_hat   = params.c * q / (2.0 * Va_safe)

    return (params.Cm0
            + params.Cma  * alpha
            + params.Cmq  * q_hat
            + params.Cmde * delta_e)


# ---------------------------------------------------------------------------
# Силы и момент
# ---------------------------------------------------------------------------

def aero_forces_moments(Va: float, alpha: float, q: float,
                        delta_e: float,
                        params: AircraftParams) -> tuple:
    """
    Аэродинамические силы в связанной СК и момент тангажа.

    Алгоритм:
      1. Коэффициенты CL, CD, Cm.
      2. Подъёмная L и сопротивление D из динамического давления q_dyn = ½ρVa².
      3. Поворот из скоростной СК (x_s ∥ Va) в связанную через угол α:
           fx =  −D·cos α  +  L·sin α
           fz =  −D·sin α  −  L·cos α
         (z_body направлена вниз, поэтому подъёмная сила fx < 0 — в смысле fz)
      4. Момент M напрямую из Cm.

    Возвращает: (fx, fz, M_pitch)
      fx      — вдоль x_body (вперёд), Н
      fz      — вдоль z_body (вниз),  Н  [fz < 0 при нормальном полёте]
      M_pitch — вокруг y_body (тангаж), Н·м  [> 0 — нос вверх]
    """
    q_dyn = 0.5 * params.rho * Va**2

    CL = coef_CL(alpha, q, delta_e, Va, params)
    CD = coef_CD(alpha, params)
    Cm = coef_Cm(alpha, q, delta_e, Va, params)

    L = q_dyn * params.S * CL
    D = q_dyn * params.S * CD
    M_pitch = q_dyn * params.S * params.c * Cm

    ca, sa = np.cos(alpha), np.sin(alpha)
    fx = -D * ca + L * sa
    fz = -D * sa - L * ca

    return fx, fz, M_pitch


# ---------------------------------------------------------------------------
# Боковой канал
# ---------------------------------------------------------------------------

def _lateral_coef(C0, C_beta, C_p, C_r, C_da, C_dr,
                  beta, p_hat, r_hat, delta_a, delta_r) -> float:
    """Линейная боковая модель: C = C0 + C_beta·β + C_p·p̂ + C_r·r̂ + C_da·δa + C_dr·δr."""
    return (C0 + C_beta * beta + C_p * p_hat + C_r * r_hat
            + C_da * delta_a + C_dr * delta_r)


def aero_lateral(Va: float, beta: float, p: float, r: float,
                 delta_a: float, delta_r: float,
                 params: AircraftParams) -> tuple:
    """
    Боковая аэродинамическая сила и моменты крена/рыскания в связанной СК.

      p̂ = b·p/(2Va),  r̂ = b·r/(2Va)
      fy     = q_dyn·S   · CY
      L_roll = q_dyn·S·b · Croll
      N_yaw  = q_dyn·S·b · Cn

    Возвращает: (fy, L_roll, N_yaw)
      fy     — вдоль y_body (вправо), Н
      L_roll — вокруг x_body, Н·м  [> 0 — правое крыло вниз]
      N_yaw  — вокруг z_body, Н·м  [> 0 — нос вправо]
    """
    pr = params
    q_dyn   = 0.5 * pr.rho * Va**2
    Va_safe = max(Va, 1.0)
    p_hat   = pr.b * p / (2.0 * Va_safe)
    r_hat   = pr.b * r / (2.0 * Va_safe)
    args    = (beta, p_hat, r_hat, delta_a, delta_r)

    CY    = _lateral_coef(pr.CY0, pr.CY_beta, pr.CY_p, pr.CY_r,
                          pr.CY_da, pr.CY_dr, *args)
    Croll = _lateral_coef(pr.Croll0, pr.Croll_beta, pr.Croll_p, pr.Croll_r,
                          pr.Croll_da, pr.Croll_dr, *args)
    Cn    = _lateral_coef(pr.Cn0, pr.Cn_beta, pr.Cn_p, pr.Cn_r,
                          pr.Cn_da, pr.Cn_dr, *args)

    return (q_dyn * pr.S * CY,
            q_dyn * pr.S * pr.b * Croll,
            q_dyn * pr.S * pr.b * Cn)

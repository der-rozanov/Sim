"""
Аэродинамическая модель продольного канала.

Коэффициенты и силы по Beard & McLain гл. 4.
Модель CL — нелинейная (sigmoid), корректно воспроизводит срыв.
"""

import numpy as np
from config import AircraftParams


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

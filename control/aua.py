"""
АУА — Автомат Углов Атаки.

Надзорный уровень поверх PitchController. При выходе УА за критическое значение
захватывает управление: опускает нос и даёт максимальную тягу до выхода из опасного режима.

Схема включения в controls_fn:
    aua_out = aua.step(alpha_meas, theta_ref_mission, thr_trim, dt)
    controller.set_pitch_setpoint(aua_out.theta_ref)
    controller.set_trim_throttle(aua_out.thr_trim)
    ctrl = controller.step(t, meas, dt)
    if aua_out.force_throttle is not None:
        ctrl[1] = aua_out.force_throttle

Состояния:
  NORMAL    (0) — нормальный полёт, alpha < alpha_warn
  WARNING   (1) — alpha_warn ≤ alpha < alpha_crit → мягкая коррекция theta_ref
  CRITICAL  (2) — alpha ≥ alpha_crit → перехват: нос вниз + максимальная тяга
  RECOVERING(3) — alpha снизился ниже alpha_crit, удерживаем защиту до alpha_exit
"""

from enum import IntEnum
import numpy as np
from dataclasses import dataclass


class AUAState(IntEnum):
    NORMAL     = 0
    WARNING    = 1
    CRITICAL   = 2
    RECOVERING = 3


# Цвета для визуализации (единообразно с s4)
AUA_COLORS = {
    AUAState.NORMAL:     "green",
    AUAState.WARNING:    "goldenrod",
    AUAState.CRITICAL:   "darkorange",
    AUAState.RECOVERING: "royalblue",
}

AUA_LABELS = {
    AUAState.NORMAL:     "НОРМ",
    AUAState.WARNING:    "ПРЕД",
    AUAState.CRITICAL:   "КРИТ",
    AUAState.RECOVERING: "ВОССТ",
}


class AUAOutput:
    """Выходные команды АУА за один шаг."""

    __slots__ = ("theta_ref", "thr_trim", "force_throttle", "state")

    def __init__(self, theta_ref: float, thr_trim: float,
                 force_throttle, state: AUAState):
        self.theta_ref = theta_ref
        self.thr_trim = thr_trim
        self.force_throttle = force_throttle   # None или float
        self.state = state


@dataclass
class AUAParams:
    """Параметры АУА. По умолчанию согласованы с AircraftParams."""
    enabled: bool = True

    # Пороги, рад
    alpha_warn: float = 0.2618   # ≈ 15°, предупреждение
    alpha_crit: float = 0.3491   # ≈ 20°, перехват управления
    alpha_exit: float = 0.1745   # ≈ 10°, выход из восстановления (гистерезис)

    # Команды при перехвате
    theta_recovery: float  = -0.1222   # ≈ -7°, нос вниз
    throttle_recovery: float = 1.0     # максимальная тяга

    # Мягкая коррекция при WARNING
    theta_warn_delta: float = -0.0524  # ≈ -3°, добавка к theta_ref


class AngleOfAttackProtector:
    """
    АУА — Автомат Углов Атаки.

    Пример включения в сценарий:
        from control.aua import AngleOfAttackProtector, AUAParams, AUAState

        aua_params = AUAParams(enabled=True,
                               alpha_warn=aircraft.alpha_warning,
                               alpha_crit=aircraft.alpha_crit)
        aua = AngleOfAttackProtector(aircraft, aua_params)

        # Внутри controls_fn:
        aua_out = aua.step(alpha_meas, theta_mission, thr_trim, dt)
        controller.set_pitch_setpoint(aua_out.theta_ref)
        controller.set_trim_throttle(aua_out.thr_trim)
        ctrl = controller.step(t, meas, dt)
        if aua_out.force_throttle is not None:
            ctrl[1] = aua_out.force_throttle   # прямая команда тяги, минуя h_Kp
    """

    def __init__(self, aircraft, params: AUAParams):
        self.enabled          = params.enabled
        self.alpha_warn       = params.alpha_warn
        self.alpha_crit       = params.alpha_crit
        self.alpha_exit       = params.alpha_exit
        self.theta_recovery   = params.theta_recovery
        self.throttle_recovery = params.throttle_recovery
        self.theta_warn_delta = params.theta_warn_delta
        self.state            = AUAState.NORMAL

    def reset(self):
        self.state = AUAState.NORMAL

    def step(self, alpha_meas: float, theta_ref: float,
             thr_trim: float, dt: float) -> AUAOutput:
        """
        Один шаг АУА.

        Args:
            alpha_meas:  измеренный УА, рад (от зонда или косвенной оценки)
            theta_ref:   уставка тангажа от миссии/контура высоты, рад
            thr_trim:    базовая тяга от миссии, о.е.
            dt:          шаг, сек (зарезервирован)

        Returns:
            AUAOutput
        """
        if not self.enabled:
            return AUAOutput(theta_ref, thr_trim, None, AUAState.NORMAL)

        # ── Автомат состояний ─────────────────────────────────────────────────
        if self.state in (AUAState.NORMAL, AUAState.WARNING):
            if alpha_meas >= self.alpha_crit:
                self.state = AUAState.CRITICAL
            elif alpha_meas >= self.alpha_warn:
                self.state = AUAState.WARNING
            else:
                self.state = AUAState.NORMAL

        elif self.state == AUAState.CRITICAL:
            if alpha_meas < self.alpha_crit:
                self.state = AUAState.RECOVERING

        else:  # RECOVERING
            if alpha_meas >= self.alpha_crit:
                self.state = AUAState.CRITICAL    # рецидив
            elif alpha_meas < self.alpha_exit:
                self.state = AUAState.NORMAL

        # ── Выходные команды ──────────────────────────────────────────────────
        if self.state == AUAState.NORMAL:
            return AUAOutput(theta_ref, thr_trim, None, self.state)

        if self.state == AUAState.WARNING:
            theta_out = np.clip(theta_ref + self.theta_warn_delta,
                                -np.pi / 2, np.pi / 2)
            return AUAOutput(theta_out, thr_trim, None, self.state)

        # CRITICAL or RECOVERING: полный перехват
        return AUAOutput(self.theta_recovery, self.throttle_recovery,
                         self.throttle_recovery, self.state)

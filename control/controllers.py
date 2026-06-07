"""
САУ (система автоматического управления) — каскадная ПИД-структура.

Три контура:
1. Внешний (theta-контур): стабилизация угла тангажа theta
2. Внутренний (q-контур): стабилизация угловой скорости q
3. Контур скорости (Va-контур): удержание воздушной скорости через тягу

Управляющие команды:
- delta_e (отклонение руля высоты), рад
- throttle (тяга/обороты), 0..1
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PIDParams:
    """Параметры ПИД контура."""
    Kp: float = 1.0    # пропорциональный коэффициент
    Ki: float = 0.0    # интегральный
    Kd: float = 0.0    # дифференциальный
    tau: float = 0.1   # фильтр производной (сек)
    integral_limit: float = 1e6  # ограничение интеграла


class PID:
    """
    ПИД-регулятор с фильтром производной и ограничением интеграла.

    Структура:
        y = Kp*e + Ki*integral(e) + Kd*d(e)/dt
    где d(e)/dt фильтруется (предотвращение noise amplification).
    """

    def __init__(self, params: PIDParams, name: str = "PID"):
        self.Kp = params.Kp
        self.Ki = params.Ki
        self.Kd = params.Kd
        self.tau = params.tau
        self.integral_limit = params.integral_limit
        self.name = name

        # Состояние
        self.integral = 0.0
        self.d_error_filtered = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Сброс состояния (при инициализации контура)."""
        self.integral = 0.0
        self.d_error_filtered = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        """
        Один шаг ПИД.

        Args:
            error: рассогласование (уставка - обратная связь)
            dt: шаг времени, сек

        Returns:
            y: выход регулятора
        """
        # Пропорциональная часть
        p_term = self.Kp * error

        # Интегральная часть (с ограничением)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.Ki * self.integral

        # Дифференциальная часть с фильтром
        if dt > 0:
            d_error = (error - self.prev_error) / dt
            # Фильтр первого порядка: d_error_filt = d_error_filt + (d_error - d_error_filt) * dt/tau
            alpha = dt / (self.tau + dt) if self.tau > 0 else 1.0
            self.d_error_filtered = self.d_error_filtered + (d_error - self.d_error_filtered) * alpha
        d_term = self.Kd * self.d_error_filtered

        self.prev_error = error
        return p_term + i_term + d_term


def saturation(value: float, min_val: float, max_val: float) -> float:
    """Ограничение значения диапазоном [min_val, max_val]."""
    return np.clip(value, min_val, max_val)


@dataclass
class PitchControlParams:
    """Параметры каскадного регулятора тангажа."""

    # Theta-контур (внешний)
    theta_Kp: float = 1.5      # пропорциональный
    theta_Ki: float = 0.1      # интегральный
    theta_Kd: float = 0.3      # дифференциальный
    theta_tau: float = 0.1     # фильтр производной, сек

    # Q-контур (внутренний, стабилизация угловой скорости)
    q_Kp: float = 0.5
    q_Ki: float = 0.05
    q_Kd: float = 0.1
    q_tau: float = 0.05

    # H-контур (управление высотой через скорость)
    h_Kp: float = 0.0        # управление тягой по высоте
    Va_ref: float = 30.0       # опорная скорость, м/с

    # Ограничения
    q_max: float = np.radians(60.0)    # макс желаемая угловая скорость, рад/с
    q_min: float = np.radians(-60.0)


class PitchController:
    """
    Каскадный контроллер тангажа (theta + q контуры).

    Структура:
        theta_ref (уставка тангажа)
          |
          v
        [PID_theta] -> q_ref (желаемая угловая скорость)
          |
          v (q_meas из гироскопа)
        [PID_q] -> delta_e (отклонение руля высоты)
          |
          v
        [Saturation] -> управляющая команда
    """

    def __init__(self, aircraft, params: PitchControlParams):
        """
        Args:
            aircraft: AircraftParams
            params: PitchControlParams
        """
        self.aircraft = aircraft

        # Theta-контур
        theta_p = PIDParams(
            Kp=params.theta_Kp,
            Ki=params.theta_Ki,
            Kd=params.theta_Kd,
            tau=params.theta_tau,
        )
        self.pid_theta = PID(theta_p, name="theta")

        # Q-контур
        q_p = PIDParams(
            Kp=params.q_Kp,
            Ki=params.q_Ki,
            Kd=params.q_Kd,
            tau=params.q_tau,
        )
        self.pid_q = PID(q_p, name="q")

        self.q_max = params.q_max
        self.q_min = params.q_min

        # Управление высотой через тягу
        self.h_Kp = params.h_Kp
        self.Va_ref = params.Va_ref

        # Уставки (будут переустановлены в start_maneuver)
        self.theta_ref = 0.0
        self.h_ref = 100.0  # высота удержания
        self.trim_throttle = 0.5  # дефолт, обновляется при инициализации

    def reset(self, state_measured: dict):
        """
        Инициализация контура на текущем состоянии.
        Используется при запуске для уставки начального режима.

        Args:
            state_measured: {'theta': theta_meas, 'q': q_meas, 'h': h_meas, ...}
        """
        theta = state_measured.get('theta', 0.0)
        h = state_measured.get('h', 100.0)
        self.theta_ref = theta  # Уставка = текущее состояние
        self.h_ref = h  # Высота удержания = текущая высота
        self.pid_theta.reset()
        self.pid_q.reset()

    def set_pitch_setpoint(self, theta_ref: float):
        """Установить уставку тангажа (рад)."""
        self.theta_ref = np.clip(theta_ref,
                                  np.radians(-30.0),
                                  np.radians(30.0))

    def set_trim_throttle(self, throttle: float):
        """Установить базовую тягу для режима уровня (поддержание высоты)."""
        self.trim_throttle = np.clip(throttle, self.aircraft.throttle_min,
                                     self.aircraft.throttle_max)

    def step(self, t: float, meas: dict, dt: float) -> np.ndarray:
        """
        Один шаг контроллера.

        Args:
            t: время, сек (для логирования)
            meas: словарь измеренных значений
                {
                    'q': q_meas,       # угловая скорость тангажа, рад/с
                    'theta': theta_meas,  # угол тангажа, рад
                    'h': h_meas,       # высота, м
                    'Va': Va_meas,     # воздушная скорость, м/с
                }
            dt: шаг времени, сек

        Returns:
            np.array([delta_e, throttle]): управляющие команды
        """
        theta_meas = meas.get('theta', 0.0)
        q_meas = meas.get('q', 0.0)
        h_meas = meas.get('h', self.h_ref)
        Va_meas = meas.get('Va', self.Va_ref)

        # Theta-контур: рассогласование по тангажу
        error_theta = self.theta_ref - theta_meas
        q_ref = self.pid_theta.step(error_theta, dt)

        # Ограничение желаемой угловой скорости
        q_ref = saturation(q_ref, self.q_min, self.q_max)

        # Q-контур: рассогласование по угловой скорости
        error_q = q_ref - q_meas
        delta_e_cmd = -self.pid_q.step(error_q, dt)  # ИНВЕРТИРОВАННЫЙ знак!

        # Насыщение рулевой команды
        delta_e = saturation(delta_e_cmd,
                            self.aircraft.delta_e_min,
                            self.aircraft.delta_e_max)

        # Управление высотой через тягу (простой P-контур)
        # При снижении высоты (h < h_ref) увеличиваем тягу
        error_h = self.h_ref - h_meas
        throttle_cmd = self.trim_throttle + self.h_Kp * error_h

        # Насыщение тяги
        throttle = saturation(throttle_cmd,
                             self.aircraft.throttle_min,
                             self.aircraft.throttle_max)

        return np.array([delta_e, throttle])

    def __call__(self, t: float, meas: dict, dt: float) -> np.ndarray:
        """Синоним для step() для удобства."""
        return self.step(t, meas, dt)


# ---------------------------------------------------------------------------
# Контур удержания воздушной скорости
# ---------------------------------------------------------------------------

@dataclass
class SpeedControlParams:
    """Параметры ПИД-регулятора воздушной скорости."""
    Va_Kp: float = 0.1    # пропорциональный
    Va_Ki: float = 0.01    # интегральный
    Va_Kd: float = 0.01     # дифференциальный
    Va_tau: float = 0.5    # фильтр производной, сек
    Va_integral_limit: float = 0.5  # ограничение интеграла


class SpeedController:
    """
    Контур удержания воздушной скорости: Va_ref → throttle.

    Структура:
        Va_ref (уставка скорости)
          |
          v
        [PID_Va] -> delta_throttle
          |
          v
        throttle = trim_throttle + delta_throttle
          |
          v
        [Saturation 0..1] -> команда тяги
    """

    def __init__(self, aircraft, params: SpeedControlParams):
        self.aircraft = aircraft

        pid_p = PIDParams(
            Kp=params.Va_Kp,
            Ki=params.Va_Ki,
            Kd=params.Va_Kd,
            tau=params.Va_tau,
            integral_limit=params.Va_integral_limit,
        )
        self.pid_Va = PID(pid_p, name="Va")

        self.Va_ref = 30.0
        self.trim_throttle = 0.5

    def reset(self):
        """Сброс интеграла ПИД. Уставка Va_ref не меняется."""
        self.pid_Va.reset()

    def set_Va_ref(self, Va_ref: float):
        """Установить уставку воздушной скорости (м/с)."""
        self.Va_ref = max(Va_ref, 1.0)

    def set_trim_throttle(self, throttle: float):
        """Установить базовую тягу (feedforward)."""
        self.trim_throttle = np.clip(throttle,
                                     self.aircraft.throttle_min,
                                     self.aircraft.throttle_max)

    def step(self, Va_meas: float, dt: float) -> float:
        """
        Один шаг контура скорости.

        Args:
            Va_meas: измеренная воздушная скорость, м/с
            dt: шаг времени, сек

        Returns:
            throttle: команда тяги 0..1
        """
        error_Va = self.Va_ref - Va_meas
        delta_throttle = self.pid_Va.step(error_Va, dt)
        throttle = saturation(
            self.trim_throttle + delta_throttle,
            self.aircraft.throttle_min,
            self.aircraft.throttle_max,
        )
        return throttle

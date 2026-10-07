"""
САУ (система автоматического управления) — каскадная ПИД-структура.

Продольный канал:
1. Внешний (theta-контур): стабилизация угла тангажа theta
2. Внутренний (q-контур): стабилизация угловой скорости q
3. Контур скорости (Va-контур): удержание воздушной скорости через тягу

Боковой канал (LateralController):
4. Курс chi -> уставка крена phi_ref
5. Крен phi (+ демпфирование по p) -> элероны delta_a
6. Скольжение beta -> руль направления delta_r (координированный разворот)

Управляющие команды:
- delta_e (отклонение руля высоты), рад
- throttle (тяга/обороты), 0..1
- delta_a (элероны), рад;  delta_r (руль направления), рад
"""

import numpy as np
from dataclasses import dataclass

from sim.state import PHI, P


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
    h_Kp: float = 1.0        # управление тягой по высоте
    Va_ref: float = 30.0     # расчётная скорость (точка настройки ПИД), м/с

    # Gain scheduling: масштабирование усиления q-контура по Va
    # delta_e *= (Va_ref / Va_meas)^2  — компенсирует падение эффективности руля
    gain_scheduling: bool = True

    # Ограничения
    q_max: float = np.radians(60.0)    # макс желаемая угловая скорость, рад/с
    q_min: float = np.radians(-60.0)
    # Диапазон масштабирования: не позволяем уйти слишком далеко от расчётной точки
    gs_scale_min: float = 0.25   # нижний предел scale (Va_meas >> Va_ref)
    gs_scale_max: float = 9.0    # верхний предел scale (Va_meas << Va_ref)


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

        # Gain scheduling
        self.gain_scheduling = params.gain_scheduling
        self.gs_scale_min = params.gs_scale_min
        self.gs_scale_max = params.gs_scale_max

        # Уставки (будут переустановлены в start_maneuver)
        self.theta_ref = 0.0
        self.q_ref = 0.0    # последняя уставка угловой скорости (для логирования)
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
        self.q_ref = q_ref

        # Q-контур: рассогласование по угловой скорости
        error_q = q_ref - q_meas
        delta_e_cmd = -self.pid_q.step(error_q, dt)  # ИНВЕРТИРОВАННЫЙ знак!

        # Gain scheduling: эффективность руля ∝ Va² → компенсируем обратно
        if self.gain_scheduling:
            Va_meas = meas.get('Va', self.Va_ref)
            Va_safe = max(Va_meas, 1.0)
            scale = np.clip((self.Va_ref / Va_safe) ** 2,
                            self.gs_scale_min, self.gs_scale_max)
            delta_e_cmd *= scale

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
    Va_Kp: float = 0.25    # пропорциональный
    Va_Ki: float = 0.05    # интегральный
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


# ---------------------------------------------------------------------------
# Боковой канал: курс -> крен -> элероны; скольжение -> руль направления
# ---------------------------------------------------------------------------

def wrap_angle(angle: float) -> float:
    """Привести угол к диапазону [−π, π) — для рассогласования по курсу."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class LateralControlParams:
    """
    Параметры боковой САУ (B&M гл. 6, последовательное замыкание контуров).

    Коэффициенты рассчитаны для параметров-аналога Aerosonde при Va = 30 м/с
    (расчёт — docs/control.md, раздел «Боковой канал»).
    """
    # Контур крена: delta_a = Kp·e_phi + Ki·∫e_phi − Kd·p
    phi_Kp: float = 0.83      # δa_max / e_phi_max = 25° / 30°
    phi_Ki: float = 0.05      # малый интеграл — снимает статическую ошибку крена
    phi_Kd: float = 0.10      # демпфирование по угловой скорости крена p (гироскоп)
    phi_integral_limit: float = 0.2   # рад·с

    # Контур курса: phi_ref = Kp·e_chi + Ki·∫e_chi
    chi_Kp: float = 4.5
    chi_Ki: float = 1.65
    chi_integral_limit: float = 0.1   # рад·с → вклад интеграла ≤ ~9.5° крена
    phi_ref_max: float = np.radians(30.0)  # ограничение уставки крена

    # Контур скольжения: delta_r = −(Kp·beta + Ki·∫beta)
    beta_hold: bool = True    # False — руль направления в нейтрали (δr = 0)
    # Ограничено шумом зонда: Kp=2 даёт дрожание руля ~1.6°/шаг при шуме β 0.6°
    beta_Kp: float = 0.5
    beta_Ki: float = 1.0
    beta_integral_limit: float = 0.2  # рад·с


class LateralController:
    """
    Боковая САУ: курс chi_ref -> (delta_a, delta_r).

    Структура:
        chi_ref (уставка курса)
          |   chi_meas (GPS: путевой угол)
          v
        [PI_chi] -> phi_ref  [Saturation ±phi_ref_max]
          |   phi_meas (ИНС), p_meas (гироскоп)
          v
        [PI_phi − Kd·p] -> delta_a  [Saturation ±delta_a_max]

        beta_meas (зонд: УС)
          v
        [PI_beta] -> delta_r  [Saturation ±delta_r_max]

    Знаки (docs/physics.md, раздел 2): delta_a > 0 — крен вправо,
    delta_r > 0 — нос влево; чтобы убрать beta > 0 (поток справа), нос
    доворачивается вправо, поэтому delta_r = −K·beta.
    """

    def __init__(self, aircraft, params: LateralControlParams):
        self.aircraft = aircraft
        self.params = params

        self.pid_chi = PID(PIDParams(Kp=params.chi_Kp, Ki=params.chi_Ki,
                                     integral_limit=params.chi_integral_limit), name="chi")
        self.pid_phi = PID(PIDParams(Kp=params.phi_Kp, Ki=params.phi_Ki,
                                     integral_limit=params.phi_integral_limit), name="phi")
        self.pid_beta = PID(PIDParams(Kp=params.beta_Kp, Ki=params.beta_Ki,
                                      integral_limit=params.beta_integral_limit), name="beta")

        self.chi_ref = 0.0
        self.phi_ref = 0.0   # последняя уставка крена (для логирования)

    def reset(self):
        """Сброс интегралов всех контуров."""
        self.pid_chi.reset()
        self.pid_phi.reset()
        self.pid_beta.reset()

    def set_course(self, chi_ref: float):
        """Установить уставку курса (путевого угла), рад."""
        self.chi_ref = wrap_angle(chi_ref)

    def step(self, meas: dict, dt: float) -> tuple:
        """
        Один шаг боковой САУ.

        Args:
            meas: {
                'chi':  путевой угол (GPS), рад
                'phi':  угол крена (ИНС), рад
                'p':    угловая скорость крена (гироскоп), рад/с
                'beta': УС (зонд), рад
            }
            dt: шаг времени, сек

        Returns:
            (delta_a, delta_r), рад
        """
        pr, ac = self.params, self.aircraft

        # Контур курса -> уставка крена (рассогласование по кратчайшему пути)
        e_chi = wrap_angle(self.chi_ref - meas['chi'])
        self.phi_ref = saturation(self.pid_chi.step(e_chi, dt),
                                  -pr.phi_ref_max, pr.phi_ref_max)

        # Контур крена -> элероны (демпфирование по измеренной p, а не по производной ошибки)
        e_phi = self.phi_ref - meas['phi']
        delta_a = saturation(self.pid_phi.step(e_phi, dt) - pr.phi_Kd * meas['p'],
                             -ac.delta_a_max, ac.delta_a_max)

        # Контур скольжения -> руль направления
        if pr.beta_hold:
            delta_r = saturation(-self.pid_beta.step(meas['beta'], dt),
                                 -ac.delta_r_max, ac.delta_r_max)
        else:
            delta_r = 0.0

        return delta_a, delta_r


# ---------------------------------------------------------------------------
# САУ по крену для продольных сценариев
# ---------------------------------------------------------------------------

@dataclass
class RollHoldParams:
    """
    Удержание крена φ_ref = 0 (крылья горизонтально).

    Kp, Kd — как в контуре крена LateralController. Интеграл больше: он
    парирует постоянный реактивный момент винта (на малой скорости и полном
    газе нужно до ~2° δa).
    """
    Kp: float = 0.83
    Ki: float = 0.3
    Kd: float = 0.10
    integral_limit: float = 0.2   # рад·с → вклад интеграла ≤ 0.06 рад ≈ 3.4° δa


class RollHold:
    """
    САУ по крену продольных сценариев: δa = PI(0 − φ) − Kd·p,  δr = 0.

    Зачем: реактивный момент винта кренит ЛА, спиральная мода неустойчива —
    без управления элеронами ЛА уходит в спираль за ~25 с (РЕШ-19).
    Датчики — ИНС (φ) и гироскоп (p) со своим генератором шума rng, чтобы не
    сдвигать последовательности шума продольных датчиков сценария.
    """

    def __init__(self, aircraft, sensor_params, rng,
                 params: RollHoldParams = None):
        self.aircraft = aircraft
        self.sp = sensor_params
        self.rng = rng
        self.params = params or RollHoldParams()
        pr = self.params
        self.pid_phi = PID(PIDParams(Kp=pr.Kp, Ki=pr.Ki,
                                     integral_limit=pr.integral_limit), name="phi_hold")

    def step(self, phi: float, p: float, dt: float) -> float:
        """Истинные φ, p → измерения → δa, рад."""
        sp, rng = self.sp, self.rng
        phi_meas = phi + rng.normal(0.0, sp.ins_angle_noise)
        p_meas   = p + sp.gyro_bias + rng.normal(0.0, sp.gyro_noise)
        delta_a = self.pid_phi.step(0.0 - phi_meas, dt) - self.params.Kd * p_meas
        return saturation(delta_a, -self.aircraft.delta_a_max, self.aircraft.delta_a_max)


def with_roll_hold(controls_fn, aircraft, sensor_params, dt: float, seed: int = 7):
    """
    Обернуть продольный controls_fn(t, state, Va, alpha) -> [δe, δt]
    удержанием крена: результат — [δe, δt, δa, 0].

    Для парных прогонов создавайте обёртку заново на каждый run() — тогда
    у обоих прогонов одинаковый шум ИНС/гироскопа крена (тот же seed).
    """
    hold = RollHold(aircraft, sensor_params, np.random.default_rng(seed))

    def fn(t, state, Va, alpha):
        c = np.asarray(controls_fn(t, state, Va, alpha), dtype=float)
        delta_a = hold.step(state[PHI], state[P], dt)
        return np.array([c[0], c[1], delta_a, 0.0])

    fn.roll_hold = hold
    return fn

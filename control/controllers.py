"""
САУ (система автоматического управления) — каскадная ПИД-структура.

Продольный канал:
1. Внешний (theta-контур): стабилизация угла тангажа theta
2. Внутренний (q-контур): стабилизация угловой скорости q
3. Контур скорости (Va-контур): удержание воздушной скорости через тягу

Боковой канал (LateralController, схема как в PX4 / ArduPilot):
4. Курс chi -> заданная скорость разворота -> уставка крена phi_ref
5. Крен (RollController): phi -> заданная скорость крена p_ref -> элероны delta_a
   (упреждение FF + ПИ по скорости крена)
6. Скольжение beta -> руль направления delta_r (координированный разворот)
   + подмешивание элеронов в РН против обратного рыскания

Управляющие команды:
- delta_e (отклонение руля высоты), рад
- throttle (тяга/обороты), 0..1
- delta_a (элероны), рад;  delta_r (руль направления), рад
"""

import numpy as np
from dataclasses import dataclass, field

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

    def unwind(self, error: float, dt: float):
        """
        Анти-виндап (условное интегрирование): отменить последнее накопление
        интеграла. Вызывается, когда выход регулятора упёрся в ограничение и
        ошибка тянет дальше в ту же сторону.
        """
        self.integral -= error * dt


def saturation(value: float, min_val: float, max_val: float) -> float:
    """Ограничение значения диапазоном [min_val, max_val]."""
    return np.clip(value, min_val, max_val)


@dataclass
class PitchControlParams:
    """Параметры каскадного регулятора тангажа."""

    # Theta-контур (внешний)
    theta_Kp: float = 16.20   # Kθ = kp_θ/kd_θ, 1/с (control/tuning.py)
    theta_Ki: float = 0.0      # интегральный
    theta_Kd: float = 0.0      # дифференциальный
    theta_tau: float = 0.1     # фильтр производной, сек

    # Q-контур (внутренний, стабилизация угловой скорости)
    q_Kp: float = 0.0381   # Kq = −kd_θ, с
    q_Ki: float = 0.0
    q_Kd: float = 0.0
    q_tau: float = 0.05

    # H-контур (управление высотой через скорость)
    # Газ по высоте: throttle = trim + h_Kp·(h_ref − h). По умолчанию выключен (0):
    # скорость держит SpeedController; при 1.0 газ дёргался от шума барометра
    # и обнулялся при наборе (h_ref контроллера не следует за заданием сценария)
    h_Kp: float = 0.0
    Va_ref: float = 16.0   # точка настройки — крейсер FPV, м/с
    # П-контур высоты для сценариев: theta_ref = alpha_trim + KH·(h_ref − h), рад/м
    KH: float = 0.0312   # ω_h = 0.5 рад/с: KH = ω_h / Va

    # Gain scheduling: масштабирование усиления q-контура по Va
    # delta_e *= (Va_ref / Va_meas)^2  — компенсирует падение эффективности руля
    gain_scheduling: bool = True

    # Ограничения
    # Предел q_ref = Kθ·40° (control/tuning.py): не отрезает руль — каскад ≡ ПД-закону.
    # При 60°/с давал δe не больше Kq·q_max ≈ 2° и ЛА не мог опустить нос.
    q_max: float = np.radians(648.0)   # макс желаемая угловая скорость, рад/с
    q_min: float = np.radians(-648.0)
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
        self.trim_elevator = 0.0  # упреждающий балансировочный δe (set_trim_elevator)

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

    def set_trim_elevator(self, delta_e: float):
        """
        Упреждающий балансировочный δe (из compute_trim): регулятор работает
        с отклонением от трима, как в линеаризации B&M — без просадки на старте.
        """
        self.trim_elevator = delta_e

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
        delta_e_cmd += self.trim_elevator

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
    Va_Kp: float = 0.106   # ωn_V = 1 рад/с, ζ = 0.8
    Va_Ki: float = 0.0868    # интегральный
    Va_Kd: float = 0.0     # дифференциальный
    Va_tau: float = 0.5    # фильтр производной, сек
    Va_integral_limit: float = 11.52  # = 1/Va_Ki — интеграл покрывает весь диапазон газа


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

        self.Va_ref = 16.0
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
class RollControlParams:
    """
    Регулятор крена по схеме PX4 / ArduPilot: угол → скорость крена → элероны.

        p_ref = (φ_ref − φ) / τ_φ,                |p_ref| ≤ p_max
        δa    = s·FF·p_ref + s²·(Kp·e_p + Ki·∫e_p),  e_p = p_ref − p,  s = Va_ref / Va

    FF — упреждение: δa, при котором установившаяся скорость крена равна p_ref
    (FF = a_φ1 / a_φ2); оно делает основную работу. ПИ по скорости крена только
    поправляет: P — против возмущений, I — против постоянного момента винта.
    Масштаб s: a_φ1 ∝ Va, a_φ2 ∝ Va², поэтому FF ∝ 1/Va, Kp, Ki ∝ 1/Va².
    Аналоги: PX4 FW_R_TC, FW_R_RMAX, FW_RR_FF, FW_RR_P, FW_RR_I;
    ArduPilot RLL2SRV_TCONST, RLL2SRV_RMAX, RLL_RATE_FF/P/I.
    Расчёт — control/tuning.py (FPV-самолёт, Va = 16 м/с).
    """
    tau_phi: float = 0.25            # с, постоянная времени крена (PX4 0.5, ArduPilot 0.45)
    p_max: float = np.radians(120.0)  # рад/с, предел p_ref (p_уст при δa = 20° ≈ 210 °/с)
    FF: float = 0.09438              # рад/(рад/с) = a_φ1 / a_φ2
    Kp: float = 0.03211              # контур p: полюс a_φ1 + a_φ2·Kp = 75 рад/с
    Ki: float = 0.1284               # нуль ПИ на 1/τ_φ
    integral_limit: float = 0.467    # рад → вклад интеграла ≤ 0.06 рад ≈ 3.4° δa
    Va_ref: float = 16.0             # м/с, скорость, на которой рассчитаны FF, Kp, Ki
    scale_min: float = 0.5           # пределы масштаба s = Va_ref / Va
    scale_max: float = 2.0


class RollController:
    """Крен φ_ref → δa (каскад угол → скорость крена). Состояние — только интеграл."""

    def __init__(self, aircraft, params: RollControlParams = None):
        self.aircraft = aircraft
        self.params = params or RollControlParams()
        pr = self.params
        self.pid_p = PID(PIDParams(Kp=pr.Kp, Ki=pr.Ki, integral_limit=pr.integral_limit),
                         name="p")
        self.p_ref = 0.0     # для логирования / окна стенда
        self.da_ff = 0.0     # упреждающая часть δa

    def reset(self):
        self.pid_p.reset()

    def step(self, phi_ref: float, phi: float, p: float, Va: float, dt: float) -> float:
        """Уставка крена и измерения (φ, p, Va) → δa, рад."""
        pr, ac = self.params, self.aircraft
        self.p_ref = saturation((phi_ref - phi) / pr.tau_phi, -pr.p_max, pr.p_max)
        s = saturation(pr.Va_ref / max(Va, 1.0), pr.scale_min, pr.scale_max)
        e_p = self.p_ref - p
        self.da_ff = s * pr.FF * self.p_ref
        da_raw = self.da_ff + s * s * self.pid_p.step(e_p, dt)
        delta_a = saturation(da_raw, -ac.delta_a_max, ac.delta_a_max)
        if da_raw != delta_a and e_p * da_raw > 0:
            self.pid_p.unwind(e_p, dt)          # элероны в упоре — интеграл не копим
        return delta_a


@dataclass
class LateralControlParams:
    """
    Параметры боковой САУ. Курс — П-контур через заданную скорость разворота
    (как L1/NPFG в PX4/ArduPilot: уставка крена из требуемого бокового ускорения):

        χ̇_ref = Vg·κ + K_chi·(χ_ref − χ),   φ_ref = atan(Vg·χ̇_ref / g),   |φ_ref| ≤ phi_ref_max

    κ — кривизна заданной траектории от навигатора (упреждение на дугах, как в L1/NPFG).

    Интеграла по курсу нет: на прямой φ = 0 при любом χ, статической ошибки нет.
    Коэффициенты рассчитаны для FPV-самолёта при Va = 16 м/с
    (control/tuning.py, docs/control.md раздел 11).
    """
    roll: RollControlParams = field(default_factory=RollControlParams)

    # Контур курса: χ/χ_ref = K/(τ_φ s² + s + K) → ζ = 1 при K = 1/(4·τ_φ)
    chi_K: float = 1.0                     # 1/с
    phi_ref_max: float = np.radians(30.0)  # ограничение уставки крена
    # Ограничение скорости изменения уставки крена (PX4 FW_PN_R_SLEW_MAX): без него
    # ступенька курса давала скачок δa ≈ 15° за шаг и рывок носа против разворота
    # (обратное рыскание элеронов, r ≈ −34 °/с)
    phi_ref_rate: float = np.radians(90.0)  # рад/с

    # Контур скольжения: delta_r = −(Kp·beta + Ki·∫beta)
    beta_hold: bool = True    # False — руль направления в нейтрали (δr = 0)
    # Ограничено шумом зонда: Kp=2 даёт дрожание руля ~1.6°/шаг при шуме β 0.6°
    beta_Kp: float = 0.5
    beta_Ki: float = 0.550   # ζ_β = 0.7
    beta_integral_limit: float = 0.2  # рад·с
    # Подмешивание элеронов в РН (ArduPilot KFF_RDDRMIX): δr += K·δa гасит момент
    # рыскания от элеронов сразу, не дожидаясь роста β. K = −N_δa / N_δr (tuning.py)
    da_dr_mix: float = -0.6536


class LateralController:
    """
    Боковая САУ: курс chi_ref -> (delta_a, delta_r).

    Структура:
        chi_ref (уставка курса)
          |   chi_meas (GPS: путевой угол), Vg (GPS: путевая скорость)
          v
        [K_chi] + Vg·κ -> χ̇_ref -> [atan(Vg·χ̇_ref/g)] -> [Saturation ±phi_ref_max]
          -> [Rate limit ±phi_ref_rate] -> phi_ref
          |   phi_meas (ИНС), p_meas (гироскоп), Va_meas (ПВД)
          v
        [RollController: 1/τ_φ -> p_ref -> FF + PI_p] -> delta_a  [Saturation ±delta_a_max]

        beta_meas (зонд: УС)
          v
        −[PI_beta] + da_dr_mix·delta_a -> delta_r  [Saturation ±delta_r_max]
        (beta_hold = False — руль направления в нейтрали, микс тоже выключен)

    Знаки (docs/physics.md, раздел 2): delta_a > 0 — крен вправо,
    delta_r > 0 — нос влево; чтобы убрать beta > 0 (поток справа), нос
    доворачивается вправо, поэтому delta_r = −K·beta.
    """

    def __init__(self, aircraft, params: LateralControlParams):
        self.aircraft = aircraft
        self.params = params

        self.roll = RollController(aircraft, params.roll)
        self.pid_beta = PID(PIDParams(Kp=params.beta_Kp, Ki=params.beta_Ki,
                                      integral_limit=params.beta_integral_limit), name="beta")

        self.chi_ref = 0.0
        self.kappa = 0.0     # кривизна заданной траектории, 1/м
        self.phi_ref = 0.0   # уставка крена после ограничителя скорости (его состояние)
        self.dr_mix = 0.0    # вклад микса элеронов в δr (для логирования)
        self._phi_init = True

    def reset(self):
        """Сброс интегралов; ограничитель уставки крена стартует с текущего крена."""
        self.roll.reset()
        self.pid_beta.reset()
        self._phi_init = True

    def set_course(self, chi_ref: float, kappa: float = 0.0):
        """
        Установить уставку курса (путевого угла), рад, и кривизну заданной
        траектории kappa, 1/м (>0 — разворот вправо; 0 — прямая или ручная уставка).
        Кривизна даёт упреждение χ̇ = Vg·kappa: на дуге радиуса R крен
        atan(Vg²/(g·R)) задаётся сразу, а не набирается ошибкой курса.
        """
        self.chi_ref = wrap_angle(chi_ref)
        self.kappa = kappa

    def step(self, meas: dict, dt: float) -> tuple:
        """
        Один шаг боковой САУ.

        Args:
            meas: {
                'chi':  путевой угол (GPS), рад
                'Vg':   путевая скорость (GPS), м/с
                'phi':  угол крена (ИНС), рад
                'p':    угловая скорость крена (гироскоп), рад/с
                'Va':   воздушная скорость (ПВД), м/с
                'beta': УС (зонд), рад
            }
            dt: шаг времени, сек

        Returns:
            (delta_a, delta_r), рад
        """
        pr, ac = self.params, self.aircraft

        # Контур курса -> уставка крена (рассогласование по кратчайшему пути)
        e_chi = wrap_angle(self.chi_ref - meas['chi'])
        chi_dot_ref = meas['Vg'] * self.kappa + pr.chi_K * e_chi
        phi_cmd = saturation(np.arctan(meas['Vg'] * chi_dot_ref / ac.g),
                             -pr.phi_ref_max, pr.phi_ref_max)
        if self._phi_init:                       # включение САУ — без скачка уставки
            self.phi_ref, self._phi_init = meas['phi'], False
        step = pr.phi_ref_rate * dt
        self.phi_ref += saturation(phi_cmd - self.phi_ref, -step, step)

        # Контур крена -> элероны
        delta_a = self.roll.step(self.phi_ref, meas['phi'], meas['p'], meas['Va'], dt)

        # Контур скольжения + микс элеронов -> руль направления
        if pr.beta_hold:
            self.dr_mix = pr.da_dr_mix * delta_a
            delta_r = saturation(-self.pid_beta.step(meas['beta'], dt) + self.dr_mix,
                                 -ac.delta_r_max, ac.delta_r_max)
        else:
            self.dr_mix, delta_r = 0.0, 0.0

        return delta_a, delta_r


# ---------------------------------------------------------------------------
# Контур высоты: ПИ h → θ_ref (B&M 6.4.3)
# ---------------------------------------------------------------------------

@dataclass
class AltitudeHoldParams:
    """
    θ_ref = α* + Kp·(h_ref − h) + Ki·∫(h_ref − h),  |θ_ref| ≤ theta_max.
    Интеграл убирает статическую ошибку высоты на скорости, отличной от точки
    балансировки (иной α*, иной δe*). Расчёт — control/tuning.py.
    """
    Kp: float = 0.09375    # рад/м, 2ζ_h·ω_h/Va, ω_h = 0.5 рад/с, ζ_h = 1.5
    Ki: float = 0.0156     # рад/(м·с), ω_h²/Va
    theta_max: float = np.radians(15.0)
    integral_limit: float = 16.76  # м·с = theta_max/Ki


class AltitudeHold:
    """ПИ-регулятор высоты с анти-виндапом (не копит интеграл, пока θ_ref в упоре)."""

    def __init__(self, params: AltitudeHoldParams = None, alpha_trim: float = 0.0):
        self.params = params or AltitudeHoldParams()
        pr = self.params
        self.pid = PID(PIDParams(Kp=pr.Kp, Ki=pr.Ki, integral_limit=pr.integral_limit),
                       name="h")
        self.alpha_trim = alpha_trim

    def reset(self):
        self.pid.reset()

    def step(self, h_ref: float, h_meas: float, dt: float) -> float:
        """→ θ_ref, рад."""
        e_h = h_ref - h_meas
        raw = self.alpha_trim + self.pid.step(e_h, dt)
        th_max = self.params.theta_max
        theta_ref = saturation(raw, -th_max, th_max)
        if raw != theta_ref and e_h * (raw - self.alpha_trim) > 0:
            self.pid.unwind(e_h, dt)
        return theta_ref


# ---------------------------------------------------------------------------
# САУ по крену для продольных сценариев
# ---------------------------------------------------------------------------

class RollHold:
    """
    САУ по крену продольных сценариев: φ_ref = 0 → RollController → δa,  δr = 0.

    Зачем: реактивный момент винта кренит ЛА, спиральная мода неустойчива —
    без управления элеронами ЛА уходит в спираль за ~25 с (РЕШ-19).
    Коэффициенты — те же, что у контура крена LateralController (RollControlParams);
    интеграл по скорости крена парирует момент винта.
    Датчики — ИНС (φ) и гироскоп (p) со своим генератором шума rng, чтобы не
    сдвигать последовательности шума продольных датчиков сценария; Va — истинная
    (масштаб коэффициентов, шум ПВД здесь не важен).
    """

    def __init__(self, aircraft, sensor_params, rng,
                 params: RollControlParams = None):
        self.sp = sensor_params
        self.rng = rng
        self.roll = RollController(aircraft, params)

    def reset(self):
        self.roll.reset()

    def step(self, phi: float, p: float, Va: float, dt: float) -> float:
        """Истинные φ, p, Va → измерения → δa, рад."""
        sp, rng = self.sp, self.rng
        phi_meas = phi + rng.normal(0.0, sp.ins_angle_noise)
        p_meas   = p + sp.gyro_bias + rng.normal(0.0, sp.gyro_noise)
        return self.roll.step(0.0, phi_meas, p_meas, Va, dt)


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
        delta_a = hold.step(state[PHI], state[P], Va, dt)
        return np.array([c[0], c[1], delta_a, 0.0])

    fn.roll_hold = hold
    return fn

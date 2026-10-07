"""Параметры симулятора. Все числа физики живут здесь."""

from dataclasses import dataclass


@dataclass
class AircraftParams:
    """
    Параметры ЛА — канонический Aerosonde (Beard & McLain, 2-е изд., 2024–;
    официальный файл авторов mavsim_public/mavsim_python/parameters/
    aerosonde_parameters.py). Решение автора 2026-10-07 (РЕШ-18): берём набор
    целиком, m = 11 кг. Параметры-аналог, не измерены на изделии автора.
    """

    # --- Масса и инерция ---
    mass: float = 11.0     # кг
    Jx:   float = 0.8244   # кг·м², момент инерции крена
    Jy:   float = 1.135    # кг·м², момент инерции тангажа
    Jz:   float = 1.759    # кг·м², момент инерции рыскания
    Jxz:  float = 0.1204   # кг·м², центробежный момент

    # --- Геометрия крыла ---
    S:    float = 0.55     # м², площадь крыла
    b:    float = 2.8956   # м, размах
    c:    float = 0.18994  # м, средняя аэродинамическая хорда

    # --- Среда ---
    rho:  float = 1.2682   # кг/м³
    g:    float = 9.81     # м/с²

    # --- Аэродинамика: продольный канал ---
    # CL = CL(α) + CLq·q̂ + CLde·δe,  q̂ = c·q/(2Va)
    CL0:  float = 0.23
    CLa:  float = 5.61
    CLq:  float = 7.95
    CLde: float = 0.13

    # CD = CDp + (CL0 + CLa·α)²/(π·e·AR) + CDq·q̂ + CDde·δe   (B&M, слайды 20, 43)
    CDp:  float = 0.043    # вредное сопротивление
    e_oswald: float = 0.9  # коэффициент Освальда
    CDq:  float = 0.0
    CDde: float = 0.0135

    # Cm = Cm0 + Cma·α + Cmq·q̂ + Cmde·δe
    Cm0:  float = 0.0135
    Cma:  float = -2.74    # < 0 — продольная статическая устойчивость
    Cmq:  float = -38.21   # демпфирование тангажа
    Cmde: float = -0.99    # δe > 0 — нос вниз

    # --- Аэродинамика: боковой канал ---
    # C* = C*0 + C*_beta·β + C*_p·p̂ + C*_r·r̂ + C*_da·δa + C*_dr·δr,
    # p̂ = b·p/(2Va), r̂ = b·r/(2Va).
    # Croll — момент крена (в B&M C_l; переименован, чтобы не путать с Cl подъёмной силы)
    # Боковая сила
    CY0:     float = 0.0
    CY_beta: float = -0.98
    CY_p:    float = 0.0
    CY_r:    float = 0.0
    CY_da:   float = 0.075
    CY_dr:   float = 0.19     # > 0, совпадает со знаком РЕШ-15
    # Момент крена
    Croll0:     float = 0.0
    Croll_beta: float = -0.13   # < 0 — поперечная устойчивость
    Croll_p:    float = -0.51   # < 0 — демпфирование крена
    Croll_r:    float = 0.25
    Croll_da:   float = 0.17    # эффективность элеронов
    Croll_dr:   float = 0.0024
    # Момент рыскания
    Cn0:     float = 0.0
    Cn_beta: float = 0.073      # > 0 — путевая устойчивость
    Cn_p:    float = 0.069
    Cn_r:    float = -0.095     # < 0 — демпфирование рыскания
    Cn_da:   float = -0.011
    Cn_dr:   float = -0.069     # эффективность руля направления

    # Нелинейная модель CL (sigmoid-смешение, B&M ур. 4.9–4.10)
    # CL = (1 - sigma)*linear + sigma*flat_plate
    M_sigmoid: float = 50.0     # крутизна перехода
    alpha_stall: float = 0.47   # рад ≈ 27°, угол срыва потока

    # --- Силовая установка: электромотор + винт (B&M 2-е изд., разд. 4.3, слайды 36–37) ---
    D_prop:   float = 0.508     # м, диаметр винта (20 дюймов)
    KV_rpm:   float = 145.0     # об/мин/В, скоростная постоянная мотора
    R_motor:  float = 0.042     # Ом, сопротивление обмотки
    i0:       float = 1.5       # А, ток холостого хода
    V_max:    float = 44.4      # В, батарея 12S (3.7 В × 12)
    # Аппроксимации CT(J), CQ(J) = C2·J² + C1·J + C0 по данным продувки винта
    C_T2: float = -0.1079
    C_T1: float = -0.06044
    C_T0: float = 0.09357
    C_Q2: float = -0.01664
    C_Q1: float = 0.004970
    C_Q0: float = 0.005230

    # --- Критические углы атаки ---
    alpha_warning: float = 0.2618   # рад ≈ 15°, предупреждение [АВТОР]
    alpha_crit:    float = 0.3491   # рад ≈ 20°, критический УА  [АВТОР]

    # --- Ограничения рулей ---
    delta_e_max: float =  0.4363   # рад ≈ +25°
    delta_e_min: float = -0.4363   # рад ≈ -25°
    delta_a_max: float = 0.4363    # рад ≈ ±25°, элероны
    delta_r_max: float = 0.4363    # рад ≈ ±25°, руль направления
    throttle_max: float = 1.0
    throttle_min: float = 0.0


@dataclass
class WindParams:
    """Параметры модели ветра."""

    Vw_const:    float = 0.0    # м/с, постоянный ветер на север (при ψ=0 > 0 = попутный)
    Vw_cross:    float = 0.0    # м/с, постоянный ветер на восток (при ψ=0 > 0 = слева направо)
    h_shear_lo:  float = 50.0   # м, нижняя граница слоя сдвига
    h_shear_hi:  float = 100.0  # м, верхняя граница
    dV_shear:    float = 0.0    # м/с, перепад скорости через слой
    gust_amp:    float = 0.0    # м/с, амплитуда горизонтального порыва (> 0 = попутный)
    gust_vwh:    float = 0.0    # м/с, амплитуда вертикального порыва (> 0 = восходящий)
    gust_vwy:    float = 0.0    # м/с, амплитуда бокового порыва (> 0 = на восток)
    gust_t0:     float = 9999.0 # с, момент начала порыва
    gust_dur:    float = 1.0    # с, длительность порыва


@dataclass
class SensorParams:
    """Шумы и смещения датчиков. Источник: B&M прил. H, если не указано иное."""

    # Гироскоп (угловая скорость тангажа q)
    gyro_noise: float = 0.002   # рад/с, СКО (ADXRS-450)
    gyro_bias:  float = 0.0     # рад/с

    # Акселерометр
    accel_noise: float = 0.025  # м/с² (ADXL-325, ~0.0025g)
    accel_bias:  float = 0.0

    # Барометр (высота)
    baro_noise: float = 0.5     # м
    baro_bias:  float = 0.0

    # СВС — датчик воздушной скорости
    airspeed_noise: float = 0.2 # м/с
    airspeed_bias:  float = 0.0

    # Зонд УА (прямое измерение alpha)
    probe_noise: float = 0.01   # рад ≈ 0.6°  [АВТОР: характеристика изделия]
    probe_bias:  float = 0.0    # рад          [АВТОР]
    # Зонд УС (прямое измерение beta тем же зондом — АУ: УА и УС)
    probe_beta_noise: float = 0.01   # рад ≈ 0.6°  [АВТОР: характеристика изделия]
    probe_beta_bias:  float = 0.0    # рад          [АВТОР]

    # ИНС — углы ориентации (крен, тангаж, рыскание)
    ins_angle_noise: float = 0.002   # рад  [МОДЕЛЬ: как шум theta в продольных сценариях]

    # GPS (координаты и скорость)
    gps_pos_noise: float = 1.0  # м
    gps_vel_noise: float = 0.1  # м/с


@dataclass
class SimConfig:
    """Настройки прогона."""

    dt:    float = 0.01     # с, шаг интегрирования
    t_end: float = 60.0     # с, длительность

    # Начальные условия
    Va0:    float = 30.0    # м/с, начальная воздушная скорость
    h0:     float = 100.0   # м, начальная высота
    theta0: float = 0.0     # рад

    # Вывод
    show_plots: bool = True
    save_plots: bool = False
    output_dir: str = "results"


def default_params():
    """Вернуть конфигурацию по умолчанию (аналог Aerosonde)."""
    return AircraftParams(), WindParams(), SensorParams(), SimConfig()

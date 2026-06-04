"""Параметры симулятора. Все числа физики живут здесь."""

from dataclasses import dataclass


@dataclass
class AircraftParams:
    """
    Параметры ЛА.
    АНАЛОГ: Aerosonde (Beard & McLain, прил. E). Подлежат замене авторскими данными.
    """

    # --- Масса и инерция ---
    mass: float = 13.5      # кг                            [АВТОР]
    Jy:   float = 1.135     # кг·м², момент инерции тангажа [АВТОР]

    # --- Геометрия крыла ---
    S:    float = 0.55      # м², площадь крыла             [АВТОР]
    b:    float = 2.8956    # м, размах                     [АВТОР]
    c:    float = 0.18994   # м, средняя хорда              [АВТОР]

    # --- Среда ---
    rho:  float = 1.2682    # кг/м³, стандартная атмосфера ~100 м
    g:    float = 9.81      # м/с²

    # --- Аэродинамика: продольный канал (линейная модель) ---
    # Коэффициент подъёмной силы: CL = CL0 + CLa*alpha + CLq*(c/2Va)*q + CLde*delta_e
    CL0:  float = 0.28      # при alpha=0, q=0, delta_e=0
    CLa:  float = 3.45      # производная по УА
    CLq:  float = 0.0       # производная по угловой скорости тангажа
    CLde: float = -0.36     # производная по рулю высоты

    # Коэффициент сопротивления: квадратичная модель
    CDp:  float = 0.0437    # вредное сопротивление (при нулевой подъёмной силе)
    e_oswald: float = 0.9   # КПД Освальда

    # Коэффициент момента тангажа: Cm = Cm0 + Cma*alpha + Cmq*(c/2Va)*q + Cmde*delta_e
    Cm0:  float = -0.02338  # при alpha=0 (< 0 — пикирующий момент)
    Cma:  float = -0.38     # производная по УА (< 0 — продольная устойчивость)
    Cmq:  float = -3.6      # демпирование тангажа
    Cmde: float = -0.5      # эффективность руля высоты

    # Нелинейная модель CL (sigmoid-смешение, B&M ур. 4.9–4.10)
    # CL = (1 - sigma)*linear + sigma*flat_plate
    M_sigmoid: float = 50.0   # крутизна перехода
    alpha_stall: float = 0.4712  # рад ≈ 27°, угол срыва потока

    # --- Силовая установка ---
    k_motor:  float = 80.0   # коэффициент двигателя        [АВТОР]
    S_prop:   float = 0.2027  # м², площадь диска винта
    C_prop:   float = 1.0     # коэффициент тяги

    # --- Критические углы атаки ---
    alpha_warning: float = 0.2618   # рад ≈ 15°, предупреждение [АВТОР]
    alpha_crit:    float = 0.3491   # рад ≈ 20°, критический УА  [АВТОР]

    # --- Ограничения рулей ---
    delta_e_max: float =  0.4363   # рад ≈ +25°
    delta_e_min: float = -0.4363   # рад ≈ -25°
    throttle_max: float = 1.0
    throttle_min: float = 0.0


@dataclass
class WindParams:
    """Параметры модели ветра."""

    Vw_const:    float = 0.0    # м/с, постоянный горизонтальный ветер (> 0 = попутный)
    h_shear_lo:  float = 50.0   # м, нижняя граница слоя сдвига
    h_shear_hi:  float = 100.0  # м, верхняя граница
    dV_shear:    float = 0.0    # м/с, перепад скорости через слой
    gust_amp:    float = 0.0    # м/с, амплитуда порыва
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

    # GPS (координаты и скорость)
    gps_pos_noise: float = 1.0  # м
    gps_vel_noise: float = 0.1  # м/с


@dataclass
class SimConfig:
    """Настройки прогона."""

    dt:    float = 0.01     # с, шаг интегрирования
    t_end: float = 30.0     # с, длительность

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

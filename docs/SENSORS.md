# Псевдодатчики

Датчики моделируются чистыми функциями `control/sensors.py`: истинное значение +
смещение (bias) + гауссов шум. Физика (`runner.run`) работает с истинным состоянием;
измерения получает только САУ. Параметры шума — `sim.config.SensorParams`
(источник — B&M прил. H, если не указано иное; поля `[АВТОР]` — характеристики изделия).

## Функции

| Функция | Что измеряет | Параметры (`SensorParams`) |
|---|---|---|
| `measure_gyro` | угловая скорость по одной оси: p, q или r | `gyro_noise` = 0.002 рад/с, `gyro_bias` |
| `measure_attitude` | угол ИНС: φ, θ или ψ | `ins_angle_noise` = 0.002 рад |
| `measure_altitude` | высота (барометр) | `baro_noise` = 0.5 м, `baro_bias` |
| `measure_airspeed` | воздушная скорость (ПВД, СВС) | `airspeed_noise` = 0.2 м/с, `airspeed_bias` |
| `measure_angle_of_attack` | УА α (зонд) | `probe_noise` = 0.01 рад, `probe_bias` [АВТОР] |
| `measure_sideslip` | УС β (тот же зонд) | `probe_beta_noise` = 0.01 рад, `probe_beta_bias` [АВТОР] |
| `measure_gps_course` | путевой угол χ = atan2(Vy, Vx) по земной скорости | `gps_vel_noise` = 0.1 м/с (на составляющие) |
| `measure_gps_velocity_earth` | земная скорость (Vx, Vh), продольная версия | `gps_vel_noise` |

Координаты GPS (шум `gps_pos_noise` = 1 м) и вертикальная скорость GPS добавляются
прямо в `NoisySensors` (ниже). Путевой угол χ при ветре отличается от курса ψ на угол
сноса — поэтому для удержания линии пути САУ использует χ (GPS), а не ψ (ИНС).

## Наборы датчиков для САУ (`control/sau.py`)

| Класс | Где | Что даёт |
|---|---|---|
| `NoisySensors(sp, rng, gps_pos, alpha_probe, gps_vz)` | сценарии s8–s10 | h — барометр, Va — ПВД, q, p — гироскоп, θ, φ — ИНС, χ — GPS, β — зонд; по флагам: координаты GPS, α — зонд, ḣ — GPS (нужна TECS) |
| `TruthSensors()` | 3D-тренажёр, стенд, конструктор | те же ключи, истинные значения |

Порядок вызовов генератора `rng` в `NoisySensors` закреплён: необязательные измерения —
в конце, чтобы прогоны с разными флагами имели одинаковый шум общих датчиков.

## Использование вне `SAU`

```python
sp  = SensorParams(gyro_noise=0.005)            # свои параметры шума
rng = np.random.default_rng(seed=42)            # воспроизводимость — через seed

def controls_fn(t, state, Va, alpha):
    q_meas = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    h_meas = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    ...
```

Измерения в `Log` не попадают (там только истинные значения); при необходимости
сценарий собирает их в буферы и передаёт в `FlightLogger.save(...)`.

## Чего нет

Дрейфа гироскопа, частоты обновления GPS, задержек и оценки состояния (фильтра) пока
нет — дорожная карта, блок 5 (docs/DEV_PLAN.md).

## Этап 1: Псевдодатчики

### Что сделано

Реализованы чистые функции для моделирования шума датчиков:

**sensors.py:**
- `measure_gyro()` — гироскоп (угловая скорость тангажа q)
- `measure_altitude()` — барометр (высота h)
- `measure_airspeed()` — СВС (воздушная скорость Va)
- `measure_angle_of_attack()` — зонд УА (угол атаки alpha)
- `measure_gps_position()` — GPS координаты
- `measure_gps_velocity_earth()` — GPS земная скорость
- `measure_accelerometer()` — акселерометр

Каждая функция: истинное значение + смещение (bias) + гауссов шум

### Параметры датчиков (config.py)

```python
class SensorParams:
    gyro_noise: float = 0.002          # рад/с, гироскоп
    gyro_bias:  float = 0.0            # смещение

    accel_noise: float = 0.025         # м/с², акселерометр
    accel_bias:  float = 0.0

    baro_noise: float = 0.5            # м, барометр
    baro_bias:  float = 0.0

    airspeed_noise: float = 0.2        # м/с, СВС
    airspeed_bias:  float = 0.0

    probe_noise: float = 0.01          # рад ≈ 0.6°, зонд УА
    probe_bias:  float = 0.0           # [АВТОР: уточнить]

    gps_pos_noise: float = 1.0         # м, GPS позиция
    gps_vel_noise: float = 0.1         # м/с, GPS скорость
```

### Использование

Датчики — чистые функции; сценарий вызывает их сам внутри `controls_fn`
(в `run()` датчиков нет — физика получает истинное состояние):

```python
sp  = SensorParams(gyro_noise=0.005, baro_noise=0.0)   # свои параметры шума
rng = np.random.default_rng(seed=42)                   # воспроизводимость — через seed rng

def controls_fn(t, state, Va, alpha):
    q_meas = measure_gyro(state[Q], sp.gyro_bias, sp.gyro_noise, rng)
    h_meas = measure_altitude(state[H], sp.baro_bias, sp.baro_noise, rng)
    ...
```

### Физическая мотивация

Независимые seed'ы для шума в парном прогоне "с зондом vs без":
- В натурных испытаниях броуновское движение молекул воздуха в каждом датчике независимо
- Синхронизированный шум — математическая условность
- `feedback_noise.md`: используем разные seed'ы для честного сравнения

### Log структура

`Log` хранит только истинные значения (`log.state`, `log.Va`, `log.alpha`,
`log.beta`). Измерения датчиков в `Log` не попадают — сценарий собирает их в
свои буферы и при необходимости передаёт в `FlightLogger.save(...)`
(`alpha_probe`, `alpha_est`, ...).

### Следующий этап (исторический текст)

**Управление (control.py):**
- Каскадная САУ из 3 ПИД-контуров
- Управление получает ИЗМЕРЕННЫЕ значения (от датчиков)
- Требование 6.4 ТЗ: парный прогон "с зондом vs без" должен быть идентичен в управлении

---

**Примечание:** Значения параметров — аналоги Aerosonde (B&M прил. H), помечены как [МОДЕЛЬ].
Автор заменит на реальные характеристики своего БПЛА.

---

## Этап 3 (Б3): датчики бокового канала

| Функция | Что измеряет | Параметры шума (SensorParams) |
|---|---|---|
| `measure_gyro` | p, q или r (одна ось) | `gyro_noise`, `gyro_bias` |
| `measure_attitude` | угол ИНС: φ, θ или ψ | `ins_angle_noise` = 0.002 рад [МОДЕЛЬ] |
| `measure_sideslip` | УС β — **тот же зонд**, что измеряет УА | `probe_beta_noise` = 0.01 рад, `probe_beta_bias` [АВТОР] |
| `measure_gps_course` | путевой угол χ = arctan(Vy/Vx) по земной скорости | `gps_vel_noise` (шум на составляющие скорости) |

Путевой угол χ при ветре отличается от курса ψ на угол сноса — поэтому для
удержания линии пути САУ использует χ (GPS), а не ψ (ИНС).

Истинная земная скорость для GPS — `sim.state.earth_velocity(state)` → (Vx, Vy, Vh).

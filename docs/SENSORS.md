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

1. **Включить/отключить датчики:**
   ```python
   cfg = SimConfig(enable_sensors=True)  # или False
   ```

2. **Задать свои параметры шума:**
   ```python
   sensors = SensorParams(
       gyro_noise=0.005,      # увеличить шум гироскопа
       probe_noise=0.02,      # шум зонда УА
       baro_noise=0.0,        # отключить шум барометра
   )
   log = run(..., sensor_params=sensors)
   ```

3. **Воспроизводимость:** seed в SimConfig
   ```python
   cfg = SimConfig(sensor_seed=42)  # одинаковый шум между прогонами
   ```

### Физическая мотивация

Независимые seed'ы для шума в парном прогоне "с зондом vs без":
- В натурных испытаниях броуновское движение молекул воздуха в каждом датчике независимо
- Синхронизированный шум — математическая условность
- `feedback_noise.md`: используем разные seed'ы для честного сравнения

### Log структура

После прогона лог содержит:
- `log.state`, `log.Va`, `log.alpha` — истинные значения
- `log.q_meas`, `log.h_meas`, `log.Va_meas`, `log.alpha_meas` — измеренные

### Следующий этап

**Управление (control.py):**
- Каскадная САУ из 3 ПИД-контуров
- Управление получает ИЗМЕРЕННЫЕ значения (от датчиков)
- Требование 6.4 ТЗ: парный прогон "с зондом vs без" должен быть идентичен в управлении

---

**Примечание:** Значения параметров — аналоги Aerosonde (B&M прил. H), помечены как [МОДЕЛЬ].
Автор заменит на реальные характеристики своего БПЛА.

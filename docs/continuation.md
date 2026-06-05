# Передача контекста — что сделано и что делать дальше

Этот файл написан для агента, который продолжает разработку.  
Читай в паре с `CLAUDE.md` (правила проекта) и `TZ_simulator.md` (техзадание).

---

## 1. Текущий статус (все тесты зелёные)

```
python check.py   →  ALL CHECKS PASSED  (24/24)
python demo.py    →  4 графика, тримовый полёт 20 с
python animate.py →  анимация полёта
```

### Что реализовано

| Файл | Статус | Описание |
|------|--------|----------|
| `config.py` | ✅ Готов | Параметры Aerosonde, dataclasses |
| `state.py` | ✅ Готов | Вектор [u,w,q,θ,x,h], `air_velocity`, `total_energy` |
| `aero.py` | ✅ Готов | CL (sigmoid+срыв), CD (квадратичный), Cm, силы |
| `wind.py` | ✅ Готов | Постоянный + сдвиг по высоте + порыв |
| `dynamics.py` | ✅ Готов | `derivatives()` — чистая функция ОДУ |
| `integrators.py` | ✅ Готов | `step_euler`, `step_rk4` |
| `runner.py` | ✅ Готов | `run()`, `Log`, `compute_trim()`, `trim_state()` |
| `plotting.py` | ✅ Готов | 4 функции графиков (русские подписи) |
| `animate.py` | ✅ Готов | Анимация по логу (изолирована) |
| `check.py` | ✅ Готов | 24 теста здравого смысла |
| `demo.py` | ✅ Готов | Точка входа, тримовый прогон |

### Чего ещё нет (нужно для Этапа 1 и 2)

| Файл | Приоритет | Описание |
|------|-----------|----------|
| `control.py` | 🔴 Критично | ПИД-каскад, без него нет стабильного полёта |
| `sensors.py` | 🔴 Критично | Псевдодатчики с шумом и смещением |
| `estimators.py` | 🟡 Важно | Косвенная оценка УА: `alpha_est = theta - gamma_GPS` |
| `scenarios.py` | 🟡 Важно | Уставки для сценариев С1–С6 |

---

## 2. Следующий шаг: `control.py`

### Архитектура каскада (3 контура)

```
Уставка h_cmd, Va_cmd
    │
    ▼
[Контур высоты]  ─── theta_cmd ───▶ [Контур тангажа] ─── q_cmd ───▶ [Контур q]
   PID(h)                              PID(theta, alpha)                PID(q)
                                                                          │
                                                                     delta_e
[Контур скорости]
   PID(Va) ─── throttle
```

### Класс PID

```python
class PID:
    def __init__(self, kp, ki, kd, i_limit=None):
        self._integrator = 0.0
        self._prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self._integrator += error * dt
        if self._i_limit:
            self._integrator = np.clip(self._integrator, -self._i_limit, self._i_limit)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp*error + self.ki*self._integrator + self.kd*derivative
```

### Сигнатура каскада

```python
def cascade(sensors, alpha_src: float, setpoints, pids: dict, dt: float) -> np.ndarray:
    """
    sensors   : SensorData (из sensors.py)
    alpha_src : alpha_probe ИЛИ alpha_est — единственное различие в парном прогоне
    setpoints : (h_cmd, Va_cmd, theta_cmd или None)
    pids      : словарь {'rate': PID, 'pitch': PID, 'alt': PID, 'speed': PID}
    Возвращает: np.array([delta_e, throttle])
    """
```

### Знаковое соглашение (критично!)

В модели Aerosonde `Cmde = -0.5`:
- **Положительный δe → момент вниз (пикирование)**
- **Отрицательный δe → момент вверх (кабрирование)**

При тримовом полёте Va=30 м/с: `δe_trim ≈ -0.078 рад ≈ -4.5°` при `α_trim ≈ 0.041 рад`.  
Это уже проверено в `runner.compute_trim()`.

Следствие для ПИД: ошибка по тангажу `e_theta = theta_cmd - theta_meas`:
- При `e_theta > 0` (нужно увеличить тангаж) → ПИД должен давать **отрицательный** δe.
- `kp_pitch < 0` или инвертировать ошибку.

---

## 3. Следующий шаг: `sensors.py`

### Структура SensorData

```python
@dataclass
class SensorData:
    theta:       float   # угол тангажа от ИНС (шум gyro), рад
    q:           float   # угловая скорость тангажа (шум gyro), рад/с
    gamma_gps:   float   # угол наклона траектории от GPS (из земных скоростей), рад
    Va_meas:     float   # воздушная скорость от СВС, м/с
    h_meas:      float   # высота от барометра, м
    alpha_probe: float   # УА от зонда (путь "с зондом"), рад
```

### Шаблон функции

```python
def measure_all(state: np.ndarray, Va_true: float, alpha_true: float,
                params: SensorParams, rng: np.random.Generator) -> SensorData:
    noise = lambda sigma: rng.normal(0.0, sigma)
    return SensorData(
        theta       = state[THETA] + noise(params.gyro_noise) + params.gyro_bias,
        q           = state[Q]     + noise(params.gyro_noise),
        gamma_gps   = kinematic_gamma(state) + noise(params.gps_vel_noise),
        Va_meas     = Va_true      + noise(params.airspeed_noise) + params.airspeed_bias,
        h_meas      = state[H]     + noise(params.baro_noise)  + params.baro_bias,
        alpha_probe = alpha_true   + noise(params.probe_noise) + params.probe_bias,
    )
```

---

## 4. Следующий шаг: `estimators.py`

```python
def estimate_kinematic(theta_meas: float, gamma_gps: float) -> float:
    """
    Косвенная оценка УА: alpha_est = theta - gamma_GPS.
    
    МОДЕЛЬ, не эксперимент. При ветре gamma_GPS считается по земной скорости,
    истинный УА — по воздушной. Расхождение = ветровая составляющая.
    Это и есть "слепота" косвенной оценки — центральный тезис работы.
    """
    return theta_meas - gamma_gps
```

---

## 5. Парный прогон (С6 — кульминация)

### Критически важное требование (TZ п. 6.4)

Оба прогона **идентичны** во всём, кроме источника α:

```python
def run_paired(scenario_fn, aircraft, wind_params, cfg):
    # Два НЕЗАВИСИМЫХ генератора шума — физически корректно
    # (броуновское движение молекул около разных датчиков независимо)
    rng_probe    = np.random.default_rng(seed=42)
    rng_noprobe  = np.random.default_rng(seed=137)

    def ctrl_probe(t, state, Va, alpha):
        sensors = measure_all(state, Va, alpha, cfg.sensors, rng_probe)
        alpha_src = sensors.alpha_probe           # прямое измерение
        return cascade(sensors, alpha_src, setpoints(t), pids_probe, cfg.dt)

    def ctrl_noprobe(t, state, Va, alpha):
        sensors = measure_all(state, Va, alpha, cfg.sensors, rng_noprobe)
        alpha_src = estimate_kinematic(sensors.theta, sensors.gamma_gps)  # косвенная
        return cascade(sensors, alpha_src, setpoints(t), pids_noprobe, cfg.dt)

    log_probe   = run(ctrl_probe,   aircraft, wind_params, cfg)
    log_noprobe = run(ctrl_noprobe, aircraft, wind_params, cfg)
    return log_probe, log_noprobe
```

Разные `seed` для `rng` — решение принято автором, физически мотивировано.

---

## 6. Сценарии (scenarios.py)

Каждый сценарий — функция `setpoints(t) -> (h_cmd, Va_cmd)`:

```python
def setpoints_C1(t):   # горизонтальный полёт
    return 100.0, 30.0

def setpoints_C2(t):   # кабрирование: рост тангажа 15-20°
    if t < 5.0: return 100.0, 30.0
    if t < 15.0: return 130.0, 30.0   # уставка по высоте вызовет рост тангажа
    return 130.0, 30.0

def setpoints_C4(t):   # выход на закритический УА
    if t < 5.0: return 100.0, 30.0
    return 100.0, 15.0   # резкое снижение скорости → рост α

def setpoints_C6(t):   # снижение в слой сдвига ветра
    return max(50.0, 100.0 - 2.0*t), 30.0   # плавное снижение
```

---

## 7. Как добавить новый модуль и не сломать существующий

1. Не изменять `dynamics.py`, `integrators.py`, `aero.py` — они **проверены**.
2. `control.py` и `sensors.py` — новые файлы, не редактируют существующие.
3. Интегрировать через `runner.run()` — передать новую `controls_fn`.
4. После реализации добавить раздел в `check.py`.
5. Запустить `check.py` — должно быть ALL CHECKS PASSED.

---

## 8. Параметры, которые автор должен предоставить

Помечены `[АВТОР]` в `config.py`. Без них работает на аналоге Aerosonde:

1. Масса, Jy, S, c, b вашего БПЛА
2. Таблицы Cl/Cd/Cm по УА
3. Уровни шума датчиков, **особенно шум зонда** (характеристика изделия)
4. Стартовые коэффициенты ПИД
5. Параметры ветра (постоянный, слой сдвига, порыв)
6. Критический и предупредительный УА для профиля
7. Таблица тяга→мощность (для С5)
8. Закон тяги: коэффициент и max оборотов

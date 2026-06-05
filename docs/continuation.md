# Передача контекста — что сделано и что делать дальше

Этот файл написан для агента, который продолжает разработку.
Читай в паре с `CLAUDE.md` (правила проекта) и `TZ_simulator.md` (техзадание).

---

## 1. Текущий статус

```
python check.py                              →  ALL CHECKS PASSED  (24/24)
python scenarios/s1_steady_flight.py         →  балансировочный полёт, 8 статичных субплотов
python scenarios/s2_pitch_up.py              →  анимация: кабрирование +5° → возврат на трим
python scenarios/s3_pitch_down.py            →  анимация: пикирование −5° → возврат на трим
python scenarios/s4_stall.py                 →  анимация: управляемый срыв + индикатор УА
python scenarios/s5_altitude_control.py      →  анимация: набор/снижение по уставке высоты
```

GIF-анимации сохранены в `results/s2.gif` … `results/s5.gif`.

### Реализованные модули

| Файл | Статус | Описание |
|------|--------|----------|
| `config.py` | ✅ | Параметры ЛА (Aerosonde-аналог), датчики, ветер, прогон |
| `state.py` | ✅ | Вектор [u,w,q,θ,x,h], `air_velocity`, `kinematic_gamma`, `total_energy` |
| `aero.py` | ✅ | CL sigmoid+срыв, CD квадратичный, Cm |
| `wind.py` | ✅ | Постоянный + сдвиг по высоте + одиночный порыв |
| `dynamics.py` | ✅ | `derivatives()` — чистая функция ОДУ продольного канала |
| `integrators.py` | ✅ | `step_euler`, `step_rk4` |
| `runner.py` | ✅ | `run()`, `Log`, `compute_trim()`, `trim_state()`, `print_summary()` |
| `sensors.py` | ✅ | Псевдодатчики с шумом: гироскоп, барометр, СВС, зонд УА, GPS |
| `control.py` | ✅ | `PID`, `PitchController` (каскад theta+q), `PitchControlParams` |
| `plotting.py` | ✅ | Статичные графики (dynamics, trajectory, energy, integrator check) |
| `check.py` | ✅ | 24 теста здравого смысла |
| `estimators.py` | ❌ | **Ещё не создан** — нужен для Этапа 2 |

### Сценарии

| Файл | Статус | Содержание |
|------|--------|------------|
| `scenarios/s1_steady_flight.py` | ✅ | С1: балансировочный полёт, статичные графики |
| `scenarios/s2_pitch_up.py` | ✅ | С2: кабрирование +5° → возврат на трим, анимация |
| `scenarios/s3_pitch_down.py` | ✅ | С3: пикирование −5° → возврат на трим, анимация |
| `scenarios/s4_stall.py` | ✅ | С4: управляемый срыв + **индикатор УА**, анимация |
| `scenarios/s5_altitude_control.py` | ✅ | : контроль высоты (см. примечание) |
| Парный прогон С6 | ❌ | **Ещё не создан** — центральный результат работы |

> **Примечание по С5:** По ТЗ С5 — сравнение энергопотребления при разном Cl/Cd
> (нужна таблица траст-теста от автора: пары тяга → мощность).
> До получения данных С5 реализован как демонстрация контроля высоты.
> Когда автор предоставит таблицу — создать `s5_power_comparison.py` рядом.

---

## 2. Архитектура контура управления

### control.py

```
PIDParams          ← dataclass: Kp, Ki, Kd, tau (фильтр производной), integral_limit
PID                ← класс: step(error, dt) → float
saturation()       ← np.clip
PitchControlParams ← dataclass: коэффициенты theta/q-контуров, h_Kp, q_max
PitchController    ← каскадный регулятор: theta_ref → q_ref → delta_e
```

### Каскад тангажа

```
theta_ref
    │
    ▼  error_theta = theta_ref - theta_meas
[PID_theta]  →  q_ref  (ограничено q_max = ±60°/с)
    │
    ▼  error_q = q_ref - q_meas
[PID_q]  →  delta_e_cmd  (знак ИНВЕРТИРОВАН: control.py строка ~229)
    │
[saturation ±25°]  →  delta_e
```

### Контур высоты (scenarios/s5_altitude_control.py)

```
h_ref
    │
    ▼  h_err = h_ref - h_meas
alpha_trim + KH * h_err  →  theta_ref  (ограничено ±15°,  KH=0.006 рад/м)
    │
    └──→  PitchController.set_pitch_setpoint()
```

Газ фиксирован = thr_trim во всех сценариях (h_Kp=0).

---

## 3. Сценарий С4 — индикатор УА (детали)

В `scenarios/s4_stall.py` реализован вычисляемый сигнал тревоги:

```python
# Уровни: 0=НОРМ, 1=ПРЕД, 2=КРИТ, 3=СРЫВ
alert_all = np.array([_alert_level(a) for a in alpha_all])
```

Пороги берутся из конфига: `alpha_warning=15°`, `alpha_crit=20°`, `alpha_stall=27°`.

В анимации:
- Отдельный субплот со ступенчатой функцией 0–3 и цветными зонами
- Фон info-box меняется: зелёный / жёлтый / оранжевый / красный
- Строка `ИНД = СРЫВ` в тексте

Результаты последнего прогона С4:
- alpha_max = 44.7°, t_ПРЕД = 14.2 с, t_КРИТ = 11.3 с, t_СРЫВ = 6.2 с

---

## 4. Следующий обязательный шаг: estimators.py + С6

### estimators.py

```python
# estimators.py — создать этот файл
from state import kinematic_gamma   # уже реализовано

def estimate_kinematic(theta_meas: float, gamma_gps: float) -> float:
    """
    Косвенная оценка УА (без зонда).
    alpha_est = theta - gamma_GPS.

    При боковом ветре gamma_GPS != gamma_air → ошибка оценки.
    Это и есть «слепота» косвенной оценки — ключевой тезис работы.
    """
    return theta_meas - gamma_gps
```

`kinematic_gamma(state)` уже есть в `state.py:87` — использовать его.

### Парный прогон С6

Оба прогона **идентичны во всём, кроме источника α** (требование ТЗ раздел 6.4):

```python
# Два независимых RNG — физически мотивировано (разные датчики)
rng_probe   = np.random.default_rng(seed=42)   # шум зонда
rng_noprobe = np.random.default_rng(seed=137)  # шум общих датчиков без зонда

# Прогон «с зондом»: alpha_src = прямое измерение
# Прогон «без зонда»: alpha_src = estimate_kinematic(theta_meas, gamma_gps)
```

Разные seed — решение принято автором, зафиксировано в memory.

Сценарий С6:
- Снижение (на базе С3) + вход в слой сдвига ветра (`WindParams(dV_shear=X)`)
- Два прогона на одном графике: зонд vs косвенная оценка
- Штилевой контроль: без ветра обе САУ должны практически совпасть

---

## 5. Важные технические детали

### Знаковое соглашение delta_e

В модели `Cmde = -0.5`:
- `delta_e > 0` → момент вниз (пикирование)
- `delta_e < 0` → момент вверх (кабрирование)

Инверсия знака в `PitchController.step()`:
```python
delta_e_cmd = -self.pid_q.step(error_q, dt)   # control.py строка ~229
```
Намеренно и корректно.

### Формула h_dot

В `dynamics.py`:
```python
dstate[H] = u * np.sin(theta) - w * np.cos(theta)
```
Это **правильная** формула (z_body вниз). Если комментарий в `state.py` расходится — доверять коду.

### Модель тяги

```
T = 0.5 * rho * S_prop * C_prop * ((k_motor * throttle)^2 - Va^2)
```
При `throttle=0` тяга отрицательна (пропеллер тормозит) — это использовалось в С4.

### PID коэффициенты (не оптимизированы, рабочие)

```
theta_Kp=1.5, theta_Ki=0.1, theta_Kd=0.3, theta_tau=0.1
q_Kp=0.5,    q_Ki=0.05,   q_Kd=0.1,    q_tau=0.05
KH (высота → theta_ref) = 0.006 рад/м
```

---

## 6. Параметры, которые автор должен предоставить

Помечены `[АВТОР]` в `config.py`:

1. Масса, Jy, S, c, b вашего БПЛА
2. Таблицы CL / CD / Cm по УА
3. Шум зонда УА (характеристика конкретного изделия)
4. Критический и предупредительный УА для вашего профиля
5. Параметры двигателя (k_motor, S_prop)
6. Параметры ветра (постоянный, слой сдвига, порыв)
7. Таблица траст-теста тяга→мощность (для С5 сравнения мощностей)

# Передача контекста — что сделано и что делать дальше

Этот файл написан для агента, который продолжает разработку.  
Читай в паре с `CLAUDE.md` (правила проекта) и `TZ_simulator.md` (техзадание).

---

## 1. Текущий статус

```
python check.py         →  ALL CHECKS PASSED  (24/24)
python demo.py          →  тримовый полёт 20 с, 4 статических графика
python animate.py       →  анимация тримового полёта
python demo_control.py  →  анимация: срыв + вывод из срыва (3 фазы)
python alt_control_demo.py  →  анимация: удержание высоты (2 скачка h_ref)
```

### Реализованные модули

| Файл | Статус | Описание |
|------|--------|----------|
| `config.py` | ✅ Готов | Параметры ЛА (Aerosonde-аналог), датчики, ветер, прогон |
| `state.py` | ✅ Готов | Вектор [u,w,q,θ,x,h], `air_velocity`, `kinematic_gamma`, `total_energy` |
| `aero.py` | ✅ Готов | CL sigmoid+срыв, CD квадратичный, Cm, аэросилы в связанной СК |
| `wind.py` | ✅ Готов | Постоянный + сдвиг по высоте + одиночный порыв |
| `dynamics.py` | ✅ Готов | `derivatives()` — чистая функция ОДУ продольного канала |
| `integrators.py` | ✅ Готов | `step_euler`, `step_rk4` |
| `runner.py` | ✅ Готов | `run()`, `Log`, `compute_trim()`, `trim_state()`, `print_summary()` |
| `sensors.py` | ✅ Готов | Псевдодатчики с шумом: гироскоп, барометр, СВС, зонд УА, GPS |
| `control.py` | ✅ Готов | `PID`, `PitchController` (каскад theta+q), `PitchControlParams` |
| `plotting.py` | ✅ Готов | 4 статических функции графиков (русские подписи) |
| `animate.py` | ✅ Готов | `animate_log()` — анимация тримового полёта по Log |
| `check.py` | ✅ Готов | 24 теста здравого смысла |
| `demo.py` | ✅ Готов | Тримовый прогон + 4 графика + проверка интегратора |
| `demo_control.py` | ✅ Готов | Анимация: 3-фазный сценарий срыва и вывода (см. ниже) |
| `alt_control_demo.py` | ✅ Готов | Анимация: удержание высоты, внешний P-контур по h |

---

## 2. Архитектура реализованного контура управления

### control.py — что там

```
PIDParams       ← dataclass: Kp, Ki, Kd, tau (фильтр производной), integral_limit
PID             ← класс: step(error, dt) → float. Фильтр d/dt первого порядка.
saturation()    ← np.clip, вынесен отдельно
PitchControlParams ← dataclass: коэффициенты theta- и q-контуров, h_Kp, q_max
PitchController ← каскадный регулятор: theta_ref → q_ref → delta_e
                  + P-контур по высоте → throttle (h_Kp, по умолчанию = 0 в демо)
```

### Каскад тангажа

```
theta_ref
    │
    ▼  error_theta = theta_ref - theta_meas
[PID_theta]  →  q_ref  (ограничено q_max = ±60°/с)
    │
    ▼  error_q = q_ref - q_meas
[PID_q]  →  delta_e_cmd  (знак инвертирован! см. control.py:229)
    │
[saturation ±25°]  →  delta_e
```

### Контур высоты → газ (в alt_control_demo.py)

```
h_ref
    │
    ▼  h_err = h_ref - h_meas
alpha_trim + KH * h_err  →  theta_ref  (ограничено ±15°)
    │
    └──→  в PitchController.set_pitch_setpoint()
```

Газ в обоих демо фиксирован = `thr_trim` (h_Kp установлен в 0).

---

## 3. Демо-файлы: сценарии

### demo_control.py — срыв и вывод

| Фаза | t, с | theta_ref | throttle |
|------|------|-----------|----------|
| 1 — трим | 0–8 | 1.9° (alpha_trim) | thr_trim = 0.398 |
| 2 — срыв | 8–25 | 20° | 0.0 (пропеллер тормозит) |
| 3 — вывод | 25–55 | −5° | thr_trim = 0.398 |

Результат: Va падает до ~14 м/с (скорость сваливания), alpha_max ≈ 44°, вывод восстанавливает Va ≈ 32 м/с, h_final ≈ 225 м (запас от h0 = 300 м).

На subplot УА — три зоны: оранжевая (>15°), красная (>20°), линия срыва (27°).  
Info-box показывает `предупрежд.` / `! критич. !` / `!! СРЫВ !!` в реальном времени.

Сохранена анимация: `stall_recovery.gif`.

### alt_control_demo.py — удержание высоты

| Фаза | t, с | h_ref |
|------|------|-------|
| 1 — трим | 0–10 | 100 м |
| 2 — набор | 10–40 | 200 м (редактируется пользователем) |
| 3 — снижение | 40–60 | 100 м |

Результат: h_final ≈ 100 м (точность < 1 м), Va почти не меняется, alpha < 5°.

---

## 4. Что ещё нужно реализовать (приоритеты по ТЗ)

| Модуль | Приоритет | Описание |
|--------|-----------|----------|
| `estimators.py` | 🔴 Критично | Косвенная оценка УА: `alpha_est = theta_meas - gamma_gps`. Это "мир без зонда". |
| `scenarios.py` | 🔴 Критично | Структурированные уставки для С1–С6, слой сдвига ветра |
| Парный прогон | 🔴 Критично | "С зондом vs без" — центральный результат главы 8 |
| Скоростной контур | 🟡 Важно | PID(Va) → throttle (сейчас газ фиксирован) |
| `check.py` для control/sensors | 🟡 Важно | Добавить тесты для новых модулей |
| Боковой канал | 🟢 Позже | Расширение вектора состояния |

---

## 5. Следующий обязательный шаг: estimators.py

```python
# estimators.py

from state import kinematic_gamma   # уже реализовано в state.py

def estimate_kinematic(theta_meas: float, gamma_gps: float) -> float:
    """
    Косвенная оценка УА (без зонда).
    alpha_est = theta - gamma_GPS.

    ВАЖНО: gamma_GPS вычисляется по земной скорости (GPS).
    При боковом ветре gamma_GPS ≠ gamma_air → ошибка оценки.
    Это и есть "слепота" косвенной оценки — ключевой тезис работы.
    """
    return theta_meas - gamma_gps
```

`kinematic_gamma()` уже есть в `state.py:87` — использовать его для вычисления gamma_gps в sensors.py.

---

## 6. Парный прогон (С6 — кульминация)

Оба прогона **идентичны во всём, кроме источника α**:

```python
def run_paired(controls_builder, aircraft, wind_params, cfg):
    # Два независимых RNG — физически мотивировано (разные датчики)
    rng_probe   = np.random.default_rng(seed=42)
    rng_noprobe = np.random.default_rng(seed=137)

    def ctrl_probe(t, state, Va, alpha):
        # alpha_src = прямое измерение зонда
        alpha_src = measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng_probe)
        return controls_builder(t, state, Va, alpha_src, rng_probe)

    def ctrl_noprobe(t, state, Va, alpha):
        # alpha_src = косвенная оценка theta - gamma_GPS
        sensors = measure_all(state, Va, alpha, sp, rng_noprobe)
        alpha_src = estimate_kinematic(sensors.theta, sensors.gamma_gps)
        return controls_builder(t, state, Va, alpha_src, rng_noprobe)

    log_probe   = run(ctrl_probe,   aircraft, wind_params, cfg)
    log_noprobe = run(ctrl_noprobe, aircraft, wind_params, cfg)
    return log_probe, log_noprobe
```

Разные seed для rng — решение принято автором, зафиксировано в memory.

---

## 7. Важные технические детали

### Знаковое соглашение delta_e

В модели `Cmde = -0.5`:
- `delta_e > 0` → момент вниз (пикирование)
- `delta_e < 0` → момент вверх (кабрирование)

В `PitchController.step()` — инверсия знака q-контура:
```python
delta_e_cmd = -self.pid_q.step(error_q, dt)   # control.py:229
```
Это намеренно и корректно. При желании набрать тангаж: q_ref > 0, error_q > 0, pid_q > 0, delta_e_cmd < 0 → кабрирование ✓.

### h_dot формула

В `dynamics.py:101`:
```python
dstate[H] = u * np.sin(theta) - w * np.cos(theta)
```
Это **правильная** формула (выведена из СК, z_body вниз). Комментарий в `state.py` содержит опечатку (знак у w обратный) — доверять коду, не комментарию.

### Модель тяги

```
T = 0.5 * rho * S_prop * C_prop * ((k_motor * throttle)^2 - Va^2)
```
При `throttle < Va/k_motor` — тяга **отрицательная** (пропеллер тормозит).  
Trim throttle = 0.398, k_motor = 80 → порог: Va = 31.8 м/с. При Va < 31.8 и trim throttle тяга положительна. Это использовалось в сценарии срыва: throttle = 0 → T ≈ −0.13 * Va² → сильное торможение.

### Стандартные PID коэффициенты (пока не оптимизированы)

```python
PitchControlParams:
    theta_Kp=1.5, theta_Ki=0.1, theta_Kd=0.3, theta_tau=0.1
    q_Kp=0.5,    q_Ki=0.05,   q_Kd=0.1,    q_tau=0.05
    h_Kp=0.01    # управление высотой через тягу (в демо = 0)
    q_max = ±60°/с

KH (altitude → theta_ref) = 0.006 рад/м  (в alt_control_demo.py)
```

---

## 8. Параметры, которые автор должен предоставить

Помечены `[АВТОР]` в `config.py`. Без них — аналог Aerosonde:

1. Масса, Jy, S, c, b вашего БПЛА
2. Таблицы CL / CD / Cm по УА
3. Шум зонда УА (характеристика конкретного изделия)
4. Критический и предупредительный УА для вашего профиля
5. Параметры двигателя (k_motor, S_prop)
6. Параметры ветра на полигоне (постоянный, слой сдвига)

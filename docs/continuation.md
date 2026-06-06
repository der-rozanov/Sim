# Передача контекста — что сделано и что делать дальше

Этот файл написан для агента, который продолжает разработку.
Читай в паре с `CLAUDE.md` (правила проекта) и `TZ_simulator.md` (техзадание).

**Актуально на: 2026-06-06**

---

## 1. Текущий статус запуска

```
python check.py                              →  ALL CHECKS PASSED  (24/24)
python scenarios/s1_steady_flight.py         →  балансировочный полёт, статичные субплоты
python scenarios/s2_pitch_up.py              →  анимация: кабрирование +5° → возврат на трим
python scenarios/s3_pitch_down.py            →  анимация: пикирование −5° → возврат на трим
python scenarios/s4_stall.py                 →  анимация: управляемый срыв + индикатор УА
python scenarios/s5_altitude_control.py      →  анимация: набор/снижение по уставке высоты
python scenarios/s6_speed_control.py         →  анимация: контроль высоты + удержание Va
python scenarios/s7_alpha_estimation.py      →  анимация: зонд vs ИНС+GPS при попутном ветре
```

---

## 2. Реализованные модули

| Файл | Статус | Описание |
|------|--------|----------|
| `config.py` | ✅ | Параметры ЛА (Aerosonde-аналог), датчики, ветер, прогон |
| `state.py` | ✅ | Вектор [u,w,q,θ,x,h], `air_velocity`, `kinematic_gamma`, `total_energy` |
| `aero.py` | ✅ | CL sigmoid+срыв, CD квадратичный, Cm |
| `wind.py` | ✅ | Постоянный + сдвиг по высоте + одиночный порыв |
| `dynamics.py` | ✅ | `derivatives()` — чистая функция ОДУ продольного канала |
| `integrators.py` | ✅ | `step_euler`, `step_rk4` |
| `runner.py` | ✅ | `run()`, `Log`, `compute_trim()`, `trim_state()`, `print_summary()` |
| `sensors.py` | ✅ | Псевдодатчики: гироскоп, барометр, СВС, зонд УА, GPS |
| `control.py` | ✅ | `PID`, `PitchController` (θ+q каскад), **`SpeedController`** (Va→throttle) |
| `estimators.py` | ✅ | **`estimate_alpha_indirect()`** — косвенная оценка УА через ИНС+GPS |
| `plotting.py` | ✅ | Статичные графики (dynamics, trajectory, energy, integrator check) |
| `check.py` | ✅ | 24 теста здравого смысла |

## 3. Сценарии

| Файл | Статус | Содержание |
|------|--------|------------|
| `scenarios/s1_steady_flight.py` | ✅ | С1: балансировочный полёт, статичные графики |
| `scenarios/s2_pitch_up.py` | ✅ | С2: кабрирование +5° → возврат на трим, анимация |
| `scenarios/s3_pitch_down.py` | ✅ | С3: пикирование −5° → возврат на трим, анимация |
| `scenarios/s4_stall.py` | ✅ | С4: управляемый срыв + **индикатор УА**, анимация |
| `scenarios/s5_altitude_control.py` | ✅ | С5: контроль высоты, газ фиксирован |
| `scenarios/s6_speed_control.py` | ✅ | С6: контроль высоты + **удержание Va** (SpeedController) |
| `scenarios/s7_alpha_estimation.py` | ✅ | С7: **сравнение зонд vs ИНС+GPS** при попутном ветре |
| Парный прогон «зонд vs без» (ТЗ С6) | ❌ | **Следующий шаг** — центральный результат работы |

> **Примечание по С5:** По ТЗ С5 — сравнение энергопотребления при разном Cl/Cd.
> До получения таблицы траст-теста от автора С5 реализован как демонстрация
> контроля высоты. Когда автор предоставит данные — создать `s5_power_comparison.py`.

---

## 4. Архитектура управления (control.py)

### 4.1 Классы и dataclass'ы

```
PIDParams           ← Kp, Ki, Kd, tau, integral_limit
PID                 ← step(error, dt) → float
saturation()        ← np.clip

PitchControlParams  ← коэффициенты theta/q-контуров, h_Kp, Va_ref, q_max
PitchController     ← каскад theta_ref → q_ref → delta_e

SpeedControlParams  ← Va_Kp=0.08, Va_Ki=0.02, Va_Kd=0.0, Va_tau=0.5
SpeedController     ← Va_ref → throttle  (НОВОЕ, добавлено в С6)
```

### 4.2 Каскад тангажа (PitchController)

```
theta_ref
    │
    ▼  error_theta = theta_ref − theta_meas
[PID_theta]  →  q_ref  (ограничено ±60°/с)
    │
    ▼  error_q = q_ref − q_meas
[PID_q]  →  delta_e_cmd  (знак ИНВЕРТИРОВАН — намеренно, строка ~229)
    │
[saturation ±25°]  →  delta_e
```

### 4.3 Контур скорости (SpeedController)

```
Va_ref (уставка)
    │
    ▼  error_Va = Va_ref − Va_meas
[PID_Va]  →  delta_throttle
    │
    ▼  throttle = trim_throttle + delta_throttle
[saturation 0..1]  →  throttle
```

Параметры (SpeedControlParams): `Va_Kp=0.08, Va_Ki=0.02, Va_Kd=0.0`

### 4.4 Контур высоты (внешний, в сценариях)

```
h_ref
    │
    ▼  h_err = h_ref − h_meas
alpha_trim + KH * h_err  →  clip(±15°)  →  theta_ref   (KH=0.006 рад/м)
    │
    └─→  PitchController.set_pitch_setpoint()
```

### 4.5 Как используется в С6 и С7

```python
# delta_e: от PitchController (как раньше)
ctrl_out = controller.step(t, meas, cfg.dt)
delta_e  = ctrl_out[0]

# throttle: от SpeedController (НОВОЕ — вместо trim_throttle=const)
throttle = spd_ctrl.step(Va_meas, cfg.dt)

return np.array([delta_e, throttle])
```

---

## 5. Оценщик угла атаки (estimators.py)

```python
def estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps) -> float:
    """alpha_est = theta_meas − gamma_gps,  gamma_gps = arctan(Vh/Vx)"""
    gamma_gps = np.arctan2(Vh_gps, Vx_gps)
    return theta_meas - gamma_gps
```

**Источник ошибки при ветре (горизонтальный Vwx):**

| Фаза | Эффект |
|------|--------|
| Уровень | gamma_gps ≈ gamma_air → bias ≈ 0 |
| Набор (попутный ветер) | Vx_earth > Vx_air → gamma_gps < gamma_air → alpha_est > alpha_true |
| Снижение (попутный ветер) | Vh_earth < 0, же Vx → gamma_gps > gamma_air → alpha_est < alpha_true |

**Результаты С7** (ветер Vwx = +5 м/с):

| | Зонд | ИНС+GPS |
|---|---|---|
| σ (шум) | 0.57° | 0.69° |
| Смещение набор | +0.006° | +0.80° |
| Смещение снижение | +0.007° | −1.26° |

---

## 6. Исправленный баг в sensors.py / state.py

**Проблема:** `measure_gps_velocity_earth()` и `kinematic_gamma()` использовали
`Vh_earth = u*sin + w*cos` — **неправильный знак**. Правильно (совпадает с
`dynamics.py` строка 101):

```python
# Было (неверно):
Vh_earth_true = u_body * st + w_body * ct

# Стало (правильно):
Vh_earth_true = u_body * st - w_body * ct
```

**Причина:** ось z_body направлена вниз. Её вертикальная земная проекция = −cos(θ).
Формула вынуждает `h_dot = u*sin(θ) − w*cos(θ)`, которая обращается в ноль при
уровне полёта (балансировочных условиях).

**Влияние:** баг не затрагивал С1–С5 (функции не использовались). Исправлен
при разработке С7.

---

## 7. Следующий обязательный шаг: парный прогон

По ТЗ (раздел 6.4) — **кульминационный результат главы 8**:

```
Два прогона ИДЕНТИЧНЫ во всём, кроме источника УА:
  «С зондом»:   alpha_src = measure_angle_of_attack(alpha, ...)
  «Без зонда»:  alpha_src = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)
```

Сценарий для парного прогона:
- Снижение (на базе С3) + вход в слой сдвига ветра (`WindParams(dV_shear=X)`)
- Ветер по высоте: до h_shear_lo — штиль, выше h_shear_hi — встречный ветер
- Оба прогона на одном графике: зонд vs косвенная оценка
- Штилевой контроль (без ветра): обе CAУ должны практически совпадать

Скелет сценария:
```python
# Общий RNG для «общих» датчиков (честное сравнение)
rng_shared = np.random.default_rng(seed=42)

# «С зондом»
alpha_src_probe = measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng_shared)

# «Без зонда»
Vx, Vh = measure_gps_velocity_earth(state[U], state[W], state[THETA], 0.0, sp.gps_vel_noise, rng_shared)
alpha_src_est = estimate_alpha_indirect(theta_meas, Vx, Vh)
```

---

## 8. Технические детали, которые надо помнить

### Знаковое соглашение delta_e

`Cmde = -0.5`: `delta_e > 0` → пикирование, `delta_e < 0` → кабрирование.
Инверсия знака в `PitchController.step()` — намеренна и корректна.

### Формула h_dot

```python
# dynamics.py строка 101 — ПРАВИЛЬНО:
dstate[H] = u * np.sin(theta) - w * np.cos(theta)
```
z_body направлена вниз. Если какой-то другой файл расходится — доверять dynamics.py.

### Модель тяги

```
T = 0.5 * rho * S_prop * C_prop * ((k_motor * throttle)² − Va²)
```
При `throttle=0` тяга отрицательна (пропеллер тормозит) — использовалось в С4.

### Балансировочная тяга

Из `compute_trim()`: `thr_trim ≈ 0.398` при Va=30 м/с (параметры Aerosonde-аналога).
SpeedController использует её как feedforward: `throttle = thr_trim + ΔT`.

### ПИД коэффициенты (рабочие, не оптимизированы)

```
PitchController:  theta_Kp=1.5  theta_Ki=0.1  theta_Kd=0.3  theta_tau=0.1
                  q_Kp=0.5      q_Ki=0.05     q_Kd=0.1      q_tau=0.05
SpeedController:  Va_Kp=0.08    Va_Ki=0.02    Va_Kd=0.0     Va_tau=0.5
KH (высота):      0.006 рад/м
```

---

## 9. Параметры, которые автор должен предоставить

Помечены `[АВТОР]` в `config.py`:

1. Масса, Jy, S, c, b своего БПЛА
2. Таблицы CL / CD / Cm по УА (или коэффициенты полиномов)
3. Шум зонда УА (характеристика конкретного изделия)
4. Критический и предупредительный УА для своего профиля
5. Параметры двигателя (k_motor, S_prop)
6. Параметры ветра (сдвиг по высоте, границы слоя, перепад)
7. Таблица траст-теста тяга→мощность (для сравнения энергопотребления)

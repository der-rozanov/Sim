# Архитектура кода

---

## 1. Граф зависимостей модулей

```
config.py            ← нет зависимостей (только dataclasses)
    │
    ├── state.py     ← config
    ├── aero.py      ← config
    ├── wind.py      ← config
    ├── sensors.py   ← numpy only, параметры из config
    ├── control.py   ← config (ограничения рулей и тяги)
    ├── estimators.py ← numpy only (НЕ зависит от sensors или control)
    │
    └── dynamics.py  ← config, state, aero
            │
            └── integrators.py ← dynamics
                    │
                    └── runner.py ← config, state, wind, integrators
                            │
                            ├── plotting.py       ← state, config
                            ├── animate.py        ← state (только индексы)
                            └── scenarios/
                                ├── s1..s5        ← runner, control, sensors, config
                                ├── s6            ← runner, control, sensors, config
                                └── s7            ← runner, control, sensors, estimators, config
```

`check.py` — импортирует все модули, зависимость только в одну сторону.

**Главное правило**: зависимости направлены только **вниз по графу**.  
`sensors.py`, `control.py`, `estimators.py` — не импортируют `runner.py` или друг друга.  
`estimators.py` намеренно не зависит от `sensors.py`: оценщик принимает уже готовые
измерения (`theta_meas`, `Vx_gps`, `Vh_gps`), а не сырое состояние.

---

## 2. Принцип чистой функции производных

`dynamics.derivatives()` — **чистая функция**:

```python
def derivatives(state, controls, t, params, wind_fn) -> dstate:
    ...  # только вычисления, без изменения внешних объектов
```

- Принимает только числа/массивы, возвращает только массив.
- Нет обращений к глобальным переменным.
- Нет присваиваний к `state` или `params`.

Зачем? Интеграторы RK4 вызывают `derivatives` четыре раза на шаг  
(для k1..k4) — если бы функция изменяла состояние, результат был бы неверен.

---

## 3. Главный цикл (`runner.run`)

```
┌─────────────────────────────────────────────────────────────┐
│  СНАРУЖИ (один раз):                                        │
│    state ← initial_state(cfg) или trim_state()              │
│    log   ← пустые массивы под N шагов                       │
│    wind_call = lambda h, t: wind(h, t, wind_params)         │
│                                                             │
│  ЦИКЛ (каждые dt секунд):                                   │
│    1. wind_vec  = wind_call(h, t)           # ветер         │
│    2. Va, alpha = air_velocity(state, wind) # УА            │
│    3. controls  = controls_fn(t, state, Va, alpha)  # САУ  │
│    4. log[i]    ← state, Va, alpha, energy, wind    # лог  │
│    5. state     = step_rk4(state, controls, ...)    # шаг  │
│    6. t         += dt                                       │
└─────────────────────────────────────────────────────────────┘
```

**Порядок важен**: сначала логируем состояние, потом интегрируем.  
Это означает, что `log[i]` содержит состояние **перед** i-м шагом,  
а управление `controls[i]` — то, которое **применено** на этом шаге.

---

## 4. Структура Log

```python
@dataclass
class Log:
    t:        ndarray (N,)      # время
    state:    ndarray (N, 12)   # [u, w, q, θ, x, h, v, p, r, φ, ψ, y] на каждом шаге
    controls: ndarray (N, 4)    # [δe, throttle, δa, δr]
    Va:       ndarray (N,)      # воздушная скорость
    alpha:    ndarray (N,)      # УА в радианах
    E_kin:    ndarray (N,)      # кинетическая энергия, Дж
    E_pot:    ndarray (N,)      # потенциальная энергия, Дж
    E_total:  ndarray (N,)      # полная механическая энергия, Дж
    wind_vec: ndarray (N, 3)    # [Vwx, Vwh, Vwy] в м/с
    beta:     ndarray (N,)      # УС в радианах
```

Столбцы 0..5 `state`, 0..1 `controls`, 0..1 `wind_vec` имеют тот же смысл,
что и в продольной версии — старый код с `log.state[:, H]`,
`log.controls[:, 0]`, `log.wind_vec[:, 1]` работает без изменений.

Обращение к компонентам состояния из лога:
```python
from state import THETA, H, U
theta = log.state[:, THETA]   # весь ряд углов тангажа
h_final = log.state[-1, H]    # финальная высота
```

---

## 5. Функция управления (`controls_fn`)

Любой вызываемый объект с сигнатурой:

```python
def my_controller(t: float, state: ndarray, Va: float, alpha: float) -> ndarray:
    delta_e  = ...   # рад
    throttle = ...   # 0.0 … 1.0
    return np.array([delta_e, throttle])                 # продольный сценарий
    # или
    return np.array([delta_e, throttle, delta_a, delta_r])  # с боковым каналом
```

Если возвращено 2 элемента, `runner` дополняет их нулями (`state.full_controls`):
δa = δr = 0. УС в `controls_fn` не передаётся (сигнатура сохранена для
совместимости) — боковая САУ вычисляет его сама через `state.air_data()`
или берёт с псевдодатчика.

Передаётся в `run()`. Примеры:
```python
# Фиксированное управление (трим без обратной связи)
run(lambda t, s, V, a: np.array([de_trim, thr_trim]), ...)

# Будущий ПИД-контроллер
run(cascade_controller, ...)
```

---

## 6. Интегратор: замороженное управление

RK4 вычисляет производные в четырёх точках внутри шага (k1..k4).  
**Управление одинаково** для всех четырёх вызовов:

```python
k1 = derivatives(state,          controls, t,      ...)
k2 = derivatives(state + dt/2*k1, controls, t+dt/2, ...)   # controls неизменны!
k3 = derivatives(state + dt/2*k2, controls, t+dt/2, ...)
k4 = derivatives(state + dt*k3,   controls, t+dt,   ...)
```

Это соответствует реальной дискретной САУ: контроллер обновляет команды  
**раз в такт** (каждые dt секунд), между обновлениями управление постоянно.

---

## 7. Ветровая функция как параметр

`derivatives` принимает `wind_fn` — это функция, а не число.  
Это позволяет легко менять модель ветра без изменения кода физики:

```python
# Штиль
wind_call = lambda h, t: (0.0, 0.0, 0.0)

# Возвращается (Vwx, Vwh, Vwy) — север, вверх, восток.
# Пара (Vwx, Vwh) без третьего элемента тоже принимается (Vwy = 0).

# Реальная модель с параметрами
wind_call = lambda h, t: wind(h, t, wind_params)

# Произвольный тестовый ветер
wind_call = lambda h, t: (5.0 * np.sin(t), 0.0)
```

---

## 8. Балансировочные условия (`compute_trim` и `trim_state`)

`compute_trim(aircraft, Va)` решает **систему 2×2** аналитически:

```
[CLa   CLde] [α  ]   [CL_req − CL0]
[Cma   Cmde] [δe ] = [−Cm0        ]
```

Это точное решение (не итеративное), поэтому начальное состояние из  
`trim_state()` — идеально балансировочное, без переходных процессов.

---

## 9. Как добавить новый сценарий

1. Написать функцию управления `controls_fn` (или использовать существующую).
2. Вызвать `run(controls_fn, aircraft, wind_params, cfg)`.
3. Передать `log` в функции `plotting.py` или `animate.animate_log`.

Физика, интегратор, лог — ничего менять не нужно.

---

## 10. Боковой канал (модель 6DOF, реализовано 2026-10-06)

Сделано расширением, не переписыванием:

1. `state.py` — индексы `V, P, R, PHI, PSI, Y` (6..11) дописаны после продольных;
   индексы управления `DE, DT, DA, DR`; `full_controls()`; матрица поворота
   `rotation_body_to_earth()`; `earth_velocity()`; `air_data()` → (Va, α, β).
   `air_velocity()` сохранён и возвращает (Va, α), как раньше.
2. `dynamics.derivatives()` — полные уравнения 6DOF (docs/physics.md, раздел 3).
3. `config.py` / `aero.py` — Jx, Jz, Jxz, боковые коэффициенты `CY_*`, `Croll_*`,
   `Cn_*`, функция `aero_lateral()`.
4. `wind.py` — третья компонента `Vwy` (постоянный боковой ветер `Vw_cross`,
   боковой порыв `gust_vwy`).

Гарантия совместимости: при v = p = r = φ = ψ = 0, δa = δr = 0 уравнения
вырождаются точно в прежние продольные (проверка 7.3 в `checks/check.py`;
регрессия по прогону с ветром и дублетом руля — совпадение u, w, q, θ, h
до бита). Все сценарии s1–s11 и lab6 работают без правок.

Адекватность бокового канала в открытом контуре — `checks/check_lateral.py`.

---

## 11. Изоляция `animate.py`

`animate.py` намеренно **не импортирует** `runner`, `dynamics`, `config`.  
Единственная зависимость — именованные индексы из `state.py` (`X`, `H`, `THETA`).

Это означает: анимацию можно использовать в любом другом проекте,  
передав ей любой объект с полями `t`, `state`, `Va`, `alpha`.

# Архитектура кода

---

## 1. Граф зависимостей модулей

```
config.py          ← нет зависимостей (только dataclasses)
    │
    ├── state.py   ← config
    ├── aero.py    ← config
    ├── wind.py    ← config
    ├── sensors.py ← (numpy only, параметры из config)
    ├── control.py ← config (для ограничений рулей и тяги)
    │
    └── dynamics.py ← config, state, aero
            │
            └── integrators.py ← dynamics
                    │
                    └── runner.py ← config, state, wind, integrators
                            │
                            ├── plotting.py      ← state, config
                            ├── animate.py       ← state (только индексы)
                            ├── demo.py          ← runner, plotting, config
                            ├── demo_control.py  ← runner, control, sensors, config
                            └── alt_control_demo.py ← runner, control, sensors, config
```

`check.py` — импортирует все модули, зависимость только в одну сторону.

**Главное правило**: зависимости направлены только **вниз по графу**.  
`sensors.py` и `control.py` не импортируют `runner.py` или друг друга.

**Главное правило**: зависимости направлены только **вниз по графу**.  
`aero.py` ничего не знает о `runner.py`, `runner.py` ничего не знает о `plotting.py`.

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
    state:    ndarray (N, 6)    # [u, w, q, θ, x, h] на каждом шаге
    controls: ndarray (N, 2)    # [δe, throttle]
    Va:       ndarray (N,)      # воздушная скорость
    alpha:    ndarray (N,)      # УА в радианах
    E_kin:    ndarray (N,)      # кинетическая энергия, Дж
    E_pot:    ndarray (N,)      # потенциальная энергия, Дж
    E_total:  ndarray (N,)      # полная механическая энергия, Дж
    wind_vec: ndarray (N, 2)    # [Vwx, Vwh] в м/с
```

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
    return np.array([delta_e, throttle])
```

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
wind_call = lambda h, t: (0.0, 0.0)

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

## 10. Как добавить боковой канал (Этап 3)

1. Расширить вектор состояния в `state.py` — добавить индексы `V, P, R, PHI, PSI, Y`.
2. Добавить боковые уравнения в `dynamics.derivatives()`.
3. Добавить боковые аэрокоэффициенты в `config.py` и `aero.py`.
4. Добавить боковые управления в `controls_fn` (элероны, руль направления).

Продольный канал при этом **не меняется** — расширение, не переписывание.

---

## 11. Изоляция `animate.py`

`animate.py` намеренно **не импортирует** `runner`, `dynamics`, `config`.  
Единственная зависимость — именованные индексы из `state.py` (`X`, `H`, `THETA`).

Это означает: анимацию можно использовать в любом другом проекте,  
передав ей любой объект с полями `t`, `state`, `Va`, `alpha`.

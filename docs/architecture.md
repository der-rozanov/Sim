# Архитектура кода

---

## 1. Граф зависимостей модулей

Слои зависят только от нижележащих:

```
sim/            физика (не знает об управлении)
  config.py       ← только dataclasses: AircraftParams, WindParams, SimConfig, SensorParams
  state.py        ← config: индексы, air_data(), earth_velocity(), матрица поворота
  aero.py, wind.py, ground.py, actuators.py ← config, state
  dynamics.py     ← config, state, aero, ground: derivatives(), propeller()
  integrators.py  ← dynamics: step_rk4(), step_euler()
      │
control/        САУ (импортирует только sim/)
  controllers.py  PID, PitchController, SpeedController, RollController, LateralController,
                  AltitudeHold, RollHold
  sensors.py      псевдодатчики (numpy)
  tecs.py, aua.py, navigation.py, lqr.py (← dynamics: линеаризация)
  tuning.py       расчёт коэффициентов по модели (← runner.compute_trim)
  sau.py          SAU: всё вместе;  blocks.py — схемы из блоков (← sau, controllers)
      │
runner.py       ← sim/: run() → Log, compute_trim(), trim_state(), print_summary()
flight_logger.py← runner, sim/: FlightLogger.save() → .flightlog, load_log()
      │
viz/            отображение (не импортирует control/)
  viewer.py       ← flight_logger: 2D-графики и анимация .flightlog
  viewer3d.py     ← flight_logger, world3d, aircraft3d, ursina: 3D
  world3d.py, mapdata.py, osmdata.py, landcover.py — миры и карты
  bench_panel.py, station.py, block_editor.py — окна tkinter (процессы-спутники)
      │
scenarios/      прогоны и интерактивные программы (← всё выше)
  s1–s7         ← runner, control (регуляторы напрямую), flight_logger, viz.viewer
  s8–s10        ← lateral_common (FullSAU → control.sau.SAU), flight_logger, viz.viewer
  GameScenario3D, Bench3D, BlockLab3D ← control.sau, sim/, viz.viewer3d
archive/probe/  сценарии сравнения «зонд vs косвенная оценка УА» (вне основной линии)
checks/         проверки (← sim/, control/, runner)
```

**Главное правило**: зависимости направлены только **вниз по графу**.
`sim/` не знает о `control/`; `viz/` не знает о `control/` и физике (читает `.flightlog`).

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
    controls: ndarray (N, 4)    # [δe, throttle, δa, δr] — положения рулей после приводов
    Va:       ndarray (N,)      # воздушная скорость
    alpha:    ndarray (N,)      # УА в радианах
    E_kin:    ndarray (N,)      # кинетическая энергия, Дж
    E_pot:    ndarray (N,)      # потенциальная энергия, Дж
    E_total:  ndarray (N,)      # полная механическая энергия, Дж
    wind_vec: ndarray (N, 3)    # [Vwx, Vwh, Vwy] в м/с
    beta:     ndarray (N,)      # УС в радианах
    controls_cmd: ndarray (N, 4) # команды САУ до приводов рулей (положения — в controls)
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

1. Написать функцию управления `controls_fn` (или собрать `SAU` — см. s8–s10).
2. Вызвать `log = run(controls_fn, aircraft, wind_params, cfg, state0=trim_state(aircraft, cfg))`.
3. Сохранить лог с уставками: `path = FlightLogger(scenario=..., ...).save(log, h_ref=..., chi_ref=...)`.
4. Показать: `show_flightlog(path)` (`viz/viewer.py`).

Физика, интегратор, лог, графики — ничего менять не нужно. Шаблон — `scenarios/s2_pitch_up.py`.

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
до бита). Все продольные сценарии работали без правок.

Адекватность бокового канала в открытом контуре — `checks/check_lateral.py`.

---

## 11. Просмотр логов (`viz/viewer.py`)

Единственный 2D-показ результатов. Вход — только `.flightlog` (`flight_logger.load_log`),
физику и САУ не импортирует. Набор панелей выбирается по содержимому лога: продольные
всегда; вид сверху, χ, φ, β, δa, δr — если ЛА маневрирует по курсу или в логе есть
уставка курса / маршрут. Уставки (`*_ref`), команды до приводов (`*_cmd`), события и
маршрут пишет `FlightLogger.save()`.

```python
from viz.viewer import show_flightlog
show_flightlog(path)                        # анимация; несколько путей — несколько окон
show_flightlog(path, static=True)           # без анимации
show_flightlog(path, save_path="out.gif")   # сохранить анимацию (.gif / .mp4)
```

CLI: `python viz/viewer.py [файл] [--static] [--speed N] [--save out.gif]`.

---

## 12. Изоляция `viewer3d.py` (3D-тренажёр)

`viewer3d.py` не импортирует `sim/`, `control/`, `runner`: вход — только
`.flightlog` через `flight_logger.load_log()`. Ursina (РЕШ-16) используется
только здесь. Подробно — `docs/viewer3d.md`.


---

## 13. Основные интерфейсы

```python
# Прогон
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
# controls_fn(t, state, Va, alpha) -> [δe, δt] или [δe, δt, δa, δr]
alpha_trim, de_trim, thr_trim = compute_trim(aircraft, Va)
s0 = trim_state(aircraft, cfg)

# Полная САУ (control/sau.py)
sau = SAU(aircraft, trim, sensors, SAUParams(), nav=None)
sau.engage(theta=..., h=..., Va=..., chi=...)
sau.set_mode(Mode.ALTITUDE)                 # PITCH | ALTITUDE | ROUTE
controls = sau.step(t, state, Va, alpha, beta, dt)

# Коэффициенты по модели (control/tuning.py)
d = design_sau(aircraft, Va=16.0)           # d.pitch, d.speed, d.lateral, d.tecs, ...

# Лог
path = FlightLogger(scenario="С2: Кабрирование", aircraft=aircraft, wind_params=wind_params,
                    cfg=cfg, trim=(alpha_trim, de_trim, thr_trim),
                    events=[{"t": 10.0, "label": "команда", "color": "steelblue"}]
                    ).save(log, theta_ref=theta_ref_buf)
data = load_log(path)                       # dict: каналы + data["meta"]
```

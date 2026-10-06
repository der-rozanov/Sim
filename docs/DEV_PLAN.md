# DEV_PLAN — Согласованная архитектура и текущий статус

Главный справочный документ разработки. Читать в паре с `CLAUDE.md`.
Детали по каждой теме — в `docs/`.

**Обновлён: 2026-10-06**

---

## Статус: что работает прямо сейчас

| Этап | Содержание | Статус |
|------|-----------|--------|
| 0 — Каркас | Модули, RK4, вектор состояния, пустой прогон | ✅ Готово |
| 1 — База | Физика, ПИД, датчики, С1–С5, индикация УА | ✅ Готово |
| 2а — Контур скорости | `SpeedController`, С6 (контроль высоты + Va) | ✅ Готово |
| 2б — Оценка УА | `estimators.py`, С7 (зонд vs ИНС+GPS при ветре) | ✅ Готово |
| 2в — Парный прогон | «С зондом vs без» (s9), АУА (s10) | ✅ Готово; тематика зонда на паузе с 2026-10-06 |
| 3 — Фичи | LQR (s11) | ✅ Готово |
| 3-Б1 — Боковой канал: физика 6DOF | Вектор 12, δa/δr, боковой ветер, check.py §7 | ✅ Готово |
| 3-Б2 — Боковой канал: открытый контур | `checks/check_lateral.py` (моды, разворот, боковой ветер) | ✅ Готово |
| 3-Б3 — Боковая САУ | `LateralController`: курс χ→φ_ref, крен φ+p→δa, β (зонд)→δr; s12 разворот, s13 боковой ветер | ✅ Готово |
| 3-Б4 — УС «зонд vs косвенная оценка» | — | ⏸ На паузе (тематика зонда остановлена автором) |

---

## Граф зависимостей модулей

```
config.py  ──────────────────────────────────────────────────────────────┐
    │                                                                     │
    ├── state.py      (air_velocity, kinematic_gamma, total_energy)       │
    ├── aero.py       (CL sigmoid, CD квадратичный, Cm)                  │
    ├── wind.py       (постоянный, порыв, сдвиг по высоте)               │
    ├── sensors.py    (гироскоп, барометр, СВС, зонд УА/УС, GPS, ИНС)    │
    ├── controllers.py (PID, PitchController, SpeedController,            │
    │                   LateralController)                                │
    ├── lqr.py, aua.py (LQR продольного канала, АУА)                      │
    ├── estimators.py (estimate_alpha_indirect — нет зависимостей ↓)     │
    │                                                                     │
    └── dynamics.py → integrators.py → runner.py → scenarios/ ──────────┘
```

Правило: зависимости только вниз. `sensors`, `control`, `estimators` не знают о `runner`.

---

## Интерфейсы модулей

### runner.run()
```python
log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)
# controls_fn(t, state, Va, alpha) -> np.array([delta_e, throttle])
# Возвращает Log: .t, .state, .controls, .Va, .alpha, .E_kin, .E_pot, .wind_vec
```

### control.PitchController
```python
controller = PitchController(aircraft, PitchControlParams())
controller.set_trim_throttle(thr_trim)
controller.reset({'theta': ..., 'q': ..., 'h': ...})
controller.set_pitch_setpoint(theta_ref)          # рад
delta_e, _ = controller.step(t, meas_dict, dt)   # meas: q, theta, h, Va
```

### control.SpeedController  *(добавлен для С6)*
```python
spd_ctrl = SpeedController(aircraft, SpeedControlParams())
spd_ctrl.set_trim_throttle(thr_trim)   # feedforward
spd_ctrl.set_Va_ref(Va_ref)            # уставка м/с
spd_ctrl.reset(Va0)                    # сброс интеграла
throttle = spd_ctrl.step(Va_meas, dt)
```

### estimators.estimate_alpha_indirect  *(добавлен для С7)*
```python
from estimators import estimate_alpha_indirect
# Нужны GPS-скорости от sensors.measure_gps_velocity_earth()
Vx_gps, Vh_gps = measure_gps_velocity_earth(state[U], state[W], state[THETA],
                                             bias=0.0, noise_std=sp.gps_vel_noise, rng=rng)
alpha_est = estimate_alpha_indirect(theta_meas, Vx_gps, Vh_gps)
# alpha_est = theta_meas - arctan(Vh_gps / Vx_gps)
```

---

## Сценарии: сигнатура controls_fn

Каждый сценарий реализует `controls_fn(t, state, Va, alpha) -> np.array([δe, throttle])`:

| Сценарий | delta_e | throttle |
|----------|---------|----------|
| s1–s4 | фиксированный или ПИД тангажа | trim_throttle = const |
| s5 | ПИД тангажа (от высоты) | trim_throttle = const |
| s6 | ПИД тангажа (от высоты) | **SpeedController** |
| s7 | ПИД тангажа (от высоты) | **SpeedController** + оценка УА |

---

## Ключевые числа (текущий config.py: m=10 кг, Va=30 м/с; пересчитано 2026-10-06)

| Величина | Значение |
|----------|----------|
| alpha_trim | 3.01° |
| de_trim | −4.97° |
| thr_trim | 0.432 |
| KH (h → theta_ref) | 0.006 рад/м |
| Va_ref (С6, С7) | 30.0 м/с |
| Ветер в С7 | Vwx = +5 м/с (попутный) |

---

## Интерфейсы бокового канала (Этап 3, Б1–Б3)

```python
from sim.state import V, P, R, PHI, PSI, Y, DA, DR, air_data, earth_velocity
Va, alpha, beta = air_data(state, wind_vec)       # wind_vec = (Vwx, Vwh, Vwy)
Vx, Vy, Vh      = earth_velocity(state)           # земная скорость (север, восток, вверх)

from control.controllers import LateralController, LateralControlParams
lat = LateralController(aircraft, LateralControlParams())
lat.set_course(chi_ref)                            # рад, путевой угол
delta_a, delta_r = lat.step({'chi': ..., 'phi': ..., 'p': ..., 'beta': ...}, dt)

# Полная САУ (высота + Va + курс) — scenarios/lateral_common.py
sau = FullSAU(aircraft, sp, cfg, rng, chi_ref_fn)
sau.bind_wind(lambda h, t: wind(h, t, wind_params))
log = run(sau, aircraft, wind_params, cfg, state0=trim_state(aircraft, cfg))
```

---

## Дорожная карта (фаза «довести симулятор до нормы», с 2026-10-06)

Предложено агентом 2026-10-06, порядок одобряется автором по шагам.
Зонд и всё связанное — на паузе.

| # | Блок | Содержание | Статус |
|---|---|---|---|
| 1 | Параметры | Единый согласованный набор ЛА (целиком Aerosonde ИЛИ целиком ЛА автора); проверить `e_oswald = 0.2`. **Вопрос автору: какой набор?** | ⬜ |
| 1 | Трим | Численная балансировка по B&M гл. 5: Va, γ (набор/снижение), R (вираж) → (α, β, φ, δe, δt, δa, δr); с нелинейным CL | ⬜ |
| 1 | Линеаризация | Один модуль якобиана + модальный анализ (продольные и боковые моды); перевести на него LQR и проверки | ⬜ |
| 2 | Продольная САУ | Упреждающий δe_trim в PitchController (убрать просадку ~7 м на старте); починить s5 (сваливание при m=10 кг); связка высота+скорость (зоны B&M или TECS) | ⬜ |
| 3 | Графика 6DOF | 3D-тренажёр `viz/viewer3d.py` (Ursina, РЕШ-16): проигрывание .flightlog ✅, пилотирование в реальном времени `scenarios/GameScenario3D.py` ✅; 2D-графики φ, ψ, β, p, r, δa, δr и вид сверху в viz/plotting ⬜ | 🔄 |
| 4 | Навигация | Следование по прямой/окружности, полёт по точкам (B&M гл. 10–11) | ⬜ |
| 5 | Реализм | Приводы рулей (инерция, скорость перекладки); турбулентность Драйдена; дрейф гироскопа, частота GPS, задержки | ⬜ |
| 6 | Гигиена | Сходимость по шагу dt; E_kin по воздушной скорости; устаревшие docs; перезапуск s7–s9 | 🔄 docs частично |

---

## Детальная документация

| Документ | Содержание |
|----------|-----------|
| `docs/architecture.md` | Граф зависимостей, главный цикл, структура Log |
| `docs/control.md` | Каскад ПИД, SpeedController, параметры, настройка |
| `docs/continuation.md` | Полный статус + технические детали для разработки |
| `docs/physics.md` | Уравнения движения, аэромодель, балансировка |
| `docs/aerodynamics.md` | Аэродинамические коэффициенты, sigmoid-модель срыва |
| `docs/SENSORS.md` | Модели датчиков, шумы, косвенная оценка УА |
| `TZ_simulator.md` | Техническое задание (исходный документ требований) |

*Этот файл находится в `docs/DEV_PLAN.md`. Ссылка в `CLAUDE.md` обновлена соответственно.*

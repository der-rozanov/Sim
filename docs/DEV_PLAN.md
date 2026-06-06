# DEV_PLAN — Согласованная архитектура и текущий статус

Главный справочный документ разработки. Читать в паре с `CLAUDE.md`.
Детали по каждой теме — в `docs/`.

**Обновлён: 2026-06-06**

---

## Статус: что работает прямо сейчас

| Этап | Содержание | Статус |
|------|-----------|--------|
| 0 — Каркас | Модули, RK4, вектор состояния, пустой прогон | ✅ Готово |
| 1 — База | Физика, ПИД, датчики, С1–С5, индикация УА | ✅ Готово |
| 2а — Контур скорости | `SpeedController`, С6 (контроль высоты + Va) | ✅ Готово |
| 2б — Оценка УА | `estimators.py`, С7 (зонд vs ИНС+GPS при ветре) | ✅ Готово |
| 2в — Парный прогон | «С зондом vs без» на сдвиге ветра — **тезис главы 8** | 🔄 Следующий шаг |
| 3 — Фичи | Боковой канал, наблюдатель, LQR | ⬜ Не начат |

---

## Граф зависимостей модулей

```
config.py  ──────────────────────────────────────────────────────────────┐
    │                                                                     │
    ├── state.py      (air_velocity, kinematic_gamma, total_energy)       │
    ├── aero.py       (CL sigmoid, CD квадратичный, Cm)                  │
    ├── wind.py       (постоянный, порыв, сдвиг по высоте)               │
    ├── sensors.py    (гироскоп, барометр, СВС, зонд УА, GPS)            │
    ├── control.py    (PID, PitchController, SpeedController)             │
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

## Ключевые числа (параметры-аналог Aerosonde, Va=30 м/с)

| Величина | Значение |
|----------|----------|
| alpha_trim | 1.92° |
| de_trim | −4.14° |
| thr_trim | 0.398 |
| KH (h → theta_ref) | 0.006 рад/м |
| Va_ref (С6, С7) | 30.0 м/с |
| Ветер в С7 | Vwx = +5 м/с (попутный) |

---

## Следующий шаг: парный прогон (ТЗ раздел 6.4)

Сценарий `s8_probe_vs_noProbe.py` (рабочее название):

1. **Один сценарий, два прогона**: снижение (база — С3) + вход в слой сдвига ветра
   (`WindParams(dV_shear=X, h_shear_lo=80, h_shear_hi=120)`)

2. **Общие датчики** — одинаковый seed; источник УА — единственное различие:
   ```python
   # Прогон 1 — «с зондом»
   alpha_src = measure_angle_of_attack(alpha, sp.probe_bias, sp.probe_noise, rng)

   # Прогон 2 — «без зонда»
   Vx, Vh = measure_gps_velocity_earth(state[U], state[W], state[THETA],
                                        0.0, sp.gps_vel_noise, rng)
   alpha_src = estimate_alpha_indirect(theta_meas, Vx, Vh)
   ```

3. **Штилевой контроль**: без ветра обе CAУ должны практически совпасть.

4. **Метрики сравнения** (обязательно для главы 8):
   - Максимальное отклонение УА от балансировочного
   - Время выхода на установившийся режим после слоя сдвига
   - Близость к alpha_warning (15°) и alpha_crit (20°)

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

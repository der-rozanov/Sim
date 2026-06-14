# АУА — Автомат Углов Атаки

**Модуль:** `control/aua.py`  
**Класс:** `AngleOfAttackProtector`  
**Назначение:** надзорный защитный уровень над каскадным контроллером тангажа.
При выходе угла атаки за критическое значение перехватывает управление и
выводит ЛА из опасного режима.

---

## 1. Место в архитектуре управления

```
Миссия / контур высоты
        │
        ▼  theta_ref (уставка тангажа)
 ┌──────────────┐
 │     АУА      │◄── alpha_meas (зонд УА)
 │ (надзорный)  │
 └──────┬───────┘
        │ theta_ref* (возможно переопределённый)
        │ thr_trim*  (возможно переопределённый)
        ▼
 ┌──────────────────┐
 │ PitchController  │◄── q_meas, theta_meas, h_meas, Va_meas
 │ (theta + q ПИД)  │
 └──────┬───────────┘
        │ [delta_e, throttle]
        │            ▲
        │            └── force_throttle (прямое переопределение от АУА)
        ▼
     ЛА / sim
```

В нормальном режиме АУА прозрачен: пропускает `theta_ref` и `thr_trim` без
изменений. При срабатывании защиты АУА **перехватывает** управление, подставляя
свои значения.

---

## 2. Схема включения в сценарий

```python
from control.aua import AngleOfAttackProtector, AUAParams, AUAOutput, AUAState

aua_params = AUAParams(
    enabled           = True,
    alpha_warn        = aircraft.alpha_warning,   # 15°
    alpha_crit        = aircraft.alpha_crit,       # 20°
    alpha_exit        = np.radians(8.0),           # 8° (гистерезис)
    theta_recovery    = np.radians(-7.0),          # нос вниз −7°
    throttle_recovery = 1.0,                       # газ 100%
)
aua = AngleOfAttackProtector(aircraft, aua_params)

# Внутри controls_fn:
aua_out = aua.step(alpha_probe, theta_mission, thr_trim, cfg.dt)
controller.set_pitch_setpoint(aua_out.theta_ref)
controller.set_trim_throttle(aua_out.thr_trim)
ctrl = controller.step(t, meas, cfg.dt)
if aua_out.force_throttle is not None:
    ctrl[1] = aua_out.force_throttle   # прямая команда, минует h_Kp
```

`force_throttle` — ключевой механизм: он подменяет тягу **после** вызова
`controller.step()`, поэтому h_Kp-контур PitchController не может снизить тягу
во время восстановления.

---

## 3. Автомат состояний

```
            alpha < alpha_warn
   ┌─────────────────────────────┐
   │                             ▼
NORMAL ──────────────────────► NORMAL
   │   alpha_warn ≤ alpha        │
   └──────────────► WARNING ─────┘ alpha < alpha_warn
                     │
                     │ alpha ≥ alpha_crit
                     ▼
                  CRITICAL ◄──── relapse (из RECOVERING)
                     │
                     │ alpha < alpha_crit
                     ▼
                RECOVERING
                     │
                     │ alpha < alpha_exit (8°)
                     ▼
                  NORMAL
```

| Состояние | Условие входа | Команды | Смысл |
|---|---|---|---|
| **NORMAL** | alpha < alpha_warn | theta_ref без изм. | Нормальный полёт |
| **WARNING** | alpha_warn ≤ alpha < alpha_crit | theta_ref − 3° | Мягкая коррекция |
| **CRITICAL** | alpha ≥ alpha_crit | theta = −7°, thr = 1.0 | Жёсткий перехват |
| **RECOVERING** | после CRITICAL, alpha < alpha_crit | theta = −7°, thr = 1.0 | Удержание защиты |

Гистерезис: выход из RECOVERING требует alpha < **8°** (не 20°). Это
предотвращает «мерцание» при alpha, болтающемся около alpha_crit.

---

## 4. Параметры AUAParams

| Параметр | По умолчанию | Описание |
|---|---|---|
| `enabled` | `True` | Выключить → АУА прозрачен (обратная совместимость) |
| `alpha_warn` | 0.2618 рад ≈ 15° | Порог предупреждения |
| `alpha_crit` | 0.3491 рад ≈ 20° | Порог перехвата управления |
| `alpha_exit` | 0.1745 рад ≈ 10° | Порог выхода из восстановления |
| `theta_recovery` | −0.1222 рад ≈ −7° | Уставка тангажа при перехвате |
| `throttle_recovery` | 1.0 | Тяга при перехвате |
| `theta_warn_delta` | −0.0524 рад ≈ −3° | Коррекция theta_ref в WARNING |

Пороги согласованы с `AircraftParams.alpha_warning` / `.alpha_crit` из
`sim/config.py` — рекомендуется передавать их явно (см. пример выше), а не
полагаться на умолчания.

---

## 5. Физика восстановления

При перехвате АУА устанавливает `theta = −7°`, `throttle = 1.0`.

**Сила тяги при T_max = 40 Н:**

```
fx = T·cos(theta) − D + m·g·sin(7°)   ← тяга + гравитация помогают разгону
fz = −L + m·g·cos(theta)              ← подъёмная сила восстанавливается по мере роста Va
```

При `theta = +15°` без АУА:
```
fx = T·cos(15°) − D − m·g·sin(15°)   ← гравитация МЕШАЕТ разгону
```

Разница в горизонтальной силе: ≈ **46 Н** в пользу АУА (при T=40 Н, m=10 кг).
Это объясняет, почему восстановление с АУА занимает 4–6 с, а без АУА ЛА
остаётся в срыве на всё время действия возмущения.

---

## 6. Роль зонда прямого измерения УА

АУА читает `alpha_probe` — сигнал непосредственно от датчика угла атаки:

```python
alpha_probe = measure_angle_of_attack(alpha_true, sp.probe_bias, sp.probe_noise, rng)
```

**Почему косвенная оценка хуже:**

| Источник alpha | Задержка | Смещение при ветре | Надёжность при манёврах |
|---|---|---|---|
| Зонд (прямой) | ~0.01 с (1 шаг) | минимальное | высокая |
| ИНС+GPS (theta − gamma) | 0.3–0.5 с | до ±2° при Vwx | деградирует при наборе/снижении |

При задержке 0.3–0.5 с и alpha, растущей со скоростью ~10°/с,
косвенная оценка даёт запоздание ~3–5° — достаточно, чтобы пропустить
момент входа в зону alpha_crit.

---

## 7. Визуальные маркеры

`AUA_COLORS` и `AUA_LABELS` из модуля используются в графиках:

| Состояние | Цвет | Метка |
|---|---|---|
| NORMAL | green | НОРМ |
| WARNING | goldenrod | ПРЕД |
| CRITICAL | darkorange | КРИТ |
| RECOVERING | royalblue | ВОССТ |

Пример субплота состояния АУА — в `scenarios/s10_aua_gust_protection.py`
(панель «Состояние АУА»).

---

## 8. Включение в новый сценарий

Минимальный шаблон (АУА с отключаемостью через флаг):

```python
from control.aua import AngleOfAttackProtector, AUAParams, AUAOutput, AUAState

USE_AUA = True   # переключатель

ap  = AUAParams(enabled=USE_AUA,
                alpha_warn=aircraft.alpha_warning,
                alpha_crit=aircraft.alpha_crit)
aua = AngleOfAttackProtector(aircraft, ap)

def controls_fn(t, state, Va, alpha_true):
    ...
    theta_mission = ...   # ваш P/PID контур высоты
    base_thr      = ...   # ваша базовая тяга

    aua_out = aua.step(alpha_true, theta_mission, base_thr, cfg.dt)
    controller.set_pitch_setpoint(aua_out.theta_ref)
    controller.set_trim_throttle(aua_out.thr_trim)
    ctrl = controller.step(t, meas, cfg.dt)
    if aua_out.force_throttle is not None:
        ctrl[1] = aua_out.force_throttle
    return ctrl
```

При `enabled=False` все вызовы `aua.step()` возвращают входные значения без
изменений — накладные расходы нулевые, поведение идентично контуру без АУА.

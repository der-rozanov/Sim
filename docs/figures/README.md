# Иллюстрации документации

Картинки лежат в `docs/img/`; всё, что можно, рисуется скриптами отсюда — после
изменения кода их можно перерисовать.

| Скрипт | Что делает | Картинки |
|---|---|---|
| `diagrams.py` | схемы (matplotlib): `python docs/figures/diagrams.py [имя]` | `overview`, `keys`, `sau_simple`, `own_scenario`, `frames`, `forces` |
| `model_plots.py` | характеристики модели по коду `sim/`: печатает таблицы для physics.md и aerodynamics.md и рисует графики | `aero_coeffs`, `polar`, `envelope`, `propeller`, `modes`, `actuator`, `gear` |
| `sau_diagrams.py` | структурные схемы САУ (тот же стиль, что `diagrams.py`): `python docs/figures/sau_diagrams.py [имя]` | `sau_structure`, `sau_pitch`, `sau_tecs`, `sau_split`, `sau_roll`, `sau_lateral`, `sau_aoa`, `sensor_model` |
| `sau_plots.py` | переходные процессы САУ на полной модели против линейной теории; печатает числа для таблиц control.md и SENSORS.md: `python docs/figures/sau_plots.py [steps outer tecs course coord aoa navfield route bandwidth sensors]` (~1.5 мин) | `sau_steps`, `sau_outer_steps`, `sau_tecs_vs_split`, `sau_course`, `sau_coord`, `sau_aoa_demo`, `sau_nav_field`, `sau_route`, `sau_bandwidth`, `sensors_noise` |
| `viewer_shots.py` | прогоняет сценарии и сохраняет кадр 2D-просмотра | `viewer_s4`, `viewer_s8` |
| `grab_screen.py` | запускает интерактивную программу, по сценарию команд (ждать, двигать окна, щёлкать, нажимать) снимает окна; только Windows | `bench_*`, `lab_editor`, `map_route`, `game_route_3d` |

3D-кадры проигрывателя снимаются им самим:

```bash
python viz/viewer3d.py <лог s8> --at 22 --cam 1 --shot docs/img/v3d_turn_cam1.png
python viz/viewer3d.py <лог s8> --at 22 --cam 2 --shot docs/img/v3d_turn_cockpit.png
python viz/viewer3d.py <лог s10, ветер> --at 150 --cam 4 --shot docs/img/v3d_route_top.png
python scenarios/GameScenario3D.py --map realmap_osm --h0 120 --shot docs/img/map3d_realmap_osm.png
```

Виды карт сверху (`maptop_*`) — копии `viz/map_cache/top_<карта>.png`, уменьшенные до
600 × 600. Спутниковые карты (`realmap`, `realmap_large`) в документацию не включаются:
условия Esri не разрешают распространять снимки. Скриншоты 3D хранятся в JPG.

Пример сценария `grab_screen.py` (стенд, ступенька курса +30°):

```
wait 12
move "=Стенд САУ" 0 0 1900 1250
click "=Стенд САУ" 482 630
wait 13
shot "=Стенд САУ" docs/img/bench_roll.png
```

Координаты щелчков зависят от размера окна и масштаба экрана: перед съёмкой снимите
окно командой `shot` и найдите нужные кнопки на снимке.

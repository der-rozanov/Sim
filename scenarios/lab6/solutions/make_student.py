# -*- coding: utf-8 -*-
"""
Генерация студенческих версий заданий из эталонных решений.

Блок между строками «# >>> SOLUTION» и «# <<< SOLUTION» заменяется на заглушку
(«# ваш код здесь» + «return None»). Всё остальное копируется как есть.

Запуск (из корня проекта):  python scenarios/lab6/solutions/make_student.py
Результат: scenarios/lab6/L1_pitch_hold.py, L2_..., L3_...

ВАЖНО: папку solutions/ студентам не раздавать.
"""

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FILES = ["L1_pitch_hold.py", "L2_altitude_hold.py", "L3_airspeed_hold.py"]

BLOCK = re.compile(r"^([ \t]*)# >>> SOLUTION\n.*?^[ \t]*# <<< SOLUTION\n", re.S | re.M)


def strip(src: str) -> str:
    return BLOCK.sub(lambda m: f"{m.group(1)}# ваш код здесь\n{m.group(1)}return None\n", src)


for name in FILES:
    src_path = os.path.join(HERE, name)
    if not os.path.exists(src_path):
        continue
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    out = strip(src)
    n = len(BLOCK.findall(src))
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(out)
    print(f"{name}: вырезано блоков решения — {n}")

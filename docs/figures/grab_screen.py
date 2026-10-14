# -*- coding: utf-8 -*-
"""
Скриншоты окон интерактивных программ для документации (docs/img/). Только Windows.

    python docs/figures/grab_screen.py СЦЕНАРИЙ.txt -- python scenarios/Bench3D.py [...]

Запускает программу и выполняет команды из файла сценария (по строке):

    wait 5                         ждать 5 с
    move ЗАГОЛОВОК X Y W H         поставить окно (часть заголовка; '=…' — целиком) в X, Y, W × H
    key  ЗАГОЛОВОК КЛАВИША [С]     нажать клавишу (буква/цифра, space) в окне; С — держать, с
    click ЗАГОЛОВОК X Y            щелчок в точке окна (от левого верхнего угла клиентской части)
    wheel ЗАГОЛОВОК X Y N          N щелчков колеса вверх (N < 0 — вниз) над точкой окна
    shot ЗАГОЛОВОК ФАЙЛ.png        снимок окна (рамка — по клиентской части)
    screen ФАЙЛ.png                снимок всего экрана

В конце закрывает программу со всеми дочерними процессами (окна схемы, карты).
Снимки — Pillow (ImageGrab; ставится вместе с ursina).
"""
import ctypes
import ctypes.wintypes as wt
import shlex
import subprocess
import sys
import time

from PIL import ImageGrab

user32 = ctypes.windll.user32
ctypes.windll.shcore.SetProcessDpiAwareness(2)      # координаты — в физических пикселях


def find(title):
    """Окно верхнего уровня, в заголовке которого есть title ('=title' — заголовок целиком)."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if (buf.value == title[1:]) if title.startswith("=") else (title in buf.value):
                found.append(hwnd)
        return True

    for _ in range(60):                              # окно могло ещё не открыться
        user32.EnumWindows(cb, 0)
        if found:
            return found[0]
        time.sleep(0.5)
    raise RuntimeError(f"окно «{title}» не найдено")


def client_box(hwnd):
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    p = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    return p.x, p.y, p.x + r.right, p.y + r.bottom


def front(hwnd):
    user32.ShowWindow(hwnd, 9)                       # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.4)


def press(hwnd, key, hold=0.15):
    front(hwnd)
    vk = {"space": 0x20, "esc": 0x1B}.get(key.lower(), ord(key.upper()))
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(hold)
    user32.keybd_event(vk, 0, 2, 0)


def click(hwnd, x, y):
    front(hwnd)
    x0, y0, _, _ = client_box(hwnd)
    user32.SetCursorPos(x0 + x, y0 + y)
    user32.mouse_event(2, 0, 0, 0, 0)
    user32.mouse_event(4, 0, 0, 0, 0)
    time.sleep(0.2)


def wheel(hwnd, x, y, n):
    front(hwnd)
    x0, y0, _, _ = client_box(hwnd)
    user32.SetCursorPos(x0 + x, y0 + y)
    for _ in range(abs(n)):
        user32.mouse_event(0x0800, 0, 0, 120 if n > 0 else -120, 0)
        time.sleep(0.2)


def main():
    script = open(sys.argv[1], encoding="utf-8").read().splitlines()
    cmd = sys.argv[sys.argv.index("--") + 1:]
    proc = subprocess.Popen(cmd)
    try:
        for line in script:
            a = shlex.split(line, comments=True)
            if not a:
                continue
            op = a[0]
            if op == "wait":
                time.sleep(float(a[1]))
            elif op == "move":
                user32.MoveWindow(find(a[1]), *map(int, a[2:6]), True)
            elif op == "key":
                press(find(a[1]), a[2], float(a[3]) if len(a) > 3 else 0.15)
            elif op == "click":
                click(find(a[1]), int(a[2]), int(a[3]))
            elif op == "wheel":
                wheel(find(a[1]), int(a[2]), int(a[3]), int(a[4]))
            elif op == "shot":
                h = find(a[1])
                front(h)
                ImageGrab.grab(bbox=client_box(h), all_screens=True).save(a[2])
                print("сохранено", a[2])
            elif op == "screen":
                ImageGrab.grab().save(a[1])
                print("сохранено", a[1])
    finally:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


if __name__ == "__main__":
    main()

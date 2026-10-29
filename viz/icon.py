"""Иконка окон проекта вместо стандартной иконки Python (рисует docs/figures/logo.py)."""
import os
import sys

ICON_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
ICON_ICO = ICON_PNG[:-3] + "ico"


def set_app_id():
    """Windows: своя группа на панели задач — иначе там иконка python.exe."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("uav.sim")


def set_win_icon(hwnd: int):
    """Иконка окна по дескриптору (Ursina). Panda3D не читает .ico по пути с кириллицей,
    поэтому — напрямую через WinAPI (LoadImageW понимает Unicode)."""
    if sys.platform != "win32":
        return
    import ctypes
    u32 = ctypes.windll.user32
    u32.LoadImageW.restype = ctypes.c_void_p
    u32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    for which, size in ((0, 16), (1, 32)):              # ICON_SMALL, ICON_BIG
        h = u32.LoadImageW(None, ICON_ICO, 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
        u32.SendMessageW(hwnd, 0x80, which, h)                    # WM_SETICON


def set_tk_icon(root):
    """Иконка окна tkinter и всех его дочерних окон."""
    import tkinter as tk
    set_app_id()
    root._icon_img = tk.PhotoImage(master=root, file=ICON_PNG)   # ссылка — иначе сборщик мусора
    root.iconphoto(True, root._icon_img)

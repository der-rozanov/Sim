# -*- coding: utf-8 -*-
"""
Просмотр полётных логов (.flightlog) — единый 2D-показ результатов прогонов.

Запуск:
    python viz/viewer.py                                   # диалог выбора файла
    python viz/viewer.py results/<файл>.flightlog          # анимация
    python viz/viewer.py results/<файл>.flightlog --speed 4 --static --save out.gif

Из сценария:
    from viz.viewer import show_flightlog
    path = logger.save(log, h_ref=..., theta_ref=...)
    show_flightlog(path)                       # несколько путей — несколько окон

Набор панелей выбирается по логу:
  продольный полёт — вид сбоку (x–h) и h, Va, θ, α, δe, δt, P;
  манёвр по курсу (или в логе есть уставка курса / маршрут) — вид сверху (E–N)
  и, кроме продольных, χ, φ, β, δa, δr; если размах по высоте больше бокового
  (пилотаж, s11) — вид сбоку; при |θ| > 60° — в равном масштабе осей.
Уставки (*_ref) — чёрный пунктир, команды до приводов (*_cmd) — тонкая линия
цвета канала, события — вертикальные линии; на α — зоны предупреждения и
критического УА, рамка справки окрашивается по индикатору УА.

Управление анимацией: Пробел — пауза/пуск, R — с начала.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flight_logger import load_log, print_log_info
from viz.icon import set_tk_icon

plt.rcParams["font.family"] = "DejaVu Sans"

ALERT_BG    = ["#e8f5e9", "#fff9c4", "#ffe0b2", "#ffcdd2"]   # НОРМ, ПРЕД, КРИТ, СРЫВ
ALERT_LABEL = ["НОРМ", "ПРЕД", "КРИТ", "СРЫВ"]


def _get(d, key):
    """Канал лога или None (нет в логе или весь NaN)."""
    a = d.get(key)
    if a is None or (a.dtype.kind == "f" and np.all(np.isnan(a))):
        return None
    return a


def _deg(a):
    return None if a is None else np.degrees(a)


def _course_deg(a):
    """Угол курса в градусах в диапазоне ±180°; на переходе через ±180° — разрыв (NaN)."""
    if a is None:
        return None
    c = np.degrees((a + np.pi) % (2.0 * np.pi) - np.pi)
    c[1:][np.abs(np.diff(c)) > 180.0] = np.nan
    return c


# ---------------------------------------------------------------------------
class FlightLogViewer:
    """
    Окно одного полётного лога.

    data       : dict из flight_logger.load_log()
    anim_speed : множитель скорости воспроизведения (1 = реальное время)
    fps        : кадров в секунду
    """

    def __init__(self, data: dict, anim_speed: float = 2.0, fps: int = 25):
        self.d, self.meta = data, data["meta"]
        self.speed, self.fps = anim_speed, fps
        self.t = data["t"]
        self.alert = data["alert"].astype(int)
        self.lateral = self._is_lateral()
        self.side_view = self._is_side_view()
        self.fig = None
        self._anim = None
        self._paused = False

    # --- выбор панелей --------------------------------------------------------
    def _is_lateral(self):
        d = self.d
        if self.meta.get("waypoints") or _get(d, "chi_ref") is not None:
            return True
        psi = np.degrees(np.unwrap(d["psi"])) if "psi" in d else np.zeros(1)
        y = d["y"] if "y" in d else np.zeros(1)
        return np.ptp(psi) > 5.0 or np.ptp(y) > 20.0

    def _is_side_view(self):
        """Вид сбоку, если полёт продольный или размах по высоте больше бокового (пилотаж)."""
        d = self.d
        if not self.lateral:
            return True
        return np.ptp(d["h"]) > np.ptp(d["y"]) and not self.meta.get("waypoints")

    def _lon_panels(self):
        d, m = self.d, self.meta
        al = m.get("alert_levels", {}).get("thresholds_deg", {})
        thr_trim = m.get("trim", {}).get("throttle")
        alpha_extra = [(_deg(_get(d, "alpha_probe")), "зонд", "steelblue"),
                       (_deg(_get(d, "alpha_est")), "ИНС+GPS", "tomato")]
        panels = [
            dict(label="h, м", y=d["h"], color="royalblue", ref=_get(d, "h_ref"), pad=5.0),
            dict(label="Va, м/с", y=d["Va"], color="steelblue", ref=_get(d, "Va_ref"), pad=1.0),
            dict(label="θ, °", y=_deg(d["theta"]), color="seagreen", ref=_deg(_get(d, "theta_ref"))),
            dict(label="α, °", y=_deg(d["alpha_true"]), color="crimson",
                 extra=[e for e in alpha_extra if e[0] is not None],
                 zones=(al.get("warning"), al.get("crit"), al.get("stall"))),
            dict(label="δe, °", y=_deg(d["delta_e"]), color="saddlebrown",
                 cmd=_deg(_get(d, "delta_e_cmd"))),
            dict(label="δt", y=d["throttle"], color="darkgreen", lim=(-0.05, 1.05),
                 hline=(thr_trim, f"трим {thr_trim:.2f}") if thr_trim is not None else None),
        ]
        if _get(d, "P_prop") is not None:
            panels.append(dict(label="P винта, Вт", y=d["P_prop"], color="purple", pad=5.0))
        return panels

    def _lat_panels(self):
        d = self.d
        chi = d["chi"] if "chi" in d else d["psi"]
        return [
            dict(label="χ, °", y=_course_deg(chi), color="teal",
                 ref=_course_deg(_get(d, "chi_ref")), clip=(-190.0, 190.0)),
            dict(label="φ, °", y=_deg(d["phi"]), color="darkorange", ref=_deg(_get(d, "phi_ref"))),
            dict(label="β, °", y=_deg(d["beta_true"]), color="slateblue"),
            dict(label="δa, °", y=_deg(d["delta_a"]), color="olive", cmd=_deg(_get(d, "delta_a_cmd"))),
            dict(label="δr, °", y=_deg(d["delta_r"]), color="indianred", cmd=_deg(_get(d, "delta_r_cmd"))),
        ]

    # --- построение фигуры ----------------------------------------------------
    def _title(self):
        m = self.meta
        w = m.get("wind", {})
        wind = f"ветер Vwx={w.get('Vw_const', 0):+.1f}"
        if w.get("Vw_cross"):
            wind += f", Vwy={w['Vw_cross']:+.1f}"
        if w.get("gust_amp"):
            wind += f", порыв {w['gust_amp']:.1f}"
        return (f"{m.get('scenario', '?')}  —  {m.get('description', '')}\n"
                f"{wind} м/с   |   ЛА: {m.get('aircraft', {}).get('name', '?')}   |   "
                f"сохранён {m.get('saved_at', '?')}")

    def build(self, animated: bool = True):
        cols = [self._lon_panels()] + ([self._lat_panels()] if self.lateral else [])
        self.fig = fig = plt.figure(figsize=(19 if self.lateral else 15, 11))
        win = getattr(fig.canvas.manager, "window", None)      # окно есть только у TkAgg
        if hasattr(win, "iconphoto"):
            set_tk_icon(win)
        fig.suptitle(self._title(), fontsize=9, fontweight="bold")
        gs = fig.add_gridspec(1, 1 + len(cols), width_ratios=[1.3] + [1] * len(cols),
                              wspace=0.28, left=0.05, right=0.98, top=0.92, bottom=0.06)

        ax_tr = fig.add_subplot(gs[0, 0])
        self._draw_track(ax_tr, animated)

        self._dyn, self._cursors = [], []      # (линия, точка, y) и курсоры времени
        first = True
        for c, panels in enumerate(cols):
            sub = gs[0, c + 1].subgridspec(len(panels), 1, hspace=0.12)
            axes = []
            for k, pn in enumerate(panels):
                ax = fig.add_subplot(sub[k, 0], sharex=axes[0] if axes else None)
                axes.append(ax)
                self._draw_panel(ax, pn, animated, legend_events=first)
                first = False
            for ax in axes[:-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
            axes[-1].set_xlabel("t, с", fontsize=9)

        if animated:
            self._start_animation()
            fig.text(0.5, 0.005, "Пробел — пауза/пуск  |  R — с начала",
                     ha="center", fontsize=8, color="gray")
        return fig

    def _draw_track(self, ax, animated):
        d = self.d
        if not self.side_view:   # вид сверху: восток → вправо, север → вверх
            X, Y = d["y"], d["x"]
            ax.set_xlabel("E (восток), м", fontsize=9)
            ax.set_ylabel("N (север), м", fontsize=9)
            ax.set_title("Траектория, вид сверху", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            wps = self.meta.get("waypoints") or []
            for k, (n, e, _h) in enumerate(wps, 1):
                ax.plot(e, n, "D", color="dodgerblue", ms=7, zorder=4)
                ax.annotate(str(k), (e, n), textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color="dodgerblue")
            self._heading = np.degrees(d["psi"])
        else:              # вид сбоку: дальность → вправо, высота → вверх
            X, Y = d["x"], d["h"]
            ax.set_xlabel("x (север), м", fontsize=9)
            ax.set_ylabel("h, м", fontsize=9)
            # фигуры пилотажа (|θ| > 60° — только в пилотаже) — в равном масштабе осей
            if np.abs(np.degrees(d["theta"])).max() > 60.0:
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title("Траектория, вид сбоку", fontsize=9)
            else:
                ax.set_title("Траектория, вид сбоку (масштабы осей разные)", fontsize=9)
            self._heading = 90.0 - np.degrees(d["theta"])   # нос по +x при θ = 0
        ax.grid(True, ls="--", alpha=0.5)
        ax.plot(X, Y, color="royalblue", lw=2.0 if not animated else 1.2,
                alpha=1.0 if not animated else 0.35, zorder=2)
        ax.plot(X[0], Y[0], "go", ms=8, zorder=5, label="старт")
        ax.plot(X[-1], Y[-1], "rs", ms=8, zorder=5, label="финиш")
        ax.legend(fontsize=8, loc="upper left")
        self._track = (X, Y)
        self._tr_line, = ax.plot([], [], color="royalblue", lw=2.0, zorder=3)
        self._tr_mark, = ax.plot([], [], marker=(3, 0, 0), ms=13, color="navy", zorder=6)
        self._info = ax.text(0.98, 0.98, "", transform=ax.transAxes, fontsize=8.5,
                             ha="right", va="top", family="monospace",
                             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                       edgecolor="gray", alpha=0.9))
        if not animated:
            self._show_frame(len(self.t) - 1)

    def _draw_panel(self, ax, pn, animated, legend_events):
        t, y, col = self.t, pn["y"], pn["color"]
        ax.grid(True, ls="--", alpha=0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(pn["label"], fontsize=8)
        ax.set_xlim(t[0], t[-1])

        series = [y] + [a for a in (pn.get("ref"), pn.get("cmd")) if a is not None] \
                     + [e[0] for e in pn.get("extra", [])]
        lo = min(np.nanmin(a) for a in series)
        hi = max(np.nanmax(a) for a in series)
        warn, crit, stall = pn.get("zones") or (None, None, None)
        if warn is not None:
            hi = max(hi, warn + 1.0)            # линия предупреждения видна всегда
        pad = max(0.15 * (hi - lo), pn.get("pad", 0.5))
        clip = pn.get("clip", (-np.inf, np.inf))
        ax.set_ylim(*(pn.get("lim") or (max(lo - pad, clip[0]), min(hi + pad, clip[1]))))
        top = ax.get_ylim()[1]
        if warn is not None:
            ax.axhspan(warn, crit, color="orange", alpha=0.12, zorder=0)
            ax.axhspan(crit, max(top, crit), color="red", alpha=0.08, zorder=0)
            for v, c, s, name in ((warn, "orange", "--", "пред"), (crit, "red", "--", "крит"),
                                  (stall, "darkred", "-", "срыв")):
                if v < top:
                    ax.axhline(v, color=c, ls=s, lw=1.0, alpha=0.8, label=f"{name} {v:.0f}°")

        if pn.get("hline"):
            v, lbl = pn["hline"]
            ax.axhline(v, color="gray", ls="--", lw=0.9, alpha=0.6, label=lbl)
        if pn.get("ref") is not None:
            ax.plot(t, pn["ref"], color="black", ls="--", lw=0.9, alpha=0.45, zorder=1,
                    label="уставка")
        if pn.get("cmd") is not None:
            ax.plot(t, pn["cmd"], color=col, lw=0.7, alpha=0.45, label="команда")
        for arr, lbl, c in pn.get("extra", []):
            ax.plot(t, arr, color=c, lw=0.9, alpha=0.6, label=lbl)
        for ev in self.meta.get("events", []):
            ax.axvline(ev["t"], color=ev.get("color", "gray"), ls=":", lw=1.0, alpha=0.75,
                       label=f"{ev['label']} t={ev['t']:.0f} с" if legend_events else None)

        if animated:
            ax.plot(t, y, color=col, lw=1.0, alpha=0.3)
            line, = ax.plot([], [], color=col, lw=1.8)
            point, = ax.plot([], [], "o", color=col, ms=4, zorder=5)
            self._dyn.append((line, point, y))
            self._cursors.append(ax.axvline(t[0], color="gray", lw=0.8, ls=":", alpha=0.6))
        else:
            ax.plot(t, y, color=col, lw=1.6)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.5, loc="upper right", ncol=2)

    # --- кадр -----------------------------------------------------------------
    def _info_text(self, i):
        d = self.d
        lv = int(self.alert[i])
        rows = [f"t  = {self.t[i]:6.1f} с",
                f"h  = {d['h'][i]:6.1f} м",
                f"Va = {d['Va'][i]:6.1f} м/с",
                f"θ  = {np.degrees(d['theta'][i]):+6.1f}°",
                f"α  = {np.degrees(d['alpha_true'][i]):+6.1f}°  {ALERT_LABEL[lv]}"]
        if self.lateral:
            chi = d["chi"][i] if "chi" in d else d["psi"][i]
            rows += [f"φ  = {np.degrees(d['phi'][i]):+6.1f}°",
                     f"χ  = {np.degrees(chi) % 360:6.1f}°",
                     f"β  = {np.degrees(d['beta_true'][i]):+6.1f}°"]
        if "P_prop" in d:
            rows.append(f"P  = {d['P_prop'][i]:6.0f} Вт")
        return "\n".join(rows), lv

    def _show_frame(self, i):
        X, Y = self._track
        self._tr_line.set_data(X[:i + 1], Y[:i + 1])
        self._tr_mark.set_data([X[i]], [Y[i]])
        self._tr_mark.set_marker((3, 0, -self._heading[i]))
        text, lv = self._info_text(i)
        self._info.set_text(text)
        self._info.get_bbox_patch().set_facecolor(ALERT_BG[lv])

    def _draw_at(self, i):
        """Кадр анимации: всё, что пройдено к отсчёту i."""
        ts = self.t[:i + 1]
        self._show_frame(i)
        for line, point, y in self._dyn:
            line.set_data(ts, y[:i + 1])
            point.set_data([self.t[i]], [y[i]])
        for c in self._cursors:
            c.set_xdata([self.t[i]])

    def goto(self, t: float):
        """Показать момент t, с (для снимков: build(animated=True), goto(t), fig.savefig)."""
        self._draw_at(int(np.searchsorted(self.t, t)))

    def _start_animation(self):
        dt = self.t[1] - self.t[0] if len(self.t) > 1 else 0.01
        stride = max(1, int(round(self.speed / self.fps / dt)))
        frames = list(range(0, len(self.t), stride))
        artists = [self._tr_line, self._tr_mark, self._info] + self._cursors \
            + [a for line, point, _ in self._dyn for a in (line, point)]

        def update(fn):
            if not self._paused:
                self._draw_at(frames[fn])
            return artists

        def on_key(event):
            if event.key == " ":
                self._paused = not self._paused
            elif event.key in ("r", "R"):
                self._paused = False
                self._anim.frame_seq = self._anim.new_frame_seq()

        self.fig.canvas.mpl_connect("key_press_event", on_key)
        self._anim = FuncAnimation(self.fig, update, frames=len(frames),
                                   interval=1000.0 / self.fps, blit=True, repeat=True)

    def save(self, path: str):
        """Сохранить анимацию: .gif (Pillow) или .mp4 (ffmpeg)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        try:
            if path.endswith(".gif"):
                from matplotlib.animation import PillowWriter
                self._anim.save(path, writer=PillowWriter(fps=self.fps))
            else:
                self._anim.save(path, fps=self.fps, extra_args=["-vcodec", "libx264"])
            print(f"Сохранено: {path}")
        except Exception as e:
            print(f"Не удалось сохранить {path}: {e}")


# ---------------------------------------------------------------------------
def show_flightlog(*paths, speed: float = 2.0, fps: int = 25,
                   static: bool = False, save_path: str = None):
    """
    Показать один или несколько .flightlog (каждый — своё окно) и ждать закрытия окон.
    save_path — сохранить анимацию первого лога (.gif / .mp4).
    """
    viewers = []
    for p in paths:
        v = FlightLogViewer(load_log(p), anim_speed=speed, fps=fps)
        v.build(animated=not static)
        viewers.append(v)
    if save_path and not static and viewers:
        viewers[0].save(save_path)
    plt.show()
    return viewers


def _open_dialog() -> str:
    """Диалог выбора файла; '' если отменён или недоступен."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Открыть полётный лог", initialdir=os.path.join(_ROOT, "results"),
            filetypes=[("Flight log", "*.flightlog"), ("NPZ", "*.npz"), ("Все файлы", "*.*")])
        root.destroy()
        return path
    except Exception as e:
        print(f"Диалог недоступен: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Просмотр полётных логов (.flightlog)")
    parser.add_argument("file", nargs="?", default=None, help="путь к .flightlog")
    parser.add_argument("--speed", type=float, default=2.0, help="скорость анимации (2.0)")
    parser.add_argument("--fps", type=int, default=25, help="кадров в секунду (25)")
    parser.add_argument("--static", action="store_true", help="графики без анимации")
    parser.add_argument("--save", default=None, help="сохранить анимацию в .gif / .mp4")
    args = parser.parse_args()

    path = args.file or _open_dialog()
    if not path:
        print("Файл не выбран.")
        return
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path += ".npz"
    print_log_info(load_log(path))
    show_flightlog(path, speed=args.speed, fps=args.fps, static=args.static, save_path=args.save)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

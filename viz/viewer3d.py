# -*- coding: utf-8 -*-
"""
3D-проигрыватель полётных логов (.flightlog) на движке Ursina (Panda3D).

Запуск:
    python viz/viewer3d.py                                  # последний лог из results/
    python viz/viewer3d.py results/<файл>.flightlog
    python viz/viewer3d.py <файл> --speed 2 --cam 3         # скорость, камера на старте

Управление:
    Пробел      — пауза / продолжение
    ← / →       — перемотка −5 / +5 с          R — в начало
    ↑ / ↓       — скорость проигрывания ×2 / ÷2
    1 2 3 4     — камера: за хвостом / из кабины / облёт мышью / обзор сверху
    F1          — уровень HUD: всё / режимы и предупреждения / пусто (viz/hud3d.py)
    F2          — профилировщик кадра: FPS и время по участкам (viz/perf.py)
    правая кнопка мыши + движение, колесо — облёт и зум (камера 3)
    M           — увеличить силуэт ×10 (для обзора)    T — показать весь путь
    клик по полосе времени внизу — переход к моменту
    Esc         — выход

Только отображение: читает .flightlog (flight_logger.load_log), физику не знает.

Системы координат
-----------------
Симулятор: земная x — север, y — восток, h — вверх; связанная СК — x вперёд,
y на правое крыло, z вниз; углы φ, θ, ψ (порядок ZYX).
Ursina (левая СК, y вверх): X — восток, Y — вверх, Z — север.
Отображение вектора (N, E, D) -> (E, −D, N) сохраняет «право/лево» и
направление разворота на экране.
"""

import sys
import os
import glob
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flight_logger import load_log
from viz.world3d import World, make_world, MAPS
from viz.aircraft3d import make_model
from viz.hud3d import Hud, HudData
from viz.perf import FrameProfiler
from viz.icon import set_app_id, set_win_icon

from ursina import (Ursina, Entity, Mesh, Text, Sky, Vec3, Color, camera, scene,
                    window, mouse, held_keys, application, time as utime,
                    DirectionalLight, AmbientLight, destroy)
from ursina.shaders.unlit_shader import unlit_shader
from panda3d.core import Filename

SUN_DIR     = np.array([0.45, 0.75, 0.35]) / np.linalg.norm([0.45, 0.75, 0.35])
FOG_CLR     = (0.72, 0.80, 0.88)


# ===========================================================================
# Подготовка данных (чистый numpy)
# ===========================================================================

def ned_to_u(n, e, d):
    """Земной вектор (N, E, D) -> координаты Ursina (X=E, Y=−D, Z=N)."""
    return np.stack([e, -d, n], axis=-1)


def body_axes(phi, theta, psi):
    """
    Оси связанной СК в земной СК (NED) по углам φ, θ, ψ (ZYX).
    Возвращает (x_b, z_b) — столбцы 1 и 3 матрицы R_b→n, формы (N, 3).
    """
    cf, sf = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    x_b = np.stack([ct * cp, ct * sp, -st], axis=-1)
    z_b = np.stack([cf * st * cp + sf * sp, cf * st * sp - sf * cp, cf * ct], axis=-1)
    return x_b, z_b


def air_velocity_axis(phi, theta, psi, alpha, beta):
    """
    Направление вектора воздушной скорости в земной СК (NED) — для маркера HUD.
    В связанной СК: (cos α cos β, sin β, sin α cos β); y_b = z_b × x_b.
    """
    x_b, z_b = body_axes(phi, theta, psi)
    y_b = np.cross(z_b, x_b)
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    k = lambda c: np.asarray(c)[..., None]
    return k(ca * cb) * x_b + k(sb) * y_b + k(sa * cb) * z_b


class Track:
    """Массивы лога, приведённые к сцене Ursina."""

    def __init__(self, data: dict):
        d = data
        self.meta = d["meta"]
        self.t = d["t"]
        n = len(self.t)
        z = np.zeros(n)
        get = lambda k: d[k] if k in d else z

        self.pos = ned_to_u(d["x"], get("y"), -d["h"])           # (N, 3)
        x_b, z_b = body_axes(get("phi"), d["theta"], get("psi"))
        self.nose = ned_to_u(x_b[:, 0], x_b[:, 1], x_b[:, 2])
        self.up   = ned_to_u(-z_b[:, 0], -z_b[:, 1], -z_b[:, 2])

        deg = np.degrees
        self.Va, self.h = d["Va"], d["h"]
        self.Vh = np.gradient(d["h"], self.t)
        v = air_velocity_axis(get("phi"), d["theta"], get("psi"), d["alpha_true"], get("beta_true"))
        self.vel = ned_to_u(v[:, 0], v[:, 1], v[:, 2])
        nan = np.full(n, np.nan)
        self.Va_ref = d["Va_ref"] if "Va_ref" in d else nan
        self.h_ref = d["h_ref"] if "h_ref" in d else nan
        self.chi_ref = (deg(d["chi_ref"]) % 360.0) if "chi_ref" in d else nan
        self.alpha = deg(d["alpha_true"])
        self.beta  = deg(get("beta_true"))
        self.phi = (deg(get("phi")) + 180.0) % 360.0 - 180.0     # крен копится при бочках
        self.theta = deg(d["theta"])
        self.psi   = (deg(get("psi")) + 360.0) % 360.0
        self.de, self.da, self.dr = d["delta_e"], get("delta_a"), get("delta_r")
        self.thr   = d["throttle"]
        self.wind  = np.stack([d["wind_x"], get("wind_y"), d["wind_h"]], axis=-1)
        self.alert = d["alert"].astype(int) if "alert" in d else z.astype(int)
        # путевой угол χ по земной скорости (разность координат)
        vx, vy = np.gradient(d["x"], self.t), np.gradient(get("y"), self.t)
        self.chi = (deg(np.arctan2(vy, vx)) + 360.0) % 360.0
        self.events = self.meta.get("events", [])
        self.b = float(self.meta.get("aircraft", {}).get("b", 2.0))
        self.c = float(self.meta.get("aircraft", {}).get("c", 0.2))
        self.name = self.meta.get("aircraft", {}).get("name", "fpv")

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def position(self, t: float) -> np.ndarray:
        """Положение с линейной интерполяцией (плавность при 60 кадрах/с)."""
        return np.array([np.interp(t, self.t, self.pos[:, k]) for k in range(3)])




# ===========================================================================
# Сцена: силуэт, след, камеры, приборная панель (общая для проигрывателя и игры)
# ===========================================================================

def _line(points, color, thickness=2.0):
    ent = Entity(model=Mesh(vertices=[(0, 0, 0), (0, 0, 0)], mode="line",
                            thickness=thickness),
                 color=color, shader=unlit_shader)
    _set_line(ent, points)
    return ent


def _set_line(ent, points):
    pts = [tuple(p) for p in np.asarray(points, float).reshape(-1, 3).tolist()]
    if len(pts) < 2:
        pts = (pts or [(0, 0, 0)]) * 2
    ent.model.vertices = pts
    ent.model.generate()


def _font():
    """Моноширинный шрифт с кириллицей (Consolas / DejaVu Sans Mono)."""
    from pathlib import Path
    for p in (r"C:\Windows\Fonts\consola.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            application.fonts_folder = Path(p).parent      # Ursina ищет шрифт по имени
            return Path(p).name
    return "VeraMono.ttf"


class View3D:
    """
    Всё, что рисуется вокруг ЛА: силуэт, след и его тень, отвес до земли,
    4 камеры, HUD (viz/hud3d.py). Источник состояния — снаружи (лог или игра):
    владелец каждый кадр вызывает render() и set_hud(HudData).
    """

    CAM_NAMES = {1: "chase", 2: "cockpit", 3: "orbit", 4: "overview"}

    def __init__(self, b: float, c: float, terrain: World, help_text: str,
                 cam: int = 1, aircraft_name: str = "fpv"):
        self.b, self.ter = b, terrain
        self.cam_mode, self.big = cam, False
        self.orbit_yaw, self.orbit_pitch, self.orbit_r = 30.0, 15.0, 12.0 * b
        self._cam_pos = None

        self.ac = make_model(aircraft_name, b, c)     # 3D-модель по типу ЛА (viz/aircraft3d.py)
        self.path_all = _line([], Color(1, 1, 1, 0.35), 1.5)
        self.path_all.enabled = False
        self.trail = _line([], Color(1.0, 0.25, 0.2, 1), 3.0)
        self.shadow = _line([], Color(0.1, 0.12, 0.1, 0.8), 2.0)
        self.drop = _line([], Color(1, 1, 1, 0.5), 1.0)
        self.route = _line([], Color(1.0, 0.83, 0.0, 0.9), 2.5)     # маршрут навигации
        self.route.enabled = False
        self.route_poles = []

        font = _font()
        self.hud = Hud(font, help_text)
        self.perf = FrameProfiler(font)
        if terrain.attribution:                 # условие использования снимков
            Text(parent=camera.ui, position=window.bottom_right + Vec3(-0.01, 0.012, 0),
                 origin=(0.5, -0.5), scale=0.55, font=font, color=Color(1, 1, 1, 0.6),
                 text=terrain.attribution)
        # строка «FPS, отставание от реального времени» — справа, над строкой подсказки по
        # клавишам (в узком окне подсказка доходит до правого края) и подписью снимка
        self.rate_txt = Text(parent=camera.ui, origin=(0.5, -0.5), scale=0.6, font=font,
                             color=Color(1, 1, 1, 0.7), text="")
        self._rate_y = -0.445 if terrain.attribution else -0.468
        self._rate_t = 0.0
        self.set_fog()

    def set_rate(self, dt, lag=None, ratio=None):
        """Строка внизу справа, раз в 0.25 с (пересборка текста дорогая).
        dt — длительность кадра, с; lag — отставание модели от реального времени, с;
        ratio — скорость модели относительно реального времени (1 — успевает)."""
        self._rate_t += dt
        if self._rate_t < 0.25:
            return
        self._rate_t = 0.0
        s = f"FPS {1.0 / max(self.perf.ema.get('frame', 1.0), 1e-6):.0f}"
        if lag is not None:
            s += f"   sim ×{ratio:.2f}   lag {lag:.1f} s"
        slow = ratio is not None and ratio < 0.98
        self.rate_txt.color = Color(1, 0.72, 0.15, 0.95) if slow else Color(1, 1, 1, 0.7)
        self.rate_txt.text = s
        self.rate_txt.position = (window.aspect_ratio / 2 - 0.01, self._rate_y)

    def set_route(self, pts, loop=False):
        """Маршрут: линия через точки (N, 3) в СК Ursina и отвесы до земли."""
        for e in self.route_poles:
            destroy(e)
        P = np.asarray(pts, float).reshape(-1, 3)
        _set_line(self.route, np.vstack([P, P[:1]]) if loop and len(P) > 2 else P)
        self.route.enabled = len(P) > 0
        self.route_poles = [_line([p, (p[0], float(self.ter.height(p[0], p[2])), p[2])],
                                  Color(1.0, 0.83, 0.0, 0.7), 1.5) for p in P]

    def set_fog(self):
        # шейдер Ursina: доля тумана = расстояние / (end − start); в обзоре — реже
        scene.fog_density = (0.0, 25000.0 if self.cam_mode == 4 else 9000.0)

    def handle_key(self, key) -> bool:
        """Клавиши камеры; True — клавиша обработана."""
        if key in ("1", "2", "3", "4"):
            self.cam_mode, self._cam_pos = int(key), None
            self.set_fog()
        elif key == "f1":
            self.hud.cycle()
        elif key == "f2":
            self.perf.toggle()
        elif key == "m":
            self.big = not self.big
        elif key == "scroll up":
            self.orbit_r = max(self.orbit_r * 0.85, 2.0 * self.b)
        elif key == "scroll down":
            self.orbit_r = min(self.orbit_r * 1.18, 3000.0)
        else:
            return False
        return True

    def render(self, pos, nose, up, de, da, dr, thr, trail, dt, overview_pts=None):
        """trail — точки следа (N, 3) в СК Ursina; overview_pts — что охватить в обзоре."""
        perf = self.perf
        with perf("aircraft"):
            scale = 25.0 if self.cam_mode == 4 else (10.0 if self.big else 1.0)
            self.ac.pose(pos, nose, up, scale)
            self.ac.surfaces(de, da, dr, thr, dt)
            self.ac.prop.enabled = self.cam_mode != 2           # винт закрывает обзор из кабины

        with perf("trail"):
            trail = np.vstack([np.asarray(trail, float).reshape(-1, 3), pos])
            _set_line(self.trail, trail)
        with perf("shadow"):
            g = trail.copy()
            g[:, 1] = self.ter.height(g[:, 0], g[:, 2]) + 0.8
            _set_line(self.shadow, g)
            gy = float(self.ter.height(pos[0], pos[2]))
            _set_line(self.drop, [pos, (pos[0], gy, pos[2])])
        perf.count("trail pts", len(trail))
        with perf("camera"):
            self.ter.follow(pos)
            self._camera(pos, nose, up, dt, trail if overview_pts is None else overview_pts)

    def _camera(self, pos, nose, up, dt, overview_pts):
        p = Vec3(*pos)
        flat = np.array([nose[0], 0.0, nose[2]])
        flat = flat / (np.linalg.norm(flat) + 1e-9)
        b = self.b
        if self.cam_mode == 1:                                    # за хвостом
            target = pos - 4.5 * b * flat + np.array([0, 1.2 * b, 0])
            k = 1.0 - np.exp(-4.0 * dt)
            self._cam_pos = target if self._cam_pos is None else \
                self._cam_pos + k * (target - self._cam_pos)
            camera.position = Vec3(*self._cam_pos)
            camera.look_at(p + Vec3(*(flat * 4 * b)))
            camera.rotation_z = 0
        elif self.cam_mode == 2:                                  # из кабины
            eye = self.ac.eye                                     # глаз пилота в осях модели
            camera.position = Vec3(*(pos + eye[2] * np.asarray(nose) + eye[1] * np.asarray(up)))
            camera.lookAt(camera.position + Vec3(*nose), Vec3(*up))
        elif self.cam_mode == 3:                                  # облёт мышью
            if mouse.right:
                self.orbit_yaw += mouse.velocity[0] * 200
                self.orbit_pitch = np.clip(self.orbit_pitch - mouse.velocity[1] * 200, -80, 85)
            yw, pt = np.radians(self.orbit_yaw), np.radians(self.orbit_pitch)
            off = self.orbit_r * np.array([np.sin(yw) * np.cos(pt), np.sin(pt), -np.cos(yw) * np.cos(pt)])
            camera.position = Vec3(*(pos + off))
            camera.look_at(p)
            camera.rotation_z = 0
        else:                                                     # обзор сверху
            P = np.vstack([np.asarray(overview_pts, float).reshape(-1, 3), pos])
            c = 0.5 * (P.min(0) + P.max(0))
            span = max(np.ptp(P[:, 0]), np.ptp(P[:, 2]), 300.0)
            camera.position = Vec3(c[0] - 0.1 * span, c[1] + 0.85 * span, c[2] - 0.6 * span)
            camera.look_at(Vec3(*c))
            camera.rotation_z = 0

    def set_hud(self, data: HudData):
        with self.perf("hud"):
            self.hud.update(data, self.cam_mode, self.CAM_NAMES[self.cam_mode])


def make_app(title: str, map_name: str = "default", size=(1600, 900)):
    """Окно Ursina, небо, свет, мир (viz/world3d.py). Возвращает (app, world)."""
    set_app_id()
    app = Ursina(title=title, size=size, borderless=False, development_mode=False)
    set_win_icon(app.win.getWindowHandle().getIntHandle())
    window.color = Color(*FOG_CLR, 1)
    camera.fov = 70
    camera.clip_plane_near = 0.3      # точность глубины на дальних планах
    camera.clip_plane_far = 40000
    terrain = make_world(map_name)
    terrain.build()
    sky = Sky()
    sky.scale = 30000
    scene.fog_color = Color(*FOG_CLR, 1)
    sun = DirectionalLight(shadows=False)
    sun.look_at(Vec3(*(-SUN_DIR)))
    AmbientLight(color=Color(0.55, 0.55, 0.6, 1))
    window.fps_counter.enabled = True
    return app, terrain


def screenshot_and_quit(path: str):
    application.base.win.saveScreenshot(Filename.fromOsSpecific(path))
    application.quit()


# ===========================================================================
# Проигрыватель
# ===========================================================================

class Player3D(Entity):

    HELP = ("Space pause  ←/→ ±5 s  ↑/↓ speed  R restart  "
            "1-4 camera  M scale  T path  Esc quit")

    def __init__(self, track: Track, terrain: World, speed=1.0, cam=1, shot=None):
        super().__init__()
        self.tr = track
        self.t_play, self.speed, self.paused = float(track.t[0]), speed, False
        self.show_path = False
        self.shot, self._frames = shot, 0

        self.view = View3D(track.b, track.c, terrain, self.HELP, cam=cam,
                           aircraft_name=track.name)
        step = max(1, int(round(0.1 / (track.t[1] - track.t[0]))))
        self._sub = np.arange(0, len(track.t), step)               # прорежение следа
        _set_line(self.view.path_all, track.pos[self._sub])

        self.bar_bg = Entity(parent=camera.ui, model="quad", color=Color(0, 0, 0, 0.45),
                             scale=(1.6, 0.018), position=(0, -0.452), collider="box")
        self.bar = Entity(parent=camera.ui, model="quad", color=Color(1, 0.45, 0.2, 0.9),
                          scale=(0.0, 0.018), position=(-0.8, -0.452), origin=(-0.5, 0))
        for ev in track.events:                              # метки событий на полосе
            fx = (ev["t"] - track.t[0]) / (track.t[-1] - track.t[0])
            Entity(parent=camera.ui, model="quad", color=Color(1, 0.85, 0.3, 1),
                   scale=(0.004, 0.03), position=(-0.8 + 1.6 * fx, -0.452))

    def input(self, key):
        if self.view.handle_key(key):
            return
        t0, t1 = self.tr.t[0], self.tr.t[-1]
        if key == "space":
            self.paused = not self.paused
        elif key in ("right arrow", "right arrow hold"):
            self.t_play = min(self.t_play + 5.0, t1)
        elif key in ("left arrow", "left arrow hold"):
            self.t_play = max(self.t_play - 5.0, t0)
        elif key == "up arrow":
            self.speed = min(self.speed * 2, 32.0)
        elif key == "down arrow":
            self.speed = max(self.speed / 2, 0.125)
        elif key == "r":
            self.t_play = t0
        elif key == "t":
            self.show_path = not self.show_path
        elif key == "left mouse down" and mouse.hovered_entity == self.bar_bg:
            fx = np.clip(mouse.point.x + 0.5, 0, 1)
            self.t_play = t0 + fx * (t1 - t0)
        elif key == "escape":
            application.quit()

    def update(self):
        tr, dt = self.tr, utime.dt
        if not self.paused:
            self.t_play = min(self.t_play + dt * self.speed, tr.t[-1])
        i = tr.index(self.t_play)
        pos = tr.position(self.t_play)
        j = int(np.searchsorted(self._sub, i))
        v = self.view
        v.path_all.enabled = self.show_path or v.cam_mode == 4
        v.render(pos, tr.nose[i], tr.up[i], tr.de[i], tr.da[i], tr.dr[i], tr.thr[i],
                 tr.pos[self._sub[:j]], dt, overview_pts=tr.pos[self._sub])
        self._hud(i)
        v.perf.frame(dt)
        v.set_rate(dt)

        self._frames += 1
        if self.shot and self._frames == 40:
            screenshot_and_quit(self.shot)

    def _hud(self, i):
        tr = self.tr
        wN, wE, _ = tr.wind[i]
        past = [e for e in tr.events if e["t"] <= tr.t[i]]
        ref = lambda a: float(a[i]) if np.isfinite(a[i]) else None
        self.view.set_hud(HudData(
            t=tr.t[i], Va=tr.Va[i], h=tr.h[i], Vh=tr.Vh[i], alpha=tr.alpha[i], beta=tr.beta[i],
            phi=tr.phi[i], psi=tr.psi[i], chi=tr.chi[i], thr=tr.thr[i], wind_n=wN, wind_e=wE,
            alert=tr.alert[i], modes=f"LOG  {tr.meta.get('scenario', '')}",
            status="PAUSED" if self.paused else f"×{self.speed:g}",
            event=f"{past[-1]['label']}  (t={past[-1]['t']:g} s)" if past else "",
            Va_ref=ref(tr.Va_ref), h_ref=ref(tr.h_ref), chi_ref=ref(tr.chi_ref),
            nose=tr.nose[i], vel=tr.vel[i]))
        fx = (tr.t[i] - tr.t[0]) / (tr.t[-1] - tr.t[0])
        self.bar.scale_x = 1.6 * fx


# ===========================================================================
# Запуск
# ===========================================================================

def play(path: str, speed: float = 1.0, cam: int = 1, shot: str = None,
         t_start: float = None, map_name: str = "default"):
    track = Track(load_log(path))
    app, terrain = make_app(f"3D: {os.path.basename(path)}", map_name)
    player = Player3D(track, terrain, speed=speed, cam=cam, shot=shot)
    if t_start is not None:
        player.t_play = float(t_start)
    try:
        app.run()
    finally:
        print(player.view.perf.summary())


def _latest_log() -> str:
    files = glob.glob(os.path.join(_ROOT, "results", "*.flightlog*"))
    if not files:
        sys.exit("В results/ нет .flightlog — сначала запустите сценарий.")
    return max(files, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="3D-проигрыватель .flightlog (Ursina)")
    ap.add_argument("path", nargs="?", help="файл .flightlog (по умолчанию — последний)")
    ap.add_argument("--speed", type=float, default=1.0, help="скорость проигрывания")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4], help="камера")
    ap.add_argument("--at", type=float, default=None, help="начать с момента t, с")
    ap.add_argument("--shot", default=None, help="сохранить скриншот и выйти (для отчётов)")
    ap.add_argument("--map", default="default", choices=list(MAPS), help="карта мира")
    a = ap.parse_args()
    play(a.path or _latest_log(), a.speed, a.cam, a.shot, a.at, a.map)


if __name__ == "__main__":
    main()

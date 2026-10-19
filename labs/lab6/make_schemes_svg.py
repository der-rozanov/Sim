# -*- coding: utf-8 -*-
"""
Схемы сборки (вид окна редактора) для заданий 1А и 1Б: SVG вставляется в HTML
между метками <!--EDA-->…<!--/EDA--> и <!--EDB-->…<!--/EDB-->.

    python labs/lab6/make_schemes_svg.py labs/lab6/Лабораторная_6.html
"""
import io, re, sys

FILL = {"src": "#e3f1e3", "reg": "#e6eefb", "math": "#f4f4f4", "nl": "#fff4e0", "out": "#fde8e8"}
EDGE = {"src": "#6aa66a", "reg": "#5b84c9", "math": "#9a9a9a", "nl": "#d39a3a", "out": "#d16a6a"}


def block(x, y, w, h, grp, title, param, n_in, has_out=True, in_labels=None):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{FILL[grp]}" stroke="{EDGE[grp]}" stroke-width="1.6"/>']
    if param:
        s.append(f'<text x="{x + w / 2}" y="{y + h / 2 - 4}" class="bt">{title}</text>')
        s.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" class="bp">{param}</text>')
    else:
        s.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" class="bt">{title}</text>')
    ports = []
    for k in range(n_in):
        py = y + h * (k + 1) / (n_in + 1)
        s.append(f'<path d="M{x - 1},{py - 5} L{x + 7},{py} L{x - 1},{py + 5} z" fill="#333"/>')
        if in_labels:
            s.append(f'<text x="{x + 11}" y="{py + 4}" class="sg">{in_labels[k]}</text>')
        ports.append((x, py))
    out = (x + w, y + h / 2)
    if has_out:
        s.append(f'<circle cx="{out[0]}" cy="{out[1]}" r="4" fill="#333"/>')
    return "\n".join(s), ports, out


def wire(pts, label=None, lpos=None, dot=None):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    s = [f'<path d="{d}" class="wr"/>']
    if dot:
        s.append(f'<circle cx="{dot[0]}" cy="{dot[1]}" r="3.5" fill="#2b4a7a"/>')
    if label:
        s.append(f'<text x="{lpos[0]}" y="{lpos[1]}" class="wl">{label}</text>')
    return "\n".join(s)


def scheme(with_q):
    parts, w = [], []
    G = 30                                        # промежуток между блоками
    y_in = {"theta_ref": 34, "theta": 104, "q": 200, "de": 272 if with_q else 200}
    ins = {}
    for key, sig in [("theta_ref", "θ_ref"), ("theta", "θ"), ("q", "q"), ("de", "δe*")]:
        if key == "q" and not with_q:
            continue
        b, _, o = block(20, y_in[key], 96, 46, "src", "Вход", sig, 0)
        parts.append(b)
        ins[key] = o
    x = 196
    b, p1, o1 = block(x, 34, 44, 64, "math", "Σ", None, 2, in_labels=["+", "−"]); parts.append(b)
    x += 44 + G + 6
    b, pk, ok = block(x, 43, 128, 46, "reg", "П-регулятор", "Kp = Kθ" if with_q else "Kp", 1); parts.append(b)
    x += 128 + G + 6
    if with_q:
        b, p2, o2 = block(x, 34, 44, 64, "math", "Σ", None, 2, in_labels=["+", "−"]); parts.append(b)
        x += 44 + G
        b, pq, oq = block(x, 43, 128, 46, "reg", "П-регулятор", "Kp = Kq", 1); parts.append(b)
        x += 128 + G
    b, pg, og = block(x, 43, 140, 46, "math", "Усилит. звено", "K = −1", 1); parts.append(b)
    x += 140 + G + 6
    b, p3, o3 = block(x, 34, 44, 64, "math", "Σ", None, 2, in_labels=["+", "+"]); parts.append(b)
    x += 44 + G
    b, ps, os_ = block(x, 43, 124, 46, "nl", "Насыщение", "−25 … 25", 1); parts.append(b)
    x += 124 + G + 6
    b, po, _ = block(x, 43, 92, 46, "out", "Выход", "δe", 1, has_out=False); parts.append(b)
    b, psc, _ = block(pk[0][0], 128, 128, 56, "out", "Осциллограф", "θ_ref, θ", 2, has_out=False); parts.append(b)

    xr, yr = ins["theta_ref"]
    w.append(wire([(xr, yr), (176, yr), (176, p1[0][1]), p1[0]], "θ_ref", (124, yr - 7)))
    w.append(wire([(176, yr), (176, psc[0][1]), psc[0]], dot=(176, yr)))
    xt, yt = ins["theta"]
    w.append(wire([(xt, yt), (164, yt), (164, p1[1][1]), p1[1]], "θ", (124, yt - 7)))
    w.append(wire([(164, yt), (164, psc[1][1]), psc[1]], dot=(164, yt)))
    w.append(wire([o1, pk[0]], "e_θ", (o1[0] + 6, o1[1] - 8)))
    if with_q:
        xm = p2[0][0] - 16
        w.append(wire([ok, (xm, ok[1]), (xm, p2[0][1]), p2[0]], "q_ref", (ok[0] - 6, ok[1] - 30)))
        xq, yq = ins["q"]
        xm2 = p2[1][0] - 8
        w.append(wire([(xq, yq), (xm2, yq), (xm2, p2[1][1]), p2[1]], "q", (124, yq - 7)))
        w.append(wire([o2, pq[0]], "e_q", (o2[0] + 5, o2[1] - 8)))
        w.append(wire([oq, pg[0]]))
    else:
        w.append(wire([ok, pg[0]]))
    xm = p3[0][0] - 16
    w.append(wire([og, (xm, og[1]), (xm, p3[0][1]), p3[0]]))
    xd, yd = ins["de"]
    xj = p3[1][0] - 8
    w.append(wire([(xd, yd), (xj, yd), (xj, p3[1][1]), p3[1]], "δe*", (124, yd - 7)))
    w.append(wire([o3, ps[0]]))
    w.append(wire([os_, po[0]], "δe", (os_[0] + 8, os_[1] - 8)))

    width = po[0][0] + 92 + 16
    height = y_in["de"] + 46 + 16
    nl = chr(10)
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" class="ed">' + nl
            + nl.join(w) + nl + nl.join(parts) + nl + "</svg>")


def scheme_b():
    """Каскад в две строки: внешний контур θ → q_ref сверху, внутренний q → δe снизу."""
    parts, w = [], []
    ins = {}
    for key, sig, y in [("theta_ref", "θ_ref", 34), ("theta", "θ", 104), ("q", "q", 210), ("de", "δe*", 290)]:
        b, _, o = block(20, y, 96, 46, "src", "Вход", sig, 0)
        parts.append(b)
        ins[key] = o
    b, p1, o1 = block(196, 34, 44, 64, "math", "Σ", None, 2, in_labels=["+", "−"]); parts.append(b)
    b, pk, ok = block(276, 43, 128, 46, "reg", "П-регулятор", "Kp = Kθ", 1); parts.append(b)
    yb = 200
    b, p2, o2 = block(196, yb, 44, 64, "math", "Σ", None, 2, in_labels=["+", "−"]); parts.append(b)
    b, pq, oq = block(276, yb + 9, 128, 46, "reg", "П-регулятор", "Kp = Kq", 1); parts.append(b)
    b, pg, og = block(440, yb + 9, 140, 46, "math", "Усилит. звено", "K = −1", 1); parts.append(b)
    b, p3, o3 = block(624, yb, 44, 64, "math", "Σ", None, 2, in_labels=["+", "+"]); parts.append(b)
    b, ps, os_ = block(704, yb + 9, 124, 46, "nl", "Насыщение", "−25 … 25", 1); parts.append(b)
    b, po, _ = block(866, yb + 9, 92, 46, "out", "Выход", "δe", 1, has_out=False); parts.append(b)

    xr, yr = ins["theta_ref"]
    w.append(wire([(xr, yr), (176, yr), (176, p1[0][1]), p1[0]], "θ_ref", (124, yr - 7)))
    xt, yt = ins["theta"]
    w.append(wire([(xt, yt), (164, yt), (164, p1[1][1]), p1[1]], "θ", (124, yt - 7)))
    w.append(wire([o1, pk[0]], "e_θ", (o1[0] + 6, o1[1] - 8)))
    w.append(wire([ok, (444, ok[1]), (444, 160), (176, 160), (176, p2[0][1]), p2[0]],
                  "q_ref", (412, ok[1] - 8)))
    w.append('<text x="300" y="152" class="wl">q_ref — уставка внутреннего контура</text>')
    xq, yq = ins["q"]
    w.append(wire([(xq, yq), (180, yq), (180, p2[1][1]), p2[1]], "q", (124, yq - 7)))
    w.append(wire([o2, pq[0]], "e_q", (o2[0] + 5, o2[1] - 8)))
    w.append(wire([oq, pg[0]]))
    w.append(wire([og, (598, og[1]), (598, p3[0][1]), p3[0]]))
    xd, yd = ins["de"]
    w.append(wire([(xd, yd), (610, yd), (610, p3[1][1]), p3[1]], "δe*", (124, yd - 7)))
    w.append(wire([o3, ps[0]]))
    w.append(wire([os_, po[0]], "δe", (os_[0] + 8, os_[1] - 8)))
    w.append('<text x="196" y="22" class="sg">внешний контур: θ → q_ref</text>')
    nl = chr(10)
    return (f'<svg viewBox="0 0 974 352" xmlns="http://www.w3.org/2000/svg" class="ed">' + nl
            + nl.join(w) + nl + nl.join(parts) + nl
            + '<text x="704" y="290" class="sg">внутренний контур: q → δe</text>' + nl + "</svg>")


if __name__ == "__main__":
    p = sys.argv[1]
    s = io.open(p, encoding="utf-8").read()
    for tag, wq in (("A", False), ("B", True)):  # B — scheme_b
        s = re.sub(rf"<!--ED{tag}-->.*?<!--/ED{tag}-->", lambda m: f"<!--ED{tag}-->\n{scheme(False) if not wq else scheme_b()}\n<!--/ED{tag}-->", s, flags=re.S)
    io.open(p, "w", encoding="utf-8").write(s)

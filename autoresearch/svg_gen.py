"""Dependency-free SVG figure generation for the IEEE paper (no matplotlib).

Provides: line_chart, bar_chart, multi_panel_bars, arch_diagram.
All figures are self-contained .svg files (white background, Helvetica).
"""
import html
from pathlib import Path

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2",
           "#db2777", "#65a30d"]


def _esc(s):
    return html.escape(str(s), quote=True)


def _write(path, svg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)
    return str(path)


def _header(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">')


def _ygrid(parts, ml, mt, pw, ph, lo, hi, nty=5, fmt="{:.3f}"):
    for i in range(nty + 1):
        v = lo + (hi - lo) * i / nty
        y = mt + ph * (1 - i / nty)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#374151">{fmt.format(v)}</text>')


def line_chart(path, title, series, xlabel="x", ylabel="y", xlabels=None,
               width=680, height=400, y_min=None, y_max=None, legend=True,
               yfmt="{:.3f}"):
    """series: list of (name, values, color). NaNs are skipped."""
    ml, mr, mt, mb = 70, 20, 48, 58
    pw, ph = width - ml - mr, height - mt - mb
    allv = [v for _, vals, _ in series for v in vals if v == v]
    lo = y_min if y_min is not None else (min(allv) if allv else 0.0)
    hi = y_max if y_max is not None else (max(allv) if allv else 1.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    nmax = max((len(vals) for _, vals, _ in series), default=1)

    def X(i, n):
        return ml + pw * (i / max(1, n - 1)) if n > 1 else ml + pw / 2

    def Y(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    p = [_header(width, height),
         f'<rect width="{width}" height="{height}" fill="white"/>',
         f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="15" '
         f'font-weight="bold">{_esc(title)}</text>']
    _ygrid(p, ml, mt, pw, ph, lo, hi, 5, yfmt)
    n = nmax
    for i in range(n):
        x = X(i, n)
        p.append(f'<line x1="{x:.1f}" y1="{mt + ph}" x2="{x:.1f}" y2="{mt + ph + 4}" '
                 f'stroke="#9ca3af"/>')
        lab = xlabels[i] if xlabels and i < len(xlabels) else str(i + 1)
        p.append(f'<text x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle" '
                 f'font-size="11" fill="#374151">{_esc(lab)}</text>')
    p.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#6b7280" stroke-width="1"/>')
    for name, vals, color in series:
        nlen = len(vals)
        pts = " ".join(f"{X(i, nlen):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v == v)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i, v in enumerate(vals):
            if v == v:
                p.append(f'<circle cx="{X(i, nlen):.1f}" cy="{Y(v):.1f}" r="3" fill="{color}"/>')
    p.append(f'<text x="{ml + pw / 2}" y="{height - 10}" text-anchor="middle" '
             f'font-size="13">{_esc(xlabel)}</text>')
    p.append(f'<text x="16" y="{mt + ph / 2}" text-anchor="middle" font-size="13" '
             f'transform="rotate(-90 16 {mt + ph / 2})">{_esc(ylabel)}</text>')
    if legend and series:
        lx = ml + 14
        for j, (name, _, color) in enumerate(series):
            y = mt + 16 + j * 18
            p.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 20}" y2="{y}" '
                     f'stroke="{color}" stroke-width="3"/>')
            p.append(f'<text x="{lx + 26}" y="{y + 4}" font-size="12">{_esc(name)}</text>')
    p.append("</svg>")
    return _write(path, "".join(p))


def bar_chart(path, title, labels, values, ylabel="value", colors=None,
              width=680, height=400, value_fmt="{:.4f}"):
    ml, mr, mt, mb = 70, 20, 48, 92
    pw, ph = width - ml - mr, height - mt - mb
    hi = max([abs(v) for v in values] + [1e-6]) * 1.18
    lo = min(0.0, min(values) if values else [0.0])
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = max(1, len(labels))

    def Y(v):
        return mt + ph * (1 - (v - lo) / (hi - lo))

    p = [_header(width, height),
         f'<rect width="{width}" height="{height}" fill="white"/>',
         f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="15" '
         f'font-weight="bold">{_esc(title)}</text>']
    _ygrid(p, ml, mt, pw, ph, lo, hi, 5, "{:.3f}")
    bw = pw / n * 0.6
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = ml + pw * (i + 0.5) / n
        y0, y1 = Y(0.0), Y(v)
        top, hgt = min(y0, y1), abs(y1 - y0)
        color = (colors[i % len(colors)] if colors else PALETTE[i % len(PALETTE)])
        p.append(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                 f'height="{max(1.0, hgt):.1f}" fill="{color}" rx="2"/>')
        p.append(f'<text x="{cx:.1f}" y="{top - 6:.1f}" text-anchor="middle" '
                 f'font-size="11" fill="#111827">{value_fmt.format(v)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{mt + ph + 16}" text-anchor="end" '
                 f'font-size="12" fill="#374151" '
                 f'transform="rotate(-32 {cx:.1f} {mt + ph + 16})">{_esc(lab)}</text>')
    p.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" '
             f'stroke="#6b7280" stroke-width="1"/>')
    p.append(f'<text x="16" y="{mt + ph / 2}" text-anchor="middle" font-size="13" '
             f'transform="rotate(-90 16 {mt + ph / 2})">{_esc(ylabel)}</text>')
    p.append("</svg>")
    return _write(path, "".join(p))


def multi_panel_bars(path, title, panels, width=920, height=420):
    """panels: list of (panel_title, labels, values)."""
    n = max(1, len(panels))
    pw_each = width // n
    ml, mt, mb = 56, 64, 92
    inner_w, inner_h = pw_each - ml - 14, height - mt - mb
    p = [_header(width, height),
         f'<rect width="{width}" height="{height}" fill="white"/>',
         f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="15" '
         f'font-weight="bold">{_esc(title)}</text>']
    for i, (pt, labels, values) in enumerate(panels):
        ox = i * pw_each
        x0 = ox + ml
        hi = max([abs(v) for v in values] + [1e-6]) * 1.18
        lo = min(0.0, min(values) if values else [0.0])
        if hi - lo < 1e-9:
            hi = lo + 1.0

        def Y(v, x0=x0, hi=hi, lo=lo):
            return mt + inner_h * (1 - (v - lo) / (hi - lo))

        p.append(f'<text x="{x0 + inner_w / 2}" y="52" text-anchor="middle" '
                 f'font-size="13" font-weight="bold">{_esc(pt)}</text>')
        for k in range(4):
            v = lo + (hi - lo) * k / 3
            y = Y(v)
            p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + inner_w}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
            p.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="#374151">{v:.3f}</text>')
        m = max(1, len(labels))
        bw = inner_w / m * 0.58
        for j, (lab, v) in enumerate(zip(labels, values)):
            cx = x0 + inner_w * (j + 0.5) / m
            y0, y1 = Y(0.0), Y(v)
            top, hgt = min(y0, y1), abs(y1 - y0)
            p.append(f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                     f'height="{max(1.0, hgt):.1f}" fill="{PALETTE[j % len(PALETTE)]}" rx="2"/>')
            p.append(f'<text x="{cx:.1f}" y="{top - 5:.1f}" text-anchor="middle" '
                     f'font-size="10">{v:.4f}</text>')
            p.append(f'<text x="{cx:.1f}" y="{mt + inner_h + 14}" text-anchor="end" '
                     f'font-size="10" transform="rotate(-32 {cx:.1f} {mt + inner_h + 14})">'
                     f'{_esc(lab)}</text>')
        p.append(f'<rect x="{x0}" y="{mt}" width="{inner_w}" height="{inner_h}" '
                 f'fill="none" stroke="#6b7280"/>')
    p.append("</svg>")
    return _write(path, "".join(p))


def arch_diagram(path, n_students=3):
    """TS-Aug pipeline: seed -> teacher/students -> candidates -> score -> select.

    Layout (960x560): left column = data, middle = models, right = selection.
    """
    W, H = 960, 560
    p = [_header(W, H), f'<rect width="{W}" height="{H}" fill="white"/>']
    p.append('<defs><marker id="arw" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
             'orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#374151"/></marker></defs>')
    p.append(f'<text x="{W / 2}" y="26" text-anchor="middle" font-size="16" '
             f'font-weight="bold">TS-Aug: Teacher-Student Ensemble Augmentation '
             f'(one iteration t)</text>')

    def box(x, y, w, h, title, sub, fill, stroke="#374151"):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        p.append(f'<text x="{x + w / 2}" y="{y + h / 2 - (8 if sub else 0)}" '
                 f'text-anchor="middle" font-size="13" font-weight="bold" '
                 f'fill="#111827">{_esc(title)}</text>')
        if sub:
            p.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" text-anchor="middle" '
                     f'font-size="11" fill="#374151">{_esc(sub)}</text>')

    def arrow(x1, y1, x2, y2, label=""):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#374151" '
                 f'stroke-width="1.6" marker-end="url(#arw)"/>')
        if label:
            p.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}" '
                     f'text-anchor="middle" font-size="10.5" fill="#4b5563">'
                     f'{_esc(label)}</text>')

    box(20, 60, 200, 64, "Few-shot seed D0", "20 classes x k shots (k=5/10)", "#dbeafe")
    box(300, 24, 260, 58, "Teacher CNN T", "train on Dt, scores P_T(c | x)", "#fee2e2")
    box(300, 128, 260, 58, f"Students S1..S{max(2, n_students)} (N=2..4)",
        "enc-dec + latent z + style prior", "#dcfce7")
    box(300, 232, 260, 54, "Candidate pool", "x' = G_i(Enc_i(x), z),  z ~ N(0, I)",
        "#fef9c3")
    box(300, 336, 260, 62, "Ensemble score",
        "S = [wT PT + wS mean peers] / (wT + wS)", "#ede9fe")
    box(640, 336, 200, 62, "Top-K selection", "accept if  S >= theta,  per class",
        "#fce7f3")
    box(640, 232, 200, 54, "Dt = D(t-1) + At", "dataset grows toward target",
        "#e0f2fe")
    box(640, 452, 200, 62, "ProtoNet evaluation", "val_acc on fixed V (1000 imgs)",
        "#f3f4f6")
    p.append(f'<text x="{W / 2}" y="{H - 16}" text-anchor="middle" font-size="11" '
             f'fill="#6b7280">repeat iterations t = 1..T until per-class target shots '
             f'reached; teacher re-fine-tuned on Dt each iteration</text>')

    arrow(220, 82, 300, 58, "train")
    arrow(220, 108, 300, 150, "train")
    arrow(430, 82, 430, 128, "")
    arrow(430, 186, 430, 232, "generate")
    arrow(430, 286, 430, 336, "")
    arrow(560, 53, 560, 330, "P_T")
    arrow(560, 157, 560, 334, "peer P_j")
    arrow(560, 367, 640, 367, "rank")
    arrow(740, 336, 740, 286, "accept A_t")
    p.append('<path d="M 640 259 L 260 259 L 260 58 L 300 53" fill="none" '
             'stroke="#374151" stroke-width="1.6" stroke-dasharray="6,4" '
             'marker-end="url(#arw)"/>')
    p.append('<text x="450" y="252" text-anchor="middle" font-size="10.5" '
             'fill="#4b5563">fine-tune teacher on Dt (next iteration)</text>')
    arrow(740, 286, 740, 452, "target reached")
    p.append("</svg>")
    return _write(path, "".join(p))

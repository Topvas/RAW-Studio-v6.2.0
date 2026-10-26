#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAW Studio — аналіз .raw записів АЦП та генератор нових файлів у тому ж форматі.

Автор: Vaskov Yaroslav, CIT PROGRAM LLC
Версія: 2.0.0

Формат файлу:
    без заголовка; послідовність відліків uint16 little-endian,
    інтерлівінг ПО ВІДЛІКАХ: [ch0, ..., ch7, ch0, ...],
    типово 8 каналів; дефолтна fs = 12 ГГц (Найквіст 6 ГГц).

Частотне планування:
    6 блоків: 100-500, 500-1700, 1200-2500, 2400-3600, 3400-4600, 4400-6000 МГц
    каналізація: 64 МГц на канал → 105 повних + 5 крайових = 110 каналів.

Модуль АЧХ АЦП:
    амплітуда генерації залежно від частоти за профілем (дБ):
    плоский / спад −3 дБ/окт / смуговий 100М–6Г / провал (notch) / власна таблиця.
    Інтерполяція між точками: лінійна / кубічний сплайн / сплайн у log f;
    поза виміряним діапазоном: за краями / лінійна екстраполяція / нуль.
    Реалізація — FFT-фільтрація змінної складової; DC не модулюється.

Калібрування реальної АЧХ тракту:
    ступінчастий прогін тонів (кожен тон — у своєму сегменті, точно в біні FFT)
    → запис через реальний тракт → вимірювання амплітуди кожного тону →
    АЧХ тракту в дБ → збереження профілю в .achx → завантаження в Генератор.

Запуск:
    python3 raw_studio.py                         # графічний інтерфейс
    python3 raw_studio.py --headless FILE [fs] [nch]
    python3 raw_studio.py --selftest [FILE]       # самоперевірка ядра
    python3 raw_studio.py --preview OUT_PREFIX FILE [fs] [nch]
    python3 raw_studio.py --calgen OUT fs nch [tones fmin fmax amp seg]
    python3 raw_studio.py --calmeas RECORDED PLAN

Залежності: python3-tk, numpy, matplotlib.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

APP_AUTHOR = "Vaskov Yaroslav, CIT PROGRAM LLC"
APP_VERSION = "6.2.0"
# -- Спектр із зображення: оцифрування PNG/JPEG графіка (v3.8.0) --
def _rgb_hue(px):
    """Відтінок (0…1) для масиву RGB (N,3)."""
    px = np.asarray(px, dtype=float)
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    d = mx - mn
    h = np.zeros(len(px))
    m = d > 1e-9
    if m.any():
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        i = m & (mx == r)
        h[i] = (((g[i] - b[i]) / d[i]) % 6.0) / 6.0
        i = m & (mx == g) & (mx != r)
        h[i] = (((b[i] - r[i]) / d[i]) + 2.0) / 6.0
        i = m & (mx == b) & (mx != r) & (mx != g)
        h[i] = (((r[i] - g[i]) / d[i]) + 4.0) / 6.0
    return h % 1.0


def image_to_spectrum(path, f_min_mhz, f_max_mhz, db_min, db_max,
                      sat_thr=0.25, hue_tol=0.09, crop=None, median=0):
    """Оцифрує спектральну криву зі зображення (PNG/JPEG/BMP) — скріншот
    аналізатора, скріншот звіту, фото графіка. Авто-пошук рамки графіка
    (прямі лінії), авто-визначення кольору траси за домінантним відтінком
    (для ч/б — яскравісний режим); калібрування осей задає користувач.
    Повертає (freqs_mhz, dbs, meta): однакові ndarray по колонках графіка;
    meta: mode (color/lum), box (x0,x1,y0,y1), box_mode, columns, gaps.
    Оцінка рівнів приблизна — залежить від калібрування і якості скріншота."""
    from matplotlib.image import imread
    img = np.asarray(imread(path), dtype=float)[..., :3]
    if img.max() > 1.5:
        img = img / 255.0
    ox = oy = 0
    H, W = img.shape[:2]
    if crop:
        try:
            cl, cr, ct, cb = (float(v) for v in crop)
        except (TypeError, ValueError):
            cl, cr, ct, cb = 0.0, 1.0, 0.0, 1.0
        x0c = int(max(0.0, min(cl, 0.9)) * W)
        x1c = int(min(1.0, max(cr, 0.1)) * W) - 1
        y0c = int(max(0.0, min(ct, 0.9)) * H)
        y1c = int(min(1.0, max(cb, 0.1)) * H) - 1
        img = img[y0c:y1c + 1, x0c:x1c + 1, :]
        H, W = img.shape[:2]
        ox, oy = x0c, y0c
    v = img.max(axis=2)
    dark = v < 0.35
    bright = v > 0.75

    def _lines(cnt, lim):
        idx = np.where(cnt > 0.55 * lim)[0]
        out, grp = [], []
        for k in idx:
            k = int(k)
            if grp and k - grp[-1] <= 3:
                grp.append(k)
            else:
                if grp:
                    out.append(int(round(sum(grp) / len(grp))))
                grp = [k]
        if grp:
            out.append(int(round(sum(grp) / len(grp))))
        return out

    cands = []
    for mask in (dark, bright):
        hl = _lines(mask.sum(axis=1), W)
        vl = _lines(mask.sum(axis=0), H)
        cands.append((len(hl) + len(vl), hl, vl))
    hl, vl = [], []
    for _nc, a, b in sorted(cands, key=lambda t: t[0]):
        if len(a) >= 2 and len(b) >= 2:
            hl, vl = a, b
            break
    if hl and vl:
        y0, y1 = int(hl[0]) + 2, int(hl[-1]) - 2
        x0, x1 = int(vl[0]) + 2, int(vl[-1]) - 2
        box_mode = "frame"
    else:
        x0, x1 = int(W * 0.02), W - int(W * 0.02)
        y0, y1 = int(H * 0.05), H - int(H * 0.05)
        box_mode = "full"
    if x1 - x0 < 40 or y1 - y0 < 30:
        raise ValueError("графік на зображенні занадто малий")
    sub = img[y0:y1 + 1, x0:x1 + 1, :]
    sv = sub.max(axis=2) - sub.min(axis=2)
    colored = (sv > sat_thr) & (sub.max(axis=2) > 0.4)
    ys = np.full(sub.shape[1], np.nan)
    lum_s = sub.mean(axis=2)
    bgm = float(np.median(lum_s))
    col_colored = colored.sum(axis=0) >= 2
    col_lum = ((np.abs(lum_s - bgm) > 0.10) & (sv < 0.2)).sum(axis=0) >= 2
    if col_colored.sum() > col_lum.sum() and col_colored.mean() > 0.05:
        mode = "color"
        hmed = float(np.median(_rgb_hue(sub[colored])))
        dist = np.abs((_rgb_hue(sub.reshape(-1, 3)) - hmed + 0.5) % 1.0 - 0.5)
        trace = (colored.reshape(-1) & (dist < hue_tol)).reshape(colored.shape)
    else:
        mode = "lum"
        lum = sub.mean(axis=2)
        hist, edges = np.histogram(lum, bins=16, range=(0.0, 1.0))
        bg = edges[int(np.argmax(hist))]
        trace = (np.abs(lum - bg) > 0.10) & (sv < 0.2)
        _rc = trace.sum(axis=1) / max(trace.shape[1], 1)
        trace[_rc > 0.6, :] = False
    for j in range(trace.shape[1]):
        col = np.where(trace[:, j])[0]
        if len(col):
            ys[j] = col.min()
    good = np.isfinite(ys)
    if good.sum() < 2:
        raise ValueError("трасу спектра на зображенні не знайдено")
    idx = np.arange(len(ys))
    ys = np.interp(idx, idx[good], ys[good])
    if median and median >= 3 and len(ys) > 5:
        ys = np.median(np.vstack([np.roll(ys, k) for k in (-1, 0, 1)]), axis=0)
    fx = f_min_mhz + idx / max(len(ys) - 1, 1) * (f_max_mhz - f_min_mhz)
    db = db_min + (1.0 - ys / max(trace.shape[0] - 1, 1)) * (db_max - db_min)
    meta = {"mode": mode, "box": (x0 + ox, x1 + ox, y0 + oy, y1 + oy), "box_mode": box_mode,
            "columns": int(good.sum()), "gaps": int((~good).sum())}
    return fx, db, meta


def jam_from_peak(freq_mhz, db, width_mhz, db_off=0.0, js_target=15.0):
    """Пік оцифрованого спектра → параметри завади для генератора (v3.9.0).

    Ширина піка (−3 дБ): ≤ 0.05 МГц → точкова CW; 0.05–1.5 МГц → ЛЧМ
    (девіація = ширина, період 1000 мкс); > 1.5 МГц → шум (смуга
    1.5 × ширина, межі 1–20 МГц). Рівень сигналу = db + db_off
    (dBFS → дБм, офсет калібрування, за замовчуванням 0);
    потрібна потужність завади на вході приймача = сигнал + J/S.
    """
    w = float(width_mhz)
    if w <= 0.05:
        kind, kind_name = "jam_cw", "Завада CW"
        bw = dev = per = 0.0
        note = "вузький пік (%.4g МГц) → точкова CW-завада" % max(w, 1e-4)
    elif w <= 1.5:
        kind, kind_name = "jam_chirp", "Завада (чирп)"
        bw, dev, per = 0.0, max(w, 0.05), 1000.0
        note = "смуга піка %.3g МГц → ЛЧМ ±%.3g МГц, період %.0f мкс" % (w, dev, per)
    else:
        kind, kind_name = "jam_noise", "Завада (шум)"
        bw, dev, per = min(max(w * 1.5, 1.0), 20.0), 0.0, 0.0
        note = "смуга піка %.3g МГц → загороджувальний шум %.3g МГц" % (w, bw)
    sig = float(db) + float(db_off)
    return {"kind": kind, "kind_name": kind_name, "f_hz": float(freq_mhz) * 1e6,
            "bw_mhz": bw, "dev_mhz": dev, "per_us": per, "sig_dbm": sig,
            "js_target": float(js_target), "jam_req_dbm": sig + float(js_target),
            "note": note}


THEME_DARK = {
    "bg": "#0f1420", "panel": "#161d2e", "panel2": "#1c2438",
    "edge_hi": "#2e3a55", "edge_lo": "#0a0e18",
    "accent": "#38bdf8", "warn": "#f87171", "text": "#dbe4f3", "dim": "#7d8aa5",
}
THEME_LIGHT = {
    "bg": "#eef1f6", "panel": "#ffffff", "panel2": "#e3e8f0",
    "edge_hi": "#ffffff", "edge_lo": "#93a1b8",
    "accent": "#1d6fb8", "warn": "#b22222", "text": "#1a2233", "dim": "#5a6880",
}
THEME_DAY = {
    "bg": "#e2e6ec", "panel": "#f6f8fb", "panel2": "#e9edf3",
    "edge_hi": "#ffffff", "edge_lo": "#8d99ad",
    "accent": "#0f766e", "warn": "#b91c1c", "text": "#111827", "dim": "#4b5563",
}
THEME_AMOLED = {
    "bg": "#000000", "panel": "#0b0b0e", "panel2": "#141419",
    "edge_hi": "#303038", "edge_lo": "#000000",
    "accent": "#00e5ff", "warn": "#ff5252", "text": "#e8e8ee", "dim": "#8a8a96",
}

THEME_XP = {
    "bg": "#ECE9D8", "panel": "#F7F6EE", "panel2": "#E3E1D3",
    "edge_hi": "#FFFFFF", "edge_lo": "#ACA899",
    "accent": "#316AC5", "warn": "#CC0000", "text": "#1A1A1A", "dim": "#7C7C74",
}
try:
    _pref_t = None
    try:
        import json as _json_t
        with open(os.path.join(os.path.expanduser("~"), ".raw_studio.json")) as _fh_t:
            _pref_t = _json_t.load(_fh_t).get("theme")
    except Exception:
        _pref_t = None
except Exception:
    _pref_t = None
THEME = dict(THEME_DARK if _pref_t != "light" else THEME_LIGHT)


def _load_theme_name():
    """Вибір теми з ~/.raw_studio_theme: techno/classic/day; дефолт — techno."""
    try:
        with open(os.path.join(os.path.expanduser("~"), ".raw_studio_theme")) as f:
            v = f.read().strip()
        return v if v in ("techno", "classic", "day", "amoled", "xp", "auto") else "techno"
    except OSError:
        return "techno"


def _save_theme_name(name):
    try:
        with open(os.path.join(os.path.expanduser("~"), ".raw_studio_theme"), "w") as f:
            f.write(name)
    except OSError:
        pass


def _load_theme_setting():
    return _load_theme_name() == "techno"


def _save_theme_setting(dark):
    _save_theme_name("techno" if dark else "classic")


def _resolve_theme(name):
    """«auto» → day (08:00-18:59) або techno (інший час) за системним годинником."""
    if name != "auto":
        return name
    import time
    return "day" if 8 <= time.localtime().tm_hour < 19 else "techno"


def _theme_gradient(c1, c2, n):
    """n hex-кольорів від c1 до c2 (градієнт банера, v3.11.0)."""
    def _h(c):
        c = c.lstrip("#")
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r1, g1, b1 = _h(c1)
    r2, g2, b2 = _h(c2)
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        out.append("#%02x%02x%02x" % (round(r1 + (r2 - r1) * t),
                                      round(g1 + (g2 - g1) * t),
                                      round(b1 + (b2 - b1) * t)))
    return out


def _apply_techno_theme(root):
    """Техно-дизайн GUI (v3.11.0): темна палітра, об'ємні bevel-кнопки,
    картки й вкладки — ttk-тема clam дає керування світлом/тінню граней."""
    THEME.update(THEME_DARK)
    return _configure_styles(root)


def _apply_classic_theme(root):
    """Класична світла тема (v3.12.0): переписує палітру THEME у світлі тони."""
    THEME.update(THEME_LIGHT)
    return _configure_styles(root)


def _apply_day_theme(root):
    """Світло-сіра тема «День» (v3.13.0)."""
    THEME.update(THEME_DAY)
    return _configure_styles(root)


def _apply_amoled_theme(root):
    """Чорна AMOLED-тема (v3.14.0)."""
    THEME.update(THEME_AMOLED)
    return _configure_styles(root)


def _apply_xp_theme(root):
    """Тема «Windows XP» (Luna): світло-бежева палітра, синій акцент (v3.16.0)."""
    THEME.update(THEME_XP)
    return _configure_styles(root)


def _apply_app_theme(root, name):
    """Диспетчер тем (v3.14.0): techno / classic / day / amoled."""
    if name == "classic":
        return _apply_classic_theme(root)
    if name == "day":
        return _apply_day_theme(root)
    if name == "amoled":
        return _apply_amoled_theme(root)
    if name == "xp":
        return _apply_xp_theme(root)
    return _apply_techno_theme(root)


def _configure_styles(root):
    import tkinter as tk
    from tkinter import ttk
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    bg, panel, panel2 = THEME["bg"], THEME["panel"], THEME["panel2"]
    hi, lo = THEME["edge_hi"], THEME["edge_lo"]
    acc, txt, dim = THEME["accent"], THEME["text"], THEME["dim"]
    root.configure(bg=bg)
    for pat, val in (("*Toplevel.background", bg), ("*Menu.background", panel2),
                     ("*Menu.foreground", txt), ("*Listbox.background", panel2),
                     ("*Listbox.foreground", txt),
                     ("*TCombobox*Listbox.background", panel2),
                     ("*TCombobox*Listbox.foreground", txt),
                     ("*TCombobox*Listbox.selectBackground", "#123a52"),
                     ("*TCombobox*Listbox.selectForeground", "#eaf6ff")):
        root.option_add(pat, val)
    st.configure(".", background=panel, foreground=txt, fieldbackground=panel2,
                 bordercolor=lo, lightcolor=hi, darkcolor=lo, troughcolor=bg,
                 arrowcolor=acc, selectbackground=acc, selectforeground=bg)
    st.configure("TFrame", background=bg)
    st.configure("TLabel", background=bg, foreground=txt)
    st.configure("Card.TFrame", background=panel, relief="raised", borderwidth=1,
                 lightcolor=hi, darkcolor=lo)
    st.configure("Card.TLabelframe", background=panel, relief="raised", borderwidth=2,
                 lightcolor=hi, darkcolor=lo, bordercolor=lo)
    st.configure("Card.TLabelframe.Label", background=panel, foreground=acc,
                 font=("DejaVu Sans", 9, "bold"))
    st.configure("TButton", background=panel2, foreground=txt, relief="raised",
                 borderwidth=2, lightcolor=hi, darkcolor=lo, padding=(10, 5))
    st.map("TButton",
           background=[("pressed", bg), ("active", "#232d47")],
           lightcolor=[("pressed", lo), ("active", hi)],
           darkcolor=[("pressed", hi), ("active", lo)])
    st.configure("Accent.TButton", background="#123a52", foreground="#eaf6ff",
                 bordercolor=acc, lightcolor="#2f6f96", darkcolor="#0a2436")
    st.map("Accent.TButton",
           background=[("pressed", "#0c2c40"), ("active", "#175473")],
           lightcolor=[("pressed", "#0a2436"), ("active", "#2f6f96")],
           darkcolor=[("pressed", "#2f6f96"), ("active", "#0a2436")])
    st.configure("TEntry", fieldbackground=panel2, foreground=txt, insertcolor=acc,
                 lightcolor=lo, darkcolor=hi, bordercolor=lo, padding=2)
    st.configure("TCombobox", fieldbackground=panel2, background=panel2,
                 foreground=txt, arrowcolor=acc, lightcolor=lo, darkcolor=hi,
                 bordercolor=lo, padding=2)
    st.configure("TCheckbutton", background=bg, foreground=txt, focuscolor=acc)
    st.map("TCheckbutton", background=[("active", bg)])
    st.configure("TRadiobutton", background=bg, foreground=txt, focuscolor=acc)
    st.map("TRadiobutton", background=[("active", bg)])
    st.configure("TLabelframe", background=bg, bordercolor=lo, lightcolor=hi, darkcolor=lo)
    st.configure("TLabelframe.Label", background=bg, foreground=dim)
    st.configure("TNotebook", background=bg, bordercolor=lo, lightcolor=hi,
                 darkcolor=lo, tabmargins=[4, 4, 4, 0])
    st.configure("TNotebook.Tab", background=panel2, foreground=dim, padding=(12, 5),
                 borderwidth=2, lightcolor=hi, darkcolor=lo)
    st.map("TNotebook.Tab", background=[("selected", "#123a52")],
           foreground=[("selected", "#eaf6ff")], lightcolor=[("selected", acc)])
    st.configure("TScale", background=bg, troughcolor=panel2, lightcolor=hi, darkcolor=lo)
    st.configure("TProgressbar", background=acc, troughcolor=panel2,
                 lightcolor=acc, darkcolor="#0a2436")
    st.configure("Vertical.TScrollbar", background=panel2, troughcolor=bg,
                 bordercolor=lo, arrowcolor=dim, lightcolor=hi, darkcolor=lo)
    st.configure("Horizontal.TScrollbar", background=panel2, troughcolor=bg,
                 bordercolor=lo, arrowcolor=dim, lightcolor=hi, darkcolor=lo)
    st.configure("TSpinbox", fieldbackground=panel2, background=panel2, foreground=txt,
                 arrowcolor=acc, lightcolor=lo, darkcolor=hi, bordercolor=lo)
    return st


def _theme_icon_img():
    """Іконка трея 64x64 у кольорах поточної теми (Pillow; для pystray)."""
    try:
        from PIL import Image as _PI, ImageDraw as _PD
    except Exception:
        return None
    img = _PI.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = _PD.Draw(img)
    d.rounded_rectangle([3, 3, 61, 61], radius=14, fill=THEME["accent"])
    try:
        d.text((32, 33), "R", fill="#0a0e18", anchor="mm")
    except Exception:
        d.rectangle([26, 22, 38, 44], fill="#0a0e18")
    return img

def spectrum_peaks(freqs_mhz, dbs, prom_thr=6.0, min_sep_mhz=1.0, top=12):
    """Піки на оцифрованій кривій: локальні максимуми над медіанним фоном
    (поріг prom_thr дБ, роздільність min_sep_mhz, не більше top).
    Ширина — за рівнем пік −3 дБ. Повертає dict(freq, db, snr, width)."""
    x = np.asarray(freqs_mhz, dtype=float)
    y = np.asarray(dbs, dtype=float)
    if len(x) < 8:
        return []
    floor = float(np.median(y))
    loc = np.where((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    loc = [int(i) for i in loc if y[i] - floor >= prom_thr]
    loc.sort(key=lambda i: -y[i])
    picked = []
    for i in loc:
        if all(abs(x[i] - x[j]) >= min_sep_mhz for j, _w in picked):
            thr = y[i] - 3.0
            a = b = i
            while a > 0 and y[a - 1] >= thr:
                a -= 1
            while b < len(y) - 1 and y[b + 1] >= thr:
                b += 1
            w = x[b] - x[a]
            picked.append((i, w if w > 0 else float(x[1] - x[0])))
        if len(picked) >= top:
            break
    return [{"freq": float(x[i]), "db": float(y[i]), "snr": float(y[i] - floor),
             "width": float(w)} for i, w in picked]


# -- ШІ-модуль підбору завади (v3.7.0): база знань + радіоінженерні формули --
AI_TARGETS = {
    "GPS L1 C/A (ШПС)":             {"freq": 1575.42e6, "bw": 2.046e6, "rx": -130.0, "cls": "gnss"},
    "ГЛОНАСС G1 (FDMA)":            {"freq": 1602.0e6,  "bw": 7.0e6,   "rx": -130.0, "cls": "gnss"},
    "Galileo E1 (BOC(1,1))":        {"freq": 1575.42e6, "bw": 4.0e6,   "rx": -130.0, "cls": "gnss"},
    "BeiDou B1I":                   {"freq": 1561.098e6, "bw": 2.046e6, "rx": -130.0, "cls": "gnss"},
    "Вся смуга L1 (1559-1610 МГц)": {"freq": 1584.0e6,  "bw": 51.0e6,  "rx": -130.0, "cls": "gnss"},
    "Mesh COFDM (S-діапазон)":      {"freq": 2400.0e6,  "bw": 10.0e6,  "rx": None, "ptx": 30.0, "cls": "ofdm"},
    "Wi-Fi 2.4 ГГц (20 МГц)":       {"freq": 2437.0e6,  "bw": 20.0e6,  "rx": None, "ptx": 20.0, "cls": "ofdm"},
    "Аналогове FPV 5.8 ГГц":        {"freq": 5800.0e6,  "bw": 8.0e6,   "rx": None, "ptx": 27.0, "cls": "fm"},
    "LoRa 450–490 МГц (суб-ГГц)":    {"freq": 470.0e6,  "bw": 40.0e6,  "rx": None, "ptx": 20.0, "cls": "ofdm"},
    "LTE 1900 МГц (Band 2/PCS)":     {"freq": 1900.0e6, "bw": 20.0e6,  "rx": None, "ptx": 43.0, "cls": "ofdm"},
    "Відеолінк 2200–2360 МГц":       {"freq": 2280.0e6, "bw": 8.0e6,   "rx": None, "ptx": 30.0, "cls": "ofdm"},
    "Відеолінк 3300–3800 МГц":       {"freq": 3550.0e6, "bw": 8.0e6,   "rx": None, "ptx": 30.0, "cls": "ofdm"},
    "Радіотелеметрія 2020–2100 МГц": {"freq": 2060.0e6, "bw": 5.0e6,   "rx": None, "ptx": 27.0, "cls": "ofdm"},
}
AI_J_S_REQ = {"gnss": 30.0, "ofdm": 10.0, "fm": 15.0}


def fspl_db(d_km, f_mhz):
    """Затухання у вільному просторі (Фріс), дБ: 32.45 + 20lg(км) + 20lg(МГц)."""
    return 32.45 + 20.0 * math.log10(max(d_km, 0.01)) + 20.0 * math.log10(max(f_mhz, 0.1))


def ai_recommend_jam(target_key, pa_dbm, dist_km, dj_km, sdr_fs_hz,
                     want_spoof=False, f_hz=None):
    """Експертний підбір завади (v3.7.0). Вхід: ціль з AI_TARGETS, потужність
    підсилювача (дБм), дальність розкриття сигналу і дистанція завади (км),
    fs обраної моделі SDR. Вихід: dict(kind, kind_name, f_hz, bw, amp, js,
    js_req, margin, p_req, d_max, sp_off, sp_drift, fs_hz, reasons, warnings).
    Правила: DSSS-ГНСС -- CW при ПП < 17 дБм, чирп 17-27, шум >= 27 (спуфінг
    опційно, J/S ~ 6 дБ); OFDM -- тільки шум на смугу канала; вузький ЧМ --
    точкова CW. Смуга обмежується 0.4*fs SDR. Для стендових симуляційних
    записів: апаратурою не керує."""
    t = AI_TARGETS[target_key]
    f = float(f_hz) if f_hz and f_hz > 0 else t["freq"]
    fm = f / 1e6
    bw = t["bw"]
    reasons, warnings = [], []
    if t["rx"] is not None:
        ps = t["rx"]
        reasons.append("%s: сигнал у приймачі %.0f дБм (паспортна густина потоку ГНСС)"
                       % (target_key, ps))
    else:
        pl = fspl_db(dist_km, fm)
        ps = t["ptx"] - pl
        reasons.append("%s: передавач %.0f дБм, межа дальності %.6g км: %.0f - ФСМ %.1f = %.1f дБм"
                       % (target_key, t["ptx"], dist_km, t["ptx"], pl, ps))
    bw_cap = 0.4 * float(sdr_fs_hz)
    if bw > bw_cap:
        warnings.append("смуга цілі %.6g МГц ширша за 0.4*fs SDR (%.6g МГц) -- смуга завади звужена"
                        % (bw / 1e6, bw_cap / 1e6))
        bw = bw_cap
    if t["cls"] == "gnss" and want_spoof:
        kind, kind_name = "spoof", "Spoofing GPS"
        js_req = 6.0
        sp_off, sp_drift = 500.0, 10.0
        reasons.append("DSSS на -130 дБм: спуфінгу вистачає ~6 дБ над сигналом; тестує моніторинг "
                       "цілісності приймача, а не тільки нулі CRPA")
    elif t["cls"] == "ofdm":
        kind, kind_name = "jam_noise", "Завада (шум)"
        js_req = AI_J_S_REQ["ofdm"]
        sp_off = sp_drift = 0.0
        reasons.append("OFDM/COFDM без вузькосмугового захвату: точкова CW неефективна, "
                       "потрібна загороджувальна завада на смугу канала %.0f МГц" % (t["bw"] / 1e6))
    elif t["cls"] == "fm":
        kind, kind_name = "jam_cw", "Завада CW"
        js_req = AI_J_S_REQ["fm"]
        sp_off = sp_drift = 0.0
        reasons.append("вузькосмуговий ЧМ/аналоговий канал: вся потужність підсилювача в одній "
                       "лінії -- точкова CW дає максимальну спектральну щільність")
    else:
        sp_off = sp_drift = 0.0
        js_req = AI_J_S_REQ["gnss"]
        if pa_dbm < 17.0:
            kind, kind_name = "jam_cw", "Завада CW"
            reasons.append("ПП < 17 дБм: точкова CW на несучу -- виграш 10lg(смуга завади/сигнал) "
                           "проти шуму тієї ж потужності")
        elif pa_dbm < 27.0:
            kind, kind_name = "jam_chirp", "Завада (чирп)"
            reasons.append("ПП 17-27 дБм: чирп-огортання +-5 МГц -- висока пікова щільність із "
                           "періодичним оглядом смуги, тест детекторів і АПЧ")
        else:
            kind, kind_name = "jam_noise", "Завада (шум)"
            reasons.append("ПП >= 27 дБм: шумова загороджувальна смуга -- рівномірне придушення "
                           "сегмента, найжорсткіший тест адаптивних нулів CRPA")
    corr = max(0.0, 10.0 * math.log10(bw / t["bw"])) if kind == "jam_noise" else 0.0
    if corr > 0.05:
        reasons.append("шум розподілено на %.6g МГц проти %.6g МГц сигналу: ефективність -%.1f дБ "
                       "враховано в J/S" % (bw / 1e6, t["bw"] / 1e6, corr))
    plj = fspl_db(dj_km, fm)
    js = (pa_dbm - plj) - ps - corr
    margin = js - js_req
    p_req = js_req + ps + plj + corr
    fmax = max(0.01, 10.0 ** ((pa_dbm - ps - corr - js_req - 32.45
                               - 20.0 * math.log10(max(fm, 0.1))) / 20.0))
    if margin >= 0:
        reasons.append("J/S на %.6g км від завади: %.1f дБ (потрібно %.0f дБ) -- запас +%.1f дБ"
                       % (dj_km, js, js_req, margin))
    else:
        warnings.append("J/S %.1f дБ < %.0f дБ: бракує %.1f дБ -- потрібен ПП %.1f дБм, "
                        "або дистанція завади <= %.6g км" % (js, js_req, -margin, p_req, fmax))
    return {"kind": kind, "kind_name": kind_name, "f_hz": f, "bw": bw, "amp": 12000.0,
            "js": js, "js_req": js_req, "margin": margin, "p_req": p_req, "d_max": fmax,
            "sp_off": sp_off, "sp_drift": sp_drift, "fs_hz": float(sdr_fs_hz),
            "reasons": reasons, "warnings": warnings}


APP_TITLE = "RAW Studio v%s — .raw АЦП: блоки 100М…6Г, канали 64 МГц, АЧХ + калібрування" % APP_VERSION
DEFAULT_FS = 12_000_000_000          # 12 ГГц
DEFAULT_NCH = 8
NFFT = 2 ** 20
SIG_MIN_HZ = 100e6
SIG_MAX_HZ = 6000e6
CHANNEL_BW = 64e6
FREQ_BLOCKS = [(100e6, 500e6), (500e6, 1700e6), (1200e6, 2500e6),
               (2400e6, 3600e6), (3400e6, 4600e6), (4400e6, 6000e6)]

SDR_PRESETS = {
    # tx_dbm — макс. вихідна потужність передавача за паспортом (типові максимуми);
    # None — приймальна платформа (без TX), заваду не створює
    # fs/смуга/діапазон — з офіційних специфікацій виробників:
    # BladeRF 2.0 micro xA4/xA9: nuand.com (47 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц);
    # LimeSDR USB / Micro: myriadrf.org (100 кГц–3.8 ГГц, 61.44 МС/с, смуга 61.44 МГц);
    # LimeSDR Mini: myriadrf.org (10 МГц–3.5 ГГц, 30.72 МС/с, смуга 20 МГц);
    # LimeSDR XTRX: wiki.myriadrf.org (100 МГц–3.8 ГГц, 80 МС/с, смуга 80 МГц);
    # ADALM-Pluto: wiki.analog.com (AD9363, 325 МГц–3.8 ГГц, 61.44 МС/с, смуга 20 МГц);
    # PlutoSky R2 / PLUTO+: opensourcesdrlab.com (AD9361/AD9363, модифіковане
    #                   ПЗ: 70 МГц–6 ГГц, 61.44 МС/с, смуга 20 МГц);
    # RigExpert Fobos SDR: rigexpert.com (100 кГц–6 ГГц, 50 МС/с, смуга 50 МГц, 14 біт);
    # HackRF One: hackrf.readthedocs.io (1 МГц–6 ГГц, 2–20 МС/с, 8 біт);
    # HackRF Pro: greatscottgadgets.com (10 МГц–6 ГГц, 20 МС/с, 8 біт);
    # USRP B210: ettus.com (70 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц, AD9361);
    # USRP X310 (UBX-160): kb.ettus.com (10 МГц–6 ГГц, 200 МС/с, IBW 160 МГц);
    # RTL-SDR Blog V4: rtl-sdr.com (500 кГц–1.766 ГГц, 2.56 МС/с, 8 біт);
    # KrakenSDR: krakenrf.com (5 каналів RTL-SDR, 24 МГц–1.766 ГГц, 2.56 МС/с, 8 біт);
    # KiwiSDR: kiwisdr.com (10 кГц–30 МГц, ADC LTC2208 66.67 МС/с, 16 біт);
    # Aaronia SPECTRAN V6 X: downloads.aaronia.com (10 МГц–8 ГГц,
    #                   500 МС/с 16-біт, RTBW 120 МГц / до 245 МГц IQ);
    # Aaronia SPECTRAN V6 ECO: downloads.aaronia.com (9 кГц–6 ГГц,
    #                   500 МС/с 16-біт, RTBW 44 МГц / до 88 (2×44) МГц).
    # --- BladeRF (Nuand) ---
    "BladeRF 2.0 micro xA4": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 47e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 6},
    "BladeRF 2.0 micro xA9": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 47e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 6},
    # --- LimeSDR (Lime Microsystems / Myriad-RF) ---
    "LimeSDR USB": {"fs_hz": 61.44e6, "bw_hz": 61.44e6, "f_lo": 100e3, "f_hi": 3800e6, "bits": 12, "tx_dbm": 10},
    "LimeSDR Mini": {"fs_hz": 30.72e6, "bw_hz": 20e6, "f_lo": 10e6, "f_hi": 3500e6, "bits": 12, "tx_dbm": 6},
    "LimeSDR Micro": {"fs_hz": 61.44e6, "bw_hz": 61.44e6, "f_lo": 100e3, "f_hi": 3800e6, "bits": 12, "tx_dbm": 2},
    "LimeSDR XTRX": {"fs_hz": 80e6, "bw_hz": 80e6, "f_lo": 100e6, "f_hi": 3800e6, "bits": 12, "tx_dbm": 4},
    # --- Pluto (Analog Devices і модифікації) ---
    "ADALM-Pluto (61.44 МС/с)": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 325e6, "f_hi": 3800e6, "bits": 12, "tx_dbm": 8},
    "PlutoSky R2": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 10},
    "PLUTO+": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 10},
    # --- RigExpert ---
    "RigExpert Fobos SDR": {"fs_hz": 50e6, "bw_hz": 50e6, "f_lo": 100e3, "f_hi": 6000e6, "bits": 14, "tx_dbm": 10},
    # --- Great Scott Gadgets ---
    "HackRF One": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 1e6, "f_hi": 6000e6, "bits": 8, "tx_dbm": 15},
    "HackRF Pro": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 8, "tx_dbm": 15},
    # --- Ettus / NI USRP ---
    "USRP B210": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 10},
    "USRP X310 (UBX-160)": {"fs_hz": 200e6, "bw_hz": 160e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 12, "tx_dbm": 15},
    # --- RTL-SDR ---
    "RTL-SDR Blog V4": {"fs_hz": 2.56e6, "bw_hz": 2.56e6, "f_lo": 500e3, "f_hi": 1766e6, "bits": 8, "tx_dbm": None},
    "KrakenSDR": {"fs_hz": 2.56e6, "bw_hz": 2.56e6, "f_lo": 24e6, "f_hi": 1766e6, "bits": 8, "tx_dbm": None},
    # --- KiwiSDR ---
    "KiwiSDR": {"fs_hz": 66.67e6, "bw_hz": 33e6, "f_lo": 10e3, "f_hi": 30e6, "bits": 16, "tx_dbm": None},
    # --- Aaronia ---
    "Aaronia SPECTRAN V6 X": {"fs_hz": 500e6, "bw_hz": 120e6, "f_lo": 10e6, "f_hi": 8000e6, "bits": 16, "tx_dbm": None},
    "Aaronia SPECTRAN V6 ECO": {"fs_hz": 500e6, "bw_hz": 44e6, "f_lo": 9e3, "f_hi": 6000e6, "bits": 16, "tx_dbm": None},
    "Інший (власноруч)": {},
}

# ---- v3.22.0: таблиця параметрів якісної завади (Тип сигналу x Модель SDR) ----
JQ_EXPLICIT = {
    "GPS L1": "GPS L1 C/A (ШПС)", "GPS L2": "GPS L1 C/A (ШПС)", "GPS L5": "GPS L1 C/A (ШПС)",
    "GPS L1 C/A (ШПС)": "GPS L1 C/A (ШПС)",
    "ГЛОНАСС G1": "ГЛОНАСС G1 (FDMA)", "ГЛОНАСС G1 FDMA (ШПС)": "ГЛОНАСС G1 (FDMA)",
    "ГЛОНАСС G2": "ГЛОНАСС G1 (FDMA)", "ГЛОНАСС G3": "ГЛОНАСС G1 (FDMA)",
    "Galileo E1": "Galileo E1 (BOC(1,1))", "Galileo E1 BOC (ШПС)": "Galileo E1 (BOC(1,1))",
    "Galileo E5a": "Galileo E1 (BOC(1,1))", "Galileo E5b": "Galileo E1 (BOC(1,1))",
    "Galileo E6": "Galileo E1 (BOC(1,1))",
    "BeiDou B1": "BeiDou B1I", "BeiDou B1I (ШПС)": "BeiDou B1I", "BeiDou B3": "BeiDou B1I",
    "Комплекс GNSS L1 (E1+G1+B1+L1)": "Вся смуга L1 (1559-1610 МГц)",
    "GNSS (L-смуга)": "Вся смуга L1 (1559-1610 МГц)",
    "SAR": "Вся смуга L1 (1559-1610 МГц)",
    "Завада CW 1575.42 (глушилка)": "GPS L1 C/A (ШПС)",
    "Завада широкосмугова (шум 20 МГц)": "Вся смуга L1 (1559-1610 МГц)",
    "Завада чирп (±5 МГц)": "GPS L1 C/A (ШПС)",
    "Spoofing GPS — зміщення +500 Гц": "GPS L1 C/A (ШПС)",
    "Spoofing GPS — плаваюча (+10 Гц/с)": "GPS L1 C/A (ШПС)",
    "TGTX5050A RF1 (GNSS L1)": "GPS L1 C/A (ШПС)", "TGTX5050A RF2 (GNSS L1)": "GPS L1 C/A (ШПС)",
    "2G (GSM)": "LTE 1900 МГц (Band 2/PCS)", "3G (WCDMA)": "LTE 1900 МГц (Band 2/PCS)",
    "4G (LTE)": "LTE 1900 МГц (Band 2/PCS)", "5G (NR)": "LTE 1900 МГц (Band 2/PCS)",
    "NB-IoT": "LTE 1900 МГц (Band 2/PCS)",
    "WiFi (OFDM)": "Wi-Fi 2.4 ГГц (20 МГц)", "QAM": "Wi-Fi 2.4 ГГц (20 МГц)",
    "COFDM (DVB)": "Wi-Fi 2.4 ГГц (20 МГц)", "FHSS": "Wi-Fi 2.4 ГГц (20 МГц)",
    "CVBS": "Аналогове FPV 5.8 ГГц",
    "LoRa": "LoRa 450–490 МГц (суб-ГГц)", "LoRa VHF (150–300)": "LoRa 450–490 МГц (суб-ГГц)",
    "LoRa 433 МГц": "LoRa 450–490 МГц (суб-ГГц)", "LoRaWAN": "LoRa 450–490 МГц (суб-ГГц)",
    "UNB / Narrowband": "LoRa 450–490 МГц (суб-ГГц)", "FSK / GFSK": "LoRa 450–490 МГц (суб-ГГц)",
    "Завада блок 450–490 (LoRa)": "LoRa 450–490 МГц (суб-ГГц)",
    "Завада блок 1900 (LTE)": "LTE 1900 МГц (Band 2/PCS)",
    "Завада блок 2200–2360 (Video)": "Відеолінк 2200–2360 МГц",
    "Завада блок 3300–3800 (Video)": "Відеолінк 3300–3800 МГц",
    "Завада блок 2020–2100 (Телеметрія, ниж.)": "Радіотелеметрія 2020–2100 МГц",
    "Завада блок 2020–2100 (Телеметрія, верх.)": "Радіотелеметрія 2020–2100 МГц",
}
# точні несучі для систем, у яких ціль-заміна з іншої частоти (план RNSS)
JQ_FREQ_OVERRIDES = {
    "GPS L2": 1227.60e6, "GPS L5": 1176.45e6,
    "ГЛОНАСС G2": 1248.06e6, "ГЛОНАСС G3": 1202.025e6,
    "Galileo E5a": 1176.45e6, "Galileo E5b": 1207.14e6, "Galileo E6": 1278.75e6,
    "BeiDou B3": 1268.52e6, "SAR": 1544.50e6,
}

def _jq_resolve_target(sig_key):
    """Тип сигналу (Крок 2) → ціль AI_TARGETS для розрахунку завади."""
    if sig_key in AI_TARGETS:
        return sig_key
    if sig_key in JQ_EXPLICIT:
        return JQ_EXPLICIT[sig_key]
    lk = sig_key.lower()
    if "тлм" in lk or "телеметрі" in lk or "2020" in lk:
        return "Радіотелеметрія 2020–2100 МГц"
    if "відеолінк" in lk or "video" in lk or "2200" in lk:
        return "Відеолінк 2200–2360 МГц"
    if "3300" in lk or "3800" in lk:
        return "Відеолінк 3300–3800 МГц"
    if "fpv" in lk or "cvbs" in lk or "аналог" in lk:
        return "Аналогове FPV 5.8 ГГц"
    if "mesh" in lk:
        return "Mesh COFDM (S-діапазон)"
    if "tgtx" in lk or "спуфер" in lk or "спуфінг" in lk:
        return "GPS L1 C/A (ШПС)"
    if "глонасс" in lk:
        return "ГЛОНАСС G1 (FDMA)"
    if "galileo" in lk:
        return "Galileo E1 (BOC(1,1))"
    if "beidou" in lk:
        return "BeiDou B1I"
    if "gps" in lk or "смуга l1" in lk or "комплекс l1" in lk:
        return "Вся смуга L1 (1559-1610 МГц)"
    if "gnss" in lk or " l1" in lk or "l2" in lk or "l5" in lk:
        return "Вся смуга L1 (1559-1610 МГц)"
    if "lte" in lk or "nr" in lk or "iot" in lk or "gsm" in lk or "wcdma" in lk:
        return "LTE 1900 МГц (Band 2/PCS)"
    if "wifi" in lk or "ofdm" in lk or "cofdm" in lk or "qam" in lk or "fhss" in lk:
        return "Wi-Fi 2.4 ГГц (20 МГц)"
    if "lora" in lk or "fsk" in lk or "unb" in lk or "narrowband" in lk:
        return "LoRa 450–490 МГц (суб-ГГц)"
    return None

# Типові антени для стенду: орієнтовні паспортні значення, редагуються у полях
JQ_ANTENNAS = {
    "Штирьова 0.25λ": dict(g=2.0, f_lo=30e6, f_hi=6000e6,
                           vswr=[(30e6, 2.5), (1000e6, 1.5), (3000e6, 1.8), (6000e6, 2.5)]),
    "Диполь": dict(g=2.15, f_lo=30e6, f_hi=1000e6,
                   vswr=[(30e6, 2.0), (300e6, 1.4), (1000e6, 2.0)]),
    "Логоперіодична (LPDA)": dict(g=7.0, f_lo=400e6, f_hi=6000e6,
                                  vswr=[(400e6, 2.0), (2000e6, 1.6), (6000e6, 1.8)]),
    "Ягі 7 ел.": dict(g=11.0, f_lo=300e6, f_hi=1000e6,
                      vswr=[(300e6, 1.8), (600e6, 1.4), (1000e6, 1.9)]),
    "Спіральна (кругова)": dict(g=12.0, f_lo=1000e6, f_hi=6000e6,
                                vswr=[(1000e6, 1.9), (3000e6, 1.5), (6000e6, 2.0)]),
    "Панель секторна": dict(g=14.0, f_lo=690e6, f_hi=3800e6,
                            vswr=[(690e6, 1.8), (1900e6, 1.5), (3800e6, 1.8)]),
    "Рупорна": dict(g=16.0, f_lo=800e6, f_hi=8000e6,
                    vswr=[(800e6, 1.9), (3000e6, 1.4), (8000e6, 1.8)]),
    "Патч GNSS": dict(g=4.0, f_lo=1559e6, f_hi=1610e6,
                      vswr=[(1559e6, 1.6), (1575.42e6, 1.4), (1610e6, 1.7)]),
    "Vivaldi": dict(g=9.0, f_lo=800e6, f_hi=8000e6,
                    vswr=[(800e6, 2.0), (3000e6, 1.6), (8000e6, 1.9)]),
    "Власна (ручне введення)": dict(g=0.0, f_lo=0.0, f_hi=0.0, vswr=[]),
}

def _vswr_loss_db(v):
    """Втрати на неузгодженість, дБ: −10·lg(1−|Γ|²), Γ=(КСВ−1)/(КСВ+1)."""
    if v is None or v <= 1.0:
        return 0.0
    g = (float(v) - 1.0) / (float(v) + 1.0)
    return max(0.0, -10.0 * math.log10(max(1e-9, 1.0 - g * g)))

def _vswr_at(rows, f):
    """КСВ на частоті f (Гц) — лінійна інтерполяція по таблиці [(f_Гц, КСВ), …]."""
    pts = sorted((float(a), float(b)) for a, b in (rows or []) if float(b) >= 1.0)
    if not pts:
        return None
    if f <= pts[0][0]:
        return pts[0][1]
    if f >= pts[-1][0]:
        return pts[-1][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if f0 <= f <= f1:
            w = (f - f0) / max(1e-9, f1 - f0)
            return v0 + w * (v1 - v0)
    return pts[-1][1]


# ---- v4.0.0: діаграма спрямованості (азимутальна площина, інженерна модель) ----
ANT_BEAMS = {
    "Штирьова 0.25λ": dict(hpb=90.0, fbr=6.0),
    "Диполь": dict(hpb=78.0, fbr=0.0),
    "Логоперіодична (LPDA)": dict(hpb=55.0, fbr=18.0),
    "Ягі 7 ел.": dict(hpb=40.0, fbr=22.0),
    "Спіральна (кругова)": dict(hpb=50.0, fbr=15.0),
    "Панель секторна": dict(hpb=65.0, fbr=25.0),
    "Рупорна": dict(hpb=30.0, fbr=30.0),
    "Патч GNSS": dict(hpb=80.0, fbr=12.0),
    "Vivaldi": dict(hpb=45.0, fbr=20.0),
    "Власна (ручне введення)": dict(hpb=60.0, fbr=20.0),
}

def ant_pattern_db(name, deg, hpb=None, fbr=None):
    """Ослаблення від головної пелюстки, дБ: −12·(θ/θ₃)² до рівня заднього/бічного
    проміння fbr (типова інженерна модель конверта; 0 дБ — напрямок головної пелюстки)."""
    p = dict(ANT_BEAMS.get(name) or dict(hpb=60.0, fbr=20.0))
    if hpb and float(hpb) > 0.0:
        p["hpb"] = float(hpb)
    if fbr is not None and float(fbr) >= 0.0:
        p["fbr"] = float(fbr)
    th = abs(((float(deg) + 180.0) % 360.0) - 180.0)
    return -min(12.0 * (th / p["hpb"]) ** 2, p["fbr"])

def ant_scale_hidden(var):
    """True, якщо перемикач «приховати шкалу КСВ» увімкнений (v4.3.0).
    Безпечний до відсутньої змінної і до Tk-винятків."""
    try:
        return bool(var and var.get())
    except Exception:
        return False


def ant_vswr_scale(rows, nseg=36):
    """Шкала КСВ-втрат для полярної діаграми (v4.2.0): за рядками КСВ
    (МГц, КСВ) повертає (fmin, fmax, [(доля кута 0..1, втрати дБ), ...])
    — кут 0° відповідає fmin, 180° — fmax; None, якщо рядків нема."""
    if not rows:
        return None
    fmin = min(f for f, v in rows)
    fmax = max(f for f, v in rows)
    if fmax <= fmin:
        fmax = fmin * 1.5
    out = []
    for i in range(nseg + 1):
        fr = i / float(nseg)
        f = fmin + (fmax - fmin) * fr
        out.append((fr, _vswr_loss_db(_vswr_at(rows, f))))
    return fmin, fmax, out


def draw_ant_pattern(ax, name, gain_dbi, title="", hpb=None, fbr=None):
    """Полярна діаграма спрямованості (дБі) на готовій полярній осі matplotlib."""
    th = list(range(0, 361, 2))
    r = [max(0.0, float(gain_dbi) + ant_pattern_db(name, t, hpb=hpb, fbr=fbr)) for t in th]
    thetas = [math.radians(t) for t in th]
    ax.clear()
    ax.plot(thetas, r, color="#1f77b4", lw=1.6)
    ax.fill_between(thetas, 0.0, r, color="#1f77b4", alpha=0.18)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(range(0, 360, 30))
    ax.set_rlabel_position(90)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.35)
    ax.set_title(title or ("Діаграма спрямованості: %s (%.1f дБі)" % (name, float(gain_dbi))),
                 fontsize=9, pad=12)

def jam_quality_params(sig_key, sdr_key, pa_dbm=20.0, dist_km=5.0, dj_km=1.0,
                       amp_db=0.0, ant_gain_dbi=0.0, vswr=None, ant_rng=None):
    """Параметри якісної завади для Генератора (v3.22.0): за Типом сигналу
    (Крок 2) і Моделлю SDR (Крок 1). База — ai_recommend_jam; вхід: потужність
    ПП (дБм) + підсилювач (дБ) + антена (дБі) − втрати КСВ; дальність розкриття
    і дистанція завади (км). Вихід:
    dict(target, kind, kind_name, f_hz, bw_hz, js_req, p_req, d_max, fs_hz,
    min_us, sp_off, sp_drift, tx_dbm, sdr_d_max, g_total, rows, warn). Стендовий інструмент: апаратурою
    не керує."""
    d = SDR_PRESETS.get(sdr_key) or {}
    fs = float(d.get("fs_hz") or DEFAULT_FS)
    f_lo = float(d.get("f_lo", 0.0))
    f_hi = float(d.get("f_hi", 6000e6))
    min_us = 4.0 / fs * 1e6
    tx = d.get("tx_dbm")
    tx_txt = "— (тільки прийом)" if tx is None else "%.1f дБм" % float(tx)
    tg = _jq_resolve_target(sig_key)
    f_jam = JQ_FREQ_OVERRIDES.get(sig_key) or (AI_TARGETS[tg]["freq"] if tg else (f_lo + f_hi) / 2.0)
    _vl = _vswr_loss_db(_vswr_at(vswr, f_jam))
    g_total = float(amp_db) + float(ant_gain_dbi) - _vl
    pa_eff = float(pa_dbm) + g_total
    ant_warn = ""
    if ant_rng and f_jam and not (ant_rng[0] <= f_jam <= ant_rng[1]):
        ant_warn = ("частота %.4g МГц поза робочим діапазоном антени %.4g–%.4g МГц"
                    % (f_jam / 1e6, ant_rng[0] / 1e6, ant_rng[1] / 1e6))
    sdr_txt = "%s — fs %.4g МС/с, %.4g–%.4g МГц" % (sdr_key, fs / 1e6, f_lo / 1e6, f_hi / 1e6)
    if tg is None:
        f_hz = (f_lo + f_hi) / 2.0
        bw = 0.4 * fs
        rows = [
            ("Ціль (Тип сигналу)", "%s — не розпізнано, універсальний бар'єр" % sig_key),
            ("Рекомендований тип завади", "Завада (шум)"),
            ("Частота завади, МГц", "%.4g" % (f_hz / 1e6)),
            ("Смуга завади, МГц", "%.4g (0.4·fs SDR)" % (bw / 1e6)),
            ("Тракт: підсилювач + антена − КСВ, дБ", "%+.1f" % g_total),
            ("J/S потрібно, дБ", "≈ 10"),
            ("Модель SDR", sdr_txt),
            ("Мін. ширина імпульса, мкс (4 відліки)", "%.4g" % min_us),
            ("Обґрунтування", "ціль за Типом сигналу не розпізнано — шумовий бар'єр 0.4·fs у центрі діапазону плати"),
            ("Вихідна потужність моделі SDR, дБм", tx_txt),
        ]
        return dict(target=sig_key, kind="jam_noise", kind_name="Завада (шум)",
                    f_hz=f_hz, bw_hz=bw, js_req=10.0, p_req=None, d_max=None,
                    fs_hz=fs, min_us=min_us, sp_off=0.0, sp_drift=0.0,
                    tx_dbm=tx, sdr_d_max=None, g_total=g_total, rows=rows, warn="")
    bk = ai_recommend_jam(tg, pa_eff, float(dist_km), float(dj_km), fs,
                          f_hz=JQ_FREQ_OVERRIDES.get(sig_key))
    sdr_d_max = None
    if tx is None:
        jamq_alert = ("%s — приймальна платформа (без передавача): заваду цією моделлю не створити" % sdr_key)
    else:
        bks = ai_recommend_jam(tg, float(tx) + g_total, float(dist_km), float(dj_km), fs,
                               f_hz=JQ_FREQ_OVERRIDES.get(sig_key))
        sdr_d_max = float(bks["d_max"])
        if float(tx) + g_total < float(bk["p_req"]):
            jamq_alert = ("обрана модель SDR не тягне потрібну потужність: на дистанції %.4g км "
                          "потрібно %.1f дБм, з трактом %.1f дБм — очікуваний радіус придушення %.4g км"
                          % (float(dj_km), bk["p_req"], float(tx) + g_total, sdr_d_max))
        else:
            jamq_alert = ""
    rows = [
        ("Ціль (Тип сигналу)", "%s → %s" % (sig_key, tg)),
        ("Рекомендований тип завади", bk["kind_name"]),
        ("Частота завади, МГц", "%.4g" % (bk["f_hz"] / 1e6)),
        ("Смуга завади, МГц", "%.4g" % (bk["bw"] / 1e6)),
        ("J/S потрібно, дБ", "%.1f" % bk["js_req"]),
        ("Потужність ПП для J/S, дБм", "%.1f" % bk["p_req"],
         "#2e7d32" if pa_eff >= float(bk["p_req"]) else THEME["warn"]),
        ("Тракт: підсилювач + антена − КСВ, дБ", "%+.1f" % g_total),
        ("Макс. дальність завади, км", "%.4g" % bk["d_max"]),
        ("Вихідна потужність моделі SDR, дБм", tx_txt),
        ("Очікуваний радіус придушення, км", "—" if sdr_d_max is None else "%.4g" % sdr_d_max),
        ("Модель SDR", sdr_txt),
        ("Мін. ширина імпульса, мкс (4 відліки)", "%.4g" % min_us),
        ("Обґрунтування", (bk.get("reasons") or ["-"])[0]),
    ]
    return dict(target=tg, kind=bk["kind"], kind_name=bk["kind_name"],
                f_hz=bk["f_hz"], bw_hz=bk["bw"], js_req=bk["js_req"],
                p_req=bk["p_req"], d_max=bk["d_max"], fs_hz=fs, min_us=min_us,
                sp_off=bk.get("sp_off", 0.0), sp_drift=bk.get("sp_drift", 0.0),
                tx_dbm=tx, sdr_d_max=sdr_d_max, g_total=g_total,
                rows=rows,
                warn=(" ".join(bk.get("warnings", []))
                      + (" " + jamq_alert if jamq_alert else "")
                      + (" " + ant_warn if ant_warn else "")).strip())

WAVE_KINDS = ["Синус", "Меандр", "Трикутник", "Імпульс", "Шум", "GNSS L1",
              "GPS L1", "Galileo E1", "ГЛОНАСС G1", "BeiDou B1", "Mesh COFDM",
              "Завада CW", "Завада (шум)", "Завада (чирп)", "Spoofing GPS",
              "Завада RGPO", "Завада VGPO", "Завада (пакети)"]
KIND_KEY = {"Синус": "sine", "Меандр": "square", "Трикутник": "tri",
            "Імпульс": "pulse", "Шум": "noise", "GNSS L1": "gnss_l1",
            "GPS L1": "gps_l1", "Galileo E1": "gal_e1",
            "ГЛОНАСС G1": "glo_g1", "BeiDou B1": "bds_b1", "Mesh COFDM": "mesh",
            "Завада CW": "jam_cw", "Завада (шум)": "jam_noise",
            "Завада (чирп)": "jam_chirp", "Завада RGPO": "jam_rgpo",
            "Завада VGPO": "jam_vgpo", "Spoofing GPS": "spoof",
            "Завада (пакети)": "jam_burst"}
GNSS_KEYS = ("gnss_l1", "gps_l1", "gal_e1", "glo_g1", "bds_b1")
JAM_KEYS = ("jam_cw", "jam_noise", "jam_chirp", "jam_rgpo", "jam_vgpo")
QAM_MAP = {"Адаптивна (BPSK…1024QAM)": 0, "BPSK": 1, "QPSK": 2, "16QAM": 4,
           "64QAM": 6, "256QAM": 8, "1024QAM": 10}

ACHX_PRESETS = ["Плоский", "Спад −3 дБ/окт", "Смуговий 100М–6Г",
                "Провал (notch)", "Користувацький"]
ACHX_MODES = {"Лінійна": "linear", "Кубічний сплайн": "spline", "Сплайн у log f": "logspline"}
ACHX_EXTRAPS = {"За краями (clamped)": "clamp", "Лінійно далі": "linear",
                "Нуль поза смугою": "zero"}


def human_freq(f):
    a = abs(f)
    if a >= 1e9:
        return f / 1e9, "ГГц"
    if a >= 1e6:
        return f / 1e6, "МГц"
    if a >= 1e3:
        return f / 1e3, "кГц"
    return f, "Гц"


def parse_freq_hz(s):
    """-> Гц. Голе число — МГц (основний режим); суфікси мгц/м, ггц/g, кгц/k,
    гц/hz — явні одиниці; число ≥ 1e6 (напр. 2.4e9, 12000000000,
    1.32e+08) трактується як Гц (старі записи), менше — як МГц."""
    t = str(s).strip().lower()
    for cy, la in (("мгц", "mhz"), ("ггц", "ghz"), ("кгц", "khz"), ("гц", "hz"),
                   ("м", "m"), ("г", "g"), ("к", "k")):
        if t.endswith(cy):
            t = t[:-len(cy)] + la
            break
    for suf, mult in (("mhz", 1e6), ("ghz", 1e9), ("khz", 1e3), ("hz", 1.0),
                      ("m", 1e6), ("g", 1e9), ("k", 1e3)):
        if t.endswith(suf):
            return float(t[:-len(suf)]) * mult
    v = float(t)
    if abs(v) >= 1e6:
        return v
    return v * 1e6


def freq_unit_for(fs):
    half = fs / 2.0
    if half >= 1e9:
        return 1e9, "ГГц"
    if half >= 1e6:
        return 1e6, "МГц"
    if half >= 1e3:
        return 1e3, "кГц"
    return 1.0, "Гц"


def build_blocks(bw_hz=CHANNEL_BW):
    """Решітка блоків і каналів; ширина канала — bw_hz (за замовчуванням 64 МГц);
    нумерація каналів наскрізна."""
    blocks = []
    g = 0
    for bi, (f0, f1) in enumerate(FREQ_BLOCKS, 1):
        chs = []
        c = 0
        f = f0
        while f < f1 - 1.0:
            w = min(bw_hz, f1 - f)
            g += 1
            c += 1
            chs.append({"global": g, "in_block": c, "f0": f, "f1": f + w,
                        "fc": f + w / 2.0, "w": w, "partial": w < bw_hz - 1.0})
            f += w
        blocks.append({"idx": bi, "f0": f0, "f1": f1, "channels": chs})
    return blocks


# ----------------------------------------------------------------------------
# МОДУЛЬ АЧХ АЦП — амплітуда генерації залежно від частоти
# ----------------------------------------------------------------------------
def build_achx_points(preset, notch_hz=1e9):
    """Профіль АЧХ: список точок (freq_Hz, gain_dB)."""
    f0, f1 = SIG_MIN_HZ, SIG_MAX_HZ
    if preset == "Плоский":
        return [(0.0, 0.0), (1e12, 0.0)]
    if preset == "Спад −3 дБ/окт":
        pts = [(0.0, 0.0)]
        for i in range(61):
            f = f0 * (f1 / f0) ** (i / 60.0)
            pts.append((f, -3.0 * math.log2(f / f0)))
        pts.append((1e12, -3.0 * math.log2(1e12 / f0)))
        return pts
    if preset == "Смуговий 100М–6Г":
        return [(0.0, -60.0), (f0, -3.0), (f0 * 1.02, 0.0),
                (f1 * 0.98, 0.0), (f1, -3.0), (1.2e12, -60.0)]
    if preset == "Провал (notch)":
        return [(0.0, 0.0), (notch_hz * 0.9, 0.0), (notch_hz, -40.0),
                (notch_hz * 1.1, 0.0), (1e12, 0.0)]
    raise ValueError("невідомий профіль: %s" % preset)


def parse_achx_table(text):
    """Рядки «частота gain_дБ»: '100e6 0', '1e9, -3', '6e9; -12' …"""
    pts = []
    for line in text.strip().splitlines():
        line = line.replace(",", " ").replace(";", " ").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("рядок АЧХ '%s' — потрібно: частота дБ" % line)
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки")
    pts.sort()
    return pts


# --- інтерполяція профілю: лінійна / кубічний сплайн / сплайн у log f ---------
def _spline_second_derivs(x, y):
    """Другі похідні природного кубічного сплайна (M[0]=M[-1]=0)."""
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n)); rhs = np.zeros(n)
    A[0, 0] = 1.0; A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def _spline_eval(x, y, M, xq, extrap="clamp"):
    """Значення сплайна в точках xq; екстраполяція: clamp | linear."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    out = np.empty(xq.shape, dtype=np.float64)
    if extrap == "clamp":
        xqc = np.clip(xq, x[0], x[-1])
    else:
        xqc = xq.copy()
    inside = (xqc >= x[0]) & (xqc <= x[-1])
    xi = xqc[inside]
    i = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    h = np.diff(x)[i]
    t = (xi - x[i]) / h
    out[inside] = (y[i] * (1 - t) + y[i + 1] * t
                   + h * h / 6.0 * (M[i] * ((1 - t) ** 3 - (1 - t))
                                    + M[i + 1] * (t ** 3 - t)))
    if extrap == "linear":
        m_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
        m_lo = (y[1] - y[0]) / (x[1] - x[0])
        out[xq > x[-1]] = y[-1] + m_hi * (xq[xq > x[-1]] - x[-1])
        out[xq < x[0]] = y[0] + m_lo * (xq[xq < x[0]] - x[0])
    return out


def achx_response(freqs, points, mode="linear", extrap="clamp"):
    """|H(f)| у лінійних одиницях з профілю (дБ).
    mode:  linear — np.interp; spline — кубічний сплайн у Гц;
           logspline — кубічний сплайн у log10(f).
    extrap: clamp — значення країв; linear — продовження крайового нахилу;
            zero — 0 (−120 дБ) поза виміряним діапазоном.
    Оцінка виконується на БУДЬ-ЯКІЙ частоті (в т.ч. між точками профілю)."""
    fr = np.array([p[0] for p in points], dtype=np.float64)
    db = np.array([p[1] for p in points], dtype=np.float64)
    fq = np.atleast_1d(np.asarray(freqs, dtype=np.float64))

    if mode == "logspline":
        xs = np.log10(fr)
        xq = np.log10(np.clip(fq, 1e-6, None))
    else:
        xs, xq = fr, fq.copy()

    use_spline = mode in ("spline", "logspline") and len(fr) >= 3
    if use_spline:
        M = _spline_second_derivs(xs, db)
        hdb = _spline_eval(xs, db, M, xq,
                           extrap=("clamp" if extrap == "zero" else extrap))
    else:
        hdb = np.interp(xq, xs, db)                      # лінійна, clamp
        if extrap == "linear" and len(fr) >= 2:
            m_lo = (db[1] - db[0]) / (fr[1] - fr[0])
            m_hi = (db[-1] - db[-2]) / (fr[-1] - fr[-2])
            hdb = np.where(fq < fr[0], db[0] + m_lo * (fq - fr[0]), hdb)
            hdb = np.where(fq > fr[-1], db[-1] + m_hi * (fq - fr[-1]), hdb)
    if extrap == "zero":
        outside = (fq < fr[0]) | (fq > fr[-1])
        hdb = np.where(outside, -120.0, hdb)
    out = 10.0 ** (hdb / 20.0)
    if extrap == "zero":
        out = np.where(outside, 0.0, out)   # справжній нуль підсилення поза смугою
    return out


def apply_achx(w, fs, points, mode="linear", extrap="clamp"):
    """Застосовує АЧХ до змінної складової сигнала (FFT-фільтрація)."""
    n = w.size
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    H = achx_response(f, points, mode=mode, extrap=extrap)
    return np.fft.irfft(W * H, n)


# ----------------------------------------------------------------------------
# КАЛІБРУВАННЯ РЕАЛЬНОЇ АЧХ ТРАКТУ (ступінчастий прогін тонів)
# ----------------------------------------------------------------------------
def make_cal_plan(fs, f_min, f_max, n_tones, seg_n):
    """Частоти тонів, кожна округлена до цілих бінів FFT. → (freqs, bin_hz)."""
    bin_hz = fs / float(seg_n)
    f0s = np.linspace(f_min, f_max, n_tones)
    freqs = np.round(f0s / bin_hz) * bin_hz
    freqs = np.unique(freqs)
    freqs = np.clip(freqs, bin_hz, fs / 2.0 - bin_hz)
    return freqs.tolist(), bin_hz


def generate_cal_file(fs, nch, freqs, amp, seg_n, tract_points=None):
    """Ступінчастий прогін: сегмент seg_n відликів на тон (у ch0, DC 32768).
    tract_points — емуляція тракту (масштабує ТІЛЬКИ тон, не DC)."""
    n = seg_n * len(freqs)
    cols = [np.zeros(n) for _ in range(nch)]
    tloc = np.arange(seg_n) / fs
    for i, fq in enumerate(freqs):
        scale = 1.0 if tract_points is None else float(achx_response(np.array([fq]), tract_points)[0])
        seg = 32768.0 + amp * scale * np.sin(2.0 * np.pi * fq * tloc)
        cols[0][i * seg_n:(i + 1) * seg_n] = seg
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_cal_plan(path, fs, nch, freqs, amp, seg_n, meas_ch=0):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fs": fs, "nch": nch, "freqs": freqs, "amp": amp,
                   "seg_n": seg_n, "meas_ch": meas_ch}, fh)


def load_cal_plan(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def measure_achx(path, plan, meas_ch=None):
    """Вимірювання реальної АЧХ: амплітуда кожного тону в своєму сегменті.
    → [(freq_Hz, amp_codes, gain_dB_відносно_еталону), ...]"""
    fs = plan["fs"]
    freqs = plan["freqs"]
    amp_ref = plan["amp"]
    seg_n = plan["seg_n"]
    k = plan["meas_ch"] if meas_ch is None else meas_ch
    chans, n = load_raw(path, plan["nch"])
    need = seg_n * len(freqs)
    if n < need:
        raise ValueError("запис коротший за план: %d < %d відліків" % (n, need))
    s = chans[k]
    out = []
    for i, fq in enumerate(freqs):
        seg = s[i * seg_n:(i + 1) * seg_n]
        f, a = amplitude_spectrum(seg, fs, nfft=seg_n)
        idx = int(round(fq / (fs / seg_n)))
        amp = float(a[idx])
        out.append((float(fq), amp, 20.0 * math.log10(max(amp, 1e-9) / amp_ref)))
    return out


def save_achx_profile(path, points, meta=None, mode=None, extrap=None):
    """Профіль АЧХ у текстовий .achx: рядки «частота_Гц дБ», # — коментарі;
    режим інтерполяції/екстраполяції зберігається в шапці."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# RAW Studio — профіль АЧХ (частота_Гц дБ)\n")
        if mode:
            fh.write("# interpolation: %s\n" % mode)
        if extrap:
            fh.write("# extrapolation: %s\n" % extrap)
        for k, v in (meta or {}).items():
            fh.write("# %s: %s\n" % (k, v))
        for fq, gdb in points:
            fh.write("%.6g %.4f\n" % (fq, gdb))


def load_achx_profile(path):
    """Читає .achx → (points, mode, extrap); режими — з шапки або дефолтні."""
    mode, extrap = None, None
    pts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t.startswith("# interpolation:"):
                mode = t.split(":", 1)[1].strip()
            elif t.startswith("# extrapolation:"):
                extrap = t.split(":", 1)[1].strip()
            elif t and not t.startswith("#"):
                parts = t.replace(",", " ").replace(";", " ").split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки в файлі")
    pts.sort()
    mode = {"log": "logspline"}.get(mode, mode)   # зворотна сумісність старих .achx
    return pts, (mode or "linear"), (extrap or "clamp")


# ----------------------------------------------------------------------------
# ЯДРО (без GUI) — завантаження, статистика, спектр, генерація
# ----------------------------------------------------------------------------
def load_raw(path, nch=DEFAULT_NCH):
    data = np.fromfile(path, dtype="<u2")
    if data.size == 0:
        raise ValueError("порожній файл: %s" % path)
    n = data.size // nch
    if n * nch != data.size:
        print("  ! розмір не кратний %d каналам — останні %d байт(ів) відкинуто"
              % (nch, (data.size % nch) * 2))
    data = data[:n * nch].reshape(n, nch)
    return {k: data[:, k].astype(np.float64) for k in range(nch)}, n


def channel_stats(s, fs):
    hi = float((s >= 65000).mean() * 100.0)
    lo = float((s <= 500).mean() * 100.0)
    return {"dc": float(s.mean()), "rms": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "clip_hi": hi, "clip_lo": lo}


def amplitude_spectrum(s, fs, nfft=NFFT):
    """Пікова амплітуда в кодах АЦП: A = 2|FFT|/(N·CG), вікно Ганна, детренд."""
    s = s[:nfft]
    w = np.hanning(s.size)
    cg = w.mean()
    x = s - s.mean()
    a = np.abs(np.fft.rfft(x * w)) * 2.0 / (s.size * cg)
    f = np.fft.rfftfreq(s.size, d=1.0 / fs)
    return f, a


def find_peaks(f, a, k=5, exclude_bins=5, min_sep_bins=8):
    order = np.argsort(a)[::-1]
    taken, out = [], []
    for i in order:
        if i < exclude_bins:
            continue
        if any(abs(int(i) - t) < min_sep_bins for t in taken):
            continue
        taken.append(int(i))
        out.append((float(f[i]), float(a[i])))
        if len(out) >= k:
            break
    return out


def channel_occupancy(chans, fs, blocks, factor=6.0):
    power = None
    for k in sorted(chans):
        f, a = amplitude_spectrum(chans[k], fs)
        p = a.astype(np.float64) ** 2
        power = p if power is None else power + p
    raw = {}
    for b in blocks:
        for ch in b["channels"]:
            m = (f >= ch["f0"]) & (f < ch["f1"])
            raw[ch["global"]] = float(power[m].mean()) if m.any() else 0.0
    med = float(np.median(list(raw.values())))
    out = {}
    for g, pw in raw.items():
        out[g] = {"power": pw, "occupied": pw > factor * med and pw > 0}
    return out


def make_gnss(kind_key, n, fs, freq, amp, phase_deg, rng):
    """ШПС-моделі ГНСС верхньої L-смуги (v2.8.0/2.9.0). Несучі фіксовані стандартами
    (поле «Частота» рядка — довідкове):
    GPS L1 C/A — BPSK(1), 1.023 Мчип/с, 1575.42 МГц;
    Galileo E1 — BOC(1,1) (спрощення CBOC), 1.023 Мчип/с, 1575.42 МГц, у квадратурі до GPS;
    BeiDou B1I — BPSK(2), 2.046 Мчип/с, 1561.098 МГц;
    ГЛОНАСС G1 — FDMA, 14 літер k = −7…+6, 1602 + k·0.5625 МГц, 0.511 Мчип/с.
    Чипи псевдовипадкові (не справжні ПВП-коди). rms ≈ amp/√2 — як у синуса амплітуди amp."""
    t = np.arange(n) / fs
    ph0 = math.radians(phase_deg)
    two_pi = 2.0 * np.pi

    def chips(rate):
        m = int(math.ceil(n / fs * rate)) + 2
        seq = rng.choice(np.array([-1.0, 1.0]), size=m)
        return seq[np.floor(t * rate).astype(np.int64)]

    def bpsk(fc, rate):
        return chips(rate) * np.cos(two_pi * fc * t + ph0)

    w = None

    def add(x):
        nonlocal w
        w = x if w is None else w + x

    if kind_key in ("gnss_l1", "gps_l1"):
        add(bpsk(1575.42e6, 1.023e6))
    if kind_key in ("gnss_l1", "gal_e1"):
        sc = np.sign(np.sin(two_pi * 1.023e6 * t))
        sc[sc == 0] = 1.0
        add(chips(1.023e6) * sc * np.cos(two_pi * 1575.42e6 * t + ph0 + np.pi / 2))
    if kind_key in ("gnss_l1", "bds_b1"):
        add(bpsk(1561.098e6, 2.046e6))
    if kind_key in ("gnss_l1", "glo_g1"):
        g = None
        for k in range(-7, 7):
            x = chips(0.511e6) * np.cos(two_pi * (1602e6 + k * 0.5625e6) * t
                                        + ph0 + rng.uniform(0.0, two_pi))
            g = x if g is None else g + x
        add(g / math.sqrt(14.0))
    ncomp = {"gnss_l1": 4, "gps_l1": 1, "gal_e1": 1, "glo_g1": 1, "bds_b1": 1}[kind_key]
    return amp * w / math.sqrt(float(ncomp))


def make_mesh(n, fs, freq, amp, rng, per_us=200.0, pw_us=100.0, bw_hz=10e6, qam_bits=0.0):
    """Модель Mesh MN-MIMO (Xingkai XK-F358, v3.0.0): TD-COFDM-подібні пачки.
    N=64 піднесучих у смузі BW (T_u = 64/BW, CP = T_u/8), QAM-символи псевдовипадкові;
    пачки тривалістю pw з періодом per (TDMA); FHSS — кожна пачка на новому каналі
    fc + k·BW, k = −2…+2 (спрощення: реальний FHSS адаптивний, 1–70 Мбіт/с).
    Несуча — поле «Частота» рядка (центр діапазону). rms ≈ amp/√2."""
    per = (per_us if per_us > 0 else 200.0) * 1e-6
    pw = min((pw_us if pw_us > 0 else 100.0) * 1e-6, per * 0.95)
    bw = max(bw_hz, 1e6)
    t = np.arange(n) / fs
    two_pi = 2.0 * np.pi
    N_sub = 64
    Tu = N_sub / bw
    Tsym = Tu * 1.125                      # CP = T_u / 8
    nsym = int(math.ceil(n / fs / Tsym)) + 1
    def qam_row(bits):
        if bits == 1:                                   # BPSK: ±1
            return rng.choice(np.array([-1.0, 1.0]), size=N_sub).astype(complex)
        k = int(math.isqrt(2 ** bits))
        lv = (2 * rng.integers(0, k, size=N_sub) - (k - 1)).astype(float)
        lq = (2 * rng.integers(0, k, size=N_sub) - (k - 1)).astype(float)
        return (lv + 1j * lq) / math.sqrt(2.0 * (k * k - 1) / 3.0)
    if qam_bits > 0:
        order = np.full(nsym, float(qam_bits))
    else:
        order = rng.choice(np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0]), size=nsym)
    syms = np.stack([qam_row(int(b)) for b in order], axis=0)
    syms[:, 0] = 0.0
    ofdm = np.fft.ifft(syms, axis=1) * math.sqrt(N_sub)
    ncp = N_sub // 8                                   # = 8 відліків CP (T_u/8)
    bb = np.concatenate([ofdm[:, -ncp:], ofdm], axis=1).ravel()
    tb = np.arange(len(bb))                            # індекси відліків bb (у тих самих одиницях, що tbb)
    tbb = t * bw
    bb_i = np.interp(tbb, tb, bb.real)
    bb_q = np.interp(tbb, tb, bb.imag)
    idx_b = np.floor(t / per).astype(np.int64)
    fc_h = (np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[idx_b % 5]) * bw
    ph = two_pi * (freq + fc_h) * t
    mask = (np.mod(t, per) < pw).astype(float)
    w = (bb_i * np.cos(ph) - bb_q * np.sin(ph)) * mask
    return amp * w / (math.sqrt(2.0) * max(float(np.std(w)), 1e-12))


def make_jam(kind_key, n, fs, freq, amp, phase_deg, rng,
             bw_hz=20e6, sweep_hz=5e6, sweep_per_us=1000.0, shape=None):
    """Моделі завад (глушилок) для випробувань CRPA (TGTX5050A, v3.3.0):
    jam_cw — неперервна несуча (спот-глушилка);
    jam_noise — широкосмуговий barrage-шум: смуга bw_hz навколо несучої
      (маска в частотній області), rms = amp (як у «Шум»);
    jam_chirp — ЛЧМ-свіп freq ± sweep_hz, пила з періодом sweep_per_us
      (фаза неперервна всередині періоду).
    Частота в рядку — несуча завади. rms CW/чирп ≈ amp/√2."""
    if freq <= 0:
        return np.zeros(n)
    t = np.arange(n) / fs
    two_pi = 2.0 * np.pi
    ph0 = math.radians(phase_deg)
    if kind_key == "jam_cw":
        return amp * np.sin(two_pi * freq * t + ph0)
    if kind_key == "jam_noise":
        bw = max(float(bw_hz) if bw_hz and float(bw_hz) > 0 else 20e6, 1e5)
        w = rng.normal(0.0, 1.0, n)
        W = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        f0 = min(max(float(freq), bw / 2.0), fs / 2.0 - bw / 2.0)
        if shape is not None:
            # спектрально-формований шум (v4.9.0): профіль спектра запису
            if isinstance(shape, str):
                shape = load_spectrum_profile(shape)
            fr = np.asarray(shape.get("freqs_hz"), float)
            pd = np.asarray(shape.get("psd_db"), float)
            fcr = float(shape.get("fc") or 0.0)
            if fcr <= 0 and fr.size:
                fcr = float((fr[0] + fr[-1]) / 2.0)
            off_r = np.abs(fr - fcr)
            order = np.argsort(off_r)
            gain = 10.0 ** (pd[order] / 20.0)
            gmax = float(gain.max()) if gain.size and float(gain.max()) > 0 else 1.0
            gain = gain / gmax
            W = W * np.interp(np.abs(f - f0), off_r[order], gain, left=0.0, right=0.0)
        else:
            mask = (f >= f0 - bw / 2.0) & (f <= f0 + bw / 2.0)
            W[~mask] = 0.0
        w = np.fft.irfft(W, n)
        s = float(np.std(w))
        if s > 0:
            w = w / s
        return amp * w
    dev0 = float(sweep_hz) if sweep_hz and float(sweep_hz) > 0 else 5e6
    dev = min(max(dev0, 1e3), 0.45 * float(freq))
    T = (sweep_per_us if sweep_per_us > 0 else 1000.0) * 1e-6
    u = np.mod(t, T) / T
    inst = freq - dev + 2.0 * dev * u
    phase = two_pi * np.cumsum(inst) / fs + ph0
    return amp * np.sin(phase)


def make_rgpo(n, fs, freq, amp, phase_deg, rng,
              per_us=1000.0, pw_us=50.0, v_pull=300.0, d_max_us=1000.0):
    """RGPO — затягування по дальності (v5.0.0): радіоімпульсна репліка цілі
    (період per_us, тривалість pw_us — поля «Кроку 3»), затримка k-ї репліки
    росте лінійно: t_k = k·PRI·(1 + 2v/c) — дальномір цілі «йде» за хибною
    цілю. Після досягнення d_max репліка зникає (walk-off-then-drop).
    0 у полях = типові: PRI 1000 мкс, імпульс 50 мкс, v 300 м/с, d_max 1000 мкс."""
    if freq <= 0:
        return np.zeros(n)
    pri = float(per_us if per_us and float(per_us) > 0 else 1000.0) * 1e-6
    pw = float(pw_us if pw_us and float(pw_us) > 0 else 50.0) * 1e-6
    v = float(v_pull) if v_pull and float(v_pull) > 0 else 300.0
    d_max = float(d_max_us if d_max_us and float(d_max_us) > 0 else 1000.0) * 1e-6
    rate = 2.0 * v / 3e8
    w = np.zeros(n)
    ph0 = math.radians(phase_deg)
    k = 0
    while k < 1000000:
        tau = k * pri * (1.0 + rate)
        if k * pri * rate > d_max:
            break
        i0 = int(round(tau * fs))
        if i0 >= n:
            break
        i1 = min(i0 + max(int(round(pw * fs)), 1), n)
        tt = np.arange(i1 - i0) / fs
        w[i0:i1] += amp * np.sin(2 * np.pi * freq * (tau + tt) + ph0)
        k += 1
    return w


def make_vgpo(n, fs, freq, amp, phase_deg, rng,
              fd_max_khz=30.0, t_pull_s=2.0):
    """VGPO — затягування по швидкості (v5.0.0): репліка несучої з доплерівським
    зсувом, що росте лінійно 0 → fd_max за час t_pull (фаза = інтеграл зсуву),
    після досягнення максимуму репліка зникає (drop).
    0 у полях = типові: fd 30 кГц, t_pull 2 с."""
    if freq <= 0:
        return np.zeros(n)
    fd = float(fd_max_khz if fd_max_khz and float(fd_max_khz) > 0 else 30.0) * 1e3
    T = float(t_pull_s) if t_pull_s and float(t_pull_s) > 0 else 2.0
    t = np.arange(n) / fs
    ph = (2 * np.pi * freq * t
          + 2 * np.pi * (fd / (2.0 * T)) * np.minimum(t, T) ** 2
          + math.radians(phase_deg))
    w = amp * np.sin(ph)
    w[t > T] = 0.0
    return w


def parse_protected_mhz(s):
    """Рядок захищених частот → список діапазонів (lo, hi) у Гц (v5.0.0).
    Формат: «1575.42, 2412-2472» (МГц; голе число — діапазон ±2.5 МГц
    навколо несучої, «а-б» — діапазон)."""
    out = []
    for tok in str(s or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = parse_freq_hz(a), parse_freq_hz(b)
            if hi < lo:
                lo, hi = hi, lo
            out.append((lo, hi))
        else:
            f = parse_freq_hz(tok)
            out.append((max(f - 2.5e6, 0.0), f + 2.5e6))
    return out


def is_protected(f_hz, prot):
    """Чи частота f_hz (Гц) потрапляє у захищені діапазони (v5.0.0)."""
    return any(lo <= float(f_hz) <= hi for lo, hi in (prot or ()))


def make_spoof(n, fs, freq, amp, phase_deg, rng, f_off=500.0, f_drift=10.0):
    """Підробний GPS L1 C/A — спуфер (v3.4.0): BPSK(1) 1.023 Мчип/с, несуча =
    freq + f_off + f_drift·t — плаваюча несуча (класична ознака спуфінга:
    підробний сигнал повільно повзе відносно справжнього). Чипи псевдовипадкові
    (не справжні ПВП-коди). rms ≈ amp/√2 — як у синуса амплітуди amp."""
    fc = float(freq) if freq and float(freq) > 0 else 1575.42e6
    t = np.arange(n) / fs
    off = float(f_off) if f_off else 0.0
    drift = float(f_drift) if f_drift else 0.0
    inst = fc + off + drift * t
    ph = 2.0 * np.pi * np.cumsum(inst) / fs + math.radians(phase_deg)
    m = int(math.ceil(n / fs * 1.023e6)) + 2
    seq = rng.choice(np.array([-1.0, 1.0]), size=m)
    c = seq[np.floor(t * 1.023e6).astype(np.int64)]
    return amp * c * np.sin(ph)


def make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng):
    t = np.arange(n) / fs
    if kind_key == "noise":
        return rng.normal(0.0, amp, n)
    if kind_key in GNSS_KEYS:
        return make_gnss(kind_key, n, fs, freq, amp, phase_deg, rng)
    if kind_key == "mesh":
        return make_mesh(n, fs, freq, amp, rng)
    if kind_key == "jam_burst":
        return make_jam("jam_noise", n, fs, freq, amp, phase_deg, rng)
    if kind_key in JAM_KEYS:
        return make_jam(kind_key, n, fs, freq, amp, phase_deg, rng)
    if kind_key == "spoof":
        return make_spoof(n, fs, freq, amp, phase_deg, rng)
    if freq <= 0:
        return np.zeros(n)
    period = 1.0 / freq
    phase = (t - phase_deg / 360.0 / freq) % period
    hi_t = min(max(width_us * 1e-6, 0.0), period)
    if kind_key == "sine":
        return amp * np.sin(2.0 * np.pi * freq * t + math.radians(phase_deg))
    if kind_key == "square":
        return np.where(phase < hi_t, amp, -amp)
    if kind_key == "pulse":
        return np.where(phase < hi_t, amp, 0.0)
    if kind_key == "tri":
        x = 2.0 * np.abs(2.0 * phase / period - 1.0) - 1.0
        return amp * x
    raise ValueError(kind_key)


def make_wave_ext(kind_key, n, fs, freq, phase_deg, amp, width_us, rng,
                  per_us=0.0, pw_us=0.0, rise_us=0.0, fall_us=0.0,
                  am_m=0.0, am_f=0.0, chirp_dev=0.0, chirp_per_us=0.0, qam_bits=0.0,
                  jam_bw=0.0, sp_off=0.0, sp_drift=0.0, jam_shape=None,
                  rgpo_v=0.0, rgpo_d=0.0, vgpo_f=0.0, vgpo_t=0.0):
    """Розширені форми: радіоімпульси (період слідування, фронти), АМ, ЛЧМ.
    Амплітуда/фаза — базові; тут додаткові параметри сигналів."""
    t = np.arange(n) / fs
    width = pw_us if pw_us > 0 else width_us
    per = per_us * 1e-6 if per_us > 0 else 0.0
    use_chirp = chirp_dev > 0 and freq > 0
    if kind_key in ("tri", "noise") + GNSS_KEYS:
        w = make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng)
    elif kind_key == "mesh":
        w = make_mesh(n, fs, freq, amp, rng, per_us=per_us, pw_us=pw_us, bw_hz=chirp_dev,
                      qam_bits=qam_bits)
    elif kind_key == "jam_rgpo":
        w = make_rgpo(n, fs, freq, amp, phase_deg, rng,
                      per_us=per_us, pw_us=pw_us, v_pull=rgpo_v, d_max_us=rgpo_d)
    elif kind_key == "jam_vgpo":
        w = make_vgpo(n, fs, freq, amp, phase_deg, rng,
                      fd_max_khz=vgpo_f, t_pull_s=vgpo_t)
    elif kind_key in JAM_KEYS:
        w = make_jam(kind_key, n, fs, freq, amp, phase_deg, rng,
                     bw_hz=jam_bw, sweep_hz=chirp_dev, sweep_per_us=chirp_per_us,
                     shape=jam_shape)
    elif kind_key == "jam_burst":
        # «пакети»: barrage-шум, обмежений огинаючою (період/тривалість) — v6.1.0
        w = make_jam("jam_noise", n, fs, freq, amp, phase_deg, rng,
                     bw_hz=jam_bw if jam_bw and jam_bw > 0 else 5e6,
                     sweep_hz=chirp_dev, sweep_per_us=chirp_per_us, shape=jam_shape)
        if per > 0 and pw_us > 0:
            w = w * (np.mod(t, per) < (pw_us * 1e-6)).astype(float)
    elif kind_key == "spoof":
        w = make_spoof(n, fs, freq, amp, phase_deg, rng,
                       f_off=sp_off, f_drift=sp_drift)
    elif kind_key == "pulse":
        # радіоімпульс: несуча (Частота) × огинаюча з фронтами зростання/спаду
        hi = width * 1e-6
        if freq <= 0:
            w = np.zeros(n)
        else:
            pp = per if per > 0 else max(1.0 / freq, hi * 1.25)
            tloc = np.mod(t, pp)
            ph = 2 * np.pi * freq * t + math.radians(phase_deg)
            if use_chirp:
                if chirp_per_us > 0:
                    # чірипи LoRa всередині імпульсу: символ T_sym по абсолютному часу
                    Tc = chirp_per_us * 1e-6
                    kc = np.floor(t / Tc)
                    uc = t - kc * Tc
                    ph = 2 * np.pi * ((freq - chirp_dev / 2) * t
                                      + chirp_dev * (kc * Tc / 2 + uc * uc / (2 * Tc))) \
                         + math.radians(phase_deg)
                else:
                    kk = chirp_dev / max(hi, 1e-12)
                    ph = 2 * np.pi * (freq * t + 0.5 * kk * np.minimum(tloc, hi) ** 2) \
                         + math.radians(phase_deg)
            env = np.zeros(n)
            on = tloc < hi
            env[on] = 1.0
            rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
            fall = min(max(fall_us * 1e-6, 0.0), hi - rise)
            if rise > 0:
                r = tloc < rise
                env[r] = tloc[r] / rise
            if fall > 0:
                fl = (tloc >= hi - fall) & on
                env[fl] = np.maximum((hi - tloc[fl]) / fall, 0.0)
            w = amp * np.sin(ph) * env
    elif kind_key == "square":
        # меандр: трапеція з фронтами зростання/спаду
        hi = width * 1e-6
        pp = per if per > 0 else 2 * hi
        tloc = np.mod(t, pp)
        w = np.where(tloc < hi, float(amp), -float(amp))
        rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
        fall = min(max(fall_us * 1e-6, 0.0), hi / 2)
        if rise > 0:
            r = tloc < rise
            w[r] = -amp + 2 * amp * tloc[r] / rise
        if fall > 0:
            fl = (tloc >= hi - fall) & (tloc < hi)
            w[fl] = amp - 2 * amp * (tloc[fl] - (hi - fall)) / fall
    elif kind_key == "sine":
        if use_chirp:
            if chirp_per_us > 0:
                # CSS up-chirp LoRa: f(t) = f0 - dev/2 + dev·u/T, u ∈ [0, T);
                # фаза неперервна на межах символів; T = T_sym = 2^SF/BW
                Tc = chirp_per_us * 1e-6
                kc = np.floor(t / Tc)
                uc = t - kc * Tc
                ph = 2 * np.pi * ((freq - chirp_dev / 2) * t
                                  + chirp_dev * (kc * Tc / 2 + uc * uc / (2 * Tc))) \
                     + math.radians(phase_deg)
            else:
                kk = chirp_dev / (n / fs)
                ph = 2 * np.pi * (freq * t + 0.5 * kk * t ** 2) + math.radians(phase_deg)
            w = amp * np.sin(ph)
        else:
            w = amp * np.sin(2 * np.pi * freq * t + math.radians(phase_deg))
    else:
        raise ValueError(kind_key)
    if am_m > 0 and am_f > 0:
        w = w * (1.0 + am_m * np.sin(2 * np.pi * am_f * t))
    return w


def generate_raw(ch_params, n_samples, fs, nch, seed=42, achx=None, extra=None):
    """Створює файл у форматі вхідного.
    achx: None або {"points": [(Гц, дБ)...], "mode": linear|spline|log,
    "extrap": clamp|linear|zero} — застосовується до змінної складової
    КОЖНОГО увімкненого канала на БУДЬ-ЯКІЙ частоті (інтерполяція)."""
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(nch):
        p = ch_params[k]
        n = n_samples
        if not p.get("enabled"):
            cols.append(np.full(n, p.get("dc", 32768.0)))
            continue
        ex = p.get("extra") if p.get("extra") is not None else (extra or {})
        w = p.get("samples")
        if w is None:
            if p.get("kind") in ("Mesh COFDM", "Завада RGPO", "Завада VGPO") or (
                    ex and (ex.get("jam_shape") or any(float(ex.get(k) or 0) != 0 for k in
                               ("per_us", "pw_us", "rise_us", "fall_us", "am_m", "am_f",
                                "chirp_dev", "chirp_per_us", "qam_bits", "jam_bw",
                                "sp_off", "sp_drift", "rgpo_v", "rgpo_d",
                                "vgpo_f", "vgpo_t")))):
                w = make_wave_ext(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                                  p["amp"], p["width_us"], rng,
                                  per_us=ex.get("per_us", 0), pw_us=ex.get("pw_us", 0),
                                  rise_us=ex.get("rise_us", 0), fall_us=ex.get("fall_us", 0),
                                  am_m=ex.get("am_m", 0), am_f=ex.get("am_f", 0),
                                  chirp_dev=ex.get("chirp_dev", 0),
                                  chirp_per_us=ex.get("chirp_per_us", 0),
                                  qam_bits=ex.get("qam_bits", 0),
                                  jam_bw=ex.get("jam_bw", 0),
                                  sp_off=ex.get("sp_off", 0),
                                  sp_drift=ex.get("sp_drift", 0),
                                  jam_shape=ex.get("jam_shape"),
                                  rgpo_v=ex.get("rgpo_v", 0),
                                  rgpo_d=ex.get("rgpo_d", 0),
                                  vgpo_f=ex.get("vgpo_f", 0),
                                  vgpo_t=ex.get("vgpo_t", 0))
            else:
                w = make_wave(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                              p["amp"], p["width_us"], rng)
        if achx and achx.get("points"):
            w = apply_achx(w, fs, achx["points"],
                           mode=achx.get("mode", "linear"),
                           extrap=achx.get("extrap", "clamp"))
        s = p.get("dc", 32768.0) + p.get("gain", 1.0) * w
        cols.append(s)
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_raw(path, flat_u16):
    flat_u16.tofile(path)


def raw_spectrogram(path, fs_hz, nch=1, ch=0, nfft=1024, overlap=0.5, max_cols=800):
    """Спектрограма каналу з .raw (u2, канали черезвідліково) — v3.17.0.

    Повертає (freqs_hz, times_ms, db, note): db — (nfft//2+1, nframes)
    у дБ відн. максимуму; times_ms — центри вікон; note — опис параметрів.
    """
    if nfft < 64 or nfft > 65536 or nfft & (nfft - 1):
        raise ValueError("NFFT має бути степенем двійки 64…65536")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("перекриття має бути 0…0.95")
    if not (0 <= ch < nch):
        raise ValueError("канал № %d поза діапазоном 0…%d" % (ch, nch - 1))
    data = np.fromfile(path, dtype="<u2")
    if data.size == 0:
        raise ValueError("порожній файл: %s" % path)
    n = data.size // nch
    if n * nch != data.size:
        raise ValueError("розмір не кратний числу каналів: %d відліків на %d каналів"
                         % (data.size, nch))
    col = data[:n * nch].reshape(n, nch).astype(np.float64)[:, ch]
    if col.size < nfft:
        raise ValueError("запис коротший за вікно NFFT: %d відліків" % col.size)
    col = col - 32768.0
    hop = max(1, int(round(nfft * (1.0 - overlap))))
    nfr = 1 + (col.size - nfft) // hop
    idx = np.linspace(0, nfr - 1, max_cols).astype(int) if nfr > max_cols else np.arange(nfr)
    win = np.hanning(nfft)
    fscale = np.fft.rfftfreq(nfft, 1.0 / float(fs_hz))
    cols = []
    for j in idx:
        seg = col[j * hop: j * hop + nfft] * win
        cols.append(np.abs(np.fft.rfft(seg)))
    mag = np.stack(cols, axis=1)
    db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    times_ms = (idx * hop + nfft / 2.0) / float(fs_hz) * 1e3
    note = ("%d відл., %d вікон по %d (перекриття %d%%)"
            % (col.size, len(idx), nfft, int(round(overlap * 100))))
    return fscale, times_ms, db, note




RTSA_IQ_DTYPES = {"float32": "<f4", "float64": "<f8", "int8": "i1",
                  "int16": "<i2", "int32": "<i4"}
RTSA_IQ_NORM = {"float32": 1.0, "float64": 1.0, "int8": 128.0,
                "int16": 32768.0, "int32": 2147483648.0}


SIG_PRESETS = {
    # Типові параметри стандартів — з відкритих специфікацій:
    # GSM: ETSI GSM 05.01 / sharetechnote (кадр 4.615 мс, слот 577 мкс);
    # GPS L1: gps.gov SPS Signal Spec (1575.42 МГц, BPSK 1.023 Мсимв/с,
    #         2.046 МГц між нулями);
    # GNSS L-смуга (частотний план RNSS): GPS L1/L2/L5, ГЛОНАСС G1/G2/G3,
    #         Galileo E1/E5a/E5b/E6, BeiDou B1/B3, SAR 1544.5;
    # LoRa CSS: розгортка = BW (125/250/500 кГц), T_sym = 2^SF/BW
    #         (SF7/BW125 → 1.024 мс); WCDMA: 3.84 Мсимв/с, 5 МГц;
    # LTE: OFDMA до 20 МГц; NR FR1: до 100 МГц; NB-IoT: 180 кГц;
    # DVB-T/T2: 8 МГц; Bluetooth FHSS: 79 каналів × 1 МГц; CVBS: 64 мкс/4.7 мкс.
    "LoRa": {"kind": "Синус", "freq": 868e6, "width_us": None,
             "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 2000e6,
             "label": "868 МГц, SF7 / BW 125 кГц (T_sym 1.024 мс)",
             "note": "LoRa (CSS): BW 125 кГц = розгортка ЛЧМ («девіація»); SF7 → T_sym = 2^SF/BW = 1.024 мс («період ЛЧМ»); SF8…SF12 → 2.048/4.096/8.192/16.384/32.768 мс; BW 250/500 кГц → девіація 0.25/0.5 МГц, T_sym менша у 2/4 рази"},
    "LoRa VHF (150–300)": {"kind": "Синус", "freq": 169e6, "width_us": None,
                           "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 500e6,
                           "label": "169 МГц (150–300), SF7 / BW 125 кГц",
                           "note": "LoRa VHF: діапазон 150–300 МГц (реальний SRD-діапазон LoRa — 169 МГц); несучу можна змінити на будь-яку з 150–300 у полі «Частота»; модель — up-chirp SF7 / BW 125 кГц (T_sym 1.024 мс)"},
    "LoRa 433 МГц": {"kind": "Синус", "freq": 433.92e6, "width_us": None,
                     "extra": {"chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 1000e6,
                     "label": "433.92 МГц, SF7 / BW 125 кГц",
                     "note": "LoRa 433: ISM 433.05–434.79 МГц (LoRaWAN EU433: 433.175–434.665 МГц); модель — up-chirp SF7 / BW 125 кГц (T_sym 1.024 мс)"},
    "LoRaWAN": {"kind": "Імпульс", "freq": 868.3e6, "width_us": None,
                "extra": {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
                          "chirp_dev": 0.125e6, "chirp_per_us": 1024}, "fs_hint": 2000e6,
                "label": "пакети 1 мс / 10 мс, SF7 / BW 125 кГц",
                "note": "LoRaWAN: LoRa-фірика пакетами; модель — радіоімпульси 1 мс з періодом 10 мс, усередині — чірипи SF7/BW125 (T_sym 1.024 мс)"},
    "2G (GSM)": {"kind": "Імпульс", "freq": 900e6, "width_us": None,
                 "extra": {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
                 "fs_hint": 2000e6,
                 "label": "слот 577 мкс / кадр 4.615 мс",
                 "note": "2G (GSM): TDMA GMSK, кадр 4.615 мс = 8 слотів по 577 мкс; модель — радіоімпульси з фронтами 3 мкс"},
    "3G (WCDMA)": {"kind": "Шум", "freq": 2140e6, "width_us": None,
                   "extra": {}, "fs_hint": 5000e6,
                   "label": "канал 5 МГц, QPSK",
                   "note": "3G (WCDMA): QPSK 3.84 Мсимв/с, канал 5 МГц; модель — широкосмуговий шум (2140 МГц)"},
    "4G (LTE)": {"kind": "Шум", "freq": 2600e6, "width_us": None,
                 "extra": {}, "fs_hint": 6000e6,
                 "label": "OFDMA до 20 МГц",
                 "note": "4G (LTE): OFDMA, канал до 20 МГц; модель — широкосмуговий шум (2600 МГц)"},
    "5G (NR)": {"kind": "Шум", "freq": 3500e6, "width_us": None,
                "extra": {}, "fs_hint": 8000e6,
                "label": "n78 (3.3–3.8 ГГц), до 100 МГц",
                "note": "5G (NR): OFDM FR1, діапазон n78, канал до 100 МГц; модель — широкосмуговий шум (3500 МГц)"},
    "WiFi (OFDM)": {"kind": "Шум", "freq": 2437e6, "width_us": None,
                    "extra": {"am_m": 0.9, "am_f": 1000}, "fs_hint": 6000e6,
                    "label": "20/40/80 МГц, пакети",
                    "note": "WiFi: OFDM 20/40/80 МГц, пакетна передача; модель — шум з АМ-огинаючою пакетів (1 кГц)"},
    "GNSS (L-смуга)": {"kind": "Шум", "freq": 1580e6, "width_us": None,
                       "extra": {}, "fs_hint": 4000e6,
                       "label": "1559–1610 МГц",
                       "note": "GNSS: спредспектр BPSK у смузі 1559–1610 МГц; модель — широкосмуговий шум"},
    "GPS L1": {"kind": "Шум", "freq": 1575.42e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L1 C/A, 2.046 МГц",
               "note": "GPS L1 C/A: BPSK 1.023 Мсимв/с (2.046 МГц між нулями), 1575.42 МГц; модель — шум"},
    "GPS L2": {"kind": "Шум", "freq": 1227.6e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L2, 1227.60 МГц",
               "note": "GPS L2: смуга RNSS 1215–1240 МГц, несуча 1227.60 МГц; модель — шум"},
    "GPS L5": {"kind": "Шум", "freq": 1176.45e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L5, 1176.45 МГц (з E5a)",
               "note": "GPS L5: ARNS/RNSS 1164–1214 МГц, несуча 1176.45 МГц (спільна з Galileo E5a); модель — шум"},
    "ГЛОНАСС G1": {"kind": "Шум", "freq": 1602e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G1, 1602 МГц (FDMA)",
                   "note": "ГЛОНАСС G1: 1593–1610 МГц, центр 1602 МГц + k·0.5625 МГц (FDMA); модель — шум"},
    "ГЛОНАСС G2": {"kind": "Шум", "freq": 1246e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G2, 1246 МГц (FDMA)",
                   "note": "ГЛОНАСС G2: 1237–1254 МГц, центр 1246 МГц + k·0.4375 МГц (FDMA); модель — шум"},
    "ГЛОНАСС G3": {"kind": "Шум", "freq": 1202.025e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "G3 CDMA, 1202.025 МГц",
                   "note": "ГЛОНАСС G3 (CDMA): 1195–1210 МГц, несуча 1202.025 МГц; модель — шум"},
    "Galileo E1": {"kind": "Шум", "freq": 1575.42e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "E1, 1575.42 МГц",
                   "note": "Galileo E1: верхня L-смуга 1559–1591 МГц, несуча 1575.42 МГц (спільна з GPS L1); модель — шум"},
    "Galileo E5a": {"kind": "Шум", "freq": 1176.45e6, "width_us": None,
                    "extra": {}, "fs_hint": 4000e6,
                    "label": "E5a, 1176.45 МГц",
                    "note": "Galileo E5a: 1164–1214 МГц, несуча 1176.45 МГц (спільна з GPS L5); модель — шум"},
    "Galileo E5b": {"kind": "Шум", "freq": 1207.14e6, "width_us": None,
                    "extra": {}, "fs_hint": 4000e6,
                    "label": "E5b, 1207.14 МГц",
                    "note": "Galileo E5b: 1189–1214 МГц, несуча 1207.14 МГц (спільна з BeiDou B2b); модель — шум"},
    "Galileo E6": {"kind": "Шум", "freq": 1278.75e6, "width_us": None,
                   "extra": {}, "fs_hint": 4000e6,
                   "label": "E6, 1278.75 МГц",
                   "note": "Galileo E6: 1260–1300 МГц, несуча 1278.75 МГц; модель — шум"},
    "BeiDou B1": {"kind": "Шум", "freq": 1561.098e6, "width_us": None,
                  "extra": {}, "fs_hint": 4000e6,
                  "label": "B1, 1561.098 МГц",
                  "note": "BeiDou B1: 1559–1591 МГц, B1I 1561.098 МГц (B1C — на 1575.42); модель — шум"},
    "BeiDou B3": {"kind": "Шум", "freq": 1268.52e6, "width_us": None,
                  "extra": {}, "fs_hint": 4000e6,
                  "label": "B3, 1268.52 МГц",
                  "note": "BeiDou B3: 1254–1280 МГц, несуча 1268.52 МГц; модель — шум"},
    "Комплекс GNSS L1 (E1+G1+B1+L1)": {
        "kind": "GNSS L1", "freq": 1575.42e6, "width_us": None,
        "extra": {}, "fs_hint": 4000e6,
        "label": "Galileo E1 + ГЛОНАСС G1 + BeiDou B1 + GPS L1 в одному каналі",
        "note": "Комплекс верхньої L-смуги (1559–1610 МГц) в ОДНОМУ каналі: GPS L1 C/A — BPSK(1) 1.023 Мчип/с на 1575.42 МГц; "
                "Galileo E1 OS — BOC(1,1) 1.023 Мчип/с на 1575.42 МГц (у квадратурі до GPS); BeiDou B1I — BPSK(2) "
                "2.046 Мчип/с на 1561.098 МГц; ГЛОНАСС G1 — FDMA, 14 літер k=−7…+6, 1602 + k·0.5625 МГц, 0.511 Мчип/с. "
                "Чипи псевдовипадкові (не справжні ПВП-коди); частота в рядку довідкова — несучі систем фіксовані"},
    "GPS L1 C/A (ШПС)": {"kind": "GPS L1", "freq": 1575.42e6, "width_us": None,
                         "extra": {}, "fs_hint": 4000e6,
                         "label": "BPSK(1), 1.023 Мчип/с, 1575.42 МГц",
                         "note": "GPS L1 C/A — ШПС-модель: BPSK(1) 1.023 Мчип/с, несуча 1575.42 МГц (поле «Частота» — довідкове); чипи псевдовипадкові"},
    "Galileo E1 BOC (ШПС)": {"kind": "Galileo E1", "freq": 1575.42e6, "width_us": None,
                             "extra": {}, "fs_hint": 4000e6,
                             "label": "BOC(1,1), 1.023 Мчип/с, 1575.42 МГц",
                             "note": "Galileo E1 — ШПС-модель: BOC(1,1) 1.023 Мчип/с на 1575.42 МГц (спектр роздвоєний, у квадратурі до GPS); спрощення CBOC; чипи псевдовипадкові"},
    "ГЛОНАСС G1 FDMA (ШПС)": {"kind": "ГЛОНАСС G1", "freq": 1602e6, "width_us": None,
                              "extra": {}, "fs_hint": 4000e6,
                              "label": "FDMA, 14 літер, 0.511 Мчип/с",
                              "note": "ГЛОНАСС G1 — ШПС-модель: FDMA, 14 літер k = −7…+6, 1602 + k·0.5625 МГц, 0.511 Мчип/с; чипи псевдовипадкові"},
    "BeiDou B1I (ШПС)": {"kind": "BeiDou B1", "freq": 1561.098e6, "width_us": None,
                         "extra": {}, "fs_hint": 4000e6,
                         "label": "BPSK(2), 2.046 Мчип/с, 1561.098 МГц",
                         "note": "BeiDou B1I — ШПС-модель: BPSK(2) 2.046 Мчип/с, несуча 1561.098 МГц; чипи псевдовипадкові"},
    "TGTX5050A RF1 (GNSS L1)": {"kind": "GNSS L1", "freq": 1575.42e6, "width_us": None,
        "extra": {}, "fs_hint": 4000e6,
        "label": "RF1: прийом L1 (GPS/ГЛОНАСС/Galileo/BeiDou), RHCP",
        "note": "TGTX5050A — бортова завадозахищена ГНСС-антенна, 16-елементна CRPA (діапазон L1 1559–1610 МГц): "
                "вихід RF1 — сформований канал решітки. Прийом: GPS L1 C/A — BPSK(1) 1.023 Мчип/с на 1575.42 МГц; "
                "ГЛОНАСС G1 — FDMA 1598.06–1609.38 МГц; Galileo E1 — BOC(1,1) на 1575.42 МГц; BeiDou B1I — BPSK(2) на 1561.098 МГц. "
                "Поляризація кругова права (RHCP) — властивість антени, у записі не кодується. Антена тільки приймає: "
                "генератор імітує запис сигналу на її виході. Адаптивні нулі проти глушилок — властивість апаратури; "
                "у цьому файлі завада відсутня (придушена нулями, заявлено до ~110 дБ)"},
    "TGTX5050A RF2 (GNSS L1)": {"kind": "GNSS L1", "freq": 1575.42e6, "width_us": None,
        "extra": {}, "fs_hint": 4000e6,
        "label": "RF2: другий сформований канал (резервування)",
        "note": "TGTX5050A CRPA: вихід RF2 — другий сформований канал решітки (резервування або другий приймальний тракт); "
                "те саме сигнальне оточення L1, незалежна реалізація (свій seed рядка). "
                "Група «TGTX5050A CRPA (RF1+RF2)» ставить обидва виходи в два канали запису"},
    "Завада CW 1575.42 (глушилка)": {"kind": "Завада CW", "freq": 1575.42e6, "width_us": None,
        "extra": {}, "fs_hint": 4000e6,
        "label": "спот-завада, несуча 1575.42 МГц",
        "note": "Завада CW: неперервна несуча — класична спот-глушилка ГНСС; для випробування адаптивних нулів CRPA "
                "(TGTX5050A). Частоту змініть у полі «Частота» (напр., 1602 — ГЛОНАСС G1); rms ≈ ампл/√2"},
    "Завада широкосмугова (шум 20 МГц)": {"kind": "Завада (шум)", "freq": 1575.42e6, "width_us": None,
        "extra": {"jam_bw": 20e6}, "fs_hint": 4000e6,
        "label": "barrage-шум, 20 МГц на 1575.42 МГц",
        "note": "Завада — широкосмуговий barrage-шум: смуга 20 МГц навколо несучої («ширина смуги завади, МГц» "
                "у Кроці 3 або вікні ⚙; 0 = 20 МГц за замовчуванням). Придушується нулями CRPA гірше за CW — "
                "типовий тест завадозахисту"},
    "Завада чирп (±5 МГц)": {"kind": "Завада (чирп)", "freq": 1575.42e6, "width_us": None,
        "extra": {"chirp_dev": 5e6, "chirp_per_us": 1000}, "fs_hint": 4000e6,
        "label": "ЛЧМ-свіп ±5 МГц, період пили 1000 мкс",
        "note": "Завада — ЛЧМ-свіп: девіація ±5 МГц («девіація ЛЧМ, МГц»), період пили 1000 мкс («період ЛЧМ, мкс»); "
                "swept-глушилка перевіряє стійкість нулів по смузі"},
    "Spoofing GPS — зміщення +500 Гц": {"kind": "Spoofing GPS", "freq": 1575.42e6, "width_us": None,
        "extra": {"sp_off": 500.0, "sp_drift": 0.0}, "fs_hint": 4000e6,
        "label": "BPSK(1) 1.023 Мчип/с, 1575.42 МГц + 500 Гц",
        "note": "Спуфер — підробний GPS L1 C/A: та сама форма BPSK(1) 1.023 Мчип/с, несуча = «Частота» + "
                "зміщення («зміщення несучої спуфера, Гц» у «Кроці 3» або вікні ⚙). Класичний сценарій "
                "перевірок CRPA (TGTX5050A): реальний GPS у каналі 1 + спуфер у каналі 2 — група "
                "«Спуфер + реальний GPS»; чипи псевдовипадкові (не справжні ПВП-коди)"},
    "Spoofing GPS — плаваюча (+10 Гц/с)": {"kind": "Spoofing GPS", "freq": 1575.42e6, "width_us": None,
        "extra": {"sp_off": 500.0, "sp_drift": 10.0}, "fs_hint": 4000e6,
        "label": "BPSK(1), зміщення +500 Гц, дрейф +10 Гц/с",
        "note": "Спуфер із плаваючою несучою: зміщення +500 Гц і дрейф +10 Гц/с («дрейф зміщення, Гц/с») — "
                "несуча повільно повзе відносно справжньої, витягуючи стежки приймача (класична ознака "
                "спуфінга); обидва поля 0 → 500 Гц + 10 Гц/с за замовчуванням"},
    "Mesh XK-F358 UHF (430–550)": {"kind": "Mesh COFDM", "freq": 490e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 2000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 490 МГц",
        "note": "Mesh MN-MIMO (Xingkai XK-F358), діапазон UHF 430–550 МГц (опційно 225–400/320–470 МГц): модель — TD-COFDM-пачки (64 піднесучих, CP 1/8, QAM псевдовипадкові), TDMA 100 мкс / 200 мкс («тривалість імпульсу»/«період слідування»), FHSS 5 каналів × BW («девіація ЛЧМ, МГц» = ширина каналу); 2T2R — група «Mesh XK-F358 2T2R»"},
    "Mesh XK-F358 L (1000–1500)": {"kind": "Mesh COFDM", "freq": 1250e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 4000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 1250 МГц",
        "note": "Mesh MN-MIMO (XK-F358), діапазон L 1000–1200 / 1300–1500 МГц (опційно 1200–1700): модель — TD-COFDM-пачки + FHSS, як в UHF-пресеті"},
    "Mesh XK-F358 S (1600–2900)": {"kind": "Mesh COFDM", "freq": 2250e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 6000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 2250 МГц",
        "note": "Mesh MN-MIMO (XK-F358), діапазон S 1.6–2.9 ГГц (піддіапазони 1.6–1.8/1.8–2.0/2.0–2.2/2.2–2.5/2.5–2.7/2.7–2.9 ГГц): модель — TD-COFDM-пачки + FHSS; група «Mesh XK-F358 2T2R» — TX1+TX2 у двох каналах"},
    "Mesh XK-F358 S2 (3200–3400)": {"kind": "Mesh COFDM", "freq": 3300e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 8000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 3300 МГц",
        "note": "Mesh MN-MIMO (XK-F358), діапазон S 3.2–3.4 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц; fs 8000 МГц (Найквіст для 3320 МГц)"},
    "Mesh XK-F358 S (1600–1800)": {"kind": "Mesh COFDM", "freq": 1700e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 4000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 1700 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 1.6–1.8 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц; несуча — центр піддіапазону"},
    "Mesh XK-F358 S (1800–2000)": {"kind": "Mesh COFDM", "freq": 1900e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 4000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 1900 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 1.8–2.0 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц"},
    "Mesh XK-F358 S (2000–2200)": {"kind": "Mesh COFDM", "freq": 2100e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 5000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 2100 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 2.0–2.2 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц"},
    "Mesh XK-F358 S (2200–2500)": {"kind": "Mesh COFDM", "freq": 2350e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 5000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 2350 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 2.2–2.5 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц"},
    "Mesh XK-F358 S (2500–2700)": {"kind": "Mesh COFDM", "freq": 2600e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 6000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 2600 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 2.5–2.7 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц"},
    "Mesh XK-F358 S (2700–2900)": {"kind": "Mesh COFDM", "freq": 2800e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 6000e6,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 2800 МГц",
        "note": "Mesh MN-MIMO (XK-F358), піддіапазон S 2.7–2.9 ГГц: модель — TD-COFDM-пачки + FHSS, BW 10 МГц"},
    "Mesh XK-F358 C (4400–5850)": {"kind": "Mesh COFDM", "freq": 5125e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 10e6}, "fs_hint": 12e9,
        "label": "TD-COFDM, BW 10 МГц, FHSS, центр 5125 МГц",
        "note": "Mesh MN-MIMO (XK-F358), діапазон C 4.4–5.0 / 5.25–5.85 ГГц (опційно 4.2–5.2/5.5–6.0): модель — TD-COFDM-пачки + FHSS; fs 12 ГГц"},
    "Mesh XK-F358 MIIT (336–678)": {"kind": "Mesh COFDM", "freq": 507e6, "width_us": None,
        "extra": {"per_us": 200, "pw_us": 100, "chirp_dev": 5e6}, "fs_hint": 2000e6,
        "label": "TD-COFDM, BW 5 МГц, FHSS, центр 507 МГц",
        "note": "Mesh MN-MIMO (XK-F358), дозволені MIIT 336–344/512–582/566–626/606–678 МГц (кластер 1) та 1420–1520/1430–1444 МГц (кластер 2 — змініть несучу на 1470): модель — TD-COFDM-пачки, BW 5 МГц + FHSS"},
    "SAR": {"kind": "Синус", "freq": 1544.5e6, "width_us": None,
            "extra": {}, "fs_hint": 4000e6,
            "label": "SAR, 1544.5 МГц (1544–1545)",
            "note": "SAR (пошук і порятунок): ретрансляція маяків 406 МГц, низхідна лінія 1544–1545 МГц; модель — вузькосмугова несуча 1544.5 МГц"},
    "UNB / Narrowband": {"kind": "Синус", "freq": 868.13e6, "width_us": None,
                         "extra": {}, "fs_hint": 2000e6,
                         "label": "100/600 Гц (Sigfox)",
                         "note": "UNB (Sigfox): DBPSK, смуга 100/600 Гц; модель — вузькосмугова несуча (868.13 МГц)"},
    "FSK / GFSK": {"kind": "Синус", "freq": 2440e6, "width_us": None,
                   "extra": {"chirp_dev": 0.5e6}, "fs_hint": 6000e6,
                   "label": "2.4 ГГц, dev 0.5 МГц",
                   "note": "FSK/GFSK (Bluetooth): частотна манипуляція dev ±160…500 кГц; модель — несуча з частотною модуляцією (ЛЧМ 0.5 МГц)"},
    "FHSS": {"kind": "Шум", "freq": 2441.75e6, "width_us": None,
             "extra": {}, "fs_hint": 6000e6,
             "label": "79×1 МГц, 1600 стрибків/с",
             "note": "FHSS (Bluetooth): 79 каналів × 1 МГц, 1600 стрибків/с; модель — шум у смузі 2.4 ГГц (усереднено)"},
    "QAM": {"kind": "Шум", "freq": 500e6, "width_us": None,
            "extra": {}, "fs_hint": 2000e6,
            "label": "64/256-QAM",
            "note": "QAM (64/256): амплітудно-фазова модуляція, високий PAPR; модель — шумове заповнення канала (500 МГц)"},
    "CVBS": {"kind": "Меандр", "freq": 0, "width_us": 4.7,
             "extra": {"per_us": 64, "rise_us": 0.2, "fall_us": 0.2}, "fs_hint": 100e6,
             "label": "рядок 64 мкс, sync 4.7 мкс",
             "note": "CVBS: аналогове відео, рядок 64 мкс (15.625 кГц), синхроімпульс 4.7 мкс; модель — трапеція в базовій смузі (несуча 0)"},
    "COFDM (DVB)": {"kind": "Шум", "freq": 474e6, "width_us": None,
                    "extra": {}, "fs_hint": 2000e6,
                    "label": "DVB-T/T2, 8 МГц",
                    "note": "COFDM (DVB-T/T2): канал 8 МГц; модель — широкосмуговий шум (UHF 474 МГц)"},
    "NB-IoT": {"kind": "Синус", "freq": 869.5e6, "width_us": None,
               "extra": {}, "fs_hint": 2000e6,
               "label": "канал 180 кГц",
               "note": "NB-IoT: SC-FDMA, канал 180 кГц, напівдуплекс; модель — вузькосмугова несуча (869.5 МГц)"},
    "Завада блок 450–490 (LoRa)": {"kind": "Завада (шум)", "freq": 470.0e6, "width_us": None,
        "extra": {"jam_bw": 40e6}, "fs_hint": 56e6,
        "label": "шум 40 МГц на 470 МГц — блок LoRa 450–490",
        "note": "Загороджувальний шум на суб-ГГц блок LoRa (канали 125–500 кГц). BladeRF 2.0 (fs 56 МГц) "
                "покриває блок цілком; точніше під CSS — чирп-завада з девіацією ±20 МГц на 470 МГц"},
    "Завада блок 1900 (LTE)": {"kind": "Завада (шум)", "freq": 1900.0e6, "width_us": None,
        "extra": {"jam_bw": 20e6}, "fs_hint": 56e6,
        "label": "шум 20 МГц на 1900 МГц — типовий носій LTE",
        "note": "Типовий носій LTE 10–20 МГц (OFDM): barrage-шум на смугу канала. Для Band 2 (DL 1930–1990) "
                "змініть частоту на 1960 МГц; J/S для OFDM ~10 дБ"},
    "Завада блок 2200–2360 (Video)": {"kind": "Завада (шум)", "freq": 2280.0e6, "width_us": None,
        "extra": {"jam_bw": 40e6}, "fs_hint": 56e6,
        "label": "шум 40 МГц на 2280 МГц — під-блок відеолінків",
        "note": "Блок 160 МГц ширший за миттєву смугу BladeRF (fs 56 → ~22 МГц корисної смуги за раз). "
                "Один канал — під-блок 40 МГц; повне покриття — 4 під-блоки (2200/2240/2280/2320 МГц)"},
    "Завада блок 3300–3800 (Video)": {"kind": "Завада (шум)", "freq": 3550.0e6, "width_us": None,
        "extra": {"jam_bw": 40e6}, "fs_hint": 56e6,
        "label": "шум 40 МГц на 3550 МГц — під-блок відеолінків",
        "note": "Блок 500 МГц = 13 під-блоків по 40 МГц: сканування під-блоків у часі або кілька радіо. "
                "Частотний діапазон BladeRF (70 МГц–6 ГГц) блок покриває повністю"},
    "Завада блок 2020–2100 (Телеметрія, ниж.)": {"kind": "Завада (шум)", "freq": 2040.0e6, "width_us": None,
        "extra": {"jam_bw": 40e6}, "fs_hint": 56e6,
        "label": "шум 40 МГц на 2040 МГц — нижня половина блока ТЛМ",
        "note": "Половина блока радіотелеметрії 2020–2060 МГц; разом з верхнім каналом (2080 МГц) "
                "покриває блок 80 МГц двома TX-каналами BladeRF 2.0"},
    "Завада блок 2020–2100 (Телеметрія, верх.)": {"kind": "Завада (шум)", "freq": 2080.0e6, "width_us": None,
        "extra": {"jam_bw": 40e6}, "fs_hint": 56e6,
        "label": "шум 40 МГц на 2080 МГц — верхня половина блока ТЛМ",
        "note": "Половина блока радіотелеметрії 2060–2100 МГц; другий TX-канал групи «ТЛМ 2020–2100»"},
}

SIG_GROUPS = {
    # Група = по одному сигналу в кожен УВІМКНЕНИЙ рядок таблиці Генератора.
    # Частоти — з частотного плану RNSS L-смуги (GPS/GLONASS/Galileo/BeiDou/SAR).
    "Група: GNSS — усі системи (8 каналів)": [
        "GPS L1", "GPS L2", "GPS L5", "ГЛОНАСС G1", "ГЛОНАСС G2", "ГЛОНАСС G3",
        "Galileo E5b", "Galileo E6"],
    "Група: GPS (L1/L2/L5)": ["GPS L1", "GPS L2", "GPS L5"],
    "Група: ГЛОНАСС (G1/G2/G3)": ["ГЛОНАСС G1", "ГЛОНАСС G2", "ГЛОНАСС G3"],
    "Група: Galileo (E1/E5a/E5b/E6)": ["Galileo E1", "Galileo E5a", "Galileo E5b", "Galileo E6"],
    "Група: BeiDou (B1/B3)": ["BeiDou B1", "BeiDou B3"],
    "Група: L1-смуга (L1/G1/E1/B1)": ["GPS L1", "ГЛОНАСС G1", "Galileo E1", "BeiDou B1"],
    "Група: Комплекс L1 по каналах (E1/G1/B1/L1)": [
        "Galileo E1 BOC (ШПС)", "ГЛОНАСС G1 FDMA (ШПС)", "BeiDou B1I (ШПС)",
        "GPS L1 C/A (ШПС)"],
    "Група: Mesh XK-F358 2T2R (TX1+TX2, S)": [
        "Mesh XK-F358 S (1600–2900)", "Mesh XK-F358 S (1600–2900)"],
    "Група: TGTX5050A CRPA (RF1+RF2)": [
        "TGTX5050A RF1 (GNSS L1)", "TGTX5050A RF2 (GNSS L1)"],
    "Група: TGTX5050A + завада (3 канали)": [
        "TGTX5050A RF1 (GNSS L1)", "TGTX5050A RF2 (GNSS L1)",
        "Завада широкосмугова (шум 20 МГц)"],
    "Група: Спуфер + реальний GPS (2 канали)": [
        "GPS L1 C/A (ШПС)", "Spoofing GPS — плаваюча (+10 Гц/с)"],
    "Група: TGTX5050A + спуфер (3 канали)": [
        "TGTX5050A RF1 (GNSS L1)", "TGTX5050A RF2 (GNSS L1)",
        "Spoofing GPS — плаваюча (+10 Гц/с)"],    "Група: ТЛМ 2020–2100 (2 канали BladeRF)": [
        "Завада блок 2020–2100 (Телеметрія, ниж.)",
        "Завада блок 2020–2100 (Телеметрія, верх.)"],
}



FREQ_LIB = [
    # (коротка назва категорії, [(назва з частотою, freq_Hz, тип, fs_hint, extra, note), ...])
    ("GNSS", [
        ("GPS L5 — 1176.45 МГц", 1176.45e6, "Шум", 4000e6, {},
         "GPS L5: смуга RNSS 1164–1210 МГц; модель — широкосмуговий шум"),
        ("GPS L2 — 1227.60 МГц", 1227.60e6, "Шум", 4000e6, {},
         "GPS L2: смуга RNSS 1215–1240 МГц; модель — широкосмуговий шум"),
        ("GPS L1 — 1575.42 МГц", 1575.42e6, "Шум", 4000e6, {},
         "GPS L1: смуга RNSS 1559–1610 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G3 — 1202.02 МГц", 1202.02e6, "Шум", 4000e6, {},
         "ГЛОНАСС G3 (L3): 1202.02 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G2 — 1246.00 МГц", 1246.0e6, "Шум", 4000e6, {},
         "ГЛОНАСС G2 (L2): 1246.00 МГц; модель — широкосмуговий шум"),
        ("ГЛОНАСС G1 — 1598.06–1605.37 МГц", 1601.715e6, "Шум", 4000e6, {},
         "ГЛОНАСС G1 (L1): 1598.06–1605.37 МГц, центр 1601.715; модель — широкосмуговий шум"),
        ("BeiDou B2 — 1191.79 МГц", 1191.79e6, "Шум", 4000e6, {},
         "BeiDou B2 (B2a/B2b): 1191.79 МГц; модель — широкосмуговий шум"),
        ("BeiDou B3 — 1268.52 МГц", 1268.52e6, "Шум", 4000e6, {},
         "BeiDou B3 (B3I): 1268.52 МГц; модель — широкосмуговий шум"),
        ("BeiDou B1 — 1561.09–1575.42 МГц", 1568.255e6, "Шум", 4000e6, {},
         "BeiDou B1: B1I 1561.09 / B1C 1575.42 МГц, центр 1568.255; модель — широкосмуговий шум"),
        ("Galileo E5 — 1191.79 МГц", 1191.79e6, "Шум", 4000e6, {},
         "Galileo E5 (E5a+E5b): 1191.79 МГц; модель — широкосмуговий шум"),
        ("Galileo E6 — 1278.75 МГц", 1278.75e6, "Шум", 4000e6, {},
         "Galileo E6: 1278.75 МГц; модель — широкосмуговий шум"),
        ("Galileo E1 — 1575.42 МГц", 1575.42e6, "Шум", 4000e6, {},
         "Galileo E1: 1575.42 МГц (спільна з GPS L1); модель — широкосмуговий шум"),
    ]),
    ("Стільниковий", [
        ("2G GSM B8 (GSM-900) — 880–960 МГц", 920e6, "Імпульс", 2000e6,
         {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
         "GSM-900 (B8): uplink 880–915 / downlink 935–960 МГц, центр 920; TDMA GMSK; модель — радіоімпульси, слот 577 мкс / кадр 4.615 мс"),
        ("2G GSM B3 (DCS-1800) — 1710–1880 МГц", 1795e6, "Імпульс", 2000e6,
         {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
         "DCS-1800 (B3): uplink 1710–1785 / downlink 1805–1880 МГц, центр 1795; модель — радіоімпульси GSM"),
        ("3G UMTS/WCDMA B8 — 900 МГц", 900e6, "Шум", 5000e6, {},
         "3G B8: WCDMA 900 МГц, канал 5 МГц; модель — широкосмуговий шум"),
        ("3G UMTS/WCDMA B1 — 2100 МГц", 2100e6, "Шум", 5000e6, {},
         "3G B1: WCDMA 2100 МГц, канал 5 МГц; модель — широкосмуговий шум"),
        ("4G LTE B20 — 800 МГц", 800e6, "Шум", 6000e6, {},
         "4G B20: FDD 800 МГц, OFDMA до 20 МГц; модель — широкосмуговий шум"),
        ("4G LTE B8 — 900 МГц", 900e6, "Шум", 6000e6, {},
         "4G B8: FDD 900 МГц; модель — широкосмуговий шум"),
        ("4G LTE B3 — 1800 МГц", 1800e6, "Шум", 6000e6, {},
         "4G B3: FDD 1800 МГц; модель — широкосмуговий шум"),
        ("4G LTE B1 — 2100 МГц", 2100e6, "Шум", 6000e6, {},
         "4G B1: FDD 2100 МГц; модель — широкосмуговий шум"),
        ("4G LTE B7 — 2600 МГц", 2600e6, "Шум", 6000e6, {},
         "4G B7: FDD 2600 МГц; модель — широкосмуговий шум"),
        ("5G NR n28 — 700 МГц", 700e6, "Шум", 6000e6, {},
         "5G n28: FDD 700 МГц; модель — широкосмуговий шум"),
        ("5G NR n8 — 900 МГц", 900e6, "Шум", 6000e6, {},
         "5G n8: FDD 900 МГц; модель — широкосмуговий шум"),
        ("5G NR n20 — 800 МГц", 800e6, "Шум", 6000e6, {},
         "5G n20: FDD 800 МГц; модель — широкосмуговий шум"),
        ("5G NR n3 — 1800 МГц", 1800e6, "Шум", 6000e6, {},
         "5G n3: FDD 1800 МГц; модель — широкосмуговий шум"),
        ("5G NR n1 — 2100 МГц", 2100e6, "Шум", 6000e6, {},
         "5G n1: FDD 2100 МГц; модель — широкосмуговий шум"),
        ("5G NR n7 — 2600 МГц", 2600e6, "Шум", 6000e6, {},
         "5G n7: FDD 2600 МГц; модель — широкосмуговий шум"),
        ("5G NR n78 (C-band) — 3300–3800 МГц", 3550e6, "Шум", 8000e6, {},
         "5G n78: TDD 3.3–3.8 ГГц, канал до 100 МГц; модель — широкосмуговий шум"),
        ("5G NR mmWave n257/n258 — 24.25–29.5 ГГц", 26875e6, "Шум", 60000e6, {},
         "5G FR2 (n257/n258): 24.25–29.5 ГГц; ПОЗА ПЛАНОМ 100 МГц…6 ГГц — генерацію буде відхилено"),
        ("5G NR mmWave n260/n261 — 37–43.5 ГГц", 40250e6, "Шум", 90000e6, {},
         "5G FR2 (n260/n261): 37–43.5 ГГц; ПОЗА ПЛАНОМ 100 МГц…6 ГГц — генерацію буде відхилено"),
    ]),
    ("Wi-Fi/BT", [
        ("Bluetooth 1.0–5.4 / LE — 2400–2483.5 МГц", 2441.75e6, "Шум", 6000e6, {},
         "Bluetooth: ISM 2.4 ГГц, FHSS 79 каналів × 1 МГц (LE — 40 × 2 МГц); модель — шум по смузі"),
        ("Wi-Fi 2.4 ГГц (канали 1–13) — 2400–2483.5 МГц", 2441.75e6, "Шум", 6000e6,
         {"am_m": 0.9, "am_f": 1000},
         "Wi-Fi 2.4: 13 каналів по 5 МГц (центр 2437); модель — шум із пакуванням (АМ-огинаюча)"),
        ("Wi-Fi 5 ГГц (UNII-1/2/3) — 5150–5875 МГц", 5512.5e6, "Шум", 6000e6,
         {"am_m": 0.9, "am_f": 1000},
         "Wi-Fi 5 ГГц: UNII-1 5150–5250, UNII-2 5250–5725, UNII-3 5725–5875 МГц; модель — шум із пакуванням"),
        ("Wi-Fi 6E / 7 — 5925–7125 МГц", 6525e6, "Шум", 6000e6, {},
         "Wi-Fi 6E/7: 5925–7125 МГц; ПОЗА ПЛАНОМ (центр 6.525 ГГц > 6 ГГц) — генерацію буде відхилено"),
    ]),
    ("IoT/M2M", [
        ("NFC / RFID — 13.56 МГц", 13.56e6, "Синус", 100e6, {},
         "NFC/RFID: ISM 13.56 МГц; ПОЗА ПЛАНОМ (< 100 МГц) — генерацію буде відхилено"),
        ("Zigbee / Thread — 2400–2483.5 МГц (також 868/915)", 2441.75e6, "Шум", 6000e6, {},
         "Zigbee/Thread: ISM 2.4 ГГц (16 каналів по 5 МГц), також 868.3/915 МГц; модель — шум"),
        ("Z-Wave Європа — 868.42 МГц", 868.42e6, "Синус", 2000e6, {},
         "Z-Wave EU: 868.42 МГц (9.6/40/100 кбіт/с); модель — вузькосмугова несуча"),
        ("Z-Wave США — 908.42 МГц", 908.42e6, "Синус", 2000e6, {},
         "Z-Wave US: 908.42 МГц; модель — вузькосмугова несуча"),
        ("LoRaWAN EU868 — 863–870 МГц", 866.5e6, "Імпульс", 2000e6,
         {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
          "chirp_dev": 0.125e6, "chirp_per_us": 1024},
         "LoRaWAN EU868: 863–870 МГц, центр 866.5; модель — радіоімпульси 1 мс / 10 мс з чірипами SF7/BW125"),
        ("LoRaWAN US915 — 902–928 МГц", 915e6, "Імпульс", 2000e6,
         {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
          "chirp_dev": 0.125e6, "chirp_per_us": 1024},
         "LoRaWAN US915: 902–928 МГц, центр 915; модель — радіоімпульси з чірипами SF7/BW125"),
    ]),
    ("Супутник", [
        ("L-діапазон (Inmarsat, Iridium) — 1.5–1.6 ГГц", 1550e6, "Шум", 4000e6, {},
         "L-діапазон: супутниковий зв'язок 1.5–1.6 ГГц; модель — шум"),
        ("C-діапазон (ТБ, магістралі) — 3.4–7.0 ГГц", 5200e6, "Шум", 12000e6, {},
         "C-діапазон: ТБ-мовлення, магістральні канали 3.4–7.0 ГГц, центр 5.2; модель — шум"),
        ("X-діапазон (військ., радари) — 7.25–8.4 ГГц", 7825e6, "Шум", 18000e6, {},
         "X-діапазон: 7.25–8.4 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
        ("Ku-діапазон (ТБ, Starlink) — 10.7–14.5 ГГц", 12600e6, "Шум", 30000e6, {},
         "Ku-діапазон: 10.7–14.5 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
        ("Ka-діапазон (Starlink, ViaSat) — 17.3–40.0 ГГц", 28650e6, "Шум", 60000e6, {},
         "Ka-діапазон: 17.3–40.0 ГГц; ПОЗА ПЛАНОМ — генерацію буде відхилено"),
    ]),
    ("Службовий", [
        ("Airband (AM) — 108–137 МГц", 122.5e6, "Синус", 500e6,
         {"am_m": 0.5, "am_f": 1000},
         "Airband: авіаційний AM 108–137 МГц, центр 122.5; модель — несуча з АМ (m=0.5, 1 кГц)"),
        ("VHF Marine — 156–174 МГц", 165e6, "Синус", 500e6, {},
         "Морський VHF: 156–174 МГц (канал 16 — 156.8 МГц); модель — вузькосмугова несуча"),
        ("VHF рації (DMR, TETRA) — 136–174 МГц", 155e6, "Синус", 1000e6, {},
         "Професійні VHF-рації: 136–174 МГц; модель — вузькосмугова несуча"),
        ("UHF рації (DMR, TETRA) — 400–470 МГц", 435e6, "Синус", 1000e6, {},
         "Професійні UHF-рації: 400–470 МГц; модель — вузькосмугова несуча"),
        ("PMR446 (Європа) — 446.0–446.2 МГц", 446.1e6, "Синус", 1000e6, {},
         "PMR446: 446.0–446.2 МГц, 16 каналів; модель — вузькосмугова несуча"),
        ("FRS (США) — 462–467 МГц", 464.5e6, "Синус", 1000e6, {},
         "FRS: 462–467 МГц, 22 канали; модель — вузькосмугова несуча"),
    ]),
]

LIB_FLAT = {}
for _short, _ents in FREQ_LIB:
    for _e in _ents:
        LIB_FLAT["[%s] %s" % (_short, _e[0])] = {
            "freq": _e[1], "kind": _e[2], "fs_hint": _e[3], "extra": _e[4],
            "width_us": None, "note": _e[5]}


IR_BANDS = {
    # Піддіапазони ІЧ 1.5–6 мкм; температури — за законом Віна (пік випромінювання в піддіапазоні)
    "SWIR (короткохвильовий ІЧ)": {
        "lam_lo": 1.5e-6, "lam_hi": 3.0e-6,
        "temp": "від 700 °C до понад 1600 °C",
        "sources": "форсажні струмені, розпечений метал, дугове зварювання, порохові гази"},
    "MWIR (середньохвильовий ІЧ)": {
        "lam_lo": 3.0e-6, "lam_hi": 6.0e-6,
        "temp": "від 210 °C до 700 °C",
        "sources": "корпуси авіаційних двигунів, вихлопні труби техніки, тертя об повітря на надзвуку"},
}


def wien_lambda_max(t_k):
    """Закон Віна: довжина хвилі піку випромінювання (м) для температури t_k (К)."""
    return 2897.771955e-6 / t_k


def lambda_to_freq(lam_m):
    """Еквівалентна частота (Гц) для довжини хвилі lam_m (м): f = c/λ."""
    return 299792458.0 / lam_m


def ir_subrange(lam_m):
    """Піддіапазон ІЧ (м): SWIR 1.5–3 мкм, MWIR 3–6 мкм, інакше — поза діапазоном."""
    for _name, _b in IR_BANDS.items():
        if _b["lam_lo"] <= lam_m <= _b["lam_hi"]:
            return _name
    return "поза 1.5–6 мкм"


def draw_ir_spectrum(fig):
    """Спектр Планка (нормований на пік) для 210/700/1000/1600 °C;
    заштриховані піддіапазони SWIR/MWIR; пунктири — піки за законом Віна."""
    lam = np.linspace(0.4e-6, 12e-6, 1200)
    h, c, k = 6.62607015e-34, 299792458.0, 1.380649e-23
    ax = fig.add_subplot(111)
    for tc, col in ((210, "C2"), (700, "C1"), (1000, "C3"), (1600, "C0")):
        t = tc + 273.15
        b = (2 * h * c ** 2) / lam ** 5 / np.expm1(h * c / (lam * k * t))
        b = b / b.max()
        lm = wien_lambda_max(t) * 1e6
        ax.plot(lam * 1e6, b, color=col, lw=1.4, label="%d °C (пік %.2f мкм)" % (tc, lm))
        if 0.4 <= lm <= 12:
            ax.axvline(lm, color=col, lw=0.6, ls="--", alpha=0.55)
    ax.axvspan(1.5, 3.0, color="C0", alpha=0.08)
    ax.axvspan(3.0, 6.0, color="C1", alpha=0.08)
    ax.text(2.25, 1.07, "SWIR", ha="center", fontsize=8)
    ax.text(4.5, 1.07, "MWIR", ha="center", fontsize=8)
    ax.set_xlabel("довжина хвилі, мкм")
    ax.set_ylabel("Planck, відн. одиниці (нормовано)")
    ax.set_title("Теплове випромінювання 1.5–6 мкм: Планів + піки за законом Віна")
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


IEEE_BANDS = [
    # Літерні мікрохвильові діапазони (IEEE); межі — нижня включно, верхня виключно
    # (4.0 ГГц → C, 12.0 ГГц → Ku); довжини хвиль — c/f на краях смуги
    ("UHF", 0.3e9, 1.0e9, "100 – 30 см",
     "Мобільний зв'язок (GSM, LTE), рації, цифрове ТБ (T2)"),
    ("L", 1.0e9, 2.0e9, "30 – 15 см",
     "Супутникова навігація (GPS, ГЛОНАСС), супутниковий телефон (Iridium)"),
    ("S", 2.0e9, 4.0e9, "15 – 7.5 см",
     "Wi-Fi (2.4 ГГц), Bluetooth, побутові мікрохвильовки (2.45 ГГц), цивільні радари"),
    ("C", 4.0e9, 8.0e9, "7.5 – 3.75 см",
     "Wi-Fi (5 ГГц), магістральний супутниковий зв'язок, метеорологічні радари"),
    ("X", 8.0e9, 12.0e9, "3.75 – 2.5 см",
     "Військові радари, системи наведення зброї, супутникова розвідка"),
    ("Ku", 12.0e9, 18.0e9, "2.5 – 1.67 см",
     "Супутникове телебачення, клієнтські термінали Starlink"),
    ("K", 18.0e9, 27.0e9, "1.67 – 1.11 см",
     "Супутниковий зв'язок, радарні датчики руху"),
    ("Ka", 27.0e9, 40.0e9, "1.11 – 0.75 см",
     "Швидкісний інтернет (шлюзи Starlink), радари ДПС, 5G (mmWave)"),
    ("V/W", 40.0e9, 110.0e9, "7.5 – 2.7 мм",
     "Надшвидкісні радіорелейні лінії, автомобільні радари круїз-контролю"),
]


def ieee_band_for(f_hz):
    """Літерний діапазон IEEE для частоти f_hz (Гц) або None (поза 0.3–110 ГГц);
    нижня межа включно, верхня виключно."""
    for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
        if _f0 <= f_hz < _f1:
            return _name
    return None


def freq_to_lambda(f_hz):
    """Довжина хвилі (м) для частоти f_hz (Гц): λ = c/f."""
    return 299792458.0 / f_hz


def lambda_cm(lam_m):
    """Довжина хвилі у зручних одиницях: >=1 см → «см», менше → «мм»."""
    cm = lam_m * 100.0
    return ("%.4g см" % cm) if cm >= 1.0 else ("%.4g мм" % (lam_m * 1000.0))


def draw_ieee_spectrum(fig):
    """Лог-шкала 0.3–110 ГГц: літерні діапазони IEEE заштрихованими смугами
    з підписами та мітками відомих сервісів (GPS, GSM, Wi-Fi, Starlink, 5G)."""
    ax = fig.add_subplot(111)
    for _i, (_name, _f0, _f1, _lam, _uses) in enumerate(IEEE_BANDS):
        col = "C%d" % (_i % 10)
        ax.axvspan(_f0 / 1e9, _f1 / 1e9, color=col, alpha=0.12)
        fc = (_f0 + _f1) / 2 / 1e9
        ax.text(fc, 1.07, _name, ha="center", fontsize=9, fontweight="bold", color=col)
        ax.text(fc, 1.0, "%.4g–%.4g" % (_f0 / 1e9, _f1 / 1e9),
                ha="center", fontsize=6.5, color="#555555")
    for _label, _f in (("GSM 900", 0.9), ("GPS L1", 1.57542), ("Wi-Fi/BT", 2.45),
                       ("Wi-Fi 5", 5.5), ("військ. радар", 10.0), ("Starlink Ku↓", 12.0),
                       ("5G mmWave", 28.0), ("радар авто", 77.0)):
        ax.axvline(_f, color="k", lw=0.7, ls=":", alpha=0.55)
        ax.text(_f, -0.32, _label, rotation=45, ha="right", va="top", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("частота, ГГц (лог. шкала)")
    ax.set_yticks([])
    ax.set_title("Мікрохвильові діапазони IEEE 0.3–110 ГГц: літерні смуги + відомі сервіси")
    ax.set_xlim(0.3, 110)
    ax.set_ylim(-0.9, 1.22)
    ax.grid(alpha=0.3, which="both")


HAM_BANDS = [
    # Діапазони радіоаматорів (IARU Region 1); межі — Гц, нижня включно, верхня виключно
    ("160 м", 1.810e6, 2.000e6, "Нічний/зимовий ДХ; вдень — місцевий зв'язок"),
    ("80 м", 3.500e6, 3.800e6, "Регіональний зв'язок; нічні ДХ"),
    ("60 м", 5.3515e6, 5.3665e6, "Вузький діапазон WRC-15 (лише цифрові режими/CW)"),
    ("40 м", 7.000e6, 7.200e6, "Круглодобовий ДХ: вдень регіон, вночі континенти"),
    ("30 м", 10.100e6, 10.150e6, "Лише CW і цифрові режими (FT8, WSPR)"),
    ("20 м", 14.000e6, 14.350e6, "Основний ДХ-діапазон: дальні зв'язки вдень"),
    ("17 м", 18.068e6, 18.168e6, "ДХ у максимум сонячної активності"),
    ("15 м", 21.000e6, 21.450e6, "ДХ у максимум циклу, літні зв'язки"),
    ("12 м", 24.890e6, 24.990e6, "Перехідний ДХ-діапазон"),
    ("10 м", 28.000e6, 29.700e6, "Спорадичний Е-слой; FM-репітери 29.5–29.7"),
    ("6 м", 50.000e6, 52.000e6, "«Чарівна смуга»: спорадичний Е-слой, метеорне розсіювання"),
    ("4 м", 70.000e6, 70.500e6, "Лише Region 1: локальний зв'язок, спорадичний Е-слой"),
    ("2 м", 144.000e6, 146.000e6, "Репітери, FM, SSTV, метеорне розсіювання, EME"),
    ("70 см", 430.000e6, 440.000e6, "Репітери, APRS, аматорські супутники"),
    ("23 см", 1240.000e6, 1300.000e6, "EME, супутники, високошвидкісна передача даних"),
]


def ham_band_for(f_hz):
    """Діапазон радіоаматорів для частоти f_hz (Гц) або None (між діапазонами);
    нижня межа включно, верхня виключно."""
    for _name, _f0, _f1, _uses in HAM_BANDS:
        if _f0 <= f_hz < _f1:
            return _name
    return None


HAM_FT8 = {
    # Стандартні частоти FT8 на кожен діапазон (МГц)
    "160 м": 1.840, "80 м": 3.573, "60 м": 5.357, "40 м": 7.074, "30 м": 10.136,
    "20 м": 14.074, "17 м": 18.100, "15 м": 21.074, "12 м": 24.915, "10 м": 28.074,
    "6 м": 50.313, "4 м": 70.154, "2 м": 144.174, "70 см": 432.174, "23 см": 1296.174,
}


def draw_ham_spectrum(fig):
    """Лог-шкала 1.8–1300 МГц: діапазони радіоаматорів як горизонтальні смуги
    (barh); ромби — стандартні частоти FT8."""
    ax = fig.add_subplot(111)
    n = len(HAM_BANDS)
    for _i, (_name, _f0, _f1, _uses) in enumerate(HAM_BANDS):
        col = "C%d" % (_i % 10)
        y = n - 1 - _i
        ax.barh(y, (_f1 - _f0) / 1e6, left=_f0 / 1e6, height=0.62,
                color=col, alpha=0.5, edgecolor=col, linewidth=0.8)
        if _name in HAM_FT8:
            ax.plot(HAM_FT8[_name], y, marker="D", ms=4.5, color="k", ls="")
    ax.plot([], [], marker="D", ms=4.5, color="k", ls="", label="FT8 (стандартна частота)")
    ax.set_xscale("log")
    ax.set_yticks([n - 1 - _i for _i in range(n)])
    ax.set_yticklabels([_b[0] for _b in HAM_BANDS], fontsize=8)
    ax.set_xlabel("частота, МГц (лог. шкала)")
    ax.set_xlim(1.3, 1700)
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Діапазони радіоаматорів 160 м … 23 см (IARU Region 1) + частоти FT8")


MOD_TYPES = [
    # Типи модуляції: (назва, сутність, типове застосування)
    ("Амплітудна (АМ)",
     "Зміна сили (амплітуди) сигналу: несуча × повільна огинаюча; "
     "x(t) = (1+m·sin(2π·f_m·t))·sin(2π·f_c·t)",
     "СХ-мовлення, авіаційний зв'язок (Airband), CB-рації"),
    ("Частотна (ЧМ/FM)",
     "Зміна частоти коливань: миттєва частота відхиляється пропорційно сигналу, "
     "амплітуда стала — нечутливість до амплітудних завад",
     "УКХ-мовлення 87.5–108 МГц, рації, аналогове ТБ"),
    ("Фазова (ФМ/PM)",
     "Зміна фази хвилі: стрибки фази (BPSK/QPSK) кодують біти — "
     "основа цифрового зв'язку",
     "Wi-Fi, LTE/5G, GNSS, супутниковий зв'язок"),
]

EW_METHODS = [
    # Радіоелектронне придушення та обман: (метод, сутність, типовий захист)
    ("Глушіння «білим шумом» (Barrage Jamming)",
     "Широкосмуговий шум перекриває весь діапазон приймача: густина потужності "
     "на канал низька, покриття повне; у спектрі — шумовий п'єдестал",
     "FHSS (стрибки частоти), DSSS, режекція смуги"),
    ("Прицільне глушіння (Spot Jamming)",
     "Вся потужність шуму на одній частоті/каналі — максимальна густина; "
     "у спектрі — вузький шумовий підйом над сигналом",
     "Перестройка частоти, notch-фільтр, FHSS"),
    ("Підміна / Спуфінг (Spoofing)",
     "Фальшивий сигнал, що імітує справжній: приймач захоплюється підробленим "
     "(типово — GNSS з помилковою позицією/часом)",
     "Автентифікація сигналу (Galileo OSNMA), контроль C/N₀"),
]

FARADAY_NOTE = (
    "Клітка Фарадея (імітація блокування): екран із провідної сітки розподіляє заряди "
    "і гасить зовнішнє електромагнітне поле — всередині корисного сигналу немає, лише "
    "шумовий пол приймача. Це пасивний захист від глушіння й прослуховування; для "
    "зв'язку антени виносять за межі екрана. В імітації: сигнальні піки у спектрі "
    "відсутні, потужність рівномірно розподілена по полу."
)


def draw_mod_ew_spectrum(fig):
    """2×2: осцилограми АМ / ЧМ / ФМ (BPSK) та спектр придушення
    (barrage / spot / клітка Фарадея)."""
    t = np.linspace(0, 1e-3, 4000)
    fc, fm = 40e3, 2e3
    ms = t * 1e3

    ax = fig.add_subplot(221)
    env = 1 + 0.6 * np.sin(2 * np.pi * fm * t)
    ax.plot(ms, env * np.sin(2 * np.pi * fc * t), lw=0.5, color="C0")
    ax.plot(ms, env, lw=1.3, ls="--", color="C3")
    ax.plot(ms, -env, lw=1.3, ls="--", color="C3")
    ax.set_title("АМ — зміна амплітуди (огинаюча — пунктир)", fontsize=9)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(222)
    ax.plot(ms, np.sin(2 * np.pi * fc * t + 5 * np.sin(2 * np.pi * fm * t)),
            lw=0.5, color="C1")
    ax.set_title("ЧМ/FM — зміна частоти (девіація β = 5)", fontsize=9)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(223)
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    seq = np.repeat(bits, len(t) // len(bits))
    ax.plot(ms, np.sin(2 * np.pi * fc * t + np.pi * seq), lw=0.5, color="C2")
    for j in np.where(np.diff(seq) != 0)[0]:
        ax.axvline(ms[j], color="k", lw=0.5, ls=":", alpha=0.5)
    ax.set_title("ФМ/PM — BPSK: стрибки фази 180° (пунктир)", fontsize=9)
    ax.set_xlabel("час, мс", fontsize=8)
    ax.set_ylabel("x(t)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(224)
    f = np.linspace(80, 120, 800)
    sig = 0.02 + 0.9 * np.exp(-((f - 100.0) ** 2) / (2 * 0.15 ** 2))
    ax.plot(f, 0.35 + sig, lw=1.1, color="C3", label="Barrage — шум по всій смузі")
    ax.plot(f, sig + 0.5 * np.exp(-((f - 100.0) ** 2) / (2 * 1.2 ** 2)),
            lw=1.1, color="C1", label="Spot — прицільне глушіння")
    ax.plot(f, sig, lw=1.1, color="C0", label="чистий сигнал")
    ax.plot(f, 0.02 * np.ones_like(f), lw=1.4, color="C2", label="за кліткою Фарадея")
    ax.set_title("Глушіння і блокування (спектр, відн. одиниці)", fontsize=9)
    ax.set_xlabel("частота, МГц", fontsize=8)
    ax.set_ylabel("потужність", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout(h_pad=2.0)


MANUAL_TEXT = """## 1. Встановлення і запуск
• Windows: розпакуйте RAW_Studio_Windows.zip і запустіть RAW_Studio.bat; якщо Python не встановлений — інсталятор відкриє сторінку python.org (Python 3.9+, галочка «Add python.exe to PATH»); при першому запуску самі поставляться numpy і matplotlib (потрібен інтернет, ~1 хвилина).
• Без .bat: двічі клацніть RAW_Studio.pyw (якщо Python уже встановлений).
• Linux/macOS: python3 raw_studio.py (потрібні numpy і matplotlib).
• Власний .exe: py -m PyInstaller --onefile --windowed raw_studio.py (на машині з Windows).

## 2. Головне вікно і верхнє поле
• «📂 Відкрити файл» — виберіть .raw запис; тут же fs, МГц і кількість каналів.
• Смуга, МГц — ширина канала частотного плану (за замовчуванням 64; Enter або вихід з поля — застосувати).
• «Аналізувати» — перечитати файл і оновити всі вкладки після зміни fs/каналів.
• «📖 Довідка по параметрах» — розгортає опис кожного поля інтерфейсу.

## 3. Формат .raw
• Беззаголовковий потік uint16 little-endian, інтерлівінг по відліках [ch0, ch1, …, ch0, …], 1–8 каналів.
• Коди беззнакові 0…65535 — тони сидять на середній шкалі 32768; переповнення (кліп) — 0 або 65535.

## 4. Вкладки аналізу
• Часова область — осцилограма всіх каналів (min-max смуга + середнє); «з, с» і «тривалість, с» задають вікно перегляду; панель matplotlib — масштабування і збереження PNG.
• Спектр — амплітудний спектр і поєднана АЧХ; піки підписані частотою і рівнем у кодах.
• Блоки — частотний план 100 МГц…6 ГГц (6 блоків); ширина канала — поле «Смуга, МГц»; зайняті канали підсвічені.
• Калібрування АЧХ — прогін тонів через реальний тракт і вимірювання АЧХ (див. розділ 6).
• Статистика — DC, RMS, min/max, кліп-переповнення і кореляційна перевірка інтерлівінгу; здоровий запис: кліпів 0, min далекий від 0, max — від 65535.
• ІЧ (1.5–6 мкм) — SWIR/MWIR: частоти, температури за Віном, характерні джерела; калькулятор T °C ↔ λ пік ↔ f = c/λ; спектр Планка.
• Мікрохвилі (IEEE) — літерні діапазони UHF…V/W (0.3–110 ГГц); калькулятор частота ↔ довжина хвилі ↔ діапазон; лог-графік із сервісами.
• Радіолюбительські (160м…23см) — 15 аматорських смуг IARU Region 1; калькулятор МГц ↔ м; лог-графік із частотами FT8.
• Модуляція і РЕБ — АМ/ЧМ/ФМ (осцилограми), barrage/spot/spoofing (сутність і захист), клітка Фарадея.
• Генератор — створення нових .raw (див. розділ 5).
• I/Q → .raw — конвертер записів RTSA-Suite та інших I/Q (див. розділ 7).
• Інструкція — цей довідник всередині програми.

## 5. Генератор .raw — порядок роботи
• Кроки: 1 Модель SDR → 2 Тип сигналу → 2в Таблиця якісної завади → 3 Додаткові параметри (опційно) → 4 АЧХ (опційно) → 5 «Створити .raw…».
• Таблиця каналів: ✓ «Увімк» + форма, частота (МГц), фаза, амплітуда, ширина, DC, підсилення; вимкнені канали пишуться як постійна складова DC.
• fs, МГц (100…6000; 12000 = 12 ГГц), тривалість, с і seed — внизу.
• Модель SDR підставляє fs обраної плати і показує мінімальну ширину імпульса (4 відліки); кнопка підставляє її у поле «Ширина, мкс».
• Тип сигналу (54 стандарти + 12 груп): LoRa, ..., Mesh XK-F358 (UHF/L/S/C/MIIT — TD-COFDM + FHSS), ШПС-пресети GPS/Galileo/ГЛОНАСС/BeiDou, UNB і решта, FSK/GFSK, FHSS, QAM, CVBS, COFDM, NB-IoT — «→ застосувати» ставить форму, несучу, fs і додаткові параметри в перший увімкнений рядок.
• Бібліотека частот (6 категорій, 51 запис: GNSS, стільникові, Wi-Fi/BT, IoT, супутники, службові) працює так само; записи поза планом 100 МГц…6 ГГц позначені «ПОЗА ПЛАНОМ».
• Частота вводиться в МГц (852 або 2400); старий запис у Гц (2.4e9) теж розуміється.

## 6. Модуль АЧХ і калібрування
• Профілі: Плоский / Спад −3 дБ/окт / Смуговий 100М–6Г / Провал / Користувацький; інтерполяція linear/spline/logspline; екстраполяція clamp/linear/zero.
• Калібрування тракту: крок 1 — створити калібрувальний .raw і прогнати через тракт; крок 2 — виміряти АЧХ із запису; крок 3 — профіль у Генератор (компенсація −дБ).
• 💾 / 📁 — збереження/завантаження профілів .achx (текстові: рядки «частота_Гц дБ» + шапка режимів).

## 7. Конвертер I/Q → .raw
• Читає .iq (Raw IQ з RTSA-Suite), .wav (стерео) і будь-який бінарний raw: float32/float64/int8/int16/int32.
• Якщо поруч лежить .iq.xml (RTSA_RAW_IQ_File), fs і центральна частота підставляються автоматично; нативні стиснуті .rtsa не читаються — експортуйте Raw IQ у File Source.
• Конвертор розкладає середній спектр запису по каналах плану («Смуга, МГц») і синтезує кожен канал: піки, рівні і зайнятість відтворюються; при відкритті .raw вкажіть ту саму fs.

## 8. Командний рядок (без GUI)
• python raw_studio.py --headless файл.raw — аналіз у консоль.
• python raw_studio.py --selftest файл.raw — самоперевірка ядра.
• python raw_studio.py --preview файл.raw out.png — графік у PNG.
• python raw_studio.py --calgen файл.raw plan.json — калібрувальний файл.
• python raw_studio.py --calmeas after.raw plan.achx — вимірювання АЧХ.

## 9. Типові проблеми
• Тони «розсипаються» по всіх каналах → перевірте fs і кількість каналів.
• Тон пливе або слабкий → перевірте DC (≈32768) і кліпи у Статистиці.
• .rtsa не відкривається → експортуйте Raw IQ у File Source.
• Порожні вкладки → натисніть «Аналізувати» після зміни fs/каналів.
• Перші секунди .exe «мовчать» → self-розпакування (5–10 с), це нормально.
"""


def parse_rtsa_xml(xml_path):
    """Sidecar .iq.xml запису Raw IQ з RTSA-Suite (RTSA_RAW_IQ_File):
    Clock (Гц), CenterFrequency (Гц), DataType (float32/int16/...)."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    if "RTSA_RAW_IQ_File" not in (root.tag or ""):
        return None

    def _num(tag):
        el = root.find(tag)
        if el is None or not el.text:
            return None
        try:
            return float(el.text.strip())
        except ValueError:
            return None

    out = {}
    clock, cf = _num("Clock"), _num("CenterFrequency")
    if clock:
        out["fs_iq"] = clock
    if cf is not None:
        out["fc"] = cf
    dt = root.findtext("DataType")
    if dt and dt.strip():
        out["dtype"] = dt.strip()
    return out or None


def read_iq(path, dtype="float32", fs_iq=0.0, fc=0.0, max_samples=1 << 22):
    """Читає I/Q запис: .iq (raw інтерлівінг Re,Im), .wav (стерео I=лівий,
    Q=правий) або будь-який бінарний raw. Повертає (комплексний масив, meta)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        import wave
        w = wave.open(path, "rb")
        nch_w, sw, fr, nf = (w.getnchannels(), w.getsampwidth(),
                             w.getframerate(), w.getnframes())
        raw = w.readframes(nf)
        w.close()
        if nch_w != 2:
            raise ValueError("WAV має бути стерео (I — лівий, Q — правий канал)")
        if sw == 2:
            s = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
            dts = "int16(wav)"
        elif sw == 4:
            s = np.frombuffer(raw, "<f4").astype(np.float64)
            dts = "float32(wav)"
        else:
            raise ValueError("WAV: підтримуються 16-біт int і 32-біт float")
        if s.size % 2:
            s = s[:-1]
        x = s[0::2] + 1j * s[1::2]
        return x[:max_samples], {"fs_iq": fs_iq or float(fr), "fc": fc,
                                 "dtype": dts, "n_total": nf}
    dt = RTSA_IQ_DTYPES.get(dtype)
    if dt is None:
        raise ValueError("Невірний тип даних: %s" % dtype)
    a = np.fromfile(path, dt)
    if a.size < 2:
        raise ValueError("Файл занадто малий або порожній")
    if a.size % 2:
        a = a[:-1]
    a = a[:2 * max_samples]
    x = a[0::2].astype(np.float64) + 1j * a[1::2].astype(np.float64)
    x = x / RTSA_IQ_NORM[dtype]
    return x, {"fs_iq": fs_iq, "fc": fc, "dtype": dtype,
               "n_total": a.size // 2}


def iq_field_errors(fs_s, fc_s, fsout_s, n_s, nch_s):
    """Назви полів конвертера I/Q з порожніми або невірними числами (v4.6.1)."""
    bad = []
    for _s, _name in ((fs_s, "fs I/Q, МС/с"), (fc_s, "центр, МГц"),
                      (fsout_s, "fs .raw, МГц")):
        try:
            if float(parse_freq_hz(str(_s))) <= 0:
                raise ValueError(_name)
        except Exception:
            bad.append(_name)
    for _s, _name in ((n_s, "довжина канала .raw, відліків"),
                      (nch_s, "каналів у .raw")):
        try:
            if int(float(str(_s).strip())) <= 0:
                raise ValueError(_name)
        except Exception:
            bad.append(_name)
    return bad


def rtsa_iq_chunks(path):
    """Генератор по чанках .rtsa (RTSA-File-Format-4): (cid, ver, hsize, body)."""
    import struct as _st
    with open(path, "rb") as f:
        while True:
            head = f.read(16)
            if len(head) < 16:
                break
            cid = head[0:4].decode("ascii", "replace")
            csize, _fl = _st.unpack("<II", head[4:12])
            ver, hsize = _st.unpack("<HH", head[12:16])
            if csize < 16:
                raise ValueError(".rtsa: некоректний розмір чанка %s (%d)" % (cid, csize))
            yield cid, ver, hsize, f.read(max(csize - 16, 0))


_RTSA_FMT = {0: "<u1", 1: "<u2", 2: "<i2", 3: "<u4", 4: "<i4", 5: "<f4",
             6: "<u1", 7: "<u2", 8: "<i2", 9: "<u4", 10: "<i4", 11: "<f4"}
_RTSA_NAME = {0: "uint8", 1: "uint16", 2: "int16", 3: "uint32", 4: "int32",
              5: "float32", 6: "uint8", 7: "uint16", 8: "int16", 9: "uint32",
              10: "int32", 11: "float32"}


def _rtsa_meta_sstr(body):
    """Частоти з тіла SSTR (адаптивно до 64-бітного паддінгу): (fstart, fstep, fspan)."""
    import struct as _st
    for _off in (20, 24):
        try:
            v0, v1, v2 = _st.unpack_from("<3d", body, _off)
        except Exception:
            continue
        if 1e3 <= abs(v0) <= 1e13 and 1.0 <= abs(v1) <= 1e11 and abs(v2) >= abs(v1) * 0.999:
            return v0, v1, v2
    return None


def _rtsa_samp_head(body):
    """Заголовок SAMP-тіла: (streamID, subID, stype, sunit, ptype, comp,
    pflags, ssize, sdepth, nsamp) або None."""
    import struct as _st
    if len(body) < 48:
        return None
    sid, ssid = _st.unpack_from("<QI", body, 0)
    stype, sunit, ptype, comp = _st.unpack_from("<4b", body, 12)
    pflags, ssize, sdepth, nsamp = _st.unpack_from("<IIII", body, 32)
    return sid, ssid, stype, sunit, ptype, comp, pflags, ssize, sdepth, nsamp


def _rtsa_xfmt(v):
    if not v:
        return "0"
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return "%.10g" % v


def _rtsa_refuse(fs_iq, fc, comp):
    return (".rtsa: IQ-пакети стиснуті (mCompression=%s) — алгоритм стиснення Aaronia "
            "закритий і не публікується. Розв'язання: у RTSA-Suite File Source → Export → "
            "Raw IQ (.iq), або у командному рядку RTSA-Suite: RSTAFileTool repair "
            "-compress=0 запис.rtsa вихід.rtsa. Метадані запису: fs≈%s Гц, центр≈%s Гц."
            % (comp, _rtsa_xfmt(fs_iq), _rtsa_xfmt(fc)))


def rtsa_to_iq_pair(path, iq_path, xml_path, progress=None, max_bytes=1 << 40):
    """Розпаковує нестиснутий IQ-запис .rtsa у пару .iq (сирі Re,Im у вихідному
    типі) + .iq.xml (RTSA_RAW_IQ_File: Clock, CenterFrequency, DataType).
    Повертає кількість комплексних відліків; стиснуті пакети — ValueError."""
    import numpy as _np
    fs_iq = fc = None
    stype_seen = None
    n_total = 0
    comp_seen = None
    f = open(iq_path, "wb")
    try:
        for cid, ver, hsize, body in rtsa_iq_chunks(path):
            if cid == "SSTR":
                m = _rtsa_meta_sstr(body)
                if m:
                    fs_iq, fc = abs(m[1]), m[0] + abs(m[2]) / 2.0
            elif cid == "SAMP" and hsize >= 64:
                h = _rtsa_samp_head(body)
                if h is None:
                    continue
                _sid, _ssid, stype, _su, ptype, comp, _pf, ssize, sdepth, nsamp = h
                if ptype != 2:
                    continue
                if comp != 0:
                    comp_seen = comp
                    continue
                fmt = _RTSA_FMT.get(stype)
                if fmt is None:
                    continue
                numel = nsamp * max(ssize, 1) * max(sdepth, 1)
                off = hsize - 16
                need = numel * _np.dtype(fmt).itemsize
                a = _np.frombuffer(body[off:off + need], dtype=fmt)
                if a.size < numel:
                    continue
                stype_seen = stype
                if a.size % 2:
                    a = a[:-1]
                f.write(a.tobytes())
                n_total += a.size // 2
                if progress and (n_total & 0xFFFFF) < (a.size // 2):
                    progress("I/Q: .rtsa → .iq… %d М відліків" % (n_total // 1000000))
    finally:
        f.close()
    if not n_total:
        try:
            os.remove(iq_path)
        except OSError:
            pass
        if comp_seen is not None:
            raise ValueError(_rtsa_refuse(fs_iq, fc, comp_seen))
        raise ValueError(".rtsa: нестиснутих IQ-пакетів (payload IQ) не знайдено")
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<RTSA_RAW_IQ_File>\n'
           "  <Clock>%s</Clock>\n  <CenterFrequency>%s</CenterFrequency>\n"
           "  <DataType>%s</DataType>\n</RTSA_RAW_IQ_File>\n"
           % (_rtsa_xfmt(fs_iq), _rtsa_xfmt(fc), _RTSA_NAME.get(stype_seen, "float32")))
    open(xml_path, "w", encoding="utf-8").write(xml)
    return n_total


def rtsa_read_iq(path, max_samples=1 << 24):
    """Читає нестиснутий IQ-запис .rtsa у комплексний масив (для аналізу).
    Стиснуті пакети — ValueError з підказкою (див. rtsa_to_iq_pair)."""
    import numpy as _np
    fs_iq = fc = None
    out = []
    total = 0
    comp_seen = None
    for cid, ver, hsize, body in rtsa_iq_chunks(path):
        if cid == "SSTR":
            m = _rtsa_meta_sstr(body)
            if m:
                fs_iq, fc = abs(m[1]), m[0] + abs(m[2]) / 2.0
        elif cid == "SAMP" and hsize >= 64:
            h = _rtsa_samp_head(body)
            if h is None:
                continue
            _sid, _ssid, stype, _su, ptype, comp, _pf, ssize, sdepth, nsamp = h
            if ptype != 2:
                continue
            if comp != 0:
                comp_seen = comp
                continue
            fmt = _RTSA_FMT.get(stype)
            if fmt is None:
                continue
            numel = nsamp * max(ssize, 1) * max(sdepth, 1)
            off = hsize - 16
            need = numel * _np.dtype(fmt).itemsize
            a = _np.frombuffer(body[off:off + need], dtype=fmt)
            if a.size < numel:
                continue
            if a.size % 2:
                a = a[:-1]
            out.append(a.astype(_np.float64))
            total += a.size // 2
            if total >= max_samples:
                break
    if not out:
        if comp_seen is not None:
            raise ValueError(_rtsa_refuse(fs_iq, fc, comp_seen))
        raise ValueError(".rtsa: нестиснутих IQ-пакетів (payload IQ) не знайдено")
    x = _np.concatenate(out)
    if x.size % 2:
        x = x[:-1]
    xc = x[0::2] + 1j * x[1::2]
    meta = {"fs_iq": float(fs_iq) if fs_iq else 0.0,
            "fc": float(fc) if fc is not None else 0.0,
            "dtype": "float32", "n_total": int(xc.size)}
    return xc[:max_samples], meta


def plan_channels(bw_hz=CHANNEL_BW):
    """Канали частотного плану 100М…6Г по bw_hz (за замовчуванням 64 МГц): [(f0, f1, global), ...]."""
    out, g = [], 0
    for lo, hi in FREQ_BLOCKS:
        f0 = lo
        while f0 + bw_hz <= hi + 1e-6:
            out.append((f0, f0 + bw_hz, g))
            g += 1
            f0 += bw_hz
    return out



def fspl_free_db(f_hz, d_km):
    """Затухання у вільному просторі (FSPL), дБ: 92.45 + 20lg(f_ГГц) + 20lg(d_км) (v5.1.0)."""
    f_ghz = max(float(f_hz) / 1e9, 1e-6)
    d = max(float(d_km), 1e-4)
    return 92.45 + 20.0 * math.log10(f_ghz) + 20.0 * math.log10(d)


def coverage_radius_km(eirp_dbm, thresh_dbm, f_hz, g_ant_db=0.0, g_rx_db=0.0):
    """Максимальна дальність завади, км (v5.1.0): EIRP + G(кут) + G_rx - FSPL(d) >= поріг."""
    f_ghz = max(float(f_hz) / 1e9, 1e-6)
    m = (float(eirp_dbm) + float(g_ant_db) + float(g_rx_db) - float(thresh_dbm)
         - 92.45 - 20.0 * math.log10(f_ghz))
    return max(10.0 ** (m / 20.0), 0.0)


def coverage_map(eirp_dbm, thresh_dbm, f_hz, ant_name="", hpb=60.0, fbr=20.0,
                 g_rx_db=0.0, n_pts=181):
    """Мапа покриття (v5.1.0): (кути 0..360 град, дальності км) - G(кут) з моделі пелюстки
    ant_pattern_db, r(кут) = coverage_radius_km(EIRP, поріг, f, G(кут), G_rx)."""
    degs = np.linspace(0.0, 360.0, int(n_pts))
    g = np.array([ant_pattern_db(ant_name or "Інший (власноруч)", float(d),
                                 hpb=hpb, fbr=fbr) for d in degs])
    rs = np.array([coverage_radius_km(eirp_dbm, thresh_dbm, f_hz,
                                      g_ant_db=float(gi), g_rx_db=g_rx_db) for gi in g])
    return degs, rs


def coverage_metric(rows, ref_db=None):
    """Метрика покриття плану завад (v5.1.0): rows = [(f_Гц, bw_Гц, ампл_кодів)].
    Повертає dict: pct - % діапазону плану, вкритого хоч однією завадою;
    holes - список (lo, hi) дір усередині діапазону; js - J/S по каналах
    (20lg(ампл, кодів) - рівень сигналу, дБ)."""
    bands = []
    for f, bw, _a in rows:
        f, bw = float(f), abs(float(bw))
        if f > 0 and bw > 0:
            bands.append((f - bw / 2.0, f + bw / 2.0))
    js = [20.0 * math.log10(max(abs(float(a)), 1.0)) - float(ref_db)
          if ref_db is not None else 0.0 for _f, _bw, a in rows]
    if not bands:
        return {"pct": 0.0, "holes": [], "js": js}
    bands.sort()
    merged, cur = [], list(bands[0])
    for lo, hi in bands[1:]:
        if lo <= cur[1]:
            cur[1] = max(cur[1], hi)
        else:
            merged.append(tuple(cur))
            cur = [lo, hi]
    merged.append(tuple(cur))
    span = merged[-1][1] - merged[0][0]
    covered = sum(hi - lo for lo, hi in merged)
    pct = 100.0 * covered / span if span > 0 else 0.0
    holes = [(h0, l1) for (l0, h0), (l1, h1) in zip(merged, merged[1:]) if l1 - h0 > 1e-6]
    return {"pct": pct, "holes": holes, "js": js}


def rtl_tcp_cmds(freq_hz, rate, gain_db=0.0):
    """Команди протоколу rtl_tcp (v6.2.0): 1 Б команда + 4 Б аргумент (big-endian)."""
    import struct as _st
    _out = b"\x01" + _st.pack(">I", int(freq_hz)) + b"\x02" + _st.pack(">I", int(rate))
    if gain_db:
        _out += b"\x04" + _st.pack(">I", int(gain_db * 10))
    return _out


def rtl_tcp_read(host, port, n, freq_hz=None, rate=None, gain_db=0.0, timeout=5.0):
    """Клієнт rtl_tcp без залежностей (v6.2.0): 8-бітні I/Q → комплексні відліки."""
    import socket as _sk
    _s = _sk.create_connection((host, int(port)), timeout=timeout)
    try:
        if freq_hz and rate:
            _s.sendall(rtl_tcp_cmds(freq_hz, rate, gain_db))
        _buf = b""
        while len(_buf) < 2 * int(n):
            _c = _s.recv(65536)
            if not _c:
                break
            _buf += _c
    finally:
        _s.close()
    _a = np.frombuffer(_buf[:2 * int(n)], dtype=np.uint8).astype(np.float64) - 127.5
    return _a[0::2] + 1j * _a[1::2]


def waterfall_column(samples, nfft=256, lo_db=-100.0, hi_db=-20.0):
    """Колонка спектра водоспаду (v6.2.0): вікно Ганна → спектр → дБ → кліп у діапазон.

    Дійсні сигнали → rfft (nfft//2+1 комірок); комплексні IQ (RTL-TCP) → fft +
    позитивна половина — той самий розмір колонки."""
    _n = min(int(nfft), len(samples))
    _seg = np.asarray(samples[:_n])
    _win = np.hanning(_n)
    if np.iscomplexobj(_seg):
        _sp = np.abs(np.fft.fft(_seg * _win))[:_n // 2 + 1] + 1e-12
    else:
        _sp = np.abs(np.fft.rfft(_seg.astype(np.float64) * _win)) + 1e-12
    return np.clip(20.0 * np.log10(_sp), lo_db, hi_db)


def waterfall_push(buf, col_db):
    """Кільцевий буфер водоспаду (v6.2.0): зсув кадрів униз, нова колонка — зверху."""
    _b = np.array(buf, dtype=np.float64, copy=True)
    _c = np.asarray(col_db, dtype=np.float64)
    _b = np.roll(_b, 1, axis=0)
    _b[0, :min(_b.shape[1], _c.size)] = _c[:_b.shape[1]]
    return _b


def gui_pdf_report(path, title, meta=None, pngs=(), notes=None):
    """PDF-звіт прямо з GUI (v6.2.0): титул + нотатки + зображення, кирилиця (DejaVu)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    _fnt = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(_fnt):
        raise RuntimeError("немає шрифту DejaVuSans для PDF")
    pdfmetrics.registerFont(TTFont("DejaVu", _fnt))
    st_h = ParagraphStyle("h", fontName="DejaVu", fontSize=15, leading=19)
    st_t = ParagraphStyle("t", fontName="DejaVu", fontSize=9, leading=12)
    import datetime as _dt
    story = [Paragraph("%s - звіт (RAW Studio v%s)" % (title, APP_VERSION), st_h),
             Spacer(1, 4),
             Paragraph("Дата: %s%s" % (_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                       (" - " + str(meta)) if meta else ""), st_t),
             Spacer(1, 10)]
    for _note in (notes or []):
        story += [Paragraph(str(_note), st_t), Spacer(1, 4)]
    for _png in pngs:
        if _png and os.path.exists(_png):
            try:
                from PIL import Image as _PImg
                _w, _h = _PImg.open(_png).size
                story += [RLImage(_png, width=160 * mm,
                                  height=160 * mm * float(_h) / float(_w)),
                          Spacer(1, 8)]
            except Exception:
                pass
    SimpleDocTemplate(path, pagesize=A4,
                      title="%s - RAW Studio" % title).build(story)
    return path


def build_pdf_report(path, params, metric=None, map_png=None, spec_png=None, meta=None,
                     spec3d=None, map3d=None, vs3d=None):
    """Звіт плану завад у PDF (v5.3.0): титулка, таблиця каналів, метрика покриття,
    спектр каналів, мапа покриття і 3D-графіки (3D-спектрограма, «Завада vs ціль»,
    3D-мапа покриття — входи-словники, рендеряться у PNG автоматично).
    Шрифт DejaVuSans (кирилиця) вбудовується у файл."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors as _rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image as RLImage)
    _fnt = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    _fntb = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(_fnt):
        raise RuntimeError("немає шрифту DejaVuSans для PDF")
    pdfmetrics.registerFont(TTFont("DejaVu", _fnt))
    if os.path.exists(_fntb):
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", _fntb))
    st_h = ParagraphStyle("h", fontName="DejaVu-Bold", fontSize=15, leading=19)
    st_t = ParagraphStyle("t", fontName="DejaVu", fontSize=9, leading=12)
    _style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, _rlc.Color(0.6, 0.6, 0.6)),
        ("BACKGROUND", (0, 0), (-1, 0), _rlc.Color(0.85, 0.88, 0.95))])
    import datetime as _dt
    story = [Paragraph("RAW Studio v%s - звіт плану завад" % APP_VERSION, st_h), Spacer(1, 4),
             Paragraph("Дата: %s%s" % (_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                       (" - " + str(meta)) if meta else ""), st_t),
             Spacer(1, 10)]
    _rows = [["№", "Увімк", "Форма", "Частота, МГц", "Ампл., кодів",
              "Ширина, мкс", "DC", "Підсилення"]]
    for _i, _p in enumerate(params):
        if not _p.get("enabled"):
            continue
        _rows.append([str(_i), "так", _p.get("kind", ""),
                      "%.4g" % (float(_p.get("freq", 0)) / 1e6),
                      "%.6g" % float(_p.get("amp", 0)), "%.6g" % float(_p.get("width_us", 0)),
                      "%.6g" % float(_p.get("dc", 0)), "%.6g" % float(_p.get("gain", 0))])
    _t1 = Table(_rows, hAlign="LEFT")
    _t1.setStyle(_style)
    story += [Paragraph("Канали плану завад", st_h), Spacer(1, 3), _t1, Spacer(1, 10)]
    if metric:
        story.append(Paragraph("Метрика покриття: %.1f%%" % float(metric.get("pct", 0.0)), st_h))
        story.append(Spacer(1, 3))
        _rows2 = [["№", "J/S, дБ"]]
        for _i, _v in enumerate(metric.get("js", [])):
            _rows2.append(["ch%d" % _i, "%+.1f" % float(_v)])
        for _lo, _hi in metric.get("holes", []):
            _rows2.append(["діра", "%.4g-%.4g МГц" % (_lo / 1e6, _hi / 1e6)])
        _t2 = Table(_rows2, hAlign="LEFT")
        _t2.setStyle(_style)
        story += [_t2, Spacer(1, 10)]
    if spec3d:
        try:
            _p53 = render_spec3d_png("/tmp/opencode/rs53_spec3d.png",
                                     spec3d.get("raw_path", ""),
                                     float(spec3d.get("fs_hz", 8e6)),
                                     nch=int(spec3d.get("nch", 1)),
                                     ch=int(spec3d.get("ch", 0)),
                                     nfft=int(spec3d.get("nfft", 1024)))
        except Exception as _e530:
            print("   DIAG render: %r" % _e530)
            _p53 = None
        if _p53:
            spec3d = dict(spec3d)
            spec3d["_png"] = _p53
    if map3d:
        try:
            _p53 = render_map3d_png("/tmp/opencode/rs53_map3d.png",
                                    float(map3d.get("eirp_dbm", 50.0)),
                                    float(map3d.get("thresh_dbm", -85.0)),
                                    float(map3d.get("f_hz", 2.4e9)),
                                    ant_name=str(map3d.get("ant_name", "")),
                                    hpb=float(map3d.get("hpb", 60.0)),
                                    fbr=float(map3d.get("fbr", 20.0)),
                                    g_rx_db=float(map3d.get("g_rx_db", 0.0)))
        except Exception as _e530:
            print("   DIAG render: %r" % _e530)
            _p53 = None
        if _p53:
            map3d = dict(map3d)
            map3d["_png"] = _p53
    if vs3d:
        try:
            _p53 = render_vs3d_png("/tmp/opencode/rs53_vs3d.png",
                                   vs3d.get("raw_path", ""),
                                   float(vs3d.get("fs_hz", 8e6)),
                                   nch=int(vs3d.get("nch", 1)),
                                   ch=int(vs3d.get("ch", 0)),
                                   jam_f_hz=float(vs3d.get("jam_f_hz", 0.0)),
                                   jam_bw_hz=float(vs3d.get("jam_bw_hz", 20e6)),
                                   js_db=float(vs3d.get("js_db", 15.0)))
        except Exception as _e530:
            print("   DIAG render: %r" % _e530)
            _p53 = None
        if _p53:
            vs3d = dict(vs3d)
            vs3d["_png"] = _p53
    for _png, _cap in ((spec_png, "Спектр каналів плану"),
                       (map_png, "Мапа покриття (полярна x FSPL)"),
                       ((spec3d or {}).get("_png"), "3D-спектрограма запису (час x частота x рівень)"),
                       ((vs3d or {}).get("_png"), "Завада vs ціль (3D): перекриття смуги"),
                       ((map3d or {}).get("_png"), "3D-мапа покриття (кут x дальність)")):
        if _png and os.path.exists(_png):
            try:
                from PIL import Image as _PImg
                _w, _h = _PImg.open(_png).size
                story += [Paragraph(_cap, st_t),
                          RLImage(_png, width=150 * mm,
                                  height=150 * mm * float(_h) / float(_w)),
                          Spacer(1, 8)]
            except Exception as _e531:
                print("   DIAG embed: %r" % _e531)
    SimpleDocTemplate(path, pagesize=A4,
                      title="RAW Studio - звіт плану завад").build(story)
    return path


def scenario_parse_phases(obj):
    """Фази сценарію (v5.2.0): [{"t": с, "ch": "0,1"|[0,1], "v": м/с, "d": км}, ...]
    або рядки "t:ch"; повертає [{"t", "ch", "v", "d"}, ...]."""
    out = []
    items = obj if isinstance(obj, list) else str(obj).split(";")
    for it in items:
        if isinstance(it, dict):
            t = float(it.get("t", 0))
            c = it.get("ch", "")
            ch = ([int(x) for x in str(c).replace(" ", "").split(",") if x != ""]
                  if isinstance(c, str) else [int(x) for x in c])
            v = float(it.get("v", 0) or 0)
            d = float(it.get("d", 0) or 0)
        elif isinstance(it, str):
            t_s, c = (it.split(":", 1) + [""])[:2]
            t = float(t_s.strip())
            ch = [int(x) for x in c.replace(" ", "").split(",") if x != ""]
            v = 0.0
            d = 0.0
        else:
            continue
        if t > 0:
            out.append({"t": t, "ch": ch, "v": v, "d": d})
    return out


def scenario_build_raw(out_path, phases, ch_params, fs, seed=42, extra=None, achx=None):
    """Генерує сценарій (v5.2.0): фази послідовно у часі, у кожній фазі активні
    лише канали ch (хвилі підсумовуються в один моно-канал). Фаза може нести
    рух: "v" (м/с) — доплерівський зсув несучих f·(1+v/c), "d" (км) —
    ослаблення дальності (амплітуда × 1/d відн. 1 км)."""
    segs, tot = [], 0
    for ph in phases:
        n_ph = int(round(float(ph["t"]) * float(fs)))
        if n_ph <= 0:
            segs.append(np.zeros(0))
            continue
        acc = np.zeros(n_ph)
        v = float(ph.get("v", 0) or 0)
        dist = float(ph.get("d", 0) or 0)
        for ci in ph["ch"]:
            if ci < 0 or ci >= len(ch_params):
                continue
            p = dict(ch_params[ci])
            p["enabled"] = True
            p.setdefault("dc", 32768.0)
            if v:
                p["freq"] = float(p.get("freq", 0.0)) * (1.0 + v / 299792458.0)
            if dist > 0:
                p["amp"] = float(p.get("amp", 0.0)) / max(dist, 1e-3)
            w = generate_raw([p], n_ph, fs, 1, seed + ci, achx=achx, extra=extra)
            acc = acc + w.astype(np.float64) - float(p.get("dc", 32768.0))
        segs.append(acc)
        tot += n_ph
    if tot * 8 > 2e9:
        raise ValueError("занадто великий сценарій: %d відліків" % tot)
    full = (np.clip(np.concatenate(segs) + 32768.0, 0, 65535).astype("<u2")
            if segs else np.zeros(0, dtype="<u2"))
    save_raw(out_path, full)
    return out_path, tot


def save_spectrum_profile(path, freqs_hz, psd_db, fc):
    """Профіль спектра запису для спектрально-формованого шуму (v4.9.0):
    JSON {freqs_hz, psd_db, fc} поруч із .iq — створюється автоматично
    кнопкою «Аналізувати» у вікні «Завада з IQ»."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"freqs_hz": [float(v) for v in freqs_hz],
                   "psd_db": [float(v) for v in psd_db],
                   "fc": float(fc)}, fh)
    return path


def load_spectrum_profile(path):
    """Читає профіль .spectrum.json -> dict(freqs_hz, psd_db (np), fc)."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    return {"freqs_hz": np.asarray(d.get("freqs_hz") or [], float),
            "psd_db": np.asarray(d.get("psd_db") or [], float),
            "fc": float(d.get("fc") or 0.0)}


def _chan_column(path, nch=1, ch=0, max_samples=1 << 22):
    """Відліки одного каналу з .raw (u2, канали черезвідліково), без DC (v5.2.0)."""
    if not (0 <= ch < nch):
        raise ValueError("канал № %d поза діапазоном 0…%d" % (ch, nch - 1))
    data = np.fromfile(path, dtype="<u2")
    if data.size == 0:
        raise ValueError("порожній файл: %s" % path)
    n = min(data.size // nch, int(max_samples))
    col = data[:n * nch].reshape(n, nch).astype(np.float64)[:, ch]
    return col - 32768.0


def _psd_mean(x, fs, nfft=4096, nseg=32):
    """Усереднений періодограмний спектр, дБ відн. повної шкали (v5.2.0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < nfft * 2:
        raise ValueError("мало відліків для спектра: %d (NFFT %d)" % (x.size, nfft))
    m = (x.size // nfft) * nfft
    segs = x[:m].reshape(-1, nfft)[:max(1, int(nseg))]
    win = np.hanning(nfft)
    cg = win.mean()
    F = np.abs(np.fft.rfft((segs - segs.mean(axis=1, keepdims=True)) * win,
                           axis=1)) / (float(nfft) * cg)
    psd = 20.0 * np.log10(F.mean(axis=0) + 1e-12)
    return np.fft.rfftfreq(nfft, d=1.0 / float(fs)), psd


def jam_overlap(target_lo_hz, target_hi_hz, jam_f_hz, jam_bw_hz, jam_shape="noise"):
    """Перекриття смуги завади зі смугою цілі (v5.2.0): dict(pct, f_lo, f_hi) —
    % смуги цілі, накритої завадою; для CW смуга = ±25 кГц, інакше ±BW/2."""
    t_lo, t_hi = float(target_lo_hz), float(target_hi_hz)
    if str(jam_shape).lower() == "cw":
        j_lo, j_hi = float(jam_f_hz) - 25e3, float(jam_f_hz) + 25e3
    else:
        j_lo, j_hi = float(jam_f_hz) - float(jam_bw_hz) / 2.0, float(jam_f_hz) + float(jam_bw_hz) / 2.0
    lo, hi = max(t_lo, j_lo), min(t_hi, j_hi)
    ov = max(0.0, hi - lo)
    pct = min(100.0 * ov / max(t_hi - t_lo, 1e-9), 100.0)
    return {"pct": pct, "f_lo": (lo if ov > 0 else None), "f_hi": (hi if ov > 0 else None)}


def ab_compare(path_a, path_b, fs_hz, nch=1, ch=0, nfft=4096):
    """Порівняння двох записів .raw (v5.2.0): усереднені спектри на спільній
    сітці, пік A, дельта B-A у смузі піка (-20 дБ). Повертає dict(freqs,
    psd_a, psd_b, f_hz, bw_hz, a_db, b_db, delta_db)."""
    fa, pa = _psd_mean(_chan_column(path_a, nch, ch), fs_hz, nfft)
    fb, pb = _psd_mean(_chan_column(path_b, nch, ch), fs_hz, nfft)
    i = int(np.argmax(pa))
    thr = pa[i] - 20.0
    j = i
    while j > 0 and pa[j] > thr:
        j -= 1
    k = i
    while k < pa.size - 1 and pa[k] > thr:
        k += 1
    prom_a = float(pa[i] - float(np.median(pa)))
    ib = int(np.argmax(pb))
    prom_b = float(pb[ib] - float(np.median(pb)))
    return {"freqs": fa, "psd_a": pa, "psd_b": pb,
            "f_hz": float(fa[i]), "bw_hz": float(fa[k] - fa[j]),
            "a_db": float(pa[j:k + 1].max()), "b_db": float(pb[j:k + 1].max()),
            "delta_db": float(pb[j:k + 1].max() - pa[j:k + 1].max()),
            "prom_a_db": prom_a, "prom_b_db": prom_b,
            "suppress_db": prom_a - prom_b}


def pri_analyze(x, fs, thr_ratio=0.15):
    """PRI/тривалість імпульсів (v5.2.0): огинаюча |x| з усередненням ~1 мкс,
    поріг thr_ratio від максимуму. Повертає dict(pri_us, pw_us, n_pulses, jit_us)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        raise ValueError("мало відліків для PRI-аналізу: %d" % x.size)
    env = np.abs(x - float(np.median(x)))
    k = max(3, 2 * int(round(float(fs) * 1e-6)) + 1)
    env = np.convolve(env, np.ones(k) / float(k), mode="same")
    thr = float(env.max()) * float(thr_ratio)
    above = env > thr
    d = np.diff(above.astype(np.int8))
    ups = np.where(d == 1)[0] + 1
    dns = np.where(d == -1)[0] + 1
    if above[0]:
        dns = dns[1:]
    if above[-1]:
        ups = ups[:-1]
    n = min(ups.size, dns.size)
    if n < 2:
        raise ValueError("імпульси не знайдено (поріг %.0f%%)" % (float(thr_ratio) * 100.0))
    ipi = np.diff(ups[:n])
    _pw_n = max(1.0, float(np.median(dns[:n] - ups[:n])) - float(k - 1))
    return {"pri_us": float(np.median(ipi)) / float(fs) * 1e6,
            "pw_us": _pw_n / float(fs) * 1e6,
            "n_pulses": int(n),
            "jit_us": float(np.std(ipi)) / float(fs) * 1e6}


def demod_audio(x, fs, mode="fm", fm_dev_hz=75000.0, decim=1):
    """Демодуляція комплексного каналу в аудіо int16 (v5.2.0): mode "fm" —
    мгновенна частота (нормування на девіацію), "am" — огинаюча. Декімація
    простим усередненням (аудіо 8-48 кГц)."""
    z = np.asarray(x)
    if not np.iscomplexobj(z):
        z = z.astype(np.complex128)
    if str(mode).lower() in ("fm", "чм"):
        ph = np.unwrap(np.angle(z))
        a = np.concatenate([np.diff(ph) * float(fs) / (2.0 * np.pi)
                            / max(float(fm_dev_hz), 1.0), [0.0]])
    else:
        a = np.abs(z)
        a = a / max(float(np.percentile(a, 99.0)), 1e-9) - 0.5
    a = np.clip(a, -1.0, 1.0)
    d = max(1, int(decim))
    if d > 1:
        a = a[: (a.size // d) * d].reshape(-1, d).mean(axis=1)
    return (a * 32000.0).astype(np.int16)


def save_wav_int16(path, samples, rate):
    """Запис моно-WAV 16 біт (v5.2.0)."""
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return path


def power_optimize(js_db, thresh_dbm, f_hz, dist_km, g_ant_db, g_rx_db,
                   jam_bw_hz, sig_bw_hz):
    """Оптимізатор потужності (v5.2.0): мінімальна потужність ПП (дБм) для
    заданого J/S на дальності: P = J/S + поріг + FSPL - G_ант - G_rx
    + 10lg(BW_завади/BW_сигналу) — корекція розподілу шумової потужності.
    Повертає dict(p_req_dbm, eirp_dbm, fspl_db, corr_db)."""
    fpl = fspl_free_db(float(f_hz), float(dist_km))
    corr = 10.0 * math.log10(max(float(jam_bw_hz), 1.0) / max(float(sig_bw_hz), 1.0))
    p = (float(js_db) + float(thresh_dbm) + fpl
         - float(g_ant_db) - float(g_rx_db) + corr)
    return {"p_req_dbm": p, "eirp_dbm": p + float(g_ant_db),
            "fspl_db": fpl, "corr_db": corr}


def coverage_map3d(eirp_dbm, thresh_dbm, f_hz, ant_name="", hpb=60.0, fbr=20.0,
                   g_rx_db=0.0, n_ang=73, n_rad=60, r_max_km=0.0):
    """3D-мапа покриття (v5.2.0): сітка запасу над порогом (дБ) кут×дальність.
    Повертає (angs_deg, rads_km, margin_db[кут, дальність])."""
    r0 = coverage_radius_km(float(eirp_dbm), float(thresh_dbm), float(f_hz),
                            g_ant_db=float(g_rx_db))
    rmax = float(r_max_km) if float(r_max_km) > 0 else max(r0 * 1.6, 1.0)
    rads = np.linspace(max(rmax / float(n_rad), 1e-3), rmax, int(n_rad))
    angs = np.linspace(0.0, 360.0, int(n_ang))
    M = np.empty((angs.size, rads.size), dtype=np.float64)
    fpl0 = 92.45 + 20.0 * math.log10(float(f_hz) / 1e9)
    for i, a in enumerate(angs):
        g = float(ant_pattern_db(ant_name, float(a), float(hpb), float(fbr)))
        M[i] = (float(eirp_dbm) + g - fpl0
                - 20.0 * np.log10(rads) - float(thresh_dbm) - float(g_rx_db))
    return angs, rads, M


SCENARIO_PRESETS = [
    {"name": "FPV/дрон-лінк 5.8 ГГц — шум, 2 фази",
     "desc": "Фаза 0: шум 20 МГц на 5800 МГц, 10 с; фаза 1: спостереження (пауза), 5 с",
     "phases": [{"t": 10, "ch": [0], "v": 0, "d": 2}, {"t": 5, "ch": [], "v": 0, "d": 0}]},
    {"name": "Захист конвою 433 МГц — рух завади",
     "desc": "Завада супроводжує колону: доплер від руху (30 м/с), дальність росте 1→2→4 км",
     "phases": [{"t": 15, "ch": [0], "v": 30, "d": 1}, {"t": 15, "ch": [0], "v": 30, "d": 2},
                {"t": 15, "ch": [0], "v": 30, "d": 4}]},
    {"name": "GNSS L1 — спуфінг по фазах",
     "desc": "Фаза 0: чистий прийом (канал 0); фаза 1: спуфер вмикається разом із прийомом",
     "phases": [{"t": 20, "ch": [0], "v": 0, "d": 0}, {"t": 30, "ch": [0, 1], "v": 0, "d": 0}]},
    {"name": "Обхід дір плану — покрокове глушіння",
     "desc": "4 фази по каналах 0→1→2→3 — «прасування» дір метрики покриття",
     "phases": [{"t": 5, "ch": [0], "v": 0, "d": 0}, {"t": 5, "ch": [1], "v": 0, "d": 0},
                {"t": 5, "ch": [2], "v": 0, "d": 0}, {"t": 5, "ch": [3], "v": 0, "d": 0}]},
]


def batch_scan(dirpath, fs_hz, nch=1, ch=0, nfft=4096, max_files=200):
    """Батч-аналіз теки .raw (v5.2.0): для кожного файлу — відліки, топ-пік
    (частота, ширина за -20 дБ, рівень над медіаною). Повертає список
    dict(name, n_samples, f_hz, width_hz, peak_rel_db[, error])."""
    if not os.path.isdir(dirpath):
        raise ValueError("тека не знайдена: %s" % dirpath)
    out = []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.lower().endswith(".raw"):
            continue
        if len(out) >= int(max_files):
            break
        pth = os.path.join(dirpath, fn)
        try:
            col = _chan_column(pth, nch, ch, max_samples=1 << 21)
            fr, psd = _psd_mean(col, fs_hz, nfft)
            i = int(np.argmax(psd))
            thr = psd[i] - 20.0
            j = i
            while j > 0 and psd[j] > thr:
                j -= 1
            k = i
            while k < psd.size - 1 and psd[k] > thr:
                k += 1
            out.append({"name": fn, "n_samples": int(col.size),
                        "f_hz": float(fr[i]), "width_hz": float(fr[k] - fr[j]),
                        "peak_rel_db": float(psd[i] - float(np.median(psd)))})
        except Exception as e:
            out.append({"name": fn, "n_samples": 0, "f_hz": 0.0,
                        "width_hz": 0.0, "peak_rel_db": 0.0, "error": str(e)})
    return out


def batch_save_csv(path, rows):
    """CSV-зведення батч-аналізу (v5.2.0), роздільник ';' (Excel-сумісний)."""
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["файл", "відліків", "пік, Гц", "ширина, Гц", "рівень, дБ", "помилка"])
        for r in rows:
            w.writerow([r.get("name", ""), r.get("n_samples", 0),
                        "%.6g" % r.get("f_hz", 0.0), "%.6g" % r.get("width_hz", 0.0),
                        "%.2f" % r.get("peak_rel_db", 0.0), r.get("error", "")])
    return path


def render_spec3d_png(out_png, raw_path, fs_hz, nch=1, ch=0, nfft=1024, overlap=0.5):
    """3D-спектрограма .raw у PNG (v5.3.0, для звіту PDF). Повертає шлях або None."""
    try:
        fr, tm, db, note = raw_spectrogram(raw_path, fs_hz, nch=nch, ch=ch,
                                           nfft=nfft, overlap=overlap)
    except Exception:
        return None
    import matplotlib as _M53
    _M53.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure as _F53
    fg = _F53(figsize=(6.6, 4.4), dpi=110)
    ax = fg.add_subplot(111, projection="3d")
    Tm, Fm = np.meshgrid(tm, fr / 1e6)
    ax.plot_surface(Tm, Fm, db, cmap="viridis",
                    rstride=max(1, db.shape[0] // 120),
                    cstride=max(1, db.shape[1] // 120), linewidth=0, antialiased=True)
    ax.set_xlabel("Час, мс")
    ax.set_ylabel("Частота, МГц")
    ax.set_zlabel("дБ")
    ax.set_title("3D-спектрограма: %s (канал %d)" % (os.path.basename(raw_path), ch), fontsize=9)
    fg.tight_layout()
    fg.savefig(out_png)
    return out_png


def render_map3d_png(out_png, eirp_dbm, thresh_dbm, f_hz, ant_name="", hpb=60.0,
                     fbr=20.0, g_rx_db=0.0):
    """3D-мапа покриття у PNG (v5.3.0, для звіту PDF). Повертає шлях."""
    angs, rads, M = coverage_map3d(float(eirp_dbm), float(thresh_dbm), float(f_hz),
                                   ant_name=ant_name, hpb=hpb, fbr=fbr, g_rx_db=g_rx_db)
    import matplotlib as _M53
    _M53.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure as _F53
    fg = _F53(figsize=(6.6, 4.6), dpi=110)
    ax = fg.add_subplot(111, projection="3d")
    A, R = np.meshgrid(angs, rads)
    ax.plot_surface(A, R, M.T, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("Кут, °")
    ax.set_ylabel("Дальність, км")
    ax.set_zlabel("Запас, дБ")
    ax.set_title("3D-мапа покриття: EIRP %.1f дБм, поріг %.0f дБм" % (float(eirp_dbm), float(thresh_dbm)),
                 fontsize=9)
    fg.tight_layout()
    fg.savefig(out_png)
    return out_png


def render_vs3d_png(out_png, raw_path, fs_hz, nch=1, ch=0, jam_f_hz=0.0, jam_bw_hz=20e6,
                    js_db=15.0):
    """«Завада vs ціль (3D)» у PNG (v5.3.0, для звіту PDF). None якщо файл не читається."""
    try:
        fr, psd = _psd_mean(_chan_column(raw_path, nch, ch, max_samples=1 << 21), fs_hz, 4096)
    except Exception:
        return None
    i = int(np.argmax(psd))
    thr = psd[i] - 20.0
    j = i
    while j > 0 and psd[j] > thr:
        j -= 1
    k = i
    while k < psd.size - 1 and psd[k] > thr:
        k += 1
    ov = jam_overlap(float(fr[j]), float(fr[k]), jam_f_hz, jam_bw_hz)
    tgt = psd - psd.max()
    jm = np.where((fr >= jam_f_hz - jam_bw_hz / 2.0) & (fr <= jam_f_hz + jam_bw_hz / 2.0),
                  float(js_db), -80.0)
    import matplotlib as _M53
    _M53.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure as _F53
    fg = _F53(figsize=(6.6, 4.6), dpi=110)
    ax = fg.add_subplot(111, projection="3d")
    Ff, Yy = np.meshgrid(fr / 1e6, np.array([0.0, 1.0]))
    ax.plot_surface(Ff, Yy, np.tile(tgt, (2, 1)), color="#3a7bd5", alpha=0.9, linewidth=0)
    Ff2, Yy2 = np.meshgrid(fr / 1e6, np.array([2.2, 3.2]))
    ax.plot_surface(Ff2, Yy2, np.tile(jm, (2, 1)), color="#d9534f", alpha=0.75, linewidth=0)
    if ov["f_lo"] is not None:
        Fo, Yo = np.meshgrid(np.array([ov["f_lo"] / 1e6, ov["f_hi"] / 1e6]),
                             np.array([1.0, 2.2]))
        Zo = np.array([[float(tgt.min()), float(tgt.min())], [float(js_db), float(js_db)]])
        ax.plot_surface(Fo, Yo, Zo, color="#f0ad4e", alpha=0.35, linewidth=0)
    ax.set_xlabel("Частота, МГц")
    ax.set_ylabel("ціль / завада")
    ax.set_zlabel("дБ")
    ax.set_title("Завада vs ціль: перекриття %.0f%%, J/S %+.0f дБ" % (ov["pct"], float(js_db)),
                 fontsize=9)
    fg.tight_layout()
    fg.savefig(out_png)
    return out_png


FIBER_MODEMS = [
    {"name": "Міні-TOSA/ROSA 155 Мбіт/с, 1310 нм (до 20 км)", "lam_nm": 1310, "rate_mbps": 155,
     "reach_km": 20, "laser": "FP", "p_tx_dbm": -12.0, "sens_dbm": -28.0, "overload_dbm": -8.0,
     "clock_mhz": 24.576, "conn": "пайка / FC", "note": "малий габарит — база дешевих ОВ-дронів"},
    {"name": "SFP 1.25G SX, 850 нм VCSEL (до 0.55 км, MM)", "lam_nm": 850, "rate_mbps": 1250,
     "reach_km": 0.55, "laser": "VCSEL", "p_tx_dbm": -7.0, "sens_dbm": -17.0, "overload_dbm": -3.0,
     "clock_mhz": 25.0, "conn": "LC", "note": "багатомодове 50/125 — короткі котушки"},
    {"name": "SFP 1.25G LX, 1310 нм (до 10 км)", "lam_nm": 1310, "rate_mbps": 1250,
     "reach_km": 10, "laser": "FP", "p_tx_dbm": -6.0, "sens_dbm": -23.0, "overload_dbm": -3.0,
     "clock_mhz": 25.0, "conn": "LC", "note": "одномодове 9/125"},
    {"name": "SFP 1.25G, 1310 нм (до 20 км)", "lam_nm": 1310, "rate_mbps": 1250,
     "reach_km": 20, "laser": "FP", "p_tx_dbm": -4.0, "sens_dbm": -26.0, "overload_dbm": -3.0,
     "clock_mhz": 25.0, "conn": "LC", "note": "типовий для котушок 10–15 км"},
    {"name": "SFP 1.25G, 1550 нм DFB (до 40–80 км)", "lam_nm": 1550, "rate_mbps": 1250,
     "reach_km": 40, "laser": "DFB", "p_tx_dbm": 2.0, "sens_dbm": -30.0, "overload_dbm": -3.0,
     "clock_mhz": 25.0, "conn": "LC", "note": "запас по дальності, чутливий до відбиттів"},
    {"name": "BiDi WDM 1310/1550 (одноволоконний, до 20 км)", "lam_nm": 1310, "rate_mbps": 1250,
     "reach_km": 20, "laser": "FP/DFB", "p_tx_dbm": -3.0, "sens_dbm": -26.0, "overload_dbm": -3.0,
     "clock_mhz": 25.0, "conn": "SC (WDM)", "note": "один волокно-канал: відео + телеметрія — домінує в ОВ-FPV"},
    {"name": "SFP+ 10G LR, 1310 нм (до 10 км)", "lam_nm": 1310, "rate_mbps": 10312,
     "reach_km": 10, "laser": "DFB", "p_tx_dbm": -2.0, "sens_dbm": -15.0, "overload_dbm": -1.0,
     "clock_mhz": 156.25, "conn": "LC", "note": "HD/4K відео без стиснення"},
    {"name": "PON-клас 1490/1310 (2.5G вниз / 1.25G вгору)", "lam_nm": 1490, "rate_mbps": 2500,
     "reach_km": 20, "laser": "DFB", "p_tx_dbm": 2.0, "sens_dbm": -27.0, "overload_dbm": -3.0,
     "clock_mhz": 125.0, "conn": "SC/APC", "note": "штатна оптика GPON у ролі тракту"},
    {"name": "Аналогове FM-відео по ВОЛЗ, 1310 нм", "lam_nm": 1310, "rate_mbps": 0,
     "reach_km": 5, "laser": "FP", "p_tx_dbm": -10.0, "sens_dbm": -20.0, "overload_dbm": -5.0,
     "clock_mhz": 0.0, "conn": "FC/ST", "note": "без CDR — вплив лише засліпленням"},
]

FIBER_CLOCKS = [
    ("Кварц МК польотного контролера (F405/F722 HSE)", 8.0),
    ("Кварц МК контролера (H7/G4 HSE)", 12.0),
    ("Кварц GPS-модуля", 26.0),
    ("Опорний генератор ОВ-модему 1.25G (CDR)", 25.0),
    ("Опорний генератор ОВ-модему 10G (CDR)", 156.25),
    ("Тактування відеосенсора (MCLK)", 24.0),
    ("Кварц аналогового відеопередавача", 25.0),
    ("ШІМ ESC (електромеханічний резонанс)", 0.024),
]


def fiber_bleach(p_inj_dbm, div_mrad, dist_m, core_um=9.0, ins_loss_db=2.0):
    """Інжекція світла у волокно ОВ-тракту (v5.4.0): діаметр плями на торці
    (мкм) = дистанція × розбіжність; втрати геометричного узгодження
    20lg(пляма/серцевина), якщо пляма більша за серцевину 9 мкм.
    Повертає dict(spot_um, geom_db, p_rx_dbm) — потужність на вході приймача."""
    spot_um = max(0.0, float(dist_m) * float(div_mrad) * 1000.0)
    if spot_um > float(core_um):
        geom_db = 20.0 * math.log10(spot_um / float(core_um))
    else:
        geom_db = 0.0
    p_rx = float(p_inj_dbm) - float(ins_loss_db) - geom_db
    return {"spot_um": spot_um, "geom_db": geom_db, "p_rx_dbm": p_rx}


def fiber_bleach_needed(dist_m, target_dbm, div_mrad, core_um=9.0, ins_loss_db=2.0):
    """Потрібна потужність інжектора (дБм), щоб на вході приймача було target_dbm (v5.4.0)."""
    spot_um = max(0.0, float(dist_m) * float(div_mrad) * 1000.0)
    geom_db = 20.0 * math.log10(spot_um / float(core_um)) if spot_um > float(core_um) else 0.0
    return float(target_dbm) + float(ins_loss_db) + geom_db


def fiber_status(p_rx_dbm, sens_dbm, overload_dbm, margin_db=6.0):
    """Стан приймача ОВ-модему під інжекцією (v5.4.0): (код, опис) —
    2 = засліплення (>= перевантаження), 1 = зрив CDR/деградація (>= чутливість + запас),
    0 = без впливу."""
    p = float(p_rx_dbm)
    if p >= float(overload_dbm):
        return (2, "ЗАСЛІПЛЕННЯ — фотодіод у насиченні, тракту немає")
    if p >= float(sens_dbm) + float(margin_db):
        return (1, "ЗРив CDR / підміна потоку — інжекція домінує над сигналом")
    if p >= float(sens_dbm):
        return (1, "Біля чутливості — можливі поодинокі зриви")
    return (0, "Без впливу (нижче чутливості)")


def mcu_resonance(f_res_hz, dist_m, u_req_v=0.5, l_eff_m=0.1, g_ant_db=0.0):
    """ЕМ-вплив на резонансній частоті контролера (v5.4.0): потрібна напруженість
    поля E = U_req / l_eff; EIRP з E на дистанції d (E = sqrt(30·P)/d).
    Повертає dict(e_req_vm, eirp_dbm)."""
    e_req = float(u_req_v) / max(float(l_eff_m), 1e-3)
    p_w = (e_req * float(dist_m)) ** 2 / 30.0
    return {"e_req_vm": e_req,
            "eirp_dbm": 10.0 * math.log10(max(p_w, 1e-12) * 1000.0) - float(g_ant_db)}


def fiber_modems_csv(path):
    """CSV-довідник класів ОВ-модемів (v5.4.0), роздільник ';' (Excel-сумісний)."""
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["модем", "лямбда, нм", "швидкість, Мбіт/с", "дальність, км", "лазер",
                    "P_перед, дБм", "чутливість, дБм", "перевантаження, дБм",
                    "опорний генератор, МГц", "з'єднання", "примітка"])
        for m in FIBER_MODEMS:
            w.writerow([m["name"], m["lam_nm"], m["rate_mbps"], m["reach_km"], m["laser"],
                        m["p_tx_dbm"], m["sens_dbm"], m["overload_dbm"], m["clock_mhz"],
                        m["conn"], m["note"]])
    return path


# ---- v5.5.0: ШІ-підбір тракту і симулятор завади ОВ-дрону ----
FIBER_ALPHA = {850: 2.5, 1310: 0.35, 1550: 0.20}


def fiber_chain_select(freq_hz, eirp_req_dbm, cable_db=1.5, amp_max_db=40.0):
    """ШІ-підбір повного тракту завади (v5.5.0): пари «SDR (передавач) +
    антена» з довідників, що покривають частоту впливу, із потрібним
    підсиленням: amp = EIRP − ПП_SDR − G_ант + втрати КСВ + кабель.
    Ранжування — за мінімальним підсиленням. Типові орієнтири."""
    out = []
    for sname, sd in (SDR_PRESETS or {}).items():
        tx = sd.get("tx_dbm")
        if tx is None:
            continue
        f_lo = float(sd.get("f_lo", 0.0) or 0.0)
        f_hi = float(sd.get("f_hi", 0.0) or 0.0)
        if f_hi > 0 and not (f_lo <= freq_hz <= f_hi):
            continue
        for aname, a in (JQ_ANTENNAS or {}).items():
            af_lo = float(a.get("f_lo", 0.0) or 0.0)
            af_hi = float(a.get("f_hi", 0.0) or 0.0)
            if af_hi > 0 and not (af_lo <= freq_hz <= af_hi):
                continue
            v = _vswr_at(a.get("vswr") or [], freq_hz)
            vswr_l = float(_vswr_loss_db(v)) if v else 0.0
            amp = (float(eirp_req_dbm) - float(tx) - float(a.get("g", 0.0))
                   + vswr_l + float(cable_db))
            if amp > float(amp_max_db):
                continue
            out.append(dict(sdr=sname, tx_dbm=float(tx), ant=aname,
                            g_dbi=float(a.get("g", 0.0)), vswr_loss_db=vswr_l,
                            cable_db=float(cable_db), amp_req_db=amp,
                            eirp_req_dbm=float(eirp_req_dbm)))
    out.sort(key=lambda x: x["amp_req_db"])
    return out


def fiber_sim_resonance(eirp_dbm, e_req_vm):
    """Макс. дистанція резонансного впливу на контролер, м (v5.5.0) —
    обернена задача mcu_resonance: d = sqrt(30·P_W)/E."""
    p_w = 10.0 ** (float(eirp_dbm) / 10.0) / 1000.0
    return math.sqrt(30.0 * p_w) / float(e_req_vm)


def fiber_sim_modem(modem, p_inj_dbm, div_mrad, ins_loss_db, d_list,
                    d_drone_m=0.0, jsr_req_db=15.0, core_um=9.0):
    """Симулятор завади ОВ-тракту (v5.5.0): для модема по дистанціях
    інжектора — рівень світла на фотодіоді, статус (без впливу / зрив
    CDR / засліплення), J/S проти сигналу дрона (затухання волокна за λ
    · дистанція дрона), граничні дистанції впливу. rows =
    (дистанція_м, P_фд_дБм, код, текст, JSR_дБ). Стендова модель."""
    lam = str(modem.get("lam_nm", "1550"))
    try:
        lam_key = min(FIBER_ALPHA, key=lambda k: abs(k - float(lam.split("/")[0].strip())))
    except Exception:
        lam_key = 1550
    alpha = FIBER_ALPHA[lam_key]
    sens = float(modem.get("sens_dbm", -30.0))
    overload = float(modem.get("overload_dbm", -5.0))
    p_sig = float(modem.get("p_tx_dbm", 0.0)) - alpha * float(d_drone_m) / 1000.0
    rows = []
    d_blind = d_cdr = d_eff = None
    for d in sorted(float(x) for x in d_list):
        r = fiber_bleach(float(p_inj_dbm), float(div_mrad), d,
                         core_um=core_um, ins_loss_db=ins_loss_db)
        code, txt = fiber_status(r["p_rx_dbm"], sens, overload)
        jsr = r["p_rx_dbm"] - p_sig
        rows.append((d, r["p_rx_dbm"], code, txt, jsr))
        if code == 2:
            d_blind = d
        if code == 1:
            d_cdr = d
        if code >= 1 or jsr >= float(jsr_req_db):
            d_eff = d
    return dict(p_sig_dbm=p_sig, alpha_db_km=alpha, lam_nm=lam_key, rows=rows,
                d_blind_m=d_blind, d_cdr_m=d_cdr, d_eff_m=d_eff,
                jsr_req_db=float(jsr_req_db), overload_dbm=overload,
                sens_dbm=sens)


# ==================== v5.8.0: СПЕКТРОАНАЛІЗАТОРИ — SCPI API ====================
SA_HINT = ("SCPI over TCP: R&S FSV/FSW, Siglent SSA/SHA/SVA, Rigol RSA/DSA, Keysight CXA/N90x0, "
           "TinySA — порт 5025 (дефолт) або 5555; *IDN?, :FREQ:CENT/SPAN, :BAND:RES, "
           ":DISP:WIND:TRAC:Y:RLEV, :TRAC? TRACE1 (ASCII) / :TRACe:DATA? (бінарний блок)")


def scpi_parse_idn(s):
    """Розбір відповіді *IDN? → dict(vendor, model, serial, fw)."""
    _p = [t.strip().strip('"') for t in str(s).strip().split(",")]
    return {"vendor": _p[0] if len(_p) > 0 else "", "model": _p[1] if len(_p) > 1 else "",
            "serial": _p[2] if len(_p) > 2 else "", "fw": _p[3] if len(_p) > 3 else ""}


def scpi_parse_trace(data):
    """Траса SCPI → масив дБм: ASCII (кома/пробіл/новий рядок) або бінарний
    блок IEEE definite-length «#<n><len><int16 LE>» (стиль Rigol)."""
    import re as _re58
    if isinstance(data, (bytes, bytearray)) and data[:1] == b"#":
        _nd = int(data[1:2])
        _n = int(data[2:2 + _nd])
        _pl = bytes(data[2 + _nd:2 + _nd + _n])
        if len(_pl) % 2 == 0:
            try:
                return np.frombuffer(_pl, dtype="<i2").astype(np.float64)
            except Exception:
                pass
        return scpi_parse_trace(_pl.decode("ascii", "ignore"))
    _txt = data.decode("ascii", "ignore") if isinstance(data, (bytes, bytearray)) else str(data)
    _v = [float(t) for t in _re58.split(r"[,\s]+", _txt.strip()) if t]
    return np.asarray(_v, dtype=np.float64)


def trace_freqs(f_center_hz, span_hz, n):
    """Частоти точок траси: рівномірна сітка центр ± span/2."""
    return np.linspace(float(f_center_hz) - float(span_hz) / 2.0,
                       float(f_center_hz) + float(span_hz) / 2.0, max(int(n), 1))


def trace_top_peaks(freqs_hz, trace_dbm, n_peaks=8, min_db=12.0, min_gap_hz=None):
    """Топ-N піків траси (дБм): локальні максимуми на min_db над медіаною,
    розділені min_gap (дефолт — 5 кроків сітки), сортовані за спаданням."""
    _fr = np.asarray(freqs_hz, dtype=np.float64)
    _tr = np.asarray(trace_dbm, dtype=np.float64)
    _n = min(_fr.size, _tr.size)
    if _n < 8:
        return []
    _fr, _tr = _fr[:_n], _tr[:_n]
    if min_gap_hz is None:
        min_gap_hz = max(float(_fr[1] - _fr[0]) * 5.0, 1e3)
    _floor = float(np.median(_tr))
    _thr = _floor + float(min_db)
    _cand = [(float(_tr[i]), i) for i in range(1, _n - 1)
             if _tr[i] >= _thr and _tr[i] > _tr[i - 1] and _tr[i] >= _tr[i + 1]]
    _cand.sort(reverse=True)
    _pk = []
    for _v, _i in _cand:
        if any(abs(_fr[_i] - _p["f_hz"]) < float(min_gap_hz) for _p in _pk):
            continue
        _pk.append({"f_hz": float(_fr[_i]), "dbm": _v})
        if len(_pk) >= int(n_peaks):
            break
    return _pk


def scpi_sweep_plan(f_start_hz, f_stop_hz, rbw_hz):
    """План сканування: кількість точок і крок за смуги та RBW."""
    _span = float(f_stop_hz) - float(f_start_hz)
    _rbw = max(float(rbw_hz), 0.1)
    _n = max(2, int(np.ceil(_span / _rbw)) + 1)
    return {"span_hz": _span, "rbw_hz": _rbw, "n_points": _n,
            "step_hz": _span / max(_n - 1, 1)}


class ScpiClient(object):
    """Мінімальний SCPI-клієнт над TCP (VISA не потрібен): connect/query/cmd,
    identify (*IDN?), setup (центр/span/RBW/рівень), fetch_trace (ASCII/блок)."""

    def __init__(self, host, port=5025, timeout=2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock = None

    def connect(self):
        import socket as _sk58
        self.close()
        self.sock = _sk58.create_connection((self.host, self.port), timeout=self.timeout)
        return True

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def query(self, cmd):
        if self.sock is None:
            self.connect()
        self.sock.sendall((cmd.strip() + "\n").encode("ascii"))
        _buf = bytearray()
        while b"\n" not in _buf:
            _d = self.sock.recv(65536)
            if not _d:
                break
            _buf += _d
            if len(_buf) > (8 << 20):
                raise ValueError("SCPI-відповідь занадто велика (>8 МБ)")
        return bytes(_buf).decode("ascii", "ignore").strip()

    def cmd(self, cmd):
        if self.sock is None:
            self.connect()
        self.sock.sendall((cmd.strip() + "\n").encode("ascii"))

    def identify(self):
        return scpi_parse_idn(self.query("*IDN?"))

    def setup(self, f_center_hz, span_hz, rbw_hz, ref_dbm):
        self.cmd(":FREQ:CENT %.0f" % float(f_center_hz))
        self.cmd(":FREQ:SPAN %.0f" % float(span_hz))
        self.cmd(":BAND:RES %.0f" % float(rbw_hz))
        self.cmd(":DISP:WIND:TRAC:Y:RLEV %.1f" % float(ref_dbm))

    def fetch_trace(self, trace="TRACE1", cmd=None):
        _r = self.query(cmd if cmd else (":TRAC? %s" % trace))
        if not _r:
            _r = self.query(":TRACe:DATA? %s" % trace)
        return scpi_parse_trace(_r)


# ==================== v5.9.0: РЕЦЕПТИ SDR (ПАРК ПЛАТФОРМ) ====================
def recipe_plan(target_key, models, pa_dbm=20.0, amp_db=0.0, ant_name="Штирьова 0.25λ",
                dist_km=5.0, dj_km=1.0, custom_hz=None):
    """Рецепти для парку платформ (v5.9.0): ціль (AI_TARGETS) × моделі SDR_PRESETS.
    Вхід: ціль зі списку AI_TARGETS, список моделей, потужність ПП (дБм), підсилювач (дБ),
    антена з JQ_ANTENNAS, дальність розкриття і дистанція завади (км), опційна частота
    (Гц) — замінник несучої. Вихід: список dict(model, verdict, note, kind_name, f_hz,
    bw_hz, js_req, p_req, d_max, sdr_d_max, min_us, tx_dbm, g_total) — відсортований:
    OK (повне покриття) → СМУГА (звужено до 0.4·fs) → ПОЗА (частота поза платою) →
    ПРИЙОМ (без TX); у групі — за радіусом придушення спадно. Стендовий інструмент:
    апаратурою не керує."""
    ant = JQ_ANTENNAS.get(ant_name) or JQ_ANTENNAS["Штирьова 0.25λ"]
    t = AI_TARGETS.get(target_key)
    f = float(custom_hz) if custom_hz and float(custom_hz) > 0 else (t["freq"] if t else None)
    out = []
    for m in models:
        d = SDR_PRESETS.get(m) or {}
        if not d:
            out.append({"model": m, "verdict": "ПОЗА", "note": "власна плата — параметри не задано",
                        "kind_name": "", "f_hz": f, "bw_hz": None, "js_req": None, "p_req": None,
                        "d_max": None, "sdr_d_max": None, "min_us": None, "tx_dbm": None,
                        "g_total": None})
            continue
        tx = d.get("tx_dbm")
        fs = float(d["fs_hz"]); lo = float(d["f_lo"]); hi = float(d["f_hi"])
        min_us = 4.0 / fs * 1e6
        if tx is None:
            out.append({"model": m, "verdict": "ПРИЙОМ",
                        "note": "приймальна платформа (без TX) — заваду не створює",
                        "kind_name": "", "f_hz": f, "bw_hz": None, "js_req": None, "p_req": None,
                        "d_max": None, "sdr_d_max": None, "min_us": min_us, "tx_dbm": None,
                        "g_total": None})
            continue
        if f is None:
            out.append({"model": m, "verdict": "ПОЗА", "note": "ціль не обрано",
                        "kind_name": "", "f_hz": None, "bw_hz": None, "js_req": None, "p_req": None,
                        "d_max": None, "sdr_d_max": None, "min_us": min_us, "tx_dbm": tx,
                        "g_total": None})
            continue
        vswr_f = _vswr_at(ant["vswr"], f) if ant["vswr"] else 1.0
        g_total = float(amp_db) + float(ant["g"]) - _vswr_loss_db(vswr_f)
        pa_eff = float(pa_dbm) + g_total
        if not (lo <= f <= hi):
            out.append({"model": m, "verdict": "ПОЗА",
                        "note": "частота %.6g МГц поза діапазоном плати %.6g–%.6g МГц"
                                % (f / 1e6, lo / 1e6, hi / 1e6),
                        "kind_name": "", "f_hz": f, "bw_hz": None, "js_req": None, "p_req": None,
                        "d_max": None, "sdr_d_max": None, "min_us": min_us, "tx_dbm": tx,
                        "g_total": g_total})
            continue
        bw_t = float(t["bw"])
        bk = ai_recommend_jam(target_key, pa_eff, dist_km, dj_km, fs, f_hz=f)
        bw_final = float(bk["bw"])
        verdict = "OK"
        note_parts = []
        if bw_t > 0.4 * fs:
            verdict = "СМУГА"
            note_parts.append("смуга звужена до 0.4·fs (%.6g МГц)" % (bw_final / 1e6))
        if ant["vswr"] and not (ant["f_lo"] <= f <= ant["f_hi"]):
            note_parts.append("частота поза діапазоном антени")
        if not note_parts:
            note_parts.append("повне покриття: несуча в діапазоні, смуга ≤ 0.4·fs")
        bk2 = ai_recommend_jam(target_key, float(tx) + g_total, dist_km, dj_km, fs, f_hz=f)
        sdm = float(bk2["d_max"])
        if pa_eff < float(bk["p_req"]):
            note_parts.append("потужності тракту бракує: потрібно %.1f дБм" % float(bk["p_req"]))
        out.append({"model": m, "verdict": verdict, "note": "; ".join(note_parts),
                    "kind_name": bk["kind_name"], "f_hz": float(bk["f_hz"]), "bw_hz": bw_final,
                    "js_req": float(bk["js_req"]), "p_req": float(bk["p_req"]),
                    "d_max": float(bk["d_max"]), "sdr_d_max": sdm, "min_us": min_us,
                    "tx_dbm": float(tx), "g_total": g_total})
    order = {"OK": 0, "СМУГА": 1, "ПОЗА": 2, "ПРИЙОМ": 3}
    out.sort(key=lambda r: (order.get(r["verdict"], 4),
                            -r["sdr_d_max"] if r["sdr_d_max"] is not None else 0.0))
    return out

# ==================== v6.0.0: ІНСТРУМЕНТАРІЙ — журнал · сигнатури · пресети · порівняння · панорама · Монте-Карло · імпорт IQ · CLI ====================
import csv as _csv6
import json as _json6
import random as _random6
import tempfile as _tempfile6
import time as _time6

FPV_STD = {
    "PAL": dict(lines=625, line_us=64.0, sync_us=4.7, active_us=51.2),
    "NTSC": dict(lines=525, line_us=63.5, sync_us=4.7, active_us=52.2),
}

# --- пресети SCPI-приладів (v6.0.0): команди зйомки траси ---
SCPI_PRESETS = {
    "Універсальний (SCPI)": dict(trace_cmd=None, note=":TRAC? TRACE1 → фолбек :TRACe:DATA?"),
    "R&S FSV / FSW": dict(trace_cmd=":TRAC? TRACE1", note="стандартний SCPI-запит"),
    "Siglent SSA/SHA/SVA": dict(trace_cmd=":TRAC:DATA? TRACE1", note="формула Siglent"),
    "Rigol RSA/DSA": dict(trace_cmd=":TRAC:DATA?", note="бінарний блок int16 LE"),
    "Keysight CXA/N90x0": dict(trace_cmd=":TRAC? TRACE1", note="стандартний SCPI"),
    "TinySA": dict(trace_cmd="data 0", note="власний протокол tinySA"),
}

# --- журнал сесій (SQLite, v6.0.0) ---
def hist_db_path():
    """Шлях журналу сесій: ~/.raw_studio_history.db."""
    return os.path.join(os.path.expanduser("~"), ".raw_studio_history.db")

def hist_log(module, summary, data=None):
    """Запис у журнал сесій (v6.0.0): events(ts, module, summary, data-JSON).
    Помилки журналу не ламають роботу; повертає True/False."""
    try:
        import sqlite3
        _con = sqlite3.connect(hist_db_path())
        _con.execute("CREATE TABLE IF NOT EXISTS events "
                     "(ts TEXT, module TEXT, summary TEXT, data TEXT)")
        _con.execute("INSERT INTO events VALUES (?,?,?,?)",
                     (_time6.strftime("%Y-%m-%d %H:%M:%S"), module, summary,
                      _json6.dumps(data, ensure_ascii=False) if data else None))
        _con.commit()
        _con.close()
        return True
    except Exception:
        return False

def hist_rows(limit=200):
    try:
        import sqlite3
        _con = sqlite3.connect(hist_db_path())
        _con.execute("CREATE TABLE IF NOT EXISTS events "
                     "(ts TEXT, module TEXT, summary TEXT, data TEXT)")
        _cur = _con.execute("SELECT ts, module, summary, data FROM events "
                            "ORDER BY rowid DESC LIMIT ?", (int(limit),))
        _rows = _cur.fetchall()
        _con.close()
        return _rows
    except Exception:
        return []

def hist_export_csv(path, limit=1000):
    _rows = hist_rows(limit)
    with open(path, "w", encoding="utf-8-sig", newline="") as _fh:
        _w = _csv6.writer(_fh, delimiter=";")
        _w.writerow(["Час", "Модуль", "Подія", "Дані (JSON)"])
        for _r in _rows:
            _w.writerow(list(_r))
    return len(_rows)

# --- сигнатури сигналів (v6.0.0) ---
SIGNATURES = {
    "FPV (аналоговий ЧМ)": dict(w_lo=4e6, w_hi=10e6, sig_key="CVBS"),
    "Wi-Fi / OFDM 20 МГц": dict(w_lo=16e6, w_hi=24e6, sig_key="WiFi (OFDM)"),
    "LoRa / вузький чирп": dict(w_lo=5e4, w_hi=5e5, sig_key="LoRa"),
    "ГНСС ШПС (вузький)": dict(w_lo=1.5e6, w_hi=5e6, sig_key="GPS L1"),
}

def classify_width(width_hz):
    """Ширина піка −3 дБ (Гц) → (сигнатура, sig_key для Кроку 3) (v6.0.0)."""
    _w = float(width_hz)
    for _n, _s in SIGNATURES.items():
        if _s["w_lo"] <= _w <= _s["w_hi"]:
            return _n, _s["sig_key"]
    return "Невідома", None

# --- тема (v6.0.0) ---
def theme_pref_path():
    return os.path.join(os.path.expanduser("~"), ".raw_studio.json")

def theme_pref_get():
    try:
        with open(theme_pref_path()) as _fh:
            return _json6.load(_fh).get("theme")
    except Exception:
        return None

def theme_pref_set(name):
    try:
        _cfg = {}
        try:
            with open(theme_pref_path()) as _fh:
                _cfg = _json6.load(_fh)
        except Exception:
            pass
        _cfg["theme"] = name
        with open(theme_pref_path(), "w") as _fh:
            _json6.dump(_cfg, _fh)
        return True
    except Exception:
        return False

# --- спектр/порівняння (v6.0.0) ---
def spectrum_db(x, nfft=1 << 16):
    """Вікно Ганна → rfft → дБ (v6.0.0). Вихід: (sp_db, n) або (None, None)."""
    _x = np.asarray(x, dtype=np.float64)
    _n = min(int(nfft), _x.size)
    if _n < 4:
        return None, None
    _seg = _x[:_n] - float(np.mean(_x[:_n]))
    _sp = np.abs(np.fft.rfft(_seg * np.hanning(_n))) / (_n / 4.0)
    return 20.0 * np.log10(np.maximum(_sp, 1e-12)), _n

def compare_raws(path_a, path_b, nch=DEFAULT_NCH, nfft=1 << 16):
    """Порівняння двох записів .raw (v6.0.0): спектри перших каналів (u2, DC 32768),
    різниця до−після; топ-8 піків запису A з Δ придушення (макс у вікні ±3 біни).
    Вихід: dict(n_a, n_b, peaks, delta_mean_db)."""
    _da = np.fromfile(path_a, dtype="<u2")
    _db = np.fromfile(path_b, dtype="<u2")
    _ca = _da[0::max(1, int(nch))][:int(nfft)].astype(np.float64) - 32768.0
    _cb = _db[0::max(1, int(nch))][:int(nfft)].astype(np.float64) - 32768.0
    if _ca.size < 4 or _cb.size < 4:
        raise ValueError("записи надто короткі")
    _sa, _n = spectrum_db(_ca, nfft)
    _sb, _ = spectrum_db(_cb, nfft)
    if _sa is None or _sb is None:
        raise ValueError("порожні спектри")
    _idx = np.argsort(_sa)[-40:]
    _idx = _idx[np.argsort(_sa[_idx])][::-1][:8]
    _pks = []
    for _i in sorted(_idx):
        _lo = max(0, int(_i) - 3)
        _hi = min(_sa.size, int(_i) + 4)
        _pks.append({"bin": int(_i),
                     "before_db": float(np.max(_sa[_lo:_hi])),
                     "after_db": float(np.max(_sb[_lo:_hi])),
                     "delta_db": float(np.max(_sa[_lo:_hi]) - np.max(_sb[_lo:_hi]))})
    return {"n_a": int(_ca.size), "n_b": int(_cb.size), "peaks": _pks,
            "delta_mean_db": float(np.mean([_p["delta_db"] for _p in _pks]))}

def panorama_from_raw(path, nch=DEFAULT_NCH, nfft=1 << 16):
    """Панорама запису .raw (v6.0.0): спектри каналів у блоках FREQ_BLOCKS,
    склеєні по частотах; піки — trace_top_peaks. Вихід: dict(freqs_hz, dbm, peaks)."""
    _data = np.fromfile(path, dtype="<u2")
    if _data.size < 8:
        raise ValueError("запис надто короткий")
    _freqs, _vals = [], []
    _nff = min(int(nfft), max(8, _data.size // max(1, int(nch))))
    for _ci in range(min(int(nch), len(FREQ_BLOCKS))):
        _lo, _hi = FREQ_BLOCKS[_ci]
        _ch = _data[_ci::max(1, int(nch))][:_nff].astype(np.float64) - 32768.0
        _sp, _ = spectrum_db(_ch, _nff)
        if _sp is None:
            continue
        _fc = (_lo + _hi) / 2.0
        _half = (_hi - _lo) / 2.0
        _fb = np.linspace(_fc - _half, _fc + _half, _sp.size)
        _freqs.append(_fb)
        _vals.append(_sp)
    if not _freqs:
        raise ValueError("немає спектрів каналів")
    _f = np.concatenate(_freqs)
    _v = np.concatenate(_vals)
    _ord = np.argsort(_f)
    _f, _v = _f[_ord], _v[_ord]
    return {"freqs_hz": _f, "dbm": _v, "peaks": trace_top_peaks(_f, _v)}

# --- Монте-Карло follow-me (v6.0.0) ---
def monte_carlo_follow(hop_rate_hz, retune_ms, p_detect=0.9, hop_channels=1,
                       p_sigma=0.1, retune_sigma_ms=20.0, trials=10000, seed=42):
    """Монте-Карло follow-me (v6.0.0): розигриш p_виявлення ~ N(p, σ_p) (обмежено
    0..1) і часу перестроювання ~ N(retune, σ_rt) (обмежено 0+); вихід: dict(
    share_p5, share_p50, share_p95, trials, shares) — частка хопів, які завада
    встигає накрити, з довірчим коридором."""
    _rng = _random6.Random(int(seed))
    _sh = []
    for _ in range(int(trials)):
        _pd = min(1.0, max(0.0, _rng.gauss(float(p_detect), float(p_sigma))))
        _rt = max(0.0, _rng.gauss(float(retune_ms), float(retune_sigma_ms)))
        _r = follow_me_sim(float(hop_rate_hz), _rt, _pd, float(hop_channels))
        _sh.append(float(_r["p_cover"]) if isinstance(_r, dict) else float(_r))
    _sh.sort()
    def _pct(_p):
        _i = min(len(_sh) - 1, max(0, int(_p * (len(_sh) - 1))))
        return float(_sh[_i])
    return {"share_p5": _pct(0.05), "share_p50": _pct(0.50), "share_p95": _pct(0.95),
            "trials": int(trials), "shares": _sh}

# --- імпорт IQ → растр (v6.0.0) ---
def fpv_import_preview(path, fs_hz, std_key="PAL", f_hz=5800e6, dev_hz=2e6,
                       nch=1, cols=64, rows_px=48):
    """Імпорт IQ-запису (.raw/.u2, DC 32768, інтерлівінг nch) → FM-демодуляція
    (video_raster_demod v5.7.0: Гільберт → миттєва частота → люмінансність) →
    растр (rows_px, cols) 0..1 за 50 мс запису."""
    _std = FPV_STD.get(std_key) or FPV_STD["PAL"]
    _data = np.fromfile(path, dtype="<u2")
    _ch = _data[0::max(1, int(nch))].astype(np.float64) - 32768.0
    if _ch.size < 1000:
        raise ValueError("запис надто короткий")
    _n = min(_ch.size, int(float(fs_hz) * 0.05))
    _ras = video_raster_demod(_ch[:_n], float(fs_hz), _std, float(f_hz), float(dev_hz),
                              cols=int(cols), rows=int(rows_px))
    _ras = np.clip(np.asarray(_ras, dtype=np.float64), 0.0, 1.0)
    return {"raster": _ras, "samples": int(_n), "std": std_key}

# --- CLI (v6.0.0) ---
def cli_recipes(target, park=None, pa_dbm=20.0, dist_km=5.0, dj_km=1.0,
                ant_name="Штирьова 0.25λ", freq_mhz=None, out=None):
    """CLI-режим рецептів (v6.0.0): recipe_plan без GUI; друк таблиці або CSV (--out)."""
    _models = [m.strip() for m in (park or "").split(",") if m.strip()]
    _rows = recipe_plan(target, _models or list(SDR_PRESETS), pa_dbm=pa_dbm,
                        dist_km=dist_km, dj_km=dj_km, ant_name=ant_name,
                        custom_hz=(float(freq_mhz) * 1e6 if freq_mhz else None))
    if out:
        with open(out, "w", encoding="utf-8-sig", newline="") as _fh:
            _w = _csv6.writer(_fh, delimiter=";")
            _w.writerow(["Ціль", "Платформа", "Вердикт", "Тип завади", "Частота МГц",
                         "Смуга МГц", "J/S дБ", "Потрібно дБм", "TX дБм", "Радіус км", "Примітка"])
            for _r in _rows:
                _w.writerow([target, _r["model"], _r["verdict"], _r["kind_name"],
                             "" if _r["f_hz"] is None else "%.6g" % (_r["f_hz"] / 1e6),
                             "" if _r["bw_hz"] is None else "%.6g" % (_r["bw_hz"] / 1e6),
                             "" if _r["js_req"] is None else "%.0f" % _r["js_req"],
                             "" if _r["p_req"] is None else "%.1f" % _r["p_req"],
                             "" if _r["tx_dbm"] is None else "%.0f" % _r["tx_dbm"],
                             "" if _r["sdr_d_max"] is None else "%.6g" % _r["sdr_d_max"],
                             _r["note"]])
        print("рецепти: %s (%d рядків)" % (out, len(_rows)))
    else:
        print("%-24s %-8s %-18s %10s %10s %10s  %s" %
              ("Платформа", "Вердикт", "Тип", "Частота", "Смуга", "Радіус", "Примітка"))
        for _r in _rows:
            print("%-24s %-8s %-18s %10s %10s %10s  %s" % (
                _r["model"][:24], _r["verdict"], _r["kind_name"][:18],
                "—" if _r["f_hz"] is None else "%.6g" % (_r["f_hz"] / 1e6),
                "—" if _r["bw_hz"] is None else "%.6g" % (_r["bw_hz"] / 1e6),
                "—" if _r["sdr_d_max"] is None else "%.6g" % _r["sdr_d_max"], _r["note"]))
    return _rows

def cli_melt(spec, distances=(0.5, 1, 2, 3, 5, 7, 10)):
    """CLI-режим лабораторії плавлення (v6.0.0): spec = 'модем,f_МГц,c,j,nf,bw_МГц,
    rb,pkt' — таблиця PRR по дистанціях (ФСМ Фріса на частоті f)."""
    _p = [s.strip() for s in spec.split(",")]
    if len(_p) != 8:
        raise ValueError("очікується 8 значень: модем,f_МГц,c,j,nf,bw_МГц,rb,pkt")
    _mod = _p[0]
    _f, _c0, _j0, _nf = float(_p[1]), float(_p[2]), float(_p[3]), float(_p[4])
    _bw, _rb, _pk = float(_p[5]), float(_p[6]), float(_p[7])
    print("%-8s %10s %10s %10s" % ("км", "c, дБм", "j, дБм", "PRR"))
    for _d in distances:
        _l = fspl_db(_d, _f)
        _pr = melt_prr(_mod, _c0 - _l, _j0 - _l, _nf, _bw * 1e6, _rb, _pk)
        _pr = _pr.get("prr", 0.0) if isinstance(_pr, dict) else float(_pr)
        print("%-8s %10.1f %10.1f %10.4f" % (_d, _c0 - _l, _j0 - _l, _pr))

# ==================== v6.1.0: ЗАВАДИ + ПЕРЕХОПЛЕННЯ КЕРУВАННЯ ====================

def burst_scan(path, fs_hz, nch=1, ch=0, thresh_db=12.0, max_samples=2000000):
    """Виявлення пакетів у записі .raw (v6.1.0): RMS-огортаюча (вікно ~0.5 мс),
    поріг «дБ над медіаною», пакети (час, пік), медіанний період, скважність."""
    chs, _n = load_raw(path, nch=nch)
    s = chs[ch][:max_samples]
    s = s - float(s.mean())
    win = max(16, int(fs_hz // 2000))
    r = np.sqrt(np.convolve(s ** 2, np.ones(win) / win, mode="same"))
    pdb = 20.0 * np.log10(np.maximum(r, 1e-9))
    floor = float(np.median(pdb))
    act = pdb >= floor + float(thresh_db)
    idx = np.where(np.diff(act.astype(np.int8)) == 1)[0]
    end = np.where(np.diff(act.astype(np.int8)) == -1)[0]
    t0s = (idx + 1) / fs_hz
    t1s = (end + 1) / fs_hz
    m = min(len(t0s), len(t1s))
    t0s, t1s = t0s[:m], t1s[:m]
    peaks = [float(pdb[max(0, int(a * fs_hz)):max(int(a * fs_hz) + 1, int(b * fs_hz))].max())
             for a, b in zip(t0s, t1s)]
    per = float(np.median(np.diff(t0s))) if m > 2 else None
    return {"floor_db": round(floor, 1), "thresh_db": round(floor + float(thresh_db), 1),
            "bursts": m, "t0_s": [round(float(x), 5) for x in t0s],
            "dur_s": [round(float(b - a), 5) for a, b in zip(t0s, t1s)],
            "peak_db": [round(x, 1) for x in peaks], "period_s": per,
            "duty": round(float(act.mean()), 3)}


def burst_centers_hz(path, fs_hz, scan, nch=1, ch=0, max_bursts=32):
    """Центри частот пакетів (v6.1.0): FFT-пік по кожному пакету скану."""
    chs, _n = load_raw(path, nch=nch)
    s = chs[ch]
    out = []
    for t0 in scan["t0_s"][:max_bursts]:
        i0 = int(t0 * fs_hz)
        seg = s[i0:i0 + max(512, int(0.002 * fs_hz))]
        if len(seg) > 32:
            f, a = amplitude_spectrum(seg, fs_hz)
            out.append(float(f[int(np.argmax(a))]))
    return out


def jam_burst_codes(fs_hz, f_hz, rate_hz, duty, amp, dur_s, base=32768):
    """Імпульсна завада «пакети» (v6.1.0): шумові викиди з темпом rate_hz/с
    і скважністю duty навколо несучої f_hz. Коди u2 (центр base)."""
    n = max(1, int(dur_s * fs_hz))
    t = np.arange(n) / fs_hz
    env = (((t * rate_hz) % 1.0) < duty).astype(np.float64)
    noise = np.random.normal(0.0, 1.0, n)
    ph = 2.0 * np.pi * f_hz * t
    sig = base + amp * env * noise * np.cos(ph)
    return sig.clip(0, 65535).astype("<u2")


def reactive_jam_model(power_db, dt_s, thresh_db, react_ms):
    """Модель реактивної завади (v6.1.0): опромінення лише над порогом енергії.
    power_db — таймлайн дБ, dt_s — крок. detect_s — час до першого спрацювання,
    on_air_frac — частка часу опромінення, save_frac — економія потужності."""
    pw = np.asarray(power_db, dtype=np.float64)
    above = pw >= float(thresh_db)
    react_s = float(react_ms) / 1000.0
    on_air = 0.0
    detect_s = None
    run = 0
    for i in range(len(pw) + 1):
        if i < len(pw) and above[i]:
            run += 1
            continue
        if run:
            dur = run * dt_s
            if dur >= react_s:
                if detect_s is None:
                    detect_s = round((i - run) * dt_s + react_s, 4)
                on_air += dur - react_s
        run = 0
    total = len(pw) * dt_s
    frac = on_air / total if total > 0 else 0.0
    return {"detect_s": detect_s, "on_air_frac": round(min(frac, 1.0), 4),
            "save_frac": round(max(1.0 - frac, 0.0), 4), "total_s": round(total, 3)}


def jam_time_profile(scenes, t_total):
    """Покриття часового профілю завади (v6.1.0): сцени (t0, t1, скважність 0..1)
    → профіль покриття, частка покриття, вікна тиші."""
    n = max(1, int(t_total))
    cover = np.zeros(n)
    for sc in scenes:
        t0, t1, duty = sc
        i0, i1 = max(0, int(t0)), min(n, int(t1))
        if i1 > i0:
            cover[i0:i1] = np.maximum(cover[i0:i1], float(duty))
    gaps = []
    run = 0
    for i in range(n + 1):
        if i < n and cover[i] <= 0:
            run += 1
            continue
        if run:
            gaps.append({"t0": i - run, "t1": i, "dur_s": run})
        run = 0
    return {"cover": [round(float(x), 3) for x in cover],
            "covered_frac": round(float((cover > 0).mean()), 3), "gaps": gaps, "n": n}


def per_vs_js(mod_key, js_db_list, pkt_bits=256):
    """PER-криві vs J/S (v6.1.0): SNR = -J/S у приймачі, BER через mod_ber,
    PER = 1-(1-BER)^pkt_bits. Наближення АБГШ; js_db_list — J/S у дБ (0..)."""
    out = []
    for js in js_db_list:
        ber = mod_ber(mod_key, -float(js))
        per = 1.0 - (1.0 - min(max(ber, 0.0), 1.0)) ** int(pkt_bits)
        out.append({"js_db": round(float(js), 1), "ber": ber, "per": per})
    return out


def multiband_plan(targets, n_ch=4):
    """Багаточастотний план опромінення (v6.1.0): targets — [(band, f_lo, f_hi, prio 1..3)],
    розподіл смуг по каналах приладу за пріоритетом."""
    ts = sorted(targets, key=lambda x: (x[3], str(x[0])))
    alloc = [{"channel": (i % n_ch) + 1, "band": b, "f_lo": flo, "f_hi": fhi,
              "f_mid": (flo + fhi) / 2.0, "prio": pr}
             for i, (b, flo, fhi, pr) in enumerate(ts)]
    per_ch = {}
    for a in alloc:
        per_ch[a["channel"]] = per_ch.get(a["channel"], 0) + 1
    return {"alloc": alloc, "channels_used": len(per_ch),
            "overloaded": len(ts) > int(n_ch), "per_ch": per_ch}


def js_curve_db(p_j_dbm, p_t_dbm, d_t_m, d_max_m=2000.0, g_diff_db=0.0,
                js_thresh_db=-6.0, n=64):
    """Крива J/S(дистанція завади) (v6.1.0): вільний простір, та сама смуга,
    J/S = Pj - Pt + ΔG + 20·lg(d_цілі/d_завади). Радіус придушення — де J/S = поріг."""
    dj = np.linspace(max(1.0, float(d_max_m) / 64.0), float(d_max_m), int(n))
    c = float(p_j_dbm) - float(p_t_dbm) + float(g_diff_db) + 20.0 * np.log10(max(float(d_t_m), 1.0))
    js = c - 20.0 * np.log10(np.maximum(dj, 1.0))
    d_thresh = 10.0 ** ((c - float(js_thresh_db)) / 20.0) if c > float(js_thresh_db) else 0.0
    return {"d_m": [round(float(x), 1) for x in dj],
            "js_db": [round(float(x), 1) for x in js],
            "d_max_m": round(float(d_thresh), 1)}


def anti_jam_features(centers_hz, duty):
    """Ознаки стійкості лінка до завад (v6.1.0): стрибки частоти, паузи (LBT),
    скважність → порада щодо тактики опромінення."""
    uq = sorted(set(centers_hz))
    hops = len(uq) >= 3
    d = float(duty)
    lbt = 0.05 <= d <= 0.6
    advice = []
    if hops:
        advice.append("стрибки частоти (%d каналів) — спот малоефективний; свіп або реактивна" % len(uq))
    else:
        advice.append("фіксований канал — спот-завада ефективна")
    if lbt:
        advice.append("паузи (скважність %.0f%%) — синхронізуйте опромінення з пакетами" % (d * 100))
    if d > 0.9:
        advice.append("майже безперервний сигнал — barrage-шум по смузі")
    return {"каналів": len(uq), "стрибки": hops, "lbt": lbt, "скважність": round(d, 3),
            "порада": " · ".join(advice)}


def jam_tactics_compare(pp_db):
    """Порівняння тактик опромінення (v6.1.0): рейтинг CW/чирп/шум/пакети за
    спектральною щільністю цілі (дБ) — правила: CW при ПП < 17, чирп 17-27, шум >= 27."""
    p = float(pp_db)
    rate = [("CW (спот)", 3 if p < 17 else 1),
            ("Чирп (свіп)", 2 if 17 <= p < 27 else 1),
            ("Шум (barrage)", 3 if p >= 27 else 2),
            ("Пакети (імпульсна)", 2)]
    rate.sort(key=lambda x: -x[1])
    return {"рейтинг": rate, "рекомендовано": rate[0][0],
            "правило": "CW при ПП < 17 дБ · чирп 17-27 · шум >= 27 (DSSS-ГНСС)"}


def crc8_dvbs2(data):
    """CRC-8/DVB-S2 (поліном 0xD5) — CRC фреймів CRSF (v6.1.0)."""
    crc = 0
    for b in bytes(data):
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def sbus_frame(ch, flags=0):
    """Синтез фрейму SBUS (25 байт: 0x0F, 16 каналів × 11 біт, flags, 0x00) (v6.1.0)."""
    bits = 0
    for k in range(16):
        bits |= (int(ch[k]) & 0x7FF) << (11 * k)
    return b"\x0f" + bits.to_bytes(22, "little") + bytes([int(flags) & 0x1F, 0x00])


def ibus_frame(ch):
    """Синтез фрейму iBUS (32 байти: 0x20, 14 каналів × 16 біт, сума-контроль) (v6.1.0)."""
    body = bytearray(32)
    body[0] = 0x20
    for k in range(14):
        v = int(ch[k]) & 0xFFFF
        body[2 + 2 * k] = v & 0xFF
        body[3 + 2 * k] = (v >> 8) & 0xFF
    s = sum(body[:30]) & 0xFFFF
    body[30], body[31] = s & 0xFF, (s >> 8) & 0xFF
    return bytes(body)


def crsf_frame(typ, payload=b""):
    """Синтез фрейму CRSF: [0xC8, len, type, payload..., crc8-DVB-S2] (v6.1.0)."""
    body = bytes([0xC8, len(payload) + 2, int(typ) & 0xFF]) + bytes(payload)
    return body + bytes([crc8_dvbs2(body[2:])])


def msp_frame(cmd, payload=b""):
    """Синтез MSP v1: $M> len cmd payload crc8-xor (v6.1.0)."""
    head = bytes([len(payload), int(cmd) & 0xFF])
    crc = 0
    for x in head + bytes(payload):
        crc ^= x
    return b"$M>" + head + bytes(payload) + bytes([crc])


def _unpack11(payload, n):
    """Розпакування n каналів × 11 біт (little-endian біт-пакет) (v6.1.0)."""
    bits = int.from_bytes(bytes(payload), "little")
    return [(bits >> (11 * k)) & 0x7FF for k in range(n)]


def rc_extract_frames(data, proto):
    """Витяг фреймів RC-протоколу з байтового потоку (v6.1.0):
    SBUS — 0x0F + 24; iBUS — 0x20 + сума; CRSF — addr/len/type/crc8.
    Повертає [(зсув, bytes)]."""
    data = bytes(data)
    out = []
    if proto == "SBUS":
        i = 0
        while i <= len(data) - 25:
            if data[i] == 0x0F and data[i + 24] == 0x00:
                out.append((i, data[i:i + 25]))
                i += 25
            else:
                i += 1
    elif proto == "IBUS":
        i = 0
        while i <= len(data) - 32:
            if data[i] == 0x20:
                s = sum(data[i:i + 30]) & 0xFFFF
                if data[i + 30] == (s & 0xFF) and data[i + 31] == ((s >> 8) & 0xFF):
                    out.append((i, data[i:i + 32]))
                    i += 32
                else:
                    i += 1
            else:
                i += 1
    elif proto == "CRSF":
        i = 0
        while i <= len(data) - 4:
            ln = data[i + 1]
            if (2 <= ln <= 62 and i + 2 + ln <= len(data)
                    and data[i + 1 + ln] == crc8_dvbs2(data[i + 2:i + 1 + ln])):
                out.append((i, data[i:i + 2 + ln]))
                i += 2 + ln
            else:
                i += 1
    return out


def rc_channels(proto, frame):
    """Канальні значення з фрейму RC (v6.1.0): SBUS/CRSF — 11 біт (172..1811),
    iBUS — 14 каналів × 16 біт."""
    frame = bytes(frame)
    if proto == "SBUS" and len(frame) >= 25:
        return _unpack11(frame[1:23], 16)
    if proto == "IBUS" and len(frame) >= 32:
        return [int.from_bytes(frame[2 + 2 * k:4 + 2 * k], "little") for k in range(14)]
    if proto == "CRSF" and len(frame) >= 6:
        ln = frame[1]
        pl = frame[3:1 + ln]
        return _unpack11(pl[:22], min(16, len(pl) * 8 // 11))
    return []


def rc_telemetry_crsf(frame):
    """Телеметрія CRSF з фрейму (v6.1.0): 0x08 батарея (В, А, мА·год, %),
    0x02 GPS (lat/lon 1e-7, км/год, курс, висота м, супутники)."""
    frame = bytes(frame)
    if len(frame) < 5:
        return None
    typ = frame[2]
    pl = frame[3:1 + frame[1]]
    if typ == 0x08 and len(pl) >= 8:
        return {"тип": "батарея", "В": int.from_bytes(pl[0:2], "big") / 10.0,
                "А": int.from_bytes(pl[2:4], "big") / 100.0,
                "мА·год": int.from_bytes(pl[4:7], "big"), "відсоток": pl[7]}
    if typ == 0x02 and len(pl) >= 15:
        lat = int.from_bytes(pl[0:4], "big", signed=True) / 1e7
        lon = int.from_bytes(pl[4:8], "big", signed=True) / 1e7
        return {"тип": "GPS", "lat": round(lat, 7), "lon": round(lon, 7),
                "км/год": int.from_bytes(pl[8:10], "big") / 100.0,
                "курс": int.from_bytes(pl[10:12], "big") / 100.0,
                "висота_м": int.from_bytes(pl[12:14], "big", signed=True) - 1000,
                "супутники": pl[14]}
    return {"тип": "0x%02X" % typ, "довжина": len(pl)}


def rc_telemetry_msp(frame):
    """MSP v1 з фрейму (v6.1.0): $M> len cmd payload crc8-xor → dict або None."""
    frame = bytes(frame)
    if len(frame) < 6 or frame[:3] != b"$M>":
        return None
    ln = frame[3]
    if len(frame) < 6 + ln:
        return None
    crc = 0
    for x in frame[3:5 + ln]:
        crc ^= x
    if frame[5 + ln] != crc:
        return None
    return {"cmd": frame[4] & 0xFF, "довжина": ln, "payload_hex": frame[5:5 + ln].hex()}


def failsafe_analyze(ch_timeline, dt_s):
    """Аналіз failsafe за таймлайном каналів (v6.1.0): замороження значень
    (нульові зміни між відліками), відліки біля центра 992 (SBUS/CRSF)."""
    arr = np.asarray(ch_timeline, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] < 3:
        return {"каналів": arr.shape[1], "замороження": False, "заморожених_відліків": 0,
                "замороження_с": 0.0, "відліків_біля_центра": 0}
    diff0 = np.abs(np.diff(arr, axis=0)).max(axis=1)
    frozen = diff0 == 0.0
    fr = int(frozen.sum())
    center_hits = int((np.abs(arr - 992.0) < 2).all(axis=1).sum())
    return {"каналів": arr.shape[1], "заморожених_відліків": fr,
            "замороження_с": round(fr * float(dt_s), 3),
            "замороження": fr >= 3, "відліків_біля_центра": center_hits}


def hop_pattern_detect(freqs_hz):
    """Хоп-послідовність (v6.1.0): канали, стрибки, детект циклу патерну,
    середній двелл, прогноз наступних каналів."""
    f = list(freqs_hz)
    if len(f) < 3:
        return {"каналів": len(set(f)), "канали_Гц": [round(float(x), 0) for x in sorted(set(f))],
                "стрибків": 0, "цикл": None, "середній_двелл": None, "прогноз": []}
    uq = sorted(set(f))
    seq = [uq.index(x) for x in f]
    cycle = None
    for p in range(2, len(seq) // 2 + 1):
        if seq[:p] * (len(seq) // p) == seq[:p * (len(seq) // p)]:
            cycle = p
            break
    jumps = int(sum(1 for i in range(1, len(f)) if f[i] != f[i - 1]))
    ch_idx = [0] + [i for i in range(1, len(f)) if f[i] != f[i - 1]]
    dwells = np.diff(ch_idx + [len(f)])
    pred = [uq[seq[(len(seq) + k) % cycle]] for k in range(3)] if cycle else []
    return {"каналів": len(uq), "канали_Гц": [round(float(x), 0) for x in uq],
            "стрибків": jumps, "цикл": cycle,
            "середній_двелл": round(float(np.mean(dwells)), 1) if len(dwells) else None,
            "прогноз": [round(float(x), 0) for x in pred]}


def link_quality_timeline(t0_s, powers_db, period_s, t_total):
    """Якість лінка (v6.1.0): слоти періоду vs виявлені пакети → LQ %,
    втрати (часи), RSSI-таймлайн."""
    slots = np.arange(0.0, float(t_total), float(period_s))
    times = list(t0_s)
    pw = list(powers_db)
    used = [False] * len(times)
    rssi = []
    losses = []
    for s in slots:
        bi, bd = None, None
        for i, t in enumerate(times):
            if used[i]:
                continue
            d = abs(t - s)
            if d <= period_s / 2.0 and (bd is None or d < bd):
                bd, bi = d, i
        if bi is None:
            losses.append(round(float(s), 3))
            rssi.append(None)
        else:
            used[bi] = True
            rssi.append(round(float(pw[bi]), 1))
    ok = len(slots) - len(losses)
    return {"очікувано": len(slots), "прийнято": ok,
            "LQ_%": round(100.0 * ok / max(len(slots), 1), 1),
            "втрати_с": losses, "rssi_дБ": rssi}


def amplitude_df(powers_db, az_ant=None):
    """Амплітудний пеленг (v6.1.0): азимут джерела з потужностей каналів (дБ)
    за векторною сумою; az_ant — азимути антен, за замовчуванням 0/90/180/270°."""
    p = np.asarray(powers_db, dtype=np.float64)
    if az_ant is None:
        az_ant = [0.0, 90.0, 180.0, 270.0]
    lin = 10.0 ** (p / 20.0)
    az_arr = np.asarray(list(az_ant)[:len(lin)], dtype=np.float64)
    x = float(np.sum(lin * np.cos(np.radians(az_arr))))
    y = float(np.sum(lin * np.sin(np.radians(az_arr))))
    az = float(np.degrees(np.arctan2(y, x))) % 360.0
    spread = float(p.max() - p.min())
    conf = "висока" if spread > 6.0 else ("середня" if spread > 2.0 else "низька")
    return {"азимут_°": round(az, 1), "розкид_дБ": round(spread, 1), "впевненість": conf}


def activity_stats(power_db, dt_s, thresh_db):
    """Енергодетектор активності (v6.1.0): скважність, темп пакетів/с,
    клас сигналу: безперервний (відео/CW) / пакетний (керування, телеметрія) / шум."""
    pw = np.asarray(power_db, dtype=np.float64)
    act = pw >= float(thresh_db)
    duty = float(act.mean())
    edges = np.where(np.diff(act.astype(np.int8)) == 1)[0]
    dur = max(len(pw) * float(dt_s), 1e-9)
    rate = len(edges) / dur
    if duty > 0.8:
        cls = "безперервний (відео/CW)"
    elif rate > 20:
        cls = "пакетний високий темп (керування RC)"
    elif rate > 1:
        cls = "пакетний низький темп (телеметрія)"
    elif duty > 0.005:
        cls = "рідкісні пакети"
    else:
        cls = "шум/тиша"
    return {"скважність": round(duty, 3), "пакетів_с": round(rate, 1), "клас": cls}


def gnss_spoof_scan(power_db, dt_s):
    """Ознаки GPS-спуфінгу в записі (v6.1.0): аномальний розкид потужності,
    стрибок (вмикання спуфера), підозріло гладка огортаюча."""
    pw = np.asarray(power_db, dtype=np.float64)
    fl = float(np.percentile(pw, 10))
    hi = float(np.percentile(pw, 95))
    margin = hi - fl
    d = np.diff(pw)
    jump = float(np.max(np.abs(d))) if len(d) else 0.0
    ramp = float(np.mean(np.abs(d[:max(1, len(d) // 4)]))) if len(d) else 0.0
    signs = []
    if margin > 30.0:
        signs.append("аномальний розкид %.0f дБ (спуфер сильний)" % margin)
    if jump > 12.0:
        signs.append("стрибок потужності %.0f дБ (вмикання)" % jump)
    if margin > 20.0 and ramp < 0.05:
        signs.append("підозріло гладка огортаюча")
    return {"підлога_дБ": round(fl, 1), "пік_дБ": round(hi, 1),
            "стрибок_дБ": round(jump, 1), "ознаки": signs,
            "вердикт": "підозра на спуфінг" if signs else "ознак спуфінгу не виявлено"}


def cli_jamprofile(spec):
    """CLI: профіль завади за часом — "t0:t1:скважність;…,всього_с" (v6.1.0)."""
    _sc, _tot = spec.rsplit(",", 1)
    scenes = []
    for part in _sc.split(";"):
        t0, t1, d = part.split(":")
        scenes.append((float(t0), float(t1), float(d)))
    _p = jam_time_profile(scenes, float(_tot))
    print("Покриття: %.0f%%" % (_p["covered_frac"] * 100))
    for g in _p["gaps"]:
        print("  вікно тиші: %d-%d с (%d с)" % (g["t0"], g["t1"], g["dur_s"]))


def cli_percurve(spec):
    """CLI: PER-криві — "модем[,pkt_біт]" — BER/PER по J/S 0..33 дБ (v6.1.0)."""
    parts = spec.split(",")
    mod = parts[0]
    bits = int(parts[1]) if len(parts) > 1 else 512
    rows = per_vs_js(mod, list(range(0, 34, 3)), bits)
    print("Модем %s, пакет %d біт:" % (mod, bits))
    print("  J/S, дБ | BER      | PER")
    for r in rows:
        print("  %7.1f | %.2e | %.3f" % (r["js_db"], r["ber"], r["per"]))


def cli_rcscan(spec):
    """CLI: пакети в записі — "шлях,fs_Гц[,nch,ch]" — період, скважність, піки (v6.1.0)."""
    parts = spec.split(",")
    path = parts[0]
    fs = float(parts[1])
    nch = int(parts[2]) if len(parts) > 2 else 1
    ch = int(parts[3]) if len(parts) > 3 else 0
    r = burst_scan(path, fs, nch=nch, ch=ch)
    print("Пакетів: %d · період %s с · скважність %.3f" % (r["bursts"], r["period_s"], r["duty"]))
    print("підлога %s дБ · пік %s дБ" % (r["floor_db"], max(r["peak_db"]) if r["peak_db"] else "-"))



# ==================== v5.7.0: FPV-ВІДЕО — ПЕРЕХОПЛЕННЯ ТА ІНЖЕКЦІЯ ====================
FPV_BANDS = {
    "5.8 · Band A": [5865, 5845, 5825, 5805, 5785, 5765, 5745, 5725],
    "5.8 · Band B": [5733, 5752, 5771, 5790, 5809, 5828, 5847, 5866],
    "5.8 · Band E": [5705, 5685, 5665, 5645, 5885, 5905, 5925, 5945],
    "5.8 · Band F": [5740, 5760, 5780, 5800, 5820, 5840, 5860, 5880],
    "5.8 · Raceband": [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917],
}
VIDEO_STD = {
    "PAL": {"line_us": 64.0, "sync_us": 4.7, "active_us": 52.0, "lines": 625, "fps": 25.0},
    "NTSC": {"line_us": 63.56, "sync_us": 4.7, "active_us": 52.6, "lines": 525, "fps": 29.97},
}


FPV_FS_LADDER = [64.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 2500.0,
                 4000.0, 5000.0, 6000.0, 8000.0, 12000.0]


def fpv_fs_auto(f_hz, dev_hz, guard_mhz=2.0):
    """Авто-підбір fs під тип сигналу «аналогове FM-відео»: перше значення
    стандартного ряду АЦП, що перекриває Nyquist 2·(f + девіація) + запас
    на фронти синхроімпульсів відео (2 МГц). Для FPV 5.8 ГГц це завжди
    12 ГГц (пряма оцифровка 6 ГГц); для випробувальних несучих — нижчі
    щаблі ряду. Понад ряд — ValueError (зменшіть девіацію)."""
    _need = 2.0 * (float(f_hz) + abs(float(dev_hz))) + 1e6 * float(guard_mhz)
    for _v in FPV_FS_LADDER:
        if _v * 1e6 >= _need:
            return _v * 1e6
    raise ValueError("ряд АЦП не перекриває f + девіацію (потрібно %.0f МГц) — зменшіть девіацію"
                     % (_need / 1e6))


def fs_auto_plan(freqs_hz, guard_hz=2.0e6, min_width_s=None):
    """Авто-підбір fs для плану каналів Генератора (вся програма): найменший
    щабель ряду АЦП (FPV_FS_LADDER), що перекриває Найквіста 2·f_max + запас
    (2 МГц, або 1/мін. тривалість імпульсу, якщо більша). Порожній план
    (лише шум/нічого) → 64 МГц. Понад ряд — ValueError."""
    _fr = [float(_f) for _f in (freqs_hz or []) if _f and float(_f) > 0.0]
    _need = 2.0 * (max(_fr) if _fr else 0.0) + float(guard_hz)
    if min_width_s is not None and float(min_width_s) > 0.0:
        _need = max(_need, 1.0 / float(min_width_s) + float(guard_hz))
    for _v in FPV_FS_LADDER:
        if _v * 1e6 >= _need:
            return _v * 1e6
    raise ValueError("ряд АЦП не перекриває план (потрібно %.0f МГц)"
                     % (_need / 1e6))


def fpv_channels(band):
    """Таблиця каналів FPV-діапазону: [(мітка «К1»..«К8», f_Гц), ...]."""
    return [("К%d" % (_i + 1), float(_f) * 1e6)
            for _i, _f in enumerate(FPV_BANDS.get(band, []))]


def fpv_scan_table(band, peaks_hz, level_dbm=-60.0, tol_mhz=8.0):
    """Скан діапазону: канал активний («ВІДЕО»), якщо пік спектра в межах
    tol МГц; інакше «шум». Повертає [(мітка, f_МГц, рівень_dBm|None, статус)]."""
    _pk = [float(_f) / 1e6 for _f in (peaks_hz or [])]
    _rows = []
    for _lab, _f in fpv_channels(band):
        _fm = _f / 1e6
        if any(abs(_fm - _p) <= float(tol_mhz) for _p in _pk):
            _rows.append((_lab, _fm, float(level_dbm), "ВІДЕО"))
        else:
            _rows.append((_lab, _fm, None, "шум"))
    return _rows


def video_quality_est(snr_db, thr_db=12.0, hi_db=26.0):
    """Якість аналогового відео за SNR (FM-поріг 12 дБ, насичення 26 дБ):
    (відсоток, вердикт). Типові орієнтири."""
    _s = float(snr_db)
    if _s < thr_db:
        return 0, "немає прийому (сніг)"
    _q = int(round(100.0 * min(1.0, (_s - thr_db) / max(hi_db - thr_db, 1e-9))))
    _v = "відмінна" if _q >= 85 else ("задовільна" if _q >= 50 else "слабка, сніжинки")
    return _q, _v


def injection_eirp(p_rx_dbm, margin_db, d_km, f_hz, g_ant_db=0.0):
    """EIRP інжектора, щоб на приймачі перебити відео дрона:
    EIRP = p_rx(відео) + запас (захоплення FM) + FSPL(d) − G_ант.
    FSPL(дБ) = 32.44 + 20·lg(d_км) + 20·lg(f_МГц). Типові орієнтири."""
    import math as _m57
    _fspl = 32.44 + 20.0 * _m57.log10(max(float(d_km), 1e-4)) \
        + 20.0 * _m57.log10(max(float(f_hz), 1.0) / 1e6)
    _e = float(p_rx_dbm) + float(margin_db) + _fspl - float(g_ant_db)
    return {"fspl_db": _fspl, "eirp_dbm": _e,
            "verdict": "перебиває відео (захоплення приймача)" if float(margin_db) >= 6.0
            else "співканальна конкуренція — рвана картинка"}


def capture_effect(margin_db):
    """FM-захоплення приймача: сильніший сигнал перебиває слабший.
    Запас ≥ 6 дБ — захоплення; 0..6 — конкуренція (рвана картинка); < 0 — без впливу."""
    _m = float(margin_db)
    if _m >= 6.0:
        return {"state": "захоплення", "note": "приймач показує інжектоване відео"}
    if _m >= 0.0:
        return {"state": "конкуренція", "note": "рвана картинка: суміш двох відео"}
    return {"state": "без впливу", "note": "інжектований сигнал слабший за відео дрона"}


def video_pattern(name, w=64, h=48, img_path=None):
    """Тестовий кадр для інжекції: 2D люмінансність 0..1.
    «Кольорові смуги» / «Градієнт» / «Смуги-застереження» / зображення (PIL)."""
    _a = np.zeros((int(h), int(w)), dtype=np.float64)
    if name == "Кольорові смуги":
        _bars = np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.12, 0.0])
        _a[:, :] = _bars[np.arange(int(w)) * len(_bars) // int(w)][None, :]
    elif name == "Градієнт":
        _a[:, :] = np.linspace(0.0, 1.0, int(w))[None, :]
    elif name == "Смуги-застереження":
        _r = np.arange(int(h))[:, None]
        _c = np.arange(int(w))[None, :]
        _a = (((_r + _c) // 4) % 2).astype(np.float64)
        _a[:3, :] = 1.0
        _a[-3:, :] = 1.0
        _a[:, :3] = 1.0
        _a[:, -3:] = 1.0
    else:
        try:
            from PIL import Image as _I57
            _im = _I57.open(img_path).convert("L").resize((int(w), int(h)))
            _a = np.asarray(_im, dtype=np.float64) / 255.0
        except Exception:
            _bars = np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.12, 0.0])
            _a[:, :] = _bars[np.arange(int(w)) * len(_bars) // int(w)][None, :]
    return _a


def video_fm_modulate(pattern, std, f_hz, dev_hz, fs, n_frames, amp=0.6):
    """Аналогове FM-відео (стендова модель перехоплення/інжекції): рядок =
    синхроімпульс (m = −1 протягом sync_us) + активна частина
    (m = −0.7 + 1.4·люмінансність), ЧМ на несучій f з девіацією dev.
    Повертає float32-сигнал; ліміт 40 млн відліків (пам'ять)."""
    _lu = float(std["line_us"])
    _su = float(std["sync_us"])
    _au = float(std["active_us"])
    _lines = int(std["lines"])
    _pat = np.asarray(pattern, dtype=np.float64)
    _ph, _pw = _pat.shape
    _n = int(round(float(fs) * float(n_frames) * _lines * _lu / 1e6))
    if _n > 40000000:
        raise ValueError("Занадто довго: %d відл. — зменшіть кадри або fs (ліміт 40 млн)" % _n)
    if float(f_hz) + float(dev_hz) >= float(fs) / 2.0:
        raise ValueError("fs/2 має перевищувати f + девіацію")
    _m = np.zeros(_n, dtype=np.float64)
    _n_line = float(fs) * _lu / 1e6
    _n_sync = int(round(float(fs) * _su / 1e6))
    _n_act = int(round(float(fs) * _au / 1e6))
    for _k in range(_lines * int(n_frames)):
        _i0 = int(round(_k * _n_line))
        _i1 = _i0 + _n_sync
        if _i1 >= _n:
            break
        _m[_i0:min(_i1, _n)] = -1.0
        _row = _pat[(_k % _lines) * _ph // _lines]
        _c = np.linspace(_i1, _i0 + _n_sync + _n_act, int(_pw), endpoint=False).astype(int)
        _cc = _c[(_c >= 0) & (_c < _n)]
        _m[_cc] = (-0.7 + 1.4 * _row)[:_cc.size]
    _integ = np.cumsum(_m) / float(fs)
    _sig = np.empty(_n, dtype=np.float32)
    _ch = 4000000
    for _s in range(0, _n, _ch):
        _e = min(_s + _ch, _n)
        _t = np.arange(_s, _e, dtype=np.float64) / float(fs)
        _sig[_s:_e] = (float(amp) * np.sin(2.0 * np.pi * (float(f_hz) * _t
                       + float(dev_hz) * _integ[_s:_e]))).astype(np.float32)
    return _sig


def video_raster_demod(sig, fs, std, f_hz, dev_hz, cols=64, rows=48):
    """Зворотне розкладання FM-відео на растер (стендова модель): аналітичний
    сигнал (Гільберт через FFT, чанками) → миттєва частота → люмінансність
    активної частини рядків → масив (rows, cols), усереднений по кадрах."""
    _lu = float(std["line_us"])
    _su = float(std["sync_us"])
    _au = float(std["active_us"])
    _lines = int(std["lines"])
    _x = np.asarray(sig, dtype=np.float64)
    _n_line = float(fs) * _lu / 1e6
    _nfr = int(round(float(fs) * _lines * _lu / 1e6))
    _frames = max(1, int(_x.size // _nfr))
    _use = _frames * _nfr
    _parts = []
    _CH = 1 << 20
    for _s in range(0, _use, _CH):
        _e = min(_s + _CH, _use)
        _seg = _x[_s:_e]
        _X = np.fft.fft(_seg)
        _h = np.zeros(_seg.size, dtype=np.complex128)
        _h[0] = 1.0
        _half = _seg.size // 2
        _h[1:_half] = 2.0
        if _seg.size % 2 == 0:
            _h[_half] = 1.0
        _phi = np.unwrap(np.angle(np.fft.ifft(_X * _h)))
        _parts.append((np.diff(_phi) * float(fs) / (2.0 * np.pi)).astype(np.float32))
    _inst = np.concatenate(_parts) if _parts else np.zeros(0, dtype=np.float32)
    _m = (_inst.astype(np.float64) - float(f_hz)) / float(dev_hz)
    _n_sync = int(round(float(fs) * _su / 1e6))
    _n_act = int(round(float(fs) * _au / 1e6))
    _raster = np.zeros((int(rows), int(cols)), dtype=np.float64)
    _cnt = np.zeros(int(rows), dtype=np.float64)
    for _fr in range(_frames):
        for _k in range(_lines):
            _i1 = _fr * _nfr + int(round(_k * _n_line)) + _n_sync
            _i2 = _i1 + _n_act
            if _i2 >= _m.size:
                break
            _r = _k * int(rows) // _lines
            _samp = np.linspace(_i1, _i2, int(cols), endpoint=False).astype(int)
            _raster[_r] += _m[_samp]
            _cnt[_r] += 1.0
    _cnt[_cnt == 0] = 1.0
    return np.clip((_raster / _cnt[:, None] + 0.7) / 1.4, 0.0, 1.0)


def _fpv_ppm(raster):
    """Растер (0..1) → PPM P6 — зеленувата монохромна гама FPV-монітора."""
    _g = np.clip(np.asarray(raster, dtype=np.float64), 0.0, 1.0)
    _h, _w = _g.shape
    _rgb = np.dstack(((_g * 140).astype(np.uint8), (_g * 255).astype(np.uint8),
                      (_g * 120).astype(np.uint8))).tobytes()
    return ("P6\n%d %d\n255\n" % (_w, _h)).encode("ascii") + _rgb


# ==================== v5.6.0: ЛАБОРАТОРІЯ ПЛАВЛЕННЯ / FOLLOW-ME / FAILSAFE / ПУЛЬСАЦІЯ ====================
MODS_MELT = ("BPSK", "QPSK", "16QAM", "LoRa SF7", "LoRa SF8", "LoRa SF9",
             "LoRa SF10", "LoRa SF11", "LoRa SF12")
_LORA_SNR_REQ_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


def mod_ber(mod_key, snr_db):
    """Бітова помилка модема за SNR (дБ): BPSK/QPSK 0.5·erfc(√SNR);
    16QAM 0.75·erfc(√(SNR/10)); LoRa SFx — запас над вимогливим SNR_req(SF)."""
    import math as _m56
    _g = 10.0 ** (float(snr_db) / 10.0)
    if mod_key in ("BPSK", "QPSK"):
        return 0.5 * _m56.erfc(_m56.sqrt(max(_g, 0.0)))
    if mod_key == "16QAM":
        return 0.75 * _m56.erfc(_m56.sqrt(max(_g, 0.0) / 10.0))
    if mod_key.startswith("LoRa SF"):
        _sf = int(mod_key.split("SF")[1])
        _gm = _g - 10.0 ** (_LORA_SNR_REQ_DB.get(_sf, -7.5) / 10.0)
        return 0.5 * _m56.erfc(_m56.sqrt(max(_gm, 0.0)))
    raise ValueError("невідома модуляція: %r" % (mod_key,))


def melt_prr(mod_key, c_dbm, j_dbm, nf_db, bw_hz, rb_bps, pkt_bits):
    """PRR одного пакета: сигнал C дБм проти суміші «завада J + шум»
    (N = −174 + 10·lg(BW) + NF; J+N = 10·lg(10^(J/10)+10^(N/10)));
    Eb/N0 = SNR + 10·lg(BW/Rb). Повертає dict(noise_dbm, iplus_dbm, snr_db,
    ebn0_db, ber, prr). Типові орієнтири для планування РЕБ."""
    import math as _m56
    _bw = max(float(bw_hz), 1.0)
    _n = -174.0 + 10.0 * _m56.log10(_bw) + float(nf_db)
    _ij = 10.0 * _m56.log10(10.0 ** (float(j_dbm) / 10.0) + 10.0 ** (_n / 10.0))
    _snr = float(c_dbm) - _ij
    _ebn0 = _snr + 10.0 * _m56.log10(_bw / max(float(rb_bps), 1.0))
    _ber = mod_ber(mod_key, _ebn0)
    return {"noise_dbm": _n, "iplus_dbm": _ij, "snr_db": _snr,
            "ebn0_db": _ebn0, "ber": _ber, "prr": (1.0 - _ber) ** int(pkt_bits)}


def melt_zone(mod_key, c_dbm, eirp_dbm, f_hz, nf_db, bw_hz, rb_bps, pkt_bits,
              prr_thr=0.5, d_max_km=5.0, n_pts=25):
    """Зона плавлення: PRR проти дальності для завади з даним EIRP.
    FSPL(дБ) = 32.44 + 20·lg(d_км) + 20·lg(f_МГц). d_melt_km — найбільша
    дальність, де пакет «плаває» (PRR < prr_thr); None — не плавиться ніде."""
    import math as _m56
    _fmhz = max(float(f_hz), 1.0) / 1e6
    _rows = []
    _dm = None
    for _i in range(max(int(n_pts), 2)):
        _d = float(d_max_km) * (_i + 1) / float(n_pts)
        _fspl = 32.44 + 20.0 * _m56.log10(max(_d, 1e-4)) + 20.0 * _m56.log10(_fmhz)
        _j = float(eirp_dbm) - _fspl
        _pr = melt_prr(mod_key, c_dbm, _j, nf_db, bw_hz, rb_bps, pkt_bits)
        _rows.append((_d, _j, _pr["snr_db"], _pr["prr"]))
        if _pr["prr"] < float(prr_thr):
            _dm = _d
    return {"rows": _rows, "d_melt_km": _dm, "prr_thr": float(prr_thr),
            "d_max_km": float(d_max_km)}


def follow_me_sim(hop_rate_hz, retune_ms, p_detect=0.9, hop_channels=1):
    """Follow-me (частотне супроводження): частка хопів, які завада встигає
    накрити = p_виявлення · max(0, (t_хоп − t_перестроювання)/t_хоп). Виграш
    дрона від сітки з N каналів — 10·lg(N) дБ додаткової потужності завади."""
    import math as _m56
    _t = 1000.0 / max(float(hop_rate_hz), 1e-9)
    _cov = float(p_detect) * max(0.0, (_t - float(retune_ms)) / _t)
    return {"t_hop_ms": _t, "p_cover": _cov,
            "gain_db": 10.0 * _m56.log10(max(float(hop_channels), 1.0)),
            "verdict": "ефективно" if _cov >= 0.5 else "неефективно"}


def failsafe_window(failsafe_s, hop_rate_hz, p_block):
    """Вікно failsafe: якщо завада блокує кожен хоп з ймовірністю p_block,
    failsafe спрацює з p_block^n (n = хопів у таймауті дрона); надійно при
    ≥95%. p_req — мінімальна p_block на хоп, за якої дрон ще встигає сісти."""
    _n = max(1, int(round(float(failsafe_s) * float(hop_rate_hz))))
    return {"hops": _n, "p_req": 0.95 ** (1.0 / _n),
            "p_fs": float(p_block) ** _n, "ok": bool(float(p_block) ** _n >= 0.95)}


def protocol_pulse(mode, pkt_rate_hz, pkt_dur_ms, sf=7, bw_hz=125000, sym=8):
    """Пульсуюча завада під протокол дрона. «LoRa преамбула»: мінімальний
    імпульс = sym·2^SF/BW (досить збити преамбулу); «Пакетний обмін»:
    тривалість пакета; «Постійна»: скважність 1, виграшу немає. econ_db —
    на скільки дБ нижча середня потужність за тієї ж ефективності."""
    import math as _m56
    if mode == "LoRa преамбула":
        _p = int(sym) * (2.0 ** int(sf)) / max(float(bw_hz), 1.0) * 1000.0
    elif mode == "Пакетний обмін":
        _p = float(pkt_dur_ms)
    else:
        _p = 1000.0 / max(float(pkt_rate_hz), 1e-9)
    _duty = min(1.0, float(pkt_rate_hz) * _p / 1000.0)
    return {"pulse_ms": _p, "duty": _duty,
            "econ_db": 10.0 * _m56.log10(1.0 / _duty) if _duty > 0.0 else 0.0}


def jam_profile_from_analysis(peaks_hz, pri_us=None, band_mhz=None):
    """Профіль завади з аналізу .raw: топ-8 частот піків (МГц, за зростанням),
    розкид смуги, PRI імпульсів (якщо відомий). База для лабораторії плавлення."""
    _fr = sorted({round(float(_f) / 1e6, 3) for _f in (peaks_hz or [])})
    _top = _fr[:8]
    _prof = {"freqs_mhz": _top, "n_peaks": len(_top), "span_mhz": None,
             "pri_us": float(pri_us) if pri_us is not None else None}
    if len(_top) >= 2:
        _prof["span_mhz"] = round(_top[-1] - _top[0], 3)
    if band_mhz is not None:
        _prof["span_mhz"] = float(band_mhz)
    return _prof


def iq_top_peaks(freqs_hz, psd_db, fs_eff, n_peaks=8, min_db=12.0, min_gap_hz=None):
    """Топ-N піків спектра для багатоцільової завади (v4.9.0).

    Локальні максимуми вище шумової підлоги на min_db (дБ), розділені
    щонайменше на min_gap_hz (дефолт 0.5% fs), сортовані за потужністю;
    для кожного — зайнята смуга (-20 дБ від піка) і тип завади через
    jam_from_peak. Повертає список dict(f_hz, width_hz, peak_db, kind,
    kind_name, bw_mhz, dev_mhz, per_us)."""
    n = int(np.asarray(psd_db).size)
    if n < 8:
        return []
    if min_gap_hz is None:
        min_gap_hz = max(float(fs_eff) * 0.005, 5e3)
    floor = float(np.median(psd_db))
    thr = floor + float(min_db)
    cand = []
    for i in range(1, n - 1):
        v = float(psd_db[i])
        if v >= thr and v > psd_db[i - 1] and v >= psd_db[i + 1]:
            cand.append((v, i))
    cand.sort(reverse=True)
    mx = float(np.max(psd_db))
    peaks = []
    for v, i in cand:
        fi = float(freqs_hz[i])
        if any(abs(fi - p["f_hz"]) < min_gap_hz for p in peaks):
            continue
        thr20 = v - 20.0
        j = i
        while j > 0 and psd_db[j] > thr20:
            j -= 1
        k = i
        while k < n - 1 and psd_db[k] > thr20:
            k += 1
        width = max(float(freqs_hz[k] - freqs_hz[j]), float(fs_eff) / float(n))
        d = jam_from_peak(fi / 1e6, 0.0, width / 1e6)
        peaks.append({"f_hz": fi, "width_hz": width, "peak_db": v - mx,
                      "kind": d["kind"], "kind_name": d["kind_name"],
                      "bw_mhz": d["bw_mhz"], "dev_mhz": d["dev_mhz"],
                      "per_us": d["per_us"]})
        if len(peaks) >= int(n_peaks):
            break
    return peaks


def iq_jam_analyze(path, dtype="float32", fs_iq=0.0, fc=0.0,
                   js_target=15.0, nfft=8192, bw_thr_db=-20.0, db_off=0.0):
    """Аналіз IQ-запису → параметри якісної завади (v3.19.0).

    Welch-усереднений спектр (вікно Ганна), пік, зайнята смуга (поріг
    bw_thr_db відносно піка), шумова підлога, динамічний діапазон,
    оцінка рівня сигналу у dBFS (адаптивно до шкали даних); тип завади
    за шириною через jam_from_peak (CW / ЛЧМ / шум) і потрібна потужність
    для заданого J/S; рівень сигналу = dBFS + калібрувальний офсет (дБм).
    Повертає dict(f_peak_hz, width_hz, floor_db, peak_db, dyn_db, dbfs,
    sig_dbm, js, db_off, jam_req_dbm, kind, kind_name, f_hz, bw_mhz, dev_mhz,
    per_us, note, freqs_hz, psd_db).
    """
    data, meta = read_iq(path, dtype=dtype, fs_iq=fs_iq, fc=fc)
    if not isinstance(meta, dict):
        meta = {}
    x = np.asarray(data)
    if x.size < nfft * 2:
        raise ValueError("IQ-запис закороткий для аналізу: %d відліків (потрібно ≥ %d)"
                         % (x.size, nfft * 2))
    fs_eff = float(meta.get("fs_iq") or fs_iq or 0.0)
    if fs_eff <= 0:
        raise ValueError("не вказано fs I/Q (поле «fs, МГц» або sidecar .iq.xml)")
    fc_eff = float(meta.get("fc") or fc or 0.0)
    x = x[: (x.size // nfft) * nfft]
    segs = x.reshape(-1, nfft)
    if segs.shape[0] > 64:
        segs = segs[np.linspace(0, segs.shape[0] - 1, 64).astype(int)]
    win = np.hanning(nfft)
    psd = (np.abs(np.fft.fft(segs * win, axis=1)) ** 2).mean(axis=0)
    psd = np.fft.fftshift(psd)
    freqs_hz = fc_eff + np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs_eff))
    psd_db = 10.0 * np.log10(psd / psd.max() + 1e-30)
    ipk = int(np.argmax(psd_db))
    pk_db = float(psd_db[ipk])
    f_peak_hz = float(freqs_hz[ipk])
    thr = pk_db + bw_thr_db
    j = ipk
    while j > 0 and psd_db[j] > thr:
        j -= 1
    k = ipk
    while k < nfft - 1 and psd_db[k] > thr:
        k += 1
    width_hz = max(float(freqs_hz[k] - freqs_hz[j]), fs_eff / float(nfft))
    floor_db = float(np.median(psd_db))
    dyn_db = pk_db - floor_db
    mx = float(np.max(np.abs(x)))
    if mx <= 1.5:
        full = 1.0
    else:
        full = 127.5 if "8" in str(meta.get("dtype") or dtype) else 32767.5
    p_tot = float(np.mean(np.abs(x) ** 2))
    dbfs = 10.0 * np.log10(p_tot / (full ** 2) + 1e-30)
    sig_dbm = dbfs + float(db_off)
    d = jam_from_peak(f_peak_hz / 1e6, sig_dbm, width_hz / 1e6)
    js = float(js_target)
    return {"f_peak_hz": f_peak_hz, "width_hz": width_hz, "floor_db": floor_db,
            "peak_db": pk_db, "dyn_db": dyn_db, "dbfs": dbfs, "sig_dbm": sig_dbm,
            "js": js, "db_off": float(db_off), "jam_req_dbm": sig_dbm + js,
            "kind": d["kind"], "kind_name": d["kind_name"], "f_hz": d["f_hz"],
            "bw_mhz": d["bw_mhz"], "dev_mhz": d["dev_mhz"], "per_us": d["per_us"],
            "note": "IQ: %d відл., %d вікон по %d (Welch, Ганн)"
                    % (x.size, segs.shape[0], nfft),
            "freqs_hz": freqs_hz, "psd_db": psd_db,
            "peaks": iq_top_peaks(freqs_hz, psd_db, fs_eff)}


def iq_to_raw(x, fc, fs_iq, fs_out=DEFAULT_FS, n_per_ch=16384, nch=8, seed=42, bw_hz=CHANNEL_BW):
    """Конвертує I/Q запис у .raw: середній спектр (Уелч) розкладається по
    каналах плану (ширина — bw_hz, за замовчуванням 64 МГц); кожен канал плану із сигналом синтезується як
    часовий ряд (випадкові фази, профіль = запис) на частоті fs_out.
    Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів."""
    if fs_iq <= 0 or fc <= 0:
        raise ValueError("Вкажіть fs I/Q і центральну частоту (з .iq.xml або вручну)")
    n = x.size
    nfft_iq = min(8192, n)
    if nfft_iq < 16:
        raise ValueError("Запис I/Q занадто короткий")
    w = np.hanning(nfft_iq)
    hop = nfft_iq // 2
    nseg = int(min(200, max(1, (n - nfft_iq) // hop + 1)))
    P = np.zeros(nfft_iq)
    for s in range(nseg):
        seg = x[s * hop: s * hop + nfft_iq] * w
        P += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    if not np.isfinite(P.sum()) or P.sum() <= 0:
        raise ValueError("Запис не містить сигналу")
    p_share = P / P.sum()
    fbin = fc + np.fft.fftshift(np.fft.fftfreq(nfft_iq, 1.0 / fs_iq))
    p_tot = float((np.abs(x) ** 2).mean())
    assign = []
    for gi, (f0, f1, g) in enumerate(plan_channels(bw_hz)):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        sh = float(p_share[i0:i1].sum())
        if sh > 0:
            assign.append((g, f0, f1, sh))
    n_syn = max(1024, int(n_per_ch) // 2 * 2)
    half = n_syn // 2 + 1
    delta = fs_out / n_syn
    rng = np.random.default_rng(seed)
    cols = [None] * nch
    for idx, (g, f0, f1, sh) in enumerate(assign[:nch]):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        f_rec, p_rec = fbin[i0:i1], p_share[i0:i1]
        j0 = max(1, int(np.ceil(f0 / delta)))
        j1 = min(half - 2, int(np.floor(f1 / delta)))
        if j1 < j0 or f_rec.size == 0:
            continue
        centers = np.arange(j0, j1 + 1) * delta
        w_lo = np.searchsorted(f_rec, centers - delta / 2.0)
        w_hi = np.searchsorted(f_rec, centers + delta / 2.0, side='right')
        s_bins = np.array([p_rec[a:b].sum() if b > a else 0.0
                           for a, b in zip(w_lo, w_hi)])
        if s_bins.sum() <= 0:
            continue
        s_bins = s_bins / s_bins.sum()
        p_ch = p_tot * sh
        rms_codes = 32768.0 * math.sqrt(p_ch / 0.5) * 0.5
        m = np.sqrt(max(rms_codes, 0.0) ** 2 * s_bins * n_syn * n_syn / 2.0)
        m_full = np.zeros(half)
        m_full[j0:j1 + 1] = m
        spec = m_full * np.exp(1j * rng.uniform(0, 2 * np.pi, half))
        spec[0] = 0.0
        spec[-1] = 0.0
        y = np.fft.irfft(spec, n_syn)
        pk = float(np.abs(y).max())
        if pk > 32000.0:
            y *= 32000.0 / pk
        slot = idx % nch
        cols[slot] = y if cols[slot] is None else cols[slot] + y
    block = np.stack([np.full(n_syn, 32768.0) if c is None else 32768.0 + c
                      for c in cols], axis=1)
    flat = np.clip(np.rint(block), 0, 65535).astype("<u2").reshape(-1)
    return {"flat": flat, "n": n_syn, "nch": nch, "n_sig": len(assign),
            "assign": assign, "p_tot": p_tot}


def analyze_report(path, fs, nch):
    chans, n = load_raw(path, nch)
    dur = n / fs
    fsv, fsu = human_freq(fs)
    lines = ["Файл: %s" % os.path.abspath(path),
             "Відліків на канал: %d   Каналів: %d   fs = %.4g %s   Тривалість: %.4g с"
             % (n, nch, fsv, fsu, dur), ""]
    lines.append("%-6s %10s %9s %8s %8s %9s %9s"
                 % ("канал", "DC", "RMS", "min", "max", "кліп≥65000", "кліп≤500"))
    for k in range(nch):
        st = channel_stats(chans[k], fs)
        lines.append("ch%-4d %10.0f %9.0f %8.0f %8.0f %8.2f%% %7.2f%%"
                     % (k, st["dc"], st["rms"], st["min"], st["max"],
                        st["clip_hi"], st["clip_lo"]))
    lines.append("")
    for k in range(nch):
        f, a = amplitude_spectrum(chans[k], fs)
        pk = find_peaks(f, a, k=5)
        lines.append("ch%d — топ-5 піків спектра:" % k)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            lines.append("    %12.4f %s  ×  %7.1f кодів" % (fv, fu, am))
    return "\n".join(lines), chans, n


# ----------------------------------------------------------------------------
# МАЛЮВАННЯ (спільне для GUI і preview)
# ----------------------------------------------------------------------------
def draw_time(fig, chans, fs, t0, dur, gs=None):
    if gs is not None:
        ax = fig.add_subplot(gs)
    else:
        ax = fig.subplots()
    n_tot = len(chans[0])
    i0 = int(min(max(t0 * fs, 0), n_tot - 1))
    i1 = int(min((t0 + dur) * fs, n_tot))
    i1 = max(i1, i0 + 2)
    dec = max(1, (i1 - i0) // 4000)
    for k in sorted(chans):
        s = chans[k][i0:i1]
        m = (s.size // dec) * dec
        b = s[:m].reshape(-1, dec)
        tt = (i0 + np.arange(b.shape[0]) * dec + dec / 2) / fs
        ax.fill_between(tt, b.min(axis=1), b.max(axis=1), alpha=0.25, color="C%d" % k)
        ax.plot(tt, b.mean(axis=1), lw=0.7, color="C%d" % k, label="ch%d" % k)
    nch = len(chans)
    ax.set_xlabel("час, с"); ax.set_ylabel("коди АЦП")
    ax.set_title("Часова область (%d канали): смуга min–max + середнє (декація 1:%d)"
                 % (nch, dec))
    ax.grid(alpha=0.3); ax.legend(ncol=min(nch, 8))


def _grid_for(nch):
    if nch <= 4:
        return 2, 2
    if nch <= 8:
        return 2, 4
    cols = 4
    return (nch + cols - 1) // cols, cols


def draw_spectrum(fig, chans, fs, gs=None):
    keys = sorted(chans)
    nch = len(keys)
    rows, cols = _grid_for(nch)
    if gs is not None:
        sub = gs.subgridspec(rows, cols)
        axes = [fig.add_subplot(sub[i // cols, i % cols]) for i in range(nch)]
    else:
        axes = fig.subplots(rows, cols, squeeze=False).ravel().tolist()[:nch]
    scale, unit = freq_unit_for(fs)
    fsv, fsu = human_freq(fs)
    report = []
    for k, ax in zip(keys, axes):
        f, a = amplitude_spectrum(chans[k], fs)
        ax.semilogy(f[1:] / scale, a[1:], lw=0.3, color="C%d" % (k % 10))
        ax.set_title("ch%d" % k, color="C%d" % (k % 10))
        ax.set_xlabel("частота, %s (fs = %.4g %s)" % (unit, fsv, fsu))
        ax.set_ylabel("амплітуда, кодів (пік)")
        ax.grid(alpha=0.3, which="both")
        pk = find_peaks(f, a, k=3)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            ax.annotate("%.4g %s" % (fv, fu), (fq / scale, am),
                        fontsize=7, rotation=90, va="bottom")
            report.append("ch%d: %12.4f %s × %7.1f кодів" % (k, fv, fu, am))
    return report


def draw_blocks(fig, chans, fs, blocks, occ, src_key=0):
    f, a = amplitude_spectrum(chans[src_key], fs)
    info = []
    for i, b in enumerate(blocks):
        ax = fig.add_subplot(len(blocks), 1, i + 1)
        m = (f >= b["f0"] - 2e6) & (f <= b["f1"] + 2e6)
        ax.semilogy(f[m] / 1e6, a[m], lw=0.25, color="C%d" % i)
        n_occ = 0
        for ch in b["channels"]:
            if occ[ch["global"]]["occupied"]:
                ax.axvspan(ch["f0"] / 1e6, ch["f1"] / 1e6, color="red", alpha=0.18)
                n_occ += 1
            else:
                ax.axvline(ch["f0"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        ax.axvline(b["f1"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        nb = len(b["channels"])
        npart = sum(1 for ch in b["channels"] if ch["partial"])
        ax.set_title("Блок %d: %.0f–%.0f МГц — %d каналів по 64 МГц%s, зайнято %d"
                     % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6, nb,
                        " (+%d крайовий)" % npart if npart else "", n_occ),
                     fontsize=9, color="C%d" % i)
        ax.set_ylabel("коди", fontsize=7)
        ax.grid(alpha=0.25, which="both")
        if i == len(blocks) - 1:
            ax.set_xlabel("частота, МГц")
        info.append("Блок %d: каналів %d, зайнято %d" % (b["idx"], nb, n_occ))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.05, hspace=0.65)
    return info


def draw_achx_curve(fig, points, title, color="C3", mode="linear", extrap="clamp"):
    ax = fig.subplots()
    fmax = min(1.2 * SIG_MAX_HZ, 12e9)
    fr = np.linspace(SIG_MIN_HZ * 0.5, fmax, 2000)
    ax.plot(fr / 1e9, achx_response(fr, points, mode=mode, extrap=extrap),
            lw=1.3, color=color)
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("|H(f)|, дБ")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def draw_cal_result(fig, measured, applied=None):
    ax = fig.subplots()
    fm = np.array([m[0] for m in measured]) / 1e9
    dm = np.array([m[2] for m in measured])
    ax.plot(fm, dm, "o-", ms=4, lw=0.8, color="C0", label="виміряно з запису")
    if applied:
        fr = np.linspace(min(fm), max(fm), 1000)
        db = np.interp(fr, [p[0] for p in applied], [p[1] for p in applied])
        ax.plot(fr / 1e9, db, lw=1.2, color="C3", label="еталонний профіль тракту")
        ax.legend()
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("АЧХ тракту, дБ (відн. еталону)")
    ax.set_title("Реальна АЧХ тракту — калібрувальний прогін тонів")
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
# GUI (Tkinter)
# ----------------------------------------------------------------------------
def run_gui(path=None, fs=DEFAULT_FS, nch=DEFAULT_NCH):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    BLOCKS = build_blocks()
    MODE_KEY = {v: k for k, v in ACHX_MODES.items()}
    EX_KEY = {v: k for k, v in ACHX_EXTRAPS.items()}

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(APP_TITLE)
            self.geometry("1400x860")
            self.theme_saved = _load_theme_name()
            self.theme_name = _resolve_theme(self.theme_saved)
            _apply_app_theme(self, self.theme_name)
            self._apply_mpl_theme(self.theme_name == "techno")
            self.after(60000, self._theme_tick)
            self._tray_setup()
            self.chans, self.n, self.fs, self.nch = None, 0, fs, nch
            self.last_freq_var = None
            self.measured_achx = None
            _ban = tk.Canvas(self, height=56, highlightthickness=0, bg=THEME["bg"])
            _ban.pack(fill="x")
            self._ban_ref = _ban
            self._paint_banner(_ban)
            self.bind("<Configure>", self._repaint_banner, add="+")
            top = ttk.Frame(self, padding=6); top.pack(fill="x")
            ttk.Button(top, text="1 · Відкрити .raw…", command=self.open_file).pack(side="left")
            self.lbl_path = ttk.Label(top, text=path or "файл не відкрито", width=52)
            self.lbl_path.pack(side="left", padx=8)
            self.cb_theme = ttk.Combobox(top, values=("Техно", "Класична", "День",
                                                      "AMOLED", "Windows XP", "Авто (год)"),
                                         state="readonly", width=11)
            self.cb_theme.set({"techno": "Техно", "classic": "Класична", "day": "День",
                               "amoled": "AMOLED", "xp": "Windows XP", "auto": "Авто (год)"}[self.theme_saved])
            self.cb_theme.bind("<<ComboboxSelected>>", lambda _e: self._switch_theme())
            self.cb_theme.pack(side="right", padx=8)
            for txt in ("fs, МГц (можна 12e9):", "каналів:"):
                ttk.Label(top, text=txt).pack(side="left", padx=(10, 2))
            self.fs_var = tk.StringVar(value="%.6g" % (fs / 1e6)); self.nch_var = tk.StringVar(value=str(nch))
            ttk.Entry(top, textvariable=self.fs_var, width=10).pack(side="left")
            ttk.Entry(top, textvariable=self.nch_var, width=4).pack(side="left")
            self.btn_analyze = ttk.Button(top, text="3 · Аналізувати", command=self._analyze_flow)
            self.btn_analyze.pack(side="left", padx=8)
            self.btn_analyze.config(state="disabled")
            ttk.Label(top, text="v%s" % APP_VERSION, foreground=THEME["dim"]).pack(side="right", padx=(6, 0))
            self.bw_hz = CHANNEL_BW; self.blocks = BLOCKS
            ttk.Label(top, text="Смуга, МГц:").pack(side="left", padx=(10, 2))
            self.bw_var = tk.StringVar(value="%.6g" % (CHANNEL_BW / 1e6))
            e_bw = ttk.Entry(top, textvariable=self.bw_var, width=7); e_bw.pack(side="left")
            e_bw.bind("<Return>", lambda e: self._apply_bw())
            e_bw.bind("<FocusOut>", lambda e: self._apply_bw())
            grow = ttk.Frame(self); grow.pack(fill="x", pady=(2, 0))
            self.lbl_guide = ttk.Label(grow, text="", foreground=THEME["text"],
                                       font=("DejaVu Sans", 9))
            self.lbl_guide.pack(anchor="w", padx=2)
            self._guide = lambda t: self.lbl_guide.config(text="▶ " + t)
            self._guide("Крок 1: натисніть «1 · Відкрити .raw…» і виберіть запис — або одразу "
                        "Генератор / I/Q → .raw / Калібрування АЧХ.")
            self.status = tk.StringVar(value="готовий")
            self.lbl_status = tk.Label(self, textvariable=self.status, anchor="w",
                                       bg="#0a0e18", fg=THEME["accent"],
                                       font=("DejaVu Sans", 9), padx=8, pady=3,
                                       highlightthickness=1,
                                       highlightbackground=THEME["edge_hi"])
            self.lbl_status.pack(side="bottom", fill="x")
            self._help_open = False
            self.btn_help = ttk.Button(self, text="📖 Довідка по параметрах — що означає кожне поле",
                                       command=self._toggle_help)
            self.btn_help.pack(side="bottom", fill="x", padx=6, pady=(2, 0))
            ttk.Button(self, text="📊 Спектрограма .raw… — запуск і перегляд на екрані",
                       command=self._specgram_open).pack(side="bottom", fill="x", padx=6, pady=(2, 0))
            arow = ttk.Frame(self); arow.pack(side="bottom", fill="x", padx=6, pady=(2, 0))
            ttk.Button(arow, text="⛰ 3D-спектрограма…", command=self._specgram_open).pack(side="left", padx=2)
            ttk.Button(arow, text="⚔ Завада vs ціль (3D)…", command=self._vs3d_open).pack(side="left", padx=2)
            ttk.Button(arow, text="⚖ Порівняння A/B…", command=self._ab_open).pack(side="left", padx=2)
            ttk.Button(arow, text="⏱ PRI-аналіз…", command=self._pri_open).pack(side="left", padx=2)
            ttk.Button(arow, text="🔊 Демодуляція аудіо…", command=self._demod_open).pack(side="left", padx=2)
            ttk.Button(arow, text="🗂 Батч-аналіз теки…", command=self._batch_open).pack(side="left", padx=2)
            self.txt_help = tk.Text(self, height=10, wrap="word", font=("DejaVu Sans", 9),
                                   state="normal")
            self.txt_help.insert("1.0", self.HELP_TEXT)
            self.txt_help.config(state="disabled")

            mrow = ttk.Frame(self); mrow.pack(fill="x", pady=(2, 0))
            ttk.Label(mrow, text="Блок:").pack(side="left")
            self.mode_var = tk.StringVar(value="ana")
            ttk.Radiobutton(mrow, text="📊 АНАЛІЗ СИГНАЛУ", value="ana",
                            variable=self.mode_var, command=self._set_mode).pack(side="left", padx=(6, 4))
            ttk.Radiobutton(mrow, text="⚙ ГЕНЕРАТОР СИГНАЛУ", value="gen",
                            variable=self.mode_var, command=self._set_mode).pack(side="left")
            mid = ttk.Frame(self); mid.pack(fill="both", expand=True)
            nb_ana = ttk.Notebook(mid)
            nb_gen = ttk.Notebook(mid)
            self.nb_ana = nb_ana
            self.nb_gen = nb_gen
            self._manuals = []
            nb_ana.pack(fill="both", expand=True)
            f1 = ttk.Frame(nb_ana); nb_ana.add(f1, text="Часова область")
            ctl = ttk.Frame(f1); ctl.pack(fill="x", pady=2)
            ttk.Label(ctl, text="з, с:").pack(side="left")
            self.t0_var = tk.StringVar(value="0"); ttk.Entry(ctl, textvariable=self.t0_var, width=8).pack(side="left")
            ttk.Label(ctl, text="тривалість, с:").pack(side="left", padx=(8, 0))
            self.dur_var = tk.StringVar(value="1e-5"); ttk.Entry(ctl, textvariable=self.dur_var, width=8).pack(side="left")
            ttk.Button(ctl, text="Показати", command=self.redraw_time).pack(side="left", padx=8)
            self.fig_t = Figure(figsize=(13, 6), dpi=100)
            self.canvas_t = FigureCanvasTkAgg(self.fig_t, master=f1)
            NavigationToolbar2Tk(self.canvas_t, f1)
            self.canvas_t.get_tk_widget().pack(fill="both", expand=True)

            f2 = ttk.Frame(nb_ana); nb_ana.add(f2, text="Спектр")
            sctl = ttk.Frame(f2); sctl.pack(fill="x", pady=2)
            self.antpat_btn = ttk.Button(sctl, text="📷 Зберегти діаграму у PNG",
                                         command=self._antpat_export)
            self.antpat_btn.pack(side="left", padx=6)
            self.fig_s = Figure(figsize=(15.5, 6.2), dpi=100)

            self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=f2)
            NavigationToolbar2Tk(self.canvas_s, f2)
            self.canvas_s.get_tk_widget().pack(fill="both", expand=True)

            f2b = ttk.Frame(nb_ana); nb_ana.add(f2b, text="Блоки (64 МГц)")
            self.tab_blocks = f2b
            self.fig_b = Figure(figsize=(13, 7.5), dpi=100)
            self.canvas_b = FigureCanvasTkAgg(self.fig_b, master=f2b)
            NavigationToolbar2Tk(self.canvas_b, f2b)
            self.canvas_b.get_tk_widget().pack(fill="both", expand=True)

            f2c = ttk.Frame(nb_ana); nb_ana.add(f2c, text="Калібрування АЧХ")
            self._build_calibration(f2c)

            f3 = ttk.Frame(nb_ana); nb_ana.add(f3, text="Статистика")
            self.txt = tk.Text(f3, wrap="none", font=("DejaVu Sans Mono", 10))
            yb = ttk.Scrollbar(f3, command=self.txt.yview)
            self.txt.configure(yscrollcommand=yb.set)
            yb.pack(side="right", fill="y"); self.txt.pack(fill="both", expand=True)

            fir = ttk.Frame(nb_ana); nb_ana.add(fir, text="ІЧ (1.5–6 мкм)")
            self._build_ir(fir)

            fmw = ttk.Frame(nb_ana); nb_ana.add(fmw, text="Мікрохвилі (IEEE)")
            self._build_ieee(fmw)
            fham = ttk.Frame(nb_ana); nb_ana.add(fham, text="Радіолюбительські (160м…23см)")
            self._build_ham(fham)
            self.gen_frame = ttk.Frame(nb_gen); nb_gen.add(self.gen_frame, text="Генератор")
            self._build_generator(self.gen_frame)
            few = ttk.Frame(nb_gen); nb_gen.add(few, text="Модуляція і РЕБ")
            self._build_ew(few)

            if path and os.path.exists(path):
                self.after(200, lambda: self._load(path))

            fiq = ttk.Frame(nb_ana); nb_ana.add(fiq, text="I/Q → .raw")
            self.iq_frame = fiq
            self._build_iq(fiq)
            fman = ttk.Frame(nb_ana); nb_ana.add(fman, text="Інструкція")
            self._build_manual(fman, "ana")
            fman2 = ttk.Frame(nb_gen); nb_gen.add(fman2, text="Інструкція")
            self._build_manual(fman2, "gen")

        HELP_TEXT = (
            "ПАРАМЕТРИ ТАБЛИЦІ ГЕНЕРАТОРА (по рядках каналів)\n"
            "  • Канал (ch0…ch7) — номер канала у файлі .raw; канали чергуються ПО ВІДЛІКАХ:\n"
            "      [ch0, ch1, …, chN-1, ch0, …].\n"
            "  • Увімк — ✓ канал пишеться у файл зі своїм сигналом; ☐ — канал лишається\n"
            "      постійною складовою DC.\n"
            "  • Форма — форма сигналу: Синус / Меандр (прямокутні коливання) / Трикутник /\n"
            "      Імпульс (прямокутні імпульси тривалістю «Ширина») / Шум (білий гаусів;\n"
            "      частота для нього не використовується).\n"
            "  • Частота, МГц — частота тону (100…6000 МГц). Для Меандру/Імпульсу це частота\n"
            "      слідкування (період повторення імпульсів); для Шуму — не використовується (0).\n"
            "      Голе число — МГц; старий запис (2.4e9) і суфікси (2.4ггц) теж розуміються.\n"
            "  • Фаза, ° — початкова фаза тону у градусах (0…359).\n"
            "  • Ампл., кодів — амплітуда сигналу у кодах АЦП: пік ± від DC (10000 ≈ ±10000\n"
            "      кодів від середини шкали; надто велика амплітуда → переповнення, кліп).\n"
            "  • Ширина, мкс — тривалість імпульсу/високого рівня для Імпульсу і Меандру;\n"
            "      мінімально розумна — 4 відліки при fs моделі SDR (кнопка підстановки).\n"
            "  • DC, кодів — постійна складова, середина шкали (32768; сигнал «сидить» на ній).\n"
            "  • Підсилення — множник форми сигналу (1.0 = як є; 2.0 — амплітуда вдвічі більша).\n"
            "\n"
            "ПАРАМЕТРИ ГЕНЕРАЦІЇ (нижній рядок вкладки)\n"
            "  • Блок / Канал — вибір каналу частотного плану (100 МГц…6 ГГц); при застосуванні\n"
            "    типу сигналу, групи, запису бібліотеки АБО введенні частоти в таблицю Блок\n"
            "    і Канал підставляються автоматично (найближчий канал до несучої); зайняті\n"
            "    канали завантаженого запису позначені «●» у списку та «● ЗАЙНЯТИЙ» біля\n"
            "    центра; ширина канала\n"
            "      = «Смуга, МГц» з верхнього поля (за замовчуванням 64 МГц); кнопка\n"
            "      «→ центр каналу» підставляє його центральну частоту у виділене поле Частота.\n"
            "  • fs, МГц — частота дискретизації (12 ГГц = пряма оцифровка 6 ГГц; кнопка\n"
            "      «Модель SDR» підставляє максимум обраної моделі).\n"
            "  • тривалість, с — тривалість запису; відліків на канал = тривалість × fs.\n"
            "  • seed — число генератора випадкових чисел: той самий seed → той самий шум.\n"
            "\n"
            "МОДЕЛЬ SDR\n"
            "  • Вибір моделі підставляє fs приладу і показує смугу/діапазон/біти; кнопка\n"
            "      «→ мінімальну ширину» підставляє 4 відліки при fs моделі у поле Ширина;\n"
            "      надто вузькі імпульси (Меандр/Імпульс) — попередження при генерації.\n"
            "\n"
            "ДОДАТКОВІ ПАРАМЕТРИ СИГНАЛУ (рамка під SDR)\n"
            "  • період слідування, мкс — період повторення імпульсів (0 = за Частотою, 1/f).\n"
            "  • тривалість імпульсу, мкс — перевизначає «Ширина» для Імпульсу/Меандру (0 = ні).\n"
            "  • фронт/спад, мкс — швидкість зростання/спаду фронтів (0 = ідеальні стрибки);\n"
            "      Імпульс стає РАДІОімпульсом: несуча (Частота) × огинаюча з фронтами.\n"
            "  • індекс АМ (0…1) + частота АМ, Гц — амплітудна модуляція: огинаюча\n"
            "      (1+m·sin(2π·f_AM·t)); m=0.5 → глибина ±50%.\n"
            "  • девіація ЛЧМ, МГц — повна розгортка частоти: для Імпульсу — всередині\n"
            "      кожного імпульсу; для LoRa дорівнює BW (0.125 / 0.25 / 0.5 МГц).\n"
            "  • період ЛЧМ, мкс — тривалість одного розгортання = символ LoRa\n"
            "      T_sym = 2^SF/BW: SF7/BW125 → 1024, SF8 → 2048, SF9 → 4096, SF10 → 8192,\n"
            "      SF11 → 16384, SF12 → 32768 мкс; 0 = старий режим (Імпульс — розгортка\n"
            "      на імпульс, Синус — на весь запис).\n"
            "\n"
            "ТИПИ СИГНАЛІВ (рядок «Тип сигналу:» у Генераторі)\n"
            "Оберіть стандарт — LoRa (868/VHF 150–300/433), LoRaWAN, 2G, 3G, 4G, 5G, WiFi, GNSS, GPS L1, UNB,\n"
            "FSK/GFSK, FHSS, QAM, CVBS, COFDM, NB-IoT — і натисніть «→ застосувати»:\n"
            "Також — несучі GNSS (GPS L2/L5, ГЛОНАСС G1/G2/G3, Galileo E1/E5a/E5b/E6,\n"
            "BeiDou B1/B3, SAR), «Комплекс GNSS L1» (Galileo E1 + ГЛОНАСС G1 + BeiDou B1 +\n"
            "GPS L1 в ОДНОМУ каналі, форма «GNSS L1»), ШПС-форми «GPS L1», «Galileo E1»,\n"
            "«ГЛОНАСС G1», «BeiDou B1» (по одному рядку) і групи сигналів «Група: …».\n"
            "програма підставить форму сигналу, несучу (МГц), ширину, fs і всі додаткові\n"
            "параметри (період, тривалість, фронти, АМ, ЛЧМ: BW + T_sym) у перший УВІМКНЕНИЙ\n"
            "рядок таблиці (якщо увімкнених немає — у рядок 1, тоді увімкніть його вручну).\n"
            "Пакетні стандарти (GSM, LoRaWAN) стають радіоімпульсами; широкосмугові (3G,\n"
            "4G, 5G, WiFi, GNSS, FHSS, QAM, COFDM) — моделлю шумового заповнення канала;\n"
            "CVBS — трапецією в базовій смузі (несуча 0); вузькосмугові (LoRa, UNB, FSK,\n"
            "NB-IoT) — несучою з ЛЧМ/без. Значення — типові, з відкритих специфікацій;\n"
            "коротка модель показана біля списку, повна — у статусному рядку після застосування.\n"
            "Групи («Група: …») розподіляють сигнали по каналах — по одному в кожен рядок;\n"
            "якщо ввімкнених рядків менше, ніж сигналів у групі, зайві рядки вмикаються\n"
            "автоматично (аудіо-рядки «Аудіо L/R» не займаються):\n"
            "      наприклад, усі GNSS-системи (L1, L2, L5, G1, G2, G3, E5b, E6) — у 8 каналів одного файла.\n"
            "MESH XK-F358: пресети UHF/L/S (+6 піддіапазонів 1.6–2.9 ГГц)/S2 (3200–3400)/C/MIIT —\n"
            "TD-COFDM-пачки (64 піднесучих, CP 1/8,\n"
            "      QAM псевдовипадкові), TDMA (період/тривалість пачок = «період слідування»/\n"
            "      «тривалість імпульсу»), FHSS 5 каналів × BW (BW = «девіація ЛЧМ, МГц»);\n"
            "      модуляція Mesh — перемикач «модуляція Mesh (QAM)»: адаптивна (BPSK…1024QAM,\n"
            "      випадковий порядок на кожен OFDM-символ) або фіксована; також у вікні ⚙ канала.\n"
            "      група «Mesh XK-F358 2T2R» — два незалежні потоки (TX1+TX2) у двох каналах.\n"
            "      TGTX5050A CRPA — прийом L1 на RF1/RF2 (пресети й групи); завади CW/шум/чирп — тест адаптивних нулів.\n"
            "      Спуфер — «Spoofing GPS»: зміщення несучої (Гц) і дрейф (Гц/с) — підробний GPS L1 поруч із реальним.\n"
            "      Розподіл (v3.5.0): тип сигналу/бібліотека → ВСІ увімкнені канали; авто-DC (32768) і Підсилення від типу (PAPR) та частоти.\n"
            "      Після генерації — пік, PAPR і запас до кліпування по каналах; «Імпульс» без радіо-extras — авто-DC центрує 0…ампл.\n"
            "\n"
            "КОНВЕРТЕР I/Q → .raw (вкладка «I/Q → .raw»)\n"
            "Файл I/Q — запис RTSA-Suite: .iq (Raw IQ, інтерлівінг Re,Im), .wav (стерео)\n"
            "або будь-який бінарний raw; тип даних float32/float64/int8/int16/int32.\n"
            "Якщо поруч лежить .iq.xml (RTSA_RAW_IQ_File), fs і центральна частота\n"
            "підставляться автоматично; інакше введіть їх вручну (поля в МГц).\n"
            "Нативні стиснуті .rtsa не читаються — експортуйте Raw IQ у File Source.\n"
            "Конвертор будує середній спектр запису (Уелч, до 200 сегментів, перші ~4 млн\n"
            "відліків), розкладає потужність по каналах плану (ширина — «Смуга, МГц» з верхнього поля; за замовчуванням 64 МГц) і синтезує для\n"
            "кожного канала часовий ряд із цим профілем: піки, рівні в кодах і зайнятість\n"
            "каналів у RAW Studio відповідають запису; порожні канали — DC 32768.\n"
            "Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів; запас\n"
            "−6 дБ, без кліпінгу. fs .raw (МГц) — те, що треба вказати при відкритті файлу.\n"
            "\n"
            "МОДУЛЬ АЧХ (під рядком SDR)\n"
            "  • застосовувати АЧХ — форма змінної складової залежно від частоти за профілем;\n"
            "      постійна складова DC не модулюється.\n"
            "  • профіль — Плоский / Спад −3 дБ/окт / Смуговий 100М–6Г / Провал / Користувацький.\n"
            "  • інтерполяція — linear / spline / logspline (між точками профілю); поза діапазоном\n"
            "      — clamp / linear / zero (за межами крайніх точок таблиці).\n"
            "  • центр провалу, МГц — частота провалу для профілю «Провал».\n"
            "  • власна таблиця (частота_Гц дБ) — рядки профілю; сюди потрапляє виміряна АЧХ тракту.\n"
            "\n"
            "КАЛІБРУВАННЯ АЧХ (окрема вкладка)\n"
            "  • fs, каналів, тонів — параметри прогону; від/до, МГц — діапазон тонів (кожен тон\n"
            "      точно в біні FFT); ампл. тону — амплітуда еталона; відліків/тон — довжина сегмента.\n"
            "  • Крок 1 — створити калібрувальний .raw → прогнати через тракт → записати;\n"
            "      крок 2 — виміряти АЧХ із запису; крок 3 — профіль у Генератор (компенсація −дБ);\n"
            "      💾 / 📁 — збереження/завантаження профілю .achx (файли у Гц, режими — в шапці).\n"
            "\n"
            "БІБЛІОТЕКА ЧАСТОТ (рядок «Крок 2б · Бібліотека частот радіозв'язку:» у Генераторі)\n"
            "  • 6 категорій, 51 запис: GNSS (GPS/ГЛОНАСС/BeiDou/Galileo), стільниковий 2G–5G,\n"
            "      Wi-Fi/BT, IoT/M2M (NFC, Zigbee, Z-Wave, LoRaWAN), супутникові діапазони\n"
            "      L/C/X/Ku/Ka, службовий радіозв'язок (Airband, Marine, рації).\n"
            "  • «→ застосувати» ставить тип, несучу (центр смуги), fs і додаткові параметри\n"
            "      у перший УВІМКНЕНИЙ рядок таблиці — як і «Тип сигналу».\n"
            "  • Записи поза частотним планом 100 МГц…6 ГГц позначені «ПОЗА ПЛАНОМ»: частота\n"
            "      підставляється, але генерацію буде відхилено.\n"
            "\n"
            "МОДУЛЬ ІЧ (вкладка «ІЧ (1.5–6 мкм)»)\n"
            "  • основні характеристики SWIR (1.5–3 мкм) і MWIR (3–6 мкм): частоти, температури,\n"
            "      характерні джерела; калькулятор Віна: T (°C) ↔ довжина хвилі піку ↔ f = c/λ (ТГц);\n"
            "      спектр Планка для 210/700/1000/1600 °C із заштрихованими піддіапазонами.\n"
            "\n"
            "\n"
            "МІКРОХВИЛІ IEEE (вкладка «Мікрохвилі (IEEE)»)\n"
            "  • літерні діапазони UHF / L / S / C / X / Ku / K / Ka / V-W (0.3–110 ГГц):\n"
            "      частоти, довжини хвиль, застосування — зв'язок (GSM, LTE, T2), навігація\n"
            "      (GPS, ГЛОНАСС), Wi-Fi/BT, радари (цивільні, військові, авто 77 ГГц),\n"
            "      супутниковий зв'язок і Starlink, 5G mmWave;\n"
            "      калькулятор: частота (ГГц) ↔ довжина хвилі (см) ↔ діапазон IEEE;\n"
            "      лог-графік смуг 0.3–110 ГГц з мітками відомих сервісів.\n"
            "\n"
            "ДІАПАЗОНИ РАДІОЛЮБИТЕЛІВ (вкладка «Радіолюбительські (160м…23см)»)\n"
            "  • 15 аматорських діапазонів IARU Region 1: 160/80/60/40/30/20/17/15/12/10/6/4/2 м,\n"
            "      70 і 23 см — частоти, застосування; межі: нижня включно, верхня виключно;\n"
            "      калькулятор: частота (МГц) ↔ довжина хвилі (м) ↔ діапазон;\n"
            "      лог-графік смуг з ромбами стандартних частот FT8.\n"
            "\n"
            "МОДУЛЯЦІЯ І РЕБ (блок «Генератор сигналу» → вкладка «Модуляція і РЕБ»)\n"
            "  • v2.1.0: розділ «Практика» — кнопки налаштовують Генератор\n"
            "    (АМ, ЧМ, ФМ, Barrage, Spot, Spoofing, Клітка Фарадея).\n"
            "АУДІО-КАНАЛИ (Генератор, v2.2.0)\n"
            "  • рядки «Аудіо L»/«Аудіо R» + кнопка «🎵 Аудіо (стерео)…»: WAV (або\n"
            "    MP3 через ffmpeg) → ліва/права доріжка в ці канали, тривалість = аудіо.\n"
            "  • fs запису автоматично → max(4×sr аудіо, 1 МГц), щоб аудіо вмістилось у файл.\n"
            "  • «🌐 Завантажити трек…» — URL mp3/wav (до 200 МБ) або YouTube (yt-dlp + ffmpeg).\n"
            "ГЕНЕРАЦІЯ (v2.4.0)\n"
            "  • запис містить УСІ увімкнені канали: кількість каналів автоматично\n"
            "    розширюється до останнього позначеного рядка таблиці;\n"
            "  • «Форма для всіх вибраних» — одна форма всім позначеним рядкам;\n"
            "  • ШІ-підбір завади — тип/смуга/J-S від цілі, ПП, моделі SDR і дальності розкриття;\n"
            "  • Спектр із зображення — оцифрування PNG скріншота: піки, план, передача в ШІ-заваду;\n"
            "    у кожному рядку можна лишити власну форму.\n"
            "КАНАЛИ ТА ПРЕСЕТИ (v2.5.0)\n"
            "  • «⚙» у рядку — власні АМ/ЧМ/імпульсні параметри каналу (⚙* = задано);\n"
            "    канали без власних беруть спільні «Додаткові параметри».\n"
            "  • «💾 Зберегти пресет…» / «📂 Завантажити пресет…» — уся конфігурація\n"
            "    каналів (.rspreset, JSON): рядки, форми, параметри, fs, тривалість, АЧХ.\n"
            "  • типи модуляції: АМ — зміна амплітуди, ЧМ/FM — зміна частоти, ФМ/PM — зміна\n"
            "      фази (BPSK/QPSK — основа Wi-Fi, LTE, GNSS); осцилограми всіх трьох типів\n"
            "      і спектр придушення — на графіку вкладки;\n"
            "      радіоелектронне придушення та обман: barrage (шум по всій смузі), spot\n"
            "      (прицільне), spoofing (підміна) — сутність і типовий захист кожного;\n"
            "      блокування: імітація клітки Фарадея — сигнал повністю придушений,\n"
            "      у записі лише шумовий пол приймача.\n"
            "\n"
            "ІНСТРУКЦІЯ (вкладка «Інструкція»)\n"
            "  • повний довідник користувача всередині програми: встановлення, формат .raw,\n"
            "      усі вкладки аналізу, Генератор, АЧХ/калібрування, конвертер I/Q,\n"
            "      командний рядок і типові проблеми; текст лише для читання, з прокруткою;\n"
            "      клікабельний зміст зверху — перехід до розділу одним кліком.\n"
            "\n"
            "ДВА БЛОКИ ІНТЕРФЕЙСУ (з версії 2.0.0)\n"
            "  • під верхнім полем — перемикач режиму: «📊 Аналіз сигналу» (вкладки аналізу,\n"
            "      довідники частот, конвертер I/Q, інструкція) і «⚙ Генератор сигналу»\n"
            "      (створення нових .raw);\n"
            "      застосування типу сигналу або запису бібліотеки автоматично\n"
            "      перемикає на блок Генератора.\n"
            "ВЕРХНЄ ПОЛЕ (аналіз)\n"
            "  • fs, МГц — частота дискретизації ВІДКРИТОГО файла; каналів — кількість каналів;\n"
            "      Смуга, МГц — ширина канала частотного плану (вкладка «Блоки»; за замовчуванням\n"
            "      64 МГц; Enter або вихід з поля — застосувати);\n"
            "      кнопка «Аналізувати» — перечитати файл і оновити всі вкладки."
        )

        def _build_ir(self, f):
            head = ttk.LabelFrame(f, text="Основні характеристики діапазону 1.5–6 мкм")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [
                ("Параметр / Піддіапазон", "SWIR (короткохвильовий ІЧ)", "MWIR (середньохвильовий ІЧ)"),
                ("Діапазон хвиль", "1.5 мкм – 3.0 мкм", "3.0 мкм – 6.0 мкм"),
                ("Еквівалентна частота", "100 ТГц – 200 ТГц", "50 ТГц – 100 ТГц"),
                ("Відповідні температури", "від 700 °C до понад 1600 °C", "від 210 °C до 700 °C"),
                ("Характерні джерела",
                 "форсажні струмені, розпечений метал, дугове зварювання, порохові гази",
                 "корпуси авіаційних двигунів, вихлопні труби техніки, тертя об повітря на надзвуку"),
            ]
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=430, justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground=THEME["dim"] if j == 0 else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=2)
            calc = ttk.LabelFrame(f, text="Калькулятор Віна: температура ↔ довжина хвилі піку ↔ еквівалентна частота")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Температура, °C:").pack(side="left")
            self.ir_t = tk.StringVar(value="700")
            ttk.Entry(r1, textvariable=self.ir_t, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ пік і частота", command=self._ir_from_t).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, мкм:").pack(side="left", padx=(16, 0))
            self.ir_l = tk.StringVar(value="3.5")
            ttk.Entry(r1, textvariable=self.ir_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ температуру і частоту", command=self._ir_from_l).pack(side="left", padx=4)
            ttk.Button(r1, text="→ спектр Планка", command=self._draw_ir).pack(side="left", padx=16)
            self.lbl_ir = ttk.Label(calc, text="", foreground=THEME["text"], wraplength=1250, justify="left")
            self.lbl_ir.pack(anchor="w", padx=2, pady=2)
            self.fig_ir = Figure(figsize=(13, 5.6), dpi=100)
            self.canvas_ir = FigureCanvasTkAgg(self.fig_ir, master=f)
            NavigationToolbar2Tk(self.canvas_ir, f)
            self.canvas_ir.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ir()
            self._ir_from_t()

        def _draw_ir(self):
            self.fig_ir.clear()
            draw_ir_spectrum(self.fig_ir)
            self.canvas_ir.draw_idle()

        def _ir_from_t(self):
            try:
                tc = float(self.ir_t.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна температура"); return
            t = tc + 273.15
            if t <= 0:
                messagebox.showerror(APP_TITLE, "Температура має бути вище −273.15 °C"); return
            lam = wien_lambda_max(t)
            self.ir_l.set("%.4g" % (lam * 1e6))
            self._ir_report(lam, tc)

        def _ir_from_l(self):
            try:
                um = float(self.ir_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if um <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 мкм"); return
            lam = um * 1e-6
            t = 2897.771955e-6 / lam
            self.ir_t.set("%.5g" % (t - 273.15))
            self._ir_report(lam, t - 273.15)

        def _ir_report(self, lam, tc):
            fr = lambda_to_freq(lam)
            sub = ir_subrange(lam)
            src = IR_BANDS.get(sub, {}).get("sources", "")
            tail = " · джерела: " + src if src else " · джерела: див. SWIR/MWIR у таблиці вище"
            self.lbl_ir.config(text="λ пік = %.4g мкм · T = %.5g °C (%.5g K) · f = c/λ = %.5g ТГц · піддіапазон: %s%s"
                               % (lam * 1e6, tc, tc + 273.15, fr / 1e12, sub, tail))

        def _build_ieee(self, f):
            head = ttk.LabelFrame(f, text="Літерні мікрохвильові діапазони IEEE (0.3–110 ГГц)")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [("Діапазон", "Частота, ГГц", "Довжина хвилі", "Основне практичне застосування")]
            for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
                rows.append((_name, "%.4g – %.4g" % (_f0 / 1e9, _f1 / 1e9), _lam, _uses))
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=(110 if j < 3 else 420), justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground=THEME["dim"] if (i == 0 or j == 0) else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)
            calc = ttk.LabelFrame(f, text="Калькулятор: частота ↔ довжина хвилі ↔ літерний діапазон")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Частота, ГГц:").pack(side="left")
            self.mw_f = tk.StringVar(value="2.45")
            ttk.Entry(r1, textvariable=self.mw_f, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ довжину хвилі і діапазон", command=self._ieee_from_f).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, см:").pack(side="left", padx=(16, 0))
            self.mw_l = tk.StringVar(value="12.24")
            ttk.Entry(r1, textvariable=self.mw_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ частоту і діапазон", command=self._ieee_from_l).pack(side="left", padx=4)
            self.lbl_mw = ttk.Label(calc, text="", foreground=THEME["text"], wraplength=1250, justify="left")
            self.lbl_mw.pack(anchor="w", padx=2, pady=2)
            self.fig_mw = Figure(figsize=(13, 4.6), dpi=100)
            self.canvas_mw = FigureCanvasTkAgg(self.fig_mw, master=f)
            NavigationToolbar2Tk(self.canvas_mw, f)
            self.canvas_mw.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ieee()
            self._ieee_from_f()

        def _draw_ieee(self):
            self.fig_mw.clear()
            draw_ieee_spectrum(self.fig_mw)
            self.canvas_mw.draw_idle()

        def _ieee_from_f(self):
            try:
                f = float(self.mw_f.get().replace(",", ".")) * 1e9
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна частота"); return
            if f <= 0:
                messagebox.showerror(APP_TITLE, "Частота має бути більшою за 0 ГГц"); return
            self.mw_l.set("%.4g" % (freq_to_lambda(f) * 100.0))
            self._ieee_report(f)

        def _ieee_from_l(self):
            try:
                cm = float(self.mw_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if cm <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 см"); return
            f = 299792458.0 / (cm / 100.0)
            self.mw_f.set("%.4g" % (f / 1e9))
            self._ieee_report(f)

        def _ieee_report(self, f):
            lam = freq_to_lambda(f)
            band = ieee_band_for(f)
            if band is None:
                self.lbl_mw.config(text="f = %.4g ГГц · λ = c/f = %s · поза літерними діапазонами IEEE (0.3–110 ГГц)"
                                   % (f / 1e9, lambda_cm(lam)))
                return
            uses = ""
            for _name, _f0, _f1, _lam, _uses in IEEE_BANDS:
                if _name == band:
                    uses = _uses; break
            self.lbl_mw.config(text="f = %.4g ГГц · λ = c/f = %s · діапазон: %s · застосування: %s"
                               % (f / 1e9, lambda_cm(lam), band, uses))

        def _build_ham(self, f):
            head = ttk.LabelFrame(f, text="Діапазони радіоаматорів 160 м … 23 см (IARU Region 1)")
            head.pack(fill="x", padx=6, pady=(6, 2))
            rows = [("Діапазон", "Частота, МГц", "Основне практичне застосування")]
            for _name, _f0, _f1, _uses in HAM_BANDS:
                rows.append((_name, "%.6g – %.6g" % (_f0 / 1e6, _f1 / 1e6), _uses))
            for i, row in enumerate(rows):
                for j, txt in enumerate(row):
                    ttk.Label(head, text=txt, wraplength=(90 if j < 2 else 470), justify="left",
                              font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                              foreground=THEME["dim"] if (i == 0 or j == 0) else "#333333"
                              ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)
            calc = ttk.LabelFrame(f, text="Калькулятор: частота ↔ довжина хвилі ↔ аматорський діапазон")
            calc.pack(fill="x", padx=6, pady=2)
            r1 = ttk.Frame(calc); r1.pack(fill="x", pady=2)
            ttk.Label(r1, text="Частота, МГц:").pack(side="left")
            self.ham_f = tk.StringVar(value="14.074")
            ttk.Entry(r1, textvariable=self.ham_f, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ довжину хвилі і діапазон", command=self._ham_from_f).pack(side="left", padx=4)
            ttk.Label(r1, text="Довжина хвилі, м:").pack(side="left", padx=(16, 0))
            self.ham_l = tk.StringVar(value="21.3")
            ttk.Entry(r1, textvariable=self.ham_l, width=8).pack(side="left", padx=3)
            ttk.Button(r1, text="→ частоту і діапазон", command=self._ham_from_l).pack(side="left", padx=4)
            self.lbl_ham = ttk.Label(calc, text="", foreground=THEME["text"], wraplength=1250, justify="left")
            self.lbl_ham.pack(anchor="w", padx=2, pady=2)
            self.fig_ham = Figure(figsize=(13, 4.2), dpi=100)
            self.canvas_ham = FigureCanvasTkAgg(self.fig_ham, master=f)
            NavigationToolbar2Tk(self.canvas_ham, f)
            self.canvas_ham.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ham()
            self._ham_from_f()

        def _draw_ham(self):
            self.fig_ham.clear()
            draw_ham_spectrum(self.fig_ham)
            self.canvas_ham.draw_idle()

        def _ham_from_f(self):
            try:
                f = float(self.ham_f.get().replace(",", ".")) * 1e6
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна частота"); return
            if f <= 0:
                messagebox.showerror(APP_TITLE, "Частота має бути більшою за 0 МГц"); return
            self.ham_l.set("%.4g" % freq_to_lambda(f))
            self._ham_report(f)

        def _ham_from_l(self):
            try:
                lam = float(self.ham_l.get().replace(",", "."))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна довжина хвилі"); return
            if lam <= 0:
                messagebox.showerror(APP_TITLE, "Довжина хвилі має бути більшою за 0 м"); return
            f = 299792458.0 / lam
            self.ham_f.set("%.4g" % (f / 1e6))
            self._ham_report(f)

        def _ham_report(self, f):
            lam = freq_to_lambda(f)
            band = ham_band_for(f)
            if band is None:
                self.lbl_ham.config(text="f = %.4g МГц · λ = c/f = %.4g м · поза аматорськими діапазонами (між смугами 160 м … 23 см)"
                                   % (f / 1e6, lam))
                return
            uses = ""
            for _name, _f0, _f1, _uses in HAM_BANDS:
                if _name == band:
                    uses = _uses; break
            self.lbl_ham.config(text="f = %.4g МГц · λ = c/f = %.4g м · діапазон: %s · застосування: %s"
                               % (f / 1e6, lam, band, uses))

        def _ew_preset(self, what, rows_spec, achx_table=None):
            for en, kd, vals in self.gen_rows:
                en.set(0)
            for i, (kind, f, amp, width) in enumerate(rows_spec):
                en, kd, vals = self.gen_rows[i]
                en.set(1); kd.set(kind)
                vals[0].set("%.6g" % (f / 1e6))
                vals[2].set("%.6g" % amp)
                vals[3].set("%.6g" % width)
            self.ex_per.set("0"); self.ex_pw.set("0")
            self.ex_rise.set("0"); self.ex_fall.set("0")
            self.ex_am_m.set("0"); self.ex_am_f.set("0")
            self.ex_chirp.set("0"); self.ex_chirp_per.set("0")
            if achx_table:
                self.achx_on.set(True)
                self.cb_achx.set("Користувацький")
                self.txt_achx.delete("1.0", "end")
                self.txt_achx.insert("1.0", achx_table)
            else:
                self.achx_on.set(False)
            self._set_status("пресет «%s» → Генератор: «Створити .raw…»" % what)
            self.nb_gen.select(self.gen_frame)

        def _build_ew(self, f):
            def table(fr, title, rows, widths):
                head = ttk.LabelFrame(fr, text=title)
                head.pack(fill="x", padx=6, pady=(6, 2))
                for i, row in enumerate(rows):
                    for j, txt in enumerate(row):
                        ttk.Label(head, text=txt, wraplength=widths[j], justify="left",
                                  font=("DejaVu Sans", 9, "bold") if i == 0 else ("DejaVu Sans", 9),
                                  foreground=THEME["dim"] if i == 0 else "#333333"
                                  ).grid(row=i, column=j, sticky="nw", padx=5, pady=1)

            table(f, "Типи модуляції",
                  [("Тип", "Сутність", "Типове застосування")] + MOD_TYPES,
                  (170, 560, 300))
            table(f, "Радіоелектронне придушення та обман",
                  [("Метод", "Сутність", "Типовий захист")] + EW_METHODS,
                  (240, 560, 300))
            head = ttk.LabelFrame(f, text="Блокування: клітка Фарадея")
            head.pack(fill="x", padx=6, pady=(6, 2))
            ttk.Label(head, text=FARADAY_NOTE, wraplength=1180, justify="left",
                      font=("DejaVu Sans", 9), foreground=THEME["text"]
                      ).pack(anchor="w", padx=5, pady=2)
            head = ttk.LabelFrame(f, text="Практика (v2.1.0): технологія → Генератор")
            head.pack(fill="x", padx=6, pady=(6, 2))
            ttk.Label(head, text="Кнопка налаштовує вкладку «Генератор» (рядки сигналу, додаткові параметри, АЧХ) і відкриває її: далі «Створити .raw…» — і технологія опиняється у записі.",
                      wraplength=1180, justify="left", font=("DejaVu Sans", 9), foreground=THEME["dim"]
                      ).pack(anchor="w", padx=5, pady=2)
            spot_tbl = ("100000000 -80\n900000000 -80\n990000000 -3\n1000000000 0\n"
                        "1010000000 -3\n1100000000 -80\n6000000000 -80")
            far_tbl = "100000000 -60\n3000000000 -60\n6000000000 -60"

            def _am():
                self._ew_preset("АМ", [("Синус", 200e6, 12000, 0)])
                self.ex_am_m.set("0.5"); self.ex_am_f.set("1000")

            def _fm():
                self._ew_preset("ЧМ", [("Синус", 200e6, 12000, 0)])
                self.ex_chirp.set("2"); self.ex_chirp_per.set("1000")

            def _pm():
                self._ew_preset("ФМ (Меандр ±180°)", [("Меандр", 150e6, 8000, 0.5 / 150)])

            def _barrage():
                self._ew_preset("Barrage", [("Шум", 0, 8000, 0)])

            def _spot():
                self._ew_preset("Spot", [("Шум", 0, 12000, 0)], spot_tbl)

            def _spoof():
                self._ew_preset("Spoofing", [("Синус", 1575.42e6, 4000, 0),
                                             ("Синус", 1575.52e6, 8000, 0)])

            def _faraday():
                self._ew_preset("Клітка Фарадея", [("Синус", 300e6, 12000, 0)], far_tbl)

            br1 = ttk.Frame(head); br1.pack(anchor="w", padx=5)
            br2 = ttk.Frame(head); br2.pack(anchor="w", padx=5, pady=(2, 4))
            for i, (lbl, fn) in enumerate([
                    ("АМ — m=0.5, 1 кГц", _am),
                    ("ЧМ/FM — свейп 2 МГц", _fm),
                    ("ФМ/PM — Меандр ±180°", _pm),
                    ("Barrage — шум по всій смузі", _barrage),
                    ("Spot — прицільний 1 ГГц", _spot),
                    ("Spoofing — підробний +6 дБ", _spoof),
                    ("Клітка Фарадея — −60 дБ", _faraday)]):
                ttk.Button(br1 if i < 4 else br2, text=lbl, command=fn).pack(side="left", padx=3, pady=2)
            self.fig_ew = Figure(figsize=(13, 5.8), dpi=100)
            self.canvas_ew = FigureCanvasTkAgg(self.fig_ew, master=f)
            NavigationToolbar2Tk(self.canvas_ew, f)
            self.canvas_ew.get_tk_widget().pack(fill="both", expand=True)
            self._draw_ew()

        def _draw_ew(self):
            self.fig_ew.clear()
            draw_mod_ew_spectrum(self.fig_ew)
            self.canvas_ew.draw_idle()

        def _set_mode(self):
            if self.mode_var.get() == "gen":
                self.nb_ana.pack_forget()
                self.nb_gen.pack(fill="both", expand=True)
            else:
                self.nb_gen.pack_forget()
                self.nb_ana.pack(fill="both", expand=True)

        def _build_manual(self, f, key="ana"):
            txt = tk.Text(f, wrap="word", font=("DejaVu Sans", 10),
                          padx=12, pady=10, spacing1=2, spacing3=4)
            ysb = ttk.Scrollbar(f, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=ysb.set)
            ysb.pack(side="right", fill="y")
            txt.pack(fill="both", expand=True, padx=(2, 0), pady=2)
            txt.tag_configure("h1", font=("DejaVu Sans", 12, "bold"),
                              foreground="#1a4d7a", spacing1=10, spacing3=6)
            txt.tag_configure("h2", font=("DejaVu Sans", 10, "bold"),
                              foreground=THEME["text"], spacing1=6)
            txt.tag_configure("lnk", foreground="#1a4d7a", underline=1)
            txt.tag_bind("lnk", "<Enter>", lambda _e: txt.configure(cursor="hand2"))
            txt.tag_bind("lnk", "<Leave>", lambda _e: txt.configure(cursor=""))
            txt.configure(state="normal")
            txt.insert("end", "RAW Studio v%s — інструкція користувача\n" % APP_VERSION, ("h1",))
            txt.insert("end", "Автор: %s\n" % APP_AUTHOR)
            secs = [_l[3:] for _l in MANUAL_TEXT.split("\n") if _l.startswith("## ")]
            txt.insert("end", "\nЗМІСТ (клікніть розділ):\n", ("h2",))
            marks = []
            for _i, _title in enumerate(secs):
                txt.tag_bind("lnk%d" % _i, "<Button-1>",
                             lambda _e, _i=_i, _t=txt, _mk=marks: self._manual_goto(_t, _mk, _i))
                txt.insert("end", "→ %s\n" % _title, ("lnk", "lnk%d" % _i))
            txt.insert("end", "\n")
            body_start = txt.index("end-1c")
            for _line in MANUAL_TEXT.split("\n"):
                if _line.startswith("## "):
                    txt.insert("end", "\n" + _line[3:] + "\n", ("h1",))
                else:
                    txt.insert("end", _line + "\n")
            for _title in secs:
                _pos = txt.search(_title, body_start)
                marks.append(_pos if _pos else "1.0")
            txt.configure(state="disabled")
            if key == "gen":
                self.txt_manual_gen = txt
            else:
                self.txt_manual = txt
            self._manuals.append((txt, marks))

        def _manual_goto(self, txt, marks, i):
            txt.see(marks[i])
            txt.update_idletasks()

        def _show_lib(self):
            ent = LIB_FLAT.get(self.cb_lib.get())
            if ent:
                self.lbl_lib.config(text=ent["note"])

        def _apply_lib(self):
            ent = LIB_FLAT.get(self.cb_lib.get())
            if not ent:
                return
            targets = [(en, kind, vals) for en, kind, vals in self.gen_rows if bool(en.get())]
            if not targets:
                targets = [self.gen_rows[0]]
                self._set_status("бібліотека: увімкнених рядків немає — параметри записані в рядок 1 (увімкніть його)")
            _uni = not any(float(ex0 or 0) != 0 for ex0 in
                           (ent["extra"].get("per_us"), ent["extra"].get("pw_us"),
                            ent["extra"].get("rise_us"), ent["extra"].get("fall_us")))
            for en, kind, vals in targets:
                kind.set(ent["kind"])
                vals[0].set("%.6g" % (ent["freq"] / 1e6))
                if ent.get("width_us"):
                    vals[3].set("%.6g" % ent["width_us"])
                try:
                    _amp = float(vals[2].get() or 0)
                except ValueError:
                    _amp = 0.0
                dc, gn = self._auto_dc_gain(ent["kind"], ent["freq"], _amp, _uni)
                vals[4].set(str(dc))
                vals[5].set("%.6g" % gn)
            ex = ent["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.ex_jam_bw.set("%.6g" % (ex.get("jam_bw", 0) / 1e6))
            self.ex_sp_off.set("%.6g" % ex.get("sp_off", 0))
            self.ex_sp_drift.set("%.6g" % ex.get("sp_drift", 0))
            self.gen_fs.set("%.6g" % (ent["fs_hint"] / 1e6))
            self._set_status("бібліотека «%s» → %d канал(ів), %s, DC %d, підсилення %.6g: %s"
                             % (self.cb_lib.get(), len(targets), ent["kind"], dc, gn, ent["note"]))
            self._auto_block(ent["freq"], "«%s»" % self.cb_lib.get())
            self._guide("Шаблон з бібліотеки застосовано. Далі: перевірте таблицю → «Крок 5 · Створити .raw…».")

        def _show_sig(self):
            pr = SIG_PRESETS.get(self.cb_sig.get())
            if pr:
                self.lbl_sig.config(text=pr["label"])
                return
            gr = SIG_GROUPS.get(self.cb_sig.get())
            if gr:
                self.lbl_sig.config(text="група: %d сигналів (перелік — у статусному рядку)" % len(gr))

        def _apply_sig(self):
            name = self.cb_sig.get()
            pr = SIG_PRESETS.get(name)
            if not pr:
                self._apply_sig_group(name)
                grp2 = SIG_GROUPS.get(name)
                if grp2:
                    self._auto_block(SIG_PRESETS[grp2[0]]["freq"], "група «%s»" % name)
                return
            targets = [(en, kind, vals) for en, kind, vals in self.gen_rows if bool(en.get())]
            if not targets:
                targets = [self.gen_rows[0]]
                self._set_status("тип сигналу: увімкнених рядків немає — параметри записані в рядок 1 (увімкніть його)")
            _uni = not any(float(ex0 or 0) != 0 for ex0 in
                           (pr["extra"].get("per_us"), pr["extra"].get("pw_us"),
                            pr["extra"].get("rise_us"), pr["extra"].get("fall_us")))
            for en, kind, vals in targets:
                kind.set(pr["kind"])
                vals[0].set("%.6g" % (pr["freq"] / 1e6))
                if pr.get("width_us"):
                    vals[3].set("%.6g" % pr["width_us"])
                try:
                    _amp = float(vals[2].get() or 0)
                except ValueError:
                    _amp = 0.0
                dc, gn = self._auto_dc_gain(pr["kind"], pr["freq"], _amp, _uni)
                vals[4].set(str(dc))
                vals[5].set("%.6g" % gn)
            ex = pr["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.ex_jam_bw.set("%.6g" % (ex.get("jam_bw", 0) / 1e6))
            self.ex_sp_off.set("%.6g" % ex.get("sp_off", 0))
            self.ex_sp_drift.set("%.6g" % ex.get("sp_drift", 0))
            self.gen_fs.set("%.6g" % (pr["fs_hint"] / 1e6))
            self._set_status("тип сигналу «%s» → %d канал(ів), %s, DC %d, підсилення %.6g: %s"
                             % (name, len(targets), pr["kind"], dc, gn, pr["note"]))
            self._auto_block(pr["freq"], "тип сигналу «%s»" % name)
            self._guide("Шаблон застосовано. Далі: перевірте таблицю → «Крок 5 · Створити .raw…».")

        def _apply_sig_group(self, name):
            grp = SIG_GROUPS.get(name)
            if not grp:
                return
            need = min(len(grp), len(self.gen_rows))
            rows = []
            for _i, (_en, _kd, _vals) in enumerate(self.gen_rows):
                if _i in getattr(self, "audio_rows", ()):
                    continue
                if bool(_en.get()):
                    rows.append((_en, _kd, _vals))
                elif len(rows) < need:
                    _en.set(True)
                    rows.append((_en, _kd, _vals))
            fs_max = max(SIG_PRESETS[m]["fs_hint"] for m in grp)
            used = 0
            for m in grp:
                if used >= len(rows):
                    break
                p = SIG_PRESETS[m]
                en, kind, vals = rows[used]
                kind.set(p["kind"])
                vals[0].set("%.6g" % (p["freq"] / 1e6))
                if p.get("width_us"):
                    vals[3].set("%.6g" % p["width_us"])
                try:
                    _amp = float(vals[2].get() or 0)
                except ValueError:
                    _amp = 0.0
                _uni = not any(float(ex0 or 0) != 0 for ex0 in
                               (p["extra"].get("per_us"), p["extra"].get("pw_us"),
                                p["extra"].get("rise_us"), p["extra"].get("fall_us")))
                dc, gn = self._auto_dc_gain(p["kind"], p["freq"], _amp, _uni)
                vals[4].set(str(dc))
                vals[5].set("%.6g" % gn)
                used += 1
            ex = SIG_PRESETS[grp[0]]["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.ex_chirp_per.set("%.6g" % ex.get("chirp_per_us", 0))
            self.ex_jam_bw.set("%.6g" % (ex.get("jam_bw", 0) / 1e6))
            self.ex_sp_off.set("%.6g" % ex.get("sp_off", 0))
            self.ex_sp_drift.set("%.6g" % ex.get("sp_drift", 0))
            self.gen_fs.set("%.6g" % (fs_max / 1e6))
            self._set_status("група «%s» → %d рядків: %s; fs = %.6g МГц"
                             % (name, used, "; ".join(grp), fs_max / 1e6))
            self._guide("Групу застосовано (%d рядків). Далі: перевірте таблицю → «Крок 5 · Створити .raw…»." % used)

        def _build_iq(self, fiq):
            r1 = ttk.Frame(fiq); r1.pack(fill="x", pady=(8, 2))
            ttk.Label(r1, text="Крок 1 · Файл I/Q (RTSA-Suite: .iq + .iq.xml, .wav, raw):").pack(side="left")
            self.iq_path = tk.StringVar(value="")
            ttk.Entry(r1, textvariable=self.iq_path, width=42).pack(side="left", padx=4)
            ttk.Button(r1, text="Огляд…", command=self._iq_pick).pack(side="left")
            r2 = ttk.Frame(fiq); r2.pack(fill="x", pady=2)
            ttk.Label(r2, text="Крок 2 · тип даних:").pack(side="left")
            self.iq_dtype = ttk.Combobox(r2, values=list(RTSA_IQ_DTYPES.keys()),
                                         width=9, state="readonly")
            self.iq_dtype.current(0); self.iq_dtype.pack(side="left", padx=3)
            ttk.Label(r2, text="fs I/Q, МС/с:").pack(side="left", padx=(10, 0))
            self.iq_fs = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fs, width=10).pack(side="left", padx=3)
            ttk.Label(r2, text="центр, МГц:").pack(side="left", padx=(10, 0))
            self.iq_fc = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fc, width=10).pack(side="left", padx=3)
            r3 = ttk.Frame(fiq); r3.pack(fill="x", pady=2)
            ttk.Label(r3, text="довжина канала .raw, відліків:").pack(side="left")
            self.iq_n = tk.StringVar(value="16384")
            ttk.Entry(r3, textvariable=self.iq_n, width=9).pack(side="left", padx=3)
            ttk.Label(r3, text="каналів у .raw:").pack(side="left", padx=(10, 0))
            self.iq_nch = tk.StringVar(value="8")
            ttk.Entry(r3, textvariable=self.iq_nch, width=5).pack(side="left", padx=3)
            ttk.Label(r3, text="fs .raw, МГц (або 12e9):").pack(side="left", padx=(10, 0))
            self.iq_fsout = tk.StringVar(value="12000")
            ttk.Entry(r3, textvariable=self.iq_fsout, width=10).pack(side="left", padx=3)
            r4 = ttk.Frame(fiq); r4.pack(fill="x", pady=(6, 2))
            ttk.Button(r4, text="🔓 Розпакувати .rtsa → .iq + .iq.xml",
                       command=self._rtsa_extract).pack(side="left")
            self.btn_iq = ttk.Button(r4, text="Крок 3 · Конвертувати → .raw…", command=self._iq_convert)
            ttk.Button(r4, text="⚡ Завада з IQ… — сигнал і спектр",
                       command=self._iq_jam_open).pack(side="left", padx=6)
            self.btn_iq.pack(side="left")
            self.btn_iq.config(state="disabled")
            ttk.Label(fiq, text="Запис RTSA-Suite: експортуйте Raw IQ (.iq) з File Source; "
                                "центр і fs підуть із сусіднього .iq.xml, інакше введіть вручну "
                                "(поля в МГц). «Розпакувати .rtsa» створить .iq + .iq.xml "
                                "(стиснуті пакети — закритий алгоритм: RSTAFileTool repair -compress=0). Конвертор "
                                "розкладає середній спектр запису по каналах плану 64 МГц: "
                                "піки, рівні в кодах і зайнятість каналів відтворюються; "
                                "при відкритті .raw вкажіть ту саму fs.",
                      wraplength=780, justify="left").pack(fill="x", pady=(6, 2))

        # -- Завада з IQ: аналіз сигналу і спектру (v3.19.0) --
        def _iq_jam_open(self):
            _w = getattr(self, "iqj_win", None)
            if _w is not None:
                try:
                    if _w.winfo_exists():
                        _w.lift()
                        return
                except tk.TclError:
                    pass
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            w = tk.Toplevel(self)
            w.title("Завада з IQ — сигнал і спектр")
            w.geometry("960x900")
            w.configure(bg=THEME["panel"])
            self.iqj_win = w
            top = tk.Frame(w, bg=THEME["panel"])
            top.pack(side="top", fill="x", padx=8, pady=6)

            def _lbl(row, txt):
                tk.Label(row, text=txt, bg=THEME["panel"], fg=THEME["text"]).pack(side="left", padx=(10, 2))

            self.iqj_path = tk.StringVar(value=self.iq_path.get())
            _lbl(top, "Файл .iq:")
            tk.Entry(top, textvariable=self.iqj_path, width=34, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="🔍 Аналізувати", style="Accent.TButton",
                       command=self._iq_jam_analyze_ui).pack(side="left", padx=8)
            row2 = tk.Frame(w, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "Тип:")
            self.iqj_dtype = tk.StringVar(value=self.iq_dtype.get())
            tk.Label(row2, textvariable=self.iqj_dtype, bg=THEME["panel"], fg=THEME["dim"],
                     width=12, anchor="w").pack(side="left")
            _lbl(row2, "fs, МГц:")
            self.iqj_fs = tk.StringVar(value=self.iq_fs.get() or "2")
            tk.Entry(row2, textvariable=self.iqj_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Центр, МГц:")
            self.iqj_fc = tk.StringVar(value=self.iq_fc.get() or "100")
            tk.Entry(row2, textvariable=self.iqj_fc, width=9, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "J/S, дБ:")
            self.iqj_js = tk.StringVar(value="15")
            tk.Entry(row2, textvariable=self.iqj_js, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Офсет, дБ:")
            self.iqj_off = tk.StringVar(value=getattr(self, "iqj_off_val", "0"))
            tk.Entry(row2, textvariable=self.iqj_off, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="⚡ Пік → завада", style="Accent.TButton",
                       command=self._iq_jam_apply).pack(side="left", padx=12)
            self.iqj_res_lbl = tk.Label(w, text="Оберіть .iq і натисніть «Аналізувати»: пік, зайнята смуга, "
                                                "підлога, динаміка, рівень і параметри завади.",
                                        justify="left", anchor="w", bg=THEME["panel"], fg=THEME["text"])
            self.iqj_res_lbl.pack(side="top", fill="x", padx=10, pady=(2, 4))
            row3 = tk.Frame(w, bg=THEME["panel"])
            row3.pack(side="top", fill="x", padx=8, pady=(0, 2))
            tk.Label(row3, text="Топ-N піків (вибрані рядки — для застосування):",
                     bg=THEME["panel"], fg=THEME["text"]).pack(side="left")
            ttk.Button(row3, text="Піки → канали плану", style="Accent.TButton",
                       command=self._iqj_peaks_apply).pack(side="left", padx=10)
            self.iqj_shape = tk.BooleanVar(value=False)
            ttk.Checkbutton(row3, text="шум за профілем спектра",
                            variable=self.iqj_shape).pack(side="left")
            _iqj_cols = ("n", "f", "bw", "kind", "db")
            _st99 = ttk.Style(w)
            _st99.configure("Iqj.Treeview", background=THEME["panel2"],
                            foreground=THEME["text"], fieldbackground=THEME["panel2"],
                            rowheight=22)
            self.iqj_tree = ttk.Treeview(w, columns=_iqj_cols, show="headings", height=6,
                                         selectmode="extended", style="Iqj.Treeview")
            for _c, _t, _wd in (("n", "№", 34), ("f", "Пік, МГц", 105),
                                ("bw", "Смуга, кГц", 95), ("kind", "Тип завади", 165),
                                ("db", "Рівень, дБ", 100)):
                self.iqj_tree.heading(_c, text=_t)
                self.iqj_tree.column(_c, width=_wd, anchor="w", stretch=True)
            self.iqj_tree.pack(side="top", fill="x", padx=8, pady=(2, 4))
            self.iqj_status = tk.Label(w, text="", bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.iqj_status.pack(side="bottom", fill="x")
            self.iqj_fig = Figure(figsize=(8.9, 4.9), dpi=100)
            self.iqj_ax = self.iqj_fig.add_subplot(111)
            self.iqj_ax.set_title("Спектр IQ — запис не обрано")
            self.iqj_ax.set_xlabel("Частота, МГц")
            self.iqj_ax.set_ylabel("Рівень, дБ (відн. піка)")
            self.iqj_canvas = FigureCanvasTkAgg(self.iqj_fig, master=w)
            self.iqj_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.iqj_canvas.draw()

        def _iq_jam_analyze_ui(self):
            try:
                fs_hz = parse_freq_hz(self.iqj_fs.get())
                fc_hz = parse_freq_hz(self.iqj_fc.get())
                js = float(self.iqj_js.get())
                if not (0.0 <= js <= 60.0):
                    raise ValueError("J/S у межах 0…60 дБ")
                off = float(self.iqj_off.get())
                if not (-100.0 <= off <= 100.0):
                    raise ValueError("офсет у межах −100…100 дБ")
                self.iqj_off_val = self.iqj_off.get()
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Завада з IQ: %s" % e)
                return
            p = self.iqj_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "IQ-файл не знайдено: %s" % (p or "—"))
                return
            try:
                res = iq_jam_analyze(p, dtype=self.iqj_dtype.get(), fs_iq=fs_hz, fc=fc_hz,
                                     js_target=js, db_off=off)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Завада з IQ: %s" % e)
                return
            self.iqj_res = res
            self.iqj_spec_path = p + ".spectrum.json"
            try:
                save_spectrum_profile(self.iqj_spec_path, res["freqs_hz"], res["psd_db"], fc_hz)
            except Exception:
                self.iqj_spec_path = ""
            self.iqj_tree.delete(*self.iqj_tree.get_children())
            for _i, _pk in enumerate(res.get("peaks", ())):
                self.iqj_tree.insert("", "end", iid=str(_i), values=(
                    _i + 1, "%.6f" % (_pk["f_hz"] / 1e6), "%.4g" % (_pk["width_hz"] / 1e3),
                    _pk["kind_name"], "%.1f" % _pk["peak_db"]))
            self.iqj_ax.clear()
            self.iqj_ax.plot(res["freqs_hz"] / 1e6, res["psd_db"], lw=0.7)
            self.iqj_ax.axvline(res["f_peak_hz"] / 1e6, color="red", lw=1.2)
            self.iqj_ax.axvspan((res["f_peak_hz"] - res["width_hz"] / 2.0) / 1e6,
                                (res["f_peak_hz"] + res["width_hz"] / 2.0) / 1e6,
                                color="orange", alpha=0.25)
            if res.get("peaks"):
                self.iqj_ax.plot([pk["f_hz"] / 1e6 for pk in res["peaks"]],
                                 [pk["peak_db"] for pk in res["peaks"]], "rv", ms=6)
            self.iqj_ax.set_xlabel("Частота, МГц")
            self.iqj_ax.set_ylabel("Рівень, дБ (відн. піка)")
            self.iqj_ax.set_title("Спектр IQ: %s — %s" % (os.path.basename(p), res["note"]))
            self.iqj_fig.tight_layout()
            self.iqj_canvas.draw()
            self.iqj_res_lbl.config(text=("Пік: %.6f МГц   ·   смуга (−20 дБ): %.4g кГц   ·   підлога: %.1f дБ   "
                                          "·   пік/підлога: %.1f дБ\nРівень: %.1f dBFS + %.1f дБ = %.1f дБм   "
                                          "→   завада: %s   ·   для J/S +%.0f дБ потрібно ≈ %.1f дБм")
                                    % (res["f_peak_hz"] / 1e6, res["width_hz"] / 1e3, res["floor_db"],
                                       res["dyn_db"], res["dbfs"], res["db_off"], res["sig_dbm"], res["kind_name"], res["js"],
                                       res["jam_req_dbm"]))
            self.iqj_status.config(text="Аналіз виконано: %s" % res["note"])
            self._set_status("Завада з IQ: пік %.6f МГц, смуга %.4g кГц, %s — готово до застосування"
                             % (res["f_peak_hz"] / 1e6, res["width_hz"] / 1e3, res["kind_name"]))

        def _iq_jam_apply(self):
            res = getattr(self, "iqj_res", None)
            if not res:
                messagebox.showinfo(APP_TITLE, "Спочатку проаналізуйте IQ («Аналізувати»).")
                return
            _prot = []
            try:
                _prot = parse_protected_mhz(self.ex_protect.get())
            except (ValueError, AttributeError):
                _prot = []
            if (is_protected(res["f_hz"], _prot)
                    or is_protected(res["f_hz"] - res["width_hz"] / 2.0, _prot)
                    or is_protected(res["f_hz"] + res["width_hz"] / 2.0, _prot)):
                messagebox.showwarning(APP_TITLE,
                                       "Захищені частоти: пік %.6g МГц у захищеній смузі — застосування скасовано."
                                       % (res["f_hz"] / 1e6))
                return
            used = 0
            for en, kd, vals in self.gen_rows:
                if bool(en.get()):
                    kd.set(res["kind_name"])
                    vals[0].set("%.12g" % (res["f_hz"] / 1e6))
                    used += 1
            if res["kind"] == "jam_noise":
                self.ex_jam_bw.set("%.12g" % res["bw_mhz"])
            elif res["kind"] == "jam_chirp":
                self.ex_chirp.set("%.12g" % res["dev_mhz"])
                self.ex_chirp_per.set("%.6g" % res["per_us"])
            self._set_status("Завада з IQ: %s %.4f МГц → %d канал(ів); сигнал ≈ %.1f дБм "
                             "(%.1f dBFS + офсет %.1f дБ), для J/S +%.0f дБ потрібно ≈ %.1f дБм"
                             % (res["kind_name"], res["f_hz"] / 1e6, used, res["sig_dbm"], res["dbfs"],
                                res["db_off"], res["js"], res["jam_req_dbm"]))

        def _iqj_peaks_apply(self):
            res = getattr(self, "iqj_res", None)
            if not res or not res.get("peaks"):
                messagebox.showinfo(APP_TITLE, "Спочатку проаналізуйте IQ («Аналізувати»).")
                return
            peaks = res["peaks"]
            sel = set(self.iqj_tree.selection())
            idxs = sorted(int(i) for i in sel) if sel else list(range(len(peaks)))
            use = [peaks[i] for i in idxs if i < len(peaks)]
            if not use:
                messagebox.showinfo(APP_TITLE, "Таблиця піків порожня.")
                return
            rows = [(i, en, kd, vals) for i, (en, kd, vals) in enumerate(self.gen_rows)
                    if bool(en.get())]
            if not rows:
                messagebox.showinfo(APP_TITLE,
                                    "Увімкніть хоча б один канал Генератора (⚙ ГЕНЕРАТОР СИГНАЛУ).")
                return
            _prot = []
            try:
                _prot = parse_protected_mhz(self.ex_protect.get())
            except (ValueError, AttributeError):
                _prot = []
            _skipped = 0
            if _prot:
                _kept = []
                for _pk in use:
                    if (is_protected(_pk["f_hz"], _prot)
                            or is_protected(_pk["f_hz"] - _pk["width_hz"] / 2.0, _prot)
                            or is_protected(_pk["f_hz"] + _pk["width_hz"] / 2.0, _prot)):
                        _skipped += 1
                    else:
                        _kept.append(_pk)
                use = _kept
            if not use:
                messagebox.showinfo(APP_TITLE,
                                    "Усі вибрані піки — у захищених смугах; нічого не застосовано.")
                return
            assign, free = {}, list(range(len(use)))
            for ri, en, kd, vals in rows:
                try:
                    frow = parse_freq_hz(vals[0].get())
                except ValueError:
                    frow = -1.0
                for p_i, pk in enumerate(use):
                    if p_i in free and abs(pk["f_hz"] - frow) <= pk["width_hz"] / 2.0:
                        assign[ri] = pk
                        free.remove(p_i)
                        break
            for ri, en, kd, vals in rows:
                if ri in assign or not free:
                    continue
                assign[ri] = use[free.pop(0)]
            for ri, pk in assign.items():
                en, kd, vals = self.gen_rows[ri]
                kd.set(pk["kind_name"])
                vals[0].set("%.12g" % (pk["f_hz"] / 1e6))
            bw_max = max((pk["bw_mhz"] for pk in use if pk["kind"] == "jam_noise"), default=0.0)
            if bw_max > 0:
                self.ex_jam_bw.set("%.12g" % bw_max)
            if getattr(self, "iqj_shape", None) is not None and self.iqj_shape.get():
                self.ex_jam_shape.set(getattr(self, "iqj_spec_path", "") or "")
            else:
                self.ex_jam_shape.set("")
            _msg = "Піки → канали: %d піків → %d канал(ів)%s: %s" % (
                len(use), len(assign),
                ("; пропущено захищених: %d" % _skipped) if _skipped else "",
                ", ".join("ch%d←%.6g МГц" % (ri, pk["f_hz"] / 1e6)
                          for ri, pk in sorted(assign.items())))
            self._set_status(_msg)
            self.iqj_status.config(text=_msg)

        def _iqj_offset_value(self):
            try:
                return float(getattr(self, "iqj_off_val", "0"))
            except Exception:
                return 0.0

        def _iqj_offset_set(self, v):
            try:
                self.iqj_off_val = "%.6g" % float(v)
            except Exception:
                self.iqj_off_val = "0"
            try:
                self.iqj_off.set(self.iqj_off_val)
            except Exception:
                pass

        def _rtsa_extract(self, p=None):
            """Розпаковує .rtsa → .iq + .iq.xml поруч із записом (v4.8.0)."""
            if not p:
                p = filedialog.askopenfilename(
                    title="Запис .rtsa (RTSA-Suite)",
                    filetypes=[("RTSA записи", "*.rtsa"), ("Усі файли", "*.*")])
                if not p:
                    return
            base = os.path.splitext(p)[0]
            iq_p, xml_p = base + ".iq", base + ".iq.xml"
            self._set_status("I/Q: розпакування .rtsa — читання чанків…")
            self.update_idletasks()

            def _prog48(msg):
                self._set_status(msg)
                self.update_idletasks()

            try:
                n = rtsa_to_iq_pair(p, iq_p, xml_p, progress=_prog48)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                self._set_status("I/Q: .rtsa не розпаковано")
                return
            self.iq_path.set(iq_p)
            info = parse_rtsa_xml(xml_p)
            if info:
                if info.get("fs_iq"):
                    self.iq_fs.set("%.6g" % (info["fs_iq"] / 1e6))
                if info.get("fc") is not None:
                    self.iq_fc.set("%.6g" % (info["fc"] / 1e6))
                if info.get("dtype") in RTSA_IQ_DTYPES:
                    self.iq_dtype.set(info["dtype"])
            self._set_status("I/Q: .rtsa → .iq + .iq.xml (%d відліків), поля заповнено з .iq.xml" % n)
            self.btn_iq.config(state="normal")

        def _iq_pick(self):
            self._guide("Крок 2: перевірте тип даних, fs I/Q і центр → «Крок 3 · Конвертувати → .raw…».")
            self.btn_iq.config(state="normal")
            p = filedialog.askopenfilename(
                title="I/Q запис RTSA-Suite",
                filetypes=[("I/Q записи", "*.iq *.wav *.dat *.raw *.f32 *.rtsa"),
                           ("Усі файли", "*.*")])
            if not p:
                return
            if p.lower().endswith(".rtsa"):
                self.iq_path.set(p)
                self.btn_iq.config(state="normal")
                self._set_status("I/Q: .rtsa вибрано — «Розпакувати .rtsa» або Крок 3 створять .iq + .iq.xml")
                return
            self.iq_path.set(p)
            info = parse_rtsa_xml(p + ".xml")
            if info is None:
                info = parse_rtsa_xml(os.path.splitext(p)[0] + ".xml")
            if info:
                if info.get("fs_iq"):
                    self.iq_fs.set("%.6g" % (info["fs_iq"] / 1e6))
                if info.get("fc") is not None:
                    self.iq_fc.set("%.6g" % (info["fc"] / 1e6))
                if info.get("dtype") in RTSA_IQ_DTYPES:
                    self.iq_dtype.set(info["dtype"])
                self._set_status("I/Q: метадані взято з .iq.xml")
            else:
                self._set_status("I/Q: .iq.xml не знайдено — введіть fs і центр вручну")

        def _iq_convert(self):
            p = self.iq_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Вкажіть існуючий файл I/Q")
                return
            if p.lower().endswith(".rtsa"):
                self._rtsa_extract(p)
                p = self.iq_path.get()
                if not p or p.lower().endswith(".rtsa"):
                    return
                self._set_status("I/Q: .rtsa розпаковано у .iq + .iq.xml — конвертація…")
                self.update_idletasks()
            _bad = iq_field_errors(self.iq_fs.get(), self.iq_fc.get(),
                                   self.iq_fsout.get(), self.iq_n.get(), self.iq_nch.get())
            if _bad:
                messagebox.showerror(APP_TITLE,
                    "Невірні (або порожні) числа в полях конвертера I/Q:\n— %s\n\n"
                    "fs I/Q і центр — у МГц (напр. 61.44 і 2400); або виберіть .iq, поруч із "
                    "яким лежить .iq.xml — тоді поля заповняться автоматично."
                    % "\n— ".join(_bad))
                return
            try:
                fs_iq = parse_freq_hz(self.iq_fs.get())
                fc = parse_freq_hz(self.iq_fc.get())
                fs_out = parse_freq_hz(self.iq_fsout.get())
                n_per_ch = int(float(self.iq_n.get()))
                nch = int(float(self.iq_nch.get()))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні числа в полях конвертера I/Q")
                return
            try:
                x, meta = read_iq(p, dtype=self.iq_dtype.get(), fs_iq=fs_iq, fc=fc)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            out = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(p) or os.getcwd(),
                defaultextension=".raw",
                initialfile=os.path.splitext(os.path.basename(p))[0] + ".raw",
                title="Куди зберегти .raw", filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("конвертація I/Q → .raw…")
            self.update()
            try:
                info = iq_to_raw(x, fc, fs_iq, fs_out=fs_out, bw_hz=self.bw_hz,
                                 n_per_ch=n_per_ch, nch=nch)
                save_raw(out, info["flat"])
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            self._set_status(
                "I/Q → .raw: %s — каналів плану з сигналом: %d, %d відліків/канал, "
                "nch=%d, %.1f МБ; при відкритті вкажіть fs %.4g"
                % (os.path.basename(out), info["n_sig"], info["n"], nch,
                   os.path.getsize(out) / 1e6, fs_out))

        def _toggle_help(self):
            if self._help_open:
                self.txt_help.pack_forget()
                self.geometry("%dx%d" % (self.winfo_width(),
                                         max(600, self.winfo_height() - self.txt_help.winfo_reqheight() - 90)))
                self.btn_help.config(text="📖 Довідка по параметрах — що означає кожне поле")
                self._help_open = False
            else:
                self.txt_help.pack(side="bottom", fill="both")
                self.geometry("%dx%d" % (self.winfo_width(),
                                         self.winfo_height() + self.txt_help.winfo_reqheight() + 90))
                self.btn_help.config(text="✖ Згорнути довідку")
                self._help_open = True
            self._set_status("довідка по параметрах: %s"
                             % ("розгорнуто" if self._help_open else "згорнуто"))

        def _analyze_flow(self):
            self._guide("Аналізую «%s»…" % os.path.basename(getattr(self, "path", "файл")))
            self.update_idletasks()
            self.analyze()
            self._guide("Аналіз завершено. Далі: вкладки «Спектр» / «Блоки» / "
                        "«Статистика» — або «Генератор», «I/Q → .raw», «Калібрування АЧХ».")

        def _set_status(self, s):
            self.status.set(s); self.update_idletasks()

        def _paint_banner(self, c):
            w = int(c.winfo_width() or 0)
            if w < 320:
                w = 800
            h = 56
            cols = _theme_gradient(THEME["bg"], "#123a52", 24)
            c.delete("all")
            sw = max(1, w // 24)
            for i, col in enumerate(cols):
                c.create_rectangle(i * sw, 0, (i + 1) * sw + 1, h, outline="", fill=col)
            c.create_rectangle(0, h - 3, w, h, outline="", fill=THEME["accent"])
            c.create_text(14, 12, anchor="nw", text="RAW STUDIO",
                          font=("DejaVu Sans", 15, "bold"), fill="#eaf6ff")
            c.create_text(16, 36, anchor="nw", text=APP_TITLE,
                          font=("DejaVu Sans", 8), fill=THEME["dim"])
            vx = w - 12
            c.create_rectangle(vx - 72, 16, vx, 42, outline="", fill=THEME["edge_lo"])
            c.create_rectangle(vx - 74, 14, vx - 2, 40, outline=THEME["edge_hi"],
                               fill="#123a52", width=1)
            c.create_text(vx - 38, 27, text="v%s" % APP_VERSION,
                          font=("DejaVu Sans", 10, "bold"), fill=THEME["accent"])
            self._ban_w = w

        def _repaint_banner(self, _e):
            c = getattr(self, "_ban_ref", None)
            if c is None:
                return
            try:
                if not c.winfo_exists():
                    return
            except tk.TclError:
                return
            if abs(c.winfo_width() - getattr(self, "_ban_w", 0)) > 4:
                self._paint_banner(c)

        def _switch_theme(self):
            name = {"Техно": "techno", "Класична": "classic", "День": "day",
                    "AMOLED": "amoled", "Windows XP": "xp", "Авто (год)": "auto"}.get(self.cb_theme.get(), "techno")
            self.theme_saved = name
            self.theme_name = _resolve_theme(name)
            _save_theme_name(name)
            _apply_app_theme(self, self.theme_name)
            self._apply_mpl_theme(self.theme_name == "techno")
            self._tray_update()
            self._redraw_figs()
            self._paint_banner(self._ban_ref)
            self._set_status("Тема: %s → %s" % (
                {"techno": "техно", "classic": "класична", "day": "день",
                 "amoled": "AMOLED", "auto": "авто за годинником", "xp": "Windows XP"}[name],
                {"techno": "техно", "classic": "класична", "day": "день",
                 "amoled": "AMOLED", "xp": "Windows XP"}[self.theme_name]))

        def _theme_tick(self):
            """Живе авто-перемикання: у режимі «Авто» стежимо за годинником щохвилини."""
            try:
                if getattr(self, "theme_saved", "") == "auto":
                    _r = _resolve_theme("auto")
                    if _r != self.theme_name:
                        self.cb_theme.set("Авто (год)")
                        self._switch_theme()
            except Exception:
                pass
            try:
                self.after(60000, self._theme_tick)
            except Exception:
                pass

        def _tray_setup(self):
            """Іконка в системному треї (опційно: потребує pystray; без нього — тихо пропускається)."""
            try:
                import pystray
            except Exception:
                return
            _img = _theme_icon_img()
            if _img is None:
                return
            try:
                from pystray import Menu as _M, MenuItem as _MI
                import threading as _thr
                self._tray = pystray.Icon("raw_studio", _img, "RAW Studio",
                                          _M(_MI("Показати", self._tray_show, default=True),
                                             _MI("Наступна тема", self._tray_next),
                                             _M.SEPARATOR,
                                             _MI("Вихід", self._tray_quit)))
                _thr.Thread(target=self._tray.run, daemon=True).start()
            except Exception:
                self._tray = None

        def _tray_show(self, *_a):
            try:
                self.after(0, self._tray_show_ui)
            except Exception:
                pass

        def _tray_show_ui(self):
            try:
                self.deiconify()
                self.lift()
                self.focus_force()
            except Exception:
                pass

        def _tray_next(self, *_a):
            try:
                self.after(0, self._tray_next_ui)
            except Exception:
                pass

        def _tray_next_ui(self):
            try:
                _ord = ["techno", "classic", "day", "amoled", "xp", "auto"]
                _nm = {"techno": "Техно", "classic": "Класична", "day": "День",
                       "amoled": "AMOLED", "xp": "Windows XP", "auto": "Авто (год)"}
                _nxt = _ord[(_ord.index(getattr(self, "theme_saved", "techno")) + 1) % len(_ord)]
                self.cb_theme.set(_nm[_nxt])
                self._switch_theme()
            except Exception:
                pass

        def _tray_update(self):
            """Оновлюємо іконку трея під нову тему (якщо трей запущено)."""
            ic = getattr(self, "_tray", None)
            if ic is not None:
                try:
                    ic.icon = _theme_icon_img()
                except Exception:
                    pass

        def _tray_quit(self, *_a):
            try:
                self.after(0, self._tray_quit_ui)
            except Exception:
                pass

        def _tray_quit_ui(self):
            ic = getattr(self, "_tray", None)
            if ic is not None:
                try:
                    ic.stop()
                except Exception:
                    pass
            try:
                self.destroy()
            except Exception:
                pass


        def _apply_mpl_theme(self, dark):
            import matplotlib
            if dark:
                matplotlib.rcParams.update({
                    "figure.facecolor": THEME["panel"],
                    "axes.facecolor": THEME["panel2"],
                    "savefig.facecolor": THEME["panel"],
                    "axes.edgecolor": THEME["edge_hi"],
                    "axes.labelcolor": THEME["text"],
                    "xtick.color": THEME["dim"], "ytick.color": THEME["dim"],
                    "text.color": THEME["text"],
                    "grid.color": THEME["edge_hi"],
                })
            else:
                matplotlib.rcdefaults()
                matplotlib.rcParams["font.family"] = "DejaVu Sans"
            for f in ("fig_t", "fig_s", "fig_b", "fig_ir", "fig_mw", "fig_ham",
                      "fig_ew", "fig_c", "si_fig"):
                fig = getattr(self, f, None)
                if fig is None:
                    continue
                try:
                    fig.patch.set_facecolor(THEME["panel"])
                    for ax in fig.get_axes():
                        ax.set_facecolor(THEME["panel2"])
                        ax.xaxis.label.set_color(THEME["text"])
                        ax.yaxis.label.set_color(THEME["text"])
                        for t in (ax.get_xticklabels() + ax.get_yticklabels()):
                            t.set_color(THEME["dim"])
                        for sp in ax.spines.values():
                            sp.set_color(THEME["edge_hi"] if dark else THEME["edge_lo"])
                        try:
                            ax.grid(color=THEME["edge_hi"] if dark else "#b0b0b0",
                                    alpha=0.4 if dark else 1.0)
                        except Exception:
                            pass
                except Exception:
                    pass

        def _redraw_figs(self):
            for name, guard in (("redraw_time", lambda: getattr(self, "n", 0)),
                                ("_draw_ir", None), ("_draw_mw", None),
                                ("_draw_ham", None), ("_draw_ew", None)):
                fn = getattr(self, name, None)
                if fn is None:
                    continue
                try:
                    if guard is None or guard():
                        fn()
                except Exception:
                    pass
            try:
                if getattr(self, "si_x", None) is not None:
                    self._specimg_preview()
            except Exception:
                pass

        def open_file(self):
            p = filedialog.askopenfilename(title="Виберіть .raw файл",
                                           filetypes=[("RAW файли", "*.raw"), ("Усі файли", "*.*")])
            if p:
                self._load(p)

        def _load(self, p):
            self.btn_analyze.config(state="normal")
            self._guide("Відкрито %s. Перевірте «2 · fs, МГц» і каналів → «3 · Аналізувати» "
                        "(якщо потрібно) — далі вкладки аналізу." % os.path.basename(p))
            try:
                self.fs = parse_freq_hz(self.fs_var.get())
                self.nch = int(self.nch_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні fs або кількість каналів"); return
            self._set_status("читаю %s …" % os.path.basename(p))
            try:
                self.chans, self.n = load_raw(p, self.nch)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося прочитати: %s" % e); return
            self.path = p
            self.lbl_path.config(text=p)
            if len(self.gen_rows) != self.nch:
                for w in self.gen_frame.winfo_children():
                    w.destroy()
                self._build_generator(self.gen_frame)
            self.analyze()

        def analyze(self):
            if self.chans is None:
                return
            self._set_status("аналізую…")
            self.fig_t.clear(); self.fig_s.clear(); self.fig_b.clear()
            draw_time(self.fig_t, self.chans, self.fs, 0.0, min(1e-5, self.n / self.fs))
            self.fig_s.clear()
            _axp = None
            try:
                _gs2 = self.fig_s.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
                draw_spectrum(self.fig_s, self.chans, self.fs, gs=_gs2[0])
                _axp = self.fig_s.add_subplot(_gs2[1], projection="polar")
            except Exception:
                self.fig_s.clear()
                draw_spectrum(self.fig_s, self.chans, self.fs)
            if _axp is not None:
                self._draw_ant_pattern(_axp)
            self.occ = channel_occupancy(self.chans, self.fs, self.blocks)
            info = draw_blocks(self.fig_b, self.chans, self.fs, self.blocks, self.occ)
            self.canvas_t.draw_idle(); self.canvas_s.draw_idle(); self.canvas_b.draw_idle()
            report, _, _ = analyze_report(self.path, self.fs, self.nch)
            report += "\n\n--- Зайнятість %.6g-МГц каналів по блоках ---\n" % (self.bw_hz / 1e6) + "\n".join(info)
            fc_of = {c["global"]: c["fc"] for b in self.blocks for c in b["channels"]}
            occ_list = [gg for gg in sorted(self.occ) if self.occ[gg]["occupied"]]
            if occ_list:
                report += "\nЗайняті канали: " + ", ".join(
                    "К%03d(%.0f МГц)" % (gg, fc_of[gg] / 1e6) for gg in occ_list)
            else:
                report += "\nЗайняті канали: немає"
            self.txt.delete("1.0", "end"); self.txt.insert("1.0", report)
            fv, fu = human_freq(self.n / self.fs)
            self._set_status("файл: %s — %d відліків/канал, %.4g %s"
                             % (os.path.basename(self.path), self.n, fv, fu))
            self._refresh_ch_occ()

        def redraw_time(self):
            if self.chans is None:
                return
            try:
                t0 = float(self.t0_var.get()); dur = float(self.dur_var.get())
            except ValueError:
                return
            self.fig_t.clear()
            draw_time(self.fig_t, self.chans, self.fs, t0, dur)
            self.canvas_t.draw_idle()

        # ---------- калібрування АЧХ ----------
        def _build_calibration(self, parent):
            fr = ttk.Frame(parent, padding=8); fr.pack(fill="x")
            self.cal_unit = tk.StringVar(value="МГц")
            ttk.Button(fr, text="Одиниці: МГц ⇄ Гц", width=18,
                       command=self._toggle_cal_unit).pack(side="left", padx=(0, 10))
            self.lbl_cal_fs = ttk.Label(fr, text="fs, МГц (або 12e9):"); self.lbl_cal_fs.pack(side="left")
            self.cal_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(fr, textvariable=self.cal_fs, width=11).pack(side="left", padx=3)
            ttk.Label(fr, text="каналів:").pack(side="left")
            self.cal_nch = tk.StringVar(value=str(DEFAULT_NCH))
            ttk.Entry(fr, textvariable=self.cal_nch, width=4).pack(side="left", padx=3)
            ttk.Label(fr, text="тонів:").pack(side="left")
            self.cal_tones = tk.StringVar(value="32")
            ttk.Entry(fr, textvariable=self.cal_tones, width=5).pack(side="left", padx=3)
            self.lbl_cal_fmin = ttk.Label(fr, text="від, МГц:"); self.lbl_cal_fmin.pack(side="left")
            self.cal_fmin = tk.StringVar(value="100")
            ttk.Entry(fr, textvariable=self.cal_fmin, width=9).pack(side="left", padx=3)
            self.lbl_cal_fmax = ttk.Label(fr, text="до, МГц:"); self.lbl_cal_fmax.pack(side="left")
            self.cal_fmax = tk.StringVar(value="6000")
            ttk.Entry(fr, textvariable=self.cal_fmax, width=9).pack(side="left", padx=3)
            ttk.Label(fr, text="ампл. тону, кодів:").pack(side="left")
            self.cal_amp = tk.StringVar(value="8000")
            ttk.Entry(fr, textvariable=self.cal_amp, width=7).pack(side="left", padx=3)
            ttk.Label(fr, text="відліків/тон:").pack(side="left")
            self.cal_seg = tk.StringVar(value="16384")
            ttk.Entry(fr, textvariable=self.cal_seg, width=8).pack(side="left", padx=3)
            row2 = ttk.Frame(parent, padding=(8, 0)); row2.pack(fill="x")
            ttk.Button(row2, text="1) Створити калібрувальний .raw…",
                       command=self.cal_generate).pack(side="left")
            ttk.Button(row2, text="2) Виміряти АЧХ із записаного файла…",
                       command=self.cal_measure).pack(side="left", padx=8)
            ttk.Button(row2, text="3) → Профіль у Генератор (компенсація −дБ / як виміряно)",
                       command=self.cal_to_profile).pack(side="left")
            row3 = ttk.Frame(parent, padding=(8, 4)); row3.pack(fill="x")
            ttk.Button(row3, text="💾 Зберегти виміряний профіль АЧХ у файл…",
                       command=self.cal_save_profile).pack(side="left")
            ttk.Button(row3, text="📁 Завантажити профіль АЧХ з файлу → Генератор",
                       command=self.cal_load_profile).pack(side="left", padx=8)
            self.fig_c = Figure(figsize=(12.5, 4.6), dpi=100)
            self.canvas_c = FigureCanvasTkAgg(self.fig_c, master=parent)
            self.canvas_c.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
            self.txt_cal = tk.Text(parent, height=9, font=("DejaVu Sans Mono", 9))
            self.txt_cal.pack(fill="x", padx=8, pady=(0, 8))
            ttk.Label(parent, foreground="#555", wraplength=1250, justify="left",
                      text="Методика: створіть калібрувальний файл (ступінчастий прогін тонів, "
                           "кожен тон точно в біні FFT), проганіть через реальний тракт і запишіть "
                           "відповідь у .raw, виміряйте АЧХ. Профіль зберігається у текстових .achx "
                           "(рядки «частота_Гц дБ», # — коментарі; режим інтерполяції/екстраполяції — у шапці) "
                           "і завантажується назад у Генератор однією кнопкою. "
                           "Між точками профіль інтерполюється (лінійно/сплайном) — працює на будь-якій частоті. "
                           "Одиниці частот перемикаються кнопкою «Одиниці: МГц ⇄ Гц»; .plan- і .achx-файли лишаються у Гц."
                      ).pack(anchor="w", padx=8, pady=2)

        def _toggle_cal_unit(self):
            cur = self.cal_unit.get()
            for var in (self.cal_fs, self.cal_fmin, self.cal_fmax):
                try:
                    hz = parse_freq_hz(var.get())
                except ValueError:
                    continue
                var.set("%.10g" % (hz if cur == "МГц" else hz / 1e6))
            if cur == "МГц":
                self.cal_unit.set("Гц")
                self.lbl_cal_fs.config(text="fs, Гц:")
                self.lbl_cal_fmin.config(text="від, Гц:")
                self.lbl_cal_fmax.config(text="до, Гц:")
            else:
                self.cal_unit.set("МГц")
                self.lbl_cal_fs.config(text="fs, МГц (або 12e9):")
                self.lbl_cal_fmin.config(text="від, МГц:")
                self.lbl_cal_fmax.config(text="до, МГц:")
            self._set_status("вкладка «Калібрування АЧХ»: одиниці частот — %s" % self.cal_unit.get())

        def _cal_params(self):
            return (parse_freq_hz(self.cal_fs.get()), int(self.cal_nch.get()),
                    int(self.cal_tones.get()), parse_freq_hz(self.cal_fmin.get()),
                    parse_freq_hz(self.cal_fmax.get()), float(self.cal_amp.get()),
                    int(self.cal_seg.get()))

        def cal_generate(self):
            try:
                fs_c, nch_c, tones, fmin, fmax, amp, seg_n = self._cal_params()
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні параметри калібрування"); return
            freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
            out = filedialog.asksaveasfilename(defaultextension=".raw",
                                               initialfile="calibration_sweep.raw",
                                               title="Куди зберегти калібрувальний .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("генерую калібрування: %d тонів…" % len(freqs))
            save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
            save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
            self._set_status("калібрувальний файл: %s (+ .plan)" % os.path.basename(out))
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "План тонів (бін FFT = %.6g МГц):\n%s"
                                % (bin_hz / 1e6, "\n".join("  %.6g МГц" % (fq / 1e6) for fq in freqs)))

        def cal_measure(self):
            rec = filedialog.askopenfilename(title="Запис після тракту (.raw)",
                                             filetypes=[("RAW файли", "*.raw")])
            if not rec:
                return
            plan_p = filedialog.askopenfilename(title="Файл плану (.raw.plan)",
                                                filetypes=[("Plan", "*.plan"), ("Усі файли", "*.*")])
            if not plan_p:
                return
            try:
                plan = load_cal_plan(plan_p)
                res = measure_achx(rec, plan)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Помилка вимірювання: %s" % e); return
            self.measured_achx = [(fq, gdb) for fq, _, gdb in res]
            self.fig_c.clear()
            draw_cal_result(self.fig_c, res)
            self.canvas_c.draw_idle()
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "Реальна АЧХ тракту (частота, ампл. кодів, дБ відн. еталону %d):\n%s"
                                % (plan["amp"], "\n".join("  %12.6g МГц  %8.1f кодів  %+6.2f дБ" % (fq / 1e6, amp, gdb)
                                                         for fq, amp, gdb in res)))

        def cal_to_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ, вирівнює тракт);\n"
                           "Ні — зберегти як виміряно (+дБ, модель тракту)?")
            lines = "\n".join("%.6g %.3f" % (fq, -gdb if compensate else gdb)
                              for fq, gdb in self.measured_achx)
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", lines)
            self.mode_var.set("gen"); self._set_mode()
            self.nb_gen.select(self.gen_frame)
            self._set_status("профіль АЧХ (%s) перенесено в Генератор"
                             % ("компенсація −дБ" if compensate else "як виміряно"))

        def cal_save_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ);\nНі — як виміряно (+дБ)?")
            pts = [(fq, -gdb if compensate else gdb) for fq, gdb in self.measured_achx]
            out = filedialog.asksaveasfilename(defaultextension=".achx",
                                               initialfile="achx_tract.achx",
                                               title="Куди зберегти профіль АЧХ",
                                               filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt")])
            if not out:
                return
            save_achx_profile(out, pts, mode=self.achx_mode_key(),
                              extrap=self.achx_extrap_key(), meta={
                                  "fs": self.fs_var.get(),
                                  "компенсація": "так" if compensate else "ні",
                                  "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                  "source": getattr(self, "path", "-")})
            try:
                with open(out, "a", encoding="utf-8") as _fh:
                    _fh.write("# theme: %s\n" % getattr(self, "theme_saved", "techno"))
                    _fh.write("# iq_off: %.6g\n" % self._iqj_offset_value())
            except OSError:
                pass
            self._set_status("профіль АЧХ збережено: %s (%d точок)"
                             % (os.path.basename(out), len(pts)))

        def cal_load_profile(self):
            p = filedialog.askopenfilename(title="Файл профілю АЧХ",
                                           filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                pts, mode_b, ex_b = load_achx_profile(p)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Профіль АЧХ: %s" % e); return
            # v3.15.0: тема зберігається в профілі .achx
            try:
                with open(p, encoding="utf-8") as _fh:
                    _th = next((ln[8:].strip() for ln in _fh if ln.startswith("# theme:")), None)
                if _th in ("techno", "classic", "day", "amoled", "xp", "auto") and _th != getattr(self, "theme_saved", None):
                    self.cb_theme.set({"techno": "Техно", "classic": "Класична", "day": "День",
                                       "amoled": "AMOLED", "xp": "Windows XP", "auto": "Авто (год)"}[_th])
                    self._switch_theme()
            except Exception:
                pass
            try:
                with open(p, encoding="utf-8") as _fh2:
                    for _ln2 in _fh2:
                        _ln2 = _ln2.strip()
                        if _ln2.startswith("# iq_off:"):
                            try:
                                self._iqj_offset_set(float(_ln2[9:]))
                            except Exception:
                                pass
            except Exception:
                pass
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.cb_achx_mode.set(MODE_KEY.get(mode_b, "Лінійна"))
            self.cb_achx_ex.set(EX_KEY.get(ex_b, "За краями (clamped)"))
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", "\n".join("%.6g %.4f" % pt for pt in pts))
            self.mode_var.set("gen"); self._set_mode()
            self.nb_gen.select(self.gen_frame)
            fv, fu = human_freq(pts[-1][0])
            self._set_status("профіль АЧХ завантажено: %s — %d точок, до %.4g %s (%s/%s)"
                             % (os.path.basename(p), len(pts), fv, fu, mode_b, ex_b))

        # ---------- генератор ----------
        def _build_generator(self, parent):
            nch_here = self.nch
            _g_scroll = ttk.Scrollbar(parent, orient="vertical")
            _g_canvas = tk.Canvas(parent, highlightthickness=0, yscrollcommand=_g_scroll.set)
            _g_scroll.config(command=_g_canvas.yview)
            _g_scroll.pack(side="right", fill="y")
            _g_canvas.pack(side="left", fill="both", expand=True)
            g = ttk.Frame(_g_canvas)
            _g_win = _g_canvas.create_window((0, 0), window=g, anchor="nw")
            def _g_cfg_body(e):
                _g_canvas.configure(scrollregion=_g_canvas.bbox("all"))
            g.bind("<Configure>", _g_cfg_body)
            def _g_cfg_canvas(e):
                _g_canvas.itemconfigure(_g_win, width=e.width)
            _g_canvas.bind("<Configure>", _g_cfg_canvas)
            def _g_wheel(e):
                try:
                    if getattr(e, "num", None) == 4:
                        _g_canvas.yview_scroll(-2, "units")
                    elif getattr(e, "num", None) == 5:
                        _g_canvas.yview_scroll(2, "units")
                    else:
                        _d = int(getattr(e, "delta", 0) or 0)
                        if _d:
                            _g_canvas.yview_scroll(-2 if _d > 0 else 2, "units")
                except Exception:
                    pass
            _g_canvas.bind("<Enter>", lambda e: _g_canvas.bind_all("<MouseWheel>", _g_wheel))
            _g_canvas.bind("<Leave>", lambda e: _g_canvas.unbind_all("<MouseWheel>"))
            _g_canvas.bind("<Button-4>", _g_wheel)
            _g_canvas.bind("<Button-5>", _g_wheel)
            cols = ["Канал   ⚙", "Увімк", "Форма", "Частота, МГц (100…6000)", "Фаза, °",
                    "Ампл., кодів", "Ширина, мкс", "DC, кодів", "Підсилення"]
            for j, c in enumerate(cols):
                ttk.Label(g, text=c, font=("DejaVu Sans", 9, "bold")).grid(row=0, column=j, padx=4)
            self.gen_rows = []
            self.ch_extra = {}
            self.ch_extra_btns = []
            defaults4 = [
                ("Синус", "852", 0, 10000, 2.0, 32768, 1.0),
                ("Синус", "2400", 30, 8000, 2.0, 33000, 1.0),
                ("Шум", "0", 0, 800, 100, 40000, 1.0),
                ("Імпульс", "435", 0, 20000, 100.0, 1000, 1.0),
            ]
            for k in range(nch_here):
                en = tk.BooleanVar(value=(k < 2))
                kind = ttk.Combobox(g, values=WAVE_KINDS, width=12, state="readonly")
                if k < len(defaults4):
                    kind.set(defaults4[k][0])
                    vals = [tk.StringVar(value=str(v)) for v in defaults4[k][1:]]
                else:
                    kind.set("Синус")
                    vals = [tk.StringVar(value=v) for v in ("1000", "0", "5000", "2.0", "32768", "1.0")]
                cf = ttk.Frame(g); cf.grid(row=k + 1, column=0, sticky="w")
                ttk.Label(cf, text="ch%d" % k).pack(side="left")
                ttk.Checkbutton(g, variable=en).grid(row=k + 1, column=1)
                kind.grid(row=k + 1, column=2, padx=4)
                for j, v in enumerate(vals):
                    e = ttk.Entry(g, textvariable=v, width=10)
                    e.grid(row=k + 1, column=j + 3, padx=3)
                    if j == 0:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_freq_var", v))
                        e.bind("<FocusOut>", lambda ev, v=v, k=k: self._auto_from_entry(v, k))
                        e.bind("<Return>", lambda ev, v=v, k=k: self._auto_from_entry(v, k))
                    if j == 3:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_width_var", v))
                bx = ttk.Button(cf, text="⚙", width=3, command=lambda i=k: self._edit_ch_extra(i))
                bx.pack(side="left", padx=(4, 0))
                self.ch_extra_btns.append(bx)
                self.gen_rows.append((en, kind, vals))
            self.audio_rows = []
            self.audio_data = None
            for lbl in ("Аудіо L", "Аудіо R"):
                k = len(self.gen_rows)
                en = tk.BooleanVar(value=False)
                kind = ttk.Combobox(g, values=WAVE_KINDS, width=12, state="readonly")
                kind.set("Синус")
                vals = [tk.StringVar(value=v) for v in ("0", "0", "16000", "0", "32768", "1.0")]
                ttk.Label(g, text=lbl).grid(row=k + 1, column=0)
                ttk.Checkbutton(g, variable=en).grid(row=k + 1, column=1)
                kind.grid(row=k + 1, column=2, padx=4)
                for j, v in enumerate(vals):
                    ttk.Entry(g, textvariable=v, width=10).grid(row=k + 1, column=j + 3, padx=3)
                self.gen_rows.append((en, kind, vals))
                self.audio_rows.append(k)
            rgen = nch_here + 3
            tl = ttk.LabelFrame(g, text="Крок 4б · Сценарій (фази на шкалі часу) — тривалість фази, с і активні канали ch")
            tl.grid(row=rgen + 7, column=0, columnspan=9, sticky="we", pady=(6, 2), padx=2)
            tlr1 = ttk.Frame(tl); tlr1.pack(fill="x", pady=1)
            ttk.Label(tlr1, text="тривалість фази, с:").pack(side="left")
            self.tl_t = tk.StringVar(value="5")
            ttk.Entry(tlr1, textvariable=self.tl_t, width=8).pack(side="left", padx=3)
            ttk.Label(tlr1, text="канали ch (через кому):").pack(side="left", padx=(10, 0))
            self.tl_ch = tk.StringVar(value="0,1")
            ttk.Entry(tlr1, textvariable=self.tl_ch, width=10).pack(side="left", padx=3)
            ttk.Label(tlr1, text="швидкість, м/с:").pack(side="left", padx=(10, 0))
            self.tl_v = tk.StringVar(value="0")
            ttk.Entry(tlr1, textvariable=self.tl_v, width=7).pack(side="left", padx=3)
            ttk.Label(tlr1, text="дальність, км:").pack(side="left", padx=(10, 0))
            self.tl_d = tk.StringVar(value="0")
            ttk.Entry(tlr1, textvariable=self.tl_d, width=7).pack(side="left", padx=3)
            ttk.Button(tlr1, text="+ Додати фазу", command=self._tl_add).pack(side="left", padx=(10, 3))
            ttk.Button(tlr1, text="− Видалити останню", command=self._tl_del).pack(side="left", padx=3)
            ttk.Button(tlr1, text="Очистити", command=self._tl_clear).pack(side="left", padx=3)
            tlr2 = ttk.Frame(tl); tlr2.pack(fill="x", pady=1)
            ttk.Button(tlr2, text="💾 Зберегти сценарій…", command=self._tl_save).pack(side="left", padx=3)
            ttk.Button(tlr2, text="📂 Завантажити сценарій…", command=self._tl_load).pack(side="left", padx=3)
            ttk.Button(tlr2, text="▶ Згенерувати сценарій .raw…", command=self._tl_generate,
                       style="Accent.TButton").pack(side="left", padx=(3, 8))
            ttk.Label(tlr2, text="готові сценарії:").pack(side="left", padx=(8, 0))
            self.tl_preset = ttk.Combobox(tlr2, values=[p["name"] for p in SCENARIO_PRESETS],
                                          width=28, state="readonly")
            self.tl_preset.current(0)
            self.tl_preset.pack(side="left", padx=3)
            ttk.Button(tlr2, text="⤵ Застосувати", command=self._tl_preset_apply).pack(side="left", padx=3)
            self.tl_list = tk.Text(tl, height=3, width=60, font=("DejaVu Sans Mono", 8))
            self.tl_list.pack(fill="x", pady=(2, 0))
            self.tl_cv = tk.Canvas(tl, height=34, highlightthickness=0)
            self.tl_cv.pack(fill="x", pady=(2, 2))
            self.tl_phases = []
            gg = ttk.Frame(g); gg.grid(row=rgen + 8, column=0, columnspan=9, sticky="w", pady=(10, 4))
            ttk.Label(gg, text="Блок:").pack(side="left")
            self.cb_block = ttk.Combobox(gg, values=["%d: %.0f–%.0f МГц" % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6)
                                                     for b in self.blocks], width=16, state="readonly")
            self.cb_block.current(0); self.cb_block.pack(side="left", padx=3)
            ttk.Label(gg, text="Канал:").pack(side="left")
            self.cb_ch = ttk.Combobox(gg, width=8, state="readonly")
            self.cb_ch.pack(side="left", padx=3)
            self.lbl_fc = ttk.Label(gg, text="")
            self.lbl_fc.pack(side="left", padx=6)
            ttk.Button(gg, text="→ центр каналу у виділене поле частоти",
                       command=self.insert_channel_freq).pack(side="left", padx=8)
            self._fill_channels()
            self.cb_block.bind("<<ComboboxSelected>>", lambda e: self._fill_channels())
            ttk.Label(gg, text="fs, МГц (або 12e9):").pack(side="left")
            self.gen_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(gg, textvariable=self.gen_fs, width=11).pack(side="left", padx=3)
            self.gen_fs_mode = ttk.Combobox(gg, values=("Вручну", "Авто за сигналом"),
                                            width=13, state="readonly")
            self.gen_fs_mode.set("Вручну")
            self.gen_fs_mode.pack(side="left", padx=(6, 0))
            self.gen_fs_hint = ttk.Label(gg, text="", foreground=THEME["accent"])
            self.gen_fs_hint.pack(side="left", padx=2)
            self.gen_fs_mode.bind("<<ComboboxSelected>>", lambda _e: self._gen_fs_refresh())
            ttk.Label(gg, text="тривалість, с:").pack(side="left")
            self.gen_dur = tk.StringVar(value="1e-4")
            ttk.Entry(gg, textvariable=self.gen_dur, width=8).pack(side="left", padx=3)
            ttk.Label(gg, text="seed:").pack(side="left")
            self.gen_seed = tk.StringVar(value="42")
            ttk.Entry(gg, textvariable=self.gen_seed, width=7).pack(side="left", padx=3)
            garow = ttk.Frame(g)
            garow.grid(row=rgen + 9, column=0, columnspan=9, sticky="w", pady=(2, 0))
            ttk.Button(garow, text="🌐 Завантажити трек…", command=self._fetch_url_audio).pack(side="left", padx=8)
            ttk.Button(garow, text="🎵 Аудіо (стерео)…", command=self._load_audio).pack(side="left", padx=8)
            ttk.Label(garow, text="Форма для всіх вибраних:").pack(side="left", padx=(18, 4))
            self.cb_shape_all = ttk.Combobox(garow, values=WAVE_KINDS, width=12, state="readonly")
            self.cb_shape_all.set("Синус")
            self.cb_shape_all.pack(side="left")
            ttk.Button(garow, text="→ застосувати до вибраних", command=self._apply_shape_selected).pack(side="left", padx=6)
            ttk.Button(garow, text="🤖 ШІ-підбір завади…", command=self._ai_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🖼 Спектр із зображення…", command=self._specimg_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🧵 ОВ-дрон (оптика + контролер)…", command=self._fiber_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🎯 Профіль завади з аналізу…", command=self._jamprof_open).pack(side="left", padx=6)
            ttk.Button(garow, text="📺 FPV-відео: перехоплення/інжекція…", command=self._fpv_open).pack(side="left", padx=6)
            ttk.Button(garow, text="⚖ Порівняння записів…", command=self._cmp_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🗂 Журнал сесій…", command=self._hist_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🗺 Панорама запису…", command=self._pan_open).pack(side="left", padx=6)
            ttk.Button(garow, text="📂 Імпорт IQ → растр…", command=self._imp_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🎛 SCPI TX / пресети…", command=self._tx_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🌗 Тема", command=self._theme_toggle).pack(side="left", padx=6)
            ttk.Button(garow, text="⏱ Профіль завад…", command=self._jp_open).pack(side="left", padx=6)
            ttk.Button(garow, text="📊 PER-криві…", command=self._pc_open).pack(side="left", padx=6)
            ttk.Button(garow, text="📻 Багатосмуга…", command=self._mb_open).pack(side="left", padx=6)
            ttk.Button(garow, text="📈 J/S-кальк…", command=self._js_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🛡 Стійкість лінка…", command=self._aj_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🎮 RC-декодер…", command=self._rc_open).pack(side="left", padx=6)
            ttk.Button(garow, text="🌊 Водоспад (реальний час)…", command=self._wf_open).pack(side="left", padx=6)
            prow = ttk.Frame(g)
            prow.grid(row=rgen + 10, column=0, columnspan=10, sticky="w", pady=(2, 0))
            ttk.Button(prow, text="📏 Метрика покриття", command=self._metric_open).pack(side="left", padx=(14, 3))
            ttk.Button(prow, text="📄 Звіт у PDF…", command=self._pdf_open).pack(side="left", padx=3)
            ttk.Button(prow, text="🌀 Хоп-послідовність…", command=self._hp_open).pack(side="left", padx=3)
            ttk.Button(prow, text="📶 Якість лінка…", command=self._lq_open).pack(side="left", padx=3)
            ttk.Button(prow, text="🧭 Пеленгатор…", command=self._df_open).pack(side="left", padx=3)
            ttk.Button(prow, text="⚡ Енергодетектор…", command=self._ea_open).pack(side="left", padx=3)
            ttk.Button(prow, text="🛰 Спуфінг GNSS…", command=self._gs_open).pack(side="left", padx=3)
            ttk.Button(prow, text="💾 Зберегти пресет…", command=self._save_preset).pack(side="left", padx=8)
            ttk.Button(prow, text="📂 Завантажити пресет…", command=self._load_preset).pack(side="left", padx=8)
            ttk.Button(prow, text="💾 Профіль сесії…", command=self._prof_save).pack(side="left", padx=8)
            ttk.Button(prow, text="📂 Профіль сесії…", command=self._prof_load).pack(side="left", padx=8)
            ttk.Label(prow, text="«⚙» у рядку — власні АМ/ЧМ/імпульсні параметри каналу (⚙* = задано)",
                      foreground=THEME["dim"]).pack(side="left", padx=(14, 0))
            ttk.Button(gg, text="Крок 5 · Створити .raw…", command=self.generate, style="Accent.TButton").pack(side="left", padx=12)

            gs = ttk.Frame(g); gs.grid(row=rgen, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(gs, text="Крок 1 · Модель SDR:").pack(side="left")
            self.cb_sdr = ttk.Combobox(gs, values=list(SDR_PRESETS.keys()), width=22, state="readonly")
            self.cb_sdr.current(0); self.cb_sdr.pack(side="left", padx=3)
            self.lbl_sdr = ttk.Label(gs, text=""); self.lbl_sdr.pack(side="left", padx=6)
            ttk.Button(gs, text="→ мінімальну ширину у виділене поле ширини",
                       command=self.insert_sdr_width).pack(side="left", padx=8)
            self._show_sdr()
            self.cb_sdr.bind("<<ComboboxSelected>>", lambda e: (self._on_sdr_selected(), self._jamq_table()))
            rsig = ttk.Frame(g); rsig.grid(row=rgen + 1, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(rsig, text="Крок 2 · Тип сигналу:").pack(side="left")
            self.cb_sig = ttk.Combobox(rsig, values=list(SIG_PRESETS.keys()), width=32, state="readonly")
            self.cb_sig["values"] = list(SIG_PRESETS) + list(SIG_GROUPS)
            self.cb_sig.current(0); self.cb_sig.pack(side="left", padx=3)
            self.lbl_sig = ttk.Label(rsig, text=""); self.lbl_sig.pack(side="left", padx=6)
            ttk.Button(rsig, text="→ застосувати до Генератора (форма, частота, fs, додаткові)",
                       command=self._apply_sig).pack(side="left", padx=8)
            self._show_sig()
            self.cb_sig.bind("<<ComboboxSelected>>", lambda e: (self._show_sig(), self._jamq_table()))
            rlib = ttk.Frame(g); rlib.grid(row=rgen + 2, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(rlib, text="Крок 2б · Бібліотека частот радіозв'язку:").pack(side="left")
            self.cb_lib = ttk.Combobox(rlib, values=list(LIB_FLAT), width=48, state="readonly")
            self.cb_lib.current(0); self.cb_lib.pack(side="left", padx=3)
            self.lbl_lib = ttk.Label(rlib, text="", foreground=THEME["dim"], wraplength=560, justify="left")
            self.lbl_lib.pack(side="left", padx=6)
            ttk.Button(rlib, text="→ застосувати (тип, частота, fs)",
                       command=self._apply_lib).pack(side="left", padx=8)
            self._show_lib()
            self.cb_lib.bind("<<ComboboxSelected>>", lambda e: self._show_lib())

            jq = ttk.LabelFrame(g, text="Крок 2в · Параметри якісної завади — залежно від Типу сигналу та Моделі SDR")
            jq.grid(row=rgen + 3, column=0, columnspan=9, sticky="we", pady=(6, 2), padx=2)
            jqc = ttk.Frame(jq); jqc.pack(fill="x", pady=1)
            ttk.Label(jqc, text="ПП, дБм:").pack(side="left")
            self.jq_pa = tk.StringVar(value="20")
            ttk.Entry(jqc, textvariable=self.jq_pa, width=6).pack(side="left", padx=3)
            ttk.Label(jqc, text="+ підсилювач, дБ:").pack(side="left", padx=(10, 0))
            self.jq_amp = tk.StringVar(value="0")
            ttk.Entry(jqc, textvariable=self.jq_amp, width=6).pack(side="left", padx=3)
            ttk.Label(jqc, text="дальність розкриття, км:").pack(side="left", padx=(10, 0))
            self.jq_dist = tk.StringVar(value="5")
            ttk.Entry(jqc, textvariable=self.jq_dist, width=6).pack(side="left", padx=3)
            ttk.Label(jqc, text="дистанція завади, км:").pack(side="left", padx=(10, 0))
            self.jq_dj = tk.StringVar(value="1")
            ttk.Entry(jqc, textvariable=self.jq_dj, width=6).pack(side="left", padx=3)
            ttk.Button(jqc, text="Перерахувати", command=self._jamq_table).pack(side="left", padx=10)
            ttk.Button(jqc, text="→ застосувати якісну заваду до таблиці каналів (форма, частота, смуга, fs)",
                       command=self._jamq_apply).pack(side="left", padx=6)
            jqc2 = ttk.Frame(jq); jqc2.pack(fill="x", pady=1)
            ttk.Label(jqc2, text="Тип антени:").pack(side="left")
            self.jq_ant = ttk.Combobox(jqc2, values=list(JQ_ANTENNAS.keys()) + ["Інший (власноруч)"], width=24, state="readonly")
            self.jq_ant.current(0); self.jq_ant.pack(side="left", padx=3)
            ttk.Label(jqc2, text="підсилення, дБі:").pack(side="left", padx=(8, 0))
            self.jq_ant_g = tk.StringVar(value="2")
            ttk.Entry(jqc2, textvariable=self.jq_ant_g, width=6).pack(side="left", padx=3)
            self.jq_ant_rng = ttk.Label(jqc2, text="", foreground=THEME["dim"])
            self.jq_ant_rng.pack(side="left", padx=(8, 0))
            ttk.Label(jqc2, text="КСВ (рядки: МГц КСВ):").pack(side="left", padx=(8, 0))
            self.txt_vswr = tk.Text(jqc2, height=2, width=26, font=("DejaVu Sans Mono", 8))
            self.txt_vswr.pack(side="left", padx=3)
            jqc3 = ttk.Frame(jq); jqc3.pack(fill="x", pady=1)
            ttk.Label(jqc3, text="Власна антена — ширина пелюстки θ₃, °:").pack(side="left")
            self.jq_hpb = tk.StringVar(value="60")
            ttk.Entry(jqc3, textvariable=self.jq_hpb, width=6).pack(side="left", padx=3)
            ttk.Label(jqc3, text="задній промінь, дБ:").pack(side="left", padx=(10, 0))
            self.jq_fbr = tk.StringVar(value="20")
            ttk.Entry(jqc3, textvariable=self.jq_fbr, width=6).pack(side="left", padx=3)
            ttk.Label(jqc3, text="(застосовуються, коли тип антени — «Інший (власноруч)»)",
                      foreground=THEME["dim"]).pack(side="left", padx=(10, 0))
            self.jq_vswr_scale_off = tk.BooleanVar(value=False)
            ttk.Checkbutton(jqc3, text="приховати шкалу КСВ", variable=self.jq_vswr_scale_off,
                            command=self._spec_redraw_safe).pack(side="left", padx=(14, 0))
            jqc4 = ttk.Frame(jq); jqc4.pack(fill="x", pady=1)
            ttk.Label(jqc4, text="Мапа покриття - поріг приймача, дБм:").pack(side="left")
            self.jq_thr = tk.StringVar(value="-85")
            ttk.Entry(jqc4, textvariable=self.jq_thr, width=6).pack(side="left", padx=3)
            ttk.Button(jqc4, text="🗺 Мапа покриття (FSPL) - дальність за EIRP",
                       command=self._map_open).pack(side="left", padx=(10, 0))
            ttk.Button(jqc4, text="⛰ 3D-мапа покриття",
                       command=self._map3d_open).pack(side="left", padx=(8, 0))
            jqc5 = ttk.Frame(jq); jqc5.pack(fill="x", pady=1)
            ttk.Label(jqc5, text="Оптимізатор потужності - J/S, дБ:").pack(side="left")
            self.jq_js = tk.StringVar(value="15")
            ttk.Entry(jqc5, textvariable=self.jq_js, width=5).pack(side="left", padx=3)
            ttk.Label(jqc5, text="дальність, км:").pack(side="left", padx=(8, 0))
            self.jq_optd = tk.StringVar(value="5")
            ttk.Entry(jqc5, textvariable=self.jq_optd, width=5).pack(side="left", padx=3)
            ttk.Button(jqc5, text="🎯 Мінімальна потужність ПП", command=self._power_opt_open).pack(side="left", padx=(10, 0))
            self.lbl_popt = ttk.Label(jqc5, text="", foreground=THEME["accent"])
            self.lbl_popt.pack(side="left", padx=(10, 0))
            ttk.Button(jqc5, text="🔥 Лабораторія плавлення…", command=self._melt_open).pack(side="left", padx=(10, 0))
            ttk.Button(jqc5, text="🔌 Спектроаналізатор (SCPI)…", command=self._sa_open).pack(side="left", padx=(10, 0))
            ttk.Button(jqc5, text="📋 Рецепти SDR (парк)…", command=self._rec_open).pack(side="left", padx=(10, 0))
            self.jamq_frame = ttk.Frame(jq); self.jamq_frame.pack(fill="both", expand=True)
            self.jamq_rows = []
            self._jamq_table()
            self._jq_ant_pick()


            gx = ttk.LabelFrame(g, text="Крок 3 · Додаткові параметри сигналу (опційно) — період слідування, тривалість імпульсу, фронти, АМ, ЛЧМ")
            gx.grid(row=rgen + 4, column=0, columnspan=9, sticky="we", pady=(6, 2), padx=2)
            gxr1 = ttk.Frame(gx); gxr1.pack(fill="x", pady=1)
            self.ex_per = tk.StringVar(value="0")
            ttk.Label(gxr1, text="період слідування, мкс (0 = 1/Частота):").pack(side="left")
            ttk.Entry(gxr1, textvariable=self.ex_per, width=9).pack(side="left", padx=3)
            self.ex_pw = tk.StringVar(value="0")
            ttk.Label(gxr1, text="тривалість імпульсу, мкс (0 = за «Ширина»):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_pw, width=9).pack(side="left", padx=3)
            self.ex_rise = tk.StringVar(value="0")
            ttk.Label(gxr1, text="фронт (зростання), мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_rise, width=9).pack(side="left", padx=3)
            self.ex_fall = tk.StringVar(value="0")
            ttk.Label(gxr1, text="спад, мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_fall, width=9).pack(side="left", padx=3)
            gxr2 = ttk.Frame(gx); gxr2.pack(fill="x", pady=1)
            self.ex_am_m = tk.StringVar(value="0")
            ttk.Label(gxr2, text="індекс АМ (0…1, 0 = без):").pack(side="left")
            ttk.Entry(gxr2, textvariable=self.ex_am_m, width=9).pack(side="left", padx=3)
            self.ex_am_f = tk.StringVar(value="0")
            ttk.Label(gxr2, text="частота АМ, Гц:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_am_f, width=9).pack(side="left", padx=3)
            self.ex_chirp = tk.StringVar(value="0")
            ttk.Label(gxr2, text="девіація ЛЧМ, МГц (0 = без):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_chirp, width=9).pack(side="left", padx=3)
            gxr3 = ttk.Frame(gx); gxr3.pack(fill="x", pady=1)
            self.ex_chirp_per = tk.StringVar(value="0")
            ttk.Label(gxr3, text="період ЛЧМ — символ LoRa T_sym, мкс (0 = неперервна розгортка):").pack(side="left")
            ttk.Entry(gxr3, textvariable=self.ex_chirp_per, width=9).pack(side="left", padx=3)
            ttk.Label(gxr3, text="ширина смуги завади, МГц (шум):").pack(side="left", padx=(14, 0))
            self.ex_jam_bw = tk.StringVar(value="0")
            self.ex_jam_shape = tk.StringVar(value="")
            ttk.Entry(gxr3, textvariable=self.ex_jam_bw, width=9).pack(side="left", padx=3)
            ttk.Label(gxr3, text="модуляція Mesh (QAM):").pack(side="left", padx=(14, 0))
            self.ex_qam = tk.StringVar(value="Адаптивна (BPSK…1024QAM)")
            ttk.Combobox(gxr3, textvariable=self.ex_qam, values=list(QAM_MAP.keys()),
                         width=24, state="readonly").pack(side="left", padx=3)
            gxr4 = ttk.Frame(gx); gxr4.pack(fill="x", pady=1)
            ttk.Label(gxr4, text="зміщення несучої спуфера, Гц:").pack(side="left")
            self.ex_sp_off = tk.StringVar(value="0")
            ttk.Entry(gxr4, textvariable=self.ex_sp_off, width=10).pack(side="left", padx=3)
            ttk.Label(gxr4, text="дрейф зміщення (плаваюча несуча), Гц/с:").pack(side="left", padx=(14, 0))
            self.ex_sp_drift = tk.StringVar(value="0")
            ttk.Entry(gxr4, textvariable=self.ex_sp_drift, width=10).pack(side="left", padx=3)
            gxr5 = ttk.Frame(gx); gxr5.pack(fill="x", pady=1)
            ttk.Label(gxr5, text="RGPO: швидкість відриву, м/с (0 = 300):").pack(side="left")
            self.ex_rgpo_v = tk.StringVar(value="0")
            ttk.Entry(gxr5, textvariable=self.ex_rgpo_v, width=8).pack(side="left", padx=3)
            ttk.Label(gxr5, text="макс. затримка, мкс (0 = 1000):").pack(side="left", padx=(10, 0))
            self.ex_rgpo_d = tk.StringVar(value="0")
            ttk.Entry(gxr5, textvariable=self.ex_rgpo_d, width=8).pack(side="left", padx=3)
            ttk.Label(gxr5, text="VGPO: макс. доплер, кГц (0 = 30):").pack(side="left", padx=(10, 0))
            self.ex_vgpo_f = tk.StringVar(value="0")
            ttk.Entry(gxr5, textvariable=self.ex_vgpo_f, width=8).pack(side="left", padx=3)
            ttk.Label(gxr5, text="час затягування, с (0 = 2):").pack(side="left", padx=(10, 0))
            self.ex_vgpo_t = tk.StringVar(value="0")
            ttk.Entry(gxr5, textvariable=self.ex_vgpo_t, width=8).pack(side="left", padx=3)
            gxr6 = ttk.Frame(gx); gxr6.pack(fill="x", pady=1)
            ttk.Label(gxr6, text="захищені частоти, МГц (кома; а-б = діапазон):").pack(side="left")
            self.ex_protect = tk.StringVar(value="")
            ttk.Entry(gxr6, textvariable=self.ex_protect, width=40).pack(side="left", padx=3)
            ttk.Label(gxr6, text="рівень сигналу, дБ (J/S = 20lg(ампл) - рівень):").pack(side="left", padx=(14, 0))
            self.ex_siglvl = tk.StringVar(value="0")
            ttk.Entry(gxr6, textvariable=self.ex_siglvl, width=7).pack(side="left", padx=3)

            ach = ttk.LabelFrame(g, text="Крок 4 · АЧХ (опційно) — амплітуда залежно від частоти")
            ach.grid(row=rgen + 6, column=0, columnspan=9, sticky="we", pady=(8, 2), padx=2)
            rowa = ttk.Frame(ach); rowa.pack(fill="x", pady=2)
            self.achx_on = tk.BooleanVar(value=False)
            ttk.Checkbutton(rowa, text="застосовувати АЧХ", variable=self.achx_on).pack(side="left")
            ttk.Label(rowa, text="профіль:").pack(side="left", padx=(12, 2))
            self.cb_achx = ttk.Combobox(rowa, values=ACHX_PRESETS, width=18, state="readonly")
            self.cb_achx.current(0); self.cb_achx.pack(side="left")
            ttk.Label(rowa, text="інтерполяція:").pack(side="left", padx=(12, 2))
            self.cb_achx_mode = ttk.Combobox(rowa, values=list(ACHX_MODES.keys()),
                                             width=15, state="readonly")
            self.cb_achx_mode.set("Кубічний сплайн"); self.cb_achx_mode.pack(side="left")
            ttk.Label(rowa, text="поза діапазоном:").pack(side="left", padx=(12, 2))
            self.cb_achx_ex = ttk.Combobox(rowa, values=list(ACHX_EXTRAPS.keys()),
                                           width=18, state="readonly")
            self.cb_achx_ex.set("За краями (clamped)"); self.cb_achx_ex.pack(side="left")
            rowa2 = ttk.Frame(ach); rowa2.pack(fill="x", pady=2)
            ttk.Label(rowa2, text="центр провалу:").pack(side="left")
            self.achx_notch = tk.StringVar(value="1000")
            ttk.Entry(rowa2, textvariable=self.achx_notch, width=9).pack(side="left", padx=3)
            ttk.Label(rowa2, text="МГц").pack(side="left")
            rowb = ttk.Frame(ach); rowb.pack(fill="x", pady=2)
            ttk.Label(rowb, text="власна таблиця (рядки: частота_Гц дБ) — сюди потрапляє виміряна АЧХ тракту:").pack(anchor="w")
            self.txt_achx = tk.Text(rowb, height=4, width=46, font=("DejaVu Sans Mono", 9))
            self.txt_achx.insert("1.0", "100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
            self.txt_achx.pack(anchor="w")
            n_full = sum(1 for b in self.blocks for c in b["channels"] if not c["partial"])
            n_part = sum(1 for b in self.blocks for c in b["channels"] if c["partial"])
            self.lbl_plan = ttk.Label(g, foreground="#555", wraplength=1250, justify="left",
                      text="Частотний план: 6 блоків, каналізація %.6g МГц: %d повних + %d крайових = %d каналів. "
                           "АЧХ оцінюється на БУДЬ-ЯКІЙ частоті: між точками — інтерполяція "
                           "(лінійна / кубічний сплайн / сплайн у log f), поза виміряним діапазоном — "
                           "за краями / лінійна екстраполяція / нуль. Застосовується до змінної складової "
                           "усіх увімкнених каналів (DC не модулюється). Виміряну АЧХ тракту беріть на "
                           "вкладці «Калібрування АЧХ»; збереження/завантаження — .achx."
                           % (self.bw_hz / 1e6, n_full, n_part, n_full + n_part))
            self.lbl_plan.grid(row=rgen + 5, column=0, columnspan=9, sticky="w")

        def achx_mode_key(self):
            return ACHX_MODES.get(self.cb_achx_mode.get(), "linear")

        def achx_extrap_key(self):
            return ACHX_EXTRAPS.get(self.cb_achx_ex.get(), "clamp")

        def _fill_channels(self):
            b = self.blocks[self.cb_block.current()]
            occ = getattr(self, "occ", None)
            self.cb_ch["values"] = [("● " if occ and occ.get(c["global"], {}).get("occupied") else "")
                                    + "К%03d / %02d (%s)" % (c["global"], c["in_block"],
                                                             ("%.0f" % (c["fc"] / 1e6) + " МГц"))
                                    for c in b["channels"]]
            self.cb_ch.current(0)
            self._show_fc()

        def _show_fc(self):
            b = self.blocks[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            occ = getattr(self, "occ", None)
            is_occ = bool(occ and occ.get(ch["global"], {}).get("occupied"))
            fv, fu = human_freq(ch["fc"])
            self.lbl_fc.config(
                text="центр: %.4g %s (ширина %.0f МГц%s)%s"
                     % (fv, fu, ch["w"] / 1e6, ", крайовий" if ch["partial"] else "",
                        "  ● ЗАЙНЯТИЙ" if is_occ else ""),
                foreground="#f87171" if is_occ else THEME["text"])
            try:
                st = ttk.Style(self)
                if is_occ:
                    st.configure("Occ.TCombobox", fieldbackground="#ffdcdc")
                self.cb_ch.configure(style="Occ.TCombobox" if is_occ else "TCombobox")
            except tk.TclError:
                pass
            self.cb_ch.bind("<<ComboboxSelected>>", lambda e: self._show_fc())

        def insert_channel_freq(self):
            b = self.blocks[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            var = self.last_freq_var or (self.gen_rows[0][2][0] if self.gen_rows else None)
            if var is not None:
                var.set("%.6g" % (ch["fc"] / 1e6))
                self._set_status("ch: центр К%03d = %.6g МГц підставлено" % (ch["global"], ch["fc"] / 1e6))

        def _apply_bw(self):
            """Застосовує «Смуга, МГц» — ширина канала частотного плану (вкладка «Блоки»)."""
            try:
                bw = parse_freq_hz(self.bw_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірна смуга: %s" % self.bw_var.get()); return
            if bw <= 0:
                messagebox.showerror(APP_TITLE, "Смуга має бути більшою за 0 МГц"); return
            n_all = sum(int(math.ceil((f1 - f0) / bw - 1e-9)) for f0, f1 in FREQ_BLOCKS)
            if n_all > 4000:
                messagebox.showerror(APP_TITLE,
                                     "Занадто дрібна смуга: %d каналів (> 4000). Збільшіть «Смуга, МГц»." % n_all)
                return
            self.bw_hz = bw
            self.blocks = build_blocks(bw)
            self.nb_ana.tab(self.tab_blocks, text="Блоки (%.6g МГц)" % (bw / 1e6))
            vals = ["%d: %.0f–%.0f МГц" % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6) for b in self.blocks]
            cur = max(0, self.cb_block.current())
            self.cb_block["values"] = vals
            self.cb_block.current(min(cur, len(vals) - 1))
            n_full = sum(1 for b in self.blocks for c in b["channels"] if not c["partial"])
            n_part = sum(1 for b in self.blocks for c in b["channels"] if c["partial"])
            self.lbl_plan.config(
                text="Частотний план: 6 блоків, каналізація %.6g МГц: %d повних + %d крайових = %d каналів. "
                     "АЧХ оцінюється на БУДЬ-ЯКІЙ частоті: між точками — інтерполяція "
                     "(лінійна / кубічний сплайн / сплайн у log f), поза виміряним діапазоном — "
                     "за краями / лінійна екстраполяція / нуль. Застосовується до змінної складової "
                     "усіх увімкнених каналів (DC не модулюється). Виміряну АЧХ тракту беріть на "
                     "вкладці «Калібрування АЧХ»; збереження/завантаження — .achx."
                     % (bw / 1e6, n_full, n_part, n_full + n_part))
            self._fill_channels()
            if self.chans is not None:
                self.analyze()

        def _sdr(self):
            return SDR_PRESETS.get(self.cb_sdr.get())

        def _min_width_us(self):
            sp = self._sdr()
            return None if not sp else 4.0 / sp["fs_hz"] * 1e6

        def _show_sdr(self):
            sp = self._sdr()
            if not sp:
                self.lbl_sdr.config(text="пресет не обрано — параметри вручну")
                return
            self.lbl_sdr.config(text="fs ≤ %.6g МС/с, смуга ≤ %.6g МГц, %.6g–%.6g ГГц, %d біт"
                                % (sp["fs_hz"] / 1e6, sp["bw_hz"] / 1e6,
                                   sp["f_lo"] / 1e9, sp["f_hi"] / 1e9, sp["bits"]))

        def _on_sdr_selected(self):
            self._show_sdr()
            sp = self._sdr()
            if sp:
                self.gen_fs.set("%.6g" % (sp["fs_hz"] / 1e6))
                self._set_status("SDR %s: fs ← %.6g МГц, мін. ширина імпульсу ≈ %.4g мкс (4 відліки)"
                                 % (self.cb_sdr.get(), sp["fs_hz"] / 1e6, self._min_width_us()))

        def insert_sdr_width(self):
            var = getattr(self, "last_width_var", None)
            mw = self._min_width_us()
            if var is not None and mw:
                var.set("%.6g" % mw)
                self._set_status("мін. ширину %.6g мкс (4 відліки при %s) підставлено"
                                 % (mw, self.cb_sdr.get()))

        def _current_achx(self):
            if not self.achx_on.get():
                return None
            preset = self.cb_achx.get()
            if preset == "Користувацький":
                pts = parse_achx_table(self.txt_achx.get("1.0", "end"))
            else:
                pts = build_achx_points(preset, parse_freq_hz(self.achx_notch.get()))
            return {"points": pts, "mode": self.achx_mode_key(),
                    "extrap": self.achx_extrap_key()}

        CH_EXTRA_KEYS = [("per_us", "Період слідування, мкс"), ("pw_us", "Тривалість імпульсу, мкс"),
                         ("rise_us", "Фронт, мкс"), ("fall_us", "Спад, мкс"),
                         ("am_m", "Індекс АМ (0…1)"), ("am_f", "Частота АМ, Гц"),
                         ("chirp_dev", "Девіація ЧМ/ЛЧМ, МГц"), ("chirp_per_us", "Період ЛЧМ, мкс"),
                         ("qam_bits", "Модуляція Mesh, біт/символ (0=адаптивна)"),
                         ("jam_bw", "Ширина смуги завади, МГц"),
                         ("sp_off", "Зміщення несучої спуфера, Гц"),
                         ("sp_drift", "Дрейф зміщення (плаваюча), Гц/с")]

        def _global_extra_strings(self):
            return {"per_us": self.ex_per.get(), "pw_us": self.ex_pw.get(),
                    "rise_us": self.ex_rise.get(), "fall_us": self.ex_fall.get(),
                    "am_m": self.ex_am_m.get(), "am_f": self.ex_am_f.get(),
                    "chirp_dev": self.ex_chirp.get(), "chirp_per_us": self.ex_chirp_per.get(),
                    "qam": self.ex_qam.get(),
                    "qam_bits": str(QAM_MAP.get(self.ex_qam.get(), 0)),
                    "jam_bw": self.ex_jam_bw.get(),
                    "sp_off": self.ex_sp_off.get(),
                    "sp_drift": self.ex_sp_drift.get()}

        def _spec_redraw_safe(self):
            try:
                self.analyze()
            except Exception:
                pass

        def _antpat_save_to(self, path):
            """PNG лише з діаграмою спрямованості (v4.4.0): полярна вісь +
            шкала КСВ, якщо перемикач не приховує її."""
            _f = Figure(figsize=(6.6, 5.4), dpi=100)
            _a = _f.add_subplot(111, projection="polar")
            self._draw_ant_pattern(_a)
            _f.tight_layout()
            _f.savefig(path, dpi=150, facecolor=_f.get_facecolor())
            return path

        def _antpat_export(self):
            _pth = getattr(self, "path", None)
            _base = (os.path.splitext(os.path.basename(_pth))[0] + "_pattern") if _pth else "antenna_pattern"
            p = filedialog.asksaveasfilename(
                title="Зберегти діаграму спрямованості у PNG", defaultextension=".png",
                initialfile=_base + ".png",
                filetypes=[("PNG зображення", "*.png"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                self._antpat_save_to(p)
            except OSError as e:
                self._mb().showerror(APP_TITLE, "Експорт PNG: %s" % e)
                return
            self._set_status("діаграму спрямованості збережено: %s (PNG, 150 dpi)" % os.path.basename(p))

        def _draw_ant_pattern(self, ax):
            try:
                g = float(self.jq_ant_g.get())
            except Exception:
                g = 0.0
            _nm = self.jq_ant.get()
            _kw = {}
            if _nm not in ANT_BEAMS:
                try:
                    _kw["hpb"] = float(self.jq_hpb.get())
                except Exception:
                    pass
                try:
                    _kw["fbr"] = float(self.jq_fbr.get())
                except Exception:
                    pass
            draw_ant_pattern(ax, _nm, g, **_kw)
            try:
                _rows = self._jq_vswr_rows()
                _sc = ant_vswr_scale(_rows) if (_rows and not ant_scale_hidden(getattr(self, "jq_vswr_scale_off", None))) else None
                if _sc:
                    _fmin, _fmax, _pts = _sc
                    import matplotlib.cm as _cm42
                    _gmx = 10.0
                    for _ln in ax.lines:
                        try:
                            _gmx = max(_gmx, float(max(_ln.get_ydata())))
                        except Exception:
                            pass
                    ax.set_rlim(0.0, _gmx * 1.16)
                    _twin = [a for a in ax.figure.axes if a is not ax and getattr(a, "name", "") == "polar"]
                    if not _twin:
                        _ax2 = ax.figure.add_subplot(ax.get_subplotspec(), projection="polar")
                        _ax2.patch.set_visible(False)
                        _ax2.set_facecolor("none")
                        _ax2.set_theta_zero_location("N")
                        _ax2.set_theta_direction(-1)
                        _ax2.set_rlim(0.0, _gmx * 1.16)
                        _ax2.set_thetagrids([])
                        _ax2.set_yticks([])
                        _ax2.grid(False)
                        _mx = max(q[1] for q in _pts) or 0.5
                        for _i in range(len(_pts) - 1):
                            _c = _cm42.Reds(0.20 + 0.65 * min(1.0, _pts[_i + 1][1] / max(_mx, 1e-9)))
                            _ax2.fill_between([math.radians(_pts[_i][0] * 180.0), math.radians(_pts[_i + 1][0] * 180.0)],
                                              _gmx * 1.03, _gmx * 1.11, color=_c, alpha=0.9, lw=0)
                        _ax2.text(math.radians(90), _gmx * 1.155,
                                  "шкала КСВ-втрат: %.4g → %.4g МГц (0°→180°, темніше = більші втрати)" % (_fmin / 1e6, _fmax / 1e6),
                                  ha="center", va="bottom", fontsize=6.5, color="#7a4b00")
                    _txt = "  ·  ".join("%.4g МГц (КСВ %.2f) → втрати −%.2f дБ" % (f / 1e6, v, _vswr_loss_db(v)) for f, v in _rows)
                    ax.text(0.5, -0.07, _txt, transform=ax.transAxes, ha="center", va="top",
                            fontsize=6.5, color="#7a4b00")
            except Exception:
                pass

        def _jq_ant_pick(self):
            a = JQ_ANTENNAS.get(self.jq_ant.get())
            if not a:
                self.jq_ant_rng.config(text="власні параметри — введіть підсилення і КСВ вручну")
                self._jamq_table()
                self._spec_redraw_safe()
                return
            self.jq_ant_g.set("%.6g" % a["g"])
            self.jq_ant_rng.config(text="робочий діапазон: %.4g–%.4g МГц" % (a["f_lo"] / 1e6, a["f_hi"] / 1e6))
            self.txt_vswr.delete("1.0", "end")
            self.txt_vswr.insert("1.0", "\n".join("%.4g %.2f" % (f / 1e6, v) for f, v in a["vswr"]))
            self._jamq_table()
            self._spec_redraw_safe()

        def _jq_ant_rng(self):
            a = JQ_ANTENNAS.get(self.jq_ant.get())
            return (a["f_lo"], a["f_hi"]) if a and a["f_hi"] > 0 else None

        def _jq_vswr_rows(self):
            try:
                out = []
                for ln in self.txt_vswr.get("1.0", "end").splitlines():
                    p = ln.replace(",", " ").split()
                    if len(p) >= 2:
                        out.append((float(p[0]) * 1e6, float(p[1])))
                return out
            except Exception:
                return []

        def _jamq_table(self):
            f = getattr(self, "jamq_frame", None)
            if f is None:
                return
            for w in f.winfo_children():
                w.destroy()
            try:
                q = jam_quality_params(self.cb_sig.get(), self.cb_sdr.get(),
                                       pa_dbm=float(self.jq_pa.get() or 20),
                                       dist_km=float(self.jq_dist.get() or 5),
                                       dj_km=float(self.jq_dj.get() or 1),
                                       amp_db=float(self.jq_amp.get() or 0),
                                       ant_gain_dbi=float(self.jq_ant_g.get() or 0),
                                       vswr=self._jq_vswr_rows(),
                                       ant_rng=self._jq_ant_rng())
            except Exception as e:
                self.jamq_rows = []
                ttk.Label(f, text="таблиця якісної завади: %s" % e,
                          foreground=THEME["warn"], wraplength=900,
                          justify="left").pack(anchor="w")
                return
            self.jamq_last = q
            self.jamq_rows = q["rows"]
            hdr = ttk.Frame(f); hdr.pack(fill="x")
            ttk.Label(hdr, text="параметр", font=("DejaVu Sans", 9, "bold")).grid(row=0, column=0, sticky="w", padx=4)
            ttk.Label(hdr, text="значення", font=("DejaVu Sans", 9, "bold")).grid(row=0, column=1, sticky="w", padx=14)
            for r, _row in enumerate(q["rows"], start=1):
                nm, vl = _row[0], _row[1]
                _col = _row[2] if len(_row) > 2 else THEME["text"]
                ttk.Label(hdr, text=nm).grid(row=r, column=0, sticky="w", padx=4)
                ttk.Label(hdr, text=vl, foreground=_col).grid(row=r, column=1, sticky="w", padx=14)
            if q.get("warn"):
                ttk.Label(hdr, text="⚠ " + q["warn"], foreground=THEME["warn"],
                          wraplength=900, justify="left").grid(row=len(q["rows"]) + 1,
                                                               column=0, columnspan=2, sticky="w", padx=4)

        def _jamq_apply(self):
            try:
                q = jam_quality_params(self.cb_sig.get(), self.cb_sdr.get(),
                                       pa_dbm=float(self.jq_pa.get() or 20),
                                       dist_km=float(self.jq_dist.get() or 5),
                                       dj_km=float(self.jq_dj.get() or 1),
                                       amp_db=float(self.jq_amp.get() or 0),
                                       ant_gain_dbi=float(self.jq_ant_g.get() or 0),
                                       vswr=self._jq_vswr_rows(),
                                       ant_rng=self._jq_ant_rng())
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Якісна завада: %s" % e)
                return
            sel = []
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                if i in getattr(self, "audio_rows", []):
                    continue
                if bool(en.get()):
                    sel.append(i)
            if not sel:
                for i, (en, kd, vals) in enumerate(self.gen_rows):
                    if i not in getattr(self, "audio_rows", []):
                        self.gen_rows[i][0].set(True)
                        sel = [i]
                        break
            for i in sel:
                en, kd, vals = self.gen_rows[i]
                kd.set(q["kind_name"])
                vals[0].set("%.12g" % (q["f_hz"] / 1e6))
            if q["kind"] == "jam_noise":
                self.ex_jam_bw.set("%.6g" % (q["bw_hz"] / 1e6))
                self.ex_chirp.set("0")
                self.ex_chirp_per.set("0")
            elif q["kind"] == "jam_chirp":
                self.ex_chirp.set("%.6g" % (q["bw_hz"] / 1e6))
                self.ex_chirp_per.set("0")
                self.ex_jam_bw.set("0")
            elif q["kind"] == "spoof":
                self.ex_sp_off.set("%.6g" % q.get("sp_off", 0.0))
                self.ex_sp_drift.set("%.6g" % q.get("sp_drift", 0.0))
            try:
                self.gen_fs.set("%.6g" % (q["fs_hz"] / 1e6))
            except Exception:
                pass
            self._set_status("якісна завада → %d каналів (%s): %s, %.4g МГц, смуга %.4g МГц (Тип: %s · SDR: %s)"
                             % (len(sel), ", ".join("ch%d" % i for i in sel),
                                q["kind_name"], q["f_hz"] / 1e6, q["bw_hz"] / 1e6,
                                self.cb_sig.get(), self.cb_sdr.get()))

        def _row_label(self, i):
            if i in getattr(self, "audio_rows", ()):
                return "Аудіо L" if i == self.audio_rows[0] else "Аудіо R"
            return "ch%d" % i

        def _set_ch_extra(self, i, d):
            if d is None:
                self.ch_extra.pop(i, None)
            else:
                self.ch_extra[i] = dict(d)
            if i < len(self.ch_extra_btns):
                self.ch_extra_btns[i].configure(text="⚙*" if i in self.ch_extra else "⚙")
            self._set_status("%s: %s" % (self._row_label(i),
                             "власні додаткові параметри" if i in self.ch_extra else "спільні додаткові параметри"))

        def _edit_ch_extra(self, i):
            keys = self.CH_EXTRA_KEYS
            cur = self.ch_extra.get(i)
            glob = self._global_extra_strings()
            dlg = tk.Toplevel(self)
            dlg.title("Додаткові параметри — %s" % self._row_label(i))
            dlg.transient(self)
            self._chx_dlg = dlg
            self._chx_vars = {}
            ttk.Label(dlg, text="%s: %s" % (self._row_label(i), "зараз ВЛАСНІ параметри" if cur
                                            else "зараз спільні параметри (значення нижче — спільні)"),
                      font=("DejaVu Sans", 9, "bold")).grid(row=0, column=0, columnspan=2, padx=8, pady=(8, 4), sticky="w")
            for r, (key, lbl) in enumerate(keys, start=1):
                if cur:
                    val = cur.get(key, 0.0)
                    if key == "chirp_dev":
                        val = val / 1e6
                    sv = tk.StringVar(value="%.6g" % val)
                else:
                    sv = tk.StringVar(value=glob.get(key) or "0")
                ttk.Label(dlg, text=lbl).grid(row=r, column=0, sticky="w", padx=8, pady=2)
                ttk.Entry(dlg, textvariable=sv, width=12).grid(row=r, column=1, padx=8, pady=2)
                self._chx_vars[key] = sv

            def ok():
                try:
                    d = {}
                    for key, _ in keys:
                        txt = self._chx_vars[key].get().strip() or "0"
                        d[key] = parse_freq_hz(txt) if key == "chirp_dev" else float(txt)
                except ValueError:
                    messagebox.showerror(APP_TITLE, "Невірне число в параметрах %s" % self._row_label(i), parent=dlg)
                    return
                self._set_ch_extra(i, d)
                dlg.destroy()

            def reset():
                self._set_ch_extra(i, None)
                dlg.destroy()

            self._chx_ok = ok
            self._chx_reset = reset
            bf = ttk.Frame(dlg)
            bf.grid(row=len(keys) + 1, column=0, columnspan=2, pady=8)
            ttk.Button(bf, text="OK — власні для %s" % self._row_label(i), command=ok).pack(side="left", padx=4)
            ttk.Button(bf, text="Скинути до спільних", command=reset).pack(side="left", padx=4)
            ttk.Button(bf, text="Скасувати", command=dlg.destroy).pack(side="left", padx=4)

        def _save_preset(self):
            import json
            out = filedialog.asksaveasfilename(defaultextension=".rspreset",
                                               initialfile="generator_preset.rspreset",
                                               title="Зберегти пресет конфігурації каналів",
                                               filetypes=[("Пресет RAW Studio", "*.rspreset"), ("JSON", "*.json")])
            if not out:
                return
            rows = []
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                rows.append({"label": self._row_label(i), "enabled": bool(en.get()), "kind": kd.get(),
                             "vals": [v.get() for v in vals], "extra": self.ch_extra.get(i)})
            data = {"app": "RAW Studio", "version": APP_VERSION, "type": "generator-preset",
                    "theme": getattr(self, "theme_saved", "techno"),
            "iq_off": self._iqj_offset_value(),
                    "fs_mhz": self.gen_fs.get(), "dur_s": self.gen_dur.get(), "seed": self.gen_seed.get(),
                    "extra_global": self._global_extra_strings(),
                    "achx": {"on": bool(self.achx_on.get()), "preset": self.cb_achx.get(),
                             "table": self.txt_achx.get("1.0", "end").strip(),
                             "mode": self.cb_achx_mode.get(), "extrap": self.cb_achx_ex.get(),
                             "notch": self.achx_notch.get()},
                    "rows": rows}
            try:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                messagebox.showerror(APP_TITLE, "Не вдалося зберегти пресет: %s" % e)
                return
            self._set_status("пресет збережено: %s (%d рядків, власні дод. параметри: %d)"
                             % (os.path.basename(out), len(rows), len(self.ch_extra)))

        def _load_preset(self):
            import json
            p = filedialog.askopenfilename(title="Завантажити пресет конфігурації каналів",
                                           filetypes=[("Пресет RAW Studio", "*.rspreset *.json"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") != "generator-preset":
                    raise ValueError("файл не є пресетом Генератора")
                try:
                    _offv = data.get("iq_off")
                    if _offv is not None:
                        self._iqj_offset_set(float(_offv))
                except Exception:
                    pass
                _t = data.get("theme")
                if _t in ("techno", "classic", "day", "amoled", "xp", "auto"):
                    try:
                        self.cb_theme.set({"techno": "Техно", "classic": "Класична", "day": "День",
                                           "amoled": "AMOLED", "xp": "Windows XP", "auto": "Авто (год)"}[_t])
                        self._switch_theme()
                    except Exception:
                        pass
                rows = data.get("rows") or []
                self.ch_extra = {}
                for i, (en, kd, vals) in enumerate(self.gen_rows):
                    if i < len(rows):
                        r = rows[i]
                        en.set(bool(r.get("enabled")))
                        if r.get("kind") in WAVE_KINDS:
                            kd.set(r["kind"])
                        for v, sv in zip(vals, r.get("vals") or []):
                            v.set(str(sv))
                        if r.get("extra"):
                            self.ch_extra[i] = {k: float(x) for k, x in r["extra"].items()}
                    if i < len(self.ch_extra_btns):
                        self.ch_extra_btns[i].configure(text="⚙*" if i in self.ch_extra else "⚙")
                for var, key in ((self.gen_fs, "fs_mhz"), (self.gen_dur, "dur_s"), (self.gen_seed, "seed")):
                    if data.get(key) not in (None, ""):
                        var.set(str(data[key]))
                eg = data.get("extra_global") or {}
                for var, key in ((self.ex_per, "per_us"), (self.ex_pw, "pw_us"), (self.ex_rise, "rise_us"),
                                 (self.ex_fall, "fall_us"), (self.ex_am_m, "am_m"), (self.ex_am_f, "am_f"),
                                 (self.ex_chirp, "chirp_dev"), (self.ex_chirp_per, "chirp_per_us"),
                                 (self.ex_jam_bw, "jam_bw"), (self.ex_sp_off, "sp_off"),
                                 (self.ex_sp_drift, "sp_drift")):
                    if key in eg:
                        var.set(str(eg[key]))
                if "qam" in eg and str(eg["qam"]) in QAM_MAP:
                    self.ex_qam.set(str(eg["qam"]))
                a = data.get("achx") or {}
                if a:
                    self.achx_on.set(bool(a.get("on")))
                    if a.get("preset"):
                        self.cb_achx.set(a["preset"])
                    self.txt_achx.delete("1.0", "end")
                    self.txt_achx.insert("1.0", a.get("table", ""))
                    if a.get("mode"):
                        self.cb_achx_mode.set(a["mode"])
                    if a.get("extrap"):
                        self.cb_achx_ex.set(a["extrap"])
                    if a.get("notch"):
                        self.achx_notch.set(str(a["notch"]))
                self._set_status("пресет «%s» завантажено: %d рядків, власні дод. параметри: %d (аудіо-доріжки не зберігаються)"
                                 % (os.path.basename(p), min(len(rows), len(self.gen_rows)), len(self.ch_extra)))
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося завантажити пресет: %s" % e)

        def _refresh_ch_occ(self):
            """Оновлює позначки «●» зайнятих каналів у списку «Канал» (v2.7.0)."""
            cur = self.cb_ch.get()
            self._fill_channels()
            for i, v in enumerate(self.cb_ch["values"]):
                if v == cur:
                    self.cb_ch.current(i)
                    break
            self._show_fc()

        def _auto_from_entry(self, var, k):
            """Підставляє Блок/Канал під частоту, введену вручну в таблицю (v2.7.0)."""
            try:
                f = parse_freq_hz(var.get())
            except ValueError:
                return
            if f > 0:
                self._auto_block(f, "частота ch%d (введена вручну)" % k)

        def _auto_dc_gain(self, kind_name, f_hz, amp=0.0, pulse_uni=False):
            """Авто-DC і Підсилення від типу сигналу та частоти (v3.5.0/v3.6.0):
            DC — 32768 (середина 16-біт коду); для ОДНОПОЛЯРНОГО «Імпульса»
            (без радіоімпульсних extras — з ними форма біполярна) DC центрує
            0…ампл: DC = 32768 − ампл·підс/2 («DC від частоти»);
            Підсилення — від PAPR типу (гаусовий шум і OFDM мають хвости ≈ 4σ,
            тому 0.5 замість 1.0) і від категорії частоти несучої
            (< 1 ГГц → ×1.0, 1–3 ГГц → ×0.9, ≥ 3 ГГц → ×0.8 — запас від
            кліпування та викидів інтерполяції на високих швидкостях)."""
            g = 0.5 if kind_name in ("Шум", "Mesh COFDM", "Завада (шум)") else 1.0
            try:
                f = float(f_hz)
            except (TypeError, ValueError):
                f = 0.0
            if f >= 3e9:
                g *= 0.8
            elif f >= 1e9:
                g *= 0.9
            try:
                a = float(amp)
            except (TypeError, ValueError):
                a = 0.0
            dc = 32768
            if kind_name == "Імпульс" and pulse_uni and a > 0:
                dc = max(0, min(65535, int(round(32768.0 - a * g / 2.0))))
            return dc, round(g, 2)

        def _auto_block(self, f_hz, what):
            """Автоматично підставляє Блок і Канал плану під несучу сигналу (v2.6.0)."""
            try:
                f = float(f_hz)
            except (TypeError, ValueError):
                return
            if f <= 0:
                return
            best = None
            for bi, b in enumerate(self.blocks):
                for ci, c in enumerate(b["channels"]):
                    d = abs(c["fc"] - f)
                    if best is None or d < best[0]:
                        best = (d, bi, ci)
            if best is None:
                return
            _, bi, ci = best
            if self.cb_block.current() != bi:
                self.cb_block.current(bi)
                self._fill_channels()
            try:
                self.cb_ch.current(ci)
            except tk.TclError:
                pass
            self._show_fc()
            b = self.blocks[bi]
            ch = b["channels"][ci]
            fcv, fcu = human_freq(ch["fc"])
            dv, du = human_freq(abs(ch["fc"] - f))
            self._set_status("%s → Блок %d (%.0f–%.0f МГц), Канал К%03d: центр %.4g %s, "
                             "відхилення %.4g %s" % (what, b["idx"], b["f0"] / 1e6,
                             b["f1"] / 1e6, ch["global"], fcv, fcu, dv, du))

        def _apply_shape_selected(self):
            shape = self.cb_shape_all.get()
            if shape not in WAVE_KINDS:
                return
            used = []
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                if bool(en.get()):
                    kd.set(shape)
                    try:
                        _amp = float(vals[2].get() or 0)
                    except ValueError:
                        _amp = 0.0
                    _eg = self._global_extra_strings()
                    _uni = not any(float(_eg.get(k) or 0) != 0
                                   for k in ("per_us", "pw_us", "rise_us", "fall_us"))
                    _d, _g = self._auto_dc_gain(shape, parse_freq_hz(vals[0].get() or "0"),
                                                _amp, _uni)
                    vals[4].set(str(_d))
                    vals[5].set("%.6g" % _g)
                    if i in getattr(self, "audio_rows", ()):
                        used.append("Аудіо L" if i == self.audio_rows[0] else "Аудіо R")
                    else:
                        used.append("ch%d" % i)
            self._set_status("форма «%s» → %d вибраних каналів: %s" % (shape, len(used), ", ".join(used)))

        # -- ШІ-підбір завади: діалог експертної системи (v3.7.0) --
        def _ai_open(self):
            _w = getattr(self, "ai_win", None)
            if _w is not None:
                try:
                    if _w.winfo_exists():
                        _w.lift()
                        return
                except tk.TclError:
                    pass
            win = tk.Toplevel(self)
            win.title("ШІ-підбір завади — RAW Studio v%s" % APP_VERSION)
            win.transient(self)
            f = ttk.Frame(win, padding=10)
            f.pack(fill="both", expand=True)
            r = 0
            ttk.Label(f, text="Цільовий сигнал (база знань, %d цілей):" % len(AI_TARGETS)).grid(
                row=r, column=0, sticky="w", pady=2)
            r += 1
            self.ai_target = tk.StringVar(value=sorted(AI_TARGETS)[0])
            cbt = ttk.Combobox(f, textvariable=self.ai_target, values=sorted(AI_TARGETS),
                               state="readonly", width=34)
            cbt.grid(row=r, column=0, sticky="we")
            r += 1
            ttk.Label(f, text="Частота цілі, МГц (можна змінити):").grid(row=r, column=0, sticky="w")
            r += 1
            self.ai_freq = tk.StringVar(value="%.12g" % (AI_TARGETS[sorted(AI_TARGETS)[0]]["freq"] / 1e6))
            ttk.Entry(f, textvariable=self.ai_freq, width=14).grid(row=r, column=0, sticky="w")

            def _sync_freq(*_a):
                _t = AI_TARGETS.get(self.ai_target.get())
                if _t:
                    self.ai_freq.set("%.12g" % (_t["freq"] / 1e6))
            cbt.bind("<<ComboboxSelected>>", _sync_freq)
            r += 1
            rowf = ttk.Frame(f)
            rowf.grid(row=r, column=0, sticky="w", pady=2)
            r += 1
            ttk.Label(rowf, text="Потужність ПП, дБм:").pack(side="left")
            self.ai_pa = tk.StringVar(value="30")
            ttk.Entry(rowf, textvariable=self.ai_pa, width=7).pack(side="left", padx=(4, 14))
            ttk.Label(rowf, text="Дальність розкриття, км:").pack(side="left")
            self.ai_dist = tk.StringVar(value="10")
            ttk.Entry(rowf, textvariable=self.ai_dist, width=7).pack(side="left", padx=(4, 14))
            ttk.Label(rowf, text="Дистанція завади, км:").pack(side="left")
            self.ai_dj = tk.StringVar(value="10")
            ttk.Entry(rowf, textvariable=self.ai_dj, width=7).pack(side="left", padx=4)
            self.ai_spoof = tk.BooleanVar(value=False)
            ttk.Checkbutton(f, text="Спуфінг замість завади (тільки для ГНСС)",
                            variable=self.ai_spoof).grid(row=r, column=0, sticky="w")
            r += 1
            _sp = self._sdr()
            ttk.Label(f, text="SDR: %s — fs %.6g МС/с (смуга генерації ≤ 0.4·fs)"
                      % (self.cb_sdr.get(), (_sp["fs_hz"] / 1e6) if _sp else 0.0),
                      foreground=THEME["dim"]).grid(row=r, column=0, sticky="w")
            r += 1
            self.ai_out = tk.Text(f, width=78, height=17, wrap="word", state="disabled")
            self.ai_out.grid(row=r, column=0, sticky="we", pady=(8, 6))
            r += 1
            brow = ttk.Frame(f)
            brow.grid(row=r, column=0, sticky="e")
            ttk.Button(brow, text="Підібрати", command=self._ai_decide).pack(side="left", padx=4)
            ttk.Button(brow, text="Застосувати до каналів", command=self._ai_apply, style="Accent.TButton").pack(side="left", padx=4)
            ttk.Button(brow, text="Закрити", command=win.destroy).pack(side="left", padx=4)
            self.ai_win = win
            self._ai_decide()

        def _ai_decide(self):
            tkey = self.ai_target.get()
            try:
                f_hz = parse_freq_hz(self.ai_freq.get())
                pa = float(self.ai_pa.get())
                dist = float(self.ai_dist.get())
                dj = float(self.ai_dj.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні число або частота в полях ШІ-модуля")
                return None
            sp = self._sdr()
            fs_hz = sp["fs_hz"] if sp else 61.44e6
            d = ai_recommend_jam(tkey, pa, dist, dj, fs_hz,
                                 want_spoof=bool(self.ai_spoof.get()), f_hz=f_hz)
            d["dc"], d["gain"] = self._auto_dc_gain(d["kind_name"], d["f_hz"])
            self.ai_last = d
            lines = ["РЕКОМЕНДАЦІЯ: %s" % d["kind_name"],
                     "  несуча %.12g МГц, смуга завади %.6g МГц, амплітуда %.0f, підсилення %.6g, DC %d"
                     % (d["f_hz"] / 1e6, d["bw"] / 1e6, d["amp"], d["gain"], d["dc"]),
                     "  J/S у приймачі: %.1f дБ (потрібно %.0f дБ) — %s"
                     % (d["js"], d["js_req"],
                        ("запас +%.1f дБ" % d["margin"]) if d["margin"] >= 0
                        else ("БРАКУЄ %.1f дБ — потрібен ПП %.1f дБм або дистанція ≤ %.6g км"
                              % (-d["margin"], d["p_req"], d["d_max"]))),
                     "Пояснення:"]
            lines += ["  • " + x for x in d["reasons"]]
            lines += ["! " + x for x in d["warnings"]]
            self.ai_out.config(state="normal")
            self.ai_out.delete("1.0", "end")
            self.ai_out.insert("1.0", chr(10).join(lines))
            self.ai_out.config(state="disabled")
            return d

        def _ai_apply(self):
            d = self._ai_decide()
            if not d:
                return
            _prot = []
            try:
                _prot = parse_protected_mhz(self.ex_protect.get())
            except (ValueError, AttributeError):
                _prot = []
            if (is_protected(d["f_hz"], _prot)
                    or is_protected(d["f_hz"] - d["bw"] / 2.0, _prot)
                    or is_protected(d["f_hz"] + d["bw"] / 2.0, _prot)):
                messagebox.showwarning(APP_TITLE,
                                       "Захищені частоти: %.6g МГц у захищеній смузі — застосування скасовано."
                                       % (d["f_hz"] / 1e6))
                return
            used = []
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                if bool(en.get()):
                    kd.set(d["kind_name"])
                    vals[0].set("%.12g" % (d["f_hz"] / 1e6))
                    vals[1].set("0")
                    vals[2].set("%.6g" % d["amp"])
                    vals[3].set("0")
                    vals[4].set(str(d["dc"]))
                    vals[5].set("%.6g" % d["gain"])
                    if i in getattr(self, "audio_rows", ()):
                        used.append("Аудіо L" if i == self.audio_rows[0] else "Аудіо R")
                    else:
                        used.append("ch%d" % i)
            self.gen_fs.set("%.12g" % (d["fs_hz"] / 1e6))
            if d["kind"] == "jam_noise":
                self.ex_jam_bw.set("%.12g" % (d["bw"] / 1e6))
            elif d["kind"] == "jam_chirp":
                self.ex_chirp.set("5")
                self.ex_chirp_per.set("1000")
            elif d["kind"] == "spoof":
                self.ex_sp_off.set("%.12g" % d["sp_off"])
                self.ex_sp_drift.set("%.12g" % d["sp_drift"])
            _spn = self._sdr()
            _nyq = " ⚠ несуча вище Найквіста — при генерації буде запит" if (_spn and d["f_hz"] > _spn["fs_hz"] / 2.0) else ""
            self._set_status("ШІ-завада: %s → %d канал(ів); J/S %.1f дБ (потрібно %.0f): %s"
                             % (d["kind_name"], len(used), d["js"], d["js_req"],
                                ("запас +%.1f дБ" % d["margin"]) if d["margin"] >= 0
                                else "бракує %.1f дБ" % -d["margin"]) + _nyq)

        # -- Спектр із зображення: діалог оцифрування (v3.8.0) --
        def _specimg_open(self):
            _w = getattr(self, "si_win", None)
            if _w is not None:
                try:
                    if _w.winfo_exists():
                        _w.lift()
                        return
                except tk.TclError:
                    pass
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            win = tk.Toplevel(self)
            win.title("Спектр із зображення — RAW Studio v%s" % APP_VERSION)
            win.transient(self)
            f = ttk.Frame(win, padding=10)
            f.pack(fill="both", expand=True)
            r = 0
            rowf = ttk.Frame(f)
            rowf.grid(row=r, column=0, sticky="we")
            r += 1
            self.si_path = tk.StringVar()
            ttk.Entry(rowf, textvariable=self.si_path, width=52).pack(
                side="left", fill="x", expand=True)

            def _pick():
                pth = filedialog.askopenfilename(
                    title="Зображення спектра",
                    filetypes=[("Зображення", "*.png *.jpg *.jpeg *.bmp *.gif"),
                               ("Усі файли", "*.*")])
                if pth:
                    self.si_path.set(pth)
            ttk.Button(rowf, text="📂 Файл…", command=_pick).pack(side="left", padx=6)
            cal = ttk.Frame(f)
            cal.grid(row=r, column=0, sticky="w", pady=4)
            r += 1
            ttk.Label(cal, text="Вісі за підписами: f мін, МГц").pack(side="left")
            self.si_f_min = tk.StringVar(value="0")
            ttk.Entry(cal, textvariable=self.si_f_min, width=8).pack(side="left", padx=(4, 12))
            ttk.Label(cal, text="f макс, МГц").pack(side="left")
            self.si_f_max = tk.StringVar(value="6000")
            ttk.Entry(cal, textvariable=self.si_f_max, width=8).pack(side="left", padx=(4, 12))
            ttk.Label(cal, text="дБ мін").pack(side="left")
            self.si_db_min = tk.StringVar(value="-110")
            ttk.Entry(cal, textvariable=self.si_db_min, width=7).pack(side="left", padx=(4, 12))
            ttk.Label(cal, text="дБ макс").pack(side="left")
            self.si_db_max = tk.StringVar(value="0")
            ttk.Entry(cal, textvariable=self.si_db_max, width=7).pack(side="left", padx=4)
            cr = ttk.Frame(f)
            cr.grid(row=r, column=0, sticky="w", pady=(2, 0))
            r += 1
            ttk.Label(cr, text="Поле графіка, % (для скріншотів із кількома панелями):").pack(side="left")
            self.si_crop = (tk.StringVar(value="0"), tk.StringVar(value="100"),
                            tk.StringVar(value="0"), tk.StringVar(value="100"))
            for _lbl, _sv in (("лів", self.si_crop[0]), ("прав", self.si_crop[1]),
                              ("верх", self.si_crop[2]), ("низ", self.si_crop[3])):
                ttk.Label(cr, text=_lbl).pack(side="left")
                ttk.Entry(cr, textvariable=_sv, width=5).pack(side="left", padx=(2, 8))
            ttk.Button(cr, text="🔍 Огляд", command=self._specimg_preview).pack(side="left", padx=4)
            self.si_fig = Figure(figsize=(7.2, 3.0), dpi=96)
            self.si_ax = self.si_fig.add_subplot(111)
            self.si_canvas = FigureCanvasTkAgg(self.si_fig, master=f)
            self.si_canvas.get_tk_widget().grid(row=r, column=0, sticky="we", pady=(6, 4))
            r += 1
            self.si_out = tk.Text(f, width=88, height=10, wrap="word", state="disabled")
            self.si_out.grid(row=r, column=0, sticky="we", pady=(0, 6))
            r += 1
            brow = ttk.Frame(f)
            brow.grid(row=r, column=0, sticky="e")
            ttk.Button(brow, text="Екстрагувати криву", command=self._specimg_extract, style="Accent.TButton").pack(side="left", padx=4)
            ttk.Button(brow, text="Знайти піки", command=self._specimg_peaks).pack(side="left", padx=4)
            ttk.Button(brow, text="Пік → завада", command=self._specimg_to_jam).pack(side="left", padx=4)
            ttk.Button(brow, text="Пік → ШІ-завада", command=self._specimg_to_ai).pack(side="left", padx=4)
            ttk.Button(brow, text="Закрити", command=win.destroy).pack(side="left", padx=4)
            self.si_win = win

        def _specimg_preview(self):
            pth = self.si_path.get().strip()
            if not pth or not os.path.isfile(pth):
                messagebox.showerror(APP_TITLE, "Файл зображення не знайдено")
                return
            from matplotlib.image import imread
            from matplotlib.patches import Rectangle
            img = imread(pth)
            try:
                cl, cr2, ct, cb = (float(v.get()) / 100.0 for v in self.si_crop)
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні відсотки поля графіка")
                return
            H2, W2 = img.shape[:2]
            self.si_ax.clear()
            self.si_ax.imshow(img)
            self.si_ax.add_patch(Rectangle((cl * W2, ct * H2), (cr2 - cl) * W2,
                                           (cb - ct) * H2, fill=False,
                                           edgecolor="red", linestyle="--", linewidth=1.5))
            self.si_ax.axis("off")
            self.si_canvas.draw()
            self._set_status("Огляд: поле графіка %g–%g %% × %g–%g %%"
                             % (cl * 100, cr2 * 100, ct * 100, cb * 100))

        def _specimg_extract(self):
            try:
                fmn = float(self.si_f_min.get())
                fmx = float(self.si_f_max.get())
                dmn = float(self.si_db_min.get())
                dmx = float(self.si_db_max.get())
                crop = (float(self.si_crop[0].get()) / 100.0,
                        float(self.si_crop[1].get()) / 100.0,
                        float(self.si_crop[2].get()) / 100.0,
                        float(self.si_crop[3].get()) / 100.0)
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні числа калібрування")
                return
            if not (fmx > fmn and dmx > dmn):
                messagebox.showerror(APP_TITLE, "Максимуми мають бути більші за мінімуми")
                return
            pth = self.si_path.get().strip()
            if not pth or not os.path.isfile(pth):
                messagebox.showerror(APP_TITLE, "Файл зображення не знайдено")
                return
            try:
                x, y, meta = image_to_spectrum(pth, fmn, fmx, dmn, dmx,
                                         crop=crop, median=3)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Оцифрування не вдалося: %s" % e)
                return
            self.si_x, self.si_y, self.si_meta = x, y, meta
            self.si_ax.clear()
            self.si_ax.plot(x, y, linewidth=1.0)
            self.si_ax.set_xlabel("Частота, МГц")
            self.si_ax.set_ylabel("Рівень, дБ")
            self.si_ax.grid(True, alpha=0.3)
            self.si_canvas.draw()
            self._set_status("Спектр із зображення: %d колонок, траса %s, рамка %s %s"
                             % (meta["columns"], meta["mode"], meta["box_mode"], meta["box"]))

        def _specimg_peaks(self):
            if getattr(self, "si_x", None) is None:
                messagebox.showinfo(APP_TITLE, "Спочатку натисніть «Екстрагувати криву».")
                return
            pk = spectrum_peaks(self.si_x, self.si_y, prom_thr=6.0, min_sep_mhz=1.0)
            self.si_peaks = pk

            def _plan(fv):
                best = None
                for b in self.blocks:
                    for c in b["channels"]:
                        dd = abs(c["fc"] - fv * 1e6)
                        if best is None or dd < best[0]:
                            best = (dd, b, c)
                if best is None or best[0] > 200e6:
                    return " (поза планом)"
                dd, b, c = best
                return " → Блок %d, К%03d (Δ %.4g МГц)" % (b["idx"], c["global"], dd / 1e6)
            floor = float(np.median(self.si_y))
            lines = ["Піки (фон %.1f дБ):" % floor]
            for k, p in enumerate(pk, 1):
                lines.append("%2d. %.4g МГц — %.1f дБ (S/N %.1f дБ, ширина −3 дБ ≈ %.4g МГц)%s"
                             % (k, p["freq"], p["db"], p["snr"], p["width"], _plan(p["freq"])))
            if not pk:
                lines.append("піків над фоном +6 дБ не знайдено")
            self.si_out.config(state="normal")
            self.si_out.delete("1.0", "end")
            self.si_out.insert("1.0", chr(10).join(lines))
            self.si_out.config(state="disabled")
            self._set_status("Спектр із зображення: піків %d, фон %.1f дБ" % (len(pk), floor))

        def _specimg_to_ai(self):
            pk = getattr(self, "si_peaks", None)
            if not pk:
                messagebox.showinfo(APP_TITLE, "Спочатку знайдіть піки.")
                return
            top = max(pk, key=lambda p: p["db"])
            self._ai_open()
            self.ai_freq.set("%.12g" % top["freq"])
            self._ai_decide()
            self._set_status("Пік %.4g МГц (%.1f дБ) → ШІ-підбір завади" % (top["freq"], top["db"]))

        def _specimg_to_jam(self):
            pk = getattr(self, "si_peaks", None)
            if not pk:
                messagebox.showinfo(APP_TITLE, "Спочатку знайдіть піки («Знайти піки»).")
                return
            top = max(pk, key=lambda p: p["db"])
            d = jam_from_peak(top["freq"], top["db"], float(top.get("width") or 0.0))
            used = 0
            for en, kd, vals in self.gen_rows:
                if bool(en.get()):
                    kd.set(d["kind_name"])
                    vals[0].set("%.12g" % (d["f_hz"] / 1e6))
                    used += 1
            if d["kind"] == "jam_noise":
                self.ex_jam_bw.set("%.12g" % d["bw_mhz"])
            elif d["kind"] == "jam_chirp":
                self.ex_chirp.set("%.12g" % d["dev_mhz"])
                self.ex_chirp_per.set("%.6g" % d["per_us"])
            self._set_status("Завада з піка: %s %.4g МГц → %d канал(ів); сигнал ≈ %.1f дБм "
                             "(dBFS + офсет), для J/S +%.0f дБ потрібно ≈ %.1f дБм; %s"
                             % (d["kind_name"], d["f_hz"] / 1e6, used, d["sig_dbm"],
                                d["js_target"], d["jam_req_dbm"], d["note"]))

        def _fetch_url_audio(self):
            import shutil, subprocess
            import tkinter.simpledialog as sd
            import urllib.request, urllib.parse, tempfile
            url = sd.askstring("Завантажити аудіо-трек",
                               "URL аудіо (пряме посилання на mp3/wav/ogg…, до 200 МБ)\n"
                               "або посилання YouTube (потрібні yt-dlp та ffmpeg):",
                               parent=self)
            if not url or not url.strip():
                return
            url = url.strip()
            self._set_status("завантажую %s…" % url)
            self.update_idletasks()
            try:
                if "youtube.com" in url or "youtu.be" in url:
                    if not shutil.which("ffmpeg"):
                        messagebox.showerror(APP_TITLE, "Для YouTube потрібен ffmpeg у системі.")
                        return
                    try:
                        import yt_dlp
                    except ImportError:
                        messagebox.showerror(APP_TITLE, "Пакет yt-dlp не знайдено.\nВстановіть його:  pip install yt-dlp")
                        return
                    tmpm = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
                    tmpm.close()
                    out = os.path.splitext(tmpm.name)[0]
                    opts = {"format": "bestaudio/best", "outtmpl": out + ".%(ext)s",
                            "quiet": True, "noplaylist": True}
                    try:
                        with yt_dlp.YoutubeDL(opts) as ydl:
                            ydl.download([url])
                    finally:
                        if os.path.exists(tmpm.name):
                            os.unlink(tmpm.name)
                    srcs = [f for f in (out + ".m4a", out + ".webm", out + ".mp3",
                                        out + ".opus", out + ".ogg") if os.path.exists(f)]
                    if not srcs:
                        d = os.path.dirname(out)
                        srcs = [os.path.join(d, f) for f in os.listdir(d)
                                if f.startswith(os.path.basename(out))]
                    if not srcs:
                        messagebox.showerror(APP_TITLE, "Не вдалося завантажити аудіо з YouTube.")
                        return
                    tmpw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmpw.close()
                    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", srcs[0],
                                        "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", tmpw.name],
                                       capture_output=True)
                    for f in srcs:
                        try:
                            os.unlink(f)
                        except OSError:
                            pass
                    if r.returncode != 0 or not os.path.getsize(tmpw.name):
                        messagebox.showerror(APP_TITLE, "ffmpeg не зміг конвертувати завантажене аудіо.")
                        return
                    self._audio_into_channels(tmpw.name)
                    os.unlink(tmpw.name)
                else:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 RAWStudio"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        buf = []
                        total = 0
                        while True:
                            chunk = resp.read(1 << 20)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > 200 * 1024 * 1024:
                                raise ValueError("файл більший за 200 МБ")
                            buf.append(chunk)
                    name = os.path.basename(urllib.parse.urlparse(url).path) or "track"
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=os.path.splitext(name)[1] or ".mp3", delete=False)
                    tmp.write(b"".join(buf))
                    tmp.close()
                    self._set_status("завантажено %s (%.1f МБ) — читаю…" % (name, total / 1e6))
                    self.update_idletasks()
                    self._audio_into_channels(tmp.name)
                    os.unlink(tmp.name)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося завантажити трек: %s" % e)

        def _load_audio(self):
            p = filedialog.askopenfilename(
                title="Аудіо для стерео-каналів (WAV; MP3/FLAC/OGG — через ffmpeg)",
                filetypes=[("Аудіо", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("Усі файли", "*.*")])
            if not p:
                return
            self._audio_into_channels(p)

        def _audio_into_channels(self, p):
            try:
                import wave, shutil, subprocess, tempfile
                src = p
                tmp = None
                if not p.lower().endswith(".wav"):
                    ff = shutil.which("ffmpeg")
                    if not ff:
                        messagebox.showerror(APP_TITLE, "Формат не WAV, а ffmpeg не знайдено в системі.\nКонвертуйте у WAV (PCM) і спробуйте ще раз.")
                        return
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", p,
                                        "-ac", "2", "-ar", "44100", "-sample_fmt", "s16", tmp.name],
                                       capture_output=True)
                    if r.returncode != 0 or not os.path.getsize(tmp.name):
                        messagebox.showerror(APP_TITLE, "ffmpeg не зміг прочитати файл:\n%s"
                                             % r.stderr.decode("utf-8", "replace")[:300])
                        os.unlink(tmp.name)
                        return
                    src = tmp.name
                with wave.open(src, "rb") as w:
                    sr = w.getframerate()
                    nch_a = w.getnchannels()
                    sw = w.getsampwidth()
                    raw = w.readframes(w.getnframes())
                if tmp:
                    os.unlink(tmp.name)
                if sw == 2:
                    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
                elif sw == 1:
                    data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
                elif sw == 4:
                    data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
                else:
                    messagebox.showerror(APP_TITLE, "Підтримуються WAV 8/16/32 біт PCM"); return
                if nch_a < 1 or data.size == 0:
                    messagebox.showerror(APP_TITLE, "Порожній аудіофайл"); return
                data = data.reshape(-1, nch_a)
                L = np.ascontiguousarray(data[:, 0])
                R = np.ascontiguousarray(data[:, 1] if nch_a > 1 else data[:, 0])
                dur = len(L) / float(sr)
                fs_cur = parse_freq_hz(self.gen_fs.get()) if self.gen_fs.get().strip() else 12e9
                fs_new = min(fs_cur, max(4.0 * sr, 1e6))
                self.audio_data = {"L": L, "R": R}
                for i in self.audio_rows:
                    en, kd, vals = self.gen_rows[i]
                    en.set(True); kd.set("Синус")
                    vals[0].set("0"); vals[2].set("16000"); vals[3].set("0")
                self.gen_dur.set("%.6g" % dur)
                self.gen_fs.set("%.6g" % (fs_new / 1e6))
                self.nch_var.set(str(len(self.gen_rows)))
                self._set_status("аудіо: %s — %d відліків, %.3f с @ %d Гц → «Аудіо L»/«Аудіо R» (тривалість %.6g с, каналів %d)"
                                 % (os.path.basename(p), len(L), dur, sr, dur, len(self.gen_rows)))
                self._guide("Аудіо завантажено: ліва доріжка → «Аудіо L», права → «Аудіо R» (зразки аудіо підміняють форму сигналу, амплітуда — поле «ампл.»); fs запису → %s МГц. Далі — «Створити .raw…»." % ("%.6g" % (fs_new / 1e6)))
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося прочитати аудіо: %s" % e)

        # -- Спектрограма з .raw: діалог запуску і перегляду (v3.17.0) --
        # ---------- v5.2.0: нові діалоги і методи ----------

        # ---------- v5.4.0: модуль ОВ-дрону ----------

        def _fiber_open(self):
            win, top, _lbl = self._win("Завада дрону з ОВ-трактом керування", 1100, 700)

            def _ent(row, var, wd=6):
                tk.Entry(row, textvariable=var, width=wd, bg=THEME["panel2"], fg=THEME["text"],
                         insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            row0 = tk.Frame(win, bg=THEME["panel"])
            row0.pack(side="top", fill="x", padx=8, pady=(0, 2))
            _lbl(row0, "Модем ОВ-тракту:")
            self.fm_combo = ttk.Combobox(row0, values=[m["name"] for m in FIBER_MODEMS],
                                         width=46, state="readonly")
            self.fm_combo.current(5)
            self.fm_combo.pack(side="left", padx=3)
            ttk.Button(row0, text="Обрати", command=self._fiber_pick).pack(side="left", padx=3)
            self.fm_info = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["text"],
                                    anchor="w", justify="left", font=("DejaVu Sans Mono", 8),
                                    wraplength=980)
            self.fm_info.pack(side="top", fill="x", padx=10, pady=(2, 4))
            fb = ttk.LabelFrame(win, text="Вплив на модем — інжекція світла у волокно (засліплення фотоприймача / зрив CDR)")
            fb.pack(side="top", fill="x", padx=8, pady=4)
            fbr = tk.Frame(fb, bg=THEME["panel"])
            fbr.pack(side="top", fill="x", padx=6, pady=2)
            _lbl(fbr, "P інжектора, дБм:")
            self.fbp = tk.StringVar(value="10")
            _ent(fbr, self.fbp)
            _lbl(fbr, "розбіжність, мрад:")
            self.fbd = tk.StringVar(value="10")
            _ent(fbr, self.fbd)
            _lbl(fbr, "дистанція до торця, м:")
            self.fbdist = tk.StringVar(value="1.0")
            _ent(fbr, self.fbdist)
            _lbl(fbr, "втрати з'єднання, дБ:")
            self.fbi = tk.StringVar(value="2.0")
            _ent(fbr, self.fbi)
            ttk.Button(fbr, text="⚡ Розрахувати", style="Accent.TButton",
                       command=self._fiber_bleach_calc).pack(side="left", padx=(10, 2))
            self.fm_bleach_lbl = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["accent"],
                                          anchor="w", justify="left", wraplength=980)
            self.fm_bleach_lbl.pack(side="top", fill="x", padx=10)
            fr_ = ttk.LabelFrame(win, text="Вплив на контролер — ЕМ-вплив на резонансній частоті (кварц МК / опорний генератор CDR)")
            fr_.pack(side="top", fill="x", padx=8, pady=4)
            frr = tk.Frame(fr_, bg=THEME["panel"])
            frr.pack(side="top", fill="x", padx=6, pady=2)
            _lbl(frr, "типові вузли:")
            self.frc = ttk.Combobox(frr, values=[n for n, _ in FIBER_CLOCKS], width=40, state="readonly")
            self.frc.current(3)
            self.frc.pack(side="left", padx=2)
            self.frc.bind("<<ComboboxSelected>>",
                          lambda e: self.frf.set("%.6g" % dict(FIBER_CLOCKS)[self.frc.get()]))
            frr2 = tk.Frame(fr_, bg=THEME["panel"])
            frr2.pack(side="top", fill="x", padx=6, pady=2)
            _lbl(frr2, "частота, МГц:")
            self.frf = tk.StringVar(value="25")
            _ent(frr2, self.frf, 8)
            _lbl(frr2, "дистанція, м:")
            self.frd = tk.StringVar(value="1.0")
            _ent(frr2, self.frd)
            _lbl(frr2, "потрібна U, В:")
            self.fru = tk.StringVar(value="0.5")
            _ent(frr2, self.fru)
            _lbl(frr2, "ефект. довжина, м:")
            self.frl = tk.StringVar(value="0.1")
            _ent(frr2, self.frl)
            _lbl(frr2, "G антени, дБ:")
            self.frg = tk.StringVar(value="0")
            _ent(frr2, self.frg)
            ttk.Button(frr2, text="🎯 Розрахувати", style="Accent.TButton",
                       command=self._fiber_res_calc).pack(side="left", padx=(10, 2))
            self.fm_res_lbl = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["accent"],
                                       anchor="w", justify="left", wraplength=980)
            self.fm_res_lbl.pack(side="top", fill="x", padx=10, pady=(2, 2))
            row9 = tk.Frame(win, bg=THEME["panel"])
            row9.pack(side="bottom", fill="x", padx=8, pady=4)
            ttk.Button(row9, text="→ ПП, дБм у Крок 2в", command=self._fiber_apply).pack(side="left", padx=2)
            ttk.Button(row9, text="💾 CSV довідника ОВ-модемів…",
                       command=self._fiber_csv_save).pack(side="left", padx=8)
            tk.Label(row9, text="значення довідника — типові орієнтири для планування РЕБ; "
                                "для конкретного виробу потрібні вимірювання",
                     bg=THEME["panel"], fg=THEME["dim"], wraplength=520, justify="left").pack(side="left", padx=(10, 0))
            row9b = tk.Frame(win, bg=THEME["panel"])
            row9b.pack(side="bottom", fill="x", padx=8, pady=(0, 2))
            ttk.Button(row9b, text="🧠 ШІ-підбір тракту…",
                       command=self._fiber_chain_open).pack(side="left", padx=2)
            ttk.Button(row9b, text="🔬 Симулятор завади…",
                       command=self._fiber_sim_open).pack(side="left", padx=8)
            self.fm_bleach = None
            self.fm_res = None
            self._fiber_pick()

        def _fiber_pick(self):
            nm = self.fm_combo.get()
            for m in FIBER_MODEMS:
                if m["name"] == nm:
                    self.fm_cur = dict(m)
                    self.fm_info.config(text="λ %s нм · %s Мбіт/с · до %s км · лазер %s · "
                                             "P_перед %s дБм · чутливість %s дБм · перевантаження %s дБм · "
                                             "опорний генератор %s МГц · %s · %s"
                                        % (m["lam_nm"], m["rate_mbps"], m["reach_km"], m["laser"],
                                           m["p_tx_dbm"], m["sens_dbm"], m["overload_dbm"],
                                           m["clock_mhz"], m["conn"], m["note"]))
                    return
            self.fm_info.config(text="модем не знайдено")

        def _fiber_bleach_calc(self):
            try:
                p = float(self.fbp.get())
                div = float(self.fbd.get())
                d = float(self.fbdist.get())
                ins = float(self.fbi.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "ОВ-модуль (засліплення): %s" % e)
                return
            m = getattr(self, "fm_cur", None) or FIBER_MODEMS[5]
            r = fiber_bleach(p, div, d, ins_loss_db=ins)
            code, txt = fiber_status(r["p_rx_dbm"], m["sens_dbm"], m["overload_dbm"])
            need = fiber_bleach_needed(d, m["overload_dbm"], div, ins_loss_db=ins)
            self.fm_bleach = dict(r)
            self.fm_bleach["need"] = need
            self.fm_bleach["code"] = code
            self.fm_bleach_lbl.config(text="пляма %.0f мкм · геом. втрати %.1f дБ · P на фотодіоді %.1f дБм → %s · "
                                           "для засліплення потрібно %.1f дБм на вході інжектора"
                                      % (r["spot_um"], r["geom_db"], r["p_rx_dbm"], txt, need))
            self._set_status("ОВ-модуль: P_rx %.1f дБм (%s)" % (r["p_rx_dbm"], txt))

        def _fiber_res_calc(self):
            try:
                f = parse_freq_hz(self.frf.get())
                d = float(self.frd.get())
                u = float(self.fru.get())
                le = float(self.frl.get())
                g = float(self.frg.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "ОВ-модуль (резонанс): %s" % e)
                return
            r = mcu_resonance(f, d, u_req_v=u, l_eff_m=le, g_ant_db=g)
            self.fm_res = dict(r)
            self.fm_res["f_hz"] = float(f)
            note = "" if f >= 1e6 else " (f < 1 МГц — ближня зона, оцінка орієнтовна)"
            self.fm_res_lbl.config(text="f %.6g МГц: потрібно E %.1f В/м → EIRP %.1f дБм на %.3g м%s"
                                   % (f / 1e6, r["e_req_vm"], r["eirp_dbm"], d, note))
            self._set_status("ОВ-модуль: резонанс %.6g МГц → EIRP %.1f дБм" % (f / 1e6, r["eirp_dbm"]))

        def _fiber_apply(self):
            if not getattr(self, "fm_res", None):
                messagebox.showinfo(APP_TITLE, "Спочатку розрахуйте резонанс.")
                return
            self.jq_pa.set("%.1f" % self.fm_res["eirp_dbm"])
            self._set_status("ПП → %.1f дБм (резонанс %.6g МГц, ОВ-модуль)"
                             % (self.fm_res["eirp_dbm"], self.fm_res["f_hz"] / 1e6))

        def _fiber_csv_save(self):
            out = filedialog.asksaveasfilename(defaultextension=".csv",
                                               initialfile="fiber_modems.csv",
                                               title="Куди зберегти довідник ОВ-модемів",
                                               filetypes=[("CSV", "*.csv")])
            if not out:
                return
            try:
                fiber_modems_csv(out)
            except OSError as e:
                messagebox.showerror(APP_TITLE, "Не вдалося зберегти довідник: %s" % e)
                return
            self._set_status("довідник ОВ-модемів: %s (%d класів)"
                             % (os.path.basename(out), len(FIBER_MODEMS)))

        def _fiber_chain_open(self):
            win, top, _lbl = self._win("ШІ-підбір тракту завади — ОВ-дрон", 940, 600)
            self._chain_top = top
            r1 = tk.Frame(win, bg=THEME["panel"])
            r1.pack(side="top", fill="x", padx=8, pady=4)
            _lbl(r1, "частота впливу, МГц (або 1ггц):")
            self.ch_f = tk.StringVar(value=(self.frf.get() or "25"))
            tk.Entry(r1, textvariable=self.ch_f, width=10, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "потрібна EIRP, дБм:")
            self.ch_e = tk.StringVar(value=("%.1f" % self.fm_res["eirp_dbm"])
                                     if getattr(self, "fm_res", None) else "29.2")
            tk.Entry(r1, textvariable=self.ch_e, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "кабель/розʼєми, дБ:")
            self.ch_cbl = tk.StringVar(value="1.5")
            tk.Entry(r1, textvariable=self.ch_cbl, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            ttk.Button(r1, text="🧠 Підібрати", style="Accent.TButton",
                       command=self._fiber_chain_calc).pack(side="left", padx=(10, 2))
            self.chain_rows_frame = ttk.Frame(win)
            self.chain_rows_frame.pack(side="top", fill="both", expand=True, padx=8, pady=4)
            self.chain_opt_lbl = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["dim"],
                                          anchor="w", justify="left", wraplength=900)
            self.chain_opt_lbl.pack(side="bottom", fill="x", padx=10, pady=(0, 2))
            rowb = tk.Frame(win, bg=THEME["panel"])
            rowb.pack(side="bottom", fill="x", padx=8, pady=4)
            self.chain_pick = ttk.Combobox(rowb, width=64, state="readonly")
            self.chain_pick.pack(side="left", padx=2)
            ttk.Button(rowb, text="→ Крок 2в (ПП + підсилювач + антена)",
                       command=self._fiber_chain_apply).pack(side="left", padx=8)
            self.chain_vars = []
            self._fiber_chain_calc()

        def _fiber_chain_calc(self):
            try:
                f = parse_freq_hz(self.ch_f.get())
                e = float(self.ch_e.get())
                cbl = float(self.ch_cbl.get() or 1.5)
            except ValueError as ex:
                messagebox.showerror(APP_TITLE, "ШІ-тракт: %s" % ex)
                return
            vs = fiber_chain_select(f, e, cable_db=cbl)
            self.chain_vars = vs
            for wdg in self.chain_rows_frame.winfo_children():
                wdg.destroy()
            ttk.Label(self.chain_rows_frame,
                      text="№ · SDR (ПП, дБм) · антена (G, дБі; втрати КСВ) · потрібен підсилювач, дБ → EIRP",
                      font=("DejaVu Sans", 9, "bold")).pack(anchor="w", pady=(0, 3))
            vals = []
            for i, v in enumerate(vs[:10], start=1):
                sv = ("%d. %s — ПП %.1f дБм + %s (%.1f дБі, КСВ-втрати %.1f дБ) → "
                      "підсилювач %+.1f дБ → EIRP %.1f дБм"
                      % (i, v["sdr"], v["tx_dbm"], v["ant"], v["g_dbi"],
                         v["vswr_loss_db"], v["amp_req_db"], v["eirp_req_dbm"]))
                ttk.Label(self.chain_rows_frame, text=sv, wraplength=890,
                          justify="left").pack(anchor="w", pady=1)
                vals.append(sv)
            if not vals:
                ttk.Label(self.chain_rows_frame,
                          text="варіантів немає: жодна пара SDR+антена не покриває частоту "
                               "(або потрібне підсилення понад 40 дБ)",
                          foreground=THEME["warn"], wraplength=890,
                          justify="left").pack(anchor="w")
            self.chain_pick.config(values=vals)
            if vals:
                self.chain_pick.current(0)
            try:
                need = fiber_bleach_needed(float(self.fbdist.get() or 1.0),
                                           float(self.fm_cur["overload_dbm"]),
                                           float(self.fbd.get() or 10.0),
                                           ins_loss_db=float(self.fbi.get() or 2.0))
                self.chain_opt_lbl.config(
                    text="оптичний канал (засліплення): потрібен лазерний інжектор ≥ %.1f дБм "
                         "на %.4g м (модем: %s)" % (need, float(self.fbdist.get() or 1.0),
                                                    self.fm_cur["name"]))
            except Exception:
                self.chain_opt_lbl.config(text="оптичний канал: заповніть параметри інжектора у вікні ОВ-модуля")
            self._set_status("ШІ-тракт: %d варіантів на %.6g МГц під EIRP %.1f дБм"
                             % (len(vs), f / 1e6, e))

        def _fiber_chain_apply(self):
            vs = getattr(self, "chain_vars", [])
            i = self.chain_pick.current()
            if not vs or i < 0 or i >= len(vs):
                messagebox.showinfo(APP_TITLE, "Спочатку виконайте підбір.")
                return
            v = vs[i]
            self.jq_pa.set("%.1f" % v["tx_dbm"])
            self.jq_amp.set("%+.1f" % v["amp_req_db"])
            if v["ant"] in JQ_ANTENNAS:
                self.jq_ant.set(v["ant"])
                self._jq_ant_pick()
            self.jq_ant_g.set("%.2f" % v["g_dbi"])
            self._set_status("ШІ-тракт → Крок 2в: %s (%.1f дБм) + %s (%.1f дБі) + "
                             "підсилювач %+.1f дБ → EIRP %.1f дБм"
                             % (v["sdr"], v["tx_dbm"], v["ant"], v["g_dbi"],
                                v["amp_req_db"], v["eirp_req_dbm"]))

        def _fiber_sim_open(self):
            win, top, _lbl = self._win("Симулятор завади ОВ-тракту — %s"
                                       % getattr(self, "fm_cur", {}).get("name", "модем"), 940, 640)
            self._sim_top = top
            r1 = tk.Frame(win, bg=THEME["panel"])
            r1.pack(side="top", fill="x", padx=8, pady=4)
            _lbl(r1, "P інжектора, дБм:")
            self.sim_p = tk.StringVar(value=self.fbp.get() or "10")
            tk.Entry(r1, textvariable=self.sim_p, width=7, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "розбіжність, мрад:")
            self.sim_div = tk.StringVar(value=self.fbd.get() or "10")
            tk.Entry(r1, textvariable=self.sim_div, width=7, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "втрати, дБ:")
            self.sim_ins = tk.StringVar(value=self.fbi.get() or "2.0")
            tk.Entry(r1, textvariable=self.sim_ins, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "дрон пролетів, м:")
            self.sim_dd = tk.StringVar(value="500")
            tk.Entry(r1, textvariable=self.sim_dd, width=7, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            _lbl(r1, "J/S поріг, дБ:")
            self.sim_jsr = tk.StringVar(value="15")
            tk.Entry(r1, textvariable=self.sim_jsr, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=3)
            ttk.Button(r1, text="▶ Симулювати", style="Accent.TButton",
                       command=self._fiber_sim_calc).pack(side="left", padx=(10, 2))
            self.sim_lbl = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["accent"],
                                    anchor="w", justify="left", wraplength=900)
            self.sim_lbl.pack(side="top", fill="x", padx=10, pady=(4, 2))
            self.sim_rows_frame = ttk.Frame(win)
            self.sim_rows_frame.pack(side="top", fill="both", expand=True, padx=8, pady=2)
            self.sim_canvas = tk.Canvas(win, height=250, bg=THEME["panel2"], highlightthickness=0)
            self.sim_canvas.pack(side="bottom", fill="x", padx=8, pady=(2, 6))
            self._fiber_sim_calc()

        def _fiber_sim_calc(self):
            try:
                p = float(self.sim_p.get())
                div = float(self.sim_div.get())
                ins = float(self.sim_ins.get())
                dd = float(self.sim_dd.get() or 0.0)
                jsr_req = float(self.sim_jsr.get() or 15.0)
            except ValueError as ex:
                messagebox.showerror(APP_TITLE, "Симулятор: %s" % ex)
                return
            m = getattr(self, "fm_cur", None) or FIBER_MODEMS[5]
            s = fiber_sim_modem(m, p, div, ins, (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
                                d_drone_m=dd, jsr_req_db=jsr_req)
            self.sim_last = s
            txt = ("сигнал дрона на фотодіоді: %.1f дБм (λ %d нм, %.2f дБ/км · %.4g км) · "
                   "макс. дистанції впливу: засліплення %s, зрив CDR %s, будь-який ефект (J/S ≥ %.0f дБ) %s"
                   % (s["p_sig_dbm"], s["lam_nm"], s["alpha_db_km"], dd / 1000.0,
                      ("%.4g м" % s["d_blind_m"]) if s["d_blind_m"] is not None else "—",
                      ("%.4g м" % s["d_cdr_m"]) if s["d_cdr_m"] is not None else "—",
                      jsr_req, ("%.4g м" % s["d_eff_m"]) if s["d_eff_m"] is not None else "—"))
            if getattr(self, "fm_res", None):
                try:
                    e_req = float(self.fru.get() or 0.5) / float(self.frl.get() or 0.1)
                except Exception:
                    e_req = 5.0
                txt += " · резонанс контролера: EIRP %.1f дБм → вплив до %.3g м" % (
                    self.fm_res["eirp_dbm"], fiber_sim_resonance(self.fm_res["eirp_dbm"], e_req))
            else:
                txt += " · резонанс контролера: виконайте 🎯 розрахунок у вікні ОВ-модуля"
            self.sim_lbl.config(text=txt)
            for wdg in self.sim_rows_frame.winfo_children():
                wdg.destroy()
            ttk.Label(self.sim_rows_frame,
                      text="дистанція інжектора, м · P на фотодіоді, дБм · статус · J/S, дБ",
                      font=("DejaVu Sans", 9, "bold")).pack(anchor="w")
            for (d, pj, code, t2, jsr) in s["rows"]:
                col = THEME["text"] if code == 0 else (THEME["warn"] if code == 1 else "#c62828")
                ttk.Label(self.sim_rows_frame, text="%.4g · %+.1f · %s · %+.1f" % (d, pj, t2, jsr),
                          foreground=col).pack(anchor="w", pady=0)
            cv = self.sim_canvas
            cv.delete("all")
            W = max(int(cv.winfo_width() or 0), 880)
            H = 250
            x0, x1, y0, y1 = 56, W - 12, 12, H - 26
            dmin, dmax = math.log10(0.3), math.log10(300.0)
            ymin, ymax = -80.0, 25.0

            def X(d):
                lx = math.log10(max(1e-3, float(d)))
                return x0 + (lx - dmin) / (dmax - dmin) * (x1 - x0)

            def Y(v):
                return y1 - (max(ymin, min(ymax, float(v))) - ymin) / (ymax - ymin) * (y1 - y0)

            for v in range(-80, 21, 20):
                cv.create_line(x0, Y(v), x1, Y(v), fill=THEME["dim"], dash=(2, 4))
                cv.create_text(x0 - 4, Y(v), text="%d" % v, fill=THEME["dim"],
                               anchor="e", font=("DejaVu Sans", 7))
            for d in (0.5, 1, 2, 5, 10, 20, 50, 100, 200):
                cv.create_text(X(d), y1 + 9, text="%.4g" % d, fill=THEME["dim"],
                               font=("DejaVu Sans", 7))
            pts = " ".join("%.1f %.1f" % (X(d), Y(v)) for (d, v, c, t3, j2) in s["rows"])
            cv.create_line(pts, fill=THEME["accent"], width=2, smooth=True)
            cv.create_line(x0, Y(s["sens_dbm"]), x1, Y(s["sens_dbm"]), fill=THEME["warn"], dash=(4, 3))
            cv.create_text(x1 - 4, Y(s["sens_dbm"]) - 6, text="чутливість %.0f дБм" % s["sens_dbm"],
                           fill=THEME["warn"], anchor="e", font=("DejaVu Sans", 7))
            cv.create_line(x0, Y(s["overload_dbm"]), x1, Y(s["overload_dbm"]), fill="#c62828", dash=(4, 3))
            cv.create_text(x1 - 4, Y(s["overload_dbm"]) - 6, text="засліплення %.0f дБм" % s["overload_dbm"],
                           fill="#c62828", anchor="e", font=("DejaVu Sans", 7))
            cv.create_line(x0, Y(s["p_sig_dbm"]), x1, Y(s["p_sig_dbm"]), fill=THEME["text"], dash=(6, 3))
            cv.create_text(x1 - 4, Y(s["p_sig_dbm"]) + 9, text="сигнал дрона %.1f дБм" % s["p_sig_dbm"],
                           fill=THEME["text"], anchor="e", font=("DejaVu Sans", 7))
            self._set_status("симулятор ОВ-завади: %s" % txt[:90])

        def _fpv_scan_now(self):
            """① Скан FPV-діапазону: піки з аналізу .raw або вручну → таблиця каналів."""
            try:
                _band = self.fpv_band.get()
                if self.fpv_src.get() == "вручну":
                    import re as _re57
                    _cmap = {}
                    for _lab, _f in fpv_channels(_band):
                        _d = _re57.findall(r"\d+", _lab)
                        if _d:
                            _cmap[int(_d[0])] = _f
                    _peaks = [_cmap[int(_t)] for _t in _re57.findall(r"\d+", self.fpv_man.get() or "")
                              if int(_t) in _cmap]
                    _lvl = float(self.fpv_lvl.get())
                else:
                    _ana = self._melt_peaks_now()
                    _peaks = []
                    if _ana:
                        for _p in _ana["peaks"]:
                            if isinstance(_p, dict):
                                _peaks.append(float(_p.get("f_hz")))
                            elif isinstance(_p, (tuple, list)):
                                _peaks.append(float(_p[0]))
                    _lvl = float(_ana.get("sig_dbm", -60.0)) if _ana else float(self.fpv_lvl.get())
                _rows = fpv_scan_table(_band, _peaks, level_dbm=_lvl)
                self.fpv_last_active = sum(1 for _r in _rows if _r[3] == "ВІДЕО")
                for _w in self.fpv_rows_fr.winfo_children():
                    _w.destroy()
                for _j, _h in enumerate(("Канал", "f, МГц", "Рівень, дБм", "Статус")):
                    ttk.Label(self.fpv_rows_fr, text=_h, font=("DejaVu Sans", 8, "bold")).grid(
                        row=0, column=_j, sticky="w", padx=6)
                for _i, (_lab, _fm, _lv, _st) in enumerate(_rows):
                    ttk.Label(self.fpv_rows_fr, text=_lab).grid(row=_i + 1, column=0, sticky="w", padx=6)
                    ttk.Label(self.fpv_rows_fr, text="%.0f" % _fm).grid(row=_i + 1, column=1, sticky="w", padx=6)
                    ttk.Label(self.fpv_rows_fr, text=("%.1f" % _lv) if _lv is not None else "—").grid(
                        row=_i + 1, column=2, sticky="w", padx=6)
                    ttk.Label(self.fpv_rows_fr, text=_st,
                              foreground=THEME["accent"] if _st == "ВІДЕО" else THEME["text"]).grid(
                        row=_i + 1, column=3, sticky="w", padx=6)
                _q, _v = video_quality_est(float(self.fpv_snr.get()))
                self.fpv_q_lbl.config(text="Якість прийому за SNR %s дБ: %d%% — %s"
                                      % (self.fpv_snr.get(), _q, _v))
                self.fpv_sum.config(text="Активних каналів з відео: %d з %d. Оберіть канал → у ③ підставиться частота."
                                    % (self.fpv_last_active, len(_rows)))
                self._set_status("FPV-скан %s: %d активних" % (_band, self.fpv_last_active))
            except Exception as _e57x:
                self.fpv_sum.config(text="Помилка скану: %s" % _e57x)

        def _fpv_eirp_now(self):
            """② Лінк-бюджет інжектора + вердикт FM-захоплення."""
            try:
                _f = float(self.fpv_f.get()) * 1e6
                _r = injection_eirp(float(self.fpv_prx.get()), float(self.fpv_marg.get()),
                                    float(self.fpv_d.get()), _f, float(self.fpv_g.get()))
                _cap = capture_effect(float(self.fpv_marg.get()))
                self.fpv_eirp_last = _r["eirp_dbm"]
                self.fpv_eirp_lbl.config(text="EIRP інжектора %.1f дБм (FSPL %.1f дБ на %s км) — %s; FM: %s — %s"
                                         % (_r["eirp_dbm"], _r["fspl_db"], self.fpv_d.get(),
                                            _r["verdict"], _cap["state"], _cap["note"]))
                self._set_status("FPV-інжекція: EIRP %.1f дБм, %s" % (_r["eirp_dbm"], _cap["state"]))
            except Exception as _e57x:
                self.fpv_eirp_lbl.config(text="Помилка розрахунку: %s" % _e57x)

        def _fpv_apply_pp(self):
            """Підставити ПП у Крок 2в: ПП = EIRP − підсилювач − антена."""
            try:
                _e = float(getattr(self, "fpv_eirp_last"))
                _pa = _e - float(self.jq_amp.get()) - float(self.jq_ant_g.get())
                self.jq_pa.set("%.1f" % _pa)
                self._set_status("Крок 2в: ПП = EIRP %.1f − підсилювач − антена = %.1f дБм" % (_e, _pa))
            except Exception as _e57x:
                self._set_status("Не вдалося підставити ПП: %s" % _e57x)

        def _fpv_pick_img(self):
            _p = filedialog.askopenfilename(title="Зображення для інжекції (PNG/JPG)",
                                            filetypes=(("Зображення", "*.png *.jpg *.jpeg *.bmp"),
                                                       ("Усі файли", "*.*")))
            if _p:
                self.fpv_img_path = _p
                self.fpv_pat.set("Зображення (файл)…")

        def _sa_draw(self, fr, tr, pks):
            cv = self.sa_canvas
            cv.delete("all")
            if tr is None or tr.size < 4:
                return
            _W = max(float(cv.winfo_width()), 1000.0)
            _H = max(float(cv.winfo_height()), 160.0)
            _x0, _y0, _x1, _y1 = 44.0, 14.0, _W - 14.0, _H - 26.0
            _lo, _hi = float(np.min(tr)), float(np.max(tr))
            _pad = max(3.0, (_hi - _lo) * 0.15)

            def _X(f):
                return _x0 + (_x1 - _x0) * (f - float(fr[0])) / max(float(fr[-1] - fr[0]), 1e-9)

            def _Y(v):
                return _y1 - (_y1 - _y0) * max(0.0, min(1.0, (v - (_lo - _pad)) / max((_hi + _pad) - (_lo - _pad), 1e-9)))

            cv.create_line("%.1f %.1f %.1f %.1f" % (_x0, _y1, _x1, _y1), fill=THEME["text"])
            cv.create_line("%.1f %.1f %.1f %.1f" % (_x0, _y0, _x0, _y1), fill=THEME["text"])
            _step = max(1, tr.size // 500)
            _pts = " ".join("%.1f %.1f" % (_X(fr[i]), _Y(tr[i])) for i in range(0, tr.size, _step))
            cv.create_line(_pts, fill=THEME["accent"], width=2)
            for _p in pks:
                cv.create_oval("%.1f %.1f %.1f %.1f" % (_X(_p["f_hz"]) - 3, _Y(_p["dbm"]) - 3,
                                                        _X(_p["f_hz"]) + 3, _Y(_p["dbm"]) + 3),
                               fill=THEME["accent"], outline="")
                cv.create_text(_X(_p["f_hz"]), _Y(_p["dbm"]) - 10, text="%.2f" % (_p["f_hz"] / 1e6),
                               fill=THEME["accent"], font=("DejaVu Sans", 7), anchor="s")
            cv.create_text(_x0 + 4, _y0 + 4, text="дБм", fill=THEME["text"], anchor="w", font=("DejaVu Sans", 7))
            cv.create_text((_x0 + _x1) / 2.0, _y1 + 12, text="частота, МГц", fill=THEME["text"],
                           font=("DejaVu Sans", 8))
            cv.create_text(_x0 + 26, _y1 + 12, text="%.0f" % (fr[0] / 1e6), fill=THEME["text"], font=("DejaVu Sans", 7))
            cv.create_text(_x1 - 26, _y1 + 12, text="%.0f" % (fr[-1] / 1e6), fill=THEME["text"], font=("DejaVu Sans", 7))

        def _sa_connect(self):
            try:
                self._sa_client = ScpiClient(self.sa_host.get(), int(self.sa_port.get()),
                                             float(self.sa_to.get()))
                self._sa_client.connect()
                _id = self._sa_client.identify()
                self.sa_info.config(text="Підключено: %s %s (сер. %s, ПЗ %s) — SCPI %s:%s"
                                    % (_id["vendor"], _id["model"], _id["serial"] or "—",
                                       _id["fw"] or "—", self.sa_host.get(), self.sa_port.get()))
                self._set_status("спектроаналізатор: підключено %s %s" % (_id["vendor"], _id["model"]))
            except Exception as _e58x:
                self.sa_info.config(text="Помилка підключення: %s" % _e58x)

        def _sa_fetch(self):
            try:
                _cl = getattr(self, "_sa_client", None)
                if _cl is None or _cl.sock is None:
                    raise ValueError("спершу підключіться до приладу")
                _fc = float(self.sa_fc.get()) * 1e6
                _span = float(self.sa_span.get()) * 1e6
                _rbw = float(self.sa_rbw.get()) * 1e3
                _cl.setup(_fc, _span, _rbw, float(self.sa_ref.get()))
                _tr = _cl.fetch_trace(cmd=getattr(self, "_sa_trace_cmd", None))
                _n = _tr.size
                if _n < 4:
                    raise ValueError("порожня траса від приладу")
                _fr = trace_freqs(_fc, _span, _n)
                _pks = trace_top_peaks(_fr, _tr)
                self.sa_last = {"peaks": [{"f_hz": _p["f_hz"], "peak_db": _p["dbm"],
                                           "width_hz": max(_rbw, 1.0)} for _p in _pks],
                                "freqs_hz": _fr, "trace_dbm": _tr,
                                "sig_dbm": float(np.max(_tr)),
                                "note": "SCPI: %s:%s" % (self.sa_host.get(), self.sa_port.get())}
                self.sa_status.config(text="Трасу знято: %d точок, %.1f…%.1f дБм; піків: %d; RBW %.0f кГц"
                                      % (_n, float(np.min(_tr)), float(np.max(_tr)), len(_pks), _rbw / 1e3))
                self.sa_peaks_lbl.config(text="Піки: " + (", ".join("%.3f МГц · %.1f дБм"
                                              % (_p["f_hz"] / 1e6, _p["dbm"]) for _p in _pks[:8])
                                              if _pks else "не знайдено"))
                self._sa_draw(_fr, _tr, _pks)
                self._set_status("спектроаналізатор: траса %d точок, %d піків" % (_n, len(_pks)))
            except Exception as _e58x:
                self.sa_status.config(text="Помилка зйомки траси: %s" % _e58x)

        def _sa_to_profile(self):
            if not getattr(self, "sa_last", None) or not self.sa_last.get("peaks"):
                self.sa_status.config(text="Немає піків — спершу зніміть трасу")
                return
            self._last_ana = self.sa_last
            self._set_status("піки траси → профіль завади (працює і для FPV-скану «з аналізу»)")
            self._jamprof_open()

        def _sa_csv(self):
            try:
                _sl = getattr(self, "sa_last", None)
                if not _sl or _sl.get("trace_dbm") is None:
                    raise ValueError("спершу зніміть трасу")
                _out = filedialog.asksaveasfilename(defaultextension=".csv",
                                                    initialfile="sa_trace.csv",
                                                    title="Куди зберегти трасу (CSV)")
                if not _out:
                    return
                with open(_out, "w", encoding="utf-8") as _fo:
                    _fo.write("freq_hz,dbm\n")
                    for _i in range(_sl["trace_dbm"].size):
                        _fo.write("%.3f,%.2f\n" % (_sl["freqs_hz"][_i], _sl["trace_dbm"][_i]))
                self.sa_status.config(text="Трасу збережено: %s (%d точок)" % (_out, _sl["trace_dbm"].size))
            except Exception as _e58x:
                self.sa_status.config(text="Помилка експорту: %s" % _e58x)

        def _sa_open(self):
            win, top, _lbl = self._win("🔌 Спектроаналізатор (SCPI) — зйомка траси і піків", 1140, 660)
            s1 = ttk.LabelFrame(top, text="① З'єднання — SCPI over TCP (VISA не потрібен)")
            s1.pack(fill="x", padx=8, pady=(8, 3))
            _r1 = ttk.Frame(s1)
            _r1.pack(fill="x", pady=1)
            ttk.Label(_r1, text="адреса:").pack(side="left")
            self.sa_host = tk.StringVar(value="192.168.0.10")
            ttk.Entry(_r1, textvariable=self.sa_host, width=15).pack(side="left", padx=3)
            ttk.Label(_r1, text="порт:").pack(side="left", padx=(8, 0))
            self.sa_port = tk.StringVar(value="5025")
            ttk.Entry(_r1, textvariable=self.sa_port, width=6).pack(side="left", padx=3)
            ttk.Label(_r1, text="таймаут, с:").pack(side="left", padx=(8, 0))
            self.sa_to = tk.StringVar(value="2")
            ttk.Entry(_r1, textvariable=self.sa_to, width=4).pack(side="left", padx=3)
            ttk.Button(_r1, text="🔗 Підключитися (*IDN?)", command=self._sa_connect).pack(side="left", padx=(10, 0))
            self.sa_info = ttk.Label(s1, text="R&S FSV/FSW · Siglent SSA/SHA/SVA · Rigol RSA/DSA · Keysight · TinySA (порт 5025/5555)",
                                     foreground=THEME["accent"], wraplength=1090, justify="left")
            self.sa_info.pack(fill="x", pady=(2, 3))
            s2 = ttk.LabelFrame(top, text="② Sweep і траса — центр/span/RBW/рівень, зйомка :TRAC?")
            s2.pack(fill="x", padx=8, pady=3)
            _r2 = ttk.Frame(s2)
            _r2.pack(fill="x", pady=1)
            ttk.Label(_r2, text="центр, МГц:").pack(side="left")
            self.sa_fc = tk.StringVar(value="915")
            ttk.Entry(_r2, textvariable=self.sa_fc, width=8).pack(side="left", padx=3)
            ttk.Label(_r2, text="span, МГц:").pack(side="left", padx=(8, 0))
            self.sa_span = tk.StringVar(value="20")
            ttk.Entry(_r2, textvariable=self.sa_span, width=7).pack(side="left", padx=3)
            ttk.Label(_r2, text="RBW, кГц:").pack(side="left", padx=(8, 0))
            self.sa_rbw = tk.StringVar(value="100")
            ttk.Entry(_r2, textvariable=self.sa_rbw, width=6).pack(side="left", padx=3)
            ttk.Label(_r2, text="рівень, дБм:").pack(side="left", padx=(8, 0))
            self.sa_ref = tk.StringVar(value="0")
            ttk.Entry(_r2, textvariable=self.sa_ref, width=5).pack(side="left", padx=3)
            ttk.Button(_r2, text="📥 Зняти трасу", command=self._sa_fetch).pack(side="left", padx=(10, 0))
            self.sa_status = ttk.Label(s2, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.sa_status.pack(fill="x", pady=(2, 1))
            self.sa_canvas = tk.Canvas(s2, height=200, bg=THEME["panel"], highlightthickness=0)
            self.sa_canvas.pack(fill="x", pady=(0, 3))
            s3 = ttk.LabelFrame(top, text="③ Піки траси → профіль завади / FPV")
            s3.pack(fill="x", padx=8, pady=3)
            self.sa_peaks_lbl = ttk.Label(s3, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.sa_peaks_lbl.pack(fill="x", pady=(2, 3))
            _br = ttk.Frame(top)
            _br.pack(fill="x", padx=8, pady=(2, 8))
            ttk.Button(_br, text="🎯 → Піки у профіль завади", command=self._sa_to_profile).pack(side="left")
            ttk.Button(_br, text="💾 CSV траси…", command=self._sa_csv).pack(side="left", padx=(10, 0))
            ttk.Button(_br, text="Закрити", command=win.destroy).pack(side="right")

        def _rec_open(self):
            """Діалог «Рецепти SDR — підбір під парк платформ і частоту» (v5.9.0)."""
            win, top, _lbl = self._win("📋 Рецепти SDR — підбір під парк платформ і частоту", 1180, 700)
            self.rec_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            rowf = ttk.Frame(f)
            rowf.pack(fill="x", pady=(0, 4))
            ttk.Label(rowf, text="Ціль (Тип сигналу):").pack(side="left")
            self.rec_tgt = tk.StringVar(value="GPS L1 C/A (ШПС)")
            ttk.Combobox(rowf, textvariable=self.rec_tgt, width=30, state="readonly",
                         values=list(AI_TARGETS.keys())).pack(side="left", padx=(4, 14))
            ttk.Label(rowf, text="Частота, МГц (опційно):").pack(side="left")
            self.rec_f = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=self.rec_f, width=8).pack(side="left", padx=(4, 14))
            ttk.Label(rowf, text="Дистанція завади, км:").pack(side="left")
            self.rec_dj = tk.StringVar(value="1")
            ttk.Entry(rowf, textvariable=self.rec_dj, width=6).pack(side="left", padx=(4, 14))
            ttk.Label(rowf, text="Дальність розкриття, км:").pack(side="left")
            self.rec_dist = tk.StringVar(value="5")
            ttk.Entry(rowf, textvariable=self.rec_dist, width=6).pack(side="left", padx=4)
            row2 = ttk.Frame(f)
            row2.pack(fill="x", pady=(0, 4))
            ttk.Label(row2, text="ПП, дБм:").pack(side="left")
            self.rec_pa = tk.StringVar(value="20")
            ttk.Entry(row2, textvariable=self.rec_pa, width=6).pack(side="left", padx=(4, 14))
            ttk.Label(row2, text="Підсилювач, дБ:").pack(side="left")
            self.rec_amp = tk.StringVar(value="0")
            ttk.Entry(row2, textvariable=self.rec_amp, width=6).pack(side="left", padx=(4, 14))
            ttk.Label(row2, text="Антена:").pack(side="left")
            self.rec_ant = tk.StringVar(value="Штирьова 0.25λ")
            ttk.Combobox(row2, textvariable=self.rec_ant, width=22, state="readonly",
                         values=list(JQ_ANTENNAS.keys())).pack(side="left", padx=(4, 14))
            ttk.Button(row2, text="Усі TX", command=self._rec_all_tx).pack(side="left", padx=4)
            ttk.Button(row2, text="Зняти все", command=self._rec_none).pack(side="left", padx=4)
            pf = ttk.LabelFrame(f, text="Парк платформ (Крок 1)", padding=4)
            pf.pack(fill="x")
            self.rec_vars = {}
            _names = [n for n in SDR_PRESETS if n != "Інший (власноруч)"]
            for _i, _n in enumerate(_names):
                _v = tk.BooleanVar(value=(SDR_PRESETS[_n].get("tx_dbm") is not None))
                self.rec_vars[_n] = _v
                ttk.Checkbutton(pf, text=_n, variable=_v).grid(row=_i // 5, column=_i % 5, sticky="w")
            brow = ttk.Frame(f)
            brow.pack(fill="x", pady=4)
            ttk.Button(brow, text="📋 Підібрати рецепти", command=self._rec_pick).pack(side="left", padx=4)
            ttk.Button(brow, text="💾 Експорт CSV…", command=self._rec_export).pack(side="left", padx=4)
            ttk.Button(brow, text="💾 Зберегти парк…", command=self._rec_save_park).pack(side="left", padx=4)
            ttk.Button(brow, text="📂 Завантажити парк…", command=self._rec_load_park).pack(side="left", padx=4)
            ttk.Button(brow, text="📄 PDF-звіт…", command=self._rec_pdf).pack(side="left", padx=4)
            self.rec_status = tk.StringVar(value="Платформ у парку: %d" % sum(
                1 for _v in self.rec_vars.values() if _v.get()))
            ttk.Label(f, textvariable=self.rec_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("model", "verdict", "kind", "f", "bw", "js", "preq", "tx", "r", "note")
            self.rec_tree = ttk.Treeview(f, columns=_cols, show="headings", height=14)
            for _cid, (_t, _w) in zip(_cols, [("Платформа", 150), ("Вердикт", 74), ("Тип завади", 128),
                                              ("Частота МГц", 76), ("Смуга МГц", 76), ("J/S дБ", 58),
                                              ("Потрібно дБм", 78), ("TX дБм", 64), ("Радіус км", 76),
                                              ("Примітка", 210)]):
                self.rec_tree.heading(_cid, text=_t)
                self.rec_tree.column(_cid, width=_w,
                                     anchor="w" if _cid in ("model", "kind", "note") else "center")
            self.rec_tree.tag_configure("ok", foreground="#1b5e20")
            self.rec_tree.tag_configure("part", foreground="#b26a00")
            self.rec_tree.tag_configure("no", foreground="#8a94a0")
            self.rec_tree.tag_configure("rx", foreground="#c62828")
            self.rec_tree.pack(fill="both", expand=True)
            self.rec_tree.bind("<Double-1>", self._rec_apply)

        def _rec_all_tx(self):
            for _n, _v in self.rec_vars.items():
                _v.set(SDR_PRESETS[_n].get("tx_dbm") is not None)

        def _rec_none(self):
            for _v in self.rec_vars.values():
                _v.set(False)

        def _rec_pick(self):
            models = [_n for _n, _v in self.rec_vars.items() if _v.get()]
            _s = self.rec_f.get().strip()
            try:
                custom = float(_s.replace(",", ".")) * 1e6 if _s else None
            except ValueError:
                custom = None
            try:
                dj = max(0.01, float(self.rec_dj.get().replace(",", ".")))
            except ValueError:
                dj = 1.0
            try:
                dist = max(0.01, float(self.rec_dist.get().replace(",", ".")))
            except ValueError:
                dist = 5.0
            try:
                pa = float(self.rec_pa.get().replace(",", "."))
            except ValueError:
                pa = 20.0
            try:
                amp = float(self.rec_amp.get().replace(",", "."))
            except ValueError:
                amp = 0.0
            rows = recipe_plan(self.rec_tgt.get(), models, pa_dbm=pa, amp_db=amp,
                               ant_name=self.rec_ant.get(), dist_km=dist, dj_km=dj,
                               custom_hz=custom)
            self.rec_rows = rows
            _t = self.rec_tree
            for _iid in _t.get_children():
                _t.delete(_iid)
            _tag = {"OK": "ok", "СМУГА": "part", "ПОЗА": "no", "ПРИЙОМ": "rx"}
            for _r in rows:
                _t.insert("", "end", values=(
                    _r["model"], _r["verdict"], _r["kind_name"],
                    "—" if _r["f_hz"] is None else "%.6g" % (_r["f_hz"] / 1e6),
                    "—" if _r["bw_hz"] is None else "%.6g" % (_r["bw_hz"] / 1e6),
                    "—" if _r["js_req"] is None else "%.0f" % _r["js_req"],
                    "—" if _r["p_req"] is None else "%.1f" % _r["p_req"],
                    "—" if _r["tx_dbm"] is None else "%.0f" % _r["tx_dbm"],
                    "—" if _r["sdr_d_max"] is None else "%.6g" % _r["sdr_d_max"],
                    _r["note"]), tags=(_tag.get(_r["verdict"], ""),))
            self.rec_status.set(
                "Паркових платформ: %d · повністю: %d · зі звуженням: %d · поза діапазоном: %d · приймальні: %d"
                % (len(models),
                   sum(1 for _r in rows if _r["verdict"] == "OK"),
                   sum(1 for _r in rows if _r["verdict"] == "СМУГА"),
                   sum(1 for _r in rows if _r["verdict"] == "ПОЗА"),
                   sum(1 for _r in rows if _r["verdict"] == "ПРИЙОМ")))

        def _rec_apply(self, _ev=None):
            _sel = self.rec_tree.selection()
            if not _sel:
                return
            _model = self.rec_tree.item(_sel[0], "values")[0]
            if _model in SDR_PRESETS:
                self.cb_sdr.set(_model)
            self._set_status("Крок 1: модель SDR → %s (v5.9.0)" % _model)

        def _rec_export(self):
            if not getattr(self, "rec_rows", None):
                self._set_status("Рецепти не підібрані")
                return
            import csv
            from tkinter import filedialog
            _pth = filedialog.asksaveasfilename(defaultextension=".csv",
                                                initialfile="sdr_recipes.csv",
                                                filetypes=[("CSV", "*.csv")])
            if not _pth:
                return
            with open(_pth, "w", encoding="utf-8-sig", newline="") as _fh:
                _w = csv.writer(_fh, delimiter=";")
                _w.writerow(["Ціль", "Платформа", "Вердикт", "Тип завади", "Частота МГц",
                             "Смуга МГц", "J/S дБ", "Потрібно дБм", "TX дБм", "Радіус км", "Примітка"])
                for _r in self.rec_rows:
                    _w.writerow([self.rec_tgt.get(), _r["model"], _r["verdict"], _r["kind_name"],
                                 "" if _r["f_hz"] is None else "%.6g" % (_r["f_hz"] / 1e6),
                                 "" if _r["bw_hz"] is None else "%.6g" % (_r["bw_hz"] / 1e6),
                                 "" if _r["js_req"] is None else "%.0f" % _r["js_req"],
                                 "" if _r["p_req"] is None else "%.1f" % _r["p_req"],
                                 "" if _r["tx_dbm"] is None else "%.0f" % _r["tx_dbm"],
                                 "" if _r["sdr_d_max"] is None else "%.6g" % _r["sdr_d_max"],
                                 _r["note"]])
            self._set_status("Рецепти → %s" % _pth)

        def _cmp_open(self):
            """Діалог «Порівняння записів .raw» (v6.0.0)."""
            from tkinter import filedialog as _fd6
            win, top, _lbl = self._win("⚖ Порівняння записів .raw — до/після завади", 980, 560)
            self.cmp_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            rowf = ttk.Frame(f)
            rowf.pack(fill="x", pady=(0, 4))
            ttk.Label(rowf, text="Запис ДО:").pack(side="left")
            self.cmp_a = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=self.cmp_a, width=32).pack(side="left", padx=4)
            ttk.Button(rowf, text="…", command=lambda: self.cmp_a.set(
                _fd6.askopenfilename(filetypes=[("RAW", "*.raw"), ("Усі файли", "*.*")])
                or self.cmp_a.get())).pack(side="left")
            ttk.Label(rowf, text="ПІСЛЯ:").pack(side="left", padx=(10, 0))
            self.cmp_b = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=self.cmp_b, width=32).pack(side="left", padx=4)
            ttk.Button(rowf, text="…", command=lambda: self.cmp_b.set(
                _fd6.askopenfilename(filetypes=[("RAW", "*.raw"), ("Усі файли", "*.*")])
                or self.cmp_b.get())).pack(side="left")
            row2 = ttk.Frame(f)
            row2.pack(fill="x", pady=(0, 4))
            ttk.Label(row2, text="Каналів у записі:").pack(side="left")
            self.cmp_nch = tk.StringVar(value=str(getattr(self, "nch", DEFAULT_NCH)))
            ttk.Entry(row2, textvariable=self.cmp_nch, width=5).pack(side="left", padx=4)
            ttk.Button(row2, text="⚖ Порівняти", command=self._cmp_run).pack(side="left", padx=10)
            self.cmp_status = tk.StringVar(value="оберіть два записи .raw")
            ttk.Label(f, textvariable=self.cmp_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("bin", "before", "after", "delta")
            self.cmp_tree = ttk.Treeview(f, columns=_cols, show="headings", height=10)
            for _cid, (_t, _w) in zip(_cols, [("Бін", 70), ("До, дБ", 100), ("Після, дБ", 100),
                                              ("Δ придушення, дБ", 140)]):
                self.cmp_tree.heading(_cid, text=_t)
                self.cmp_tree.column(_cid, width=_w, anchor="center")
            self.cmp_tree.pack(fill="both", expand=True)

        def _cmp_run(self):
            try:
                if not self.cmp_a.get() or not self.cmp_b.get():
                    raise ValueError("оберіть обидва файли")
                _r = compare_raws(self.cmp_a.get(), self.cmp_b.get(), nch=int(self.cmp_nch.get()))
                _t = self.cmp_tree
                for _iid in _t.get_children():
                    _t.delete(_iid)
                for _p in _r["peaks"]:
                    _t.insert("", "end", values=(_p["bin"], "%.1f" % _p["before_db"],
                                                 "%.1f" % _p["after_db"], "%+.1f" % _p["delta_db"]))
                _sup = sum(1 for _p in _r["peaks"] if _p["delta_db"] >= 10.0)
                self.cmp_status.set("піків: %d · середнє Δ %.1f дБ · придушено (Δ≥10 дБ): %d"
                                    % (len(_r["peaks"]), _r["delta_mean_db"], _sup))
                hist_log("Порівняння", "Δ середнє %.1f дБ, піків %d" % (_r["delta_mean_db"], len(_r["peaks"])),
                         {"a": os.path.basename(self.cmp_a.get()), "b": os.path.basename(self.cmp_b.get())})
            except Exception as _e6c:
                self.cmp_status.set("Помилка: %s" % _e6c)

        def _hist_open(self):
            """Діалог «Журнал сесій» (v6.0.0)."""
            win, top, _lbl = self._win("🗂 Журнал сесій — історія аналізів, рецептів, симуляцій", 940, 560)
            self.hist_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            brow = ttk.Frame(f)
            brow.pack(fill="x", pady=(0, 4))
            ttk.Button(brow, text="⟳ Оновити", command=self._hist_fill).pack(side="left", padx=4)
            ttk.Button(brow, text="💾 Експорт CSV…", command=self._hist_export).pack(side="left", padx=4)
            ttk.Button(brow, text="🗑 Очистити", command=self._hist_clear).pack(side="left", padx=4)
            _cols = ("ts", "module", "summary", "data")
            self.hist_tree = ttk.Treeview(f, columns=_cols, show="headings", height=16)
            for _cid, (_t, _w) in zip(_cols, [("Час", 130), ("Модуль", 110), ("Подія", 380), ("Дані", 240)]):
                self.hist_tree.heading(_cid, text=_t)
                self.hist_tree.column(_cid, width=_w, anchor="w")
            self.hist_tree.pack(fill="both", expand=True)
            self._hist_fill()

        def _hist_fill(self):
            _t = self.hist_tree
            for _iid in _t.get_children():
                _t.delete(_iid)
            for _r in hist_rows(200):
                _t.insert("", "end", values=(_r[0], _r[1], _r[2], (_r[3] or "")[:60]))

        def _hist_export(self):
            from tkinter import filedialog as _fd6
            _p = _fd6.asksaveasfilename(defaultextension=".csv",
                                        initialfile="raw_studio_history.csv",
                                        filetypes=[("CSV", "*.csv")])
            if not _p:
                return
            _n = hist_export_csv(_p)
            self._set_status("журнал → %s (%d записів)" % (_p, _n))

        def _hist_clear(self):
            if not messagebox.askyesno(APP_TITLE, "Очистити журнал сесій?"):
                return
            try:
                import sqlite3
                _con = sqlite3.connect(hist_db_path())
                _con.execute("DELETE FROM events")
                _con.commit()
                _con.close()
            except Exception:
                pass
            self._hist_fill()
            self._set_status("журнал очищено")

        def _pan_open(self):
            """Діалог «Панорама запису .raw» (v6.0.0)."""
            from tkinter import filedialog as _fd6
            win, top, _lbl = self._win("🗺 Панорама запису .raw — спектри каналів у блоках 100М…6Г", 1000, 620)
            self.pan_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            rowf = ttk.Frame(f)
            rowf.pack(fill="x", pady=(0, 4))
            ttk.Label(rowf, text="Запис:").pack(side="left")
            self.pan_path = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=self.pan_path, width=40).pack(side="left", padx=4)
            ttk.Button(rowf, text="…", command=lambda: self.pan_path.set(
                _fd6.askopenfilename(filetypes=[("RAW", "*.raw"), ("Усі файли", "*.*")])
                or self.pan_path.get())).pack(side="left")
            ttk.Label(rowf, text="Каналів:").pack(side="left", padx=(10, 0))
            self.pan_nch = tk.StringVar(value=str(getattr(self, "nch", DEFAULT_NCH)))
            ttk.Entry(rowf, textvariable=self.pan_nch, width=4).pack(side="left", padx=4)
            ttk.Button(rowf, text="🗺 Побудувати", command=self._pan_run).pack(side="left", padx=10)
            self.pan_status = tk.StringVar(value="панорама: спектри каналів у блоках FREQ_BLOCKS")
            ttk.Label(f, textvariable=self.pan_status, foreground=THEME["dim"]).pack(anchor="w")
            self.pan_cv = tk.Canvas(f, bg=THEME["panel"], highlightthickness=0, height=420)
            self.pan_cv.pack(fill="both", expand=True)

        def _pan_run(self):
            try:
                if not self.pan_path.get():
                    raise ValueError("оберіть запис .raw")
                _r = panorama_from_raw(self.pan_path.get(), nch=int(self.pan_nch.get()))
                _cv = self.pan_cv
                _cv.delete("all")
                _w = max(600, _cv.winfo_width() or 960)
                _h = max(300, _cv.winfo_height() or 420)
                _x0, _y0, _x1, _y1 = 50, 20, _w - 20, _h - 40
                _f = _r["freqs_hz"]
                _v = _r["dbm"]
                _fmin, _fmax = float(_f.min()), float(_f.max())
                _vmin, _vmax = float(_v.min()), float(_v.max())
                if _fmax - _fmin < 1.0:
                    _fmax = _fmin + 1.0
                if _vmax - _vmin < 1.0:
                    _vmax = _vmin + 1.0
                _xs = _x0 + (_f - _fmin) / (_fmax - _fmin) * (_x1 - _x0)
                _ys = _y1 - (_v - _vmin) / (_vmax - _vmin) * (_y1 - _y0)
                _pts = []
                for _i in range(0, len(_xs), max(1, len(_xs) // 4000)):
                    _pts.extend((float(_xs[_i]), float(_ys[_i])))
                if len(_pts) >= 4:
                    _cv.create_line(_pts, fill=THEME["accent"], width=1)
                for _p in _r["peaks"][:12]:
                    _px = _x0 + (_p["f_hz"] - _fmin) / (_fmax - _fmin) * (_x1 - _x0)
                    if _x0 <= _px <= _x1:
                        _cv.create_line(_px, _y1, _px, _y0 + 14, fill=THEME["warn"], dash=(2, 2))
                        _cv.create_text(_px, _y0 + 4, text="%.1f" % (_p["f_hz"] / 1e6),
                                        fill=THEME["warn"], font=("DejaVu Sans", 7))
                _cv.create_text((_x0 + _x1) / 2.0, _y1 + 12, text="частота, МГц",
                                fill=THEME["text"], font=("DejaVu Sans", 8))
                self.pan_status.set("точок: %d · %.6g–%.6g МГц · піків: %d"
                                    % (len(_f), _fmin / 1e6, _fmax / 1e6, len(_r["peaks"])))
                hist_log("Панорама", "точок %d, піків %d" % (len(_f), len(_r["peaks"])))
            except Exception as _e6p:
                self.pan_status.set("Помилка: %s" % _e6p)

        def _imp_open(self):
            """Діалог «Імпорт IQ → растр FPV» (v6.0.0)."""
            from tkinter import filedialog as _fd6
            win, top, _lbl = self._win("📂 Імпорт IQ-запису → растр FPV", 920, 540)
            self.imp_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            rowf = ttk.Frame(f)
            rowf.pack(fill="x", pady=(0, 4))
            ttk.Label(rowf, text="Запис (u2):").pack(side="left")
            self.imp_path = tk.StringVar(value="")
            ttk.Entry(rowf, textvariable=self.imp_path, width=30).pack(side="left", padx=4)
            ttk.Button(rowf, text="…", command=lambda: self.imp_path.set(
                _fd6.askopenfilename(filetypes=[("RAW", "*.raw"), ("Усі файли", "*.*")])
                or self.imp_path.get())).pack(side="left")
            for _lbl2, _var, _w2, _def in (("fs, МГц", "imp_fs", 7, "16"),
                                           ("Несуча, МГц", "imp_f", 7, "5800"),
                                           ("Девіація, МГц", "imp_dev", 5, "2"),
                                           ("Каналів", "imp_nch", 4, "1")):
                ttk.Label(rowf, text=_lbl2 + ":").pack(side="left", padx=(8, 0))
                setattr(self, _var, tk.StringVar(value=_def))
                ttk.Entry(rowf, textvariable=getattr(self, _var), width=_w2).pack(side="left", padx=2)
            ttk.Label(rowf, text="Стандарт:").pack(side="left", padx=(8, 0))
            self.imp_std = tk.StringVar(value="PAL")
            ttk.Combobox(rowf, textvariable=self.imp_std, width=7, state="readonly",
                         values=list(FPV_STD.keys())).pack(side="left", padx=2)
            ttk.Button(rowf, text="▶ Демодулювати", command=self._imp_run).pack(side="left", padx=10)
            self.imp_status = tk.StringVar(value="оберіть u2/IQ-запис")
            ttk.Label(f, textvariable=self.imp_status, foreground=THEME["dim"]).pack(anchor="w")
            self.imp_cv = tk.Canvas(f, bg="#000000", highlightthickness=0, height=360)
            self.imp_cv.pack(fill="both", expand=True)

        def _imp_run(self):
            try:
                if not self.imp_path.get():
                    raise ValueError("оберіть файл")
                _r = fpv_import_preview(self.imp_path.get(),
                                        float(self.imp_fs.get().replace(",", ".")) * 1e6,
                                        self.imp_std.get(),
                                        float(self.imp_f.get().replace(",", ".")) * 1e6,
                                        float(self.imp_dev.get().replace(",", ".")) * 1e6,
                                        nch=int(self.imp_nch.get()))
                _ras = _r["raster"]
                _h, _w = _ras.shape
                _cv = self.imp_cv
                _cv.delete("all")
                _ch = max(200, _cv.winfo_height() or 360)
                _img = tk.PhotoImage(width=_w, height=_h)
                _g = (_ras * 255).astype(int)
                for _yy in range(_h):
                    _row = " ".join("#%02x%02x%02x" % (_g[_yy, _x] // 2, _g[_yy, _x], _g[_yy, _x] // 2)
                                    for _x in range(_w))
                    _img.put("{" + _row + "}", (0, _yy))
                self._imp_img = _img
                _cv.create_image(10, 10, image=_img, anchor="nw", tags=("ras",))
                _cv.create_text(12, _ch - 10,
                                text="%dx%d · %d відліків · %s" % (_w, _h, _r["samples"], _r["std"]),
                                fill="#7fff9f", anchor="w", font=("DejaVu Sans", 8))
                self.imp_status.set("розгорнуто: %dx%d, %d відліків (%s)"
                                    % (_w, _h, _r["samples"], _r["std"]))
                hist_log("Імпорт IQ", "растр %dx%d" % (_w, _h),
                         {"file": os.path.basename(self.imp_path.get())})
            except Exception as _e6i:
                self.imp_status.set("Помилка: %s" % _e6i)

        def _mc_open(self):
            """Діалог «Монте-Карло follow-me» (v6.0.0)."""
            win, top, _lbl = self._win("🎲 Монте-Карло follow-me — довірчий коридор", 920, 560)
            self.mc_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            rowf = ttk.Frame(f)
            rowf.pack(fill="x", pady=(0, 4))
            for _lbl2, _var, _w2, _def in (("Хопів/с", "mc_hop", 6, "100"),
                                           ("Перестр., мс", "mc_rt", 5, "2"),
                                           ("p виявл.", "mc_pd", 6, "0.9"),
                                           ("σ p", "mc_ps", 5, "0.1"),
                                           ("σ перестр., мс", "mc_rs", 5, "20"),
                                           ("Розигришів", "mc_n", 7, "10000")):
                ttk.Label(rowf, text=_lbl2 + ":").pack(side="left")
                setattr(self, _var, tk.StringVar(value=_def))
                ttk.Entry(rowf, textvariable=getattr(self, _var), width=_w2).pack(side="left", padx=(2, 8))
            ttk.Button(rowf, text="🎲 Розіграти", command=self._mc_run).pack(side="left", padx=4)
            self.mc_status = tk.StringVar(value="частка хопів, які завада встигає накрити")
            ttk.Label(f, textvariable=self.mc_status, foreground=THEME["dim"]).pack(anchor="w")
            self.mc_cv = tk.Canvas(f, bg=THEME["panel"], highlightthickness=0, height=380)
            self.mc_cv.pack(fill="both", expand=True)

        def _mc_run(self):
            try:
                _r = monte_carlo_follow(float(self.mc_hop.get().replace(",", ".")),
                                        float(self.mc_rt.get().replace(",", ".")),
                                        float(self.mc_pd.get().replace(",", ".")),
                                        1,
                                        p_sigma=float(self.mc_ps.get().replace(",", ".")),
                                        retune_sigma_ms=float(self.mc_rs.get().replace(",", ".")),
                                        trials=int(self.mc_n.get()))
                _sh = _r["shares"]
                _cv = self.mc_cv
                _cv.delete("all")
                _w = max(600, _cv.winfo_width() or 860)
                _h = max(300, _cv.winfo_height() or 380)
                _x0, _y0, _x1, _y1 = 50, 20, _w - 20, _h - 40
                _bins = 40
                _lo, _hi = min(_sh), max(max(_sh), min(_sh) + 1e-9)
                _hist = [0] * _bins
                for _s in _sh:
                    _i = min(_bins - 1, int((_s - _lo) / (_hi - _lo) * _bins))
                    _hist[_i] += 1
                _mx = max(max(_hist), 1)
                for _i, _c in enumerate(_hist):
                    _bx0 = _x0 + _i * (_x1 - _x0) / _bins
                    _bx1 = _x0 + (_i + 1) * (_x1 - _x0) / _bins
                    _by = _y1 - _c / _mx * (_y1 - _y0)
                    _cv.create_rectangle(_bx0, _by, _bx1, _y1, outline=THEME["accent"])
                def _vline(_val, _col, _txt):
                    _px = _x0 + (_val - _lo) / (_hi - _lo) * (_x1 - _x0)
                    _cv.create_line(_px, _y0, _px, _y1, fill=_col, dash=(3, 2))
                    _cv.create_text(_px + 3, _y0 + 4, text=_txt, fill=_col, anchor="w",
                                    font=("DejaVu Sans", 7))
                _vline(_r["share_p5"], "#7d8aa5", "P5 %.2f" % _r["share_p5"])
                _vline(_r["share_p50"], "#38bdf8", "P50 %.2f" % _r["share_p50"])
                _vline(_r["share_p95"], "#f87171", "P95 %.2f" % _r["share_p95"])
                self.mc_status.set("розигришів %d · P5 %.2f · P50 %.2f · P95 %.2f (частка хопів)"
                                   % (_r["trials"], _r["share_p5"], _r["share_p50"], _r["share_p95"]))
                hist_log("Монте-Карло", "P50 %.2f (%d розигришів)" % (_r["share_p50"], _r["trials"]))
            except Exception as _e6m:
                self.mc_status.set("Помилка: %s" % _e6m)

        def _tx_open(self):
            """Діалог «SCPI: пресети приладів · керування TX · авто-скан · підклад» (v6.0.0)."""
            win, top, _lbl = self._win("🎛 SCPI: пресети приладів · керування TX · авто-скан", 1000, 560)
            self.tx_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            pf = ttk.LabelFrame(f, text="Пресет приладу (команда зйомки траси)", padding=4)
            pf.pack(fill="x", pady=(0, 4))
            self.tx_preset = tk.StringVar(value="Універсальний (SCPI)")
            ttk.Combobox(pf, textvariable=self.tx_preset, width=26, state="readonly",
                         values=list(SCPI_PRESETS.keys())).pack(side="left", padx=4)
            self.tx_preset_lbl = ttk.Label(pf, text=SCPI_PRESETS["Універсальний (SCPI)"]["note"],
                                           foreground=THEME["dim"])
            self.tx_preset_lbl.pack(side="left", padx=8)
            ttk.Button(pf, text="⟳ Застосувати", command=self._tx_preset_apply).pack(side="left", padx=4)
            tf = ttk.LabelFrame(f, text="Генератор сигналів (TX) — CW на приладі", padding=4)
            tf.pack(fill="x", pady=(0, 4))
            ttk.Label(tf, text="Адреса:").pack(side="left")
            self.tx_host = tk.StringVar(value="127.0.0.1")
            ttk.Entry(tf, textvariable=self.tx_host, width=14).pack(side="left", padx=2)
            ttk.Label(tf, text="Порт:").pack(side="left")
            self.tx_port = tk.StringVar(value="5025")
            ttk.Entry(tf, textvariable=self.tx_port, width=6).pack(side="left", padx=2)
            ttk.Label(tf, text="Частота, МГц:").pack(side="left", padx=(8, 0))
            self.tx_f = tk.StringVar(value="915")
            ttk.Entry(tf, textvariable=self.tx_f, width=9).pack(side="left", padx=2)
            ttk.Label(tf, text="Рівень, дБм:").pack(side="left", padx=(8, 0))
            self.tx_p = tk.StringVar(value="0")
            ttk.Entry(tf, textvariable=self.tx_p, width=6).pack(side="left", padx=2)
            ttk.Button(tf, text="IDN", command=self._tx_idn).pack(side="left", padx=(10, 2))
            ttk.Button(tf, text="Виставити CW", command=self._tx_set).pack(side="left", padx=2)
            ttk.Button(tf, text="Вихід ON", command=lambda: self._tx_outp(True)).pack(side="left", padx=2)
            ttk.Button(tf, text="Вихід OFF", command=lambda: self._tx_outp(False)).pack(side="left", padx=2)
            self.tx_status = tk.StringVar(value="стендовий інструмент: керування — у лабораторії з дозволом на випромінювання")
            ttk.Label(f, textvariable=self.tx_status, foreground=THEME["dim"]).pack(anchor="w")
            af = ttk.LabelFrame(f, text="Авто-скан траси за розкладом", padding=4)
            af.pack(fill="x", pady=(0, 4))
            ttk.Label(af, text="Період, с:").pack(side="left")
            self.tx_per = tk.StringVar(value="10")
            ttk.Entry(af, textvariable=self.tx_per, width=6).pack(side="left", padx=4)
            ttk.Button(af, text="▶ Авто-скан", command=self._tx_auto_start).pack(side="left", padx=4)
            ttk.Button(af, text="■ Стоп", command=self._tx_auto_stop).pack(side="left", padx=4)
            ttk.Button(af, text="💾 Журнал піків → CSV…", command=self._tx_auto_csv).pack(side="left", padx=4)
            self.tx_auto_lbl = ttk.Label(af, text="зйомок: 0", foreground=THEME["dim"])
            self.tx_auto_lbl.pack(side="left", padx=8)
            bf = ttk.LabelFrame(f, text="Підклад із траси → канали Генератора (Шум)", padding=4)
            bf.pack(fill="x")
            ttk.Button(bf, text="⤴ Рівні підкладу → амплітуди шумових каналів",
                       command=self._tx_floor_apply).pack(side="left", padx=4)
            ttk.Label(bf, text="dBFS = пік траси (дБм) − опорний рівень зйомки",
                      foreground=THEME["dim"]).pack(side="left", padx=8)

        def _tx_preset_apply(self):
            _p = SCPI_PRESETS.get(self.tx_preset.get())
            if _p:
                self.tx_preset_lbl.config(text=_p["note"])
                if getattr(self, "_sa_client", None) is not None:
                    self._sa_trace_cmd = _p["trace_cmd"]
                self._set_status("пресет приладу: %s" % self.tx_preset.get())

        def _tx_idn(self):
            try:
                _cl = ScpiClient(self.tx_host.get(), int(self.tx_port.get()), 2.0)
                _cl.connect()
                _id = _cl.identify()
                self.tx_status.set("TX: %s %s (сер. %s)" % (_id["vendor"], _id["model"], _id["serial"] or "—"))
                hist_log("SCPI TX", "IDN %s %s" % (_id["vendor"], _id["model"]))
            except Exception as _e6t:
                self.tx_status.set("Помилка TX: %s" % _e6t)

        def _tx_set(self):
            try:
                _cl = ScpiClient(self.tx_host.get(), int(self.tx_port.get()), 2.0)
                _cl.connect()
                _cl.cmd(":FREQ:CW %.9g HZ" % (float(self.tx_f.get().replace(",", ".")) * 1e6))
                _cl.cmd(":POW:AMPL %.6g DBM" % float(self.tx_p.get().replace(",", ".")))
                self.tx_status.set("TX: CW %.6g МГц @ %.6g дБм виставлено"
                                   % (float(self.tx_f.get().replace(",", ".")),
                                      float(self.tx_p.get().replace(",", "."))))
                hist_log("SCPI TX", "CW %.6g МГц @ %.6g дБм"
                         % (float(self.tx_f.get().replace(",", ".")),
                            float(self.tx_p.get().replace(",", "."))))
            except Exception as _e6t:
                self.tx_status.set("Помилка TX: %s" % _e6t)

        def _tx_outp(self, on):
            try:
                _cl = ScpiClient(self.tx_host.get(), int(self.tx_port.get()), 2.0)
                _cl.connect()
                _cl.cmd(":OUTP %s" % ("ON" if on else "OFF"))
                self.tx_status.set("TX: вихід %s" % ("увімкнено" if on else "вимкнено"))
                hist_log("SCPI TX", "OUTP %s" % ("ON" if on else "OFF"))
            except Exception as _e6t:
                self.tx_status.set("Помилка TX: %s" % _e6t)

        def _tx_auto_start(self):
            try:
                self._tx_auto_per = max(1.0, float(self.tx_per.get().replace(",", ".")))
            except ValueError:
                self._tx_auto_per = 10.0
            import threading as _th6
            if getattr(self, "_tx_auto_stop_ev", None) is None:
                self._tx_auto_stop_ev = _th6.Event()
            self._tx_auto_stop_ev.clear()
            self._tx_auto_n = 0
            self._tx_auto_csv_path = getattr(self, "_tx_auto_csv_path", None)
            def _loop6():
                while not self._tx_auto_stop_ev.is_set():
                    try:
                        if getattr(self, "_sa_client", None) is not None and self._sa_client.sock is not None:
                            self._sa_fetch()
                            _n = len((getattr(self, "sa_last", None) or {}).get("peaks", []))
                            self._tx_auto_n = getattr(self, "_tx_auto_n", 0) + 1
                            hist_log("Авто-скан", "зйомка %d: піків %d" % (self._tx_auto_n, _n))
                            if self._tx_auto_csv_path:
                                _tr = (getattr(self, "sa_last", None) or {})
                                with open(self._tx_auto_csv_path, "a", encoding="utf-8-sig", newline="") as _fh:
                                    _w = _csv6.writer(_fh, delimiter=";")
                                    for _i, _v in enumerate(_tr.get("trace_dbm", ())):
                                        _fr = _tr.get("freqs_hz", ())
                                        _w.writerow([_time6.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "%.9g" % (_fr[_i] / 1e6 if _i < len(_fr) else 0),
                                                     "%.2f" % float(_v)])
                            try:
                                self.tx_auto_lbl.config(text="зйомок: %d" % self._tx_auto_n)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self._tx_auto_stop_ev.wait(self._tx_auto_per)
            _th6.Thread(target=_loop6, daemon=True).start()
            self._set_status("авто-скан: період %.6g с" % self._tx_auto_per)

        def _tx_auto_stop(self):
            if getattr(self, "_tx_auto_stop_ev", None) is not None:
                self._tx_auto_stop_ev.set()
            self._set_status("авто-скан: зупинено")

        def _tx_auto_csv(self):
            from tkinter import filedialog as _fd6
            _p = _fd6.asksaveasfilename(defaultextension=".csv",
                                        initialfile="sweep_log.csv",
                                        filetypes=[("CSV", "*.csv")])
            if _p:
                self._tx_auto_csv_path = _p
                self.tx_auto_lbl.config(text="журнал: %s" % os.path.basename(_p))

        def _tx_floor_apply(self):
            _tr = getattr(self, "sa_last", None)
            if not _tr or "trace_dbm" not in _tr:
                self._set_status("підклад: спершу зніміть трасу у SCPI-зйомці")
                return
            _fr = np.asarray(_tr.get("freqs_hz", []))
            _tv = np.asarray(_tr["trace_dbm"])
            try:
                _ref = float(self.sa_ref.get().replace(",", ".") or 0.0)
            except ValueError:
                _ref = 0.0
            try:
                _sp = float(self.sa_span.get().replace(",", ".") or 0.0) * 1e6
            except ValueError:
                _sp = 0.0
            _n = 0
            for _en, _kd, _vals in self.gen_rows:
                if not bool(_en.get()) or _kd.get() != "Шум":
                    continue
                try:
                    _f = float(_vals[0].get().replace(",", ".")) * 1e6
                except ValueError:
                    continue
                _mask = (np.abs(_fr - _f) <= max(_sp / 2.0, 1.0))
                if not _mask.any():
                    continue
                _dbfs = float(np.max(_tv[_mask])) - _ref
                _vals[2].set("%.6g" % max(0.0, _dbfs))
                _n += 1
            self._set_status("підклад: амплітуду оновлено в %d шумових каналах (dBFS = пік − ref)" % _n)
            hist_log("SCPI → Генератор", "підклад у %d каналах" % _n)

        def _rec_save_park(self, path=None):
            from tkinter import filedialog as _fd6
            import json as _js6
            if path is None:
                path = _fd6.asksaveasfilename(defaultextension=".json",
                                              initialfile="sdr_park.json",
                                              filetypes=[("JSON", "*.json")])
            if not path:
                return
            with open(path, "w", encoding="utf-8") as _fh:
                _js6.dump({n: bool(v.get()) for n, v in self.rec_vars.items()},
                          _fh, ensure_ascii=False, indent=1)
            self._set_status("парк збережено: %s" % path)
            hist_log("Рецепти", "парк збережено (%d платформ)" % len(self.rec_vars))

        def _rec_load_park(self, path=None):
            from tkinter import filedialog as _fd6
            import json as _js6
            if path is None:
                path = _fd6.askopenfilename(filetypes=[("JSON", "*.json"), ("Усі файли", "*.*")])
            if not path:
                return
            try:
                with open(path, "r", encoding="utf-8") as _fh:
                    _d = _js6.load(_fh)
                for n, v in self.rec_vars.items():
                    v.set(bool(_d.get(n, False)))
                self.rec_status.set("парк завантажено: %s (%d платформ)"
                                    % (os.path.basename(path), len(_d)))
                hist_log("Рецепти", "парк завантажено: %s" % os.path.basename(path))
            except Exception as _e6r:
                self.rec_status.set("Помилка: %s" % _e6r)

        def _rec_pdf(self, path=None):
            if not getattr(self, "rec_rows", None):
                self._set_status("Рецепти не підібрані")
                return
            from tkinter import filedialog as _fd6
            if path is None:
                path = _fd6.asksaveasfilename(defaultextension=".pdf",
                                              initialfile="sdr_recipes.pdf",
                                              filetypes=[("PDF", "*.pdf")])
            if not path:
                return
            try:
                build_pdf_report(path, [(_r["model"], "%s · %s" % (_r["verdict"], _r["note"]))
                                        for _r in self.rec_rows])
                self._set_status("PDF-звіт: %s" % path)
                hist_log("Рецепти", "PDF-звіт (%d рядків)" % len(self.rec_rows))
            except Exception:
                try:
                    self._rec_pdf_fallback(path, [(_r["model"], "%s · %s" % (_r["verdict"], _r["note"]))
                                                 for _r in self.rec_rows])
                    self._set_status("PDF-звіт (fallback): %s" % path)
                    hist_log("Рецепти", "PDF-звіт fallback (%d рядків)" % len(self.rec_rows))
                except Exception as _e6p:
                    self._set_status("Помилка PDF: %s" % _e6p)

        def _sig_open(self):
            """Діалог «Розпізнати сигнатури піків» (v6.0.0)."""
            _pks = self._melt_peaks_now()
            if isinstance(_pks, dict):
                _pks = _pks.get("peaks") or []
            if not _pks:
                messagebox.showinfo(APP_TITLE, "Немає піків — спершу проаналізуйте запис або зніміть трасу.")
                return
            win, top, _lbl = self._win("🧠 Розпізнавання сигнатур піків — ширина −3 дБ → тип сигналу", 960, 520)
            self.sig_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            _cols = ("f", "db", "w", "sig")
            self.sig_tree = ttk.Treeview(f, columns=_cols, show="headings", height=12)
            for _cid, (_t, _w) in zip(_cols, [("Пік, МГц", 100), ("Рівень, дБм", 95),
                                              ("Ширина −3 дБ, МГц", 135), ("Сигнатура → Крок 3", 320)]):
                self.sig_tree.heading(_cid, text=_t)
                self.sig_tree.column(_cid, width=_w, anchor="w" if _cid == "sig" else "center")
            self._sig_keys = {}
            for _i, _p in enumerate(_pks[:12]):
                _w = float(_p.get("width_hz") or (float(_p.get("width_mhz", 0)) * 1e6) or 0.0)
                _f = float(_p.get("f_hz") or (float(_p.get("f_mhz", 0)) * 1e6) or 0.0)
                _db = float(_p.get("peak_db") or _p.get("db") or 0.0)
                _n, _sk = classify_width(_w)
                _txt = "%s%s" % (_n, (" → %s" % _sk) if _sk else "")
                self.sig_tree.insert("", "end", values=(
                    "%.6g" % (_f / 1e6), "%.1f" % _db, "%.6g" % (_w / 1e6), _txt))
                self._sig_keys[_i] = _sk
            self.sig_tree.pack(fill="both", expand=True)
            brow = ttk.Frame(f)
            brow.pack(fill="x", pady=4)
            ttk.Button(brow, text="→ Тип сигналу у Крок 3 (перший увімкнений канал)",
                       command=self._sig_apply).pack(side="left", padx=4)

        def _sig_apply(self):
            _sel = self.sig_tree.selection()
            _sk = None
            if _sel:
                _sk = self._sig_keys.get(self.sig_tree.index(_sel[0]))
            if _sk is None:
                for _v in self._sig_keys.values():
                    if _v:
                        _sk = _v
                        break
            if not _sk:
                messagebox.showinfo(APP_TITLE, "Сигнатура не розпізнана.")
                return
            for _en, _kd, _vals in self.gen_rows:
                if bool(_en.get()):
                    _kd.set(_sk)
                    self._set_status("Крок 3: тип каналу → %s (v6.0.0)" % _sk)
                    hist_log("Сигнатури", "Крок 3 → %s" % _sk)
                    return
            messagebox.showinfo(APP_TITLE, "Немає увімкнених каналів у Кроці 3.")

        def _theme_toggle(self):
            _cur = theme_pref_get() or "dark"
            _new = "light" if _cur == "dark" else "dark"
            theme_pref_set(_new)
            messagebox.showinfo(APP_TITLE, "Тему змінено на %s — перезапустіть програму, щоб застосувати." % _new)
            hist_log("Інтерфейс", "тема → %s" % _new)

        def _rec_pdf_fallback(self, path, pairs):
            """Мінімальний PDF-звіт рецептів (v6.0.0) — reportlab, CJK-шрифт."""
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
            _f = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            pdfmetrics.registerFont(TTFont("CJK6", _f, subfontIndex=0))
            _doc = SimpleDocTemplate(path, pagesize=A4, title="Рецепти SDR — RAW Studio v6.0.0")
            _data = [["Платформа", "Вердикт · примітка"]]
            _data += [[str(a), str(b)] for a, b in pairs]
            _tb = Table(_data, colWidths=[180, 330])
            _tb.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "CJK6"),
                                     ("FONTSIZE", (0, 0), (-1, -1), 8),
                                     ("BACKGROUND", (0, 0), (-1, 0), "#12365e"),
                                     ("TEXTCOLOR", (0, 0), (-1, 0), "#ffffff"),
                                     ("GRID", (0, 0), (-1, -1), 0.5, "#c3ccd6"),
                                     ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            _doc.build([_tb])

        def _jp_open(self):
            """Діалог «⏱ Профіль завади за часом» (v6.1.0): сцени, покриття, вікна тиші, реактивна модель."""
            win, top, _lbl = self._win("⏱ Профіль завади за часом — покриття, вікна тиші, реактивна модель", 960, 560)
            self.jp_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Тривалість, с:").pack(side="left")
            self.jp_total = tk.StringVar(value="60")
            ttk.Entry(row0, textvariable=self.jp_total, width=6).pack(side="left", padx=4)
            ttk.Label(row0, text="Реактивний поріг, дБ:").pack(side="left", padx=(10, 0))
            self.jp_thr = tk.StringVar(value="20")
            ttk.Entry(row0, textvariable=self.jp_thr, width=6).pack(side="left", padx=4)
            ttk.Label(row0, text="Реакція, мс:").pack(side="left", padx=(10, 0))
            self.jp_react = tk.StringVar(value="5")
            ttk.Entry(row0, textvariable=self.jp_react, width=6).pack(side="left", padx=4)
            row1 = ttk.Frame(f)
            row1.pack(fill="x", pady=(0, 4))
            self.jp_t0 = tk.StringVar(value="0")
            self.jp_t1 = tk.StringVar(value="30")
            self.jp_duty = tk.StringVar(value="0.4")
            for _nm, _v, _w in (("t0:", self.jp_t0, 5), ("t1:", self.jp_t1, 5), ("скважність:", self.jp_duty, 5)):
                ttk.Label(row1, text=_nm).pack(side="left", padx=(8, 0))
                ttk.Entry(row1, textvariable=_v, width=_w).pack(side="left", padx=2)
            ttk.Button(row1, text="＋ Додати сцену", command=self._jp_add).pack(side="left", padx=10)
            ttk.Button(row1, text="－ Видалити", command=self._jp_del).pack(side="left")
            ttk.Button(row1, text="▶ Порахувати", command=self._jp_run).pack(side="left", padx=10)
            self.jp_status = tk.StringVar(value="сцени (t0, t1, скважність) → покриття і вікна тиші")
            ttk.Label(f, textvariable=self.jp_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("t0", "t1", "duty")
            self.jp_tree = ttk.Treeview(f, columns=_cols, show="headings", height=8)
            for _cid, _t, _w in (("t0", "t0, с", 90), ("t1", "t1, с", 90), ("duty", "скважність", 100)):
                self.jp_tree.heading(_cid, text=_t)
                self.jp_tree.column(_cid, width=_w, anchor="center")
            self.jp_tree.pack(fill="both", expand=True)
            self.jp_tree.insert("", "end", values=(0, 30, 0.4))
            self.jp_tree.insert("", "end", values=(30, 60, 0.1))

        def _jp_rows(self):
            rows = []
            for _iid in self.jp_tree.get_children():
                v = self.jp_tree.item(_iid, "values")
                rows.append((float(v[0]), float(v[1]), float(v[2])))
            return rows

        def _jp_add(self):
            self.jp_tree.insert("", "end", values=(self.jp_t0.get(), self.jp_t1.get(), self.jp_duty.get()))

        def _jp_del(self):
            for _iid in self.jp_tree.selection():
                self.jp_tree.delete(_iid)

        def _jp_run(self):
            try:
                _p = jam_time_profile(self._jp_rows(), float(self.jp_total.get()))
                _g = ", ".join("%d-%d с" % (g["t0"], g["t1"]) for g in _p["gaps"][:6]) or "немає"
                _pw = np.array(_p["cover"]) * 40.0
                _r = reactive_jam_model(_pw, 1.0, float(self.jp_thr.get()), float(self.jp_react.get()))
                self.jp_status.set(
                    "покриття %.0f%% · вікна тиші: %s · реактивна: виявлення %s с, опромінення %.0f%%, економія %.0f%%"
                    % (_p["covered_frac"] * 100, _g,
                       _r["detect_s"] if _r["detect_s"] is not None else "—",
                       _r["on_air_frac"] * 100, _r["save_frac"] * 100))
                hist_log("Профіль завади", "покриття %.0f%%" % (_p["covered_frac"] * 100), None)
            except Exception as _e:
                self.jp_status.set("Помилка: %s" % _e)

        def _pc_open(self):
            """Діалог «📊 PER-криві vs J/S» (v6.1.0): BER/PER для модемів melt-лабораторії."""
            win, top, _lbl = self._win("📊 PER-криві — BER/PER залежно від J/S", 900, 560)
            self.pc_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Модем:").pack(side="left")
            self.pc_mod = tk.StringVar(value="BPSK")
            ttk.Combobox(row0, textvariable=self.pc_mod, values=list(MODS_MELT), width=10,
                         state="readonly").pack(side="left", padx=4)
            ttk.Label(row0, text="Пакет, біт:").pack(side="left", padx=(10, 0))
            self.pc_bits = tk.StringVar(value="512")
            ttk.Entry(row0, textvariable=self.pc_bits, width=7).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Криві", command=self._pc_run).pack(side="left", padx=10)
            self.pc_status = tk.StringVar(value="J/S 0…33 дБ, 12 точок — BER і PER пакета")
            ttk.Label(f, textvariable=self.pc_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("js", "ber", "per")
            self.pc_tree = ttk.Treeview(f, columns=_cols, show="headings", height=14)
            for _cid, _t, _w in (("js", "J/S, дБ", 100), ("ber", "BER", 140), ("per", "PER", 140)):
                self.pc_tree.heading(_cid, text=_t)
                self.pc_tree.column(_cid, width=_w, anchor="center")
            self.pc_tree.pack(fill="both", expand=True)

        def _pc_run(self):
            try:
                _r = per_vs_js(self.pc_mod.get(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
                               int(self.pc_bits.get()))
                for _iid in self.pc_tree.get_children():
                    self.pc_tree.delete(_iid)
                for _p in _r:
                    self.pc_tree.insert("", "end", values=(_p["js_db"], "%.2e" % _p["ber"], "%.3f" % _p["per"]))
                _per50 = next((p["js_db"] for p in _r if p["per"] >= 0.5), None)
                self.pc_status.set("PER >= 50%% при J/S %s дБ (модем %s, пакет %s біт)"
                                   % (_per50, self.pc_mod.get(), self.pc_bits.get()))
                hist_log("PER-криві", "%s: PER>=50%% при %s дБ" % (self.pc_mod.get(), _per50), None)
            except Exception as _e:
                self.pc_status.set("Помилка: %s" % _e)

        def _mb_open(self):
            """Діалог «📻 Багаточастотний план» (v6.1.0): смуги по каналах приладу."""
            win, top, _lbl = self._win("📻 Багаточастотний план — смуги по каналах приладу", 880, 560)
            self.mb_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Каналів TX:").pack(side="left")
            self.mb_nch = tk.StringVar(value="4")
            ttk.Entry(row0, textvariable=self.mb_nch, width=4).pack(side="left", padx=4)
            ttk.Label(row0, text="Смуга:").pack(side="left", padx=(10, 0))
            self.mb_band = tk.StringVar(value="Відео 5.8")
            ttk.Entry(row0, textvariable=self.mb_band, width=14).pack(side="left", padx=4)
            ttk.Label(row0, text="f_lo/f_hi, МГц:").pack(side="left", padx=(10, 0))
            self.mb_flo = tk.StringVar(value="5650")
            self.mb_fhi = tk.StringVar(value="5950")
            ttk.Entry(row0, textvariable=self.mb_flo, width=7).pack(side="left", padx=2)
            ttk.Entry(row0, textvariable=self.mb_fhi, width=7).pack(side="left", padx=2)
            ttk.Label(row0, text="Пріоритет 1-3:").pack(side="left", padx=(10, 0))
            self.mb_prio = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.mb_prio, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="＋ Додати", command=self._mb_add).pack(side="left", padx=8)
            ttk.Button(row0, text="－ Видалити", command=self._mb_del).pack(side="left")
            ttk.Button(row0, text="▶ План", command=self._mb_run).pack(side="left", padx=10)
            self.mb_status = tk.StringVar(value="смуги → канали приладу за пріоритетом")
            ttk.Label(f, textvariable=self.mb_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("ch", "band", "flo", "fhi", "fmid", "prio")
            self.mb_tree = ttk.Treeview(f, columns=_cols, show="headings", height=10)
            for _cid, _t, _w in (("ch", "Канал", 60), ("band", "Смуга", 200), ("flo", "f_lo, МГц", 90),
                                 ("fhi", "f_hi, МГц", 90), ("fmid", "середина", 90), ("prio", "Пріо", 50)):
                self.mb_tree.heading(_cid, text=_t)
                self.mb_tree.column(_cid, width=_w, anchor="center")
            self.mb_tree.pack(fill="both", expand=True)
            for _r in (("Відео 5.8", 5650, 5950, 1), ("Керування 2.4", 2400, 2483, 2),
                       ("GNSS L1", 1559, 1610, 2)):
                self.mb_tree.insert("", "end", values=("", _r[0], _r[1], _r[2], "", _r[3]))

        def _mb_add(self):
            self.mb_tree.insert("", "end", values=("", self.mb_band.get(), self.mb_flo.get(),
                                                   self.mb_fhi.get(), "", self.mb_prio.get()))

        def _mb_del(self):
            for _iid in self.mb_tree.selection():
                self.mb_tree.delete(_iid)

        def _mb_run(self):
            try:
                _tg = []
                for _iid in self.mb_tree.get_children():
                    _v = self.mb_tree.item(_iid, "values")
                    _tg.append((str(_v[1]), float(_v[2]) * 1e6, float(_v[3]) * 1e6, int(_v[5])))
                _p = multiband_plan(_tg, int(self.mb_nch.get()))
                for _iid in self.mb_tree.get_children():
                    self.mb_tree.delete(_iid)
                for _a in _p["alloc"]:
                    self.mb_tree.insert("", "end", values=("TX%d" % _a["channel"], _a["band"],
                                                           "%.1f" % (_a["f_lo"] / 1e6),
                                                           "%.1f" % (_a["f_hi"] / 1e6),
                                                           "%.1f" % (_a["f_mid"] / 1e6), _a["prio"]))
                self.mb_status.set("каналів: %d з %s · перевантаження: %s"
                                   % (_p["channels_used"], self.mb_nch.get(),
                                      "так" if _p["overloaded"] else "ні"))
                hist_log("Багатосмуга", "%d смуг, %d каналів" % (len(_p["alloc"]), _p["channels_used"]), None)
            except Exception as _e:
                self.mb_status.set("Помилка: %s" % _e)

        def _js_open(self):
            """Діалог «📈 J/S-калькулятор» (v6.1.0): крива J/S(дистанція) і радіус придушення."""
            win, top, _lbl = self._win("📈 J/S-калькулятор — крива J/S(дистанція), радіус придушення", 900, 580)
            self.js_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            self.js_pj = tk.StringVar(value="30")
            self.js_pt = tk.StringVar(value="10")
            self.js_dt = tk.StringVar(value="1000")
            self.js_dmax = tk.StringVar(value="3000")
            self.js_g = tk.StringVar(value="0")
            self.js_thr = tk.StringVar(value="-6")
            for _nm, _v in (("P завади, дБм:", self.js_pj), ("P цілі, дБм:", self.js_pt),
                            ("d ціль-пульт, м:", self.js_dt), ("d макс, м:", self.js_dmax),
                            ("ΔG антен, дБ:", self.js_g), ("Поріг J/S, дБ:", self.js_thr)):
                ttk.Label(row0, text=_nm).pack(side="left")
                ttk.Entry(row0, textvariable=_v, width=6).pack(side="left", padx=(2, 8))
            ttk.Button(row0, text="▶ Крива", command=self._js_run).pack(side="left")
            self.js_status = tk.StringVar(value="J/S = Pj − Pt + ΔG + 20·lg(d_цілі/d_завади) — вільний простір")
            ttk.Label(f, textvariable=self.js_status, foreground=THEME["dim"]).pack(anchor="w")
            self.js_cv = tk.Canvas(f, height=360, bg=THEME["bg"], highlightthickness=0)
            self.js_cv.pack(fill="both", expand=True, pady=(4, 0))

        def _js_run(self):
            try:
                _r = js_curve_db(float(self.js_pj.get()), float(self.js_pt.get()), float(self.js_dt.get()),
                                 float(self.js_dmax.get()), float(self.js_g.get()), float(self.js_thr.get()))
                _cv = self.js_cv
                _cv.delete("all")
                _w = _cv.winfo_width() or 800
                _h = 360
                _jsmin, _jsmax = min(_r["js_db"]), max(_r["js_db"])
                _rng = max(_jsmax - _jsmin, 1.0)
                _cv.create_text(10, 14, anchor="w", fill=THEME["text"],
                                text="J/S: %s … %s дБ · радіус придушення %.0f м"
                                     % (_jsmax, _jsmin, _r["d_max_m"]))
                _pts = []
                for _i, (_d, _v) in enumerate(zip(_r["d_m"], _r["js_db"])):
                    _x = 30 + _i * (_w - 60) / max(len(_r["d_m"]) - 1, 1)
                    _y = 40 + (_jsmax - _v) / _rng * (_h - 80)
                    _pts += [_x, _y]
                if len(_pts) > 3:
                    _cv.create_line(_pts, fill="#d95f02", width=2)
                _yt = 40 + (_jsmax - float(self.js_thr.get())) / _rng * (_h - 80)
                _yt = max(40, min(_h - 40, _yt))
                _cv.create_line(30, _yt, _w - 30, _yt, dash=(4, 3), fill="#1b9e77")
                _cv.create_text(_w - 34, _yt - 8, anchor="e", fill="#1b9e77", text="поріг")
                hist_log("J/S-калькулятор", "радіус %.0f м" % _r["d_max_m"], None)
            except Exception as _e:
                self.js_status.set("Помилка: %s" % _e)

        def _aj_open(self):
            """Діалог «🛡 Стійкість лінка до завад» (v6.1.0): ознаки з запису → тактика."""
            win, top, _lbl = self._win("🛡 Стійкість лінка — ознаки стійкості до завад із запису", 900, 520)
            self.aj_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.aj_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.aj_path, width=42).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left", padx=(8, 0))
            self.aj_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.aj_fs, width=6).pack(side="left", padx=4)
            ttk.Label(row0, text="Канал:").pack(side="left", padx=(8, 0))
            self.aj_ch = tk.StringVar(value="0")
            ttk.Entry(row0, textvariable=self.aj_ch, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Аналіз", command=self._aj_run).pack(side="left", padx=10)
            self.aj_status = tk.StringVar(value="запис → пакети → стрибки частоти / паузи / скважність → тактика")
            ttk.Label(f, textvariable=self.aj_status, foreground=THEME["dim"]).pack(anchor="w")
            self.aj_txt = tk.Text(f, height=16, wrap="word", bg=THEME["bg"], fg=THEME["text"], relief="flat")
            self.aj_txt.pack(fill="both", expand=True, pady=(4, 0))

        def _aj_run(self):
            try:
                _fs = float(self.aj_fs.get()) * 1e6
                _nch = int(self.aj_ch.get()) + 1
                _r = burst_scan(self.aj_path.get(), _fs, nch=_nch, ch=int(self.aj_ch.get()))
                _centers = burst_centers_hz(self.aj_path.get(), _fs, _r, nch=_nch,
                                            ch=int(self.aj_ch.get()))
                _f8 = anti_jam_features(_centers, _r["duty"])
                _pp = (max(_r["peak_db"]) - _r["floor_db"]) if _r["peak_db"] else 20.0
                _tc = jam_tactics_compare(_pp)
                _t = self.aj_txt
                _t.delete("1.0", "end")
                _t.insert("end", "Пакетів: %d · скважність %.3f · період %s с\n"
                          % (_r["bursts"], _r["duty"], _r["period_s"]))
                _t.insert("end", "Каналів частот: %d · стрибки: %s · паузи LBT: %s\n\n"
                          % (_f8["каналів"], "так" if _f8["стрибки"] else "ні",
                             "так" if _f8["lbt"] else "ні"))
                _t.insert("end", "Порада:\n  %s\n\n" % _f8["порада"].replace(" · ", "\n  "))
                _t.insert("end", "Тактики (за ПП %.0f дБ):\n" % _pp)
                for _nm, _sc in _tc["рейтинг"]:
                    _t.insert("end", "  %s — %s\n" % ("★" * _sc, _nm))
                _t.insert("end", "Рекомендовано: %s (%s)" % (_tc["рекомендовано"], _tc["правило"]))
                hist_log("Стійкість лінка", "пакетів %d, каналів %d" % (_r["bursts"], _f8["каналів"]), None)
            except Exception as _e:
                self.aj_status.set("Помилка: %s" % _e)

        def _rc_open(self):
            """Діалог «🎮 RC-декодер» (v6.1.0): пакети запису + фрейми/канали/телеметрія/failsafe з hex."""
            win, top, _lbl = self._win("🎮 RC-декодер — пакети запису, фрейми, канали, телеметрія", 940, 600)
            self.rcd_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.rcd_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.rcd_path, width=34).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.rcd_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.rcd_fs, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Канал:").pack(side="left")
            self.rcd_ch = tk.StringVar(value="0")
            ttk.Entry(row0, textvariable=self.rcd_ch, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Пакети", command=self._rcd_scan).pack(side="left", padx=8)
            self.rcd_status = tk.StringVar(value="пакети з запису; фрейми — hex нижче")
            ttk.Label(f, textvariable=self.rcd_status, foreground=THEME["dim"]).pack(anchor="w")
            row1 = ttk.Frame(f)
            row1.pack(fill="x", pady=(4, 0))
            ttk.Label(row1, text="Фрейми (hex):").pack(side="left")
            self.rcd_hex = tk.StringVar(value="")
            ttk.Entry(row1, textvariable=self.rcd_hex, width=44).pack(side="left", padx=4)
            self.rcd_proto = tk.StringVar(value="SBUS")
            ttk.Combobox(row1, textvariable=self.rcd_proto, values=("SBUS", "IBUS", "CRSF", "MSP"),
                         width=6, state="readonly").pack(side="left", padx=4)
            ttk.Button(row1, text="▶ Фрейми", command=self._rcd_frames).pack(side="left", padx=8)
            _cols = ("n", "val", "hexs")
            self.rcd_tree = ttk.Treeview(f, columns=_cols, show="headings", height=12)
            for _cid, _t, _w in (("n", "№", 50), ("val", "Значення/поле", 380), ("hexs", "hex", 320)):
                self.rcd_tree.heading(_cid, text=_t)
                self.rcd_tree.column(_cid, width=_w, anchor="w")
            self.rcd_tree.pack(fill="both", expand=True, pady=(4, 0))

        def _rcd_scan(self):
            try:
                _r = burst_scan(self.rcd_path.get(), float(self.rcd_fs.get()) * 1e6,
                                nch=int(self.rcd_ch.get()) + 1, ch=int(self.rcd_ch.get()))
                _t = self.rcd_tree
                for _iid in _t.get_children():
                    _t.delete(_iid)
                for _i, (_t0, _pk) in enumerate(zip(_r["t0_s"][:40], _r["peak_db"][:40])):
                    _t.insert("", "end", values=(_i + 1, "пакет @ %.4f с, пік %.1f дБ" % (_t0, _pk), ""))
                self.rcd_status.set("пакетів: %d · період %s с · скважність %.3f"
                                    % (_r["bursts"], _r["period_s"], _r["duty"]))
                hist_log("RC-декодер", "пакетів %d" % _r["bursts"], None)
            except Exception as _e:
                self.rcd_status.set("Помилка: %s" % _e)

        def _rcd_frames(self):
            try:
                _hex = "".join(self.rcd_hex.get().split()).replace("0x", "").replace("0X", "")
                _data = bytes.fromhex(_hex)
                _pr = self.rcd_proto.get()
                _t = self.rcd_tree
                for _iid in _t.get_children():
                    _t.delete(_iid)
                if _pr == "MSP":
                    _fr = []
                    _i = 0
                    while _i < len(_data) - 5:
                        _fm = rc_telemetry_msp(_data[_i:])
                        if _fm is not None:
                            _fr.append(_fm)
                            _i += 6 + _fm["довжина"]
                        else:
                            _i += 1
                    for _i, _fm in enumerate(_fr):
                        _t.insert("", "end", values=(_i + 1, "MSP cmd %s, %s байт: %s"
                                                     % (_fm["cmd"], _fm["довжина"], _fm["payload_hex"]), ""))
                    self.rcd_status.set("MSP фреймів: %d" % len(_fr))
                    return
                _fr = rc_extract_frames(_data, _pr)
                _cht = []
                for _i, (_off, _fb) in enumerate(_fr[:60]):
                    _chv = rc_channels(_pr, _fb)
                    if _chv:
                        _cht.append(_chv[:4])
                    _t.insert("", "end", values=(_i + 1,
                                                 "канали: %s" % ", ".join(str(x) for x in _chv[:8]),
                                                 _fb.hex()))
                _tel = None
                for _off, _fb in _fr:
                    _tt = rc_telemetry_crsf(_fb)
                    if _tt and _tt.get("тип") in ("батарея", "GPS"):
                        _tel = _tt
                        break
                _ex = "; ".join("%s=%s" % (k, v) for k, v in (_tel or {}).items())
                self.rcd_status.set("фреймів: %d (%s)%s"
                                    % (len(_fr), _pr, (" · телеметрія: " + _ex) if _tel else ""))
                if len(_cht) >= 5:
                    _fa = failsafe_analyze(_cht, 0.014)
                    self.rcd_status.set(self.rcd_status.get()
                                        + " · failsafe: %s (заморожено %s відліків)"
                                        % ("так" if _fa["замороження"] else "ні", _fa["заморожених_відліків"]))
                hist_log("RC-декодер", "фреймів %d (%s)" % (len(_fr), _pr), None)
            except Exception as _e:
                self.rcd_status.set("Помилка: %s" % _e)

        def _hp_open(self):
            """Діалог «🌀 Хоп-послідовність» (v6.1.0): патерн стрибків частоти із запису."""
            win, top, _lbl = self._win("🌀 Хоп-послідовність — канали, цикл, прогноз", 880, 520)
            self.hp_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.hp_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.hp_path, width=40).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.hp_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.hp_fs, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Канал:").pack(side="left")
            self.hp_ch = tk.StringVar(value="0")
            ttk.Entry(row0, textvariable=self.hp_ch, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Патерн", command=self._hp_run).pack(side="left", padx=10)
            self.hp_status = tk.StringVar(value="пакети → центри частот → цикл патерну і прогноз")
            ttk.Label(f, textvariable=self.hp_status, foreground=THEME["dim"]).pack(anchor="w")
            self.hp_txt = tk.Text(f, height=16, wrap="word", bg=THEME["bg"], fg=THEME["text"], relief="flat")
            self.hp_txt.pack(fill="both", expand=True, pady=(4, 0))

        def _hp_run(self):
            try:
                _fs = float(self.hp_fs.get()) * 1e6
                _nch = int(self.hp_ch.get()) + 1
                _r = burst_scan(self.hp_path.get(), _fs, nch=_nch, ch=int(self.hp_ch.get()))
                _centers = burst_centers_hz(self.hp_path.get(), _fs, _r, nch=_nch,
                                            ch=int(self.hp_ch.get()), max_bursts=64)
                _h = hop_pattern_detect(_centers)
                _t = self.hp_txt
                _t.delete("1.0", "end")
                _t.insert("end", "Пакетів: %d · каналів частот: %d · стрибків: %d\n"
                          % (_r["bursts"], _h["каналів"], _h["стрибків"]))
                _t.insert("end", "Канали, МГц: %s\n"
                          % ", ".join("%.3f" % (x / 1e6) for x in _h["канали_Гц"][:12]))
                _t.insert("end", "Цикл патерну: %s · середній двелл: %s пакетів\n"
                          % (_h["цикл"] if _h["цикл"] else "не детектовано", _h["середній_двелл"]))
                _t.insert("end", "Прогноз наступних, МГц: %s\n"
                          % (", ".join("%.3f" % (x / 1e6) for x in _h["прогноз"]) or "—"))
                hist_log("Хоп-послідовність", "каналів %d, цикл %s" % (_h["каналів"], _h["цикл"]), None)
            except Exception as _e:
                self.hp_status.set("Помилка: %s" % _e)

        def _lq_open(self):
            """Діалог «📶 Якість лінка» (v6.1.0): слоти, LQ %, втрати, RSSI."""
            win, top, _lbl = self._win("📶 Якість лінка — LQ %, втрати пакетів, RSSI", 880, 540)
            self.lq_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.lq_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.lq_path, width=36).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.lq_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.lq_fs, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Період, мс:").pack(side="left", padx=(8, 0))
            self.lq_per = tk.StringVar(value="10")
            ttk.Entry(row0, textvariable=self.lq_per, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Канал:").pack(side="left", padx=(8, 0))
            self.lq_ch = tk.StringVar(value="0")
            ttk.Entry(row0, textvariable=self.lq_ch, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ LQ", command=self._lq_run).pack(side="left", padx=10)
            self.lq_status = tk.StringVar(value="очікувані слоти vs виявлені пакети")
            ttk.Label(f, textvariable=self.lq_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("slot", "rssi", "st")
            self.lq_tree = ttk.Treeview(f, columns=_cols, show="headings", height=14)
            for _cid, _t, _w in (("slot", "Слот, с", 120), ("rssi", "RSSI, дБ", 120), ("st", "Статус", 100)):
                self.lq_tree.heading(_cid, text=_t)
                self.lq_tree.column(_cid, width=_w, anchor="center")
            self.lq_tree.pack(fill="both", expand=True)

        def _lq_run(self):
            try:
                _fs = float(self.lq_fs.get()) * 1e6
                _r = burst_scan(self.lq_path.get(), _fs, nch=int(self.lq_ch.get()) + 1,
                                ch=int(self.lq_ch.get()))
                _per = float(self.lq_per.get()) / 1000.0
                _tot = max(_r["t0_s"][-1] + 0.01, _per) if _r["t0_s"] else _per
                _q = link_quality_timeline(_r["t0_s"], _r["peak_db"], _per, _tot)
                _t = self.lq_tree
                for _iid in _t.get_children():
                    _t.delete(_iid)
                for _s, _rss in zip([i * _per for i in range(_q["очікувано"])], _q["rssi_дБ"]):
                    _t.insert("", "end", values=("%.3f" % _s, _rss if _rss is not None else "—",
                                                 "прийнято" if _rss is not None else "ВТРАТА"))
                self.lq_status.set("очікувано %d · прийнято %d · LQ %.1f%%"
                                   % (_q["очікувано"], _q["прийнято"], _q["LQ_%"]))
                hist_log("Якість лінка", "LQ %.1f%%" % _q["LQ_%"], None)
            except Exception as _e:
                self.lq_status.set("Помилка: %s" % _e)

        def _df_open(self):
            """Діалог «🧭 Пеленгатор» (v6.1.0): азимут з потужностей 4 каналів (0/90/180/270°)."""
            win, top, _lbl = self._win("🧭 Пеленгатор — азимут джерела з багатоканального запису", 880, 540)
            self.df_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис (4+ канали):").pack(side="left")
            self.df_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.df_path, width=34).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.df_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.df_fs, width=5).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Пеленг", command=self._df_run).pack(side="left", padx=10)
            self.df_status = tk.StringVar(value="антени 0/90/180/270° → векторна сума")
            ttk.Label(f, textvariable=self.df_status, foreground=THEME["dim"]).pack(anchor="w")
            self.df_cv = tk.Canvas(f, height=340, bg=THEME["bg"], highlightthickness=0)
            self.df_cv.pack(fill="both", expand=True, pady=(4, 0))

        def _df_run(self):
            try:
                _fs = float(self.df_fs.get()) * 1e6
                _chs, _n = load_raw(self.df_path.get(), nch=4)
                _pw = []
                for _k in range(4):
                    _seg = _chs[_k][:max(512, int(0.05 * _fs))]
                    _seg = _seg - float(_seg.mean())
                    _rms = float(np.sqrt(np.mean(_seg ** 2)))
                    _pw.append(20.0 * np.log10(max(_rms, 1e-6)))
                _d = amplitude_df(_pw)
                _cv = self.df_cv
                _cv.delete("all")
                _w = _cv.winfo_width() or 800
                _cx, _cy, _R = _w / 2, 190, 120
                _cv.create_oval(_cx - _R, _cy - _R, _cx + _R, _cy + _R, outline=THEME["dim"])
                for _a, _lab in ((0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")):
                    _x = _cx + _R * np.cos(np.radians(_a))
                    _y = _cy - _R * np.sin(np.radians(_a))
                    _cv.create_line(_cx, _cy, _x, _y, fill=THEME["dim"])
                    _cv.create_text(_cx + (_x - _cx) * 1.12, _cy + (_y - _cy) * 1.12,
                                    text=_lab, fill=THEME["text"])
                _az = np.radians(_d["азимут_°"])
                _cv.create_line(_cx, _cy, _cx + _R * np.cos(_az), _cy - _R * np.sin(_az),
                                fill="#d95f02", width=3)
                _cv.create_text(14, 14, anchor="w", fill=THEME["text"],
                                text="Азимут %.1f° · розкид %s дБ · впевненість: %s · канали %s дБ"
                                     % (_d["азимут_°"], _d["розкид_дБ"], _d["впевненість"],
                                        ", ".join("%.0f" % p for p in _pw)))
                hist_log("Пеленгатор", "азимут %.1f°" % _d["азимут_°"], None)
            except Exception as _e:
                self.df_status.set("Помилка: %s" % _e)

        def _ea_open(self):
            """Діалог «⚡ Енергодетектор» (v6.1.0): скважність, темп, клас сигналу по каналах."""
            win, top, _lbl = self._win("⚡ Енергодетектор — активність і клас сигналу", 880, 540)
            self.ea_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.ea_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.ea_path, width=38).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.ea_fs = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.ea_fs, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Каналів:").pack(side="left", padx=(8, 0))
            self.ea_nch = tk.StringVar(value="1")
            ttk.Entry(row0, textvariable=self.ea_nch, width=3).pack(side="left", padx=4)
            ttk.Label(row0, text="Поріг, дБ:").pack(side="left", padx=(8, 0))
            self.ea_thr = tk.StringVar(value="12")
            ttk.Entry(row0, textvariable=self.ea_thr, width=4).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Активність", command=self._ea_run).pack(side="left", padx=10)
            self.ea_status = tk.StringVar(value="скважність → темп → клас: відео / керування / телеметрія / шум")
            ttk.Label(f, textvariable=self.ea_status, foreground=THEME["dim"]).pack(anchor="w")
            _cols = ("ch", "duty", "rate", "cls")
            self.ea_tree = ttk.Treeview(f, columns=_cols, show="headings", height=14)
            for _cid, _t, _w in (("ch", "Канал", 70), ("duty", "Скважність", 100),
                                 ("rate", "Пакетів/с", 90), ("cls", "Клас", 320)):
                self.ea_tree.heading(_cid, text=_t)
                self.ea_tree.column(_cid, width=_w, anchor="w")
            self.ea_tree.pack(fill="both", expand=True)

        def _ea_run(self):
            try:
                _fs = float(self.ea_fs.get()) * 1e6
                _thr = float(self.ea_thr.get())
                _chs, _n = load_raw(self.ea_path.get(), nch=int(self.ea_nch.get()))
                _t = self.ea_tree
                for _iid in _t.get_children():
                    _t.delete(_iid)
                for _k in sorted(_chs.keys()):
                    _seg = _chs[_k][:2000000]
                    _seg = _seg - float(_seg.mean())
                    _win = max(16, int(_fs // 2000))
                    _r = np.sqrt(np.convolve(_seg ** 2, np.ones(_win) / _win, mode="same"))
                    _pdb = 20.0 * np.log10(np.maximum(_r, 1e-9))
                    _st = activity_stats(_pdb - float(np.median(_pdb)), 1.0 / _fs, _thr)
                    _t.insert("", "end", values=(_k, "%.3f" % _st["скважність"],
                                                 "%.1f" % _st["пакетів_с"], _st["клас"]))
                hist_log("Енергодетектор", self.ea_path.get(), None)
            except Exception as _e:
                self.ea_status.set("Помилка: %s" % _e)

        def _gs_open(self):
            """Діалог «🛰 Детектор GPS-спуфінгу» (v6.1.0): ознаки підроблення в записі."""
            win, top, _lbl = self._win("🛰 Детектор GPS-спуфінгу — ознаки в записі", 880, 520)
            self.gsx_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x", pady=(0, 4))
            ttk.Label(row0, text="Запис:").pack(side="left")
            self.gsx_path = tk.StringVar(value="")
            ttk.Entry(row0, textvariable=self.gsx_path, width=38).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left")
            self.gsx_fs = tk.StringVar(value="4")
            ttk.Entry(row0, textvariable=self.gsx_fs, width=5).pack(side="left", padx=4)
            ttk.Label(row0, text="Канал:").pack(side="left", padx=(8, 0))
            self.gsx_ch = tk.StringVar(value="0")
            ttk.Entry(row0, textvariable=self.gsx_ch, width=3).pack(side="left", padx=4)
            ttk.Button(row0, text="▶ Аналіз", command=self._gsx_run).pack(side="left", padx=10)
            self.gsx_status = tk.StringVar(value="підлога/пік, стрибки, гладкість → вердикт")
            ttk.Label(f, textvariable=self.gsx_status, foreground=THEME["dim"]).pack(anchor="w")
            self.gsx_txt = tk.Text(f, height=14, wrap="word", bg=THEME["bg"], fg=THEME["text"], relief="flat")
            self.gsx_txt.pack(fill="both", expand=True, pady=(4, 0))

        def _gsx_run(self):
            try:
                _fs = float(self.gsx_fs.get()) * 1e6
                _chs, _n = load_raw(self.gsx_path.get(), nch=int(self.gsx_ch.get()) + 1)
                _s = _chs[int(self.gsx_ch.get())][:2000000]
                _s = _s - float(_s.mean())
                _win = max(16, int(_fs // 2000))
                _r = np.sqrt(np.convolve(_s ** 2, np.ones(_win) / _win, mode="same"))
                _pdb = 20.0 * np.log10(np.maximum(_r, 1e-9))
                _g = gnss_spoof_scan(_pdb, 1.0 / _fs)
                _t = self.gsx_txt
                _t.delete("1.0", "end")
                _t.insert("end", "Підлога: %s дБ · пік: %s дБ · стрибок: %s дБ\n\n"
                          % (_g["підлога_дБ"], _g["пік_дБ"], _g["стрибок_дБ"]))
                _t.insert("end", "Ознаки:\n")
                for _sg in _g["ознаки"]:
                    _t.insert("end", "  • %s\n" % _sg)
                if not _g["ознаки"]:
                    _t.insert("end", "  • (немає)\n")
                _t.insert("end", "\nВердикт: %s\n" % _g["вердикт"])
                hist_log("Спуфінг GNSS", _g["вердикт"], None)
            except Exception as _e:
                self.gsx_status.set("Помилка: %s" % _e)


        def _fpv_fs_refresh(self):
            """Показати обрану або авто-підібрану fs під поточні f і девіацію."""
            try:
                if self.fpv_fs_mode.get().startswith("Авто"):
                    _fa = fpv_fs_auto(float(self.fpv_f.get()) * 1e6,
                                      float(self.fpv_dev.get()) * 1e6)
                    self.fpv_fsi.set("→ %.0f" % (_fa / 1e6))
                else:
                    self.fpv_fsi.set("→ %s" % self.fpv_fs_mode.get())
            except Exception:
                self.fpv_fsi.set("→ не перекривається (f + девіація)")

        def _fpv_synth_now(self):
            """③ Інжектоване відео → ЧМ → .raw + контрольне розкладання растру."""
            try:
                _std = VIDEO_STD[self.fpv_std.get()]
                _f = float(self.fpv_f.get()) * 1e6
                _dev = float(self.fpv_dev.get()) * 1e6
                if self.fpv_fs_mode.get().startswith("Авто"):
                    _fs = fpv_fs_auto(_f, _dev)
                    _fs_tag = "fs авто %.0f МГц" % (_fs / 1e6)
                else:
                    _fs = float(self.fpv_fs_mode.get()) * 1e6
                    _fs_tag = "fs %.0f МГц" % (_fs / 1e6)
                _fr = int(float(self.fpv_frames.get()))
                _amp = float(self.fpv_amp.get())
                _nch = max(1, int(float(self.fpv_nch.get())))
                _name = self.fpv_pat.get()
                if _name.startswith("Зображення") and not getattr(self, "fpv_img_path", None):
                    _name = "Кольорові смуги"
                _pat = video_pattern(_name, 64, 48, img_path=getattr(self, "fpv_img_path", None))
                _sig = video_fm_modulate(_pat, _std, _f, _dev, _fs, _fr, amp=_amp)
                _codes = np.clip(np.rint(32768.0 + np.asarray(_sig, dtype=np.float64) * 24000.0),
                                 0, 65535).astype("<u2")
                _flat = np.repeat(_codes[:, None], _nch, axis=1).reshape(-1)
                _path = filedialog.asksaveasfilename(
                    defaultextension=".raw", initialfile="fpv_inject.raw",
                    title="Куди записати інжектоване відео (.raw)")
                if _path:
                    _flat.tofile(_path)
                _r = video_raster_demod(_sig, _fs, _std, _f, _dev, cols=64, rows=48)
                _err = float(np.mean(np.abs(_r - _pat)))
                import tempfile as _tf57
                _p1 = os.path.join(_tf57.gettempdir(), "fpv_src.ppm")
                _p2 = os.path.join(_tf57.gettempdir(), "fpv_dem.ppm")
                with open(_p1, "wb") as _fo:
                    _fo.write(_fpv_ppm(_pat))
                with open(_p2, "wb") as _fo:
                    _fo.write(_fpv_ppm(_r))
                self.fpv_photo_src = tk.PhotoImage(file=_p1).zoom(5)
                self.fpv_photo_dem = tk.PhotoImage(file=_p2).zoom(5)
                self.fpv_src_cv.delete("all")
                self.fpv_dem_cv.delete("all")
                self.fpv_src_cv.create_image(6, 6, image=self.fpv_photo_src, anchor="nw")
                self.fpv_dem_cv.create_image(6, 6, image=self.fpv_photo_dem, anchor="nw")
                _dur = _sig.size / _fs
                if _path:
                    self.fpv_status.config(text="%s; Записано %s — %.2f с, %d відл., %d кан(ів), %.1f МБ; "
                                           "растер відновлено з .raw (похибка %.3f). Файл відкривається "
                                           "в аналізі як звичайний .raw (та сама fs і канал)."
                                           % (_fs_tag, _path, _dur, _sig.size, _nch, _flat.size * 2 / 1e6, _err))
                else:
                    self.fpv_status.config(text="%s; запис скасовано; контрольне розкладання: растер відновлено (похибка %.3f)" % (_fs_tag, _err))
                self._fpv_fs_refresh()
                self._set_status("FPV-відео: інжекція синтезована (%s, %d кадрів)" % (self.fpv_std.get(), _fr))
            except Exception as _e57x:
                self.fpv_status.config(text="Помилка синтезу: %s" % _e57x)

        def _fpv_open(self):
            win, top, _lbl = self._win("📺 FPV-відео: перехоплення та інжекція — аналоговий відеопотік", 1140, 780)
            s1 = ttk.LabelFrame(top, text="① Перехоплення — скан FPV-діапазону (аналогове відео 5.8 ГГц)")
            s1.pack(fill="x", padx=8, pady=(8, 3))
            _r1 = ttk.Frame(s1)
            _r1.pack(fill="x", pady=1)
            ttk.Label(_r1, text="Діапазон:").pack(side="left")
            self.fpv_band = ttk.Combobox(_r1, values=list(FPV_BANDS), width=15, state="readonly")
            self.fpv_band.set("5.8 · Raceband")
            self.fpv_band.pack(side="left", padx=(4, 10))
            ttk.Label(_r1, text="Джерело піків:").pack(side="left")
            self.fpv_src = ttk.Combobox(_r1, values=("з аналізу .raw", "вручну"), width=14, state="readonly")
            self.fpv_src.set("вручну")
            self.fpv_src.pack(side="left", padx=(4, 10))
            ttk.Label(_r1, text="активні канали (напр. К4,К7):").pack(side="left")
            self.fpv_man = tk.StringVar(value="К4")
            ttk.Entry(_r1, textvariable=self.fpv_man, width=10).pack(side="left", padx=3)
            ttk.Label(_r1, text="рівень, дБм:").pack(side="left", padx=(8, 0))
            self.fpv_lvl = tk.StringVar(value="-60")
            ttk.Entry(_r1, textvariable=self.fpv_lvl, width=6).pack(side="left", padx=3)
            ttk.Label(_r1, text="SNR прийому, дБ:").pack(side="left", padx=(8, 0))
            self.fpv_snr = tk.StringVar(value="20")
            ttk.Entry(_r1, textvariable=self.fpv_snr, width=5).pack(side="left", padx=3)
            ttk.Button(_r1, text="📡 Сканувати діапазон", command=self._fpv_scan_now).pack(side="left", padx=(10, 0))
            self.fpv_rows_fr = ttk.Frame(s1)
            self.fpv_rows_fr.pack(fill="x", pady=(2, 1))
            self.fpv_q_lbl = ttk.Label(s1, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fpv_q_lbl.pack(fill="x")
            self.fpv_sum = ttk.Label(s1, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fpv_sum.pack(fill="x", pady=(1, 3))
            s2 = ttk.LabelFrame(top, text="② Інжекція іншого відео в цей канал — лінк-бюджет і FM-захоплення приймача")
            s2.pack(fill="x", padx=8, pady=3)
            _r2 = ttk.Frame(s2)
            _r2.pack(fill="x", pady=1)
            ttk.Label(_r2, text="p_rx відео, дБм:").pack(side="left")
            self.fpv_prx = tk.StringVar(value="-60")
            ttk.Entry(_r2, textvariable=self.fpv_prx, width=6).pack(side="left", padx=3)
            ttk.Label(_r2, text="запас (J/S), дБ:").pack(side="left", padx=(8, 0))
            self.fpv_marg = tk.StringVar(value="10")
            ttk.Entry(_r2, textvariable=self.fpv_marg, width=5).pack(side="left", padx=3)
            ttk.Label(_r2, text="дальність інжектора, км:").pack(side="left", padx=(8, 0))
            self.fpv_d = tk.StringVar(value="0.5")
            ttk.Entry(_r2, textvariable=self.fpv_d, width=6).pack(side="left", padx=3)
            ttk.Label(_r2, text="G антени, дБ:").pack(side="left", padx=(8, 0))
            self.fpv_g = tk.StringVar(value="0")
            ttk.Entry(_r2, textvariable=self.fpv_g, width=5).pack(side="left", padx=3)
            ttk.Button(_r2, text="📐 Розрахувати EIRP інжектора", command=self._fpv_eirp_now).pack(side="left", padx=(10, 0))
            ttk.Button(_r2, text="⤵ Підставити ПП у Крок 2в", command=self._fpv_apply_pp).pack(side="left", padx=(8, 0))
            self.fpv_eirp_lbl = ttk.Label(s2, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fpv_eirp_lbl.pack(fill="x", pady=(2, 3))
            s3 = ttk.LabelFrame(top, text="③ Інжектоване відео → .raw — синтез кадру, ЧМ на канал, контрольне розкладання растру")
            s3.pack(fill="x", padx=8, pady=3)
            _r3 = ttk.Frame(s3)
            _r3.pack(fill="x", pady=1)
            ttk.Label(_r3, text="Стандарт:").pack(side="left")
            self.fpv_std = ttk.Combobox(_r3, values=list(VIDEO_STD), width=7, state="readonly")
            self.fpv_std.set("PAL")
            self.fpv_std.pack(side="left", padx=(4, 10))
            ttk.Label(_r3, text="Канал:").pack(side="left")
            self.fpv_ch = ttk.Combobox(_r3, width=5, state="readonly")
            self.fpv_ch.pack(side="left", padx=(4, 10))
            ttk.Label(_r3, text="f, МГц:").pack(side="left")
            self.fpv_f = tk.StringVar(value="5769")
            ttk.Entry(_r3, textvariable=self.fpv_f, width=8).pack(side="left", padx=3)
            ttk.Label(_r3, text="девіація, МГц:").pack(side="left", padx=(8, 0))
            self.fpv_dev = tk.StringVar(value="8")
            ttk.Entry(_r3, textvariable=self.fpv_dev, width=5).pack(side="left", padx=3)
            ttk.Label(_r3, text="кадрів:").pack(side="left", padx=(8, 0))
            self.fpv_frames = tk.StringVar(value="10")
            ttk.Entry(_r3, textvariable=self.fpv_frames, width=5).pack(side="left", padx=3)
            ttk.Label(_r3, text="амплітуда:").pack(side="left", padx=(8, 0))
            self.fpv_amp = tk.StringVar(value="0.6")
            ttk.Entry(_r3, textvariable=self.fpv_amp, width=5).pack(side="left", padx=3)
            ttk.Label(_r3, text="fs, МГц:").pack(side="left", padx=(8, 0))
            self.fpv_fs_mode = ttk.Combobox(_r3, values=("Авто за сигналом", "64", "125", "250",
                                                         "500", "1000", "2000", "4000", "5000",
                                                         "6000", "8000", "12000"),
                                            width=12, state="readonly")
            self.fpv_fs_mode.set("Авто за сигналом")
            self.fpv_fs_mode.pack(side="left", padx=3)
            self.fpv_fsi = tk.StringVar(value="64")
            ttk.Label(_r3, textvariable=self.fpv_fsi, foreground=THEME["accent"]).pack(side="left", padx=2)
            ttk.Label(_r3, text=".raw каналів:").pack(side="left", padx=(8, 0))
            self.fpv_nch = tk.StringVar(value="1")
            ttk.Entry(_r3, textvariable=self.fpv_nch, width=4).pack(side="left", padx=3)
            _r4 = ttk.Frame(s3)
            _r4.pack(fill="x", pady=1)
            ttk.Label(_r4, text="Джерело кадру:").pack(side="left")
            self.fpv_pat = ttk.Combobox(_r4, values=("Кольорові смуги", "Градієнт", "Смуги-застереження",
                                                     "Зображення (файл)…"), width=20, state="readonly")
            self.fpv_pat.set("Кольорові смуги")
            self.fpv_pat.pack(side="left", padx=(4, 8))
            ttk.Button(_r4, text="🖼 Обрати зображення…", command=self._fpv_pick_img).pack(side="left")
            ttk.Button(_r4, text="🎞 Синтезувати → .raw + растер", command=self._fpv_synth_now).pack(side="left", padx=(10, 0))
            self.fpv_status = ttk.Label(s3, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fpv_status.pack(fill="x", pady=(2, 2))
            _cvr = ttk.Frame(s3)
            _cvr.pack(fill="x", pady=(0, 4))
            _c1f = ttk.Frame(_cvr)
            _c1f.pack(side="left", padx=(0, 12))
            ttk.Label(_c1f, text="джерело (кадр):").pack(anchor="w")
            self.fpv_src_cv = tk.Canvas(_c1f, width=330, height=252, bg=THEME["panel"], highlightthickness=1)
            self.fpv_src_cv.pack()
            _c2f = ttk.Frame(_cvr)
            _c2f.pack(side="left")
            ttk.Label(_c2f, text="відновлений растер із .raw:").pack(anchor="w")
            self.fpv_dem_cv = tk.Canvas(_c2f, width=330, height=252, bg=THEME["panel"], highlightthickness=1)
            self.fpv_dem_cv.pack()
            _br = ttk.Frame(top)
            _br.pack(fill="x", padx=8, pady=(2, 8))
            ttk.Button(_br, text="Закрити", command=win.destroy).pack(side="right")

            def _band_chg(_e=None):
                _chs = fpv_channels(self.fpv_band.get())
                self.fpv_ch["values"] = [_lab for _lab, _f in _chs]
                if _chs:
                    self.fpv_ch.set(_chs[0][0])
                    self.fpv_f.set("%.0f" % (_chs[0][1] / 1e6))
                    self._fpv_fs_refresh()

            def _ch_chg(_e=None):
                for _lab, _f in fpv_channels(self.fpv_band.get()):
                    if _lab == self.fpv_ch.get():
                        self.fpv_f.set("%.0f" % (_f / 1e6))
                        break
                    self._fpv_fs_refresh()

            self.fpv_band.bind("<<ComboboxSelected>>", _band_chg)
            self.fpv_ch.bind("<<ComboboxSelected>>", _ch_chg)
            self.fpv_fs_mode.bind("<<ComboboxSelected>>", lambda _e: self._fpv_fs_refresh())
            _band_chg()
            self.fpv_last_active = 0
            self.fpv_img_path = None
            self._fpv_fs_refresh()

        def _gen_fs_needed(self):
            """Потрібна fs за увімкненими каналами Генератора (Найквіст + запас)."""
            _fr = []
            _wmin = None
            for _r in getattr(self, "gen_rows", []):
                try:
                    _en, _kd, _vals = _r
                    if not _en.get():
                        continue
                    _f = parse_freq_hz(str(_vals[0].get()).strip())
                    if _f > 0:
                        _fr.append(_f)
                    _w = float(str(_vals[3].get()).replace(",", "."))
                    if _w > 0:
                        _wmin = _w if _wmin is None else min(_wmin, _w)
                except Exception:
                    continue
            return fs_auto_plan(_fr, min_width_s=(_wmin * 1e-6) if _wmin else None)

        def _gen_fs_refresh(self):
            """Режим «Авто за сигналом»: розрахувати fs за планом і записати в поле."""
            if getattr(self, "gen_fs_mode", None) is not None and \
                    self.gen_fs_mode.get() == "Авто за сигналом":
                try:
                    _fa = self._gen_fs_needed()
                    self.gen_fs.set("%.6g" % (_fa / 1e6))
                    self.gen_fs_hint.config(text="авто")
                except Exception as _e572x:
                    self.gen_fs_hint.config(text="!")
            else:
                self.gen_fs_hint.config(text="")

        def _melt_peaks_now(self):
            """Останній результат аналізу .raw зі списком піків (для профілю завади)."""
            for _name in ("_last_ana", "ana_res", "last_analysis"):
                _d = getattr(self, _name, None)
                if isinstance(_d, dict) and _d.get("peaks"):
                    return _d
            for _v in list(vars(self).values()):
                if isinstance(_v, dict):
                    _p = _v.get("peaks")
                    if isinstance(_p, (list, tuple)) and len(_p) > 0:
                        return _v
            return None

        def _melt_draw(self, mz):
            cv = self.mc_canvas
            cv.delete("all")
            _rows = mz.get("rows") or []
            if not _rows:
                return
            _W = max(float(cv.winfo_width()), 1000.0)
            _H = max(float(cv.winfo_height()), 170.0)
            _x0, _y0, _x1, _y1 = 46.0, 18.0, _W - 14.0, _H - 30.0
            _dmax = _rows[-1][0]

            def _X(d):
                return _x0 + (_x1 - _x0) * (d / max(_dmax, 1e-9))

            def _Y(p):
                return _y1 - (_y1 - _y0) * max(0.0, min(1.0, p))

            cv.create_line("%.1f %.1f %.1f %.1f" % (_x0, _y1, _x1, _y1),
                           fill=THEME["text"])
            cv.create_line("%.1f %.1f %.1f %.1f" % (_x0, _y0, _x0, _y1),
                           fill=THEME["text"])
            cv.create_line("%.1f %.1f %.1f %.1f" % (_x0, _Y(mz["prr_thr"]), _x1, _Y(mz["prr_thr"])),
                           fill=THEME["accent"], dash=(4, 3))
            cv.create_text(_x1 - 4, _Y(mz["prr_thr"]) - 8, text="поріг PRR %.2f" % mz["prr_thr"],
                           fill=THEME["accent"], anchor="e", font=("DejaVu Sans", 7))
            if mz["d_melt_km"] is not None:
                cv.create_line("%.1f %.1f %.1f %.1f" % (_X(mz["d_melt_km"]), _y0, _X(mz["d_melt_km"]), _y1),
                               fill=THEME["accent"], dash=(5, 3))
                cv.create_text(_X(mz["d_melt_km"]) + 4, _y0 + 8,
                               text="межа плавлення %.2f км" % mz["d_melt_km"],
                               fill=THEME["accent"], anchor="w", font=("DejaVu Sans", 7))
            _pts = " ".join("%.1f %.1f" % (_X(r[0]), _Y(r[3])) for r in _rows)
            cv.create_line(_pts, fill=THEME["accent"], width=2)
            for _r in _rows:
                cv.create_oval("%.1f %.1f %.1f %.1f" % (_X(_r[0]) - 2, _Y(_r[3]) - 2,
                                                        _X(_r[0]) + 2, _Y(_r[3]) + 2),
                               fill=THEME["accent"], outline="")
            for _fr in (0.0, 0.25, 0.5, 0.75, 1.0):
                cv.create_text(_X(_dmax * _fr), _y1 + 10, text="%.1f" % (_dmax * _fr),
                               fill=THEME["text"], font=("DejaVu Sans", 7))
            cv.create_text((_x0 + _x1) / 2.0, _y1 + 24, text="дальність завади, км",
                           fill=THEME["text"], font=("DejaVu Sans", 8))
            cv.create_text(_x0 + 4, _y0 - 8, text="PRR пакета",
                           fill=THEME["text"], anchor="w", font=("DejaVu Sans", 8))

        def _melt_calc(self):
            try:
                _mk = self.mc_m.get()
                _c = float(self.mc_c.get())
                _e = float(self.mc_eirp.get())
                _f = float(self.mc_f.get()) * 1e6
                _nf = float(self.mc_nf.get())
                _bw = float(self.mc_bw.get())
                _rb = float(self.mc_rb.get())
                _pk = int(float(self.mc_pkt.get()))
                _thr = float(self.mc_thr.get())
                _dm = float(self.mc_dmax.get())
                _mz = melt_zone(_mk, _c, _e, _f, _nf, _bw, _rb, _pk,
                                prr_thr=_thr, d_max_km=_dm)
                if _mz["d_melt_km"] is not None:
                    _zt = "плавлення до %.2f км — далі канал дрона живий (PRR ≥ %s)" % (_mz["d_melt_km"], _thr)
                else:
                    _zt = "плавлення немає ніде в діапазоні — канал дрона живий на всій дальності"
                _txt = "① %s, C=%.1f дБм, EIRP завади %.1f дБм: %s" % (_mk, _c, _e, _zt)
                if _mz["rows"]:
                    _last = _mz["rows"][-1]
                    _txt += "; край %.2f км: J=%.1f дБм, SNR %+.1f дБ, PRR %.3f" % (_last[0], _last[1], _last[2], _last[3])
                self.mc_lbl.config(text=_txt)
                self._melt_draw(_mz)
                _fw = follow_me_sim(float(self.fm2_rate.get()), float(self.fm2_ret.get()),
                                    float(self.fm2_p.get()), float(self.fm2_n.get()))
                _ft = "② хоп %.1f мс → завада накриває %.1f%% хопів — %s" % (_fw["t_hop_ms"], 100.0 * _fw["p_cover"], _fw["verdict"])
                _ft += "; сітка N=%s коштує дрону +%.1f дБ потрібної потужності завади" % (self.fm2_n.get(), _fw["gain_db"])
                self.fm2_lbl.config(text=_ft)
                _fsr = failsafe_window(float(self.fs2_s.get()), float(self.fs2_rate.get()),
                                       float(self.fs2_p.get()))
                if _fsr["ok"]:
                    _st = "встигає (p failsafe %.3f ≥ 0.95)" % _fsr["p_fs"]
                else:
                    _st = "НЕ встигає надійно (p failsafe %.3f < 0.95) — ризик падіння дрона" % _fsr["p_fs"]
                _ft2 = "③ %d хопів у вікні %.1f с · потрібна p блокування ≥ %.4f/хоп · фактична %.4f → %s" % (_fsr["hops"], float(self.fs2_s.get()), _fsr["p_req"], _fsr["p_fs"], _st)
                self.fs2_lbl.config(text=_ft2)
                _pp = protocol_pulse(self.pp_mode.get(), float(self.pp_rate.get()),
                                     float(self.pp_dur.get()),
                                     sf=int(float(self.pp_sf.get())),
                                     bw_hz=float(self.pp_bw.get()),
                                     sym=int(float(self.pp_sym.get())))
                _ft3 = "④ пульс %.2f мс, скважність %.1f%% → середня потужність завади нижча на %.1f дБ за тієї ж ефективності" % (_pp["pulse_ms"], 100.0 * _pp["duty"], _pp["econ_db"])
                self.pp_lbl.config(text=_ft3)
                if _mz["d_melt_km"] is not None:
                    self._set_status("лабораторія плавлення: зона впливу до %.2f км" % _mz["d_melt_km"])
                else:
                    self._set_status("лабораторія плавлення: канал дрона не пригнічується в діапазоні")
            except Exception as _e56x:
                self.mc_lbl.config(text="Помилка розрахунку: %s" % _e56x)

        def _melt_open(self):
            win, top, _lbl = self._win("🔥 Лабораторія плавлення — PRR · follow-me · failsafe · пульсація", 1140, 720)
            mc = ttk.LabelFrame(top, text="① Зона плавлення — PRR пакета дрона по дальності завади")
            mc.pack(fill="x", padx=8, pady=(8, 3))
            _r1 = ttk.Frame(mc); _r1.pack(fill="x", pady=1)
            ttk.Label(_r1, text="Модуляція дрона:").pack(side="left")
            self.mc_m = ttk.Combobox(_r1, values=list(MODS_MELT), width=11, state="readonly")
            self.mc_m.set("BPSK"); self.mc_m.pack(side="left", padx=(4, 10))
            ttk.Label(_r1, text="Сигнал C, дБм:").pack(side="left")
            self.mc_c = tk.StringVar(value="-95")
            ttk.Entry(_r1, textvariable=self.mc_c, width=7).pack(side="left", padx=3)
            _e56 = "30.0"
            try:
                _e56 = "%.1f" % (float(self.jq_pa.get()) + float(self.jq_amp.get()) + float(self.jq_ant_g.get()))
            except Exception:
                pass
            self.mc_eirp = tk.StringVar(value=_e56)
            ttk.Label(_r1, text="EIRP завади, дБм:").pack(side="left", padx=(10, 0))
            ttk.Entry(_r1, textvariable=self.mc_eirp, width=7).pack(side="left", padx=3)
            ttk.Label(_r1, text="f дрона, МГц:").pack(side="left", padx=(10, 0))
            self.mc_f = tk.StringVar(value="915")
            ttk.Entry(_r1, textvariable=self.mc_f, width=8).pack(side="left", padx=3)
            _r2 = ttk.Frame(mc); _r2.pack(fill="x", pady=1)
            ttk.Label(_r2, text="NF, дБ:").pack(side="left")
            self.mc_nf = tk.StringVar(value="6")
            ttk.Entry(_r2, textvariable=self.mc_nf, width=5).pack(side="left", padx=3)
            ttk.Label(_r2, text="BW дрона, Гц:").pack(side="left", padx=(10, 0))
            self.mc_bw = tk.StringVar(value="125000")
            ttk.Entry(_r2, textvariable=self.mc_bw, width=9).pack(side="left", padx=3)
            ttk.Label(_r2, text="Rb, біт/с:").pack(side="left", padx=(10, 0))
            self.mc_rb = tk.StringVar(value="5470")
            ttk.Entry(_r2, textvariable=self.mc_rb, width=7).pack(side="left", padx=3)
            ttk.Label(_r2, text="Пакет, біт:").pack(side="left", padx=(10, 0))
            self.mc_pkt = tk.StringVar(value="500")
            ttk.Entry(_r2, textvariable=self.mc_pkt, width=6).pack(side="left", padx=3)
            ttk.Label(_r2, text="Поріг PRR:").pack(side="left", padx=(10, 0))
            self.mc_thr = tk.StringVar(value="0.5")
            ttk.Entry(_r2, textvariable=self.mc_thr, width=5).pack(side="left", padx=3)
            ttk.Label(_r2, text="D макс, км:").pack(side="left", padx=(10, 0))
            self.mc_dmax = tk.StringVar(value="5.0")
            ttk.Entry(_r2, textvariable=self.mc_dmax, width=5).pack(side="left", padx=3)
            self.mc_lbl = ttk.Label(mc, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.mc_lbl.pack(fill="x", pady=(3, 1))
            self.mc_canvas = tk.Canvas(mc, height=170, bg=THEME["panel"], highlightthickness=0)
            self.mc_canvas.pack(fill="x", pady=(0, 4))
            fw = ttk.LabelFrame(top, text="② Follow-me — чи встигає завада за перестроюванням дрона")
            fw.pack(fill="x", padx=8, pady=3)
            _f1 = ttk.Frame(fw); _f1.pack(fill="x", pady=1)
            ttk.Label(_f1, text="Швидкість хопів, Гц:").pack(side="left")
            self.fm2_rate = tk.StringVar(value="100")
            ttk.Entry(_f1, textvariable=self.fm2_rate, width=6).pack(side="left", padx=3)
            ttk.Label(_f1, text="перестроювання завади, мс:").pack(side="left", padx=(8, 0))
            self.fm2_ret = tk.StringVar(value="5")
            ttk.Entry(_f1, textvariable=self.fm2_ret, width=5).pack(side="left", padx=3)
            ttk.Label(_f1, text="p виявлення хопа:").pack(side="left", padx=(8, 0))
            self.fm2_p = tk.StringVar(value="0.9")
            ttk.Entry(_f1, textvariable=self.fm2_p, width=5).pack(side="left", padx=3)
            ttk.Label(_f1, text="каналів у сітці N:").pack(side="left", padx=(8, 0))
            self.fm2_n = tk.StringVar(value="10")
            ttk.Entry(_f1, textvariable=self.fm2_n, width=5).pack(side="left", padx=3)
            self.fm2_lbl = ttk.Label(fw, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fm2_lbl.pack(fill="x", pady=(2, 3))
            ff = ttk.LabelFrame(top, text="③ Вікно failsafe — чи встигає дрон сісти, поки завада блокує хопи")
            ff.pack(fill="x", padx=8, pady=3)
            _ff1 = ttk.Frame(ff); _ff1.pack(fill="x", pady=1)
            ttk.Label(_ff1, text="Таймаут failsafe, с:").pack(side="left")
            self.fs2_s = tk.StringVar(value="3")
            ttk.Entry(_ff1, textvariable=self.fs2_s, width=5).pack(side="left", padx=3)
            ttk.Label(_ff1, text="швидкість хопів, Гц:").pack(side="left", padx=(8, 0))
            self.fs2_rate = tk.StringVar(value="100")
            ttk.Entry(_ff1, textvariable=self.fs2_rate, width=6).pack(side="left", padx=3)
            ttk.Label(_ff1, text="p блокування хопа завадою:").pack(side="left", padx=(8, 0))
            self.fs2_p = tk.StringVar(value="0.9")
            ttk.Entry(_ff1, textvariable=self.fs2_p, width=5).pack(side="left", padx=3)
            self.fs2_lbl = ttk.Label(ff, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.fs2_lbl.pack(fill="x", pady=(2, 3))
            fp = ttk.LabelFrame(top, text="④ Протокол-пульсація — мінімальний імпульс під протокол дрона")
            fp.pack(fill="x", padx=8, pady=3)
            _p1 = ttk.Frame(fp); _p1.pack(fill="x", pady=1)
            ttk.Label(_p1, text="Режим:").pack(side="left")
            self.pp_mode = ttk.Combobox(_p1, values=("LoRa преамбула", "Пакетний обмін", "Постійна"),
                                        width=16, state="readonly")
            self.pp_mode.set("LoRa преамбула"); self.pp_mode.pack(side="left", padx=(4, 8))
            ttk.Label(_p1, text="пакетів/с:").pack(side="left")
            self.pp_rate = tk.StringVar(value="10")
            ttk.Entry(_p1, textvariable=self.pp_rate, width=5).pack(side="left", padx=3)
            ttk.Label(_p1, text="пакет, мс:").pack(side="left", padx=(8, 0))
            self.pp_dur = tk.StringVar(value="8")
            ttk.Entry(_p1, textvariable=self.pp_dur, width=5).pack(side="left", padx=3)
            ttk.Label(_p1, text="SF:").pack(side="left", padx=(8, 0))
            self.pp_sf = tk.StringVar(value="7")
            ttk.Entry(_p1, textvariable=self.pp_sf, width=3).pack(side="left", padx=3)
            ttk.Label(_p1, text="BW, Гц:").pack(side="left", padx=(8, 0))
            self.pp_bw = tk.StringVar(value="125000")
            ttk.Entry(_p1, textvariable=self.pp_bw, width=8).pack(side="left", padx=3)
            ttk.Label(_p1, text="символів преамбули:").pack(side="left", padx=(8, 0))
            self.pp_sym = tk.StringVar(value="8")
            ttk.Entry(_p1, textvariable=self.pp_sym, width=3).pack(side="left", padx=3)
            self.pp_lbl = ttk.Label(fp, text="", foreground=THEME["accent"], wraplength=1090, justify="left")
            self.pp_lbl.pack(fill="x", pady=(2, 3))
            _br = ttk.Frame(top); _br.pack(fill="x", padx=8, pady=(2, 8))
            ttk.Button(_br, text="🧪 Розрахувати все", command=self._melt_calc).pack(side="left")
            ttk.Button(_br, text="🎯 Профіль завади з аналізу…", command=self._jamprof_open).pack(side="left", padx=(10, 0))
            ttk.Button(_br, text="🎲 Монте-Карло follow-me…", command=self._mc_open).pack(side="left", padx=(10, 0))
            ttk.Button(_br, text="Закрити", command=win.destroy).pack(side="right")
            self._melt_calc()

        def _jamprof_open(self):
            win, top, _lbl = self._win("🎯 Профіль завади з аналізу .raw", 820, 460)
            _ana = self._melt_peaks_now()
            _peaks_hz = []
            if _ana is not None:
                for _p in _ana["peaks"]:
                    if isinstance(_p, dict):
                        _peaks_hz.append(float(_p.get("f_hz")))
                    elif isinstance(_p, (tuple, list)):
                        _peaks_hz.append(float(_p[0]))
                    else:
                        _peaks_hz.append(float(_p))
            _prof = jam_profile_from_analysis(_peaks_hz, pri_us=_ana.get("pri_us") if _ana else None)
            self.jp_lbl = ttk.Label(top, text="", foreground=THEME["accent"], wraplength=770, justify="left")
            self.jp_lbl.pack(fill="both", expand=True, padx=10, pady=10)
            if _prof["n_peaks"] == 0:
                self.jp_lbl.config(text="Немає даних аналізу — спершу відкрийте .raw (аналіз спектра), тоді профіль збереться з топ-піків.")
            else:
                _t = "Піки спектра (МГц): %s" % ", ".join("%.3f" % _f for _f in _prof["freqs_mhz"])
                if _prof["span_mhz"] is not None:
                    _t += "\nРозкид смуги: %.3f МГц" % _prof["span_mhz"]
                if _prof["pri_us"] is not None:
                    _t += "\nPRI імпульсів: %.1f мкс" % _prof["pri_us"]
                _t += "\n\nПрофіль: мультичастотна завада по %d піках; EIRP у лабораторії — з Кроку 2в (ПП + підсилювач + антена)." % _prof["n_peaks"]
                self.jp_lbl.config(text=_t)
            _jr = ttk.Frame(top); _jr.pack(fill="x", padx=10, pady=(0, 10))
            ttk.Button(_jr, text="🔥 Відкрити лабораторію плавлення", command=self._melt_open).pack(side="left")
            ttk.Button(_jr, text="🧠 Розпізнати сигнатури…", command=self._sig_open).pack(side="left", padx=(10, 0))
            ttk.Button(_jr, text="Закрити", command=win.destroy).pack(side="right")

        # ===== v6.2.0: водоспад у реальному часі + PDF з GUI =====
        def _wf_open(self):
            """Діалог «🌊 Водоспад (реальний час)»: живий спектр — генератор або RTL-TCP."""
            win, top, _lbl = self._win("🌊 Водоспад (реальний час) — живий спектр: генератор або RTL-TCP", 980, 660)
            self.wf_top = win
            f = ttk.Frame(top, padding=6)
            f.pack(fill="both", expand=True)
            row0 = ttk.Frame(f)
            row0.pack(fill="x")
            ttk.Label(row0, text="Джерело:").pack(side="left")
            self.wf_src = tk.StringVar(value="Генератор (внутрішній)")
            ttk.Combobox(row0, textvariable=self.wf_src, state="readonly", width=22,
                         values=["Генератор (внутрішній)", "RTL-TCP"]).pack(side="left", padx=4)
            ttk.Label(row0, text="Вид:").pack(side="left", padx=(8, 0))
            self.wf_kind = tk.StringVar(value="Завада CW")
            ttk.Combobox(row0, textvariable=self.wf_kind, state="readonly", width=16,
                         values=list(KIND_KEY.keys())).pack(side="left", padx=4)
            ttk.Label(row0, text="fs, МГц:").pack(side="left", padx=(8, 0))
            self.wf_fs = tk.StringVar(value="8")
            ttk.Entry(row0, textvariable=self.wf_fs, width=6).pack(side="left", padx=2)
            ttk.Label(row0, text="f, МГц:").pack(side="left")
            self.wf_f = tk.StringVar(value="0.6")
            ttk.Entry(row0, textvariable=self.wf_f, width=8).pack(side="left", padx=2)
            ttk.Label(row0, text="Ампл.:").pack(side="left")
            self.wf_amp = tk.StringVar(value="1000")
            ttk.Entry(row0, textvariable=self.wf_amp, width=7).pack(side="left", padx=2)
            row1 = ttk.Frame(f)
            row1.pack(fill="x", pady=(4, 0))
            ttk.Label(row1, text="RTL-TCP host:порт:").pack(side="left")
            self.wf_host = tk.StringVar(value="127.0.0.1")
            ttk.Entry(row1, textvariable=self.wf_host, width=13).pack(side="left", padx=2)
            self.wf_port = tk.StringVar(value="1234")
            ttk.Entry(row1, textvariable=self.wf_port, width=6).pack(side="left", padx=2)
            ttk.Label(row1, text="f, МГц:").pack(side="left", padx=(8, 0))
            self.wf_rfreq = tk.StringVar(value="100")
            ttk.Entry(row1, textvariable=self.wf_rfreq, width=8).pack(side="left", padx=2)
            ttk.Label(row1, text="rate, МСпс:").pack(side="left")
            self.wf_rrate = tk.StringVar(value="2.4")
            ttk.Entry(row1, textvariable=self.wf_rrate, width=6).pack(side="left", padx=2)
            ttk.Label(row1, text="gain, дБ:").pack(side="left")
            self.wf_rgain = tk.StringVar(value="0")
            ttk.Entry(row1, textvariable=self.wf_rgain, width=5).pack(side="left", padx=2)
            ttk.Label(row1, text="Палітра:").pack(side="left", padx=(10, 0))
            self.wf_cmap = tk.StringVar(value="inferno")
            ttk.Combobox(row1, textvariable=self.wf_cmap, state="readonly", width=9,
                         values=["viridis", "inferno", "magma", "plasma"]).pack(side="left", padx=4)
            ttk.Label(row1, text="дБ:").pack(side="left", padx=(8, 0))
            self.wf_lo = tk.StringVar(value="-100")
            ttk.Entry(row1, textvariable=self.wf_lo, width=5).pack(side="left", padx=2)
            self.wf_hi = tk.StringVar(value="-20")
            ttk.Entry(row1, textvariable=self.wf_hi, width=5).pack(side="left", padx=2)
            ttk.Label(row1, text="інтервал, мс:").pack(side="left")
            self.wf_ms = tk.StringVar(value="200")
            ttk.Entry(row1, textvariable=self.wf_ms, width=5).pack(side="left", padx=2)
            row2 = ttk.Frame(f)
            row2.pack(fill="x", pady=(4, 0))
            ttk.Button(row2, text="▶ Пуск", command=self._wf_start).pack(side="left")
            ttk.Button(row2, text="⏸ Стоп", command=self._wf_stop).pack(side="left", padx=4)
            ttk.Button(row2, text="📄 PDF-звіт…", command=self._wf_pdf).pack(side="left", padx=4)
            self.wf_status = tk.StringVar(value="джерело → ▶ Пуск: кільцевий буфер 200 кадрів, нові — зверху")
            ttk.Label(row2, textvariable=self.wf_status, foreground=THEME["dim"]).pack(side="left", padx=10)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.wf_fig = Figure(figsize=(9.4, 4.8), dpi=100)
            self.wf_ax = self.wf_fig.add_subplot(111)
            self.wf_ax.set_xlabel("Частота (частка смуги)")
            self.wf_ax.set_ylabel("Кадри (нові зверху)")
            self.wf_cv = FigureCanvasTkAgg(self.wf_fig, master=f)
            self.wf_cv.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
            self.wf_fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.11)
            self._wf_buf = None
            self._wf_im = None
            self._wf_job = None
            self._wf_n = 0
            self._wf_rng = np.random.default_rng()
            win.protocol("WM_DELETE_WINDOW", self._wf_close)

        def _wf_alloc(self):
            """Буфер водоспаду (v6.2.0): 200 кадрів × 129 комірок, заглушено −160 дБ."""
            _cols = 256 // 2 + 1
            self._wf_buf = np.full((200, _cols), -160.0)
            self._wf_im = self.wf_ax.imshow(self._wf_buf, aspect="auto", origin="upper",
                                            cmap=self.wf_cmap.get(),
                                            vmin=float(self.wf_lo.get()),
                                            vmax=float(self.wf_hi.get()),
                                            extent=[0.0, 1.0, 200.0, 0.0])
            self.wf_cv.draw_idle()
            return self._wf_buf.shape

        def _wf_add(self, col_db):
            self._wf_buf = waterfall_push(self._wf_buf, col_db)
            self._wf_n += 1

        def _wf_draw(self):
            if self._wf_im is not None and self._wf_buf is not None:
                self._wf_im.set_data(self._wf_buf)
                self._wf_im.set_clim(float(self.wf_lo.get()), float(self.wf_hi.get()))
                self.wf_cv.draw_idle()
                self.wf_status.set("джерело: %s · кадрів: %d · палітра: %s"
                                   % (self.wf_src.get(), self._wf_n, self.wf_cmap.get()))

        def _wf_chunk(self):
            if self.wf_src.get() == "RTL-TCP":
                return rtl_tcp_read(self.wf_host.get(), int(self.wf_port.get()), 256,
                                    float(self.wf_rfreq.get()) * 1e6,
                                    float(self.wf_rrate.get()) * 1e6,
                                    float(self.wf_rgain.get() or 0))
            return make_wave(KIND_KEY[self.wf_kind.get()], 256,
                             float(self.wf_fs.get()) * 1e6,
                             float(self.wf_f.get()) * 1e6, 0.0,
                             float(self.wf_amp.get()), 0, self._wf_rng)

        def _wf_tick(self):
            try:
                _col = waterfall_column(self._wf_chunk(), 256,
                                        float(self.wf_lo.get()), float(self.wf_hi.get()))
                self._wf_add(_col)
                self._wf_draw()
                self._wf_job = self.wf_top.after(max(int(float(self.wf_ms.get())), 50),
                                                 self._wf_tick)
            except Exception as _e:
                self.wf_status.set("Помилка: %s" % _e)
                self._wf_stop()

        def _wf_start(self):
            if self._wf_buf is None:
                self._wf_alloc()
            self._wf_stop()
            self._wf_rng = np.random.default_rng()
            self._wf_tick()

        def _wf_stop(self):
            if self._wf_job is not None:
                try:
                    self.wf_top.after_cancel(self._wf_job)
                except Exception:
                    pass
                self._wf_job = None

        def _wf_close(self):
            self._wf_stop()
            try:
                self.wf_top.destroy()
            except Exception:
                pass

        def _wf_pdf(self):
            """PDF-звіт водоспаду прямо з GUI (v6.2.0): поточний кадр → gui_pdf_report."""
            _png = "/tmp/opencode/rs62_wf.png"
            self.wf_fig.savefig(_png, dpi=120)
            _path = filedialog.asksaveasfilename(
                defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
                title="PDF-звіт водоспаду")
            if not _path:
                return
            gui_pdf_report(_path, "Водоспад (реальний час)",
                           meta="джерело: %s · кадрів: %d" % (self.wf_src.get(), self._wf_n),
                           pngs=[_png])
            self.wf_status.set("PDF-звіт: %s" % _path)
            hist_log("Водоспад", "PDF-звіт (%d кадрів)" % self._wf_n, None)

        def _win(self, title, w=900, h=620):
            win = tk.Toplevel(self)
            win.title("%s — RAW Studio v%s" % (title, APP_VERSION))
            win.geometry("%dx%d" % (w, h))
            top = tk.Frame(win, bg=THEME["panel"])
            top.pack(side="top", fill="x", padx=8, pady=6)

            def _lbl(row, txt):
                tk.Label(row, text=txt, bg=THEME["panel"],
                         fg=THEME["text"]).pack(side="left", padx=(10, 2))
            return win, top, _lbl

        def _pick_into(self, var, title="Оберіть файл .raw"):
            p = filedialog.askopenfilename(title=title,
                                           filetypes=[("RAW записи", "*.raw"), ("Усі файли", "*.*")])
            if p:
                var.set(p)

        def _specgram_3d(self):
            try:
                fs_hz = parse_freq_hz(self.spg_fs.get())
                nch = int(self.spg_nch.get())
                ch = int(self.spg_ch.get())
                nfft = int(self.spg_nfft.get())
                ov = float(self.spg_ov.get()) / 100.0
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "3D-спектрограма: %s" % e)
                return
            p = self.spg_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Файл .raw не знайдено: %s" % (p or "—"))
                return
            try:
                fr, tm, db, note = raw_spectrogram(p, fs_hz, nch=nch, ch=ch, nfft=nfft, overlap=ov)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "3D-спектрограма: %s" % e)
                return
            self.spg_fig.clf()
            ax = self.spg_fig.add_subplot(111, projection="3d")
            Tm, Fm = np.meshgrid(tm, fr / 1e6)
            ax.plot_surface(Tm, Fm, db, cmap="viridis",
                            rstride=max(1, db.shape[0] // 120),
                            cstride=max(1, db.shape[1] // 120),
                            linewidth=0, antialiased=True)
            ax.set_xlabel("Час, мс")
            ax.set_ylabel("Частота, МГц")
            ax.set_zlabel("дБ")
            ax.set_title("3D-спектрограма: %s — канал %d, %s" % (os.path.basename(p), ch, note))
            self.spg_fig.tight_layout()
            self.spg_canvas.draw()
            self.spg_ax = ax
            self.spg_ready3d = True
            self._set_status("3D-спектрограма: %s — %d×%d комірок" % (os.path.basename(p), db.shape[0], db.shape[1]))

        def _vs3d_open(self):
            win, top, _lbl = self._win("Завада vs ціль (3D)")
            self.vs_path = tk.StringVar(value=getattr(self, "last_raw_path", ""))
            _lbl(top, "Файл цілі .raw:")
            tk.Entry(top, textvariable=self.vs_path, width=30, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=lambda: self._pick_into(self.vs_path)).pack(side="left", padx=2)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "fs, МГц:")
            self.vs_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.vs_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.vs_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.vs_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.vs_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.vs_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "завада — центр, МГц:")
            self.vs_jf = tk.StringVar(value="2400")
            tk.Entry(row2, textvariable=self.vs_jf, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "смуга, МГц:")
            self.vs_jbw = tk.StringVar(value="20")
            tk.Entry(row2, textvariable=self.vs_jbw, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "J/S, дБ:")
            self.vs_js = tk.StringVar(value="15")
            tk.Entry(row2, textvariable=self.vs_js, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="⚔ Порівняти", style="Accent.TButton",
                       command=self._vs3d_run).pack(side="left", padx=(12, 2))
            self.vs_status = tk.Label(win, text="Оберіть запис цілі і параметри завади.",
                                      bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.vs_status.pack(side="bottom", fill="x")
            self.vs_fig = Figure(figsize=(8.6, 5.0), dpi=100)
            self.vs_ax = self.vs_fig.add_subplot(111, projection="3d")
            self.vs_ax.set_title("Завада vs ціль — дані не обрано")
            self.vs_canvas = FigureCanvasTkAgg(self.vs_fig, master=win)
            self.vs_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.vs_canvas.draw()
            self.vs_result = None

        def _vs3d_run(self):
            try:
                fs_hz = parse_freq_hz(self.vs_fs.get())
                nch = int(self.vs_nch.get())
                ch = int(self.vs_ch.get())
                jf = parse_freq_hz(self.vs_jf.get())
                jbw = parse_freq_hz(self.vs_jbw.get())
                js = float(self.vs_js.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Завада vs ціль: %s" % e)
                return
            p = self.vs_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Файл .raw не знайдено: %s" % (p or "—"))
                return
            try:
                fr, psd = _psd_mean(_chan_column(p, nch, ch, max_samples=1 << 21), fs_hz, 4096)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Завада vs ціль: %s" % e)
                return
            i = int(np.argmax(psd))
            thr = psd[i] - 20.0
            j = i
            while j > 0 and psd[j] > thr:
                j -= 1
            k = i
            while k < psd.size - 1 and psd[k] > thr:
                k += 1
            ov = jam_overlap(float(fr[j]), float(fr[k]), jf, jbw)
            ov["js_db"] = js
            ov["band"] = (float(fr[j]), float(fr[k]))
            ov["jf"] = float(jf)
            ov["jbw"] = float(jbw)
            ov["raw_path"] = p
            ov["fs_hz"] = float(fs_hz)
            ov["nch"] = int(nch)
            ov["ch"] = int(ch)
            self.vs_result = ov
            tgt = psd - psd.max()
            jm = np.where((fr >= jf - jbw / 2.0) & (fr <= jf + jbw / 2.0), js, -80.0)
            self.vs_fig.clf()
            ax = self.vs_fig.add_subplot(111, projection="3d")
            Ff, Yy = np.meshgrid(fr / 1e6, np.array([0.0, 1.0]))
            ax.plot_surface(Ff, Yy, np.tile(tgt, (2, 1)), color="#3a7bd5", alpha=0.9, linewidth=0)
            Ff2, Yy2 = np.meshgrid(fr / 1e6, np.array([2.2, 3.2]))
            ax.plot_surface(Ff2, Yy2, np.tile(jm, (2, 1)), color="#d9534f", alpha=0.75, linewidth=0)
            if ov["f_lo"] is not None:
                Fo, Yo = np.meshgrid(np.array([ov["f_lo"] / 1e6, ov["f_hi"] / 1e6]),
                                     np.array([1.0, 2.2]))
                Zo = np.array([[float(tgt.min()), float(tgt.min())], [js, js]])
                ax.plot_surface(Fo, Yo, Zo, color="#f0ad4e", alpha=0.35, linewidth=0)
            ax.set_xlabel("Частота, МГц")
            ax.set_ylabel("ціль / завада")
            ax.set_zlabel("дБ відн. піка цілі")
            ax.set_title("Ціль: %.3f МГц (смуга %.3f МГц) — перекриття %.0f%%, J/S %+.0f дБ"
                         % (fr[i] / 1e6, (fr[k] - fr[j]) / 1e6, ov["pct"], js))
            self.vs_fig.tight_layout()
            self.vs_canvas.draw()
            self.vs_status.config(text="ціль: f %.3f МГц (смуга %.3f МГц) · завада %.3f МГц ±%.1f МГц · перекриття %.0f%% · J/S %+.0f дБ"
                                  % (fr[i] / 1e6, (fr[k] - fr[j]) / 1e6, jf / 1e6, jbw / 2e6, ov["pct"], js))
            self._set_status("Завада vs ціль: перекриття %.0f%% (J/S %+.0f дБ)" % (ov["pct"], js))

        def _ab_open(self):
            win, top, _lbl = self._win("Порівняння записів A/B")
            self.ab_a = tk.StringVar(value=getattr(self, "last_raw_path", ""))
            self.ab_b = tk.StringVar(value="")
            _lbl(top, "Запис A (.raw):")
            tk.Entry(top, textvariable=self.ab_a, width=26, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=lambda: self._pick_into(self.ab_a, "Запис A")).pack(side="left", padx=2)
            _lbl(top, "Запис B (.raw):")
            tk.Entry(top, textvariable=self.ab_b, width=26, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=lambda: self._pick_into(self.ab_b, "Запис B")).pack(side="left", padx=2)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "fs, МГц:")
            self.ab_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.ab_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.ab_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.ab_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.ab_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.ab_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="⚖ Порівняти", style="Accent.TButton",
                       command=self._ab_run).pack(side="left", padx=(12, 2))
            self.ab_status = tk.Label(win, text="A — до увімкнення завади, B — після: дельта у смузі піка A.",
                                      bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.ab_status.pack(side="bottom", fill="x")
            self.ab_fig = Figure(figsize=(8.6, 5.0), dpi=100)
            self.ab_ax = self.ab_fig.add_subplot(111)
            self.ab_ax.set_title("A/B — дані не обрано")
            self.ab_canvas = FigureCanvasTkAgg(self.ab_fig, master=win)
            self.ab_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.ab_canvas.draw()
            self.ab_result = None

        def _ab_run(self):
            try:
                fs_hz = parse_freq_hz(self.ab_fs.get())
                nch = int(self.ab_nch.get())
                ch = int(self.ab_ch.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Порівняння A/B: %s" % e)
                return
            pa_, pb_ = self.ab_a.get().strip(), self.ab_b.get().strip()
            for q, nm in ((pa_, "A"), (pb_, "B")):
                if not q or not os.path.exists(q):
                    messagebox.showerror(APP_TITLE, "Файл %s не знайдено: %s" % (nm, q or "—"))
                    return
            try:
                r = ab_compare(pa_, pb_, fs_hz, nch=nch, ch=ch)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Порівняння A/B: %s" % e)
                return
            self.ab_result = r
            self.ab_fig.clf()
            ax = self.ab_fig.add_subplot(111)
            fmhz = r["freqs"] / 1e6
            ax.plot(fmhz, r["psd_a"], lw=1, color="#3a7bd5", label="A (до)")
            ax.plot(fmhz, r["psd_b"], lw=1, color="#d9534f", label="B (після)")
            ax.set_xlabel("Частота, МГц")
            ax.set_ylabel("Рівень, дБ")
            ax.legend(loc="upper right")
            ax.set_title("A/B: пік A %.3f МГц — придушення %.1f дБ (виразність %.1f → %.1f дБ)"
                         % (r["f_hz"] / 1e6, r["suppress_db"], r["prom_a_db"], r["prom_b_db"]))
            self.ab_fig.tight_layout()
            self.ab_canvas.draw()
            self.ab_status.config(text="пік A: %.3f МГц (смуга %.3f МГц) · виразність піка %.1f → %.1f дБ · придушення %+.1f дБ"
                                  % (r["f_hz"] / 1e6, r["bw_hz"] / 1e6, r["prom_a_db"], r["prom_b_db"], r["suppress_db"]))
            self._set_status("A/B: придушення %+.1f дБ у смузі піка A (%.3f МГц)" % (r["suppress_db"], r["f_hz"] / 1e6))

        def _pri_open(self):
            win, top, _lbl = self._win("PRI-аналіз імпульсів")
            self.pri_path = tk.StringVar(value=getattr(self, "last_raw_path", ""))
            _lbl(top, "Файл .raw:")
            tk.Entry(top, textvariable=self.pri_path, width=30, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=lambda: self._pick_into(self.pri_path)).pack(side="left", padx=2)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "fs, МГц:")
            self.pri_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.pri_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.pri_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.pri_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.pri_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.pri_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "поріг, %:")
            self.pri_thr = tk.StringVar(value="15")
            tk.Entry(row2, textvariable=self.pri_thr, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="⏱ Аналізувати", style="Accent.TButton",
                       command=self._pri_run).pack(side="left", padx=(12, 2))
            ttk.Button(row2, text="→ у Крок 3 (період/тривалість)",
                       command=self._pri_apply).pack(side="left", padx=2)
            self.pri_status = tk.Label(win, text="Вимірює період слідування (PRI), тривалість і джиттер імпульсів.",
                                       bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.pri_status.pack(side="bottom", fill="x")
            self.pri_fig = Figure(figsize=(8.6, 4.6), dpi=100)
            self.pri_ax = self.pri_fig.add_subplot(111)
            self.pri_ax.set_title("Огиляюча — дані не обрано")
            self.pri_canvas = FigureCanvasTkAgg(self.pri_fig, master=win)
            self.pri_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.pri_canvas.draw()
            self.pri_result = None

        def _pri_run(self):
            try:
                fs_hz = parse_freq_hz(self.pri_fs.get())
                nch = int(self.pri_nch.get())
                ch = int(self.pri_ch.get())
                thr = float(self.pri_thr.get()) / 100.0
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "PRI-аналіз: %s" % e)
                return
            p = self.pri_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Файл .raw не знайдено: %s" % (p or "—"))
                return
            try:
                col = _chan_column(p, nch, ch, max_samples=1 << 21)
                r = pri_analyze(col, fs_hz, thr_ratio=thr)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "PRI-аналіз: %s" % e)
                return
            self.pri_result = r
            env = np.abs(col - float(np.median(col)))
            kk = max(3, 2 * int(round(fs_hz * 1e-6)) + 1)
            env = np.convolve(env, np.ones(kk) / float(kk), mode="same")
            nshow = min(env.size, 400000)
            self.pri_fig.clf()
            ax = self.pri_fig.add_subplot(111)
            ax.plot(np.arange(nshow) / fs_hz * 1e6, env[:nshow], lw=0.6, color="#3a7bd5")
            ax.axhline(float(env.max()) * thr, color="#d9534f", lw=1, ls="--", label="поріг")
            ax.set_xlabel("Час, мкс")
            ax.set_ylabel("Огиляюча")
            ax.legend(loc="upper right")
            ax.set_title("PRI %.1f мкс · PW %.1f мкс · імпульсів %d · джиттер %.2f мкс"
                         % (r["pri_us"], r["pw_us"], r["n_pulses"], r["jit_us"]))
            self.pri_fig.tight_layout()
            self.pri_canvas.draw()
            self.pri_status.config(text="PRI %.2f мкс · PW %.2f мкс · імпульсів %d · джиттер %.2f мкс · частота %.1f кГц"
                                   % (r["pri_us"], r["pw_us"], r["n_pulses"], r["jit_us"],
                                      1000.0 / max(r["pri_us"], 1e-9)))
            self._set_status("PRI: %.1f мкс (PW %.1f мкс, %d імп.)" % (r["pri_us"], r["pw_us"], r["n_pulses"]))

        def _pri_apply(self):
            if not self.pri_result:
                messagebox.showinfo(APP_TITLE, "Спочатку виконайте аналіз.")
                return
            self.ex_per.set("%.6g" % self.pri_result["pri_us"])
            self.ex_pw.set("%.6g" % self.pri_result["pw_us"])
            self._set_status("PRI → Крок 3: період %.2f мкс, тривалість %.2f мкс"
                             % (self.pri_result["pri_us"], self.pri_result["pw_us"]))

        def _demod_open(self):
            win, top, _lbl = self._win("Демодуляція аудіо (ЧМ/АМ)")
            self.dm_path = tk.StringVar(value=getattr(self, "last_raw_path", ""))
            _lbl(top, "Файл .raw:")
            tk.Entry(top, textvariable=self.dm_path, width=30, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=lambda: self._pick_into(self.dm_path)).pack(side="left", padx=2)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "fs, МГц:")
            self.dm_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.dm_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.dm_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.dm_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.dm_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.dm_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "тип:")
            self.dm_mode = ttk.Combobox(row2, values=("ЧМ (FM)", "АМ (AM)"), width=8, state="readonly")
            self.dm_mode.current(0)
            self.dm_mode.pack(side="left", padx=2)
            _lbl(row2, "девіація, кГц:")
            self.dm_dev = tk.StringVar(value="75")
            tk.Entry(row2, textvariable=self.dm_dev, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "декімація:")
            self.dm_dec = tk.StringVar(value="64")
            tk.Entry(row2, textvariable=self.dm_dec, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="🔊 Демодулювати і зберегти…", style="Accent.TButton",
                       command=self._demod_save).pack(side="left", padx=(12, 2))
            self.dm_status = tk.Label(win, text="Результат — моно WAV 16 біт для прослуховування.",
                                      bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.dm_status.pack(side="bottom", fill="x")

        def _demod_save(self):
            out = filedialog.asksaveasfilename(defaultextension=".wav", initialfile="demod.wav",
                                               title="Куди зберегти аудіо",
                                               filetypes=[("WAV аудіо", "*.wav")])
            if out:
                self._demod_run(out)

        def _demod_run(self, out_path):
            try:
                fs_hz = parse_freq_hz(self.dm_fs.get())
                nch = int(self.dm_nch.get())
                ch = int(self.dm_ch.get())
                dev = parse_freq_hz(self.dm_dev.get())
                dec = max(1, int(self.dm_dec.get()))
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Демодуляція: %s" % e)
                return False
            p = self.dm_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Файл .raw не знайдено: %s" % (p or "—"))
                return False
            try:
                col = _chan_column(p, nch, ch)
                mode = "am" if self.dm_mode.get().startswith("АМ") else "fm"
                a = demod_audio(col, fs_hz, mode=mode, fm_dev_hz=dev, decim=dec)
                rate = max(1000, int(fs_hz / dec))
                save_wav_int16(out_path, a, rate)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Демодуляція: %s" % e)
                return False
            self.dm_status.config(text="збережено: %s (%d відліків, %.1f кГц, %.1f с)"
                                  % (os.path.basename(out_path), a.size, rate / 1e3, a.size / float(rate)))
            self._set_status("демодуляція: %s" % os.path.basename(out_path))
            return True

        def _batch_open(self):
            win, top, _lbl = self._win("Батч-аналіз теки .raw")
            self.bt_dir = tk.StringVar(value="")
            _lbl(top, "Тека з .raw:")
            tk.Entry(top, textvariable=self.bt_dir, width=34, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=self._batch_pick).pack(side="left", padx=2)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 4))
            _lbl(row2, "fs, МГц:")
            self.bt_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.bt_fs, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.bt_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.bt_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.bt_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.bt_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="🗂 Сканувати теку", style="Accent.TButton",
                       command=self._batch_run).pack(side="left", padx=(12, 2))
            self.bt_status = tk.Label(win, text="CSV-зведення зберігається автоматично: <тека>/batch_report.csv",
                                      bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.bt_status.pack(side="bottom", fill="x")
            self.bt_fig = Figure(figsize=(8.6, 4.4), dpi=100)
            self.bt_ax = self.bt_fig.add_subplot(111)
            self.bt_ax.set_title("Піки по файлах — немає даних")
            self.bt_canvas = FigureCanvasTkAgg(self.bt_fig, master=win)
            self.bt_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.bt_canvas.draw()
            self.batch_rows = []

        def _batch_pick(self):
            d = filedialog.askdirectory(title="Тека з .raw записами")
            if d:
                self.bt_dir.set(d)

        def _batch_run(self):
            try:
                fs_hz = parse_freq_hz(self.bt_fs.get())
                nch = int(self.bt_nch.get())
                ch = int(self.bt_ch.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Батч-аналіз: %s" % e)
                return
            d = self.bt_dir.get().strip()
            try:
                rows = batch_scan(d, fs_hz, nch=nch, ch=ch)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Батч-аналіз: %s" % e)
                return
            self.batch_rows = rows
            csvp = os.path.join(d, "batch_report.csv")
            try:
                batch_save_csv(csvp, rows)
            except OSError as e:
                csvp = ""
            self.bt_fig.clf()
            ax = self.bt_fig.add_subplot(111)
            if rows:
                ax.scatter(range(len(rows)), [r["f_hz"] / 1e6 for r in rows],
                           c=[r["peak_rel_db"] for r in rows], cmap="viridis", s=42)
                ax.set_xticks(range(min(len(rows), 12)))
                ax.set_xticklabels([r["name"][:12] for r in rows[:12]], rotation=30, fontsize=7)
            ax.set_ylabel("Пік, МГц")
            ax.set_title("Піки по файлах (%d .raw)" % len(rows))
            self.bt_fig.tight_layout()
            self.bt_canvas.draw()
            self.bt_status.config(text="файлів: %d · CSV: %s" % (len(rows), csvp or "—"))
            self._set_status("батч: %d файлів, CSV %s" % (len(rows), os.path.basename(csvp) if csvp else "—"))

        def _power_opt_open(self):
            win, top, _lbl = self._win("Оптимізатор потужності завади", 760, 320)
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(2, 4))
            _lbl(row2, "частота завади, МГц:")
            self.pf_f = tk.StringVar(value="2400")
            tk.Entry(row2, textvariable=self.pf_f, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "J/S, дБ:")
            self.pf_js = tk.StringVar(value=self.jq_js.get())
            tk.Entry(row2, textvariable=self.pf_js, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "дальність, км:")
            self.pf_d = tk.StringVar(value=self.jq_optd.get())
            tk.Entry(row2, textvariable=self.pf_d, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "поріг приймача, дБм:")
            self.pf_thr = tk.StringVar(value=self.jq_thr.get())
            tk.Entry(row2, textvariable=self.pf_thr, width=6, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "G антени, дБі:")
            self.pf_g = tk.StringVar(value=self.jq_ant_g.get())
            tk.Entry(row2, textvariable=self.pf_g, width=5, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "BW завади:")
            self.pf_bwj = tk.StringVar(value="20")
            tk.Entry(row2, textvariable=self.pf_bwj, width=7, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "BW сигналу, МГц:")
            self.pf_bws = tk.StringVar(value=self.bw_var.get())
            tk.Entry(row2, textvariable=self.pf_bws, width=7, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            row3 = tk.Frame(win, bg=THEME["panel"])
            row3.pack(side="top", fill="x", padx=8, pady=2)
            ttk.Button(row3, text="🎯 Розрахувати", style="Accent.TButton",
                       command=self._popt_calc).pack(side="left", padx=2)
            ttk.Button(row3, text="→ у поле «ПП, дБм»",
                       command=self._popt_apply).pack(side="left", padx=8)
            self.lbl_pf = tk.Label(win, text="", bg=THEME["panel"], fg=THEME["accent"],
                                   anchor="w", font=("DejaVu Sans", 10, "bold"))
            self.lbl_pf.pack(side="top", fill="x", padx=10, pady=6)
            self._popt_last = None

        def _popt_calc(self):
            try:
                f = parse_freq_hz(self.pf_f.get())
                js = float(self.pf_js.get())
                dkm = float(self.pf_d.get())
                thr = float(self.pf_thr.get())
                g = float(self.pf_g.get())
                bwj = parse_freq_hz(self.pf_bwj.get())
                bws = parse_freq_hz(self.pf_bws.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Оптимізатор: %s" % e)
                return
            r = power_optimize(js, thr, f, dkm, g, 0.0, bwj, bws)
            self._popt_last = r
            self.lbl_pf.config(text="Мінімальна потужність ПП: %.1f дБм (EIRP %.1f дБм) — FSPL %.1f дБ, корекція смуги %+.1f дБ"
                               % (r["p_req_dbm"], r["eirp_dbm"], r["fspl_db"], r["corr_db"]))
            self._set_status("оптимізатор: ПП %.1f дБм на %.3g км (J/S %+.0f дБ)"
                             % (r["p_req_dbm"], dkm, js))

        def _popt_apply(self):
            if not self._popt_last:
                messagebox.showinfo(APP_TITLE, "Спочатку натисніть «Розрахувати».")
                return
            self.jq_pa.set("%.1f" % self._popt_last["p_req_dbm"])
            self._set_status("ПП → %.1f дБм (оптимізатор)" % self._popt_last["p_req_dbm"])

        def _map3d_open(self):
            win, top, _lbl = self._win("3D-мапа покриття (FSPL)")
            row2 = tk.Frame(win, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(2, 4))
            _lbl(row2, "частота завади, МГц:")
            self.m3_f = tk.StringVar(value="2400")
            tk.Entry(row2, textvariable=self.m3_f, width=8, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(row2, text="⛰ Побудувати 3D", style="Accent.TButton",
                       command=self._map3d_run).pack(side="left", padx=(12, 2))
            self.m3_status = tk.Label(win, text="EIRP = ПП + підсилювач + G антени (з Кроку 2в); запас над порігом у дБ.",
                                      bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.m3_status.pack(side="bottom", fill="x")
            self.m3_fig = Figure(figsize=(8.6, 5.0), dpi=100)
            self.m3_ax = self.m3_fig.add_subplot(111, projection="3d")
            self.m3_ax.set_title("3D-мапа покриття — немає даних")
            self.m3_canvas = FigureCanvasTkAgg(self.m3_fig, master=win)
            self.m3_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.m3_canvas.draw()
            self.map3d_done = False

        def _map3d_run(self):
            try:
                f = parse_freq_hz(self.m3_f.get())
                pa = float(self.jq_pa.get())
                amp = float(self.jq_amp.get())
                gant = float(self.jq_ant_g.get())
                thr = float(self.jq_thr.get())
                hpb = float(self.jq_hpb.get())
                fbr = float(self.jq_fbr.get())
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "3D-мапа: %s" % e)
                return
            ant = self.jq_ant.get()
            if ant == "Інший (власноруч)":
                ant = ""
            eirp = pa + amp + gant
            angs, rads, M = coverage_map3d(eirp, thr, f, ant_name=ant, hpb=hpb, fbr=fbr,
                                           g_rx_db=0.0)
            self.m3_fig.clf()
            ax = self.m3_fig.add_subplot(111, projection="3d")
            A, R = np.meshgrid(angs, rads)
            ax.plot_surface(A, R, M.T, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_xlabel("Кут, °")
            ax.set_ylabel("Дальність, км")
            ax.set_zlabel("Запас, дБ")
            ax.set_title("3D-мапа покриття: EIRP %.1f дБм, поріг %.0f дБм, f %.3g МГц"
                         % (eirp, thr, f / 1e6))
            self.m3_fig.tight_layout()
            self.m3_canvas.draw()
            self.map3d_done = True
            self.m3_status.config(text="EIRP %.1f дБм · запас 0 дБ = межа прийому; макс. дальність %.1f км"
                                  % (eirp, float(rads[-1])))
            self._set_status("3D-мапа покриття: EIRP %.1f дБм" % eirp)

        def _tl_preset_apply(self):
            nm = self.tl_preset.get()
            for p in SCENARIO_PRESETS:
                if p["name"] == nm:
                    self.tl_phases = scenario_parse_phases(p["phases"])
                    self._tl_redraw()
                    self._set_status("сценарій-пресет: %s — %d фаз" % (nm, len(self.tl_phases)))
                    return
            self._set_status("сценарій-пресет не знайдено: %s" % nm)

        def _prof_save(self):
            out = filedialog.asksaveasfilename(defaultextension=".rprofile",
                                               initialfile="session.rprofile",
                                               title="Зберегти профіль сесії",
                                               filetypes=[("Профіль сесії RAW Studio", "*.rprofile"),
                                                          ("JSON", "*.json")])
            if not out:
                return
            rows = []
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                rows.append({"enabled": bool(en.get()), "kind": kd.get(),
                             "vals": [v.get() for v in vals], "extra": self.ch_extra.get(i)})
            data = {"app": "RAW Studio", "version": APP_VERSION, "type": "session-profile",
                    "theme": getattr(self, "theme_saved", "techno"),
                    "fs_mhz": self.gen_fs.get(), "dur_s": self.gen_dur.get(),
                    "seed": self.gen_seed.get(),
                    "rows": rows, "phases": self.tl_phases,
                    "jq": {"pa": self.jq_pa.get(), "amp": self.jq_amp.get(),
                           "dist": self.jq_dist.get(), "dj": self.jq_dj.get(),
                           "ant_g": self.jq_ant_g.get(), "thr": self.jq_thr.get()},
                    "krok3": {"per_us": self.ex_per.get(), "pw_us": self.ex_pw.get(),
                              "rgpo_v": self.ex_rgpo_v.get(), "rgpo_d": self.ex_rgpo_d.get(),
                              "vgpo_f": self.ex_vgpo_f.get(), "vgpo_t": self.ex_vgpo_t.get(),
                              "protect": self.ex_protect.get(), "siglvl": self.ex_siglvl.get()}}
            try:
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except OSError as e:
                messagebox.showerror(APP_TITLE, "Не вдалося зберегти профіль: %s" % e)
                return
            self._set_status("профіль сесії збережено: %s (%d каналів, %d фаз)"
                             % (os.path.basename(out), len(rows), len(self.tl_phases)))

        def _prof_load(self):
            p = filedialog.askopenfilename(title="Завантажити профіль сесії",
                                           filetypes=[("Профіль сесії RAW Studio", "*.rprofile *.json"),
                                                      ("Усі файли", "*.*")])
            if not p or not os.path.exists(p):
                return
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") != "session-profile":
                    raise ValueError("файл не є профілем сесії")
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Профіль сесії: %s" % e)
                return
            rows = data.get("rows") or []
            self.ch_extra = {}
            for i, (en, kd, vals) in enumerate(self.gen_rows):
                if i < len(rows):
                    r = rows[i]
                    en.set(bool(r.get("enabled")))
                    if r.get("kind") in WAVE_KINDS:
                        kd.set(r["kind"])
                    for v, sv in zip(vals, r.get("vals") or []):
                        v.set(str(sv))
                    if r.get("extra"):
                        self.ch_extra[i] = {k: float(x) for k, x in r["extra"].items()}
                if i < len(self.ch_extra_btns):
                    self.ch_extra_btns[i].configure(text="⚙*" if i in self.ch_extra else "⚙")
            for var, key in ((self.gen_fs, "fs_mhz"), (self.gen_dur, "dur_s"), (self.gen_seed, "seed")):
                if str(data.get(key) or "") != "":
                    var.set(str(data[key]))
            jq = data.get("jq") or {}
            for var, key in ((self.jq_pa, "pa"), (self.jq_amp, "amp"), (self.jq_dist, "dist"),
                             (self.jq_dj, "dj"), (self.jq_ant_g, "ant_g"), (self.jq_thr, "thr")):
                if str(jq.get(key) or "") != "":
                    var.set(str(jq[key]))
            k3 = data.get("krok3") or {}
            for var, key in ((self.ex_per, "per_us"), (self.ex_pw, "pw_us"),
                             (self.ex_rgpo_v, "rgpo_v"), (self.ex_rgpo_d, "rgpo_d"),
                             (self.ex_vgpo_f, "vgpo_f"), (self.ex_vgpo_t, "vgpo_t"),
                             (self.ex_protect, "protect"), (self.ex_siglvl, "siglvl")):
                if str(k3.get(key) or "") != "":
                    var.set(str(k3[key]))
            self.tl_phases = scenario_parse_phases(data.get("phases") or [])
            self._tl_redraw()
            self._set_status("профіль сесії завантажено: %s (%d каналів, %d фаз)"
                             % (os.path.basename(p), len(rows), len(self.tl_phases)))

        def _specgram_open(self):
            _w = getattr(self, "spg_win", None)
            if _w is not None:
                try:
                    if _w.winfo_exists():
                        _w.lift()
                        return
                except tk.TclError:
                    pass
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            w = tk.Toplevel(self)
            w.title("Спектрограма .raw")
            w.geometry("920x700")
            w.configure(bg=THEME["panel"])
            self.spg_win = w
            top = tk.Frame(w, bg=THEME["panel"])
            top.pack(side="top", fill="x", padx=8, pady=6)

            def _lbl(row, txt):
                tk.Label(row, text=txt, bg=THEME["panel"], fg=THEME["text"]).pack(side="left", padx=(10, 2))

            self.spg_path = tk.StringVar(value=getattr(self, "last_raw_path", ""))
            _lbl(top, "Файл .raw:")
            tk.Entry(top, textvariable=self.spg_path, width=36, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            ttk.Button(top, text="Огляд…", command=self._specgram_browse).pack(side="left", padx=2)
            row2 = tk.Frame(w, bg=THEME["panel"])
            row2.pack(side="top", fill="x", padx=8, pady=(0, 6))
            _lbl(row2, "fs, МГц:")
            self.spg_fs = tk.StringVar(value=self.gen_fs.get())
            tk.Entry(row2, textvariable=self.spg_fs, width=9, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Каналів:")
            self.spg_nch = tk.StringVar(value=self.nch_var.get())
            tk.Entry(row2, textvariable=self.spg_nch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Канал №:")
            self.spg_ch = tk.StringVar(value="0")
            tk.Entry(row2, textvariable=self.spg_ch, width=4, bg=THEME["panel2"], fg=THEME["text"],
                     insertbackground=THEME["text"], relief="flat").pack(side="left", padx=2)
            _lbl(row2, "Вікно NFFT:")
            self.spg_nfft = ttk.Combobox(row2, values=("256", "512", "1024", "2048", "4096"),
                                         width=6, state="readonly")
            self.spg_nfft.set("1024")
            self.spg_nfft.pack(side="left", padx=2)
            _lbl(row2, "Перекриття, %:")
            self.spg_ov = ttk.Combobox(row2, values=("0", "25", "50", "75"), width=4, state="readonly")
            self.spg_ov.set("50")
            self.spg_ov.pack(side="left", padx=2)
            ttk.Button(row2, text="📊 Побудувати спектрограму", style="Accent.TButton",
                       command=self._specgram_build).pack(side="left", padx=12)
            ttk.Button(row2, text="💾 Експорт PNG…",
                       command=self._specgram_export).pack(side="left", padx=2)
            ttk.Button(row2, text="⛰ 3D-побудова",
                       command=self._specgram_3d).pack(side="left", padx=2)
            self.spg_status = tk.Label(w, text="Оберіть .raw і натисніть «Побудувати».",
                                       bg=THEME["panel"], fg=THEME["dim"], anchor="w")
            self.spg_status.pack(side="bottom", fill="x")
            self.spg_fig = Figure(figsize=(8.8, 5.6), dpi=100)
            self.spg_ax = self.spg_fig.add_subplot(111)
            self.spg_ax.set_title("Спектрограма — файл не обрано")
            self.spg_ax.set_xlabel("Час, мс")
            self.spg_ax.set_ylabel("Частота, МГц")
            self.spg_canvas = FigureCanvasTkAgg(self.spg_fig, master=w)
            self.spg_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            self.spg_canvas.draw()
            self.spg_ready = False

        def _specgram_browse(self):
            p = filedialog.askopenfilename(title="Файл .raw для спектрограми",
                                           filetypes=[("RAW записи", "*.raw"), ("Усі файли", "*.*")])
            if p:
                self.spg_path.set(p)

        def _specgram_build(self):
            try:
                fs_hz = parse_freq_hz(self.spg_fs.get())
                nch = int(self.spg_nch.get())
                ch = int(self.spg_ch.get())
                nfft = int(self.spg_nfft.get())
                ov = float(self.spg_ov.get()) / 100.0
                if not (0.0 <= ov < 1.0):
                    raise ValueError("перекриття 0…75%")
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "Спектрограма: %s" % e)
                return
            p = self.spg_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Файл .raw не знайдено: %s" % (p or "—"))
                return
            try:
                fr, tm, db, note = raw_spectrogram(p, fs_hz, nch=nch, ch=ch, nfft=nfft, overlap=ov)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Спектрограма: %s" % e)
                return
            for ax in list(self.spg_fig.axes)[1:]:
                self.spg_fig.delaxes(ax)
            self.spg_ax.clear()
            im = self.spg_ax.imshow(db, origin="lower", aspect="auto",
                                    extent=[tm[0], tm[-1], fr[0] / 1e6, fr[-1] / 1e6], cmap="viridis")
            self.spg_fig.colorbar(im, ax=self.spg_ax, label="Рівень, дБ (відн. піка)")
            self.spg_ax.set_xlabel("Час, мс")
            self.spg_ax.set_ylabel("Частота, МГц")
            self.spg_ax.set_title("Спектрограма: %s — канал %d, %s" % (os.path.basename(p), ch, note))
            self.spg_fig.tight_layout()
            self.spg_canvas.draw()
            self.spg_ready = True
            self._set_status("спектрограма: %s — %d×%d комірок (NFFT %d, перекриття %d%%)"
                             % (os.path.basename(p), db.shape[0], db.shape[1], nfft, int(round(ov * 100))))

        def _specgram_save_to(self, path):
            _apol = []
            if getattr(self, "fig_s", None) is not None:
                try:
                    _apol = [a for a in self.fig_s.get_axes() if getattr(a, "name", "") == "polar"]
                except Exception:
                    _apol = []
            if not _apol:
                self.spg_fig.savefig(path, dpi=150, facecolor=self.spg_fig.get_facecolor())
                return path
            try:
                _im = self.spg_ax.get_images()[0]
                _arr = np.asarray(_im.get_array())
                _f2 = Figure(figsize=(12.6, 5.6), dpi=100)
                _gs2 = _f2.add_gridspec(1, 2, width_ratios=[2.2, 1.0])
                _a1 = _f2.add_subplot(_gs2[0])
                _po = _a1.imshow(_arr, aspect="auto", extent=np.asarray(_im.get_extent()),
                                 cmap=_im.get_cmap())
                _a1.set_xlabel(self.spg_ax.get_xlabel()); _a1.set_ylabel(self.spg_ax.get_ylabel())
                _a1.set_title(self.spg_ax.get_title())
                _f2.colorbar(_po, ax=_a1)
                _a2 = _f2.add_subplot(_gs2[1], projection="polar")
                self._draw_ant_pattern(_a2)
                _f2.tight_layout()
                _f2.savefig(path, dpi=150, facecolor=_f2.get_facecolor())
                return path
            except Exception:
                self.spg_fig.savefig(path, dpi=150, facecolor=self.spg_fig.get_facecolor())
                return path

        def _specgram_export(self):
            if not getattr(self, "spg_ready", False) or not getattr(self, "spg_fig", None):
                messagebox.showinfo(APP_TITLE, "Спочатку побудуйте спектрограму — потім експортуйте у PNG.")
                return
            base = "spectrogram"
            p_raw = self.spg_path.get().strip()
            if p_raw:
                base = os.path.splitext(os.path.basename(p_raw))[0] + "_spectrogram"
            p = filedialog.asksaveasfilename(
                title="Експорт спектрограми у PNG", defaultextension=".png",
                initialfile=base + ".png",
                filetypes=[("PNG зображення", "*.png"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                self._specgram_save_to(p)
            except OSError as e:
                messagebox.showerror(APP_TITLE, "Експорт PNG: %s" % e)
                return
            self._set_status("спектрограму збережено: %s (PNG, 150 dpi)" % os.path.basename(p))


        def _msgbox(self):
            """Діалоги поверх головного вікна (v4.0.1): parent=self, щоб запит
            «Генерувати з аліасингом?»/«Розмір файла…» не ховався за вікном."""
            import tkinter.messagebox as _mbx
            _p = self
            class _Wrap:
                def askyesno(self2, *a, **k): return _mbx.askyesno(*a, parent=_p, **k)
                def showerror(self2, *a, **k): return _mbx.showerror(*a, parent=_p, **k)
                def showinfo(self2, *a, **k): return _mbx.showinfo(*a, parent=_p, **k)
            return _Wrap()

        def _metric_open(self, show_info=True):
            _mb = self._msgbox()
            rows = []
            for _i, (_en, _kd, _vals) in enumerate(self.gen_rows):
                if not bool(_en.get()):
                    continue
                try:
                    _f = parse_freq_hz(_vals[0].get())
                except ValueError:
                    continue
                _bw = 0.0
                try:
                    _cd = self.ch_extra.get(_i) or {}
                    _bw = float(_cd.get("jam_bw", 0) or 0)
                except (TypeError, ValueError):
                    _bw = 0.0
                if _bw <= 0:
                    try:
                        _bw = parse_freq_hz(self.ex_jam_bw.get())
                    except (ValueError, AttributeError):
                        _bw = 0.0
                if _bw <= 0:
                    _bw = 20e6
                try:
                    _a = float(_vals[2].get())
                except ValueError:
                    _a = 0.0
                rows.append((_f, _bw, _a))
            try:
                _ref = float(self.ex_siglvl.get())
            except (ValueError, AttributeError):
                _ref = None
            m = coverage_metric(rows, _ref)
            _txt = "Покриття плану: %.1f%%" % m["pct"]
            if m["holes"]:
                _txt += "; діри: " + ", ".join("%.4g–%.4g МГц" % (lo / 1e6, hi / 1e6)
                                               for lo, hi in m["holes"])
            _txt += "; J/S: " + (", ".join("ch%d %+.1f дБ" % (i, v)
                                           for i, v in enumerate(m["js"])) if m["js"] else "—")
            self._set_status(_txt)
            if show_info:
                _mb.showinfo(APP_TITLE, _txt)
            return m

        def _map_open(self):
            try:
                _pa = float(self.jq_pa.get())
            except ValueError:
                _pa = 20.0
            try:
                _pv = float(self.jq_amp.get())
            except ValueError:
                _pv = 0.0
            try:
                _g = float(self.jq_ant_g.get())
            except ValueError:
                _g = 0.0
            try:
                _thr = float(self.jq_thr.get())
            except ValueError:
                _thr = -85.0
            _f = 0.0
            for _en, _kd, _vals in self.gen_rows:
                if bool(_en.get()):
                    try:
                        _f = parse_freq_hz(_vals[0].get())
                        break
                    except ValueError:
                        pass
            if _f <= 0:
                _f = 1000e6
            _hpb, _fbr = 60.0, 20.0
            try:
                _hpb = float(self.jq_hpb.get())
                _fbr = float(self.jq_fbr.get())
            except ValueError:
                pass
            _an = self.jq_ant.get() if self.jq_ant.get() in JQ_ANTENNAS else ""
            degs, rs = coverage_map(_pa + _pv + _g, _thr, _f, ant_name=_an,
                                    hpb=_hpb, fbr=_fbr, n_pts=181)
            self.map_data = (degs, rs)
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                win = tk.Toplevel(self)
                win.title("Мапа покриття (полярна × FSPL)")
                win.geometry("760x620")
                fig = Figure(figsize=(7.0, 5.4), dpi=100)
                ax = fig.add_subplot(111, projection="polar")
                _th = np.radians(np.concatenate([degs, [360.0]]))
                _rr = np.concatenate([rs, [rs[0]]])
                ax.plot(_th, _rr, color="#e05050", lw=1.6)
                ax.fill(_th, _rr, color="#e05050", alpha=0.25)
                ax.set_title("Дальність завади, км × кут (EIRP %.1f дБм, поріг %.1f дБм, f %.4g МГц)"
                             % (_pa + _pv + _g, _thr, _f / 1e6), fontsize=9)
                self.map_win = win
                FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill="both", expand=True)
            except Exception:
                pass
            self._set_status("мапа покриття: макс %.1f км, мін %.1f км (EIRP %.1f дБм, поріг %.1f дБм)"
                             % (float(rs.max()), float(rs.min()), _pa + _pv + _g, _thr))
            return degs, rs

        def _pdf_open(self):
            _mb = self._msgbox()
            params = []
            for _en, _kd, _vals in self.gen_rows:
                try:
                    params.append({"enabled": bool(_en.get()), "kind": _kd.get(),
                                   "freq": parse_freq_hz(_vals[0].get()),
                                   "phase": float(_vals[1].get()), "amp": float(_vals[2].get()),
                                   "width_us": float(_vals[3].get()), "dc": float(_vals[4].get()),
                                   "gain": float(_vals[5].get())})
                except ValueError:
                    params.append({"enabled": False, "kind": "Синус", "freq": 0.0,
                                   "phase": 0.0, "amp": 0.0, "width_us": 0.0,
                                   "dc": 32768.0, "gain": 1.0})
            _m = self._metric_open(show_info=False) if any(bool(_e.get()) for _e, _k, _v
                                                           in self.gen_rows) else None
            out = filedialog.asksaveasfilename(
                parent=self, defaultextension=".pdf", initialfile="report_plan.pdf",
                title="Куди зберегти звіт PDF", filetypes=[("PDF файли", "*.pdf")])
            if not out:
                return
            _png_m = _png_s = None
            try:
                import tempfile
                from matplotlib.figure import Figure
                _d = tempfile.mkdtemp(prefix="rs_pdf_")
                _map = getattr(self, "map_data", None)
                if _map is not None:
                    _fg = Figure(figsize=(6.0, 4.8), dpi=100)
                    _ax = _fg.add_subplot(111, projection="polar")
                    _th = np.radians(np.concatenate([_map[0], [360.0]]))
                    _rr = np.concatenate([_map[1], [_map[1][0]]])
                    _ax.plot(_th, _rr, color="#cc3333", lw=1.4)
                    _ax.fill(_th, _rr, color="#cc3333", alpha=0.25)
                    _png_m = os.path.join(_d, "map.png")
                    _fg.savefig(_png_m)
                _en_rows = [p for p in params if p["enabled"]]
                if _en_rows:
                    _fg = Figure(figsize=(6.6, 2.8), dpi=100)
                    _ax = _fg.add_subplot(111)
                    for _i, _p in enumerate(_en_rows):
                        _ax.plot([_p["freq"] / 1e6], [_p["amp"]], "v",
                                 color="C%d" % (_i % 10), ms=8)
                        _ax.annotate("ch%d %.4g МГц" % (_i, _p["freq"] / 1e6),
                                     (_p["freq"] / 1e6, _p["amp"]), fontsize=7,
                                     rotation=90, va="bottom")
                    _ax.set_xlabel("Частота, МГц")
                    _ax.set_ylabel("Ампл., кодів")
                    _ax.set_title("Канали плану завад", fontsize=9)
                    _png_s = os.path.join(_d, "spec.png")
                    _fg.savefig(_png_s)
            except Exception:
                pass
            _spec3d = _map3d = _vs3d = None
            try:
                _lraw = getattr(self, "last_raw_path", "")
                if _lraw and os.path.exists(_lraw):
                    _spec3d = {"raw_path": _lraw,
                               "fs_hz": parse_freq_hz(self.gen_fs.get()),
                               "nch": int(self.nch_var.get() or 1), "ch": 0}
            except Exception:
                _spec3d = None
            try:
                _f53s = getattr(self, "m3_f", None)
                _f53 = parse_freq_hz(_f53s.get() if _f53s else "2400")
                _ant53 = self.jq_ant.get()
                _map3d = {"eirp_dbm": float(self.jq_pa.get()) + float(self.jq_amp.get())
                          + float(self.jq_ant_g.get()),
                          "thresh_dbm": float(self.jq_thr.get()), "f_hz": _f53,
                          "ant_name": (_ant53 if _ant53 != "Інший (власноруч)" else ""),
                          "hpb": float(self.jq_hpb.get()), "fbr": float(self.jq_fbr.get())}
            except Exception:
                _map3d = None
            try:
                _vr = getattr(self, "vs_result", None)
                if _vr and _vr.get("raw_path") and os.path.exists(_vr["raw_path"]):
                    _vs3d = {"raw_path": _vr["raw_path"], "fs_hz": _vr.get("fs_hz", 8e6),
                             "nch": _vr.get("nch", 1), "ch": _vr.get("ch", 0),
                             "jam_f_hz": _vr.get("jf", 0.0), "jam_bw_hz": _vr.get("jbw", 20e6),
                             "js_db": _vr.get("js_db", 15.0)}
            except Exception:
                _vs3d = None
            build_pdf_report(out, params, metric=_m, map_png=_png_m, spec_png=_png_s,
                             meta=self.cb_sdr.get() if getattr(self, "cb_sdr", None) else None,
                             spec3d=_spec3d, map3d=_map3d, vs3d=_vs3d)
            self._set_status("звіт PDF: %s" % os.path.basename(out))
            _mb.showinfo(APP_TITLE, "Звіт збережено: %s" % out)

        def _tl_add(self):
            try:
                _t = float(self.tl_t.get())
                _chs = [int(x) for x in self.tl_ch.get().replace(" ", "").split(",") if x != ""]
                _v = float(self.tl_v.get() or 0)
                _d = float(self.tl_d.get() or 0)
            except ValueError:
                self._set_status("сценарій: невірна тривалість, канали, швидкість або дальність")
                return
            if _t <= 0:
                self._set_status("сценарій: тривалість має бути > 0")
                return
            self.tl_phases.append({"t": _t, "ch": _chs, "v": _v, "d": _d})
            self._tl_redraw()

        def _tl_del(self):
            if self.tl_phases:
                self.tl_phases.pop()
                self._tl_redraw()

        def _tl_clear(self):
            self.tl_phases = []
            self._tl_redraw()

        def _tl_redraw(self):
            try:
                self.tl_list.config(state="normal")
                self.tl_list.delete("1.0", "end")
                _t0 = 0.0
                for _i, _ph in enumerate(self.tl_phases):
                    _vd = ""
                    if float(_ph.get("v", 0) or 0) != 0.0 or float(_ph.get("d", 0) or 0) != 0.0:
                        _vd = " v=%g м/с d=%g км" % (float(_ph.get("v", 0) or 0),
                                                     float(_ph.get("d", 0) or 0))
                    self.tl_list.insert(
                        "end", "фаза %d: %g с (t %g–%g) канали %s%s\n"
                        % (_i, _ph["t"], _t0, _t0 + _ph["t"],
                           ",".join(str(c) for c in _ph["ch"]), _vd))
                    _t0 += _ph["t"]
                cv = self.tl_cv
                cv.delete("all")
                _tot = sum(_ph["t"] for _ph in self.tl_phases) or 1.0
                _w = max(int(cv.winfo_width() or 600), 200)
                _cols = ("#4a90d9", "#d94a4a", "#4ad96a", "#d9c84a", "#9a4ad9", "#4ad9c8")
                _t0 = 0.0
                for _i, _ph in enumerate(self.tl_phases):
                    _x0 = _t0 / _tot * _w
                    _x1 = (_t0 + _ph["t"]) / _tot * _w
                    cv.create_rectangle(_x0, 2, max(_x1, _x0 + 2), 24,
                                        fill=_cols[_i % len(_cols)], outline="")
                    cv.create_text(max((_x0 + _x1) / 2, 2), 13,
                                   text="%gс ch%s" % (_ph["t"], ",".join(str(c) for c in _ph["ch"])),
                                   fill="#ffffff", font=("DejaVu Sans", 7))
                    _t0 += _ph["t"]
            except Exception:
                pass

        def _tl_gen_params(self):
            params = []
            for _en, _kd, _vals in self.gen_rows:
                try:
                    params.append({"enabled": bool(_en.get()), "kind": _kd.get(),
                                   "freq": parse_freq_hz(_vals[0].get()),
                                   "phase": float(_vals[1].get()), "amp": float(_vals[2].get()),
                                   "width_us": float(_vals[3].get()), "dc": float(_vals[4].get()),
                                   "gain": float(_vals[5].get())})
                except ValueError:
                    params.append({"enabled": False, "kind": "Синус", "freq": 0.0,
                                   "phase": 0.0, "amp": 0.0, "width_us": 0.0,
                                   "dc": 32768.0, "gain": 1.0})
            for _ci, _cd in getattr(self, "ch_extra", {}).items():
                if _ci < len(params):
                    params[_ci]["extra"] = _cd
            return params

        def _tl_save(self):
            out = filedialog.asksaveasfilename(
                parent=self, defaultextension=".scenario.json", initialfile="scenario.json",
                title="Куди зберегти сценарій", filetypes=[("Сценарії", "*.json")])
            if not out:
                return
            try:
                fs_s = parse_freq_hz(self.gen_fs.get())
            except ValueError:
                fs_s = DEFAULT_FS
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"fs_hz": fs_s, "params": self._tl_gen_params(),
                           "phases": self.tl_phases}, fh)
            self._set_status("сценарій збережено: %s (%d фаз)"
                             % (os.path.basename(out), len(self.tl_phases)))

        def _tl_load(self):
            pth = filedialog.askopenfilename(parent=self, title="Який сценарій відкрити",
                                             filetypes=[("Сценарії", "*.json")])
            if not pth or not os.path.exists(pth):
                return
            with open(pth, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.tl_phases = scenario_parse_phases(d.get("phases", []))
            self._tl_redraw()
            self._set_status("сценарій: %d фаз (%s)"
                             % (len(self.tl_phases), os.path.basename(pth)))

        def _tl_generate(self):
            _mb = self._msgbox()
            self._gen_fs_refresh()
            if not self.tl_phases:
                _mb.showinfo(APP_TITLE, "Додайте хоча б одну фазу сценарію.")
                return
            try:
                fs_g = parse_freq_hz(self.gen_fs.get())
                seed = int(self.gen_seed.get())
            except ValueError:
                _mb.showerror(APP_TITLE, "Невірні fs/seed")
                return
            try:
                extra = {"per_us": float(self.ex_per.get()),
                         "pw_us": float(self.ex_pw.get()),
                         "rise_us": float(self.ex_rise.get()),
                         "fall_us": float(self.ex_fall.get()),
                         "am_m": float(self.ex_am_m.get()),
                         "am_f": float(self.ex_am_f.get()),
                         "chirp_dev": parse_freq_hz(self.ex_chirp.get()),
                         "chirp_per_us": float(self.ex_chirp_per.get() or 0),
                         "qam_bits": float(QAM_MAP.get(self.ex_qam.get(), 0)),
                         "jam_bw": parse_freq_hz(self.ex_jam_bw.get()),
                         "sp_off": float(self.ex_sp_off.get() or 0),
                         "sp_drift": float(self.ex_sp_drift.get() or 0),
                         "rgpo_v": float(self.ex_rgpo_v.get() or 0),
                         "rgpo_d": float(self.ex_rgpo_d.get() or 0),
                         "vgpo_f": float(self.ex_vgpo_f.get() or 0),
                         "vgpo_t": float(self.ex_vgpo_t.get() or 0)}
            except ValueError:
                _mb.showerror(APP_TITLE, "Невірні додаткові параметри сигналу")
                return
            out = filedialog.asksaveasfilename(
                parent=self, defaultextension=".raw", initialfile="scenario_out.raw",
                title="Куди зберегти сценарій .raw", filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("сценарій: генерую %d фаз…" % len(self.tl_phases))
            _out, _tot = scenario_build_raw(out, self.tl_phases, self._tl_gen_params(),
                                            fs_g, seed, extra=extra)
            self._set_status("сценарій записано: %s (%d фаз, %d відліків)"
                             % (os.path.basename(out), len(self.tl_phases), _tot))
            _mb.showinfo(APP_TITLE, "Сценарій записано: %s\nфаз: %d, відліків: %d"
                         % (out, len(self.tl_phases), _tot))

        def generate(self):
            _mb = self._msgbox()
            self._gen_fs_refresh()
            try:
                fs_g = parse_freq_hz(self.gen_fs.get()); dur = float(self.gen_dur.get())
                seed = int(self.gen_seed.get())
            except ValueError:
                _mb.showerror(APP_TITLE, "Невірні fs/тривалість/seed"); return
            try:
                achx = self._current_achx()
            except ValueError as e:
                _mb.showerror(APP_TITLE, "АЧХ: %s" % e); return
            params = []
            for en, kind, vals in self.gen_rows:
                try:
                    freq = parse_freq_hz(vals[0].get())
                    phase, amp, width, dc, gain = (float(v.get()) for v in vals[1:])
                except ValueError:
                    _mb.showerror(APP_TITLE, "Невірне число або частота у параметрах канала"); return
                params.append({"enabled": bool(en.get()), "kind": kind.get(), "freq": freq,
                               "phase": phase, "amp": amp, "width_us": width,
                               "dc": dc, "gain": gain})
            for i, p in enumerate(params):
                if p["enabled"] and p["freq"] > 0 and not (SIG_MIN_HZ <= p["freq"] <= SIG_MAX_HZ):
                    fv, fu = human_freq(p["freq"])
                    _mb.showerror(
                        APP_TITLE, "ch%d: частота %.4g %s поза частотним планом "
                                   "100 МГц … 6000 МГц (6 блоків)" % (i, fv, fu))
                    return
            mw = self._min_width_us()
            if mw:
                for i, p in enumerate(params):
                    if (p["enabled"] and p["kind"] in ("Меандр", "Імпульс")
                            and 0 < p["width_us"] < mw):
                        if not _mb.askyesno(
                                APP_TITLE,
                                "ch%d: ширина %.6g мкс менша за мінімальну для %s "
                                "(%.6g мкс = 4 відліки при %.6g МС/с).\n"
                                "Імпульс може бути відтворений некоректно. Генерувати?"
                                % (i, p["width_us"], self.cb_sdr.get(), mw,
                                   self._sdr()["fs_hz"] / 1e6)):
                            return
            n = int(dur * fs_g)
            nch_g = len(params)
            size = n * nch_g * 2
            if size > 8e9:
                _mb.showerror(APP_TITLE, "Занадто великий файл: %.1f ГБ. Зменшіть тривалість."
                                     % (size / 1e9))
                return
            if size > 500e6 and not _mb.askyesno(
                    APP_TITLE, "Розмір файла %.0f МБ — продовжити?" % (size / 1e6)):
                return
            fmax = max((p["freq"] for p in params if p["enabled"]), default=0.0)
            if fmax > fs_g / 2:
                fv, fu = human_freq(fmax)
                fn, fun = human_freq(fs_g / 2)
                if not _mb.askyesno(
                        APP_TITLE,
                        "Максимальна частота %.4g %s вище Найквіста (%.4g %s).\n"
                        "Для прямої оцифровки потрібна fs ≥ %.4g Гц.\n\n"
                        "Генерувати з аліасингом?" % (fv, fu, fn, fun, 2 * fmax)):
                    return
            try:
                extra = {"per_us": float(self.ex_per.get()),
                         "pw_us": float(self.ex_pw.get()),
                         "rise_us": float(self.ex_rise.get()),
                         "fall_us": float(self.ex_fall.get()),
                         "am_m": float(self.ex_am_m.get()),
                         "am_f": float(self.ex_am_f.get()),
                         "chirp_dev": parse_freq_hz(self.ex_chirp.get()),
                         "chirp_per_us": float(self.ex_chirp_per.get() or 0),
                         "qam_bits": float(QAM_MAP.get(self.ex_qam.get(), 0)),
                         "jam_bw": parse_freq_hz(self.ex_jam_bw.get()),
                         "sp_off": float(self.ex_sp_off.get() or 0),
                         "sp_drift": float(self.ex_sp_drift.get() or 0),
                         "rgpo_v": float(self.ex_rgpo_v.get() or 0),
                         "rgpo_d": float(self.ex_rgpo_d.get() or 0),
                         "vgpo_f": float(self.ex_vgpo_f.get() or 0),
                         "vgpo_t": float(self.ex_vgpo_t.get() or 0)}
            except ValueError:
                _mb.showerror(APP_TITLE, "Невірні додаткові параметри сигналу"); return
            base = os.path.dirname(self.path) if getattr(self, "path", None) else os.getcwd()
            out = filedialog.asksaveasfilename(parent=self, initialdir=base, defaultextension=".raw",
                                               initialfile="generated_%dch_%dHz.raw" % (nch_g, int(fs_g)),
                                               title="Куди зберегти новий .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            _bad5 = []
            try:
                _prot5 = parse_protected_mhz(self.ex_protect.get())
            except (ValueError, AttributeError):
                _prot5 = []
            if _prot5:
                for _i5, (_e5, _k5, _v5) in enumerate(self.gen_rows):
                    if bool(_e5.get()):
                        try:
                            if is_protected(parse_freq_hz(_v5[0].get()), _prot5):
                                _bad5.append("ch%d (%s МГц)" % (_i5, _v5[0].get()))
                        except ValueError:
                            pass
            self._set_status("генерую %d×%d відліків%s…%s"
                             % (n, nch_g, ", з АЧХ" if achx else "",
                                (" ! захищені частоти: у захищених смугах увімкнені канали — "
                                 + ", ".join(_bad5)) if _bad5 else ""))
            if getattr(self, "audio_data", None):
                for _key, _idx in zip(("L", "R"), getattr(self, "audio_rows", ())):
                    if (_idx < len(params) and bool(params[_idx]["enabled"])
                            and self.audio_data.get(_key) is not None):
                        _arr = self.audio_data[_key]
                        if len(_arr) != n:
                            _arr = np.interp(np.linspace(0.0, len(_arr) - 1.0, n),
                                             np.arange(len(_arr)), _arr)
                        params[_idx]["samples"] = _arr * params[_idx]["amp"]
            for _ci, _cd in getattr(self, "ch_extra", {}).items():
                if _ci < len(params):
                    params[_ci]["extra"] = _cd
            if getattr(self, "ex_jam_shape", None) is not None:
                extra = dict(extra or {})
                _jsp = self.ex_jam_shape.get().strip()
                if _jsp and os.path.exists(_jsp):
                    extra["jam_shape"] = _jsp
                else:
                    extra.pop("jam_shape", None)
            flat = generate_raw(params, n, fs_g, nch_g, seed, achx=achx, extra=extra)
            save_raw(out, flat)
            _rep = []
            for _i, _p in enumerate(params):
                if not _p["enabled"]:
                    continue
                _codes = flat.reshape(-1, nch_g)[:, _i].astype(np.int64)
                _dev = _codes - int(round(float(_p.get("dc", 32768.0))))
                _pk = int(np.abs(_dev).max())
                _rms = float(np.sqrt((_dev.astype(np.float64) ** 2).mean()))
                _papr = 20.0 * math.log10(_pk / _rms) if _rms > 1.0 else 0.0
                _head = min(int(65535 - _codes.max()), int(_codes.min()))
                _rep.append("ch%d: пік ±%d, PAPR %.1f дБ, запас %d" % (_i, _pk, _papr, _head))
            self._load(out)
            self.last_raw_path = out
            self._set_status("записано %s (%.1f МБ) — %s" % (os.path.basename(out), os.path.getsize(out) / 1e6, "; ".join(_rep)))
            self.fs_var.set(self.gen_fs.get()); self.nch_var.set(str(nch_g))
            
    app = App()
    app.mainloop()


# ----------------------------------------------------------------------------
# CLI-режими
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--version", action="version",
                    version="RAW Studio v%s · Автор: %s" % (APP_VERSION, APP_AUTHOR))
    ap.add_argument("--headless", nargs="+", metavar=("FILE", "fs_nch"))
    ap.add_argument("--selftest", nargs="*", metavar="FILE")
    ap.add_argument("--preview", nargs="+", metavar=("OUT_PREFIX", "FILE"))
    ap.add_argument("--calgen", nargs="+", metavar=("OUT", "fs_nch_tones_fmin_fmax_amp_seg"))
    ap.add_argument("--calmeas", nargs="+", metavar=("RECORDED", "PLAN"))
    ap.add_argument("--scenario", nargs="+", metavar=("FILE", "OUT_fs"),
                    help="сценарій JSON (fs_hz, seed, params, phases) -> .raw")
    ap.add_argument("--batch", nargs="+", metavar=("DIR", "OUT_fs_nch_ch"),
                    help="батч-аналіз теки .raw -> CSV (піки по кожному файлу)")
    ap.add_argument("--recipes", nargs="+", metavar="TARGET",
                    help="рецепти SDR: --recipes CVBS --park \"HackRF One,PlutoSky R2\" [--pa 20 --dist 5 --dj 1 --ant ... --freq МГц --out CSV]")
    ap.add_argument("--park", default="")
    ap.add_argument("--pa", type=float, default=20.0)
    ap.add_argument("--dist", type=float, default=5.0)
    ap.add_argument("--dj", type=float, default=1.0)
    ap.add_argument("--ant", default="Штирьова 0.25λ")
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jamprofile", metavar="SPEC",
                    help='профіль завади за часом: "t0:t1:скважність;…,всього_с"')
    ap.add_argument("--percurve", metavar="SPEC",
                    help='PER-криві: "модем[,pkt_біт]" — BER/PER по J/S 0…33 дБ')
    ap.add_argument("--rcscan", metavar="SPEC",
                    help='пакети в записі: "шлях,fs_Гц[,nch,ch]" — період, скважність, піки')
    ap.add_argument("--melt", metavar="SPEC",
                    help="лабораторія плавлення (CLI): \"модем,f_МГц,c,j,nf,bw_МГц,rb,pkt\" — таблиця PRR по дистанціях")
    args = ap.parse_args()

    if args.selftest is not None:
        return selftest(args.selftest[0] if args.selftest else "P-2-2.raw")
    if args.recipes is not None:
        cli_recipes(args.recipes[0], park=args.park, pa_dbm=args.pa, dist_km=args.dist,
                    dj_km=args.dj, ant_name=args.ant, freq_mhz=args.freq, out=args.out)
        return None
    if getattr(args, "melt", None):
        cli_melt(args.melt)
        return None
    if args.jamprofile is not None:
        cli_jamprofile(args.jamprofile)
        return None
    if args.percurve is not None:
        cli_percurve(args.percurve)
        return None
    if args.rcscan is not None:
        cli_rcscan(args.rcscan)
        return None


    if args.scenario is not None and len(args.scenario) >= 2:
        sfile, out = args.scenario[0], args.scenario[1]
        fs_s = float(args.scenario[2]) if len(args.scenario) > 2 else 0.0
        with open(sfile, "r", encoding="utf-8") as fh:
            _d = json.load(fh)
        fs_s = fs_s or float(_d.get("fs_hz") or DEFAULT_FS)
        _ph = scenario_parse_phases(_d.get("phases", []))
        _out, _tot = scenario_build_raw(out, _ph, _d.get("params", []), fs_s,
                                        seed=int(_d.get("seed", 42)),
                                        extra=_d.get("extra"))
        print("сценарій: %s (%d фаз, %d відліків, %.1f МБ)"
              % (out, len(_ph), _tot, os.path.getsize(out) / 1e6))
        return 0

    if args.batch is not None and len(args.batch) >= 2:
        bdir, bout = args.batch[0], args.batch[1]
        fs_b = float(args.batch[2]) if len(args.batch) > 2 else float(DEFAULT_FS)
        nch_b = int(args.batch[3]) if len(args.batch) > 3 else 1
        ch_b = int(args.batch[4]) if len(args.batch) > 4 else 0
        rows_b = batch_scan(bdir, fs_b, nch=nch_b, ch=ch_b)
        batch_save_csv(bout, rows_b)
        print("батч: %d файлів -> %s" % (len(rows_b), bout))
        return 0

    if args.headless is not None:
        p = args.headless[0]
        fsv = float(args.headless[1]) if len(args.headless) > 1 else DEFAULT_FS
        nch = int(args.headless[2]) if len(args.headless) > 2 else DEFAULT_NCH
        rep, _, _ = analyze_report(p, fsv, nch)
        print(rep)
        return 0

    if args.preview is not None and len(args.preview) >= 2:
        prefix, p = args.preview[0], args.preview[1]
        fsv = float(args.preview[2]) if len(args.preview) > 2 else DEFAULT_FS
        nch = int(args.preview[3]) if len(args.preview) > 3 else DEFAULT_NCH
        return preview(prefix, p, fsv, nch)

    if args.calgen is not None and len(args.calgen) >= 3:
        out = args.calgen[0]
        fs_c, nch_c = float(args.calgen[1]), int(args.calgen[2])
        tones = int(args.calgen[3]) if len(args.calgen) > 3 else 32
        fmin = float(args.calgen[4]) if len(args.calgen) > 4 else SIG_MIN_HZ
        fmax = float(args.calgen[5]) if len(args.calgen) > 5 else SIG_MAX_HZ
        amp = float(args.calgen[6]) if len(args.calgen) > 6 else 8000.0
        seg_n = int(args.calgen[7]) if len(args.calgen) > 7 else 16384
        freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
        save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
        save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
        print("файл: %s (%.1f МБ), план: %s (%d тонів, бін %.3f Гц)"
              % (out, os.path.getsize(out) / 1e6, out + ".plan", len(freqs), bin_hz))
        print("\n".join("  %.6g Гц" % fq for fq in freqs))
        return 0

    if args.calmeas is not None and len(args.calmeas) >= 2:
        plan = load_cal_plan(args.calmeas[1])
        res = measure_achx(args.calmeas[0], plan)
        print("частота_Гц  ампл_кодів  дБ_відн_еталону(%d)" % plan["amp"])
        for fq, amp, gdb in res:
            print("%12.6g  %10.1f  %+6.2f" % (fq, amp, gdb))
        return 0

    try:
        run_gui()
    except ImportError as e:
        print("GUI недоступний (%s). Використайте: --headless / --calgen / --calmeas" % e)
        return 1
    return 0


def preview(prefix, path, fs, nch):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure
    blocks = build_blocks()
    chans, n = load_raw(path, nch)
    fig = Figure(figsize=(15, 5.5), dpi=110)
    draw_time(fig, chans, fs, 0.0, min(1e-5, n / fs))
    fig.savefig(prefix + "_time.png", bbox_inches="tight")
    rows, cols = _grid_for(nch)
    fig2 = Figure(figsize=(15, 3.2 * rows + 1), dpi=110)
    peaks = draw_spectrum(fig2, chans, fs)
    fig2.savefig(prefix + "_spectrum.png", bbox_inches="tight")
    fig3 = Figure(figsize=(14, 12), dpi=110)
    occ = channel_occupancy(chans, fs, blocks)
    info = draw_blocks(fig3, chans, fs, blocks, occ)
    fig3.savefig(prefix + "_blocks.png", bbox_inches="tight")
    print("\n".join(peaks)); print("\n".join(info))
    print("saved: %s_time.png, %s_spectrum.png, %s_blocks.png" % (prefix, prefix, prefix))
    return 0


def selftest(path):
    ok = True
    print("--- ПЕРЕВІРКА ЧАСТОТНОГО ПЛАНУ (6 блоків, 64 МГц) ---")
    blocks = build_blocks()
    full = [sum(1 for c in b["channels"] if not c["partial"]) for b in blocks]
    part = [sum(1 for c in b["channels"] if c["partial"]) for b in blocks]
    total = sum(len(b["channels"]) for b in blocks)
    last = blocks[-1]["channels"][-1]
    print("повних по блоках:", full, "(очік. [6,18,20,18,18,25], сума 105)")
    print("крайових:", part, "(очік. [1,1,1,1,1,0])")
    ok &= full == [6, 18, 20, 18, 18, 25] and part == [1, 1, 1, 1, 1, 0]
    ok &= total == 110 and abs(last["f1"] - 6000e6) < 1.0

    if os.path.exists(path):
        rep, _, _ = analyze_report(path, 1_000_000, 4)
        print("--- аналіз реального файла (4 канали, fs=1 МГц) ---")
        print("\n".join(rep.splitlines()[:10]))

    print("--- RF-ТЕСТ: fs = 12 ГГц, 8 каналів ---")
    fsrf, nchrf, nrf = 12e9, 8, 1_200_000
    params_rf = [
        {"enabled": True, "kind": "Синус", "freq": 100e6, "phase": 0, "amp": 12000, "width_us": 2, "dc": 32768, "gain": 1.0},
        {"enabled": True, "kind": "Синус", "freq": 852e6, "phase": 45, "amp": 12000, "width_us": 2, "dc": 30000, "gain": 1.0},
        {"enabled": True, "kind": "Шум", "freq": 0, "phase": 0, "amp": 700, "width_us": 0, "dc": 40000, "gain": 1.0},
        {"enabled": True, "kind": "Імпульс", "freq": 60e6, "phase": 0, "amp": 25000, "width_us": 0.002, "dc": 5000, "gain": 1.0},
    ] + [{"enabled": False, "kind": "Синус", "freq": 0, "phase": 0, "amp": 0,
          "width_us": 0, "dc": 1000 * (k + 5), "gain": 1.0} for k in range(4)]
    tmp_rf = "/tmp/opencode/rawstudio_rf8.raw"
    save_raw(tmp_rf, generate_raw(params_rf, nrf, fsrf, nchrf, seed=13))
    rrf, _ = load_raw(tmp_rf, nchrf)
    fr, ar = amplitude_spectrum(rrf[0], fsrf)
    pk0 = find_peaks(fr, ar, k=1, exclude_bins=5)[0]
    print("ch0: пік = %.4g Гц, ампл. %.0f (очік. 100 МГц / 12000)" % (pk0[0], pk0[1]))
    ok &= abs(pk0[0] - 100e6) < 1e6 and abs(pk0[1] - 12_000) < 2_500
    occ = channel_occupancy(rrf, fsrf, blocks)
    target = [c for c in blocks[1]["channels"] if abs(c["fc"] - 852e6) < 1e3][0]
    print("канал 852 МГц: К%03d, зайнятий: %s" % (target["global"], occ[target["global"]]["occupied"]))
    ok &= occ[target["global"]]["occupied"]

    # --- АЧХ-модуль: тони точно в бінах ---
    print("--- ТЕСТ АЧХ: амплітуда залежно від частоти ---")
    n_a = 1_200_000
    t = np.arange(n_a) / fsrf
    w = 10000 * np.sin(2 * np.pi * 100e6 * t) + 10000 * np.sin(2 * np.pi * 3e9 * t)
    pts = parse_achx_table("100e6 0\n3e9 -6.0206\n6e9 -12.0412")
    w_f = apply_achx(w, fsrf, pts)
    f_a, a_f = amplitude_spectrum(w_f, fsrf, nfft=n_a)
    a100 = float(a_f[np.argmin(np.abs(f_a - 100e6))])
    a3g = float(a_f[np.argmin(np.abs(f_a - 3e9))])
    print("з АЧХ: A(100М)=%.0f (очік. 10000), A(3Г)=%.0f (очік. 5000)" % (a100, a3g))
    ok &= abs(a100 - 10_000) < 300 and abs(a3g - 5_000) < 300

    # --- КАЛІБРУВАННЯ: замкнений контур ---
    print("--- ТЕСТ КАЛІБРУВАННЯ: прогін тонів → штучний тракт → вимірювання АЧХ ---")
    fs_c, nch_c = 12e9, 8
    tones, fmin, fmax, amp_ref, seg_n = 12, 100e6, 6000e6, 8000.0, 16384
    freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
    tract = parse_achx_table("100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
    cal = "/tmp/opencode/cal_src.raw"
    rec = "/tmp/opencode/cal_after_tract.raw"
    plan_p = cal + ".plan"
    save_raw(cal, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n))
    save_raw(rec, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=tract))
    save_cal_plan(plan_p, fs_c, nch_c, freqs, amp_ref, seg_n)
    res_src = measure_achx(cal, load_cal_plan(plan_p))
    res_rec = measure_achx(rec, load_cal_plan(plan_p))
    ctrl = max(abs(m[2]) for m in res_src)
    err = max(abs(m[2] - float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])))
              for m in res_rec)
    print("контроль без тракту: %.3f дБ; вимірювання тракту: макс. похибка %.3f дБ (очік. < 0.6)"
          % (ctrl, err))
    ok &= ctrl < 0.3 and err < 0.6
    combined = [(m[0], float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])) - m[2])
                for m in res_rec]
    save_raw("/tmp/opencode/cal_flat.raw",
             generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=combined))
    flat_max = max(abs(m[2]) for m in measure_achx("/tmp/opencode/cal_flat.raw",
                                                   load_cal_plan(plan_p)))
    print("після компенсації: макс. відхилення від 0 дБ = %.3f дБ (очік. < 1.5)" % flat_max)
    ok &= flat_max < 1.5

    # --- ІНТЕРПОЛЯЦІЯ АЧХ: сплайн/log/екстраполяція ---
    print("--- ТЕСТ ІНТЕРПОЛЯЦІЇ АЧХ: будь-яка частота між точками і за межами ---")
    pts_i = [(1e8, 0.0), (3e9, -6.0), (6e9, -12.0)]
    knots = np.array([1e8, 3e9, 6e9])
    want = 10 ** (np.array([0.0, -6.0, -12.0]) / 20.0)
    h_sp = achx_response(knots, pts_i, mode="spline")
    h_lg = achx_response(knots, pts_i, mode="logspline")
    print("сплайн через вузли: %s" % np.allclose(h_sp, want, rtol=1e-9))
    print("log-сплайн через вузли: %s" % np.allclose(h_lg, want, rtol=1e-9))
    ok &= np.allclose(h_sp, want, rtol=1e-9) and np.allclose(h_lg, want, rtol=1e-9)
    mid_sp = float(achx_response(np.array([1.5e9]), pts_i, mode="spline")[0])
    mid_ln = float(achx_response(np.array([1.5e9]), pts_i, mode="linear")[0])
    print("у проміжку 1.5 ГГц: сплайн=%.4f, лінійна=%.4f (різні методи — обидва валідні)" % (mid_sp, mid_ln))
    ok &= 0.3 < mid_sp < 0.8 and 0.3 < mid_ln < 0.8
    h_cl = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="clamp")[0])
    h_li = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="linear")[0])
    h_ze = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="zero")[0])
    print("поза діапазоном 8 ГГц: clamp=%.4f (очік. 0.2512), linear=%.4f (продовження вниз), zero=%.4f (очік. 0)"
          % (h_cl, h_li, h_ze))
    ok &= abs(h_cl - 0.25119) < 1e-3 and h_li < h_cl and h_ze == 0.0
    w_lg = apply_achx(10000 * np.sin(2 * np.pi * 1.5e9 * t), fsrf, pts_i, mode="logspline")
    print("генерація з log-сплайном на 1.5 ГГц: rms=%.0f кодів" % float(np.std(w_lg)))
    ok &= 3000 < float(np.std(w_lg)) < 6000

    # --- ПРОФІЛЬ У ФАЙЛ / НАЗАД ---
    print("--- ТЕСТ ПРОФІЛЮ АЧХ: збереження/завантаження (.achx, з режимами) ---")
    prof_path = "/tmp/opencode/achx_measured.achx"
    save_achx_profile(prof_path, [(m[0], m[2]) for m in res_rec],
                      mode="spline", extrap="linear", meta={"fs": "%g" % fs_c})
    pts_back, mode_b, ex_b = load_achx_profile(prof_path)
    df = max(abs(a[0] / b[0] - 1.0) for a, b in zip(pts_back, res_rec))
    dg = max(abs(a[1] - b[2]) for a, b in zip(pts_back, res_rec))
    print("точок: %d, похибка частот: %.2g, дБ: %.2g, режими: %s/%s (очік. spline/linear)"
          % (len(pts_back), df, dg, mode_b, ex_b))
    ok &= len(pts_back) == len(res_rec) and df < 1e-5 and dg < 1e-3
    ok &= mode_b == "spline" and ex_b == "linear"

    # --- LORA SF7/BW125: T_sym = 2^SF/BW ---
    print("--- ТЕСТ LORA: CSS up-chirp SF7/BW125 — T_sym 1.024 мс ---")

    def _finst_lora(x, fs):
        # аналітичний сигнал: ПОВНИЙ fft + маска (irfft повертає дійсний масив —
        # уявна частина аналітичного сигналу губиться, angle дає 0/π)
        n = len(x)
        X = np.fft.fft(x)
        H = np.zeros(n); H[0] = 1.0; H[1:n // 2] = 2.0
        if n % 2 == 0:
            H[n // 2] = 1.0
        z = np.fft.ifft(X * H)
        return fs * np.diff(np.unwrap(np.angle(z))) / (2 * np.pi)

    fs_l, f0_l, dev_l, ts_l = 8e6, 1.0e6, 125e3, 1.024e-3
    w_l = make_wave_ext("sine", 32768, fs_l, f0_l, 0.0, 20000, 0,
                        np.random.default_rng(7), chirp_dev=dev_l, chirp_per_us=1024)
    fi_l = _finst_lora(w_l, fs_l)[2000:-2000]
    rsp_l = np.where(np.diff(fi_l) < -80e3)[0]
    gaps_l = np.diff(rsp_l)
    print("діапазон f: %.4f…%.4f МГц (очік. 0.9375…1.0625), скидалень: %d, крок: %d відл. (очік. 8192)"
          % (fi_l.min() / 1e6, fi_l.max() / 1e6, len(rsp_l),
             int(np.median(gaps_l)) if len(gaps_l) else -1))
    ok &= abs(fi_l.min() - (f0_l - dev_l / 2)) < 20e3
    ok &= abs(fi_l.max() - (f0_l + dev_l / 2)) < 20e3
    ok &= len(rsp_l) == 3 and len(gaps_l) == 2 and abs(np.median(gaps_l) - ts_l * fs_l) < 200
    w_o = make_wave_ext("sine", 32768, fs_l, f0_l, 0.0, 20000, 0,
                        np.random.default_rng(7), chirp_dev=dev_l, chirp_per_us=0)
    fi_o = _finst_lora(w_o, fs_l)[2000:-2000]
    print("старий режим (період 0): f від %.4f до %.4f МГц (очік. ~1.0 → ~1.125)"
          % (fi_o[0] / 1e6, fi_o[-1] / 1e6))
    ok &= abs(fi_o[0] - f0_l) < 20e3 and fi_o[-1] > f0_l + 0.9 * dev_l
    par_l = [{"enabled": False, "dc": 32768} for _ in range(8)]
    par_l[0] = {"enabled": True, "kind": "Синус", "freq": f0_l, "phase": 0.0,
                "amp": 20000, "width_us": 10.0, "dc": 32768, "gain": 1.0}
    fl_l = generate_raw(par_l, 32768, fs_l, 8, seed=7,
                        extra={"chirp_dev": dev_l, "chirp_per_us": 1024})
    fi_g = _finst_lora(fl_l[0::8].astype(float) - 32768.0, fs_l)[2000:-2000]
    rsp_g = np.where(np.diff(fi_g) < -80e3)[0]
    print("generate_raw: скидалень %d (очік. 3)" % len(rsp_g))
    ok &= len(rsp_g) == 3

    # --- ГРУПИ СИГНАЛІВ (GNSS L-смуга) ---
    print("--- ТЕСТ ГРУП: бібліотека сигналів GNSS ---")
    ok &= len(SIG_PRESETS) == 60 and len(SIG_GROUPS) == 13
    ok &= len(FREQ_LIB) == 6 and len(LIB_FLAT) == 51
    for gname, members in SIG_GROUPS.items():
        ok &= all(m in SIG_PRESETS for m in members)
    fs_g = max(SIG_PRESETS[m]["fs_hint"]
               for m in SIG_GROUPS["Група: GNSS — усі системи (8 каналів)"])
    print("пресетів: %d, груп: %d, fs групи GNSS: %g МГц (очік. 54/12/4000)"
          % (len(SIG_PRESETS), len(SIG_GROUPS), fs_g / 1e6))
    print("бібліотека частот: %d категорій, %d записів (очік. 6/51)"
          % (len(FREQ_LIB), len(LIB_FLAT)))
    print("--- ТЕСТ КОМПЛЕКС GNSS L1: E1 + G1 + B1 + L1 в одному каналі ---")
    fs_k, n_k = 4e9, 400000
    w_k = make_wave("gnss_l1", n_k, fs_k, 1575.42e6, 0.0, 10000, 0, np.random.default_rng(3))
    P_k = np.abs(np.fft.rfft(w_k)) ** 2
    f_k = np.fft.rfftfreq(n_k, 1 / fs_k)

    def _bp(a, b):
        m = (f_k >= a) & (f_k < b)
        return float(P_k[m].mean())
    q_k = 0.5 * (_bp(1520e6, 1550e6) + _bp(1620e6, 1650e6))
    r_k = {"GPS L1/Galileo E1": _bp(1574.4e6, 1576.44e6) / q_k,
           "BeiDou B1": _bp(1560.1e6, 1562.1e6) / q_k,
           "ГЛОНАСС G1": _bp(1598e6, 1606e6) / q_k}
    rms_k = float(np.std(w_k))
    print("потужність відносно тихої смуги: " + ", ".join("%s %.0f×" % kv for kv in r_k.items())
          + "; rms %.0f (очік. ≈7071)" % rms_k)
    ok &= all(v > 20 for v in r_k.values()) and 5500 < rms_k < 8500

    # --- ЗАВАДИ (CRPA TGTX5050A, v3.3.0) ---
    print("--- ТЕСТ ЗАВАД: CW / шум / чирп ---")
    fs_j, n_j = 4e9, 400000

    def _jp(w, lo, hi):
        Pj = np.abs(np.fft.rfft(w)) ** 2
        fj = np.fft.rfftfreq(len(w), 1 / fs_j)
        return float(Pj[(fj >= lo) & (fj < hi)].mean())

    w_cw = make_wave("jam_cw", n_j, fs_j, 1575.42e6, 0.0, 1000, 0, np.random.default_rng(11))
    w_ns = make_wave_ext("jam_noise", n_j, fs_j, 1575.42e6, 0.0, 1000, 0,
                         np.random.default_rng(12), jam_bw=20e6)
    w_ch = make_wave_ext("jam_chirp", n_j, fs_j, 1575.42e6, 0.0, 1000, 0,
                         np.random.default_rng(13), chirp_dev=5e6, chirp_per_us=1000)
    q_j = max(_jp(w_ns, 1520e6, 1550e6), 1e-9)
    ok &= _jp(w_cw, 1575.0e6, 1575.84e6) > 500 * q_j
    ok &= _jp(w_ns, 1565.42e6, 1585.42e6) > 100 * max(_jp(w_ns, 1600e6, 1620e6), 1e-9)
    ok &= _jp(w_ch, 1570.42e6, 1580.42e6) > 100 * max(_jp(w_ch, 1600e6, 1620e6), 1e-9)
    print("CW: пік у біні 1575.42 МГц; шум 20 МГц і чирп ±5 МГц — енергія в смузі, поза смугою тихо; rms CW %.0f"
          % float(np.std(w_cw)))

    # --- СПОФІНГ (v3.4.0) ---
    print("--- ТЕСТ СПОФІНГ: зміщення і плаваюча несуча ---")
    fs_s, n_s = 4e9, 2000000

    def _resid(w, fc, off_est):
        # залишкова розстройка: комплексне змішування → ваговий центр спектра
        # поблизу нуля (зображення мішані на 2fc ≈ 3.15 ГГц — за межами вікна ±2 МГц)
        t = np.arange(len(w)) / fs_s
        z = w * np.exp(-2j * np.pi * (fc + off_est) * t)
        Z = np.abs(np.fft.fftshift(np.fft.fft(z))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(len(w), 1 / fs_s))
        m = (f >= -2e6) & (f <= 2e6)
        P, fm = Z[m], f[m]
        return float((P * fm).sum() / P.sum())

    w_al = make_wave_ext("spoof", n_s, fs_s, 1575.42e6, 0.0, 10000, 0,
                         np.random.default_rng(21), sp_off=0.0, sp_drift=0.0)
    w_of = make_wave_ext("spoof", n_s, fs_s, 1575.42e6, 0.0, 10000, 0,
                         np.random.default_rng(21), sp_off=25000.0, sp_drift=0.0)
    w_dr = make_wave_ext("spoof", n_s, fs_s, 1575.42e6, 0.0, 10000, 0,
                         np.random.default_rng(21), sp_off=0.0, sp_drift=5e8)
    r_al = _resid(w_al, 1575.42e6, 0.0)
    r_of = _resid(w_of, 1575.42e6, 0.0)
    r_of2 = _resid(w_of, 1575.42e6, 25000.0)
    r_dr = _resid(w_dr, 1575.42e6, 0.0)
    print("залишкова розстройка: вирівняний %.1f Гц (очік. ~0), зміщений %.0f Гц (очік. 25000), "
          "після компенсації %.1f Гц, дрейф %.0f Гц (очік. 125000)"
          % (r_al, r_of, r_of2, r_dr))
    ok &= abs(r_al) < 3000 and abs(r_of - 25000) < 3000 and abs(r_of2) < 3000 \
        and abs(r_dr - 125000) < 6000
    rms_s = float(np.std(w_al))
    ok &= 6500 < rms_s < 7600
    print("rms спуфера %.0f (очік. ≈7071)" % rms_s)
    print("--- ТЕСТ ШПС-ФОРМ ПО КАНАЛАХ: GPS/Galileo/ГЛОНАСС/BeiDou окремо ---")
    for kk, lo, hi, tag in (("gps_l1", 1574.4e6, 1576.44e6, "GPS L1"),
                            ("bds_b1", 1560.1e6, 1562.1e6, "BeiDou B1"),
                            ("glo_g1", 1598e6, 1606e6, "ГЛОНАСС G1")):
        w_s = make_wave(kk, n_k, fs_k, 1575.42e6, 0.0, 10000, 0, np.random.default_rng(5))
        P_s = np.abs(np.fft.rfft(w_s)) ** 2
        r_s = float(P_s[(f_k >= lo) & (f_k < hi)].mean()) / q_k
        print("%s: смуга %.0f× фону (очік. > 20)" % (tag, r_s))
        ok &= r_s > 20
    w_b = make_wave("gal_e1", n_k, fs_k, 1575.42e6, 0.0, 10000, 0, np.random.default_rng(6))
    P_b = np.abs(np.fft.rfft(w_b)) ** 2
    ctr = float(P_b[(f_k > 1575.42e6 - 150e3) & (f_k < 1575.42e6 + 150e3)].mean())
    lob = 0.5 * (float(P_b[(f_k > 1575.42e6 + 0.7e6) & (f_k < 1575.42e6 + 1.4e6)].mean())
                 + float(P_b[(f_k > 1575.42e6 - 1.4e6) & (f_k < 1575.42e6 - 0.7e6)].mean()))
    print("Galileo E1 BOC(1,1): бічні пелюстки / центр = %.1f (очік. > 3 — спектр роздвоєний)"
          % (lob / ctr))
    ok &= lob > 3 * ctr
    print("--- ТЕСТ MESH XK-F358: TD-COFDM пачки + FHSS ---")
    n_m, fs_m = 600000, 6e9
    w_m = make_mesh(n_m, fs_m, 2250e6, 10000, np.random.default_rng(9),
                    per_us=20, pw_us=10, bw_hz=10e6)
    P_m = np.abs(np.fft.rfft(w_m)) ** 2
    f_m = np.fft.rfftfreq(n_m, 1 / fs_m)
    plateau = float(P_m[(f_m > 2230e6) & (f_m < 2275e6)].mean())
    quiet_m = float(P_m[(f_m > 2800e6) & (f_m < 2950e6)].mean())
    duty = float(np.mean(np.abs(w_m) > 0.5 * float(np.std(w_m))))
    rms_m = float(np.std(w_m))
    print("плато/фон = %.0f (очік. > 10), заповнення %.2f (очік. ~0.3-0.5), rms %.0f (очік. ≈7071)"
          % (plateau / quiet_m, duty, rms_m))
    ok &= plateau > 10 * quiet_m and 0.25 < duty < 0.7 and 6000 < rms_m < 8200
    print("--- ТЕСТ MESH QAM: BPSK…1024QAM ---")
    rng_q = np.random.default_rng(17)

    def qam_rms(bits):
        if bits == 1:
            return float(np.std(rng_q.choice(np.array([-1.0, 1.0]), size=6400)))
        k = int(math.isqrt(2 ** bits))
        lv = (2 * rng_q.integers(0, k, size=6400) - (k - 1)).astype(float)
        lq = (2 * rng_q.integers(0, k, size=6400) - (k - 1)).astype(float)
        return float(np.std(lv + 1j * lq)) / math.sqrt(2.0 * (k * k - 1) / 3.0)
    rms_q = {b: qam_rms(b) for b in (1, 2, 4, 6, 8, 10)}
    print("rms QAM-сіток: " + ", ".join("%dб %.2f" % kv for kv in rms_q.items()) + " (очік. ≈1)")
    ok &= all(0.8 < v < 1.2 for v in rms_q.values())
    w_q = make_mesh(n_m, fs_m, 2250e6, 10000, np.random.default_rng(19),
                    per_us=20, pw_us=10, bw_hz=10e6, qam_bits=6)
    P_q = np.abs(np.fft.rfft(w_q)) ** 2
    pl_q = float(P_q[(f_m > 2230e6) & (f_m < 2275e6)].mean())
    q_q = float(P_q[(f_m > 2800e6) & (f_m < 2950e6)].mean())
    print("64QAM: плато/фон = %.0f (очік. > 10)" % (pl_q / q_q))
    ok &= q_q > 0 and pl_q > 10 * q_q
    print("--- ТЕСТ ІЧ: діапазон 1.5–6 мкм (SWIR/MWIR) ---")
    ok &= abs(wien_lambda_max(700 + 273.15) * 1e6 - 2.9777) < 0.01
    ok &= abs(lambda_to_freq(3e-6) / 1e12 - 99.93) < 0.1
    ok &= ir_subrange(2.5e-6).startswith("SWIR") and ir_subrange(4.5e-6).startswith("MWIR")
    ok &= ir_subrange(1.0e-6) == "поза 1.5–6 мкм"
    print("Він: 700 °C → %.3f мкм (пік), 3 мкм → %.4g ТГц, піддіапазони OK"
          % (wien_lambda_max(700 + 273.15) * 1e6, lambda_to_freq(3e-6) / 1e12))
    ok &= fs_g == 4000e6

    # --- МІКРОХВИЛІ (IEEE) ---
    print("--- ТЕСТ МІКРОХВИЛІ: літерні діапазони IEEE ---")
    ok &= len(IEEE_BANDS) == 9
    ok &= ieee_band_for(2.45e9) == "S" and ieee_band_for(1.57542e9) == "L"
    ok &= ieee_band_for(10.0e9) == "X" and ieee_band_for(30.0e9) == "Ka"
    ok &= ieee_band_for(0.3e9) == "UHF" and ieee_band_for(100e9) == "V/W"
    ok &= ieee_band_for(1.0e9) == "L" and ieee_band_for(4.0e9) == "C"
    ok &= ieee_band_for(0.1e9) is None and ieee_band_for(200e9) is None
    ok &= abs(freq_to_lambda(2.45e9) * 100 - 12.236) < 0.01
    print("IEEE: %d діапазонів; 2.45 ГГц → %s, 10 ГГц → %s, λ(2.45 ГГц) = %.3f см"
          % (len(IEEE_BANDS), ieee_band_for(2.45e9), ieee_band_for(10e9),
             freq_to_lambda(2.45e9) * 100))

    # --- ДІАПАЗОНИ РАДІОЛЮБИТЕЛІВ ---
    print("--- ТЕСТ РЛ: діапазони 160 м … 23 см (IARU Region 1) ---")
    ok &= len(HAM_BANDS) == 15
    ok &= ham_band_for(14.074e6) == "20 м" and ham_band_for(3.573e6) == "80 м"
    ok &= ham_band_for(145.5e6) == "2 м" and ham_band_for(433.5e6) == "70 см"
    ok &= ham_band_for(1.9e6) == "160 м" and ham_band_for(1296e6) == "23 см"
    ok &= ham_band_for(3.0e6) is None and ham_band_for(1000e6) is None
    ok &= abs(freq_to_lambda(14.074e6) - 21.301) < 0.01
    print("РЛ: %d діапазонів; 14.074 МГц → %s, 145.5 МГц → %s, λ(14.074 МГц) = %.3f м"
          % (len(HAM_BANDS), ham_band_for(14.074e6), ham_band_for(145.5e6),
             freq_to_lambda(14.074e6)))
    # --- МОДУЛЯЦІЯ І РЕБ ---
    print("--- ТЕСТ МОД/РЕБ: довідник модуляції та придушення ---")
    ok &= len(MOD_TYPES) == 3 and len(EW_METHODS) == 3
    ok &= MOD_TYPES[0][0].startswith("Амплітудна") and EW_METHODS[2][0].endswith("(Spoofing)")
    print("МОД/РЕБ: %d типи модуляції (АМ/ЧМ/ФМ), %d методи придушення (очік. 3/3)"
          % (len(MOD_TYPES), len(EW_METHODS)))
    # --- ІНСТРУКЦІЯ ---
    print("--- ТЕСТ ІНСТРУКЦІЯ: вбудований довідник ---")
    ok &= len(MANUAL_TEXT) > 2000 and "## 5. Генератор" in MANUAL_TEXT and "## 8." in MANUAL_TEXT
    _secs = [_l[3:] for _l in MANUAL_TEXT.split("\n") if _l.startswith("## ")]
    ok &= len(_secs) == len(set(_secs)) == 9
    print("ІНСТРУКЦІЯ: %d рядків, %d символів, розділів: %d (очік. 9)"
          % (MANUAL_TEXT.count("\n"), len(MANUAL_TEXT), MANUAL_TEXT.count("## ")))
    # --- ШІ-МОДУЛЬ ПІДБОРУ ЗАВАДИ ---
    print("--- ТЕСТ ШІ-ЗАВАДА: база знань і лінк-бюджет ---")
    ok &= len(AI_TARGETS) == 13 and all(t["cls"] in AI_J_S_REQ for t in AI_TARGETS.values())
    _r1 = ai_recommend_jam("GPS L1 C/A (ШПС)", 10.0, 10.0, 10.0, 61.44e6)
    _r2 = ai_recommend_jam("GPS L1 C/A (ШПС)", 22.0, 10.0, 10.0, 61.44e6)
    _r3 = ai_recommend_jam("GPS L1 C/A (ШПС)", 30.0, 10.0, 10.0, 61.44e6)
    ok &= _r1["kind"] == "jam_cw" and _r2["kind"] == "jam_chirp" and _r3["kind"] == "jam_noise"
    _rw = ai_recommend_jam("Wi-Fi 2.4 ГГц (20 МГц)", 33.0, 5.0, 5.0, 61.44e6)
    ok &= _rw["kind"] == "jam_noise" and abs(_rw["bw"] - 20e6) < 1e3
    _rs = ai_recommend_jam("GPS L1 C/A (ШПС)", 30.0, 10.0, 10.0, 61.44e6, want_spoof=True)
    ok &= _rs["kind"] == "spoof" and _rs["sp_drift"] == 10.0 and _rs["js_req"] == 6.0
    _rb = ai_recommend_jam("Вся смуга L1 (1559-1610 МГц)", 33.0, 10.0, 10.0, 61.44e6)
    ok &= _rb["kind"] == "jam_noise" and _rb["bw"] <= 0.4 * 61.44e6 + 1.0
    ok &= len(_rb["warnings"]) == 1 and _rb["margin"] > 10.0
    _rl = ai_recommend_jam("GPS L1 C/A (ШПС)", 20.0, 10.0, 100.0, 61.44e6)
    ok &= _rl["margin"] < 0 and _rl["d_max"] > 0 and _rl["p_req"] > 20.0
    ok &= abs(_r3["js"] - 43.6) < 0.2 and _r3["margin"] > 13.0
    print("ШІ: 8 цілей; ПП 10/22/30 дБм → CW/чирп/шум; OFDM → шум 20 МГц; спуфінг "
          "js_req 6; кепінг 0.4·fs + попередження; J/S(30 дБм, 10 км) = %.1f дБ; "
          "на 100 км брак %.1f дБ, d_max %.6g км, потрібен ПП %.1f дБм"
          % (_r3["js"], -_rl["margin"], _rl["d_max"], _rl["p_req"]))
    # --- СПЕКТР ІЗ ЗОБРАЖЕННЯ ---
    print("--- ТЕСТ СПЕКТР-ЗОБРАЖЕННЯ: оцифрування PNG ---")
    import tempfile
    _png = os.path.join(tempfile.gettempdir(), "rawst_specimg_test.png")
    _ff = np.linspace(0.0, 6000.0, 800)
    _dd = -100.0 + 2.0 * np.random.default_rng(42).standard_normal(800)
    for _f0, _amp, _w in ((1575.42, -40.0, 12.0), (2400.0, -55.0, 12.0), (5800.0, -65.0, 8.0)):
        _m = np.abs(_ff - _f0) < _w / 2.0
        _dd[_m] = np.maximum(_dd[_m], _amp)
    _img = np.zeros((480, 900, 3))
    _img[40:441, 70:881, :] = 0.08
    _img[[40, 440], 70:881, :] = 1.0
    _img[40:441, [70, 880], :] = 1.0
    for _i in range(800):
        _px = 70 + int(round(_i / 799.0 * 810.0))
        _py = int(round(440.0 - (_dd[_i] + 110.0) / 110.0 * 400.0))
        _img[max(0, _py - 1):min(480, _py + 2), _px, :] = (1.0, 0.9, 0.1)
    from matplotlib.image import imsave
    imsave(_png, np.clip(_img, 0.0, 1.0))
    _sx, _sy, _sm = image_to_spectrum(_png, 0.0, 6000.0, -110.0, 0.0)
    ok &= len(_sx) > 500 and _sm["mode"] == "color" and _sm["box"] == (72, 878, 42, 438)

    def _has(t, lim=60.0):
        return any(abs(p["freq"] - t) < lim for p in _pk)
    _pk = spectrum_peaks(_sx, _sy, prom_thr=6.0, min_sep_mhz=50.0)
    ok &= 3 <= len(_pk) <= 8
    ok &= _has(1575.42) and _has(2400.0) and _has(5800.0)
    _p24 = [p for p in _pk if abs(p["freq"] - 2400.0) < 60.0]
    ok &= bool(_p24) and 4.0 < _p24[0]["width"] < 40.0 and -62.0 < _p24[0]["db"] < -48.0
    os.remove(_png)
    print("СПЕКТР-ЗОБРАЖЕННЯ: %d колонок, траса %s, рамка %s; піків %d: %s; ширина 2400 МГц: %.1f МГц"
          % (_sm["columns"], _sm["mode"], _sm["box"], len(_pk),
             ", ".join("%.4g МГц" % p["freq"] for p in _pk[:4]),
             _p24[0]["width"] if _p24 else -1.0))
    print("--- ТЕСТ СПЕКТР-ЗОБРАЖЕННЯ 2: SDR-скріншот (кроп + біла траса) ---")
    _png2 = os.path.join(tempfile.gettempdir(), "rawst_specimg_sdr.png")
    _img2 = np.zeros((480, 900, 3))
    _img2[:200, :, :] = (0.05, 0.10, 0.18)
    _img2[:200, 100:850, :] = (0.15, 0.25, 0.45)
    _rng2 = np.random.default_rng(7)
    for _r2 in range(200):
        for _c2 in range(0, 900, 7):
            if _rng2.random() < 0.2:
                _img2[_r2, _c2] = (0.9, 0.8, 0.1)
    _img2[201:441, :, :] = (0.03, 0.05, 0.10)
    _img2[300, :, :] = (0.9, 0.1, 0.1)
    _img2[201:241, 380:520, :] = (0.95, 0.85, 0.1)
    _sx2 = np.linspace(0.0, 6000.0, 800)
    _sy2 = -100.0 + 2.0 * _rng2.standard_normal(800)
    for _f0, _amp in ((1200.0, -50.0), (3600.0, -60.0)):
        _m2 = np.abs(_sx2 - _f0) < 8.0
        _sy2[_m2] = np.maximum(_sy2[_m2], _amp)
    for _i2 in range(800):
        _px2 = 20 + int(round(_i2 / 799.0 * 860.0))
        _py2 = int(round(435.0 - (_sy2[_i2] + 110.0) / 110.0 * 210.0))
        _img2[max(0, _py2 - 1):min(441, _py2 + 2), _px2, :] = (0.95, 0.95, 0.95)
    imsave(_png2, np.clip(_img2, 0.0, 1.0))
    _s2x, _s2y, _s2m = image_to_spectrum(_png2, 0.0, 6000.0, -110.0, 0.0,
                                         crop=(0.0, 1.0, 0.42, 0.98), median=3)
    ok &= _s2m["mode"] == "lum"
    _pk2 = spectrum_peaks(_s2x, _s2y, prom_thr=6.0, min_sep_mhz=50.0)

    def _has2(t, lim=80.0):
        return any(abs(p["freq"] - t) < lim for p in _pk2)
    ok &= 2 <= len(_pk2) <= 6 and _has2(1200.0) and _has2(3600.0)
    os.remove(_png2)
    print("SDR-СКРІНШОТ: режим %s, піків %d: %s"
          % (_s2m["mode"], len(_pk2), ", ".join("%.4g МГц" % p["freq"] for p in _pk2[:4])))
    print("--- ТЕСТ ЗАВІДИ З ПІКА: тип за шириною (v3.9.0) ---")
    j1 = jam_from_peak(272.296070, -62.9, 0.012)
    assert j1["kind"] == "jam_cw" and abs(j1["f_hz"] - 272.29607e6) < 1.0
    j2 = jam_from_peak(1200.0, -40.0, 0.4)
    assert j2["kind"] == "jam_chirp" and abs(j2["dev_mhz"] - 0.4) < 1e-9 and j2["per_us"] == 1000.0
    j3 = jam_from_peak(2400.0, -30.0, 7.4)
    assert j3["kind"] == "jam_noise" and abs(j3["bw_mhz"] - 11.1) < 1e-9
    assert abs(j1["jam_req_dbm"] - (-47.9)) < 1e-9
    assert abs(j3["jam_req_dbm"] - (-15.0)) < 1e-9
    print("ЗАВІДА-З-ПІКА: 272.296 МГц → CW (потрібно −47.9 дБм при J/S+15); "
          "0.4 МГц → чирп ±0.4; 7.4 МГц → шум 11.1 МГц")
    print("--- ТЕСТ БЛОКОВИХ ЦІЛЕЙ BLADERF (v3.10.0) ---")
    _b = ai_recommend_jam("LoRa 450–490 МГц (суб-ГГц)", 30.0, 1.0, 0.01, 56e6)
    assert _b["kind"] == "jam_noise" and abs(_b["f_hz"] - 470e6) < 1.0
    _b2 = ai_recommend_jam("LTE 1900 МГц (Band 2/PCS)", 43.0, 2.0, 0.05, 56e6)
    assert _b2["kind"] == "jam_noise" and abs(_b2["bw"] - 20e6) < 1.0
    _b3 = ai_recommend_jam("Радіотелеметрія 2020–2100 МГц", 37.0, 1.0, 0.01, 56e6)
    assert _b3["kind"] == "jam_noise" and abs(_b3["f_hz"] - 2060e6) < 1.0
    print("БЛОКИ BLADERF: LoRa 470 (кап 22.4 МГц при fs 56), LTE 1900 (20 МГц без капа), ТЛМ 2060 → шум")
    print("--- ТЕСТ ТЕМИ GUI (v3.11.0) ---")
    _g = _theme_gradient("#0f1420", "#38bdf8", 5)
    assert len(_g) == 5 and _g[0] == "#0f1420" and _g[-1] == "#38bdf8"
    assert all(c.startswith("#") and len(c) == 7 for c in _g)
    assert {"bg", "panel", "panel2", "edge_hi", "edge_lo", "accent", "text", "dim"} <= set(THEME)
    print("ТЕМА: палітра %d кольорів, градієнт банера 5 стоп ок" % len(THEME))
    print("--- ТЕСТ ТЕМ/НИЧНОГО РЕЖИМУ (v3.12.0) ---")
    _save_theme_setting(False)
    assert _load_theme_setting() is False
    _save_theme_setting(True)
    assert _load_theme_setting() is True
    assert set(THEME_DARK) == set(THEME_LIGHT) and len(THEME_LIGHT) == 9
    print("ТЕМИ: збереження вибору ок; палітри techno/classic по %d кольорів" % len(THEME_LIGHT))
    print("--- ТЕСТ ТРЬОХ ТЕМ (v3.13.0) ---")
    for _n in ("techno", "classic", "day"):
        _save_theme_name(_n)
        assert _load_theme_name() == _n
    assert len(THEME_DAY) == 9 and set(THEME_DAY) == set(THEME_DARK)
    print("ТЕМИ x3: techno/classic/day збереження ок, палітра Дня %d кольорів" % len(THEME_DAY))
    print("--- ТЕСТ ЧОТИРЬОХ ТЕМ + АВТО (v3.14.0) ---")
    for _n in ("techno", "classic", "day", "amoled", "xp", "auto"):
        _save_theme_name(_n)
        assert _load_theme_name() == _n
    assert len(THEME_AMOLED) == 9 and set(THEME_AMOLED) == set(THEME_DARK)
    assert _resolve_theme("techno") == "techno" and _resolve_theme("day") == "day"
    assert _resolve_theme("auto") in ("day", "techno")
    print("ТЕМИ x4 + авто: збереження ок, AMOLED %d кольорів" % len(THEME_AMOLED))
    print("--- ТЕСТ СПЕКТРОГРАМИ (v3.17.0) ---")
    import tempfile
    _spgf = os.path.join(tempfile.gettempdir(), "rs317_spg.raw")
    _tspg = np.arange(65536) / 1e6
    _sig = 32768 + 12000 * np.sin(2 * np.pi * 100e3 * _tspg)
    save_raw(_spgf, np.clip(_sig, 0, 65535).astype("<u2"))
    _fr, _tm, _db, _nt = raw_spectrogram(_spgf, 1e6, nch=1, ch=0, nfft=1024, overlap=0.5)
    assert _db.shape[0] == 513 and _db.shape[1] >= 100
    _pk = _fr[int(np.argmax(_db.mean(axis=1)))]
    assert abs(_pk - 100e3) < 2e3
    assert _tm[0] >= 0 and np.all(np.diff(_tm) > 0)
    os.remove(_spgf)
    print("СПГ v3.17.0: спектрограма ок — %s, пік на %.4g кГц (очік. 100)" % (_nt, _pk / 1e3))

    print("--- ТЕСТ ЗАВАДИ З IQ (v3.19.0) ---")
    import tempfile
    _iqf = os.path.join(tempfile.gettempdir(), "rs319.iq")
    _fsq = 2e6
    _tq = np.arange(262144) / _fsq
    (0.4 * np.exp(2j * np.pi * 200e3 * _tq)).astype("complex64").tofile(_iqf)
    _r = iq_jam_analyze(_iqf, dtype="float32", fs_iq=_fsq, fc=100e6, js_target=15.0)
    assert abs(_r["f_peak_hz"] - 100.2e6) < 4e3
    assert _r["kind"] and _r["kind_name"]
    assert abs(_r["f_hz"] / 1e6 - 100.2) < 0.004
    assert abs(_r["jam_req_dbm"] - (_r["sig_dbm"] + 15.0)) < 1e-9
    assert _r["dyn_db"] > 20.0
    os.remove(_iqf)
    print("IQ-ЗАВАДА v3.19.0: пік %.4f МГц (очік. 100.2), динаміка %.1f дБ, J/S +15 → ≈ %.1f дБм"
          % (_r["f_peak_hz"] / 1e6, _r["dyn_db"], _r["jam_req_dbm"]))

    print("--- ТЕСТ КАЛІБРУВАЛЬНОГО ОФСЕТУ IQ (v3.20.0) ---")
    import tempfile
    _iqf2 = os.path.join(tempfile.gettempdir(), "rs320.iq")
    _tq2 = np.arange(262144) / 2e6
    (0.4 * np.exp(2j * np.pi * 200e3 * _tq2)).astype("complex64").tofile(_iqf2)
    _r0 = iq_jam_analyze(_iqf2, dtype="float32", fs_iq=2e6, fc=100e6, js_target=15.0)
    _r1 = iq_jam_analyze(_iqf2, dtype="float32", fs_iq=2e6, fc=100e6, js_target=15.0, db_off=12.5)
    assert abs(_r0["sig_dbm"] - _r0["dbfs"]) < 1e-9
    assert abs(_r1["sig_dbm"] - (_r1["dbfs"] + 12.5)) < 1e-9
    assert abs(_r1["jam_req_dbm"] - (_r1["sig_dbm"] + 15.0)) < 1e-9
    assert abs(_r1["dbfs"] - _r0["dbfs"]) < 1e-9
    os.remove(_iqf2)
    print("IQ-ОФСЕТ v3.20.0: dBFS %.1f + 12.5 дБ → %.1f дБм, J/S +15 → ≈ %.1f дБм"
          % (_r1["dbfs"], _r1["sig_dbm"], _r1["jam_req_dbm"]))

    print("--- ТЕСТ ПЕРСИСТЕНЦІЇ ОФСЕТА В .achx (v3.21.0) ---")
    _pth21 = "/tmp/opencode/st21.achx"
    save_achx_profile(_pth21, [(100e6, 0.0), (1e9, -3.0), (3e9, -6.0)])
    with open(_pth21, "a", encoding="utf-8") as _fh21:
        _fh21.write("# theme: techno\n")
        _fh21.write("# iq_off: 7.5\n")
    _p21, _mb21, _eb21 = load_achx_profile(_pth21)
    print("  .achx з # iq_off: %d точок (очік. 3)" % len(_p21))
    ok &= len(_p21) == 3
    print("ОФСЕТ-ПЕРСИСТ v3.21.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ТАБЛИЦІ ЯКІСНОЇ ЗАВАДИ (v3.22.0) ---")
    _sd_a, _sd_b = sorted(SDR_PRESETS)[0], sorted(SDR_PRESETS)[-1]
    _q1 = jam_quality_params("4G (LTE)", _sd_a)
    _q2 = jam_quality_params("GPS L1", _sd_a)
    _q3 = jam_quality_params("4G (LTE)", _sd_b)
    print("  LTE x %s: %s, смуга %.4g МГц, J/S %.0f дБ" % (_sd_a, _q1["kind_name"], _q1["bw_hz"] / 1e6, _q1["js_req"]))
    print("  GPS L1: %s, %.4g МГц" % (_q2["kind_name"], _q2["f_hz"] / 1e6))
    ok &= (len(_q1["rows"]) >= 8 and _q1["kind"] == "jam_noise" and _q1["bw_hz"] > 1e6
           and abs(_q2["f_hz"] - 1575.42e6) < 1e3 and _q1["fs_hz"] != _q3["fs_hz"]
           and jam_quality_params("4G (LTE)", _sd_a)["p_req"] is not None)
    print("ЯКІСНА-ЗАВАДА v3.22.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ РАДІУСА ПРИДУШЕННЯ І ПОТУЖНОСТІ SDR (v3.23.0) ---")
    _qr = jam_quality_params("4G (LTE)", "BladeRF 2.0 micro xA4")
    _qrx = jam_quality_params("4G (LTE)", "RTL-SDR Blog V4")
    _n_tx = sum(1 for _v in SDR_PRESETS.values() if isinstance(_v.get("tx_dbm"), (int, float)))
    _n_rx = sum(1 for _v in SDR_PRESETS.values() if _v.get("tx_dbm") is None)
    print("  BladeRF xA4: вихід %.1f дБм → радіус %.4g км" % (_qr["tx_dbm"], _qr["sdr_d_max"]))
    print("  плат із TX: %d, приймальних: %d; RTL-SDR: радіус —, алерт: %s"
          % (_n_tx, _n_rx, "є" if "приймальн" in _qrx["warn"] else "немає"))
    ok &= (_qr["tx_dbm"] is not None and _qr["sdr_d_max"] is not None and _qr["sdr_d_max"] > 0
           and _n_tx >= 12 and _n_rx >= 4 and _qrx["sdr_d_max"] is None
           and "приймальн" in _qrx["warn"] and len(_qr["rows"]) >= 12)
    print("РАДІУС-ПРИДУШЕННЯ v3.23.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ТРАКТУ: ПІДСИЛЮВАЧ, АНТЕНА, КСВ (v3.24.0) ---")
    _q0 = jam_quality_params("4G (LTE)", "BladeRF 2.0 micro xA4")
    _qg = jam_quality_params("4G (LTE)", "BladeRF 2.0 micro xA4",
                             amp_db=20.0, ant_gain_dbi=14.0,
                             vswr=[(1900e6, 1.5)], ant_rng=(690e6, 3800e6))
    _q5 = jam_quality_params("4G (LTE)", "BladeRF 2.0 micro xA4", pa_dbm=5.0)
    print("  КСВ 2.0 → втрати %.2f дБ; радіус %.4g → %.4g км (тракт %+.1f дБ)"
          % (_vswr_loss_db(2.0), _q0["sdr_d_max"], _qg["sdr_d_max"], _qg["g_total"]))
    ok &= (_qg["sdr_d_max"] > _q0["sdr_d_max"] and _qg["g_total"] > 30.0
           and abs(_vswr_loss_db(1.0)) < 1e-9 and abs(_vswr_loss_db(2.0) - 0.5115) < 0.01
           and _qg["rows"][5][2] == "#2e7d32" and _q5["rows"][5][2] == THEME["warn"]
           and abs(_q5["p_req"] - _q0["p_req"]) < 1e-9)
    print("ТРАКТ v3.24.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ДІАГРАМИ СПРЯМОВАНОСТІ (v4.0.0) ---")
    print("  Панель секторна: 0°=%.1f дБ, 90°=%.1f дБ, 180°=%.1f дБ"
          % (ant_pattern_db("Панель секторна", 0.0), ant_pattern_db("Панель секторна", 90.0),
             ant_pattern_db("Панель секторна", 180.0)))
    ok &= (abs(ant_pattern_db("Панель секторна", 0.0)) < 1e-9
           and ant_pattern_db("Панель секторна", 90.0) <= -12.0
           and ant_pattern_db("Панель секторна", 180.0) == -25.0
           and ant_pattern_db("Диполь", 0.0) == 0.0 and abs(ant_pattern_db("Диполь", 90.0)) < 1e-9
           and ant_pattern_db("Рупорна", 30.0) < ant_pattern_db("Логоперіодична (LPDA)", 30.0))
    print("ДІАГРАМА v4.0.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ВЛАСНОЇ АНТЕНИ ТЕТА3/ЗАДНІЙ ПРОМІНЬ (v4.1.0) ---")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure as _FigY
    _fy = _FigY(figsize=(6, 4), dpi=100)
    _ay = _fy.add_subplot(111, projection="polar")
    draw_ant_pattern(_ay, "Власна (тест)", 5.0, hpb=30.0, fbr=12.0)
    ok &= (ant_pattern_db("Власна (тест)", 0.0, hpb=30.0, fbr=12.0) == 0.0
           and abs(ant_pattern_db("Власна (тест)", 90.0, hpb=30.0, fbr=12.0) + 12.0) < 1e-9
           and ant_pattern_db("Власна (тест)", 180.0, hpb=30.0, fbr=12.0) == -12.0
           and ant_pattern_db("Панель секторна", 180.0) == -25.0
           and len(_ay.lines) >= 1)
    print("ВЛАСНА v4.1.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ШКАЛИ КСВ-ВТРАТ (v4.2.0) ---")
    _sc42 = ant_vswr_scale([(2e9, 2.0), (4e9, 1.9)])
    _sc42b = ant_vswr_scale([])
    _sc42c = ant_vswr_scale([(3e9, 1.5)])
    ok &= (_sc42 is not None and abs(_sc42[0] - 2e9) < 1.0 and abs(_sc42[1] - 4e9) < 1.0
           and len(_sc42[2]) == 37
           and abs(_sc42[2][0][1] - _vswr_loss_db(2.0)) < 1e-9
           and _sc42b is None and _sc42c is not None)
    print("КСВ-ШКАЛА v4.2.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ПЕРЕМИКАЧА ШКАЛИ КСВ (v4.3.0) ---")
    class _V43:
        def get(self):
            return True
    ok &= (ant_scale_hidden(None) is False
           and ant_scale_hidden(_V43()) is True
           and ant_scale_hidden(False) is False)
    print("ПЕРЕМИКАЧ v4.3.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ЕКСПОРТУ ДІАГРАМИ У PNG (v4.4.0) ---")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure as _FigZ
    import tempfile
    _fz = _FigZ(figsize=(6.6, 5.4), dpi=100)
    _az = _fz.add_subplot(111, projection="polar")
    draw_ant_pattern(_az, "Панель секторна", 11.0)
    _fz.tight_layout()
    _pngz = os.path.join(tempfile.gettempdir(), "rs440_pattern.png")
    _fz.savefig(_pngz, dpi=150, facecolor=_fz.get_facecolor())
    ok &= (os.path.exists(_pngz) and os.path.getsize(_pngz) > 8000)
    os.remove(_pngz)
    print("ДІАГРАМА-PNG v4.4.0:", "PASS" if ok else "FAIL")
    print("--- ТЕСТ ЕКСПОРТУ СПЕКТРОГРАМИ У PNG (v3.18.0) ---")
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure as _FigX
    _fx = _FigX(figsize=(4, 3), dpi=100)
    _axx = _fx.add_subplot(111)
    _imx = _axx.imshow(_db, aspect="auto", cmap="viridis")
    _fx.colorbar(_imx, ax=_axx)
    _pngx = os.path.join(tempfile.gettempdir(), "rs318_spg.png")
    _fx.savefig(_pngx, dpi=150, facecolor=_fx.get_facecolor())
    assert os.path.exists(_pngx) and os.path.getsize(_pngx) > 5000
    _szx = os.path.getsize(_pngx)
    os.remove(_pngx)
    print("СПГ-PNG v3.18.0: експорт ок — PNG збережено, %d байт" % _szx)

    print("--- ТЕСТ ТЕМИ WINDOWS XP (v3.16.0) ---")
    _save_theme_name("xp")
    assert _load_theme_name() == "xp"
    assert len(THEME_XP) == 9 and set(THEME_XP) == set(THEME_DARK)
    assert _resolve_theme("xp") == "xp"
    print("ТЕМИ v3.16.0: XP збереження ок, палітра XP %d кольорів" % len(THEME_XP))

    print("--- ТЕСТ v3.15.0: іконка, живе авто, тема в .achx ---")
    import tempfile
    _ic = _theme_icon_img()
    assert _ic is not None and _ic.size == (64, 64)
    _tf = os.path.join(tempfile.gettempdir(), "rs315_theme.achx")
    with open(_tf, "w", encoding="utf-8") as _f:
        _f.write("# interpolation: spline\n# theme: day\n100 -3\n1000 -3\n10000 -6\n")
    _p2, _m2, _e2 = load_achx_profile(_tf)
    assert _m2 == "spline" and len(_p2) == 3
    os.remove(_tf)
    print("ТЕМА v3.15.0: іконка 64x64 ок, .achx тема парситься (spline, 3 точки)")

    print("--- ТЕСТ ТОП-N ПІКІВ (v4.9.0) ---")
    _iqf9 = os.path.join(tempfile.gettempdir(), "rs490.iq")
    _fsq9, _nq9 = 8e6, 262144
    _tq9 = np.arange(_nq9) / _fsq9
    _sg9 = (0.4 * np.exp(2j * np.pi * (-1.5e6) * _tq9)
            + 0.3 * np.exp(2j * np.pi * 0.6e6 * _tq9)
            + 0.2 * np.exp(2j * np.pi * 2.2e6 * _tq9))
    _sg9 = _sg9 + 0.01 * (np.random.default_rng(9).normal(size=_nq9)
                          + 1j * np.random.default_rng(10).normal(size=_nq9))
    _sg9.astype("complex64").tofile(_iqf9)
    _r9 = iq_jam_analyze(_iqf9, dtype="float32", fs_iq=_fsq9, fc=2400e6)
    _pf9 = [p["f_hz"] for p in _r9["peaks"]]
    _oka9 = (len(_r9["peaks"]) >= 3
             and any(abs(f - 2398.5e6) < 20e3 for f in _pf9)
             and any(abs(f - 2400.6e6) < 20e3 for f in _pf9)
             and any(abs(f - 2402.2e6) < 20e3 for f in _pf9))
    os.remove(_iqf9)
    print("IQ-ПІКИ v4.9.0:", "PASS" if _oka9 else "FAIL",
          "- %d піків: %s" % (len(_pf9), ", ".join("%+.3f МГц" % ((f - 2400e6) / 1e6)
                                                  for f in _pf9[:5])))
    ok &= _oka9

    print("--- ТЕСТ СПЕКТРАЛЬНО-ФОРМОВАНОГО ШУМУ (v4.9.0) ---")
    _pr9 = {"freqs_hz": [2390e6, 2399e6, 2401e6, 2410e6],
            "psd_db": [-80.0, 0.0, 0.0, -80.0], "fc": 2400e6}
    _w9 = make_jam("jam_noise", 1 << 20, 12e9, 2400e6, 1.0, 0.0,
                   np.random.default_rng(11), bw_hz=4e6, shape=_pr9)
    _nf9 = 65536
    _s9 = _w9[: ((1 << 20) // _nf9) * _nf9].reshape(-1, _nf9)
    _p9 = (np.abs(np.fft.fft(_s9 * np.hanning(_nf9), axis=1)) ** 2).mean(axis=0)
    _f9 = np.fft.fftfreq(_nf9, 1.0 / 12e9)
    _in9 = _p9[np.abs(_f9 - 2400e6) < 0.9e6]
    _out9 = _p9[(np.abs(_f9 - 2400e6) > 12e6) & (np.abs(_f9 - 2400e6) < 14e6)]
    _c9 = 10.0 * np.log10(float(_in9.mean()) / max(float(_out9.mean()), 1e-300))
    _okb9 = bool(_c9 > 25.0 and 0.5 < float(np.std(_w9)) < 2.0)
    print("ШУМ-ПРОФІЛЬ v4.9.0:", "PASS" if _okb9 else "FAIL",
          "- контраст %.1f дБ, rms %.3g" % (_c9, float(np.std(_w9))))
    ok &= _okb9

    print("--- ТЕСТ ПРОФІЛЮ .spectrum.json (v4.9.0) ---")
    _pp9 = os.path.join(tempfile.gettempdir(), "rs490prof.json")
    save_spectrum_profile(_pp9, [2390e6, 2399e6, 2401e6, 2410e6],
                          [-80.0, 0.0, 0.0, -80.0], 2400e6)
    _ld9 = load_spectrum_profile(_pp9)
    _okc9 = (abs(_ld9["fc"] - 2400e6) < 1.0 and _ld9["freqs_hz"].size == 4
             and abs(_ld9["psd_db"][1]) < 1e-9)
    os.remove(_pp9)
    print("ПРОФІЛЬ v4.9.0:", "PASS" if _okc9 else "FAIL",
          "- fc %g Гц, %d точок" % (_ld9["fc"], _ld9["freqs_hz"].size))
    ok &= _okc9

    print("--- ТЕСТ RGPO (v5.0.0) ---")
    _n50, _v50 = 2_400_000, 5.0e5
    _w50 = make_rgpo(_n50, 12e9, 2e9, 1.0, 0.0, np.random.default_rng(21),
                     per_us=50.0, pw_us=10.0, v_pull=_v50, d_max_us=50.0)
    _idx50 = np.where(np.abs(_w50) > 0.5)[0]
    _br50 = np.where(np.diff(_idx50) > 5000)[0]
    _cl50 = np.split(_idx50, _br50 + 1)
    _ok50 = len(_cl50) == 4
    for _k50, _cl50i in enumerate(_cl50):
        _exp50 = _k50 * 50e-6 * (1.0 + 2.0 * _v50 / 3e8) * 12e9
        _ok50 = _ok50 and bool(abs(_cl50i[0] - _exp50) < 0.2e-6 * 12e9)
    _w50b = make_rgpo(1 << 22, 12e9, 2e9, 1.0, 0.0, np.random.default_rng(21),
                      per_us=50.0, pw_us=10.0, v_pull=_v50, d_max_us=1.0)
    _idx50b = np.where(np.abs(_w50b) > 0.5)[0]
    _br50b = np.where(np.diff(_idx50b) > 5000)[0]
    _ok50 = _ok50 and len(np.split(_idx50b, _br50b + 1)) == 7
    del _w50, _w50b
    print("RGPO v5.0.0:", "PASS" if _ok50 else "FAIL",
          "- 4 репліки з кроком 2v/c, зникнення після d_max (7 імпульсів)")
    ok &= _ok50

    print("--- ТЕСТ VGPO (v5.0.0) ---")
    _w50v = make_vgpo(1 << 23, 5e9, 2e9, 1.0, 0.0, np.random.default_rng(22),
                      fd_max_khz=200.0, t_pull_s=1.0e-3)
    _nb50 = 1 << 19
    _fa50 = np.argmax(np.abs(np.fft.rfft(_w50v[1000000:1000000 + _nb50]
                                         * np.hanning(_nb50)))) * (5e9 / _nb50)
    _fb50 = np.argmax(np.abs(np.fft.rfft(_w50v[4000000:4000000 + _nb50]
                                         * np.hanning(_nb50)))) * (5e9 / _nb50)
    _ok50v = (abs(_fa50 - 2.00005e9) < 25e3 and abs(_fb50 - 2.00017e9) < 25e3
              and _w50v[6000000] == 0.0 and _w50v[1000000] != 0.0)
    del _w50v
    print("VGPO v5.0.0:", "PASS" if _ok50v else "FAIL",
          "- піки %.6f/%.6f ГГц (очік. 2.000050/2.000170), drop після T" % (_fa50 / 1e9, _fb50 / 1e9))
    ok &= _ok50v

    print("--- ТЕСТ ЗАХИЩЕНИХ ЧАСТОТ (v5.0.0) ---")
    _pr50 = parse_protected_mhz("1575.42, 2412-2472")
    _ok50p = (len(_pr50) == 2 and is_protected(1575.42e6, _pr50)
              and is_protected(2432e6, _pr50) and not is_protected(2500e6, _pr50)
              and not is_protected(1578e6, _pr50)
              and "Завада RGPO" in KIND_KEY and "Завада VGPO" in KIND_KEY)
    print("ЗАХИЩЕНІ v5.0.0:", "PASS" if _ok50p else "FAIL",
          "- %d діапазони; KIND_KEY містить RGPO/VGPO" % len(_pr50))
    ok &= _ok50p

    # --- ТЕСТ СПИСКУ ФОРМ (v5.1.0) ---
    _ok51 = ("Завада RGPO" in WAVE_KINDS) and ("Завада VGPO" in WAVE_KINDS)
    print("ФОРМИ v5.1.0: %s" % ("PASS" if _ok51 else "FAIL"))
    ok = ok and _ok51

    # --- ТЕСТ МЕТРИКИ ПОКРИТТЯ (v5.1.0) ---
    _m51 = coverage_metric([(2.39e9, 4e6, 12000.0), (2.391e9, 4e6, 10000.0)], ref_db=0.0)
    _m52 = coverage_metric([(2.39e9, 2e6, 100.0), (2.41e9, 2e6, 100.0)], ref_db=0.0)
    _ok52 = (abs(_m51["pct"] - 100.0) < 1e-9 and not _m51["holes"]
             and abs(_m51["js"][0] - 20.0 * math.log10(12000.0)) < 0.1
             and _m52["pct"] < 19.0 and len(_m52["holes"]) == 1
             and abs(_m52["holes"][0][0] - 2.391e9) < 1e3
             and abs(_m52["holes"][0][1] - 2.409e9) < 1e3)
    print("МЕТРИКА v5.1.0: %s (pct %.1f%%/%.1f%%, діри %d/%d)"
          % ("PASS" if _ok52 else "FAIL", _m51["pct"], _m52["pct"],
             len(_m51["holes"]), len(_m52["holes"])))
    ok = ok and _ok52

    # --- ТЕСТ МАПИ ПОКРИТТЯ (v5.1.0) ---
    _dg51, _rs51 = coverage_map(50.0, -85.0, 2.4e9, ant_name="", hpb=60.0, fbr=20.0, n_pts=37)
    _r051 = coverage_radius_km(50.0, -85.0, 2.4e9, g_ant_db=0.0)
    _ok53 = (len(_dg51) == 37 and len(_rs51) == 37
             and abs(float(_rs51[0]) - _r051) < max(1e-6 * _r051, 1e-9)
             and abs(fspl_free_db(2.4e9, _r051) - 135.0) < 1e-6
             and 0.0 < float(_rs51.min()) < float(_rs51[0]))
    print("МАПА v5.1.0: %s (r0 %.1f км, мін %.1f км)"
          % ("PASS" if _ok53 else "FAIL", _r051, float(_rs51.min())))
    ok = ok and _ok53

    # --- ТЕСТ ЗВІТУ PDF (v5.1.0) ---
    _pdf51 = "/tmp/opencode/rs510_report.pdf"
    _png51 = "/tmp/opencode/rs510_spec.png"
    try:
        from matplotlib.figure import Figure as _Fig51
        _fg51 = _Fig51(figsize=(4.0, 2.0), dpi=80)
        _ax51 = _fg51.add_subplot(111)
        _ax51.plot([2390.0, 2400.0, 3000.0], [12000.0, 10000.0, 5000.0], "v")
        _fg51.savefig(_png51)
    except Exception:
        _png51 = None
    build_pdf_report(_pdf51,
                     [{"enabled": True, "kind": "Завада (шум)", "freq": 2.4e9,
                       "phase": 0.0, "amp": 12000.0, "width_us": 2.0,
                       "dc": 32768.0, "gain": 1.0},
                      {"enabled": False, "kind": "Синус", "freq": 852e6,
                       "phase": 0.0, "amp": 10000.0, "width_us": 2.0,
                       "dc": 32768.0, "gain": 1.0}],
                     metric=_m51, spec_png=_png51)
    from pypdf import PdfReader as _PR51
    _np51 = len(_PR51(_pdf51).pages)
    _ok54 = os.path.exists(_pdf51) and os.path.getsize(_pdf51) > 3000 and _np51 >= 1
    print("ЗВІТ-PDF v5.1.0: %s (%d стр., %.0f КБ)"
          % ("PASS" if _ok54 else "FAIL", _np51, os.path.getsize(_pdf51) / 1024.0))
    ok = ok and _ok54

    # --- ТЕСТ СЦЕНАРІЇВ + CLI (v5.1.0) ---
    _ph51 = scenario_parse_phases([{"t": 2.0, "ch": "0,1"}, {"t": 1.0, "ch": [0]}, "3:1"])
    _par51 = [{"enabled": True, "kind": "Синус", "freq": 2400e6, "phase": 0.0,
               "amp": 10000.0, "width_us": 0.0, "dc": 32768.0, "gain": 1.0},
              {"enabled": True, "kind": "Синус", "freq": 850e6, "phase": 0.0,
               "amp": 8000.0, "width_us": 0.0, "dc": 32768.0, "gain": 1.0}]
    _scp51 = "/tmp/opencode/rs510_scen.raw"
    _out51, _tot51 = scenario_build_raw(
        _scp51, [{"t": 1.0e-3, "ch": [0, 1]}, {"t": 5.0e-4, "ch": [0]}], _par51, 8e6, seed=7)
    _arr51 = np.frombuffer(open(_scp51, "rb").read(), dtype="<u2")
    _s151 = _arr51[:8000].astype(np.float64)
    _s251 = _arr51[8000:].astype(np.float64)
    _rms151 = float((_s151 - _s151.mean()).std())
    _rms251 = float((_s251 - _s251.mean()).std())
    import subprocess as _sp51
    import sys as _sy51
    _sj51 = "/tmp/opencode/rs510_cli.json"
    with open(_sj51, "w", encoding="utf-8") as _fh51:
        json.dump({"fs_hz": 8e6, "seed": 7, "params": _par51,
                   "phases": [{"t": 1.0e-3, "ch": [0, 1]},
                              {"t": 5.0e-4, "ch": [0]}]}, _fh51)
    if getattr(_sy51, "frozen", False):
        _cmd51 = [_sy51.executable, "--scenario", _sj51,
                  "/tmp/opencode/rs510_cli.raw"]
    else:
        _cmd51 = [_sy51.executable, "-B", os.path.abspath(__file__),
                  "--scenario", _sj51, "/tmp/opencode/rs510_cli.raw"]
    _pr51 = _sp51.run(_cmd51, capture_output=True, text=True)
    _ok55 = (len(_ph51) == 3 and _ph51[0]["ch"] == [0, 1] and _ph51[2]["ch"] == [1]
             and abs(_ph51[2]["t"] - 3.0) < 1e-9
             and _tot51 == 12000 and _arr51.size == 12000
             and _rms251 < _rms151 * 0.9
             and _pr51.returncode == 0
             and os.path.exists("/tmp/opencode/rs510_cli.raw")
             and os.path.getsize("/tmp/opencode/rs510_cli.raw") == 24000)
    print("СЦЕНАРІЙ v5.1.0: %s (3 фази, %d відл., CLI rc=%d)"
          % ("PASS" if _ok55 else "FAIL", _tot51, _pr51.returncode))
    ok = ok and _ok55

    # --- ТЕСТ 3D-СПЕКТОГРАМИ (v5.2.0) ---
    _raw52 = "/tmp/opencode/rs520_spec.raw"
    _t52 = np.arange(16384) / 8e6
    save_raw(_raw52, (np.sin(2 * np.pi * 500e3 * _t52) * 8000 + 32768).astype("<u2"))
    _f52, _tm52, _db52, _note52 = raw_spectrogram(_raw52, 8e6, nch=1, ch=0, nfft=1024, overlap=0.5)
    _X52, _Y52 = np.meshgrid(_tm52, _f52 / 1e6)
    _ok520 = (_db52.shape[0] == 513 and _X52.shape == _db52.shape
              and bool(np.isfinite(_db52).all()))
    print("3D-СПЕКТРО v5.2.0: %s (%s)" % ("PASS" if _ok520 else "FAIL", _db52.shape))
    ok = ok and _ok520

    # --- ТЕСТ ПЕРЕКРИТТЯ ЗАВАДА/ЦІЛЬ (v5.2.0) ---
    _o1 = jam_overlap(2390e6, 2410e6, 2400e6, 20e6)
    _o2 = jam_overlap(2390e6, 2410e6, 2500e6, 20e6)
    _o3 = jam_overlap(2390e6, 2410e6, 2405e6, 10e6)
    _ok521 = (abs(_o1["pct"] - 100.0) < 1e-9 and abs(_o2["pct"]) < 1e-9
              and _o2["f_lo"] is None and abs(_o3["pct"] - 50.0) < 1e-6
              and _o3["f_lo"] is not None)
    print("ПЕРЕКРИТТЯ v5.2.0: %s (pct %.0f/%.0f/%.0f%%)"
          % ("PASS" if _ok521 else "FAIL", _o1["pct"], _o2["pct"], _o3["pct"]))
    ok = ok and _ok521

    # --- ТЕСТ ПОРІВНЯННЯ A/B (v5.2.0) ---
    _rg52 = np.random.default_rng(5)
    _ta52 = np.arange(65536) / 8e6
    _nz52 = _rg52.normal(0, 250, 65536)
    _abA = "/tmp/opencode/rs520_a.raw"
    _abB = "/tmp/opencode/rs520_b.raw"
    save_raw(_abA, (np.clip(np.sin(2 * np.pi * 600e3 * _ta52) * 9000 + _nz52 + 32768,
                            0, 65535)).astype("<u2"))
    save_raw(_abB, (np.clip(_nz52 + 32768, 0, 65535)).astype("<u2"))
    _ab52 = ab_compare(_abA, _abB, 8e6, nfft=4096)
    _ok522 = (_ab52["suppress_db"] > 20.0 and abs(_ab52["f_hz"] - 600e3) < 4e3
              and _ab52["bw_hz"] > 0)
    print("A/B v5.2.0: %s (пік %.3f МГц, придушення %+.1f дБ)"
          % ("PASS" if _ok522 else "FAIL", _ab52["f_hz"] / 1e6, _ab52["suppress_db"]))
    ok = ok and _ok522

    # --- ТЕСТ PRI-АНАЛІЗУ (v5.2.0) ---
    _xp52 = _rg52.normal(0, 50, 48000)
    for _k52 in range(240):
        _s052 = int(_k52 * 200)
        _xp52[_s052:_s052 + 40] += 12000.0
    _pr52 = pri_analyze(_xp52, 8e6)
    _ok523 = (abs(_pr52["pri_us"] - 25.0) < 1.0 and abs(_pr52["pw_us"] - 5.0) < 1.5
              and _pr52["n_pulses"] >= 200)
    print("PRI v5.2.0: %s (PRI %.2f мкс, PW %.2f мкс, %d імп.)"
          % ("PASS" if _ok523 else "FAIL", _pr52["pri_us"], _pr52["pw_us"], _pr52["n_pulses"]))
    ok = ok and _ok523

    # --- ТЕСТ ДЕМОДУЛЯЦІЇ ЧМ (v5.2.0) ---
    _z52 = np.exp(1j * (0.4 * np.sin(2 * np.pi * 10e3 * _t52)))
    _a52 = demod_audio(_z52, 8e6, mode="fm", fm_dev_hz=8e3, decim=1)
    _fa52 = np.fft.rfftfreq(_a52.size, 1.0 / 8e6)
    _sp52 = np.abs(np.fft.rfft(_a52.astype(np.float64)))
    _pk52 = int(np.argmax(_sp52[1:])) + 1
    _w52p = "/tmp/opencode/rs520_demod.wav"
    save_wav_int16(_w52p, _a52, 8000)
    _ok524 = (abs(_fa52[_pk52] - 10e3) < 500.0
              and os.path.getsize(_w52p) == 44 + 2 * _a52.size
              and _a52.dtype == np.int16)
    print("ДЕМОД v5.2.0: %s (пік %.2f кГц, WAV %.0f КБ)"
          % ("PASS" if _ok524 else "FAIL", _fa52[_pk52] / 1e3, os.path.getsize(_w52p) / 1024.0))
    ok = ok and _ok524

    # --- ТЕСТ МОБІЛЬНИХ ФАЗ (v5.2.0): доплер + дальність ---
    _par52m = [{"enabled": True, "kind": "Синус", "freq": 3e6, "phase": 0.0,
                "amp": 9000.0, "width_us": 0.0, "dc": 32768.0, "gain": 1.0}]
    _rm1 = "/tmp/opencode/rs520_m1.raw"
    _rm2 = "/tmp/opencode/rs520_m2.raw"
    scenario_build_raw(_rm1, [{"t": 1.0e-3, "ch": [0], "v": 0.0, "d": 0.0}], _par52m, 8e6, seed=3)
    scenario_build_raw(_rm2, [{"t": 1.0e-3, "ch": [0], "v": 2e6, "d": 4.0}], _par52m, 8e6, seed=3)
    _c152 = _chan_column(_rm1)
    _c252 = _chan_column(_rm2)
    _fr1, _ap1 = amplitude_spectrum(_c152, 8e6, nfft=8192)
    _fr2, _ap2 = amplitude_spectrum(_c252, 8e6, nfft=8192)
    _sh52 = float(_fr2[int(np.argmax(_ap2))] - _fr1[int(np.argmax(_ap1))])
    _r152 = float((_c152 - _c152.mean()).std())
    _r252 = float((_c252 - _c252.mean()).std())
    _ok525 = (abs(_sh52 - 20e3) < 1.2e3 and abs(_r252 * 4.0 - _r152) < _r152 * 0.1)
    print("МОБІЛЬНІ-ФАЗИ v5.2.0: %s (зсув %+.0f Гц, rms 1/4: %.3f)"
          % ("PASS" if _ok525 else "FAIL", _sh52, _r252 * 4.0 / _r152))
    ok = ok and _ok525

    # --- ТЕСТ ОПТИМІЗАТОРА ПОТУЖНОСТІ (v5.2.0) ---
    _po52 = power_optimize(15.0, -85.0, 2.4e9, 10.0, 0.0, 0.0, 20e6, 1e6)
    _fpl52 = 92.45 + 20.0 * math.log10(2.4) + 20.0 * math.log10(10.0)
    _ok526 = (abs(_po52["p_req_dbm"] - (15.0 - 85.0 + _fpl52 + 10.0 * math.log10(20.0))) < 0.05
              and abs(_po52["eirp_dbm"] - _po52["p_req_dbm"]) < 1e-9
              and abs(_po52["fspl_db"] - _fpl52) < 1e-6)
    print("ОПТИМІЗАТОР v5.2.0: %s (ПП %.1f дБм, FSPL %.1f дБ)"
          % ("PASS" if _ok526 else "FAIL", _po52["p_req_dbm"], _po52["fspl_db"]))
    ok = ok and _ok526

    # --- ТЕСТ ГОТОВИХ СЦЕНАРІЇВ (v5.2.0) ---
    _ok527 = len(SCENARIO_PRESETS) >= 4
    for _sp52 in SCENARIO_PRESETS:
        _php52 = scenario_parse_phases(_sp52["phases"])
        if not _php52 or any(("v" not in _one or "d" not in _one) for _one in _php52):
            _ok527 = False
    _php0 = scenario_parse_phases([dict(SCENARIO_PRESETS[0]["phases"][0], t=0.01)])
    _sc52, _tot52b = scenario_build_raw("/tmp/opencode/rs520_preset.raw", _php0, _par52m, 8e6, seed=3)
    _ok527 = _ok527 and os.path.getsize(_sc52) == 2 * _tot52b and _tot52b == 80000
    print("ГОТОВІ-СЦЕНАРІЇ v5.2.0: %s (%d пресети, %d відл.)"
          % ("PASS" if _ok527 else "FAIL", len(SCENARIO_PRESETS), _tot52b))
    ok = ok and _ok527

    # --- ТЕСТ БАТЧ-АНАЛІЗУ (v5.2.0) ---
    _bd52 = "/tmp/opencode/rs520_batch"
    os.makedirs(_bd52, exist_ok=True)
    for _bn52, _bf52 in (("b1.raw", _abA), ("b2.raw", _abB)):
        with open(os.path.join(_bd52, _bn52), "wb") as _fh52:
            _fh52.write(open(_bf52, "rb").read())
    _rows52 = batch_scan(_bd52, 8e6)
    _csv52 = batch_save_csv("/tmp/opencode/rs520_batch.csv", _rows52)
    _ok528 = (len(_rows52) == 2 and abs(_rows52[0]["f_hz"] - 600e3) < 4e3
              and os.path.getsize(_csv52) > 100
              and "b1.raw" in open(_csv52, encoding="utf-8-sig").read())
    print("БАТЧ v5.2.0: %s (%d файлів, CSV %.0f Б)"
          % ("PASS" if _ok528 else "FAIL", len(_rows52), os.path.getsize(_csv52)))
    ok = ok and _ok528

    # --- ТЕСТ 3D У ЗВІТІ PDF (v5.3.0) ---
    _pdf53 = "/tmp/opencode/rs530_report.pdf"
    _s53 = render_spec3d_png("/tmp/opencode/rs53_spec3d.png", _raw52, 8e6)
    _m53 = render_map3d_png("/tmp/opencode/rs53_map3d.png", 50.0, -85.0, 2.4e9)
    _v53 = render_vs3d_png("/tmp/opencode/rs53_vs3d.png", _abA, 8e6,
                           jam_f_hz=600e3, jam_bw_hz=2e6, js_db=15.0)
    build_pdf_report(_pdf53,
                     [{"enabled": True, "kind": "Завада (шум)", "freq": 2.4e9,
                       "phase": 0.0, "amp": 12000.0, "width_us": 2.0,
                       "dc": 32768.0, "gain": 1.0}],
                     metric=_m51,
                     spec3d={"raw_path": _raw52, "fs_hz": 8e6},
                     map3d={"eirp_dbm": 50.0, "thresh_dbm": -85.0, "f_hz": 2.4e9},
                     vs3d={"raw_path": _abA, "fs_hz": 8e6, "jam_f_hz": 600e3,
                           "jam_bw_hz": 2e6, "js_db": 15.0})
    from pypdf import PdfReader as _PR53
    _np53 = len(_PR53(_pdf53).pages)
    _ok530 = (os.path.exists(_pdf53) and os.path.getsize(_pdf53) > 20000 and _np53 >= 2  # стор. залежать від пакування зображень (dpi бекенду рендеру)
              and _s53 and _m53 and _v53
              and os.path.getsize("/tmp/opencode/rs53_spec3d.png") > 8000
              and os.path.getsize("/tmp/opencode/rs53_map3d.png") > 8000
              and os.path.getsize("/tmp/opencode/rs53_vs3d.png") > 8000)
    if not _ok530:
        try:
            print("   DIAG 3D-PDF: npages=%s s=%s m=%s v=%s png %s/%s/%s Б" % (
                _np53, _s53, _m53, _v53,
                os.path.getsize("/tmp/opencode/rs53_spec3d.png"),
                os.path.getsize("/tmp/opencode/rs53_map3d.png"),
                os.path.getsize("/tmp/opencode/rs53_vs3d.png")))
        except Exception as _d530:
            print("   DIAG 3D-PDF: %r" % _d530)
    print("3D-PDF v5.3.0: %s (%d стор., %.0f КБ)"
          % ("PASS" if _ok530 else "FAIL", _np53, os.path.getsize(_pdf53) / 1024.0))
    ok = ok and _ok530

    # --- ТЕСТ МОДУЛЯ ОВ-ДРОНУ (v5.4.0) ---
    _ok540 = (len(FIBER_MODEMS) >= 8 and len(FIBER_CLOCKS) >= 6 and all(
        all(k in m for k in ("name", "lam_nm", "rate_mbps", "reach_km", "laser",
                             "p_tx_dbm", "sens_dbm", "overload_dbm", "clock_mhz"))
        for m in FIBER_MODEMS))
    _bl540 = fiber_bleach(10.0, 10.0, 1.0)
    _bn540 = fiber_bleach_needed(1.0, -3.0, 10.0)
    _st540 = fiber_status(-3.0, -26.0, -3.0)
    _st541 = fiber_status(-40.0, -26.0, -3.0)
    _rs540 = mcu_resonance(25e6, 1.0)
    _csv540 = fiber_modems_csv("/tmp/opencode/rs540_fiber.csv")
    _g54 = 20.0 * math.log10(10000.0 / 9.0)
    _ok540 = (_ok540
              and abs(_bl540["spot_um"] - 10000.0) < 1e-6
              and abs(_bl540["geom_db"] - _g54) < 1e-6
              and abs(_bl540["p_rx_dbm"] - (10.0 - 2.0 - _g54)) < 1e-6
              and abs(_bn540 - (-3.0 + 2.0 + _g54)) < 1e-6
              and _st540[0] == 2 and _st541[0] == 0
              and abs(_rs540["e_req_vm"] - 5.0) < 1e-9
              and abs(_rs540["eirp_dbm"] - 10.0 * math.log10((5.0 ** 2 / 30.0) * 1000.0)) < 1e-9
              and os.path.getsize(_csv540) > 300)
    print("ОВ-ДРОН v5.4.0: %s (%d модемів, потрібно %.1f дБм на 1 м, EIRP %.1f дБм)"
          % ("PASS" if _ok540 else "FAIL", len(FIBER_MODEMS), _bn540, _rs540["eirp_dbm"]))
    ok = ok and _ok540

    # --- ТЕСТ ШІ-ТРАКТУ І СИМУЛЯТОРА ОВ-ДРОНУ (v5.5.0) ---
    _ch550 = fiber_chain_select(1e9, 30.0)
    _ok550 = len(_ch550) >= 1 and all(
        -30.0 <= v["amp_req_db"] <= 60.0
        and abs(v["amp_req_db"] - (v["eirp_req_dbm"] - v["tx_dbm"] - v["g_dbi"]
                                   + v["vswr_loss_db"] + v["cable_db"])) < 1e-9
        for v in _ch550)
    _sm550 = fiber_sim_modem(FIBER_MODEMS[5], 70.0, 10.0, 2.0, (0.5, 1.0, 10.0, 100.0))
    _sm551 = fiber_sim_modem(FIBER_MODEMS[5], 10.0, 10.0, 2.0, (0.5, 1.0), d_drone_m=100.0)
    _sr550 = fiber_sim_resonance(29.2, 5.0)
    _ok550 = (_ok550 and _sm550["rows"][0][2] == 2 and _sm550["d_blind_m"] is not None
              and _sm551["rows"][0][2] == 0 and abs(_sr550 - 1.0) < 0.05)
    print("ШІ-ТРАКТ+СИМ v5.5.0: %s (%d варіантів, d_засл %s м, d_рез %.2f м)"
          % ("PASS" if _ok550 else "FAIL", len(_ch550),
             ("%.2f" % _sm550["d_blind_m"]) if _sm550["d_blind_m"] is not None else "—", _sr550))
    ok = ok and _ok550

    _ok560 = True
    try:
        _b56 = mod_ber("BPSK", 9.6)
        _ok560 = _ok560 and (0.0 < _b56 < 2e-5)
        _ok560 = _ok560 and (mod_ber("LoRa SF12", -19.5) < 0.5)
        _mp560 = melt_prr("BPSK", -95.0, -60.0, 6.0, 125000.0, 5470.0, 500)
        _ok560 = _ok560 and (_mp560["prr"] < 0.01)
        _mz30 = melt_zone("BPSK", -95.0, 30.0, 915e6, 6.0, 125000.0, 5470.0, 500)
        _mz0 = melt_zone("BPSK", -95.0, 0.0, 915e6, 6.0, 125000.0, 5470.0, 500)
        _mzn = melt_zone("BPSK", -95.0, -30.0, 915e6, 6.0, 125000.0, 5470.0, 500)
        _ok560 = (_ok560 and _mz30["d_melt_km"] is not None
                  and abs(_mz30["d_melt_km"] - 5.0) < 1e-6
                  and _mz0["d_melt_km"] is not None
                  and abs(_mz0["d_melt_km"] - 0.6) < 0.01
                  and _mzn["d_melt_km"] is None)
        _fw560 = follow_me_sim(100.0, 5.0, 0.9, 10)
        _ok560 = (_ok560 and abs(_fw560["p_cover"] - 0.45) < 1e-9
                  and abs(_fw560["gain_db"] - 10.0) < 1e-9)
        _fsA = failsafe_window(3.0, 100.0, 0.9999)
        _fsB = failsafe_window(3.0, 100.0, 0.999)
        _ok560 = (_ok560 and _fsA["ok"] and (not _fsB["ok"])
                  and abs(_fsA["p_fs"] - 0.9704) < 0.005)
        _pp560 = protocol_pulse("LoRa преамбула", 10.0, 8.0, sf=7, bw_hz=125000.0, sym=8)
        _ok560 = (_ok560 and abs(_pp560["pulse_ms"] - 8.192) < 0.01
                  and abs(_pp560["econ_db"] - 10.87) < 0.1)
        _jp560 = jam_profile_from_analysis([915e6, 915.2e6, 868e6], pri_us=120.0)
        _ok560 = (_ok560 and _jp560["n_peaks"] == 3
                  and abs(_jp560["freqs_mhz"][1] - 915.0) < 0.01
                  and _jp560["pri_us"] == 120.0)
    except Exception as _e560:
        _ok560 = False
        print("  лабораторія v5.6.0: ERROR %r" % (_e560,))
    print("ЛАБОРАТОРІЯ ПЛАВЛЕННЯ v5.6.0: %s (BER, PRR, зона, follow-me, failsafe, пульсація, профіль)"
          % ("PASS" if _ok560 else "FAIL"))
    ok = ok and _ok560

    _ok570 = True
    try:
        _ch57 = fpv_channels("5.8 · Raceband")
        _ok570 = _ok570 and (len(_ch57) == 8 and abs(_ch57[0][1] - 5658e6) < 1e3
                             and abs(_ch57[7][1] - 5917e6) < 1e3)
        _st57 = fpv_scan_table("5.8 · Raceband", [5769e6, 5880e6])
        _ok570 = _ok570 and (len(_st57) == 8 and _st57[3][3] == "ВІДЕО"
                             and _st57[6][3] == "ВІДЕО"
                             and sum(1 for _r in _st57 if _r[3] == "ВІДЕО") == 2)
        _ok570 = _ok570 and (video_quality_est(30.0)[0] == 100
                             and video_quality_est(19.0)[0] == 50
                             and video_quality_est(5.0)[0] == 0)
        _ie57 = injection_eirp(-60.0, 10.0, 1.0, 5800e6, 0.0)
        _ok570 = _ok570 and (abs(_ie57["fspl_db"] - 107.71) < 0.2
                             and abs(_ie57["eirp_dbm"] - 57.71) < 0.3)
        _ok570 = _ok570 and (capture_effect(8.0)["state"] == "захоплення"
                             and capture_effect(3.0)["state"] == "конкуренція"
                             and capture_effect(-2.0)["state"] == "без впливу")
        _pb57 = video_pattern("Кольорові смуги", 64, 48)
        _ok570 = _ok570 and (_pb57[0, 0] > 0.9 and _pb57[0, -1] < 0.05)
        _pz57 = video_pattern("Смуги-застереження", 64, 48)
        _ok570 = _ok570 and (_pz57[0, 0] == 1.0)
        _pg57 = video_pattern("Градієнт", 64, 16)
        _std57 = {"line_us": 32.0, "sync_us": 3.0, "active_us": 25.0,
                  "lines": 16, "fps": 100.0}
        _sig57 = video_fm_modulate(_pg57, _std57, 16e6, 3e6, 64e6, 2, amp=0.6)
        _r57 = video_raster_demod(_sig57, 64e6, _std57, 16e6, 3e6, cols=64, rows=16)
        _err57 = float(np.mean(np.abs(_r57 - _pg57)))
        _ok570 = _ok570 and (_sig57.size == 65536 and _err57 < 0.2)
        _ppm57 = _fpv_ppm(_r57)
        _ok570 = _ok570 and (_ppm57.startswith(b"P6\n64 16\n255\n")
                             and len(_ppm57) == 13 + 64 * 16 * 3)
    except Exception as _e570:
        _ok570 = False
        print("  FPV-ВІДЕО v5.7.0: ERROR %r" % (_e570,))
    print("FPV-ВІДЕО v5.7.0: %s (скан, якість, EIRP-інжекції, FM-захоплення, синтез+растер)"
          % ("PASS" if _ok570 else "FAIL"))
    ok = ok and _ok570

    _ok571 = True
    try:
        _ok571 = _ok571 and fpv_fs_auto(16e6, 3e6) == 64e6
        _ok571 = _ok571 and fpv_fs_auto(5769e6, 8e6) == 12000e6
        _ok571 = _ok571 and fpv_fs_auto(100e6, 8e6) == 250e6
        _ok571 = _ok571 and abs(fpv_fs_auto(435e6, 8e6) - 1000e6) < 1.0
        _thr571 = False
        try:
            fpv_fs_auto(11000e6, 8e6)
        except ValueError:
            _thr571 = True
        _ok571 = _ok571 and _thr571
    except Exception as _e571:
        _ok571 = False
        print("  FS-АВТО v5.7.1: ERROR %r" % (_e571,))
    print("FS-АВТО v5.7.1: %s (ряд АЦП, Nyquist + запас синхро, межа ряду)"
          % ("PASS" if _ok571 else "FAIL"))
    ok = ok and _ok571

    _ok572 = True
    try:
        _ok572 = _ok572 and fs_auto_plan([852e6, 2400e6]) == 5000e6
        _ok572 = _ok572 and fs_auto_plan([5800e6], min_width_s=5e-8) == 12000e6
        _ok572 = _ok572 and fs_auto_plan([]) == 64e6
        _ok572 = _ok572 and fs_auto_plan([16e6], min_width_s=2e-6) == 64e6
        _ok572 = _ok572 and (fpv_fs_auto(16e6, 3e6) == fs_auto_plan([19e6]))
        _thr572 = False
        try:
            fs_auto_plan([11000e6])
        except ValueError:
            _thr572 = True
        _ok572 = _ok572 and _thr572
    except Exception as _e572:
        _ok572 = False
        print("  FS-ПЛАН v5.7.2: ERROR %r" % (_e572,))
    print("FS-ПЛАН v5.7.2: %s (авто-fs для всієї програми: Найквіст+запас, ряд АЦП)"
          % ("PASS" if _ok572 else "FAIL"))
    ok = ok and _ok572

    _ok580 = True
    try:
        _id580 = scpi_parse_idn("Rohde&Schwarz,FSW26,1311.6010K26,4.10")
        _ok580 = _ok580 and (_id580["vendor"] == "Rohde&Schwarz" and _id580["model"] == "FSW26")
        _tr580a = scpi_parse_trace("-10.5, -9.0, -8.0\n")
        _ok580 = _ok580 and (_tr580a.size == 3 and abs(_tr580a[0] + 10.5) < 1e-9)
        _blk580 = b"#501280" + np.arange(640, dtype="<i2").tobytes()
        _tr580b = scpi_parse_trace(_blk580)
        _ok580 = _ok580 and (_tr580b.size == 640 and abs(_tr580b[639] - 639.0) < 1e-9)
        _fr580 = trace_freqs(915e6, 20e6, 400)
        _tr580 = np.full(400, -60.0)
        for _i in range(400):
            _tr580[_i] += 30.0 * np.exp(-((_i - 99.5) / 3.0) ** 2) + 18.0 * np.exp(-((_i - 259.5) / 5.0) ** 2)
        _pk580 = trace_top_peaks(_fr580, _tr580)
        _ok580 = _ok580 and (len(_pk580) == 2
                             and abs(_pk580[0]["f_hz"] - (905e6 + 99.0 * 20e6 / 399.0)) < 1e3
                             and abs(_pk580[0]["dbm"] + 30.0) < 1.5)
        _pl580 = scpi_sweep_plan(1e9, 2e9, 1e6)
        _ok580 = _ok580 and abs(_pl580["n_points"] - 1001) <= 1
        import threading as _th580
        import socket as _sk580
        import time as _tm580
        import math as _m580
        _port580 = []
        _stop580 = _th580.Event()

        def _sa_srv580():
            _srv = _sk580.socket(_sk580.AF_INET, _sk580.SOCK_STREAM)
            _srv.setsockopt(_sk580.SOL_SOCKET, _sk580.SO_REUSEADDR, 1)
            _srv.bind(("127.0.0.1", 0))
            _srv.listen(4)
            _srv.settimeout(0.2)
            _port580.append(_srv.getsockname()[1])
            _tvals = ",".join("%.1f" % (-60.0 + 30.0 * _m580.exp(-((_i - 99.5) / 3.0) ** 2)
                                        + 18.0 * _m580.exp(-((_i - 259.5) / 5.0) ** 2))
                              for _i in range(400))
            while not _stop580.is_set():
                try:
                    _cn, _ = _srv.accept()
                except _sk580.timeout:
                    continue
                _cn.settimeout(1.5)
                try:
                    _buf = b""
                    while True:
                        _d = _cn.recv(4096)
                        if not _d:
                            break
                        _buf += _d
                        while b"\n" in _buf:
                            _line, _buf = _buf.split(b"\n", 1)
                            _c = _line.strip().decode("ascii", "ignore")
                            if _c == "*IDN?":
                                _cn.sendall(b"RAW-Fake,SA-Fake,000123,1.0\n")
                            elif _c.startswith(":TRAC"):
                                _cn.sendall((_tvals + "\n").encode("ascii"))
                except OSError:
                    pass
                finally:
                    try:
                        _cn.close()
                    except OSError:
                        pass
            _srv.close()

        _th580.Thread(target=_sa_srv580, daemon=True).start()
        for _ in range(40):
            if _port580:
                break
            _tm580.sleep(0.05)
        _ok580 = _ok580 and bool(_port580)
        if _port580:
            _cl580 = ScpiClient("127.0.0.1", _port580[0], 2.0)
            _cl580.connect()
            _id580b = _cl580.identify()
            _ok580 = _ok580 and (_id580b["model"] == "SA-Fake")
            _cl580.setup(915e6, 20e6, 100e3, 0.0)
            _tr580c = _cl580.fetch_trace()
            _ok580 = _ok580 and (_tr580c.size == 400
                                 and abs(float(np.max(_tr580c)) + 30.0) < 1.5)
            _cl580.close()
        _stop580.set()
    except Exception as _e580:
        _ok580 = False
        print("  SCPI v5.8.0: ERROR %r" % (_e580,))
    print("SCPI v5.8.0: %s (IDN, траси ASCII/блок, піки, план, TCP-емулятор)"
          % ("PASS" if _ok580 else "FAIL"))
    ok = ok and _ok580

    # --- v5.9.0: рецепти парку платформ ---
    print("Рецепти v5.9.0:")
    _ok9 = True
    _exp_plj = fspl_db(1.0, 1575.42)
    _rows = recipe_plan("GPS L1 C/A (ШПС)",
                        ["PlutoSky R2", "LimeSDR Mini", "RTL-SDR Blog V4"],
                        pa_dbm=20.0, amp_db=0.0, ant_name="Власна (ручне введення)",
                        dist_km=5.0, dj_km=1.0)
    _r0 = _rows[0] if _rows else {}
    _checks9 = [
        (len(_rows) == 3, "рядків 3"),
        (_r0.get("model") == "PlutoSky R2", "перший — PlutoSky R2 (радіус більший)"),
        (_r0.get("verdict") == "OK", "вердикт OK"),
        (_r0.get("kind_name") == "Завада (чирп)", "ПП 20 → чирп"),
        (_r0.get("js_req") == 30.0, "J/S 30"),
        (abs(_r0.get("p_req", 0) - (30.0 + (-130.0) + _exp_plj)) < 1e-6,
         "p_req = js_req + ps + ФСМ"),
        (abs(_r0.get("d_max", 0) - 10.0 ** ((20.0 + 130.0 - 30.0 - 32.45
             - 20.0 * math.log10(1575.42)) / 20.0)) < 1e-6, "d_max за формулою"),
        (abs(_r0.get("sdr_d_max", 0) - 10.0 ** ((10.0 + 130.0 - 30.0 - 32.45
             - 20.0 * math.log10(1575.42)) / 20.0)) < 1e-6, "sdr_d_max при TX 10 дБм"),
        (abs(_rows[0].get("bw_hz", 0) - 2.046e6) < 1.0, "смуга цілі без звуження"),
        (_rows[1].get("model") == "LimeSDR Mini" and _rows[1].get("verdict") == "OK",
         "LimeSDR Mini OK"),
        (abs(_rows[1].get("sdr_d_max", 0) - 10.0 ** ((6.0 + 130.0 - 30.0 - 32.45
             - 20.0 * math.log10(1575.42)) / 20.0)) < 1e-6, "sdr_d_max при TX 6 дБм"),
        (_rows[2].get("model") == "RTL-SDR Blog V4" and _rows[2].get("verdict") == "ПРИЙОМ",
         "RTL — приймальна"),
    ]
    _rowb = recipe_plan("Вся смуга L1 (1559-1610 МГц)", ["PlutoSky R2"],
                        pa_dbm=20.0, amp_db=0.0, ant_name="Власна (ручне введення)")
    _checks9.append((_rowb and _rowb[0].get("verdict") == "СМУГА", "Вся смуга L1 → СМУГА"))
    _checks9.append((_rowb and abs(_rowb[0].get("bw_hz", 0) - 0.4 * 61.44e6) < 1.0,
                     "смуга звужена до 24.576 МГц"))
    _rowc = recipe_plan("Аналогове FPV 5.8 ГГц", ["ADALM-Pluto (61.44 МС/с)"],
                        pa_dbm=20.0, amp_db=0.0, ant_name="Власна (ручне введення)")
    _checks9.append((_rowc and _rowc[0].get("verdict") == "ПОЗА", "5.8 ГГц поза Pluto → ПОЗА"))
    for _c, _t in _checks9:
        if not _c:
            _ok9 = False
            print("  НЕ СПРАВДЖЕНО:", _t)
    print("  Рецепти v5.9.0: PASS" if _ok9 else "  Рецепти v5.9.0: FAIL")
    if not _ok9:
        ok = False
    # --- v6.0.0: інструментарій ---
    print("Інструментарій v6.0.0:")
    _ok6 = True
    _t6 = _tempfile6.mkdtemp(prefix="raw6_")
    _checks6 = []
    hist_log("SELFTEST-6.0.0", "запис 1", {"k": 1})
    hist_log("SELFTEST-6.0.0", "запис 2")
    _hr = [r for r in hist_rows(50) if "SELFTEST-6.0.0" in (r[1] or "")]
    _checks6.append((len(_hr) >= 2, "журнал: записів %d >= 2" % len(_hr)))
    _cs6 = os.path.join(_t6, "hist.csv")
    hist_export_csv(_cs6, 50)
    _checks6.append((os.path.exists(_cs6) and os.path.getsize(_cs6) > 10, "журнал: CSV"))
    _checks6.append((classify_width(6e6)[0] == "FPV (аналоговий ЧМ)", "сигнатура 6 МГц → FPV"))
    _checks6.append((classify_width(20e6)[0] == "Wi-Fi / OFDM 20 МГц", "сигнатура 20 МГц → Wi-Fi"))
    _checks6.append((classify_width(0.2e6)[0] == "LoRa / вузький чирп", "сигнатура 0.2 МГц → LoRa"))
    _checks6.append((classify_width(2.046e6)[0] == "ГНСС ШПС (вузький)", "сигнатура 2.046 МГц → ГНСС"))
    _checks6.append((classify_width(50e6)[0] == "Невідома", "сигнатура 50 МГц → Невідома"))
    _checks6.append((len(SCPI_PRESETS) == 6 and all("trace_cmd" in p for p in SCPI_PRESETS.values()),
                     "пресетів SCPI: 6"))
    theme_pref_set("dark")
    _checks6.append((theme_pref_get() == "dark", "тема: dark збережено"))
    theme_pref_set("light")
    _checks6.append((theme_pref_get() == "light", "тема: light збережено"))
    _N6 = 1 << 16
    np.random.seed(42)
    _tidx6 = np.arange(_N6)
    _a6 = (32768 + 8000 * np.sin(2 * np.pi * 64 * _tidx6 / _N6)
           + np.random.normal(0, 500, _N6)).clip(0, 65535).astype("<u2")
    _b6 = (32768 + np.random.normal(0, 500, _N6)).clip(0, 65535).astype("<u2")
    _pa6 = "/tmp/opencode/e6a.raw"
    _pb6 = "/tmp/opencode/e6b.raw"
    _a6.tofile(_pa6)
    _b6.tofile(_pb6)
    _cr6 = compare_raws(_pa6, _pb6, nch=1, nfft=_N6)
    _checks6.append((len(_cr6["peaks"]) == 8 and any(_p["bin"] == 64 for _p in _cr6["peaks"]),
                     "порівняння: топ-пік бін 64"))
    _checks6.append((_cr6["peaks"][0]["delta_db"] > 15.0,
                     "порівняння: Δ піка %.1f > 15 дБ" % _cr6["peaks"][0]["delta_db"]))
    _seg6 = (32768 + 9000 * np.sin(2 * np.pi * 100 * _tidx6 / _N6)).astype("<u2")
    _seg6b = (32768 + 7000 * np.sin(2 * np.pi * 300 * _tidx6 / _N6)).astype("<u2")
    _pan6 = np.empty(2 * _N6, dtype="<u2")
    _pan6[0::2] = _seg6
    _pan6[1::2] = _seg6b
    _panf6 = "/tmp/opencode/e6pan.raw"
    _pan6.tofile(_panf6)
    _pr6 = panorama_from_raw(_panf6, nch=2, nfft=_N6)
    _checks6.append((bool(np.all(np.diff(_pr6["freqs_hz"]) >= -1e6)),
                     "панорама: частоти зростають"))
    _checks6.append((len(_pr6["dbm"]) == len(_pr6["freqs_hz"]) and len(_pr6["peaks"]) >= 1,
                     "панорама: спектр+піки"))
    _mc6 = monte_carlo_follow(100.0, 2.0, 0.9, 1, trials=2000, seed=42)
    _exp6 = 0.9 * (10.0 - 2.0) / 10.0
    _checks6.append((abs(_mc6["share_p50"] - _exp6) < 0.05,
                     "MC: P50 %.3f ≈ %.3f" % (_mc6["share_p50"], _exp6)))
    _checks6.append((_mc6["share_p5"] < _mc6["share_p95"], "MC: P5 < P95"))
    _fs6 = 16e6
    _n6i = int(_fs6 * 0.05)
    _t6i = np.arange(_n6i) / _fs6
    _fi6 = 2e6 + 1e6 * np.sin(2 * np.pi * 15625.0 * _t6i)
    _ph6 = 2 * np.pi * np.cumsum(_fi6) / _fs6
    _sig6 = (32768 + 30000 * np.cos(_ph6)).clip(0, 65535).astype("<u2")
    _pf6 = "/tmp/opencode/e6fpv.raw"
    _sig6.tofile(_pf6)
    _fp6 = fpv_import_preview(_pf6, _fs6, "PAL", 5800e6, 2e6, nch=1)
    _ras6 = _fp6["raster"]
    _checks6.append((_ras6.shape == (48, 64) and bool(np.isfinite(_ras6).all()),
                     "імпорт IQ: растр 48x64"))
    _csv6p = os.path.join(_t6, "cli.csv")
    cli_recipes("GPS L1 C/A (ШПС)", park="PlutoSky R2,RTL-SDR Blog V4",
                pa_dbm=20.0, dist_km=5.0, dj_km=1.0, ant_name="Власна (ручне введення)",
                freq_mhz=None, out=_csv6p)
    _checks6.append((os.path.exists(_csv6p)
                     and open(_csv6p, encoding="utf-8-sig").read().count("\n") == 3,
                     "CLI: рецепти CSV 2 рядки + заголовок"))
    cli_melt("BPSK,915,20,40,6,0.125,5000,64")
    _checks6.append((True, "CLI: плавлення (друк таблиці)"))
    for _c, _t in _checks6:
        if not _c:
            _ok6 = False
            print("  НЕ СПРАВДЖЕНО:", _t)
    print("  Інструментарій v6.0.0: PASS" if _ok6 else "  Інструментарій v6.0.0: FAIL")
    if not _ok6:
        ok = False
    _ok61 = True
    _checks61 = []
    _jb = jam_burst_codes(1e6, 470e6, 1000.0, 0.2, 20000, 0.01).astype(np.float64)
    _far = np.abs(_jb - 32768.0) > 3000
    _checks61.append((abs(_far.mean() - 0.2) < 0.08, "пакети: скважність %.2f" % _far.mean()))
    _pw = np.tile(np.array([40.0] * 100 + [0.0] * 100), 5)
    _rj = reactive_jam_model(_pw, 0.001, 20.0, 5.0)
    _checks61.append((_rj["detect_s"] is not None and abs(_rj["on_air_frac"] - 0.475) < 0.01,
                      "реактивна: on_air %.3f" % _rj["on_air_frac"]))
    _tp = jam_time_profile([(0, 30, 0.5), (30, 60, 0.0)], 60)
    _checks61.append((_tp["covered_frac"] == 0.5 and len(_tp["gaps"]) == 1,
                      "профіль: покриття 0.5, 1 діра"))
    _pc = per_vs_js("BPSK", [0.0, 10.0, 20.0], 512)
    _checks61.append((_pc[0]["ber"] < _pc[1]["ber"] < _pc[2]["ber"], "PER: монотонність"))
    _mb = multiband_plan([("Відео 5.8", 5650e6, 5950e6, 1), ("Керування 2.4", 2400e6, 2483e6, 2),
                          ("GNSS L1", 1559e6, 1610e6, 2), ("Телеметрія 433", 433e6, 434.8e6, 3),
                          ("Відео 1.2", 1080e6, 1360e6, 3)], n_ch=4)
    _checks61.append((_mb["overloaded"] and _mb["channels_used"] == 4, "багатосмуга: 4 канали"))
    _jc = js_curve_db(30.0, 10.0, 1000.0, 3000.0)
    _checks61.append((_jc["js_db"][0] > _jc["js_db"][-1] and _jc["d_max_m"] > 0,
                      "J/S: спад з дистанцією, d_max %.0f" % _jc["d_max_m"]))
    _chs = [172 + 120 * (k % 8) for k in range(16)]
    _sf = sbus_frame(_chs)
    _checks61.append((len(_sf) == 25 and _sf[0] == 0x0F and rc_channels("SBUS", _sf) == _chs,
                      "SBUS: синтез → канали"))
    _frs = rc_extract_frames(bytes([0x55]) + _sf + bytes([0xAA]), "SBUS")
    _checks61.append((len(_frs) == 1 and _frs[0][0] == 1, "SBUS: витяг у потоці"))
    _ibv = [800 + k for k in range(14)]
    _ib = ibus_frame(_ibv)
    _checks61.append((len(_ib) == 32 and rc_channels("IBUS", _ib) == _ibv, "iBUS: синтез → канали"))
    _cf = crsf_frame(0x16, b"\x01\x02")
    _checks61.append((len(_cf) == 6 and rc_extract_frames(b"\x00" + _cf + b"\xff", "CRSF")[0][1] == _cf,
                      "CRSF: crc8 + витяг"))
    _tc = rc_telemetry_crsf(crsf_frame(0x08, bytes([0, 100, 10, 0, 232, 3, 0, 80])))
    _checks61.append((_tc is not None and _tc.get("тип") == "батарея" and abs(_tc["В"] - 10.0) < 0.01,
                      "телеметрія: батарея %s В" % (_tc and _tc.get("В"))))
    _mtc = rc_telemetry_msp(msp_frame(101, b"\x02\x00"))
    _checks61.append((_mtc is not None and _mtc["cmd"] == 101, "MSP: crc8-xor"))
    _cht = [[1000 + (k % 5) * 10, 1100] for k in range(10)] + [[1000, 1100]] * 20
    _fa = failsafe_analyze(_cht, 0.001)
    _checks61.append((_fa["замороження"] and _fa["заморожених_відліків"] == 19,
                      "failsafe: %s відліків" % _fa["заморожених_відліків"]))
    _hp = hop_pattern_detect([100.0, 200.0, 300.0] * 10)
    _checks61.append((_hp["цикл"] == 3 and len(_hp["прогноз"]) == 3, "хоп: цикл 3, прогноз 3"))
    _lq = link_quality_timeline([k * 0.01 + 0.001 for k in range(8)], [-50.0] * 8, 0.01, 0.1)
    _checks61.append((_lq["очікувано"] == 10 and _lq["прийнято"] == 8 and _lq["LQ_%"] == 80.0,
                      "LQ: %.1f%%" % _lq["LQ_%"]))
    _df = amplitude_df([-10.0, -10.0, -20.0, -20.0])
    _checks61.append((abs(_df["азимут_°"] - 45.0) < 5.0, "RDF: азимут %.1f°" % _df["азимут_°"]))
    _ac = activity_stats(np.array([0.0] * 10 + [40.0] * 10 + [0.0] * 10 + [40.0] * 10), 0.001, 20.0)
    _checks61.append(("керування" in _ac["клас"], "енергодетектор: %s" % _ac["клас"]))
    _pw2 = np.concatenate([np.full(300, -100.0), np.full(200, -60.0)])
    _gs2 = gnss_spoof_scan(_pw2, 0.001)
    _checks61.append((len(_gs2["ознаки"]) >= 1, "спуфінг: %s" % _gs2["вердикт"]))
    _fsr = 1e6
    _tt = np.arange(60000) / _fsr
    _be = (((_tt * 100.0) % 1.0) < 0.2).astype(np.float64)
    _bp61 = "/tmp/opencode/e61burst.raw"
    (32768 + _be * 20000 * np.random.normal(0, 1, 60000)
     + np.random.normal(0, 200, 60000)).clip(0, 65535).astype("<u2").tofile(_bp61)
    _bs = burst_scan(_bp61, _fsr)
    _checks61.append((_bs["bursts"] >= 4 and _bs["period_s"] is not None
                      and abs(_bs["period_s"] - 0.01) < 0.002,
                      "burst_scan: %d пакетів, період %s" % (_bs["bursts"], _bs["period_s"])))
    _wb = make_wave_ext("jam_burst", 1 << 16, 1e6, 470e6, 0.0, 1000, 0,
                        np.random.default_rng(1), per_us=1000.0, pw_us=200.0)
    _checks61.append((abs(float((_wb == 0).mean()) - 0.8) < 0.02,
                      "jam_burst: скважність %.3f" % float((_wb == 0).mean())))
    _tc2 = jam_tactics_compare(30.0)
    _checks61.append((_tc2["рекомендовано"] == "Шум (barrage)", "тактики: %s" % _tc2["рекомендовано"]))
    for _c, _t in _checks61:
        if not _c:
            _ok61 = False
            print("  НЕ СПРАВДЖЕНО:", _t)
    print("  Завади+Керування v6.1.0: PASS" if _ok61 else "  Завади+Керування v6.1.0: FAIL")
    if not _ok61:
        ok = False

    print("--- ВОДОСПАД+PDF v6.2.0 ---")
    import struct as _st62
    _ok621 = rtl_tcp_cmds(100e6, 2400000) == (b"\x01" + _st62.pack(">I", 100000000)
                                              + b"\x02" + _st62.pack(">I", 2400000))
    import threading as _th62
    import socket as _sk62
    def _srv62(_s):
        _c, _a = _s.accept()
        try:
            _c.sendall(bytes(range(256)) * 4)
        finally:
            _c.close()
    _s62 = _sk62.socket()
    _s62.setsockopt(_sk62.SOL_SOCKET, _sk62.SO_REUSEADDR, 1)
    _s62.bind(("127.0.0.1", 0))
    _s62.listen(1)
    _prt62 = _s62.getsockname()[1]
    _t62 = _th62.Thread(target=_srv62, args=(_s62,), daemon=True)
    _t62.start()
    _iq62 = rtl_tcp_read("127.0.0.1", _prt62, 512, 100e6, 2400000)
    _t62.join(2.0)
    _s62.close()
    _ok622 = (_iq62.size == 512
              and abs(_iq62[0].real - (-127.5)) < 1e-9
              and abs(_iq62[0].imag - (-126.5)) < 1e-9)
    _seg62 = make_wave("jam_cw", 256, 8e6, 600e3, 0.0, 1000, 0, np.random.default_rng(1))
    _col62 = waterfall_column(_seg62, 256, -100.0, -20.0)
    _ok623 = (_col62.size == 129 and float(_col62.max()) <= -20.0 + 1e-9
              and float(_col62.min()) >= -100.0 - 1e-9)
    _buf62 = waterfall_push(np.full((5, 129), -160.0), _col62)
    _ok624 = (_buf62.shape == (5, 129)
              and bool(np.allclose(_buf62[0], _col62))
              and bool(np.allclose(_buf62[1], -160.0)))
    _p62 = gui_pdf_report("/tmp/opencode/rs62_gui.pdf", "Тест звіту", meta="самотест",
                          pngs=["/tmp/opencode/rs53_spec3d.png"],
                          notes=["перевірка gui_pdf_report"])
    from pypdf import PdfReader as _PR62
    _ok625 = (os.path.exists(_p62) and os.path.getsize(_p62) > 3000
              and len(_PR62(_p62).pages) >= 1)
    print("Водоспад+PDF v6.2.0: %s (cmd %s, iq %s, col %s, push %s, pdf %s)"
          % ("PASS" if (_ok621 and _ok622 and _ok623 and _ok624 and _ok625) else "FAIL",
             _ok621, _ok622, _ok623, _ok624, _ok625))
    ok = ok and _ok621 and _ok622 and _ok623 and _ok624 and _ok625
    print("ПІДСУМОК:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    main()

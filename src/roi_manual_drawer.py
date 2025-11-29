#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Manual Drawer + Auto-Segmentation + ROI Manager (v2 25.11.27)
--------------------------------------------------------------------------------
v2 Fix:
- ROI 중간 수정 기능 추가 (edit 모드에서만 작동함)
"""

import os, re, glob, zipfile, platform, tempfile, json, time, sys
import numpy as np
import threading

import matplotlib
if platform.system().lower().startswith("win"):
    matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']

# 🔧 키 충돌 완전 제거
matplotlib.rcParams['keymap.save'] = []     
matplotlib.rcParams['keymap.pan'] = []      
matplotlib.rcParams['keymap.zoom'] = []     
matplotlib.rcParams['keymap.fullscreen'] = []  
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import matplotlib.path as mpath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import patheffects as mpatheffects

from tifffile import imread, imwrite
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.measure import find_contours, approximate_polygon
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize
from skimage import exposure, filters

from tkinter import (
    Tk, filedialog, simpledialog, messagebox, StringVar, BooleanVar, DoubleVar,
    Entry, Label, Button, Frame, Radiobutton, Checkbutton, Scale, HORIZONTAL,
    DISABLED, NORMAL, OptionMenu, Toplevel, Listbox, Scrollbar, END, SINGLE
)
import tkinter as _tk

# -------------------- Language resources --------------------
LANG_DEFAULT = "ko"
LANG_CURRENT = LANG_DEFAULT

STRINGS = {
    "ko": {
        "title_startup": "ROI 그리기/채널/모드 선택",
        "label_folder": "TIF 폴더",
        "btn_browse": "찾기",
        "label_channel": "시작 채널(번호)",
        "label_mode": "모드",
        "mode_new": "신규(New)",
        "mode_edit": "수정(Edit)",
        "timelapse_cb": "Timelapse(시간 포함: 파일명=SXX_TXX_X)",
        "label_stage": "Stage (숫자)",
        "label_time": "Timepoint (숫자)",
        "cb_include": "Edit 모드에서 ROI 없는 파일도 포함",
        "label_tol": "경계 단순화 tolerance",
        "label_min_area": "최소 area (px²)",
        "label_color": "Pseudocolor",
        "btn_ok": "확인",
        "btn_cancel": "취소",
        "err_title": "오류",
        "err_folder": "유효한 폴더를 선택하세요",
        "err_channel": "채널 번호는 0~999 사이 숫자",
        "err_stage": "Stage는 숫자",
        "err_time": "Timepoint는 숫자",
        "err_tol": "Tolerance는 0.1~5.0",
        "err_min_area": "최소 area는 0 이상",
        "exit_cancelled": "작업이 취소되었습니다.",
        "err_no_tif": "[오류] 폴더에 TIF 파일이 없습니다.",
        "hud_instr": "P: 새 ROI | Enter/Space/Y: 확정 | Esc/N: 취소 | U: 마지막 되돌리기 | C: 모두 지우기 | Ctrl+S/Cmd+S: 저장 | Q: 빠르게 종료 | PageUp/Down: ROI 이동 | Delete: ROI 삭제 | Tab/Shift+Tab: 채널 순환 | +/-: gamma | [: p_low | ]: p_high",
        "log_folder": "폴더: {folder}",
        "log_mode": "모드: {mode} | 시작채널(번호): {start_ch} | timelapse={timelapse} | include_no_roi={include_no} | 대상 Stage/Time {count}개",
        "log_paths": "ROI: {outdir} | mask: {mask_dir} | overlay: {overlay_dir} | zip: {zip_dir}",
        "log_filter": "필터: Stage={stage}, Time={time}",
        "log_params": "tolerance={tolerance:.2f}, min_area={min_area:.0f}, pseudocolor={color_mode}",
        "log_overlay": "overlay: FAST(PIL) mode, max side = {maxpx}px",
        "log_force_quit": "사용자 요청으로 전체 종료(Ctrl+Q/Cmd+Q).",
        "log_no_tasks": "[정보] 처리할 항목이 없습니다. (mode={mode}, include_no_roi={include_no}, timelapse={timelapse})",
    },
    "en": {
        "title_startup": "ROI Drawer / Channel / Mode",
        "label_folder": "TIF folder",
        "btn_browse": "Browse",
        "label_channel": "Start channel (number)",
        "label_mode": "Mode",
        "mode_new": "New",
        "mode_edit": "Edit",
        "timelapse_cb": "Timelapse (filename=SXX_TXX_X)",
        "label_stage": "Stage (number)",
        "label_time": "Timepoint (number)",
        "cb_include": "In Edit mode include files without ROI",
        "label_tol": "Boundary simplify tolerance",
        "label_min_area": "Min area (px²)",
        "label_color": "Pseudocolor",
        "btn_ok": "OK",
        "btn_cancel": "Cancel",
        "err_title": "Error",
        "err_folder": "Select a valid folder",
        "err_channel": "Channel number must be 0~999",
        "err_stage": "Stage must be a number",
        "err_time": "Timepoint must be a number",
        "err_tol": "Tolerance must be 0.1~5.0",
        "err_min_area": "Min area must be >= 0",
        "exit_cancelled": "Operation was cancelled.",
        "err_no_tif": "[Error] No TIF files in the folder.",
        "hud_instr": "P: new ROI | Enter/Space/Y: accept | Esc/N: cancel | U: undo last | C: clear all | Ctrl+S/Cmd+S: save | Q: quick quit | PageUp/Down: move ROI | Delete: delete ROI | Tab/Shift+Tab: cycle channel | +/-: gamma | [: p_low | ]: p_high",
        "log_folder": "Folder: {folder}",
        "log_mode": "Mode: {mode} | start_ch: {start_ch} | timelapse={timelapse} | include_no_roi={include_no} | targets {count}",
        "log_paths": "ROI: {outdir} | mask: {mask_dir} | overlay: {overlay_dir} | zip: {zip_dir}",
        "log_filter": "Filter: Stage={stage}, Time={time}",
        "log_params": "tolerance={tolerance:.2f}, min_area={min_area:.0f}, pseudocolor={color_mode}",
        "log_overlay": "overlay: FAST(PIL) mode, max side = {maxpx}px",
        "log_force_quit": "User requested full quit (Ctrl+Q/Cmd+Q).",
        "log_no_tasks": "[Info] Nothing to process. (mode={mode}, include_no_roi={include_no}, timelapse={timelapse})",
    },
}


def t(key: str, default=None, lang=None) -> str:
    """Simple i18n lookup with ko default."""
    lng = (lang or LANG_CURRENT or LANG_DEFAULT)
    if lng not in STRINGS:
        lng = LANG_DEFAULT
    if key in STRINGS[lng]:
        return STRINGS[lng][key]
    if key in STRINGS.get(LANG_DEFAULT, {}):
        return STRINGS[LANG_DEFAULT][key]
    return default if default is not None else key

# [FIX] Helper function moved to top level scope
def pick_lang_from_argv(argv):
    lang = LANG_DEFAULT
    for i, a in enumerate(argv):
        al = str(a).lower()
        if al in ("-mode", "--mode") and (i + 1) < len(argv):
            if str(argv[i+1]).lower().startswith("en"):
                lang = "en"
        if al in ("-mode=en", "--mode=en", "/mode=en"):
            lang = "en"
    return lang


# ===================== 설정(Overlay 초고속) =====================
FAST_OVERLAY = True           # True → PIL로 빠르게 그리고 저장
FAST_OVERLAY_MAXPX = 1400     # 긴 변 최대 픽셀(overlay만 다운스케일), 분석용 아님

# ===================== 유틸/로깅 =====================
_t0 = time.time()
def log(msg: str):
    dt = time.time() - _t0
    print(f"[{dt:7.2f}s] {msg}")

def maximize_figure_window(fig):
    try:
        mgr = fig.canvas.manager
        backend = matplotlib.get_backend().lower()
        if "tkagg" in backend and hasattr(mgr, "window"):
            try:
                mgr.window.state("zoomed")
            except Exception:
                try:
                    mgr.window.attributes("-zoomed", True)
                except Exception:
                    mgr.window.attributes("-fullscreen", True)
            return
        if ("qt5" in backend or "qt6" in backend or "qtagg" in backend) and hasattr(mgr, "window"):
            mgr.window.showMaximized(); return
        if "wx" in backend and hasattr(mgr, "frame"):
            mgr.frame.Maximize(True); return
    except Exception:
        pass
    try:
        fig.set_size_inches(18, 10, forward=True)
        fig.tight_layout()
    except Exception:
        pass

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_tifs(folder):
    exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    uniq = {}
    for p in files:
        norm = os.path.normcase(os.path.abspath(p))
        if norm not in uniq:
            uniq[norm] = p
    return sorted(uniq.values(), key=natural_key)

def ensure_outdirs(folder):
    root = os.path.join(folder, "roi")
    mask_dir = os.path.join(root, "mask")
    overlay_dir = os.path.join(root, "overlay")
    zip_dir = os.path.join(root, "zip")
    for d in (root, mask_dir, overlay_dir, zip_dir):
        os.makedirs(d, exist_ok=True)
    return root, mask_dir, overlay_dir, zip_dir

# ---------- 파싱 & 표준 저장명(2자리) + 레거시 호환 ----------
def fmt_stage(n):  # S01
    return f"S{int(n):02d}"

def fmt_time(n):   # t00
    return f"t{int(n):02d}"

def parse_tokens(basename: str, timelapse: bool):
    name = os.path.splitext(basename)[0]
    ch = None
    m_ch = re.search(r'(?:[_-](\d+)$)|(?:[_-](?:ch|c)(\d+)$)', name, flags=re.IGNORECASE)
    if m_ch:
        ch = int(next(g for g in m_ch.groups() if g is not None))
    m_s = re.search(r'(?i)S(\d+)', name)
    s_num = int(m_s.group(1)) if m_s else None
    t_num = None
    if timelapse:
        m_t = re.search(r'(?i)t(\d+)', name)
        t_num = int(m_t.group(1)) if m_t else None
    return s_num, t_num, ch

def clean_base_for_save(basename: str, timelapse: bool):
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    if s_num is None:
        name = os.path.splitext(basename)[0]
        return re.sub(r'([_-])\d+$', '', name)
    if timelapse and (t_num is not None):
        return f"{fmt_stage(s_num)}_{fmt_time(t_num)}"
    return fmt_stage(s_num)

def find_json_path(outdir, basename, timelapse):
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    norm = clean_base_for_save(basename, timelapse)
    candidates = [os.path.join(outdir, norm + ".json")]
    if s_num is not None:
        legacy = f"S{int(s_num)}"
        if timelapse and t_num is not None:
            legacy = f"{legacy}_t{int(t_num)}"
        candidates.append(os.path.join(outdir, legacy + ".json"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

# ------------- channel detection & filtering -------------
CH_PATTERNS = [
    re.compile(r'[-_](\d+)(?=\.|$)', re.IGNORECASE),
    re.compile(r'(?:ch|c)(\d+)(?=[._-]|$)', re.IGNORECASE),
]

def detect_channel(base):
    name = os.path.splitext(base)[0]
    for pat in CH_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    low = name.lower()
    if any(k in low for k in ['ecfp', 'cfp', 'donor']):
        return 1
    if any(k in low for k in ['yfret', 'fret', 'acceptor', 'yfp']):
        return 2
    return None

def build_channel_map(files_all, s_num, t_num, timelapse):
    cmap = {}
    for _p in files_all:
        _base = os.path.basename(_p)
        _s, _t, _c = parse_tokens(_base, timelapse)
        if _c is None:
            continue
        if _s == s_num and (((_t is None) and (t_num is None)) or (_t == t_num)):
            cmap[int(_c)] = _p
    return dict(sorted(cmap.items()))

# -------------------- Pseudocolor & normalization --------------------
PCOLORS = {
    "grayscale": None,
    "cyan":   np.array([0.0, 1.0, 1.0]),
    "blue":   np.array([0.0, 0.0, 1.0]),
    "green":  np.array([0.0, 1.0, 0.0]),
    "red":    np.array([1.0, 0.0, 0.0]),
    "yellow": np.array([1.0, 1.0, 0.0]),
}

def normalize_to_rgb(img, mode="grayscale", p_low=1.0, p_high=99.0, gamma=1.0, invert=False):
    # (유틸 함수)
    vmin = np.percentile(img, float(p_low))
    vmax = np.percentile(img, float(p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = (img - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    if gamma is not None and gamma > 0:
        x = np.power(x, 1.0/float(gamma))
    if invert:
        x = 1.0 - x
    if mode not in PCOLORS or PCOLORS[mode] is None:
        return np.dstack([x, x, x])
    rgb = PCOLORS[mode].reshape(1, 1, 3)
    return np.clip(x[..., None] * rgb, 0, 1)

# -------------------- TIFF 리더 --------------------
def read_tiff_with_fallback(path, page=0):
    try:
        return imread(path, key=page)
    except Exception as e:
        msg = str(e)
        if ("imagecodecs" not in msg) and ("COMPRESSION" not in msg):
            raise
        with Image.open(path) as im:
            try:
                im.seek(page)
            except EOFError:
                im.seek(0)
            return np.array(im)

# -------------------- Auto-segmentation inside polygon --------------------
def polygon_area(xy):
    x = xy[:, 0]; y = xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def segment_inside_polygon(img, poly, thr_param=90.0, min_area=40, tolerance=1.0,
                           mode: str = "percentile"):
    H, W = img.shape[:2]
    path = mpath.Path(np.asarray(poly))
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    inside = path.contains_points(pts).reshape(H, W)

    vals = img[inside]
    if vals.size == 0:
        return None, None, None

    thr_param = float(thr_param)

    if mode.lower() == "bnd":
        m = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        if (s <= 0) or (not np.isfinite(s)):
            thr = float(np.percentile(vals, 90.0))
        else:
            thr = m + thr_param * s
    else:
        thr = float(np.percentile(vals, thr_param))

    cand = (img >= thr) & inside

    lab, n = ndi.label(cand)
    if n == 0:
        return thr, None, None

    sizes = ndi.sum(cand, lab, index=np.arange(1, n+1))
    k = int(np.argmax(sizes)) + 1
    mask = (lab == k)
    mask = ndi.binary_fill_holes(mask)

    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return thr, None, None

    polys = []
    for c in contours:
        xy = np.c_[c[:, 1], c[:, 0]]
        area = polygon_area(xy)
        if area >= float(min_area):
            xy_s = approximate_polygon(xy, tolerance=float(tolerance))
            if len(xy_s) >= 3:
                polys.append((area, xy_s))

    if not polys:
        return thr, None, None

    best = max(polys, key=lambda t: t[0])[1]
    return thr, mask, best

# -------------------- Interactive ROI GUI --------------------
def polygon_centroid(points):
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        return float(P[:, 0].mean()), float(P[:, 1].mean())
    x = P[:, 0]; y = P[:, 1]
    x1 = np.r_[x, x[0]]; y1 = np.r_[y, y[0]]
    cross = x1[:-1] * y1[1:] - x1[1:] * y1[:-1]
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-6:
        return float(x.mean()), float(y.mean())
    cx = (1/(6*A)) * np.sum((x1[:-1] + x1[1:]) * cross)
    cy = (1/(6*A)) * np.sum((y1[:-1] + y1[1:]) * cross)
    return float(cx), float(cy)

# ----------------------- startup GUI -----------------------
# [FIX] Moved startup_gui here before Main Class for clarity and scope
def startup_gui(lang: str = LANG_DEFAULT):
    root = Tk()
    root.title(t("title_startup", "ROI Drawer / Channel / Mode", lang=lang))
    root.resizable(False, False)

    folder_var      = StringVar()
    ch_var          = StringVar(value="3")
    mode_var        = StringVar(value="new")
    stage_var       = StringVar(value="")
    time_var        = StringVar(value="")
    include_no_roi  = BooleanVar(value=False)
    timelapse_var   = BooleanVar(value=False)
    tol_var         = DoubleVar(value=1.0)
    min_area_var    = StringVar(value="40")
    color_var       = StringVar(value="grayscale")
    bnd_mode_var    = BooleanVar(value=False)

    def browse():
        p = filedialog.askdirectory(title=t("label_folder", "TIF folder", lang=lang))
        if p:
            folder_var.set(p)

    def update_widgets():
        is_edit = (mode_var.get() == "edit")
        is_time = bool(timelapse_var.get()) and is_edit
        e_stage.configure(state=(NORMAL if is_edit else DISABLED))
        e_time.configure(state=(NORMAL if is_time else DISABLED))
        cb_include.configure(state=(NORMAL if is_edit else DISABLED))

    def on_any_change(*_):
        update_widgets()

    def on_ok():
        path = folder_var.get().strip()
        if not path or not os.path.isdir(path):
            messagebox.showerror(t("err_title", "Error", lang=lang), t("err_folder", "Select a valid folder", lang=lang)); return
        try:
            ch = int(ch_var.get().strip())
            if ch < 0 or ch > 999:
                raise ValueError
        except Exception:
            messagebox.showerror(t("err_title", "Error", lang=lang), t("err_channel", "Channel number must be 0~999", lang=lang)); return

        s_num = t_num = None
        if mode_var.get() == "edit":
            st = stage_var.get().strip()
            if st:
                try:
                    s_num = int(st)
                except Exception:
                    messagebox.showerror(t("err_title", "Error", lang=lang), t("err_stage", "Stage must be a number", lang=lang)); return
            if timelapse_var.get():
                tt = time_var.get().strip()
                if tt:
                    try:
                        t_num = int(tt)
                    except Exception:
                        messagebox.showerror(t("err_title", "Error", lang=lang), t("err_time", "Timepoint must be a number", lang=lang)); return

        try:
            tol = float(tol_var.get())
            if not (0.1 <= tol <= 5.0):
                raise ValueError
        except Exception:
            messagebox.showerror(t("err_title", "Error", lang=lang), t("err_tol", "Tolerance must be 0.1~5.0", lang=lang)); return

        try:
            ma = float(min_area_var.get())
            if ma < 0:
                raise ValueError
        except Exception:
            messagebox.showerror(t("err_title", "Error", lang=lang), t("err_min_area", "Min area must be >= 0", lang=lang)); return

        root.selected = {
            "folder": path, "channel": ch, "mode": mode_var.get(),
            "stage": s_num, "time": t_num, "include_no_roi": bool(include_no_roi.get()),
            "timelapse": bool(timelapse_var.get()),
            "tolerance": tol, "min_area": ma,
            "color_mode": color_var.get(),
            "bnd_mode": bool(bnd_mode_var.get())
        }
        root.destroy()

    def on_cancel():
        root.selected = None
        root.destroy()

    pad = {"padx": 8, "pady": 6}

    Label(root, text=t("label_folder", "TIF folder", lang=lang)).grid(row=0, column=0, sticky="w", **pad)
    f1 = Frame(root); f1.grid(row=0, column=1, sticky="ew", **pad)
    Entry(f1, textvariable=folder_var, width=50).pack(side="left")
    Button(f1, text=t("btn_browse", "Browse", lang=lang), width=8, command=browse).pack(side="left", padx=6)

    Label(root, text=t("label_channel", "Start channel (number)", lang=lang)).grid(row=1, column=0, sticky="w", **pad)
    Entry(root, textvariable=ch_var, width=10).grid(row=1, column=1, sticky="w", **pad)

    Label(root, text=t("label_mode", "Mode", lang=lang)).grid(row=2, column=0, sticky="w", **pad)
    f2 = Frame(root); f2.grid(row=2, column=1, sticky="w", **pad)
    Radiobutton(f2, text=t("mode_new", "New", lang=lang),  variable=mode_var, value="new", command=on_any_change).pack(side="left", padx=4)
    Radiobutton(f2, text=t("mode_edit", "Edit", lang=lang), variable=mode_var, value="edit", command=on_any_change).pack(side="left", padx=4)

    Checkbutton(root, text=t("timelapse_cb", "Timelapse (filename=SXX_TXX_X)", lang=lang), variable=timelapse_var, command=on_any_change).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))

    Label(root, text=t("label_stage", "Stage (number)", lang=lang)).grid(row=4, column=0, sticky="w", **pad)
    e_stage = Entry(root, textvariable=stage_var, width=10); e_stage.grid(row=4, column=1, sticky="w", **pad)
    Label(root, text=t("label_time", "Timepoint (number)", lang=lang)).grid(row=5, column=0, sticky="w", **pad)
    e_time = Entry(root, textvariable=time_var, width=10); e_time.grid(row=5, column=1, sticky="w", **pad)

    cb_include = Checkbutton(root, text=t("cb_include", "Include files without ROI in Edit mode", lang=lang), variable=include_no_roi, command=on_any_change)
    cb_include.grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    Label(root, text=t("label_tol", "Boundary simplify tolerance", lang=lang)).grid(row=7, column=0, sticky="w", **pad)
    s_tol = Scale(root, variable=tol_var, from_=0.1, to=5.0, resolution=0.1, orient=HORIZONTAL, length=260)
    s_tol.grid(row=7, column=1, sticky="w", **pad)
    Label(root, text=t("label_min_area", "Min area (px^2)", lang=lang)).grid(row=8, column=0, sticky="w", **pad)
    Entry(root, textvariable=min_area_var, width=10).grid(row=8, column=1, sticky="w", **pad)

    Label(root, text=t("label_color", "Pseudocolor", lang=lang)).grid(row=9, column=0, sticky="w", **pad)
    OptionMenu(root, color_var, "grayscale", "cyan", "blue", "green", "red", "yellow").grid(row=9, column=1, sticky="w", **pad)
    # 🔸 BND-like 모드 토글
    Checkbutton(
        root,
        text="BND-like threshold (mean + α·std)",
        variable=bnd_mode_var
    ).grid(row=10, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    fbtn = Frame(root); fbtn.grid(row=11, column=0, columnspan=2, pady=10)
    Button(fbtn, text=t("btn_ok", "OK", lang=lang), width=12, command=on_ok).pack(side="left", padx=6)
    Button(fbtn, text=t("btn_cancel", "Cancel", lang=lang), width=12, command=on_cancel).pack(side="left", padx=6)

    update_widgets()
    root.mainloop()
    if not getattr(root, "selected", None):
        raise SystemExit(t("exit_cancelled", "Operation was cancelled.", lang=lang))
    return root.selected

# =========================================================
# ROI Manager Window Class
# =========================================================
class ROIManagerWindow:
    def __init__(self, annotator):
        self.annot = annotator
        self.root = Toplevel()
        self.root.title("ROI Manager")
        self.root.geometry("250x450")
        self.root.attributes("-topmost", True)
        
        # 리스트박스 & 스크롤바
        frame_list = Frame(self.root)
        frame_list.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.scrollbar = Scrollbar(frame_list, orient="vertical")
        self.listbox = Listbox(frame_list, selectmode=SINGLE, 
                               yscrollcommand=self.scrollbar.set, font=("Consolas", 10))
        self.scrollbar.config(command=self.listbox.yview)
        
        self.scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        
        # 버튼들
        btn_frame = Frame(self.root)
        btn_frame.pack(side="bottom", fill="x", padx=5, pady=5)
        
        # [NEW] Redraw Button added
        Button(btn_frame, text="Redraw (수정)", command=self.redraw_selected, bg="#ffffdd").pack(side="top", fill="x", padx=2, pady=2)
        
        f_btm = Frame(btn_frame)
        f_btm.pack(side="top", fill="x", pady=2)
        Button(f_btm, text="Delete", command=self.delete_selected, bg="#ffdddd").pack(side="left", expand=True, fill="x", padx=1)
        Button(f_btm, text="Save & Close", command=self.save_close, bg="#ddffdd").pack(side="left", expand=True, fill="x", padx=1)

        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_req)
        self.listbox.bind('<Delete>', lambda e: self.delete_selected())
        
        self.refresh_list()

    def refresh_list(self):
        sel = self.listbox.curselection()
        old_idx = sel[0] if sel else -1
        
        self.listbox.delete(0, END)
        for i in range(len(self.annot.rois)):
            roi_item = self.annot.rois[i]
            if roi_item is None:
                self.listbox.insert(END, f"ROI #{i+1} [Redrawing...]")
            else:
                self.listbox.insert(END, f"ROI #{i+1}")
            
        if old_idx >= 0:
            new_len = self.listbox.size()
            if new_len > 0:
                new_idx = min(old_idx, new_len - 1)
                self.listbox.selection_set(new_idx)
                self.listbox.see(new_idx)

    def on_select(self, event):
        sel = self.listbox.curselection()
        if not sel: 
            self.annot.highlight_roi(-1)
            return
        idx = int(sel[0])
        self.annot.highlight_roi(idx)

    def redraw_selected(self):
        sel = self.listbox.curselection()
        if not sel: return
        idx = int(sel[0])
        self.annot.start_redraw(idx)
        self.refresh_list()

    def delete_selected(self):
        sel = self.listbox.curselection()
        if not sel: return
        idx = int(sel[0])
        self.annot.delete_roi_by_index(idx)
        self.refresh_list()
        sel_new = self.listbox.curselection()
        if sel_new:
            self.annot.highlight_roi(int(sel_new[0]))
        else:
            self.annot.highlight_roi(-1)

    def save_close(self):
        self.annot.close_and_save()

    def on_close_req(self):
        self.annot.close_and_save()

# -------------------- Main Annotator Class --------------------
class ROIAnnotator:
    def __init__(self, image, title="ROI Drawer (Auto-Seg)", init_thresh_p: float = 70.0,
                 initial_rois=None, tolerance: float = 1.0, min_area: float = 40.0,
                 color_mode: str = "grayscale",
                 last_view: dict | None = None,
                 bnd_mode: bool = False,
                 # [NEW] 모드 인자 추가 (기본값: 'new')
                 current_mode: str = 'new'):
        self.image = image.astype(np.float32, copy=False)
        self.rois = [] if initial_rois is None else [np.asarray(p, float) for p in initial_rois]
        self.current_selector = None
        self.force_quit = False
        self.roi_manager = None
        self.highlight_patch = None
        self.replace_idx = None 
        
        # [NEW] 현재 모드 저장
        self.current_mode = current_mode

        self.thresh_p = float(init_thresh_p)
        self.tolerance = float(tolerance)
        self.min_area = float(min_area)
        self.bnd_mode = bool(bnd_mode)

        if last_view is None: last_view = {}
        self.p_low  = float(last_view.get('p_low', 1.0))
        self.p_high = float(last_view.get('p_high', 99.0))
        self.gamma  = float(last_view.get('gamma', 1.0))
        self.invert = bool(last_view.get('invert', False))
        self.color_mode = str(last_view.get('color_mode', color_mode))

        self.use_bandpass   = bool(last_view.get('use_bandpass', False))
        self.sigma_small    = float(last_view.get('sigma_small', 1.2))
        self.sigma_large    = float(last_view.get('sigma_large', 9.0))
        self.use_unsharp    = bool(last_view.get('use_unsharp', False))
        self.unsharp_amount = float(last_view.get('unsharp_amount', 0.7))
        self.unsharp_radius = float(last_view.get('unsharp_radius', 2.0))
        self.use_clahe      = bool(last_view.get('use_clahe', False))
        self.clahe_clip     = float(last_view.get('clahe_clip', 0.03))
        self.edge_overlay   = bool(last_view.get('edge_overlay', False))
        self.local_norm     = False

        self.fig, self.ax = plt.subplots()
        maximize_figure_window(self.fig)
        self._update_bg_rgb()
        self.ax.imshow(self.bg_rgb)
        self.ax.set_title(title)
        self.ax.set_axis_off()
        self._draw_hud()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        if self.rois:
            self._redraw_all()
        self._attach_view_tool_to_toolbar()

    def highlight_roi(self, idx):
        if self.highlight_patch:
            try: self.highlight_patch.remove()
            except: pass
            self.highlight_patch = None
            
        if idx < 0 or idx >= len(self.rois):
            self.fig.canvas.draw_idle()
            return
        
        if self.rois[idx] is None:
            self.fig.canvas.draw_idle()
            return

        poly = self.rois[idx]
        self.highlight_patch = MplPolygon(poly, closed=True, fill=False, edgecolor='red', linewidth=3.0, alpha=1.0)
        self.ax.add_patch(self.highlight_patch)
        self.fig.canvas.draw_idle()

    def delete_roi_by_index(self, idx):
        if 0 <= idx < len(self.rois):
            del self.rois[idx]
            self._redraw_all()

    def close_and_save(self):
        plt.close(self.fig)
        if self.roi_manager:
            try: self.roi_manager.root.destroy()
            except: pass

    def export_view_params(self) -> dict:
        return {
            'p_low': self.p_low, 'p_high': self.p_high, 'gamma': self.gamma, 'invert': self.invert,
            'color_mode': self.color_mode,
            'use_bandpass': self.use_bandpass, 'sigma_small': self.sigma_small, 'sigma_large': self.sigma_large,
            'use_unsharp': self.use_unsharp, 'unsharp_amount': self.unsharp_amount, 'unsharp_radius': self.unsharp_radius,
            'use_clahe': self.use_clahe, 'clahe_clip': self.clahe_clip, 'edge_overlay': self.edge_overlay
        }

    def _get_parent_tk(self):
        try: return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except: return None

    def _ask_yes_no_modal(self, title, message, default_yes=True):
        parent = self._get_parent_tk()
        win = _tk.Toplevel(master=parent)
        win.title(title)
        win.transient(parent)
        win.grab_set()
        try: win.attributes("-topmost", True)
        except: pass
        
        frm = _tk.Frame(win, padx=12, pady=10)
        frm.pack(fill="both", expand=True)
        _tk.Label(frm, text=message, justify="left").pack(anchor="w")
        
        btn_frm = _tk.Frame(frm)
        btn_frm.pack(pady=(10, 0), fill="x")
        ans = {"val": default_yes}

        def _yes(*_): ans["val"] = True; win.destroy()
        def _no(*_): ans["val"] = False; win.destroy()

        b1 = _tk.Button(btn_frm, text="Yes", width=10, command=_yes)
        b2 = _tk.Button(btn_frm, text="No", width=10, command=_no)
        b1.pack(side="left", padx=5); b2.pack(side="left", padx=5)

        for k in ("y", "Y", "Return", "space"): win.bind(f"<{k}>", _yes)
        for k in ("n", "N", "Escape"): win.bind(f"<{k}>", _no)

        win.lift(); b1.focus_set(); win.wait_window()
        try: parent.attributes("-topmost", False)
        except: pass
        return bool(ans["val"])
    
    def _ask_float_modal(self, title, prompt, initial=70.0, minv=0.0, maxv=100.0, step=0.5):
        parent = self._get_parent_tk()
        win = _tk.Toplevel(master=parent)
        win.title(title)
        win.transient(parent)
        win.grab_set()
        try: win.attributes("-topmost", True)
        except: pass

        frm = _tk.Frame(win, padx=12, pady=10)
        frm.pack(fill="both", expand=True)
        _tk.Label(frm, text=prompt, justify="left").pack(anchor="w", pady=(0,6))

        var = _tk.StringVar(value=f"{float(initial):.1f}")
        ent = _tk.Entry(frm, textvariable=var, width=12, justify="right")
        ent.pack(anchor="w")

        btn_frm = _tk.Frame(frm); btn_frm.pack(pady=(10,0), fill="x")
        ans = {"val": None}

        def _ok(*_):
            try: x = float(var.get())
            except: x = None
            if x is not None and (x < minv or x > maxv): x = None
            if x is None: 
                try: win.bell()
                except: pass
                return
            ans["val"] = x
            win.destroy()

        def _cancel(*_): ans["val"] = None; win.destroy()

        b_ok = _tk.Button(btn_frm, text="OK", width=8, command=_ok)
        b_no = _tk.Button(btn_frm, text="Cancel", width=8, command=_cancel)
        b_ok.pack(side="left", padx=5); b_no.pack(side="left", padx=5)

        win.bind("<Return>", _ok); win.bind("<Escape>", _cancel)
        win.update_idletasks(); win.lift(); ent.focus_set(); ent.selection_range(0, _tk.END)
        win.wait_window()
        try: parent.attributes("-topmost", False)
        except: pass
        return ans["val"]

    def _hud_str(self):
        instructions = t("hud_instr", default="", lang=LANG_CURRENT)
        inv = "ON" if self.invert else "off"
        bp  = f"BP={'ON' if self.use_bandpass else 'off'}"
        thr_info = f"ThreshMode={'BND' if self.bnd_mode else 'Percentile'}({self.thresh_p:.1f})"
        view_line = (f"[View] p_low={self.p_low:.1f}%, p_high={self.p_high:.1f}%, "
                     f"gamma={self.gamma:.2f}, invert={inv} | {bp} | {thr_info}")
        return (instructions + "\n" + view_line) if instructions else view_line

    def _draw_hud(self):
        instructions = t("hud_instr", default="", lang=LANG_CURRENT)
        self.ax.figure.text(
            0.02, 0.02, instructions,
            color="w", fontsize=9,
            bbox=dict(facecolor="k", alpha=0.4)
        )
        self._hud_txt = self.ax.figure.text(
            0.02, 0.96, self._hud_str(),
            color="yellow", fontsize=10,
            bbox=dict(facecolor="k", alpha=0.35)
        )

    def _render_pipeline(self, img):
        im = img
        if self.use_bandpass:
            im = ndi.gaussian_filter(im, self.sigma_small) - ndi.gaussian_filter(im, self.sigma_large)
        if self.use_unsharp:
            im = im + self.unsharp_amount * (im - ndi.gaussian_filter(im, self.unsharp_radius))
        return im

    def _to_rgb(self, x01):
        if self.use_clahe:
            x01 = exposure.equalize_adapthist(x01, clip_limit=self.clahe_clip)
        if self.color_mode not in PCOLORS or PCOLORS[self.color_mode] is None:
            rgb = np.dstack([x01, x01, x01])
        else:
            rgb = np.clip(x01[..., None] * PCOLORS[self.color_mode].reshape(1,1,3), 0, 1)
        if self.edge_overlay:
            ed = filters.sobel(x01)
            rgb[...,1] = np.clip(rgb[...,1] + ed*0.8, 0, 1)
        return rgb

    def _update_bg_rgb(self, poly_for_local=None):
        im = self._render_pipeline(self.image)
        if self.local_norm and poly_for_local is not None:
            H, W = im.shape[:2]
            path = mpath.Path(np.asarray(poly_for_local))
            yy, xx = np.mgrid[0:H, 0:W]
            inside = path.contains_points(
                np.vstack((xx.ravel(), yy.ravel())).T
            ).reshape(H, W)
            vals = im[inside]
            if vals.size > 10:
                vmin = np.percentile(vals, self.p_low)
                vmax = np.percentile(vals, self.p_high)
            else:
                vmin = np.percentile(im, self.p_low)
                vmax = np.percentile(im, self.p_high)
        else:
            vmin = np.percentile(im, self.p_low)
            vmax = np.percentile(im, self.p_high)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        x = np.clip((im - vmin) / (vmax - vmin), 0, 1)
        x = np.power(x, 1.0/max(self.gamma, 1e-6))
        if self.invert:
            x = 1.0 - x
        self.bg_rgb = self._to_rgb(x)

    def _refresh_view(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self._update_bg_rgb()
        self.ax.imshow(self.bg_rgb)
        self._draw_hud()
        for i, poly in enumerate(self.rois, 1):
            if poly is None: continue
            C = np.asarray(poly)
            self.ax.add_patch(MplPolygon(C, closed=True, fill=False, ec='lime', lw=2.0))
            cx, cy = polygon_centroid(C)
            t = self.ax.text(
                cx, cy, str(i),
                color='yellow', fontsize=12, fontweight='bold',
                ha='center', va='center', zorder=9999
            )
            try: t.set_path_effects([mpatheffects.withStroke(linewidth=2, foreground='black')])
            except: pass
        
        if self.roi_manager:
            sel = self.roi_manager.listbox.curselection()
            if sel: self.highlight_roi(int(sel[0]))

        self.fig.canvas.draw_idle()

    def _redraw_all(self):
        self._refresh_view()

    def start_redraw(self, idx):
        if idx < 0 or idx >= len(self.rois): return
        self.replace_idx = idx
        self.rois[idx] = None
        self._redraw_all()
        self.start_polygon()

    def start_polygon(self):
        if self.current_selector is not None:
            try: 
                self.current_selector.disconnect_events()
                self.current_selector.set_visible(False)
            except Exception: pass
            self.current_selector = None
            self.fig.canvas.draw_idle()
            
        try:
            self.current_selector = PolygonSelector(
                self.ax, 
                self.on_select, 
                useblit=True,
                lineprops=dict(color='yellow', linewidth=2, alpha=0.9),
                markerprops=dict(marker='o', markersize=5, mec='yellow', mfc='yellow', alpha=0.9)
            )
        except TypeError:
            try:
                self.current_selector = PolygonSelector(
                    self.ax, 
                    self.on_select, 
                    useblit=True,
                    props=dict(color='yellow', linewidth=2, alpha=0.9),
                    handle_props=dict(marker='o', markersize=5, mec='yellow', mfc='yellow', alpha=0.9)
                )
            except TypeError:
                self.current_selector = PolygonSelector(self.ax, self.on_select, useblit=True)
        
        self.fig.canvas.draw_idle()

    def on_select(self, verts):
        rough_poly = np.array(verts, dtype=float)
        if self.current_selector:
            try:
                self.current_selector.disconnect_events()
                self.current_selector.set_visible(False)
            except: pass
            self.current_selector = None
            
        while True:
            thr, mask, poly = segment_inside_polygon(
                self.image, rough_poly, thr_param=self.thresh_p,
                min_area=self.min_area, tolerance=self.tolerance,
                mode=("bnd" if self.bnd_mode else "percentile")
            )

            self.ax.clear()
            self.ax.set_axis_off()
            self._update_bg_rgb(rough_poly if self.local_norm else None)
            self.ax.imshow(self.bg_rgb)

            P = rough_poly
            self.ax.plot(P[:, 0], P[:, 1], '-', color='yellow', linewidth=1.5, alpha=0.8)
            self.ax.plot([P[-1, 0], P[0, 0]], [P[-1, 1], P[0, 1]], '-', color='yellow', linewidth=1.0, alpha=0.8)

            if poly is not None:
                C = np.asarray(poly)
                self.ax.plot(C[:, 0], C[:, 1], '-', color='lime', linewidth=2.5, alpha=0.95)
                self.ax.plot([C[-1, 0], C[0, 0]], [C[-1, 1], C[0, 1]], '-', color='lime', linewidth=2.5, alpha=0.95)
                title = f"Accepted? (Threshold p={self.thresh_p:.1f})"
            else:
                title = f"No ROI found (Threshold p={self.thresh_p:.1f})"

            self.ax.set_title(title)
            self._draw_hud()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

            if self.bnd_mode: msg = f"BND-mode α={self.thresh_p:.2f} Accept?"
            else: msg = f"Percentile p={self.thresh_p:.1f} Accept?"

            ans = False
            if poly is not None:
                ans = self._ask_yes_no_modal("Confirm", msg, default_yes=True)

            if ans and poly is not None:
                new_poly = np.asarray(poly)
                if self.replace_idx is not None:
                    self.rois[self.replace_idx] = new_poly
                    self.replace_idx = None
                else:
                    self.rois.append(new_poly)
                
                if self.roi_manager:
                    self.roi_manager.refresh_list()
                    if self.replace_idx is None: 
                        self.roi_manager.listbox.selection_clear(0, END)
                        self.roi_manager.listbox.selection_set(END)
                        self.roi_manager.listbox.see(END)
                break
            else:
                newp = self._ask_float_modal("Retry Threshold", "New value:", initial=self.thresh_p)
                if newp is None: break
                self.thresh_p = float(newp)

        self._redraw_all()

    def on_scroll(self, event):
        step = 0.25
        width = float(self.p_high - self.p_low)
        width = max(0.1, min(width, 99.9))
        if event.button == 'up':
            new_low = min(100.0 - width - 0.1, self.p_low + step)
            new_high = new_low + width
        else:
            new_low = max(0.0, self.p_low - step)
            new_high = new_low + width
            if new_high > 100.0: new_high = 100.0; new_low = max(0.0, new_high - width)
        self.p_low = float(new_low); self.p_high = float(new_high)
        self._refresh_view()

    def on_key(self, event):
        k = (event.key or '').strip()
        if k in ('ctrl+q', 'cmd+q', 'q'): self.close_and_save(); return
        if k.lower() == 'p': self.start_polygon(); return
        if k.lower() == 'u':
            if self.rois:
                self.rois.pop()
                if self.roi_manager: self.roi_manager.refresh_list()
                self._redraw_all()
            return
        if k.lower() == 'c':
            self.rois.clear()
            if self.roi_manager: self.roi_manager.refresh_list()
            self._redraw_all(); return
        if k in ('ctrl+s', 'cmd+s', 'ctrl+enter', 'cmd+enter'): self.close_and_save(); return
        if k == 'delete':
            if self.roi_manager: self.roi_manager.delete_selected()
            return
        if k in ('a', 'A'): self.p_low = max(0.0, self.p_low - 1.0); self._refresh_view(); return
        if k in ('d', 'D'): self.p_low = min(self.p_high-0.1, self.p_low + 1.0); self._refresh_view(); return
        if k in ('s', 'S'): self.p_high = max(self.p_low+0.1, self.p_high - 1.0); self._refresh_view(); return
        if k in ('f', 'F'): self.p_high = min(100.0, self.p_high + 1.0); self._refresh_view(); return
        if k == 'g': self.gamma = max(0.1, self.gamma - 0.1); self._refresh_view(); return
        if k == 'G': self.gamma = min(5.0, self.gamma + 0.1); self._refresh_view(); return
        if k.lower() == 'i': self.invert = not self.invert; self._refresh_view(); return
        if k == '0': self.color_mode = 'grayscale'; self._refresh_view(); return
        if k == '1': self.color_mode = 'cyan';      self._refresh_view(); return
        if k == '2': self.color_mode = 'blue';      self._refresh_view(); return
        if k == '3': self.color_mode = 'green';     self._refresh_view(); return
        if k == '4': self.color_mode = 'red';       self._refresh_view(); return
        if k == '5': self.color_mode = 'yellow';    self._refresh_view(); return
        if k.lower() == 'r': self.p_low, self.p_high, self.gamma, self.invert = 1.0, 99.0, 1.0, False; self._refresh_view(); return
        if k.lower() == 'h':
            msg = "Help: P=New ROI, Del=Delete Selected, Q=Quit/Save, MouseWheel=Brightness"
            parent = self._get_parent_tk()
            messagebox.showinfo("Help", msg, parent=parent); return

    def _attach_view_tool_to_toolbar(self):
        try:
            backend = matplotlib.get_backend().lower()
            if 'tkagg' not in backend: return
            toolbar = self.fig.canvas.manager.toolbar
        except: return
        if getattr(toolbar, "_roi_view_tool_patched", False): return

        def open_view_tool(*args, **kwargs):
            if hasattr(self, "_view_tool_win") and self._view_tool_win.winfo_exists():
                self._view_tool_win.lift(); return
            parent = self._get_parent_tk()
            win = _tk.Toplevel(parent); win.title("View / Gamma / Percentile")
            try: win.attributes("-topmost", True)
            except: pass
            frm = _tk.Frame(win); frm.pack(padx=10, pady=10)
            var_p_low = _tk.DoubleVar(value=self.p_low)
            var_p_high = _tk.DoubleVar(value=self.p_high)
            var_gamma = _tk.DoubleVar(value=self.gamma)
            _tk.Label(frm, text="p_low (%)").grid(row=0, column=0, sticky="w")
            s_low = _tk.Scale(frm, from_=0, to=50, resolution=0.5, orient="horizontal", length=260, variable=var_p_low)
            s_low.grid(row=0, column=1, padx=6, pady=4)
            _tk.Label(frm, text="p_high (%)").grid(row=1, column=0, sticky="w")
            s_high = _tk.Scale(frm, from_=50, to=100, resolution=0.5, orient="horizontal", length=260, variable=var_p_high)
            s_high.grid(row=1, column=1, padx=6, pady=4)
            _tk.Label(frm, text="gamma").grid(row=2, column=0, sticky="w")
            s_gamma = _tk.Scale(frm, from_=0.2, to=3.0, resolution=0.05, orient="horizontal", length=260, variable=var_gamma)
            s_gamma.grid(row=2, column=1, padx=6, pady=4)
            def apply_and_refresh(*_):
                pl = float(var_p_low.get()); ph = float(var_p_high.get())
                if pl >= ph: pl = min(pl, ph - 0.1)
                self.p_low = max(0.0, min(ph-0.1, pl))
                self.p_high = max(self.p_low+0.1, min(100.0, ph))
                self.gamma = max(0.05, float(var_gamma.get()))
                self._refresh_view()
            for v in (var_p_low, var_p_high, var_gamma): v.trace_add("write", apply_and_refresh)
            def on_close(): self._view_tool_win = None; win.destroy()
            win.protocol("WM_DELETE_WINDOW", on_close); self._view_tool_win = win

        try:
            toolbar.configure_subplots = open_view_tool
            btn = None
            for key in ("Subplots", "subplots", "Configure subplots"):
                if hasattr(toolbar, "_buttons") and key in toolbar._buttons: btn = toolbar._buttons[key]; break
            if btn is not None:
                try: btn.configure(command=open_view_tool)
                except: pass
                try: btn.tooltip_string = "View / Gamma / Percentile"
                except: pass
            toolbar._roi_view_tool_patched = True
        except: pass

    def show(self):
        # [NEW] ROI Manager Window Condition
        # 'edit' 모드일 때만 ROI Manager 창을 띄움
        if self.current_mode == 'edit':
            if _tk._default_root is None:
                root = _tk.Tk(); root.withdraw()
            self.roi_manager = ROIManagerWindow(self)
        
        plt.show()
        
        final_rois = [r for r in self.rois if r is not None]
        return final_rois, float(self.thresh_p), self.export_view_params()

# -------------------- Channel-switching annotator --------------------
class ROIAnnotatorCH(ROIAnnotator):
    def __init__(self, image_path_by_channel, selected_channel, *args, **kwargs):
        self.channel_map = dict(sorted({int(k): v for k, v in image_path_by_channel.items()}.items()))
        _fallback = kwargs.pop('image', None)
        if self.channel_map:
            self.selected_channel = selected_channel if selected_channel in self.channel_map else sorted(self.channel_map.keys())[0]
            img0 = self._load_image(self.selected_channel)
        else:
            self.selected_channel = selected_channel
            if isinstance(_fallback, np.ndarray): img0 = _fallback
            else: img0 = np.zeros((512, 512), np.float32)
        super().__init__(image=img0, *args, **kwargs)
        self.ax.set_title(self.ax.get_title() + f"  |  CH={self.selected_channel}")
        self._channel_win = None
        self._build_channel_window()
        try: self.fig.canvas.mpl_connect('close_event', self._on_close)
        except: pass

    def _on_close(self, evt):
        try:
            if self._channel_win: self._channel_win.destroy()
        except: pass

    def _load_image(self, ch):
        path = self.channel_map[int(ch)]
        img = read_tiff_with_fallback(path)
        if getattr(img, 'ndim', 2) > 2: img = img[..., 0] if img.ndim == 3 else img[0, ...]
        return img.astype(np.float32, copy=False)

    def _apply_channel_change(self, ch):
        if not self.channel_map: return
        self.selected_channel = int(ch)
        self.image = self._load_image(self.selected_channel)
        try:
            base = re.sub(r"\s\|\sCH=.*$", "", self.ax.get_title())
            self.ax.set_title(base + f"  |  CH={self.selected_channel}")
        except: pass
        self._refresh_view()
        self.fig.canvas.draw_idle()

    def _cycle_channel(self, step):
        ks = list(sorted(self.channel_map.keys()))
        if not ks: return
        i = ks.index(self.selected_channel)
        self._apply_channel_change(ks[(i + step) % len(ks)])
        if hasattr(self, '_ch_var'): self._ch_var.set(str(self.selected_channel))

    def _build_channel_window(self):
        if len(self.channel_map) <= 1: return
        try: parent = self.fig.canvas.get_tk_widget().winfo_toplevel()
        except: return
        win = _tk.Toplevel(master=parent)
        win.title("Channel Select")
        base_w, base_h = 300, 180
        win.geometry(f"{base_w}x{base_h}")
        try: win.attributes('-topmost', True)
        except: pass
        frm = _tk.Frame(win); frm.pack(padx=12, pady=12)
        lbl = _tk.Label(frm, text="Select Channel")
        lbl.grid(row=0, column=0, sticky='w')
        self._ch_var = _tk.StringVar(value=str(self.selected_channel))
        opts = [str(k) for k in sorted(self.channel_map.keys())]
        om = _tk.OptionMenu(frm, self._ch_var, *opts, command=lambda v: self._apply_channel_change(int(v)))
        om.grid(row=0, column=1, sticky='we', padx=10)
        self._channel_win = win

    def on_key(self, event):
        k = (event.key or '').strip()
        if k == 'tab': self._cycle_channel(+1); return
        if k == 'shift+tab': self._cycle_channel(-1); return
        return super().on_key(event)

# ----------------------- IO helpers -----------------------
# (생략 - save_imagej_roi_zip, save_roi_bundle 등 기존 코드와 동일)
def save_imagej_roi_zip(polys, out_zip_path):
    try: import roifile as rf
    except Exception as e: raise RuntimeError("roifile 패키지가 필요합니다: pip install roifile") from e
    tmpdir = tempfile.mkdtemp(prefix="roi_zip_")
    names = []
    for i, poly in enumerate(polys, 1):
        arr = np.asarray(poly, dtype=np.float32)
        roi = rf.ImagejRoi.frompoints(arr, name=f"roi_{i}")
        rp = os.path.join(tmpdir, f"roi_{i}.roi")
        roi.tofile(rp)
        names.append(rp)
    with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in names: z.write(p, arcname=os.path.basename(p))

def _apply_view_and_color(img, view_params: dict):
    # (간소화: 뷰 파라미터 적용 로직)
    im = img.astype(np.float32, copy=False)
    vmin = np.percentile(im, float(view_params.get('p_low', 1.0)))
    vmax = np.percentile(im, float(view_params.get('p_high', 99.0)))
    if vmax <= vmin: vmax = vmin + 1e-6
    x = np.clip((im - vmin) / (vmax - vmin), 0, 1)
    x = np.power(x, 1.0/max(float(view_params.get('gamma', 1.0)), 1e-6))
    if bool(view_params.get('invert', False)): x = 1.0 - x
    mode = str(view_params.get('color_mode', 'grayscale'))
    if mode not in PCOLORS or PCOLORS[mode] is None: rgb = np.dstack([x, x, x])
    else: rgb = np.clip(x[..., None] * PCOLORS[mode].reshape(1,1,3), 0, 1)
    return rgb

def save_roi_bundle(outdir, mask_dir, overlay_dir, zip_dir, base_S_t, rois, img, view_params):
    # (간소화: 저장 로직)
    H, W = img.shape[:2]
    json_path = os.path.join(outdir, f"{base_S_t}.json")
    mask_path = os.path.join(mask_dir, f"{base_S_t}_mask.tif")
    png_path  = os.path.join(overlay_dir, f"{base_S_t}_overlay.png")
    zip_path  = os.path.join(zip_dir, f"{base_S_t}.zip")

    data = {
        "name": base_S_t,
        "image_shape": {"height": int(H), "width": int(W)},
        "rois": [np.asarray(p, float).tolist() for p in rois],
        "view_params": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in view_params.items()
        }
    }
    tmp_json = json_path + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_json, json_path)
    log(f"  JSON 저장: {json_path}")

    try:
        mask = np.zeros((H, W), dtype=np.uint8)
        for poly in rois:
            if len(poly) >= 3:
                C = np.asarray(poly, float)
                rr, cc = sk_polygon(C[:, 1], C[:, 0], shape=(H, W))
                mask[rr, cc] = 255
        tmp_mask = mask_path + ".tmp"
        imwrite(tmp_mask, mask, dtype=np.uint8)
        os.replace(tmp_mask, mask_path)
        log(f"  mask 저장: {mask_path}")
    except Exception as e: log(f"  [경고] mask 저장 실패: {e}")

    try:
        bg_rgb = _apply_view_and_color(img, view_params)
        Hs, Ws = bg_rgb.shape[:2]
        scale = min(1.0, FAST_OVERLAY_MAXPX / max(Hs, Ws)) if FAST_OVERLAY else 1.0
        if scale < 1.0:
            bg_rgb = resize(bg_rgb, (int(Hs*scale), int(Ws*scale)), preserve_range=True, anti_aliasing=True)
        canvas = (bg_rgb * 255).astype(np.uint8)
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()
        for i, poly in enumerate(rois, 1):
            P = (np.asarray(poly, float) * scale)
            xy = [tuple(p) for p in P]
            if len(xy) >= 2:
                draw.line(xy + [xy[0]], width=2, fill=(0, 255, 0))
                cx = float(np.mean(P[:, 0])); cy = float(np.mean(P[:, 1]))
                draw.text((cx, cy), str(i), fill=(255, 210, 0), font=font)
        tmp_png = png_path + ".tmp"
        pil_img.save(tmp_png, format="PNG", optimize=True)
        os.replace(tmp_png, png_path)
        log(f"  overlay 저장: {png_path}")
    except Exception as e: log(f"  [경고] overlay 저장 실패: {e}")

    try:
        save_imagej_roi_zip(rois, zip_path)
        log(f"  zip 저장: {zip_path}")
    except Exception as e: log(f"  [경고] ImageJ ROI ZIP 저장 실패: {e}")
    return json_path, mask_path, png_path, zip_path

# ----------------------- startup GUI & Main -----------------------
# (기존 main 및 startup_gui 함수 - 일부 업데이트)
def main():
    global LANG_CURRENT
    LANG_CURRENT = pick_lang_from_argv(sys.argv[1:])
    lang = LANG_CURRENT

    params        = startup_gui(lang=lang)
    folder        = params["folder"]
    start_ch      = params["channel"]
    mode          = params["mode"]
    s_filter      = params["stage"]
    t_filter      = params["time"]
    include_no    = params["include_no_roi"]
    timelapse     = params["timelapse"]
    tolerance     = params["tolerance"]
    min_area      = params["min_area"]
    color_mode    = params["color_mode"]
    bnd_mode      = params.get("bnd_mode", False)

    files_all = list_tifs(folder)
    if not files_all:
        print(t("err_no_tif", "[Error] No TIF files in the folder.", lang=lang))
        return

    outdir, mask_dir, overlay_dir, zip_dir = ensure_outdirs(folder)

    groups = {}
    for p in files_all:
        base = os.path.basename(p)
        s_num, t_num, _ = parse_tokens(base, timelapse)
        if s_num is None: continue
        key = (s_num, (t_num if timelapse else None))
        groups.setdefault(key, []).append(p)

    tasks = []
    for (s_num, t_num), flist in sorted(groups.items()):
        rep = sorted(flist, key=natural_key)[0]
        base_S_t = clean_base_for_save(os.path.basename(rep), timelapse)
        json_path = find_json_path(outdir, os.path.basename(rep), timelapse)
        has_roi = os.path.exists(json_path)

        def pass_filter():
            if mode != "edit": return True
            if (s_filter is not None) and (s_num != s_filter): return False
            if timelapse and (t_filter is not None) and (t_num != t_filter): return False
            return True

        if not pass_filter(): continue

        init_rois = None
        if has_roi:
            try:
                with open(json_path, "r", encoding="utf-8") as jf: data = json.load(jf)
                init_rois = [np.asarray(p, float) for p in data.get("rois", [])]
            except Exception: init_rois = None

        if mode == "new":
            if not has_roi: tasks.append(((s_num, t_num), rep, base_S_t, init_rois))
        else:
            if has_roi or include_no: tasks.append(((s_num, t_num), rep, base_S_t, init_rois))

    if not tasks:
        print(f"[정보] 처리할 대상이 없습니다. (mode={mode}, include_no_roi={include_no}, timelapse={timelapse})")
        return

    log(f"폴더: {folder}")
    log(f"모드: {mode} | 시작채널(선호): {start_ch} | timelapse={timelapse} | include_no_roi={include_no} | 대상 Stage/Time {len(tasks)}개")
    log(f"ROI: {outdir} | mask: {mask_dir} | overlay: {overlay_dir} | zip: {zip_dir}")
    log(f"threshold_mode = {'BND(mean+α·std)' if bnd_mode else 'percentile(p)'}")
    if mode == "edit":
        tf_msg = t_filter if (timelapse and t_filter is not None) else "N/A"
        log(f"필터: Stage={s_filter if s_filter is not None else 'ALL'}, Time={tf_msg}")
    log(f"tolerance={tolerance:.2f}, min_area={min_area:.0f}, pseudocolor={color_mode}")
    if FAST_OVERLAY: log(f"overlay: FAST(PIL) mode, max side = {FAST_OVERLAY_MAXPX}px")

    if bnd_mode: last_thresh = 1.5
    else: last_thresh = 70.0

    last_view_params = {
        'p_low': 1.0, 'p_high': 99.0, 'gamma': 1.0, 'invert': False,
        'color_mode': color_mode,
        'use_bandpass': False, 'sigma_small': 1.2, 'sigma_large': 9.0,
        'use_unsharp': False, 'unsharp_amount': 0.7, 'unsharp_radius': 2.0,
        'use_clahe': False, 'clahe_clip': 0.03, 'edge_overlay': False
    }

    for idx, ((s_num, t_num), rep, base_S_t, init_rois) in enumerate(tasks, 1):
        log(f"[{idx}/{len(tasks)}] Stage={s_num} Time={('t%02d'%t_num) if (t_num is not None) else 'N/A'} → 저장명: {base_S_t}")

        channel_map = build_channel_map(files_all, s_num, t_num, timelapse)
        rep_img = read_tiff_with_fallback(rep)
        if rep_img.ndim > 2: rep_img = rep_img[..., 0] if rep_img.ndim == 3 else rep_img[0, ...]
        rep_img = rep_img.astype(np.float32, copy=False)

        if len(channel_map) >= 1:
            start_for_this = start_ch if start_ch in channel_map else (sorted(channel_map.keys())[0])
            annot = ROIAnnotatorCH(
                image_path_by_channel=channel_map,
                selected_channel=start_for_this,
                title=f"{'(Edit) ' if mode=='edit' else '(New) '}{os.path.basename(rep)}  →  저장명: {base_S_t}",
                init_thresh_p=last_thresh,
                initial_rois=init_rois,
                tolerance=tolerance,
                min_area=min_area,
                color_mode=color_mode,
                last_view=last_view_params,
                image=rep_img,
                bnd_mode=bnd_mode,
                current_mode=mode  # [FIX] 모드 전달
            )
        else:
            annot = ROIAnnotator(
                rep_img,
                title=f"{'(Edit) ' if mode=='edit' else '(New) '}{os.path.basename(rep)}  →  저장명: {base_S_t}",
                init_thresh_p=last_thresh,
                initial_rois=init_rois,
                tolerance=tolerance,
                min_area=min_area,
                color_mode=color_mode,
                last_view=last_view_params,
                bnd_mode=bnd_mode,
                current_mode=mode  # [FIX] 모드 전달
            )

        rois, last_thresh, view_params = annot.show()
        if hasattr(annot, 'selected_channel'):
            view_params['selected_channel'] = int(annot.selected_channel)
        if getattr(annot, 'force_quit', False):
            log("사용자 요청으로 전체 종료(Ctrl+Q/Cmd+Q).")
            return

        log(f"  획득 ROI 수: {len(rois)}")

        if rois:
            img_to_save = annot.image if hasattr(annot, 'image') else rep_img
            log("  저장 시작(JSON/mask/overlay/zip)…")
            _ = save_roi_bundle(outdir, mask_dir, overlay_dir, zip_dir,
                                base_S_t, rois, img_to_save, view_params)
            log("  저장 완료")
        else:
            log("  [스킵] ROI 없음 → 저장하지 않음")

        last_view_params = view_params

    log("완료: 모든 작업이 종료되었습니다.")

if __name__ == "__main__":
    try:
        if matplotlib.get_backend().lower() == 'agg':
            matplotlib.use('TkAgg')
    except Exception:
        pass
    main()
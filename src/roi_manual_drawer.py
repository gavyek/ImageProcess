#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Manual Drawer + Auto-Segmentation + Fast Overlay + ImageJ ROI ZIP (v1.1)
--------------------------------------------------------------------------------
v1.1 patch
-- 영어모드 가능 (-mode EN으로 실행하면 영어로 나타남)

필요 패키지: numpy matplotlib pillow tifffile scipy scikit-image roifile
"""

import os, re, glob, zipfile, platform, tempfile, json, time, sys
import numpy as np

import matplotlib
if platform.system().lower().startswith("win"):
    matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']

# 🔧 키 충돌 완전 제거 (아이콘만으로 pan/zoom)
matplotlib.rcParams['keymap.save'] = []     # S키 충돌 방지
matplotlib.rcParams['keymap.pan'] = []      # P키로 pan 비활성화
matplotlib.rcParams['keymap.zoom'] = []     # Z키로 zoom 비활성화
matplotlib.rcParams['keymap.fullscreen'] = []  # 'f' 충돌 제거
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
    DISABLED, NORMAL, OptionMenu
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
    """
    timelapse=True  : S\\d+, t\\d+ 정수 추출
    timelapse=False : S\\d+ 정수만 추출
    channel: 끝 토큰(_X) 또는 _chX/_cX
    """
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
    """동일 Stage/Time의 {채널번호: 파일경로} 맵"""
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

# -------------------- TIFF 리더 (tifffile → Pillow 폴백) --------------------
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
    """
    mode = 'percentile' : thr_param = percentile p (예: 70 → 70th percentile)
    mode = 'bnd'        : thr_param = alpha (thr = mean + alpha*std)
    """
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
        # BND-like: mean + alpha*std (ROI 내부 픽셀 기준)
        m = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        if (s <= 0) or (not np.isfinite(s)):
            # 분산이 거의 없으면 percentile 방식으로 fallback
            thr = float(np.percentile(vals, 90.0))
        else:
            thr = m + thr_param * s
    else:
        # 기본: percentile 방식 (이전과 동일)
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

class ROIAnnotator:
    def __init__(self, image, title="ROI Drawer (Auto-Seg)", init_thresh_p: float = 70.0,
                 initial_rois=None, tolerance: float = 1.0, min_area: float = 40.0,
                 color_mode: str = "grayscale",
                 last_view: dict | None = None,
                 bnd_mode: bool = False):
        self.image = image.astype(np.float32, copy=False)
        self.rois = [] if initial_rois is None else [np.asarray(p, float) for p in initial_rois]
        self.current_selector = None
        self.force_quit = False

        # 🔸 이 값은 mode에 따라 의미가 달라짐
        #  - percentile 모드: percentile p (예: 70 → 70th percentile)
        #  - BND 모드      : alpha (thr = mean + alpha*std)
        self.thresh_p = float(init_thresh_p)
        self.tolerance = float(tolerance)
        self.min_area = float(min_area)
        self.bnd_mode = bool(bnd_mode)   # 🔸 새로 추가된 플래그

        if last_view is None:
            last_view = {}
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

    def _find_roi_index_under_cursor(self, x, y):
        """커서(x,y) 아래 있는 ROI의 인덱스(0-based)를 반환. 없으면 -1."""
        if not self.rois:
            return -1
        pt = (x, y)
        # 가장 먼저 그린 ROI부터 검사; 마지막 ROI를 우선하고 싶으면 reversed(range())
        for i in reversed(range(len(self.rois))):
            poly = np.asarray(self.rois[i], float)
            if len(poly) >= 3:
                path = mpath.Path(poly)
                if path.contains_point(pt):
                    return i
        return -1

    def _scale_roi(self, idx, factor, center):
        """idx번째 ROI를 center(픽셀좌표) 기준으로 factor 배 스케일."""
        if idx < 0 or idx >= len(self.rois):
            return False
        poly = np.asarray(self.rois[idx], float)
        cx, cy = center
        poly[:, 0] = (poly[:, 0] - cx) * factor + cx
        poly[:, 1] = (poly[:, 1] - cy) * factor + cy
        self.rois[idx] = poly
        return True

    # ---- Tk parent helper ----
    def _get_parent_tk(self):
        try:
            return self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            return None

    # ---- Custom Yes/No modal (Y/N/Space/Enter/Esc) ----
    def _ask_yes_no_modal(self, title, message, default_yes=True):
        parent = self._get_parent_tk()
        win = _tk.Toplevel(master=parent)
        win.title(title)
        win.transient(parent)
        win.grab_set()
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        # 위치/크기
        frm = _tk.Frame(win, padx=12, pady=10)
        frm.pack(fill="both", expand=True)
        _tk.Label(frm, text=message, justify="left").pack(anchor="w")
        # 버튼
        btn_frm = _tk.Frame(frm)
        btn_frm.pack(pady=(10, 0), fill="x")
        ans = {"val": default_yes}

        def _yes(*_):
            ans["val"] = True
            win.destroy()
        def _no(*_):
            ans["val"] = False
            win.destroy()

        b1 = _tk.Button(btn_frm, text="Yes (Y/Space/Enter)", width=20, command=_yes)
        b2 = _tk.Button(btn_frm, text="No (N/Esc)", width=12, command=_no)
        b1.pack(side="left", padx=5)
        b2.pack(side="left", padx=5)

        # 키 바인딩
        for k in ("y", "Y", "Return", "space"):
            win.bind(f"<{k}>", _yes)
        for k in ("n", "N", "Escape"):
            win.bind(f"<{k}>", _no)

        # 포커스/맨앞
        win.lift()
        b1.focus_set()
        win.wait_window()
        try:
            parent.attributes("-topmost", False)
        except Exception:
            pass
        return bool(ans["val"])
    
    def _ask_float_modal(self, title, prompt, initial=70.0, minv=0.0, maxv=100.0, step=0.5):
        parent = self._get_parent_tk()
        win = _tk.Toplevel(master=parent)
        win.title(title)
        win.transient(parent)
        win.grab_set()
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass

        frm = _tk.Frame(win, padx=12, pady=10)
        frm.pack(fill="both", expand=True)

        _tk.Label(frm, text=prompt, justify="left").pack(anchor="w", pady=(0,6))

        var = _tk.StringVar(value=f"{float(initial):.1f}")
        ent = _tk.Entry(frm, textvariable=var, width=12, justify="right")
        ent.pack(anchor="w")

        # 버튼
        btn_frm = _tk.Frame(frm); btn_frm.pack(pady=(10,0), fill="x")
        ans = {"val": None}

        def _clamp(v):
            try:
                x = float(v)
            except Exception:
                return None
            if x < minv or x > maxv:
                return None
            return x

        def _ok(*_):
            x = _clamp(var.get())
            if x is None:
                # 간단 경고
                try:
                    win.bell()
                except Exception:
                    pass
                ent.focus_set()
                ent.selection_range(0, _tk.END)
                return
            ans["val"] = x
            win.destroy()

        def _cancel(*_):
            ans["val"] = None
            win.destroy()

        b_ok = _tk.Button(btn_frm, text="OK (Enter)", width=12, command=_ok)
        b_no = _tk.Button(btn_frm, text="Cancel (Esc)", width=12, command=_cancel)
        b_ok.pack(side="left", padx=5); b_no.pack(side="left", padx=5)

        # 키 바인딩: Enter/ESC, ↑/↓로 빠르게 증감
        win.bind("<Return>", _ok)
        win.bind("<Escape>", _cancel)

        def _inc(*_):
            try:
                x = float(var.get())
            except Exception:
                x = initial
            x = min(maxv, x + step)
            var.set(f"{x:.1f}")
            ent.icursor(_tk.END)

        def _dec(*_):
            try:
                x = float(var.get())
            except Exception:
                x = initial
            x = max(minv, x - step)
            var.set(f"{x:.1f}")
            ent.icursor(_tk.END)

        win.bind("<Up>", _inc)
        win.bind("<Down>", _dec)

        # 포커스/선택 즉시
        win.update_idletasks()
        win.lift()
        ent.focus_set()
        ent.selection_range(0, _tk.END)

        win.wait_window()
        try:
            parent.attributes("-topmost", False)
        except Exception:
            pass
        return ans["val"]

    def _hud_str(self):
        instructions = t("hud_instr", default="", lang=LANG_CURRENT)
        inv = "ON" if self.invert else "off"
        bp  = f"BP={'ON' if self.use_bandpass else 'off'}(σs={self.sigma_small:.1f},σl={self.sigma_large:.1f})"
        us  = f"US={'ON' if self.use_unsharp else 'off'}(k={self.unsharp_amount:.1f},r={self.unsharp_radius:.1f})"
        ch  = f"CLAHE={'ON' if self.use_clahe else 'off'}(clip={self.clahe_clip:.3f})"
        ed  = f"EDGE={'ON' if self.edge_overlay else 'off'}"
        ln  = f"LocalNorm={'ON' if self.local_norm else 'off'}"

        # 🔸 추가: threshold 모드 표시
        if self.bnd_mode:
            thr_info = f"ThreshMode=BND(mean+α·std, α={self.thresh_p:.2f})"
        else:
            thr_info = f"ThreshMode=percentile(p={self.thresh_p:.1f})"

        view_line = (f"[View] p_low={self.p_low:.1f}%, p_high={self.p_high:.1f}%, "
                     f"gamma={self.gamma:.2f}, invert={inv}, color={self.color_mode} | "
                     f"{bp} | {us} | {ch} | {ed} | {ln} | {thr_info}")
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
            C = np.asarray(poly)
            self.ax.add_patch(MplPolygon(C, closed=True, fill=False, ec='lime', lw=2.0))
            cx, cy = polygon_centroid(C)
            t = self.ax.text(
                cx, cy, str(i),
                color='yellow', fontsize=12, fontweight='bold',
                ha='center', va='center', zorder=9999
            )
            try:
                t.set_path_effects([mpatheffects.withStroke(linewidth=2, foreground='black')])
            except Exception:
                pass
        self.fig.canvas.draw_idle()

    def _redraw_all(self):
        self._refresh_view()

    def start_polygon(self):
        if self.current_selector is not None:
            try:
                self.current_selector.disconnect_events()
            except Exception:
                pass
            self.current_selector = None
        try:
            self.current_selector = PolygonSelector(
                self.ax, self.on_select, useblit=True,
                lineprops=dict(color='yellow', linewidth=2, alpha=0.9),
                markerprops=dict(marker='o', markersize=5, mec='yellow', mfc='yellow', alpha=0.9)
            )
        except TypeError:
            try:
                self.current_selector = PolygonSelector(
                    self.ax, self.on_select, useblit=True,
                    props=dict(color='yellow', linewidth=2, alpha=0.9),
                    handle_props=dict(marker='o', markersize=5, mec='yellow', mfc='yellow', alpha=0.9)
                )
            except TypeError:
                self.current_selector = PolygonSelector(self.ax, self.on_select, useblit=True)
        self.fig.canvas.draw_idle()

    def on_select(self, verts):
        rough_poly = np.array(verts, dtype=float)
        while True:
            # 🔸 여기서 mode에 따라 percentile 또는 BND 방식 선택
            thr, mask, poly = segment_inside_polygon(
                self.image,
                rough_poly,
                thr_param=self.thresh_p,
                min_area=self.min_area,
                tolerance=self.tolerance,
                mode=("bnd" if self.bnd_mode else "percentile")
            )

            self.ax.clear()
            self.ax.set_axis_off()
            self._update_bg_rgb(rough_poly if self.local_norm else None)
            self.ax.imshow(self.bg_rgb)

            P = rough_poly
            self.ax.plot(P[:, 0], P[:, 1], '-', color='yellow', linewidth=1.5, alpha=0.8)
            self.ax.plot([P[-1, 0], P[0, 0]], [P[-1, 1], P[0, 1]],
                         '-', color='yellow', linewidth=1.0, alpha=0.8)

            if poly is not None:
                C = np.asarray(poly)
                self.ax.plot(C[:, 0], C[:, 1], '-', color='lime', linewidth=2.5, alpha=0.95)
                self.ax.plot([C[-1, 0], C[0, 0]], [C[-1, 1], C[0, 1]],
                             '-', color='lime', linewidth=2.5, alpha=0.95)

                if self.bnd_mode:
                    title = (f"BND-mode α={self.thresh_p:.2f}  "
                             f"(tol={self.tolerance:.2f}, minA={self.min_area:.0f}) → Accept?")
                else:
                    title = (f"Threshold p={self.thresh_p:.1f}th  "
                             f"(tol={self.tolerance:.2f}, minA={self.min_area:.0f}) → Accept?")
            else:
                if self.bnd_mode:
                    title = (f"No foreground at α={self.thresh_p:.2f}  "
                             f"(tol={self.tolerance:.2f}, minA={self.min_area:.0f}). Adjust?")
                else:
                    title = (f"No foreground at p={self.thresh_p:.1f}th  "
                             f"(tol={self.tolerance:.2f}, minA={self.min_area:.0f}). Adjust?")

            self.ax.set_title(title)
            self._draw_hud()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

            # ---- 모드에 따라 다른 문구의 Yes/No 모달 ----
            if self.bnd_mode:
                msg = (f"BND-like mode\n"
                       f"α={self.thresh_p:.2f} 로 추출된 ROI를 사용할까요?\n"
                       "(Yes: Y/Space/Enter,  No: N/Esc)")
            else:
                msg = (f"임계값 p={self.thresh_p:.1f} (percentile)로 추출된 ROI를 사용할까요?\n"
                       "(Yes: Y/Space/Enter,  No: N/Esc)")

            ans = False
            if poly is not None:
                ans = self._ask_yes_no_modal("Auto-segmentation", msg, default_yes=True)

            if ans and poly is not None:
                self.rois.append(np.asarray(poly))
                break
            else:
                # ---- 임계값 재설정 ----
                parent = self._get_parent_tk()
                if parent is not None:
                    parent.lift(); parent.focus_force()
                    try: parent.attributes("-topmost", True)
                    except Exception: pass

                try:
                    if self.bnd_mode:
                        prompt = (f"새 α 값을 입력하세요 (예: 1.0 ~ 3.0)\n"
                                  f"현재 α = {self.thresh_p:.2f}")
                        newp = self._ask_float_modal(
                            "α 재설정 (BND-mode)",
                            prompt,
                            initial=float(self.thresh_p),
                            minv=-1.0,
                            maxv=5.0,
                            step=0.1
                        )
                    else:
                        prompt = (f"새 percentile p 값을 입력하세요 (0–100)\n"
                                  f"현재 p = {self.thresh_p:.1f}")
                        newp = self._ask_float_modal(
                            "임계값 재설정 (percentile)",
                            prompt,
                            initial=float(self.thresh_p),
                            minv=0.0,
                            maxv=100.0,
                            step=0.5
                        )
                finally:
                    if parent is not None:
                        try: parent.attributes("-topmost", False)
                        except Exception: pass

                if newp is None:
                    break
                self.thresh_p = float(newp)

        if self.current_selector is not None:
            try:
                self.current_selector.disconnect_events()
            except Exception:
                pass
            self.current_selector = None
        self._redraw_all()


    def on_scroll(self, event):
        """
        - 그냥 휠: 밝기(브라이트니스) 이동 → p_low/p_high를 '같이' 평행 이동(윈도우 폭 유지)
        - Ctrl(또는 Cmd)+휠: 커서 아래 ROI 스케일 조정
        · 위로(Scroll Up)   → 확대 (기본 1.05배)
        · 아래로(Scroll Down) → 축소 (기본 0.95배)
        """
        # Matplotlib에서 modifier는 event.key에 'control', 'ctrl', 'cmd' 등이 들어올 수 있음
        key = (event.key or "").lower()
        is_ctrl = ("ctrl" in key) or ("control" in key) or ("cmd" in key)
        step = 0.25  # 🔧 밝기 이동 step (퍼센트 포인트)
        width = float(self.p_high - self.p_low)
        # 안전 가드
        width = max(0.1, min(width, 99.9))

        if event.button == 'up':
            # 더 밝게(분포 상위 쪽으로 이동)
            new_low = min(100.0 - width - 0.1, self.p_low + step)
            new_high = new_low + width
        else:  # 'down'
            new_low = max(0.0, self.p_low - step)
            new_high = new_low + width
            if new_high > 100.0:
                new_high = 100.0
                new_low = max(0.0, new_high - width)

        # 적용
        self.p_low = float(new_low)
        self.p_high = float(new_high)
        self._refresh_view()


    def on_key(self, event):
        k = (event.key or '').strip()
        if k in ('ctrl+q', 'cmd+q'):
            self.force_quit = True
            plt.close(self.fig); return
        if k.lower() == 'q':
            plt.close(self.fig); return
        if k.lower() == 'p':
            self.start_polygon(); return
        if k.lower() == 'u':
            if self.rois:
                self.rois.pop(); self._redraw_all()
            return
        if k.lower() == 'c':
            self.rois.clear(); self._redraw_all(); return
        if k in ('ctrl+s', 'cmd+s', 'ctrl+enter', 'cmd+enter'):
            plt.close(self.fig); return
        # Space/Enter → PolygonSelector 내부에서는 모달에서 이미 처리

        # 밝기/HUD 즉시 반영
        if k in ('a', 'A'):
            self.p_low = max(0.0, min(self.p_high-0.1, self.p_low - (5.0 if k == 'A' else 1.0))); self._refresh_view(); return
        if k in ('d', 'D'):
            self.p_low = max(0.0, min(self.p_high-0.1, self.p_low + (5.0 if k == 'D' else 1.0))); self._refresh_view(); return
        if k in ('s', 'S'):
            self.p_high = max(self.p_low+0.1, self.p_high - (5.0 if k == 'S' else 1.0)); self._refresh_view(); return
        if k in ('f', 'F'):
            self.p_high = min(100.0, max(self.p_low+0.1, self.p_high + (5.0 if k == 'F' else 1.0))); self._refresh_view(); return

        if k == 'g':
            self.gamma = max(0.1, self.gamma - 0.1); self._refresh_view(); return
        if k == 'G':
            self.gamma = min(5.0, self.gamma + 0.1); self._refresh_view(); return
        if k.lower() == 'i':
            self.invert = not self.invert; self._refresh_view(); return

        if k == '0': self.color_mode = 'grayscale'; self._refresh_view(); return
        if k == '1': self.color_mode = 'cyan';      self._refresh_view(); return
        if k == '2': self.color_mode = 'blue';      self._refresh_view(); return
        if k == '3': self.color_mode = 'green';     self._refresh_view(); return
        if k == '4': self.color_mode = 'red';       self._refresh_view(); return
        if k == '5': self.color_mode = 'yellow';    self._refresh_view(); return

        if k.lower() == 'b':
            self.use_bandpass = not self.use_bandpass; self._refresh_view(); return
        if k == ',':  self.sigma_small = max(0.2, self.sigma_small - 0.2); self._refresh_view(); return
        if k == '.':  self.sigma_small = min(5.0, self.sigma_small + 0.2); self._refresh_view(); return
        if k == '[':  self.sigma_large = max(self.sigma_small+0.5, self.sigma_large - 1.0); self._refresh_view(); return
        if k == ']':  self.sigma_large = min(30.0, self.sigma_large + 1.0); self._refresh_view(); return

        if k.lower() == 'x':
            self.use_unsharp = not self.use_unsharp; self._refresh_view(); return
        if k == 'k':  self.unsharp_amount = max(0.0, self.unsharp_amount - 0.1); self._refresh_view(); return
        if k == 'K':  self.unsharp_amount = min(3.0, self.unsharp_amount + 0.1); self._refresh_view(); return
        if k == 'l':  self.unsharp_radius = max(0.5, self.unsharp_radius - 0.5); self._refresh_view(); return
        if k == 'L':  self.unsharp_radius = min(10.0, self.unsharp_radius + 0.5); self._refresh_view(); return

        if k.lower() == 't':
            self.use_clahe = not self.use_clahe; self._refresh_view(); return
        if k == 'y':  self.clahe_clip = max(0.005, self.clahe_clip - 0.005); self._refresh_view(); return
        if k == 'Y':  self.clahe_clip = min(0.100, self.clahe_clip + 0.005); self._refresh_view(); return

        if k.lower() == 'e':
            self.edge_overlay = not self.edge_overlay; self._refresh_view(); return
        if k.lower() == 'o':
            self.local_norm = not self.local_norm; self._refresh_view(); return

        if k.lower() == 'r':
            self.p_low, self.p_high, self.gamma, self.invert = 1.0, 99.0, 1.0, False
            self.use_bandpass=False; self.sigma_small=1.2; self.sigma_large=9.0
            self.use_unsharp=False;  self.unsharp_amount=0.7; self.unsharp_radius=2.0
            self.use_clahe=False;    self.clahe_clip=0.03
            self.edge_overlay=False; self.local_norm=False
            self._refresh_view(); return

        if k.lower() == 'h':
            msg = (
            "밝기/콘트라스트: a/A,d/D,s/S,f/F | 감마: g/G, 마우스휠 | 반전: i\n"
            "Band-pass(DoG): b, σs: , .  |  σl: [ ]\n"
            "Unsharp: x, 강도 k/K, 반경 l/L\n"
            "CLAHE: t, clip: y/Y | Edge overlay: e\n"
            "Local normalize(rough poly 내부 기준): o\n"
            "색상 0~5 | 리셋 r | 저장 Ctrl+S/Cmd+S | 새 ROI p | 삭제 u | 전체지우기 c | 종료 q"
            )
            parent = self._get_parent_tk()
            messagebox.showinfo("단축키 도움말", msg, parent=parent); return

    def export_view_params(self) -> dict:
        return {
            'p_low': self.p_low, 'p_high': self.p_high, 'gamma': self.gamma, 'invert': self.invert,
            'color_mode': self.color_mode,
            'use_bandpass': self.use_bandpass, 'sigma_small': self.sigma_small, 'sigma_large': self.sigma_large,
            'use_unsharp': self.use_unsharp, 'unsharp_amount': self.unsharp_amount, 'unsharp_radius': self.unsharp_radius,
            'use_clahe': self.use_clahe, 'clahe_clip': self.clahe_clip, 'edge_overlay': self.edge_overlay
        }

    def _attach_view_tool_to_toolbar(self):
        """툴바 'Subplot config' 버튼을 밝기/감마 컨트롤 창으로 교체."""
        try:
            backend = matplotlib.get_backend().lower()
            if 'tkagg' not in backend:
                return
            toolbar = self.fig.canvas.manager.toolbar
        except Exception:
            return

        if getattr(toolbar, "_roi_view_tool_patched", False):
            return

        def open_view_tool(*args, **kwargs):
            if hasattr(self, "_view_tool_win") and self._view_tool_win.winfo_exists():
                self._view_tool_win.lift()
                return
            parent = self._get_parent_tk()
            win = _tk.Toplevel(parent); win.title("View / Gamma / Percentile")
            try: win.attributes("-topmost", True)
            except Exception: pass
            frm = _tk.Frame(win); frm.pack(padx=10, pady=10)

            var_p_low  = _tk.DoubleVar(value=self.p_low)
            var_p_high = _tk.DoubleVar(value=self.p_high)
            var_gamma  = _tk.DoubleVar(value=self.gamma)

            _tk.Label(frm, text="p_low (%)").grid(row=0, column=0, sticky="w")
            s_low = _tk.Scale(frm, from_=0, to=50, resolution=0.5,
                              orient="horizontal", length=260,
                              variable=var_p_low); s_low.grid(row=0, column=1, padx=6, pady=4)

            _tk.Label(frm, text="p_high (%)").grid(row=1, column=0, sticky="w")
            s_high = _tk.Scale(frm, from_=50, to=100, resolution=0.5,
                               orient="horizontal", length=260,
                               variable=var_p_high); s_high.grid(row=1, column=1, padx=6, pady=4)

            _tk.Label(frm, text="gamma").grid(row=2, column=0, sticky="w")
            s_gamma = _tk.Scale(frm, from_=0.2, to=3.0, resolution=0.05,
                                orient="horizontal", length=260,
                                variable=var_gamma); s_gamma.grid(row=2, column=1, padx=6, pady=4)

            def apply_and_refresh(*_):
                pl = float(var_p_low.get()); ph = float(var_p_high.get())
                if pl >= ph: pl = min(pl, ph - 0.1)
                self.p_low  = max(0.0, min(ph-0.1, pl))
                self.p_high = max(self.p_low+0.1, min(100.0, ph))
                self.gamma  = max(0.05, float(var_gamma.get()))
                self._refresh_view()
            for v in (var_p_low, var_p_high, var_gamma):
                v.trace_add("write", apply_and_refresh)

            def on_close():
                self._view_tool_win = None; win.destroy()
            win.protocol("WM_DELETE_WINDOW", on_close)
            self._view_tool_win = win

        try:
            toolbar.configure_subplots = open_view_tool
            btn = None
            for key in ("Subplots", "subplots", "Configure subplots"):
                if hasattr(toolbar, "_buttons") and key in toolbar._buttons:
                    btn = toolbar._buttons[key]; break
            if btn is not None:
                try: btn.configure(command=open_view_tool)
                except Exception: pass
                try: btn.tooltip_string = "View / Gamma / Percentile"
                except Exception: pass
            toolbar._roi_view_tool_patched = True
        except Exception:
            pass

    def show(self):
        plt.show()
        return self.rois, float(self.thresh_p), self.export_view_params()

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
            if isinstance(_fallback, np.ndarray):
                img0 = _fallback
            else:
                img0 = np.zeros((512, 512), np.float32)

        super().__init__(image=img0, *args, **kwargs)

        self.ax.set_title(self.ax.get_title() + f"  |  CH={self.selected_channel}")
        self._channel_win = None
        self._build_channel_window()
        try:
            self.fig.canvas.mpl_connect('close_event', self._on_close)
        except Exception:
            pass

    def _on_close(self, evt):
        try:
            if self._channel_win:
                self._channel_win.destroy()
        except Exception:
            pass

    def _load_image(self, ch):
        path = self.channel_map[int(ch)]
        img = read_tiff_with_fallback(path)
        if getattr(img, 'ndim', 2) > 2:
            img = img[..., 0] if img.ndim == 3 else img[0, ...]
        return img.astype(np.float32, copy=False)

    def _apply_channel_change(self, ch):
        if not self.channel_map:
            return
        self.selected_channel = int(ch)
        self.image = self._load_image(self.selected_channel)
        try:
            base = re.sub(r"\s\|\sCH=.*$", "", self.ax.get_title())
            self.ax.set_title(base + f"  |  CH={self.selected_channel}")
        except Exception:
            pass
        self._refresh_view()
        self.fig.canvas.draw_idle()

    def _cycle_channel(self, step):
        ks = list(sorted(self.channel_map.keys()))
        if not ks:
            return
        i = ks.index(self.selected_channel)
        self._apply_channel_change(ks[(i + step) % len(ks)])
        if hasattr(self, '_ch_var'):
            self._ch_var.set(str(self.selected_channel))

    def _build_channel_window(self):
        if len(self.channel_map) <= 1:
            return
        try:
            parent = self.fig.canvas.get_tk_widget().winfo_toplevel()
        except Exception:
            log("[경고] TkAgg가 아니어서 Channel 창을 띄우지 못했습니다. TAB/Shift+TAB으로 전환하세요.")
            return

        win = _tk.Toplevel(master=parent)
        win.title("Channel Select")
        base_w, base_h = 300, 180
        win.geometry(f"{base_w}x{base_h}")
        try:
            win.attributes('-topmost', True)
        except Exception:
            pass
        try:
            win.update_idletasks()
            px = parent.winfo_rootx(); py = parent.winfo_rooty()
            x = max(0, px + 40); y = max(0, py + 80)
            win.geometry(f"{base_w}x{base_h}+{x}+{y}")
        except Exception:
            pass

        try:
            import tkinter.font as tkFont
            base_font = tkFont.Font(master=win, size=12)
            bold_font = tkFont.Font(master=win, size=12, weight="bold")
        except Exception:
            base_font = None; bold_font = None

        frm = _tk.Frame(win); frm.pack(padx=12, pady=12)
        lbl = _tk.Label(frm, text="Select Channel", font=bold_font or None)
        lbl.grid(row=0, column=0, sticky='w')

        self._ch_var = _tk.StringVar(value=str(self.selected_channel))
        opts = [str(k) for k in sorted(self.channel_map.keys())]
        om = _tk.OptionMenu(frm, self._ch_var, *opts,
                            command=lambda v: self._apply_channel_change(int(v)))
        om.grid(row=0, column=1, sticky='we', padx=10)
        if base_font is not None:
            om.config(font=base_font)
            try: om["menu"].config(font=base_font)
            except Exception: pass

        self._label_path = _tk.Label(
            frm,
            text=os.path.basename(self.channel_map[self.selected_channel]),
            fg='#666', font=base_font or None, wraplength=320, justify='left'
        )
        self._label_path.grid(row=1, column=0, columnspan=2, sticky='w', pady=(8, 0))

        def on_var(*_):
            try:
                ch = int(self._ch_var.get())
                path = os.path.basename(self.channel_map[ch])
                self._label_path.configure(text=path)
            except Exception:
                pass
        try:
            self._ch_var.trace_add('write', on_var)
        except Exception:
            self._ch_var.trace('w', on_var)

        self._channel_win = win

    def on_key(self, event):
        k = (event.key or '').strip()
        if k == 'tab':
            self._cycle_channel(+1); return
        if k == 'shift+tab':
            self._cycle_channel(-1); return
        return super().on_key(event)

# ----------------------- IO helpers -----------------------
def save_imagej_roi_zip(polys, out_zip_path):
    try:
        import roifile as rf
    except Exception as e:
        raise RuntimeError("roifile 패키지가 필요합니다: pip install roifile") from e
    tmpdir = tempfile.mkdtemp(prefix="roi_zip_")
    names = []
    for i, poly in enumerate(polys, 1):
        arr = np.asarray(poly, dtype=np.float32)
        roi = rf.ImagejRoi.frompoints(arr, name=f"roi_{i}")
        rp = os.path.join(tmpdir, f"roi_{i}.roi")
        roi.tofile(rp)
        names.append(rp)
    with zipfile.ZipFile(out_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in names:
            z.write(p, arcname=os.path.basename(p))

def _apply_view_and_color(img, view_params: dict):
    im = img.astype(np.float32, copy=False)
    if view_params.get('use_bandpass', False):
        im = ndi.gaussian_filter(im, float(view_params.get('sigma_small', 1.2))) - \
             ndi.gaussian_filter(im, float(view_params.get('sigma_large', 9.0)))
    if view_params.get('use_unsharp', False):
        im = im + float(view_params.get('unsharp_amount', 0.7)) * \
            (im - ndi.gaussian_filter(im, float(view_params.get('unsharp_radius', 2.0))))
    vmin = np.percentile(im, float(view_params.get('p_low', 1.0)))
    vmax = np.percentile(im, float(view_params.get('p_high', 99.0)))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    x = np.clip((im - vmin) / (vmax - vmin), 0, 1)
    x = np.power(x, 1.0/max(float(view_params.get('gamma', 1.0)), 1e-6))
    if bool(view_params.get('invert', False)):
        x = 1.0 - x

    mode = str(view_params.get('color_mode', 'grayscale'))
    if bool(view_params.get('use_clahe', False)):
        x = exposure.equalize_adapthist(x, clip_limit=float(view_params.get('clahe_clip', 0.03)))
    if mode not in PCOLORS or PCOLORS[mode] is None:
        rgb = np.dstack([x, x, x])
    else:
        rgb = np.clip(x[..., None] * PCOLORS[mode].reshape(1,1,3), 0, 1)
    if bool(view_params.get('edge_overlay', False)):
        ed = filters.sobel(x)
        rgb[...,1] = np.clip(rgb[...,1] + ed*0.8, 0, 1)
    return rgb

def save_roi_bundle(outdir, mask_dir, overlay_dir, zip_dir, base_S_t, rois, img, view_params):
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
    except Exception as e:
        log(f"  [경고] mask 저장 실패: {e}")

    try:
        bg_rgb = _apply_view_and_color(img, view_params)
        Hs, Ws = bg_rgb.shape[:2]
        scale = min(1.0, FAST_OVERLAY_MAXPX / max(Hs, Ws)) if FAST_OVERLAY else 1.0
        if scale < 1.0:
            bg_rgb = resize(bg_rgb, (int(Hs*scale), int(Ws*scale)), preserve_range=True, anti_aliasing=True)
        canvas = (bg_rgb * 255).astype(np.uint8)
        pil_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_img)
        font = None
        for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(font_name, 18)
                break
            except Exception:
                font = None
        if font is None:
            font = ImageFont.load_default()

        for i, poly in enumerate(rois, 1):
            P = (np.asarray(poly, float) * scale)
            xy = [tuple(p) for p in P]
            if len(xy) >= 2:
                draw.line(xy + [xy[0]], width=2, fill=(0, 255, 0))
                cx = float(np.mean(P[:, 0])); cy = float(np.mean(P[:, 1]))
                try:
                    draw.text((cx, cy), str(i), fill=(255, 210, 0), font=font, anchor="mm")
                except TypeError:
                    draw.text((cx, cy), str(i), fill=(255, 210, 0), font=font)

        tmp_png = png_path + ".tmp"
        pil_img.save(tmp_png, format="PNG", optimize=True)
        os.replace(tmp_png, png_path)
        log(f"  overlay 저장: {png_path}")
    except Exception as e:
        log(f"  [경고] overlay 저장 실패: {e}")

    try:
        save_imagej_roi_zip(rois, zip_path)
        log(f"  zip 저장: {zip_path}")
    except Exception as e:
        log(f"  [경고] ImageJ ROI ZIP 저장 실패: {e}")
        zip_path = None

    return json_path, mask_path, png_path, zip_path

# ----------------------- startup GUI -----------------------
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
    bnd_mode_var    = BooleanVar(value=False)   # 🔸 새로 추가

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
            "bnd_mode": bool(bnd_mode_var.get())   # 🔸 추가
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

# ----------------------- lang picker -----------------------
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

# ----------------------- main -----------------------
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
        if s_num is None:
            continue
        key = (s_num, (t_num if timelapse else None))
        groups.setdefault(key, []).append(p)

    tasks = []
    for (s_num, t_num), flist in sorted(groups.items()):
        rep = sorted(flist, key=natural_key)[0]
        base_S_t = clean_base_for_save(os.path.basename(rep), timelapse)
        json_path = find_json_path(outdir, os.path.basename(rep), timelapse)
        has_roi = os.path.exists(json_path)

        def pass_filter():
            if mode != "edit":
                return True
            if (s_filter is not None) and (s_num != s_filter):
                return False
            if timelapse and (t_filter is not None) and (t_num != t_filter):
                return False
            return True

        if not pass_filter():
            continue

        init_rois = None
        if has_roi:
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                init_rois = [np.asarray(p, float) for p in data.get("rois", [])]
            except Exception:
                init_rois = None

        if mode == "new":
            if not has_roi:
                tasks.append(((s_num, t_num), rep, base_S_t, init_rois))
        else:
            if has_roi or include_no:
                tasks.append(((s_num, t_num), rep, base_S_t, init_rois))

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
    if FAST_OVERLAY:
        log(f"overlay: FAST(PIL) mode, max side = {FAST_OVERLAY_MAXPX}px")

    if bnd_mode:
        last_thresh = 1.5   # alpha 기본값
    else:
        last_thresh = 70.0  # percentile 기본값 (70th)

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
        if rep_img.ndim > 2:
            rep_img = rep_img[..., 0] if rep_img.ndim == 3 else rep_img[0, ...]
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
                bnd_mode=bnd_mode          # 🔸 새로 추가
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
                bnd_mode=bnd_mode          # 🔸 새로 추가
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

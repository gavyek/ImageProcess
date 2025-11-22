#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fluor Intensity Builder (v1 + Integrated Log/Progress)

- 메인 GUI 하단 Log 창 (ScrolledText)
- print / 에러 메시지 → Log 창 + 로그 파일 + 콘솔 동시 기록
- 메인 창 내부 Progressbar + 상태/ETA 표시
- 실행 중: 실행 버튼 및 입력 위젯 비활성화, 취소 버튼으로 중단 요청
- 처리 완료 후에도 GUI 유지 (종료 버튼 또는 X로 닫기)
"""

import os, re, glob, json, time, datetime, math, traceback, sys, threading
import numpy as np
import pandas as pd

from tifffile import imread, imwrite
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.path as mpath

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from concurrent.futures import ProcessPoolExecutor, as_completed

# ===================== 공통 유틸 =====================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_tifs(folder):
    exts = ("*.tif","*.tiff","*.TIF","*.TIFF")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    uniq = {}
    for p in files:
        norm = os.path.normcase(os.path.abspath(p))
        if norm not in uniq:
            uniq[norm] = p
    return sorted(uniq.values(), key=natural_key)

# ---------- 2자리 저장명 & 타임랩스 파서 ----------

def fmt_stage(n):  # 'S01'
    return f"S{int(n):02d}"

def fmt_time(n):   # 't00'
    return f"t{int(n):02d}"

def parse_tokens(basename: str, timelapse: bool):
    """
    timelapse=True  : S\\d+, t\\d+ 정수 추출
    timelapse=False : S\\d+ 정수만 추출
    channel: 'chX'/'cX' 우선, 없으면 마지막 숫자(단, tXX와 '문자열'이 완전히 같은 건 채널에서 제외)
    return: (stage_num or None, time_num or None, channel or None)
    """
    name = os.path.splitext(basename)[0]

    # stage
    s_num = None
    ms = re.search(r'(?i)(?:^|[_-])S(\d+)(?=$|[_-])', name)
    if ms:
        s_num = int(ms.group(1))

    # time
    t_num = None
    t_str = None
    if timelapse:
        mt = re.search(r'(?i)(?:^|[_-])t(\d+)(?=$|[_-])', name)
        if mt:
            t_str = mt.group(1)  # "03"
            t_num = int(t_str)

    # channel
    ch = None
    m_ch = re.search(r'(?i)(?:^|[_-])(ch|c)(\d{1,3})(?=$|[_-])', name)
    if m_ch:
        ch = int(m_ch.group(2))
    else:
        tokens = re.split(r'[_-]', name)
        nums = [tok for tok in tokens if tok.isdigit()]
        if timelapse and t_str is not None:
            nums = [n for n in nums if n != t_str]
        if nums:
            ch = int(nums[-1])

    return s_num, t_num, ch

def clean_base_for_save(basename: str, timelapse: bool):
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    if s_num is None:
        name = os.path.splitext(basename)[0]
        return re.sub(r'([_-])\d+$', '', name)
    if timelapse and (t_num is not None):
        return f"{fmt_stage(s_num)}_{fmt_time(t_num)}"
    return fmt_stage(s_num)

def find_roi_basepath(roi_dir, basename, timelapse):
    """표준(S01[_t00]) 우선, 없으면 레거시(S1[_t0]) base 반환"""
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    norm = clean_base_for_save(basename, timelapse)
    cands = [os.path.join(roi_dir, norm)]
    if s_num is not None:
        legacy = f"S{int(s_num)}"
        if timelapse and (t_num is not None):
            legacy = f"{legacy}_t{int(t_num)}"
        cands.append(os.path.join(roi_dir, legacy))
    for b in cands:
        if os.path.exists(b + ".json") or os.path.exists(b + ".png"):
            return b
    return cands[0]

# ---------- TIFF 리더(폴백) ----------

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

def read_2d(path):
    a = read_tiff_with_fallback(path)
    if a.ndim > 2:
        a = a[..., 0] if a.ndim == 3 else a[0, ...]
    return a.astype(np.float32, copy=False)

# ---------- 키 빌드 ----------

def build_keymap(files, timelapse: bool):
    """
    returns dict key -> {ch: path}
    key: (Sxx, txx or None)
    """
    key2ch = {}
    for p in files:
        base = os.path.basename(p)
        s_num, t_num, ch = parse_tokens(base, timelapse)
        if s_num is None or ch is None:
            continue
        s = fmt_stage(s_num)
        t = fmt_time(t_num) if (timelapse and t_num is not None) else None
        key = (s, t)
        key2ch.setdefault(key, {})[ch] = p

    def sort_key(item):
        s, t = item[0]
        s_idx = int(re.search(r'\d+', s).group()) if s else -1
        t_idx = int(re.search(r'\d+', t).group()) if t else -1
        return (s_idx, t_idx)

    return dict(sorted(key2ch.items(), key=sort_key))

# ===================== ROI/계산 유틸 =====================

def rasterize_polygon(poly, shape):
    H, W = shape
    path = mpath.Path(np.asarray(poly, dtype=float))
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    return path.contains_points(pts).reshape(H, W)

def load_roi_polys_or_mask(roi_folder, s, t, timelapse, img_shape=None):
    """
    1) JSON: {"rois":[ [[x,y],...], ... ]} (다중 ROI)
    2) PNG mask: 흑백/이진(white=True). 여러 ROI = union만 제공
    """
    base = f"{s}_{t}" if (timelapse and t is not None) else s
    roi_base = find_roi_basepath(roi_folder, base, timelapse)

    # JSON
    json_path = roi_base + ".json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        polys = []
        for poly in data.get("rois", []):
            P = np.asarray(poly, dtype=float)
            if P.shape[0] >= 3:
                polys.append(P)
        if polys:
            return polys, None

    # PNG mask
    png_path = roi_base + ".png"
    if os.path.exists(png_path):
        with Image.open(png_path) as im:
            mask = np.array(im.convert("L")) > 0
        if (img_shape is not None) and (mask.shape != img_shape):
            H, W = img_shape
            mh, mw = mask.shape
            mask = mask[:min(H, mh), :min(W, mw)]
            padH = max(0, H - mask.shape[0])
            padW = max(0, W - mask.shape[1])
            if padH or padW:
                pad = ((0, padH), (0, padW))
                mask = np.pad(mask, pad, mode="constant", constant_values=False)
        return None, mask
    return None, None

def count_rois_fast(roi_folder, s, t, timelapse):
    """총 진행률 계산용: JSON이면 len(rois), PNG mask면 1, 둘 다 없으면 0"""
    base = f"{s}_{t}" if (timelapse and t is not None) else s
    roi_base = find_roi_basepath(roi_folder, base, timelapse)
    json_path = roi_base + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rois = data.get("rois", [])
            return max(0, int(len(rois)))
        except:
            return 0
    png_path = roi_base + ".png"
    if os.path.exists(png_path):
        return 1
    return 0

def _vals_in_scope(img2d, scope_mask):
    return img2d.ravel() if scope_mask is None else img2d[scope_mask]

def bg_value(img2d, mode="percentile", p=1.0, scope_mask=None, stride=4):
    vals = _vals_in_scope(img2d, scope_mask)
    if vals.size == 0:
        return 0.0
    if stride and stride > 1:
        vals = vals[::int(stride)]
        if vals.size == 0:
            return 0.0
    if mode == "percentile":
        return float(np.percentile(vals, p))
    elif mode == "hist-mode":
        hist, bins = np.histogram(vals, bins=2048)
        if hist.sum() <= 0:
            return float(np.percentile(vals, p))
        cdf = np.cumsum(hist).astype(float)
        cdf /= cdf[-1]
        target = float(p) / 100.0
        idx = int(np.searchsorted(cdf, target, side="left"))
        thr = bins[-1] if idx >= len(bins) - 1 else 0.5 * (bins[idx] + bins[idx + 1])
        return float(thr)
    else:
        return 0.0

def bg_correct(img2d, mode="percentile", p=1.0, scope_mask=None, clip_neg=True, stride=4):
    B = bg_value(img2d, mode=mode, p=p, scope_mask=scope_mask, stride=stride)
    J = img2d - B
    if clip_neg:
        J[J < 0] = 0.0
    return J, B

def quantify_stats(vals):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(mean=np.nan, median=np.nan, std=np.nan, p5=np.nan, p95=np.nan,
                    vmin=np.nan, vmax=np.nan, vsum=np.nan, npx=0)
    return dict(mean=float(np.mean(vals)),
                median=float(np.median(vals)),
                std=float(np.std(vals)),
                p5=float(np.percentile(vals, 5)),
                p95=float(np.percentile(vals, 95)),
                vmin=float(np.min(vals)),
                vmax=float(np.max(vals)),
                vsum=float(np.sum(vals)),
                npx=int(vals.size))

def quantify_per_roi_multi(images_dict, polys=None, union_mask=None):
    any_img = next(iter(images_dict.values()))
    H, W = any_img.shape
    rows = []
    if polys is not None:
        for i, poly in enumerate(polys, 1):
            m = rasterize_polygon(poly, (H, W))
            row = {"roi": i, "area_px": int(m.sum())}
            for ch, img in sorted(images_dict.items()):
                st = quantify_stats(img[m])
                for k, v in st.items():
                    row[f"ch{ch}_{k}"] = v
            rows.append(row)
    elif union_mask is not None:
        m = union_mask.astype(bool, copy=False)
        row = {"roi": 1, "area_px": int(m.sum())}
        for ch, img in sorted(images_dict.items()):
            st = quantify_stats(img[m])
            for k, v in st.items():
                row[f"ch{ch}_{k}"] = v
        rows.append(row)
    else:
        m = np.ones_like(any_img, dtype=bool)
        row = {"roi": 0, "area_px": int(m.sum())}
        for ch, img in sorted(images_dict.items()):
            st = quantify_stats(img[m])
            for k, v in st.items():
                row[f"ch{ch}_{k}"] = v
        rows.append(row)
    return rows

def auto_minmax(vals, p_lo=1.0, p_hi=99.0):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)

# ===================== 컬러맵 유틸 =====================

COLOR_CHOICES = ["Cyan", "Yellow", "Green", "Red", "Blue", "Magenta", "Grayscale"]
CMAP_CHOICES = ["jet","turbo","viridis","plasma","magma","inferno","cividis"]
SB_ANCHORS = ["br","bl","tr","tl"]

def _single_color_rgb(name):
    name = name.lower()
    lut = {
        "cyan":     (0.0, 1.0, 1.0),
        "yellow":   (1.0, 1.0, 0.0),
        "green":    (0.0, 1.0, 0.0),
        "red":      (1.0, 0.0, 0.0),
        "blue":     (0.0, 0.0, 1.0),
        "magenta":  (1.0, 0.0, 1.0),
        "grayscale": None
    }
    return lut.get(name, None)

def get_cmap_for_color(color_name):
    """단색 그라데이션 컬러맵(검정→지정색). Grayscale은 'gray'."""
    if color_name is None:
        return "gray"
    if color_name.lower() == "grayscale":
        return "gray"
    rgb = _single_color_rgb(color_name)
    if rgb is None:
        return "gray"
    r, g, b = rgb
    cdict = {
        'red':   ((0.0, 0.0, 0.0), (1.0, r, r)),
        'green': ((0.0, 0.0, 0.0), (1.0, g, g)),
        'blue':  ((0.0, 0.0, 0.0), (1.0, b, b)),
    }
    return LinearSegmentedColormap('single_' + color_name, segmentdata=cdict)

# ===================== 그리기 유틸 =====================

def draw_scalebar(ax, img_w, img_h, bar_px, bar_um,
                  frac_margin=0.05, lw=3, anchor="br", font_size=10):
    if anchor not in ("br", "bl", "tr", "tl"):
        anchor = "br"
    margin_x = int(img_w * 0.05)
    margin_y = int(img_h * 0.05)
    if anchor in ("br", "tr"):
        x_start = img_w - margin_x - bar_px
    else:
        x_start = margin_x
    if anchor in ("br", "bl"):
        y = img_h - margin_y
    else:
        y = margin_y
    x_end = x_start + bar_px
    ax.plot([x_start, x_end], [y, y], color='w', linewidth=lw)
    ax.text(
        (x_start + x_end) / 2,
        y - max(10, int(0.02 * img_h)) if anchor in ("br", "bl") else y + max(10, int(0.02 * img_h)),
        f"{bar_um:.0f} µm",
        color='w',
        ha='center',
        va=('bottom' if anchor in ("br", "bl") else 'top'),
        fontsize=font_size,
        bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none')
    )

def add_short_colorbar(fig, ax, vmin, vmax, cmap='jet', label="Intensity (a.u.)"):
    bbox = ax.get_position()
    ax_h = bbox.height
    ax_y0 = bbox.y0
    ax_x1 = bbox.x1
    pad = 0.01
    cb_w = 0.02
    cb_h = ax_h * (2.0 / 3.0)
    cb_y0 = ax_y0 + (ax_h - cb_h) / 2.0
    cb_x0 = ax_x1 + pad
    cb_ax = fig.add_axes([cb_x0, cb_y0, cb_w, cb_h])
    cb_ax.set_facecolor('black')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cb = mpl.colorbar.ColorbarBase(
        cb_ax, cmap=cmap_obj, norm=norm, orientation='vertical'
    )
    cb.set_label(label, rotation=90, color='w')
    cb.ax.yaxis.label.set_color('w')
    cb.set_ticks([vmin, vmax])
    cb.ax.set_yticklabels(
        [f"{vmin:.2f}", f"{vmax:.2f}"], color='w'
    )
    cb.ax.tick_params(color='w', labelcolor='w', length=3)
    cb.outline.set_edgecolor('w')

def save_png_colormap(img2d, out_path, vmin=None, vmax=None, cmap='jet',
                      mask=None, scalebar_um=None, px_um=None,
                      show_colorbar=False, dpi=300, out_px=None,
                      cbar_label="Intensity (a.u.)",
                      bar_anchor="br", bar_font=10):
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    shown = np.array(img2d, copy=True)
    if mask is not None:
        shown = np.ma.array(shown, mask=~mask)
        cmap_obj = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
        try:
            cmap_obj.set_bad(alpha=0.0)
        except Exception:
            pass
        ax.imshow(shown, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if (scalebar_um is not None) and (px_um is not None) and (scalebar_um > 0):
        H, W = shown.shape[:2]
        bar_px = int(round(float(scalebar_um) / float(px_um)))
        bar_px = max(2, min(bar_px, int(0.8 * W)))
        draw_scalebar(
            ax, W, H, bar_px, bar_px * float(px_um),
            anchor=bar_anchor, font_size=bar_font
        )
    if show_colorbar and (vmin is not None) and (vmax is not None):
        add_short_colorbar(fig, ax, vmin, vmax, cmap=cmap, label=cbar_label)
    fig.tight_layout(pad=0)
    if out_px:
        fig.set_size_inches(out_px[0] / dpi, out_px[1] / dpi)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

# ===================== GUI Log Redirector =====================

class GuiLogger:
    """
    stdout/stderr를 ScrolledText + 파일에 동시에 쓰는 로거.
    """
    def __init__(self, text_widget, orig_stream):
        self.text_widget = text_widget
        self.orig = orig_stream
        self.log_path = None
        self.lock = threading.Lock()

    def set_log_path(self, path):
        self.log_path = path
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[START] {ts}\n")

    def write(self, msg):
        if self.orig:
            try:
                self.orig.write(msg)
            except Exception:
                pass
        if not msg:
            return
        if self.log_path:
            with self.lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
        def append():
            try:
                self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
            except Exception:
                pass
        try:
            self.text_widget.after(0, append)
        except Exception:
            pass

    def flush(self):
        if self.orig:
            try:
                self.orig.flush()
            except Exception:
                pass

# ===================== Excel 저장 =====================

def save_excel(rows_all, keymap, xls_dir):
    df = pd.DataFrame(rows_all)
    if df.empty:
        return
    base_cols = ["stage", "time", "roi", "area_px",
                 "bg_mode", "bg_scope", "clip_neg", "bg_stride"]
    dyn_cols = sorted(
        [c for c in df.columns if c not in base_cols],
        key=natural_key
    )
    df = df[base_cols + dyn_cols]

    df["stage_idx"] = df["stage"].str.extract(
        r"S(\d+)", expand=False).astype(int)
    if df["time"].notna().any():
        df["time_idx"] = df["time"].fillna("t0").str.extract(
            r"t(\d+)", expand=False).astype(int)
    else:
        df["time_idx"] = 0
    df["roi_lab"] = "s" + df["stage_idx"].astype(str) + \
        "c" + df["roi"].astype(str)
    df["roi_id"] = df["stage"] + "_roi" + df["roi"].astype(str)

    xlsx = os.path.join(xls_dir, "fluor_intensity_perROI.xlsx")
    csv = os.path.join(xls_dir, "fluor_intensity_perROI.csv")

    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="per_ROI")

        is_tl = any(k[1] is not None for k in keymap.keys())
        if not is_tl:
            ch_list = sorted({
                int(m.group(1))
                for col in df.columns
                if (m := re.match(r"ch(\d+)_mean", col))
            })
            for ch in ch_list:
                keep = ["stage", "roi", "roi_id", "area_px"] + \
                    [c for c in df.columns if c.startswith(f"ch{ch}_")]
                keep = [c for c in keep if c in df.columns]
                sub = df[keep].copy().sort_values(["stage", "roi"])
                sub.insert(0, "No.", range(1, len(sub) + 1))
                sub.to_excel(w, index=False, sheet_name=f"ch{ch}")
        else:
            for ch in sorted({
                int(m.group(1))
                for col in df.columns
                if (m := re.match(r"ch(\d+)_mean", col))
            }):
                mean_mat = df.pivot(
                    index="time_idx",
                    columns="roi_lab",
                    values=f"ch{ch}_mean"
                ).sort_index()
                med_mat = df.pivot(
                    index="time_idx",
                    columns="roi_lab",
                    values=f"ch{ch}_median"
                ).sort_index()
                mean_mat.to_excel(w, sheet_name=f"ch{ch}_mean_matrix")
                med_mat.to_excel(w, sheet_name=f"ch{ch}_median_matrix")

    df.to_csv(csv, index=False)
    print(f"[저장 완료] {xlsx} 및 CSV")

# ===================== 워커 함수 (ProcessPool용) =====================

def _process_key_task(task):
    """
    한 (Stage, Time) key 처리:
    - 채널 로딩
    - ROI/Mask 로딩
    - BG 보정
    - ROI 정량
    - (선택) TIF/PNG 저장
    반환: {rows, steps, logs}
    """
    try:
        s = task["s"]
        t = task["t"]
        stid = task["stid"]
        chs_to_quant = task["chs_to_quant"]
        chmap = task["chmap"]
        roi_dir = task["roi_dir"]
        timelapse = task["timelapse"]

        imgs_raw = {}
        for ch in chs_to_quant:
            pth = chmap.get(ch)
            if pth is None:
                continue
            imgs_raw[ch] = read_2d(pth)
        if not imgs_raw:
            return {"rows": [], "steps": 1,
                    "logs": [f"[SKIP] {stid} — 채널 없음"]}

        any_img = next(iter(imgs_raw.values()))
        H, W = any_img.shape

        polys, union_mask = load_roi_polys_or_mask(
            roi_dir, s, t, timelapse, img_shape=(H, W)
        )
        if (polys is None) and (union_mask is None) and task["skip_no_roi"]:
            return {"rows": [], "steps": 1,
                    "logs": [f"[SKIP] {stid} — ROI 없음"]}

        # BG scope
        scope_mask = None
        if task["bg_scope"] == "roi_union":
            if polys is not None:
                u = np.zeros((H, W), dtype=bool)
                for P in polys:
                    u |= rasterize_polygon(P, (H, W))
                scope_mask = u
            elif union_mask is not None:
                scope_mask = union_mask

        # BG 보정
        imgs_bc = {}
        bg_used = {}
        p_glob = float(task["percentile"])
        for ch, img in imgs_raw.items():
            if task["per_channel_p"]:
                pp = float(task["ch_p_map"].get(ch, p_glob))
            else:
                pp = p_glob
            bc, B = bg_correct(
                img,
                mode=task["bg_mode"],
                p=pp,
                scope_mask=scope_mask,
                clip_neg=task["clip_neg"],
                stride=int(task["bg_stride"])
            )
            imgs_bc[ch] = bc
            bg_used[ch] = {"bg": float(B), "p": float(pp)}

        # ROI 정량
        per_roi = quantify_per_roi_multi(
            imgs_bc, polys=polys, union_mask=union_mask
        )

        rows = []
        for r in per_roi:
            r.update({
                "stage": s,
                "time": t if timelapse else None,
                "bg_scope": task["bg_scope"],
                "bg_mode": task["bg_mode"],
                "clip_neg": bool(task["clip_neg"]),
                "bg_stride": int(task["bg_stride"]),
            })
            for ch in chs_to_quant:
                if ch in bg_used:
                    r[f"ch{ch}_bg"] = bg_used[ch]["bg"]
                    r[f"ch{ch}_p"] = bg_used[ch]["p"]
                r[f"ch{ch}_color"] = task["ch_color_map"].get(
                    ch, "Grayscale"
                )
            rows.append(r)

        logs = [f"[DONE-QUANT] {stid} ROI={len(per_roi)}"]
        steps = max(1, len(per_roi))

        if not (task["do_tif"] or task["do_png"]):
            return {"rows": rows, "steps": steps, "logs": logs}

        H, W = any_img.shape
        auto_lo = float(task["auto_lo"])
        auto_hi = float(task["auto_hi"])
        px_um = task["px_um"]

        # union mask
        union_for_mask = None
        if polys is not None:
            u = np.zeros((H, W), dtype=bool)
            for P in polys:
                u |= rasterize_polygon(P, (H, W))
            union_for_mask = u
        elif union_mask is not None:
            union_for_mask = union_mask

        # ---- TIF 출력 ----
        if task["do_tif"]:
            for ch, bc in imgs_bc.items():
                img_to_save = bc
                if task["tif_mask_outside"] and (union_for_mask is not None):
                    masked = np.zeros_like(bc, dtype=np.float32)
                    masked[union_for_mask] = bc[union_for_mask]
                    img_to_save = masked
                out32 = os.path.join(
                    task["tif32_dir"],
                    f"{stid}_ch{ch}_bgcorr.tif"
                )
                imwrite(out32, img_to_save.astype(np.float32))

                vals = img_to_save[np.isfinite(img_to_save)]
                if vals.size > 0:
                    lo, hi = auto_minmax(vals, auto_lo, auto_hi)
                    clip_ = np.clip(img_to_save, lo, hi)
                    norm = (clip_ - lo) / (hi - lo + 1e-12)
                    out16 = os.path.join(
                        task["tif16_dir"],
                        f"{stid}_ch{ch}_bgcorr_preview.tif"
                    )
                    imwrite(
                        out16,
                        (norm * 65535).astype(np.uint16),
                        photometric='minisblack'
                    )

        # ---- PNG 출력 ----
        if task["do_png"]:

            def get_png_dirs_for_ch(ch):
                tag = f"ch{ch}"
                full_dir = os.path.join(task["png_root"], "full", tag)
                crop_dir = os.path.join(task["png_root"], "crop", tag)
                os.makedirs(full_dir, exist_ok=True)
                os.makedirs(crop_dir, exist_ok=True)
                return full_dir, crop_dir

            def get_vminmax(data_vals, cmin, cmax, cmap_on):
                if not cmap_on:
                    return None, None
                vmin = float(cmin) if cmin != "" else None
                vmax = float(cmax) if cmax != "" else None
                if (vmin is None) or (vmax is None) or (vmax <= vmin):
                    lo, hi = auto_minmax(data_vals, auto_lo, auto_hi)
                    vmin = lo if vmin is None else vmin
                    if (vmax is None) or (vmax <= vmin):
                        vmax = hi
                return vmin, vmax

            for ch, bc in imgs_bc.items():
                vals = bc[np.isfinite(bc)]

                # FULL
                if task["full_on"]:
                    cmap_full = (
                        task["full_cmap"]
                        if task["full_cmap_on"]
                        else get_cmap_for_color(
                            task["ch_color_map"].get(ch, "Grayscale")
                        )
                    )
                    vmin, vmax = get_vminmax(
                        vals, task["full_cmin"], task["full_cmax"],
                        task["full_cmap_on"]
                    )
                    full_union = union_for_mask if task["full_mask_outside"] else None
                    full_dir, _ = get_png_dirs_for_ch(ch)
                    outp = os.path.join(full_dir, f"{stid}_ch{ch}.png")
                    save_png_colormap(
                        bc, outp,
                        vmin=vmin, vmax=vmax, cmap=cmap_full,
                        mask=full_union,
                        scalebar_um=(
                            task["full_sb_um"]
                            if task["full_sb_on"] else None
                        ),
                        px_um=px_um,
                        show_colorbar=bool(
                            task["full_cbar_on"] and task["full_cmap_on"]
                        ),
                        dpi=int(task["full_png_dpi"]),
                        out_px=None,
                        cbar_label=f"ch{ch} Intensity",
                        bar_anchor=task["full_sb_anchor"],
                        bar_font=int(task["full_sb_font"])
                    )

                # CROP
                if task["crop_on"] and (polys is not None or union_for_mask is not None):
                    _, crop_dir = get_png_dirs_for_ch(ch)
                    cmap_crop = (
                        task["crop_cmap"]
                        if task["crop_cmap_on"]
                        else get_cmap_for_color(
                            task["ch_color_map"].get(ch, "Grayscale")
                        )
                    )

                    if polys is not None:
                        roi_list = list(range(1, len(polys) + 1))
                        if task["subset_on"] and (task["subset_roi"] is not None):
                            k = int(task["subset_roi"])
                            if 1 <= k <= len(polys):
                                roi_list = [k]
                            else:
                                roi_list = []

                        for i in roi_list:
                            P = np.asarray(polys[i - 1])
                            minx, maxx = P[:, 0].min(), P[:, 0].max()
                            miny, maxy = P[:, 1].min(), P[:, 1].max()
                            pad = max(10, int(0.05 * max(W, H)))
                            x0 = max(int(minx) - pad, 0)
                            x1 = min(int(maxx) + pad, W - 1)
                            y0 = max(int(miny) - pad, 0)
                            y1 = min(int(maxy) + pad, H - 1)
                            P2 = P.copy()
                            P2[:, 0] -= x0
                            P2[:, 1] -= y0
                            local_mask = rasterize_polygon(
                                P2, (y1 - y0 + 1, x1 - x0 + 1)
                            )
                            crop = bc[y0:y1 + 1, x0:x1 + 1]
                            use_vals = crop[local_mask] if task["crop_mask_outside"] else crop[np.isfinite(crop)]
                            vmin, vmax = get_vminmax(
                                use_vals, task["crop_cmin"], task["crop_cmax"],
                                task["crop_cmap_on"]
                            )
                            used_mask = local_mask if task["crop_mask_outside"] else None
                            out_px = (
                                (task["crop_w"], task["crop_h"])
                                if task["fixed_crop"] else None
                            )
                            out_png_path = os.path.join(
                                crop_dir,
                                f"{stid}_roi{i}_ch{ch}.png"
                            )
                            save_png_colormap(
                                crop, out_png_path,
                                vmin=vmin, vmax=vmax, cmap=cmap_crop,
                                mask=used_mask,
                                scalebar_um=(
                                    task["crop_sb_um"]
                                    if task["crop_sb_on"] else None
                                ),
                                px_um=px_um,
                                show_colorbar=bool(
                                    task["crop_cbar_on"]
                                    and task["crop_cmap_on"]
                                ),
                                dpi=int(task["png_dpi"]),
                                out_px=out_px,
                                cbar_label=f"ch{ch} Intensity",
                                bar_anchor=task["crop_sb_anchor"],
                                bar_font=int(task["crop_sb_font"])
                            )

                            if task["save_raw_crop_tif"]:
                                raw = imgs_raw.get(ch, None)
                                if raw is not None:
                                    raw_crop = raw[y0:y1 + 1, x0:x1 + 1]
                                    out_raw = os.path.join(
                                        task["tif32_dir"],
                                        f"{stid}_roi{i}_ch{ch}_raw.tif"
                                    )
                                    imwrite(
                                        out_raw,
                                        raw_crop.astype(np.float32)
                                    )

                    else:
                        # union mask만 있는 경우
                        m = union_for_mask.astype(bool)
                        ys, xs = np.where(m)
                        if ys.size > 0:
                            miny, maxy = ys.min(), ys.max()
                            minx, maxx = xs.min(), xs.max()
                            pad = max(10, int(0.05 * max(W, H)))
                            x0 = max(int(minx) - pad, 0)
                            x1 = min(int(maxx) + pad, W - 1)
                            y0 = max(int(miny) - pad, 0)
                            y1 = min(int(maxy) + pad, H - 1)
                            local_mask = m[y0:y1 + 1, x0:x1 + 1]
                            crop = bc[y0:y1 + 1, x0:x1 + 1]
                            vmin, vmax = get_vminmax(
                                crop[local_mask],
                                task["crop_cmin"], task["crop_cmax"],
                                task["crop_cmap_on"]
                            )
                            used_mask = local_mask if task["crop_mask_outside"] else None
                            out_px = (
                                (task["crop_w"], task["crop_h"])
                                if task["fixed_crop"] else None
                            )
                            out_png_path = os.path.join(
                                task["png_root"], "crop",
                                f"{stid}_roi1_ch{ch}.png"
                            )
                            save_png_colormap(
                                crop, out_png_path,
                                vmin=vmin, vmax=vmax, cmap=cmap_crop,
                                mask=used_mask,
                                scalebar_um=(
                                    task["crop_sb_um"]
                                    if task["crop_sb_on"] else None
                                ),
                                px_um=px_um,
                                show_colorbar=bool(
                                    task["crop_cbar_on"]
                                    and task["crop_cmap_on"]
                                ),
                                dpi=int(task["png_dpi"]),
                                out_px=out_px,
                                cbar_label=f"ch{ch} Intensity",
                                bar_anchor=task["crop_sb_anchor"],
                                bar_font=int(task["crop_sb_font"])
                            )

        return {"rows": rows, "steps": steps, "logs": logs}

    except Exception as e:
        return {
            "rows": [], "steps": 1,
            "logs": [f"[ERROR][WORKER] {task.get('stid','?')}: {e}"]
        }

# ===================== 통합 GUI 앱 =====================

class FluorIntensityApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fluor Intensity Builder (v3.1.1 + Log)")
        self.root.resizable(False, False)

        self.running = False
        self.cancel_requested = False
        self.progress_total = 1
        self.progress_done = 0
        self.start_ts = None

        self.gui_logger = None

        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    # ---------- GUI 구성 ----------

    def _build_gui(self):
        pad = {'padx':8, 'pady':6}

        # 경로
        self.img_v = tk.StringVar()
        self.roi_v = tk.StringVar()
        self.out_v = tk.StringVar()

        self.tl_v = tk.BooleanVar(value=False)

        # 채널
        self.n_ch_v = tk.StringVar(value="2")
        self.ch_num_vars = []
        self.ch_color_vars = []

        # BG
        self.scope_v = tk.StringVar(value="full")
        self.bgmode_v = tk.StringVar(value="percentile")
        self.p_v = tk.StringVar(value="1.0")
        self.per_ch_v = tk.BooleanVar(value=False)
        self.ch_p_vars = {i: tk.StringVar(value="1.0") for i in [1,2,3,4]}
        self.clip_v = tk.BooleanVar(value=True)
        self.bg_stride_v = tk.StringVar(value="4")

        # 픽셀크기
        self.px_v = tk.StringVar(value="")

        # 출력 옵션
        self.out_xls_v = tk.BooleanVar(value=True)
        self.out_tif_v = tk.BooleanVar(value=True)
        self.out_png_v = tk.BooleanVar(value=True)
        self.tif_mask_v = tk.BooleanVar(value=True)
        self.save_raw_crop_tif_v = tk.BooleanVar(value=False)

        # PNG / Full
        self.full_on_v = tk.BooleanVar(value=False)
        self.full_mask_v = tk.BooleanVar(value=False)
        self.full_sb_v = tk.BooleanVar(value=False)
        self.full_sb_um_v = tk.StringVar(value="20")
        self.full_sb_anchor_v = tk.StringVar(value="br")
        self.full_sb_font_v = tk.StringVar(value="10")
        self.full_cmap_on_v = tk.BooleanVar(value=False)
        self.full_cmap_v = tk.StringVar(value="jet")
        self.full_cmin_v = tk.StringVar(value="")
        self.full_cmax_v = tk.StringVar(value="")
        self.full_cbar_v = tk.BooleanVar(value=False)
        self.full_dpi_v = tk.StringVar(value="300")

        # PNG / Crop
        self.crop_on_v = tk.BooleanVar(value=True)
        self.crop_mask_v = tk.BooleanVar(value=True)
        self.crop_sb_v = tk.BooleanVar(value=False)
        self.crop_sb_um_v = tk.StringVar(value="20")
        self.crop_sb_anchor_v = tk.StringVar(value="br")
        self.crop_sb_font_v = tk.StringVar(value="10")
        self.crop_cmap_on_v = tk.BooleanVar(value=False)
        self.crop_cmap_v = tk.StringVar(value="jet")
        self.crop_cmin_v = tk.StringVar(value="")
        self.crop_cmax_v = tk.StringVar(value="")
        self.crop_cbar_v = tk.BooleanVar(value=False)
        self.png_dpi_v = tk.StringVar(value="300")
        self.fixed_crop_v = tk.BooleanVar(value=True)
        self.crop_w_v = tk.StringVar(value="500")
        self.crop_h_v = tk.StringVar(value="500")

        # 부분 추출
        self.subset_on_v = tk.BooleanVar(value=False)
        self.subset_stage_v = tk.StringVar(value="")
        self.subset_time_v = tk.StringVar(value="")
        self.subset_roi_v = tk.StringVar(value="")

        # Auto clip
        self.auto_lo_v = tk.StringVar(value="1.0")
        self.auto_hi_v = tk.StringVar(value="99.0")

        # 병렬 처리
        self.proc_parallel_v = tk.BooleanVar(value=True)
        self.proc_workers_v = tk.StringVar(value="0")

        # ----- 경로 입력 -----
        tk.Label(self.root, text="이미지 폴더").grid(row=0, column=0, sticky="w", **pad)
        fr = tk.Frame(self.root); fr.grid(row=0, column=1, sticky="ew", **pad)
        tk.Entry(fr, textvariable=self.img_v, width=52).pack(side="left")
        tk.Button(fr, text="찾기", width=8,
                  command=lambda:self._browse_dir(self.img_v)).pack(side="left", padx=6)

        tk.Label(self.root, text="ROI 폴더").grid(row=1, column=0, sticky="w", **pad)
        fr = tk.Frame(self.root); fr.grid(row=1, column=1, sticky="ew", **pad)
        tk.Entry(fr, textvariable=self.roi_v, width=52).pack(side="left")
        tk.Button(fr, text="찾기", width=8,
                  command=lambda:self._browse_dir(self.roi_v)).pack(side="left", padx=6)

        tk.Label(self.root, text="출력 루트(선택)").grid(row=2, column=0, sticky="w", **pad)
        fr = tk.Frame(self.root); fr.grid(row=2, column=1, sticky="ew", **pad)
        tk.Entry(fr, textvariable=self.out_v, width=52).pack(side="left")
        tk.Button(fr, text="찾기", width=8,
                  command=lambda:self._browse_dir(self.out_v)).pack(side="left", padx=6)

        tk.Checkbutton(
            self.root,
            text="Timelapse(시간축 있음: 파일명 SXX_TXX_X)",
            variable=self.tl_v,
            command=self._toggle_subset
        ).grid(row=3, column=0, columnspan=2,
               sticky="w", padx=8, pady=(4, 0))

        # ----- 채널 설정 -----
        chf = tk.LabelFrame(self.root, text="채널 설정")
        chf.grid(row=4, column=0, columnspan=2,
                 sticky="we", padx=8, pady=(6, 6))
        tk.Label(chf, text="총 채널 수").grid(row=0, column=0, sticky="e", padx=6)
        cb_nc = ttk.Combobox(
            chf, textvariable=self.n_ch_v,
            values=["1", "2", "3", "4"], width=6, state="readonly"
        )
        cb_nc.grid(row=0, column=1, sticky="w")
        self.ch_rows_frame = tk.Frame(chf)
        self.ch_rows_frame.grid(row=1, column=0, columnspan=6,
                                sticky="we", pady=(6, 0))
        self.n_ch_v.trace_add("write", lambda *_: self._build_channel_rows())
        self._build_channel_rows()

        # ----- 채널 설정 + BG 옵션 (한 줄에 정리) -----
        ch_bg_frame = tk.Frame(self.root)
        ch_bg_frame.grid(row=4, column=0, columnspan=2,
                         sticky="we", padx=8, pady=(6, 6))
        ch_bg_frame.grid_columnconfigure(0, weight=1)
        ch_bg_frame.grid_columnconfigure(1, weight=1)

        # 채널 설정
        chf = tk.LabelFrame(ch_bg_frame, text="채널 설정")
        chf.grid(row=0, column=0, sticky="nwe", padx=(0, 6))
        tk.Label(chf, text="총 채널 수").grid(row=0, column=0, sticky="e", padx=6)
        cb_nc = ttk.Combobox(
            chf, textvariable=self.n_ch_v,
            values=["1", "2", "3", "4"], width=6, state="readonly"
        )
        cb_nc.grid(row=0, column=1, sticky="w")

        self.ch_rows_frame = tk.Frame(chf)
        self.ch_rows_frame.grid(row=1, column=0, columnspan=6,
                                sticky="we", pady=(6, 0))
        self.n_ch_v.trace_add("write", lambda *_: self._build_channel_rows())
        self._build_channel_rows()

        # BG 옵션
        bgf = tk.LabelFrame(ch_bg_frame, text="BG 옵션")
        bgf.grid(row=0, column=1, sticky="nwe")

        tk.Label(bgf, text="Scope").grid(row=0, column=0, sticky="e", padx=6, pady=4)
        ttk.Combobox(
            bgf, textvariable=self.scope_v,
            values=["full", "roi_union"], width=10, state="readonly"
        ).grid(row=0, column=1, sticky="w")

        tk.Label(bgf, text="Mode").grid(row=0, column=2, sticky="e", padx=6, pady=4)
        ttk.Combobox(
            bgf, textvariable=self.bgmode_v,
            values=["percentile", "hist-mode"], width=12, state="readonly"
        ).grid(row=0, column=3, sticky="w")

        tk.Label(bgf, text="percentile p(%)").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        tk.Entry(bgf, textvariable=self.p_v, width=8).grid(row=1, column=1, sticky="w")

        tk.Checkbutton(bgf, text="채널별 p", variable=self.per_ch_v).grid(
            row=1, column=2, sticky="w", padx=4
        )

        row_p = tk.Frame(bgf)
        row_p.grid(row=1, column=3, columnspan=3, sticky="w")
        for i in [1, 2, 3, 4]:
            tk.Label(row_p, text=f"ch{i}").pack(side="left")
            tk.Entry(row_p, textvariable=self.ch_p_vars[i], width=4).pack(side="left", padx=(0, 4))

        tk.Checkbutton(bgf, text="음수 clip", variable=self.clip_v).grid(
            row=2, column=0, sticky="w", padx=6, pady=(2, 4)
        )

        tk.Label(bgf, text="bg_stride").grid(row=2, column=1, sticky="e", padx=6, pady=(2, 4))
        tk.Entry(bgf, textvariable=self.bg_stride_v, width=6).grid(
            row=2, column=2, sticky="w", pady=(2, 4)
        )

        # ----- 픽셀 크기 -----
        tk.Label(self.root, text="픽셀크기 (µm/px) — 스케일바용").grid(
            row=6, column=0, sticky="w", **pad
        )
        tk.Entry(self.root, textvariable=self.px_v,
                 width=12).grid(row=6, column=1, sticky="w", **pad)

        # ----- 출력 그룹 -----
        outf = tk.LabelFrame(self.root, text="출력 그룹")
        outf.grid(row=7, column=0, columnspan=2,
                  sticky="we", padx=8, pady=(6, 6))
        tk.Checkbutton(outf, text="XLS/CSV",
                       variable=self.out_xls_v).grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        tk.Checkbutton(outf, text="TIF (BG-correct / ROI masking)",
                       variable=self.out_tif_v).grid(
            row=0, column=1, sticky="w", padx=8, pady=4
        )
        tk.Checkbutton(outf, text="PNG",
                       variable=self.out_png_v,
                       command=self._toggle_png_group).grid(
            row=0, column=2, sticky="w", padx=8, pady=4
        )
        tk.Checkbutton(outf, text="TIF ROI 외부 0 마스킹",
                       variable=self.tif_mask_v).grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )
        tk.Checkbutton(outf, text="원본 Crop TIF 저장",
                       variable=self.save_raw_crop_tif_v).grid(
            row=1, column=1, sticky="w", padx=8, pady=4
        )

        # ----- PNG 옵션 -----
        pngf = tk.LabelFrame(self.root, text="PNG 옵션")
        pngf.grid(row=8, column=0, columnspan=2,
                  sticky="we", padx=8, pady=(4, 6))

        # Full
        self.png_full_box = tk.LabelFrame(pngf, text="Full image 저장")
        self.png_full_box.grid(row=0, column=0, sticky="we", padx=6, pady=6)
        tk.Checkbutton(self.png_full_box, text="활성화",
                       variable=self.full_on_v,
                       command=self._toggle_full_box).grid(
            row=0, column=0, sticky="w", padx=6
        )
        self.chk_full_mask = tk.Checkbutton(
            self.png_full_box, text="ROI 외부 마스킹",
            variable=self.full_mask_v
        )
        self.chk_full_mask.grid(row=0, column=1, sticky="w", padx=6)

        self.full_inner = tk.Frame(self.png_full_box)
        self.full_inner.grid(row=1, column=0, columnspan=4,
                             sticky="we", padx=2)

        tk.Checkbutton(self.full_inner, text="Scale bar 추가",
                       variable=self.full_sb_v,
                       command=self._toggle_full_box).grid(
            row=0, column=0, sticky="w", padx=6
        )
        tk.Label(self.full_inner, text="길이(µm):").grid(
            row=0, column=1, sticky="e")
        self.e_full_sb_um = tk.Entry(
            self.full_inner, textvariable=self.full_sb_um_v,
            width=6
        )
        self.e_full_sb_um.grid(row=0, column=2, sticky="w")
        tk.Label(self.full_inner, text="위치:").grid(
            row=0, column=3, sticky="e")
        self.cb_full_sb_anchor = ttk.Combobox(
            self.full_inner,
            textvariable=self.full_sb_anchor_v,
            values=SB_ANCHORS, width=4, state="readonly"
        )
        self.cb_full_sb_anchor.grid(row=0, column=4, sticky="w")
        tk.Label(self.full_inner, text="글자크기:").grid(
            row=0, column=5, sticky="e")
        self.e_full_sb_font = tk.Entry(
            self.full_inner, textvariable=self.full_sb_font_v,
            width=4
        )
        self.e_full_sb_font.grid(row=0, column=6, sticky="w")

        tk.Checkbutton(self.full_inner, text="Color map 재설정",
                       variable=self.full_cmap_on_v,
                       command=self._toggle_full_box).grid(
            row=1, column=0, sticky="w", padx=6
        )
        tk.Label(self.full_inner, text="Color map:").grid(
            row=1, column=1, sticky="e"
        )
        self.cb_full_cmap = ttk.Combobox(
            self.full_inner, textvariable=self.full_cmap_v,
            values=CMAP_CHOICES, width=10, state="readonly"
        )
        self.cb_full_cmap.grid(row=1, column=2, sticky="w")
        tk.Label(self.full_inner, text="최소값:").grid(
            row=1, column=3, sticky="e")
        self.e_full_cmin = tk.Entry(
            self.full_inner, textvariable=self.full_cmin_v,
            width=8
        )
        self.e_full_cmin.grid(row=1, column=4, sticky="w")
        tk.Label(self.full_inner, text="최대값:").grid(
            row=1, column=5, sticky="e")
        self.e_full_cmax = tk.Entry(
            self.full_inner, textvariable=self.full_cmax_v,
            width=8
        )
        self.e_full_cmax.grid(row=1, column=6, sticky="w")
        self.cb_full_cbar = tk.Checkbutton(
            self.full_inner, text="Color bar 표시",
            variable=self.full_cbar_v
        )
        self.cb_full_cbar.grid(row=1, column=7, sticky="w", padx=6)

        tk.Label(self.full_inner, text="PNG DPI:").grid(
            row=2, column=0, sticky="e", padx=6)
        self.e_full_dpi = tk.Entry(
            self.full_inner, textvariable=self.full_dpi_v,
            width=6
        )
        self.e_full_dpi.grid(row=2, column=1, sticky="w")

        # Crop
        self.png_crop_box = tk.LabelFrame(pngf, text="Crop image 저장")
        self.png_crop_box.grid(row=0, column=1, sticky="we",
                               padx=6, pady=6)
        tk.Checkbutton(self.png_crop_box, text="활성화",
                       variable=self.crop_on_v,
                       command=self._toggle_crop_box).grid(
            row=0, column=0, sticky="w", padx=6
        )
        self.chk_crop_mask = tk.Checkbutton(
            self.png_crop_box, text="ROI 외부 마스킹",
            variable=self.crop_mask_v
        )
        self.chk_crop_mask.grid(row=0, column=1,
                                sticky="w", padx=6)

        self.crop_inner = tk.Frame(self.png_crop_box)
        self.crop_inner.grid(row=1, column=0, columnspan=4,
                             sticky="we", padx=2)

        tk.Checkbutton(self.crop_inner, text="Scale bar 추가",
                       variable=self.crop_sb_v,
                       command=self._toggle_crop_box).grid(
            row=0, column=0, sticky="w", padx=6
        )
        tk.Label(self.crop_inner, text="길이(µm):").grid(
            row=0, column=1, sticky="e")
        self.e_crop_sb_um = tk.Entry(
            self.crop_inner, textvariable=self.crop_sb_um_v,
            width=6
        )
        self.e_crop_sb_um.grid(row=0, column=2, sticky="w")
        tk.Label(self.crop_inner, text="위치:").grid(
            row=0, column=3, sticky="e")
        self.cb_crop_sb_anchor = ttk.Combobox(
            self.crop_inner,
            textvariable=self.crop_sb_anchor_v,
            values=SB_ANCHORS, width=4, state="readonly"
        )
        self.cb_crop_sb_anchor.grid(row=0, column=4, sticky="w")
        tk.Label(self.crop_inner, text="글자크기:").grid(
            row=0, column=5, sticky="e")
        self.e_crop_sb_font = tk.Entry(
            self.crop_inner, textvariable=self.crop_sb_font_v,
            width=4
        )
        self.e_crop_sb_font.grid(row=0, column=6, sticky="w")

        tk.Checkbutton(self.crop_inner, text="Color map 재설정",
                       variable=self.crop_cmap_on_v,
                       command=self._toggle_crop_box).grid(
            row=1, column=0, sticky="w", padx=6
        )
        tk.Label(self.crop_inner, text="Color map:").grid(
            row=1, column=1, sticky="e")
        self.cb_crop_cmap = ttk.Combobox(
            self.crop_inner,
            textvariable=self.crop_cmap_v,
            values=CMAP_CHOICES, width=10,
            state="readonly"
        )
        self.cb_crop_cmap.grid(row=1, column=2, sticky="w")
        tk.Label(self.crop_inner, text="최소값:").grid(
            row=1, column=3, sticky="e")
        self.e_crop_cmin = tk.Entry(
            self.crop_inner, textvariable=self.crop_cmin_v,
            width=8
        )
        self.e_crop_cmin.grid(row=1, column=4, sticky="w")
        tk.Label(self.crop_inner, text="최대값:").grid(
            row=1, column=5, sticky="e")
        self.e_crop_cmax = tk.Entry(
            self.crop_inner, textvariable=self.crop_cmax_v,
            width=8
        )
        self.e_crop_cmax.grid(row=1, column=6, sticky="w")
        self.cb_crop_cbar = tk.Checkbutton(
            self.crop_inner, text="Color bar 표시",
            variable=self.crop_cbar_v
        )
        self.cb_crop_cbar.grid(row=1, column=7, sticky="w", padx=6)

        # PNG/TIF 공통
        self.png_misc_box = tk.LabelFrame(
            pngf, text="PNG/TIF 공통 자동 스케일"
        )
        self.png_misc_box.grid(row=1, column=0, columnspan=2,
                               sticky="we", padx=6, pady=(0, 6))
        tk.Label(self.png_misc_box, text="PNG DPI:").grid(
            row=0, column=0, sticky="e", padx=6)
        tk.Entry(self.png_misc_box, textvariable=self.png_dpi_v,
                 width=6).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(
            self.png_misc_box,
            text="고정 crop 픽셀 크기 사용",
            variable=self.fixed_crop_v,
            command=self._toggle_crop_box
        ).grid(row=0, column=2, sticky="w", padx=8)
        tk.Label(self.png_misc_box, text="WxH(px):").grid(
            row=0, column=3, sticky="e")
        self.e_cw = tk.Entry(
            self.png_misc_box, textvariable=self.crop_w_v,
            width=6
        )
        self.e_cw.grid(row=0, column=4, sticky="w")
        tk.Label(self.png_misc_box, text="×").grid(
            row=0, column=5, sticky="w")
        self.e_ch = tk.Entry(
            self.png_misc_box, textvariable=self.crop_h_v,
            width=6
        )
        self.e_ch.grid(row=0, column=6, sticky="w")

        tk.Label(self.png_misc_box,
                 text="Auto clip p_low / p_high (%)").grid(
            row=1, column=0, sticky="e", padx=6
        )
        tk.Entry(self.png_misc_box, textvariable=self.auto_lo_v,
                 width=6).grid(row=1, column=1, sticky="w")
        tk.Label(self.png_misc_box, text="/").grid(
            row=1, column=2, sticky="w")
        tk.Entry(self.png_misc_box, textvariable=self.auto_hi_v,
                 width=6).grid(row=1, column=3, sticky="w")

        # ----- 부분 추출 -----
        subf = tk.LabelFrame(self.root, text="부분 추출")
        subf.grid(row=9, column=0, columnspan=2,
                  sticky="we", padx=8, pady=(6, 6))
        tk.Checkbutton(subf, text="부분 추출 활성화",
                       variable=self.subset_on_v,
                       command=self._toggle_subset).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        tk.Label(subf, text="Stage(필수)").grid(
            row=1, column=0, sticky="e", padx=8)
        self.e_stage = tk.Entry(
            subf, textvariable=self.subset_stage_v,
            width=10, state=tk.DISABLED
        )
        self.e_stage.grid(row=1, column=1, sticky="w")
        tk.Label(subf, text="Time(선택; timelapse)").grid(
            row=1, column=2, sticky="e", padx=8)
        self.e_time = tk.Entry(
            subf, textvariable=self.subset_time_v,
            width=10, state=tk.DISABLED
        )
        self.e_time.grid(row=1, column=3, sticky="w")
        tk.Label(subf, text="ROI(선택)").grid(
            row=1, column=4, sticky="e", padx=8)
        self.e_roi = tk.Entry(
            subf, textvariable=self.subset_roi_v,
            width=10, state=tk.DISABLED
        )
        self.e_roi.grid(row=1, column=5, sticky="w")

        # ----- 병렬 처리 -----
        par_box = tk.LabelFrame(self.root, text="병렬 처리")
        par_box.grid(row=10, column=0, columnspan=2,
                     sticky="we", padx=8, pady=(4, 6))
        tk.Checkbutton(
            par_box,
            text="ProcessPoolExecutor 사용",
            variable=self.proc_parallel_v
        ).grid(row=0, column=0, sticky="w", padx=8)
        tk.Label(par_box, text="프로세스 개수 (0=자동)").grid(
            row=0, column=1, sticky="e", padx=8)
        tk.Entry(par_box, textvariable=self.proc_workers_v,
                 width=6).grid(row=0, column=2, sticky="w")

        # ----- 실행 / 취소 / 종료 버튼 -----
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=11, column=0, columnspan=2, pady=(4, 4))
        self.btn_run = tk.Button(
            btn_frame, text="실행", width=12,
            command=self.on_run
        )
        self.btn_run.pack(side="left", padx=6)
        self.btn_cancel = tk.Button(
            btn_frame, text="중단", width=12,
            command=self.on_cancel, state=tk.DISABLED
        )
        self.btn_cancel.pack(side="left", padx=6)
        self.btn_exit = tk.Button(
            btn_frame, text="종료", width=12,
            command=self.on_close
        )
        self.btn_exit.pack(side="left", padx=6)

        # ----- Progress + Log -----
        prog_frame = tk.Frame(self.root)
        prog_frame.grid(row=12, column=0, columnspan=2,
                        sticky="we", padx=8, pady=(2, 2))
        tk.Label(prog_frame, text="Progress").grid(
            row=0, column=0, sticky="w")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.pbar = ttk.Progressbar(
            prog_frame, length=500,
            mode="determinate", variable=self.progress_var,
            maximum=1.0
        )
        self.pbar.grid(row=0, column=1, sticky="we", padx=6)
        self.lab_status = tk.Label(prog_frame, text="")
        self.lab_status.grid(row=1, column=0,
                             columnspan=2, sticky="w")
        self.lab_eta = tk.Label(
            prog_frame,
            text="경과: 00:00:00 | 남은: --:--:--"
        )
        self.lab_eta.grid(row=2, column=0,
                          columnspan=2, sticky="w")

        tk.Label(self.root, text="Log").grid(
            row=13, column=0, sticky="nw", padx=8)
        self.log_text = scrolledtext.ScrolledText(
            self.root, height=12, width=95
        )
        self.log_text.grid(
            row=13, column=1, sticky="we",
            padx=8, pady=(0, 8)
        )

        # stdout/stderr redirector 초기화
        self.gui_logger = GuiLogger(self.log_text, sys.stdout)
        sys.stdout = self.gui_logger
        sys.stderr = self.gui_logger

        # 초기 상태
        self._toggle_png_group()
        self._toggle_subset()
        self._toggle_full_box()
        self._toggle_crop_box()

    # ---------- UI helpers ----------

    def _browse_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _build_channel_rows(self):
        for w in self.ch_rows_frame.winfo_children():
            w.destroy()
        self.ch_num_vars.clear()
        self.ch_color_vars.clear()
        try:
            nC = int(self.n_ch_v.get())
        except:
            nC = 1
        tk.Label(self.ch_rows_frame, text="채널 #").grid(
            row=0, column=0, padx=6, sticky="w")
        tk.Label(self.ch_rows_frame, text="컬러").grid(
            row=0, column=1, padx=6, sticky="w")
        for i in range(nC):
            num_v = tk.StringVar(value=str(i + 1))
            cbn = ttk.Combobox(
                self.ch_rows_frame, textvariable=num_v,
                values=["0", "1", "2", "3", "4"],
                width=6, state="readonly"
            )
            cbn.grid(row=i + 1, column=0, padx=6,
                     pady=2, sticky="w")
            col_v = tk.StringVar(
                value=("Grayscale" if i == 0 else "Cyan")
            )
            cbc = ttk.Combobox(
                self.ch_rows_frame, textvariable=col_v,
                values=COLOR_CHOICES, width=12,
                state="readonly"
            )
            cbc.grid(row=i + 1, column=1, padx=6,
                     pady=2, sticky="w")
            self.ch_num_vars.append(num_v)
            self.ch_color_vars.append(col_v)

    def _set_frame_state_recursive(self, w, state):
        try:
            w.configure(state=state)
        except tk.TclError:
            pass
        for ch in w.winfo_children():
            self._set_frame_state_recursive(ch, state)

    def _toggle_png_group(self, *_):
        enabled = self.out_png_v.get()
        st = tk.NORMAL if enabled else tk.DISABLED
        self._set_frame_state_recursive(self.png_full_box, st)
        self._set_frame_state_recursive(self.png_crop_box, st)
        self._set_frame_state_recursive(self.png_misc_box, st)
        self._toggle_full_box()
        self._toggle_crop_box()

    def _toggle_full_box(self, *_):
        on = self.full_on_v.get() and self.out_png_v.get()
        st = tk.NORMAL if on else tk.DISABLED
        self._set_frame_state_recursive(self.full_inner, st)
        self.chk_full_mask.configure(state=st)

    def _toggle_crop_box(self, *_):
        on = self.crop_on_v.get() and self.out_png_v.get()
        st = tk.NORMAL if on else tk.DISABLED
        self._set_frame_state_recursive(self.crop_inner, st)
        self.chk_crop_mask.configure(state=st)
        st_fix = tk.NORMAL if (on and self.fixed_crop_v.get() and self.out_png_v.get()) else tk.DISABLED
        self.e_cw.configure(state=st_fix)
        self.e_ch.configure(state=st_fix)

    def _toggle_subset(self, *_):
        on = self.subset_on_v.get()
        self.e_stage.configure(state=(tk.NORMAL if on else tk.DISABLED))
        self.e_roi.configure(state=(tk.NORMAL if on else tk.DISABLED))
        if on and self.tl_v.get():
            self.e_time.configure(state=tk.NORMAL)
        else:
            self.e_time.configure(state=tk.DISABLED)

    # ---------- 실행/취소/종료 ----------

    def _set_all_inputs_state(self, state):
        # Log 영역, Progress, 종료/중단 버튼은 항상 활성 유지
        for child in self.root.winfo_children():
            if child in (self.log_text,):
                continue
            if child == self.btn_exit:
                continue
            if child == self.btn_cancel and state == tk.DISABLED:
                # 실행 끝날 때만 cancel 비활성화
                child.configure(state=tk.DISABLED)
                continue
            if child == self.btn_run:
                child.configure(state=state)
                continue
            try:
                child.configure(state=state)
            except tk.TclError:
                for g in child.winfo_children():
                    if g in (self.log_text, self.btn_exit, self.btn_run, self.btn_cancel):
                        continue
                    try:
                        g.configure(state=state)
                    except tk.TclError:
                        pass

    def on_run(self):
        if self.running:
            print("[알림] 이미 실행 중입니다.")
            return

        cfg = self._validate_and_collect()
        if cfg is None:
            return

        res_root = cfg["out_root"]
        log_dir = ensure_dir(os.path.join(res_root, "logs"))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"run_{ts}.txt")
        self.gui_logger.set_log_path(log_path)

        self.running = True
        self.cancel_requested = False
        self.start_ts = time.time()
        self.progress_done = 0
        self.progress_total = 1
        self.progress_var.set(0.0)
        self.lab_status.config(text="초기화 중...")
        self.lab_eta.config(text="경과: 00:00:00 | 남은: --:--:--")

        self._set_all_inputs_state(tk.DISABLED)
        self.btn_cancel.configure(state=tk.NORMAL)

        print(f"[정보] Fluor Intensity Builder 실행 시작")
        print(f"img_dir={cfg['img_dir']}")
        print(f"roi_dir={cfg['roi_dir']}")
        print(f"out_root={cfg['out_root']}")

        th = threading.Thread(target=self._run_pipeline, args=(cfg,), daemon=True)
        th.start()

    def on_cancel(self):
        if self.running:
            self.cancel_requested = True
            print("[알림] 중단 요청됨. 현재 작업 마무리 후 종료합니다.")

    def on_close(self):
        if self.running:
            if not messagebox.askyesno(
                "종료 확인",
                "현재 작업이 진행 중입니다. 정말 종료할까요?"
            ):
                return
        self.root.destroy()

    # ---------- 설정 검증 ----------

    def _validate_and_collect(self):
        img = self.img_v.get().strip()
        roi = self.roi_v.get().strip()
        out = self.out_v.get().strip()

        if not img or not os.path.isdir(img):
            messagebox.showerror("오류", "유효한 이미지 경로를 선택하세요.")
            return None
        if not roi or not os.path.isdir(roi):
            messagebox.showerror("오류", "유효한 ROI 경로를 선택하세요.")
            return None
        if not out:
            out = os.path.join(img, "RES_INT")

        # 채널
        try:
            nC = int(self.n_ch_v.get())
            assert 1 <= nC <= 4
        except:
            messagebox.showerror("오류", "채널 수는 1~4")
            return None

        chs = []
        ch_color_map = {}
        for i in range(nC):
            try:
                cnum = int(self.ch_num_vars[i].get())
                assert 0 <= cnum <= 4
            except:
                messagebox.showerror(
                    "오류",
                    f"{i+1}번째 채널 번호가 올바르지 않습니다 (0~4)."
                )
                return None
            col = self.ch_color_vars[i].get()
            chs.append(cnum)
            ch_color_map[cnum] = col
        if len(set(chs)) != len(chs):
            messagebox.showerror("오류", "채널 번호가 중복되었습니다.")
            return None

        # BG
        try:
            p = float(self.p_v.get())
            assert 0.0 <= p <= 10.0
        except:
            messagebox.showerror("오류", "percentile p는 0~10")
            return None

        ch_p_map = {i: p for i in [1, 2, 3, 4]}
        if self.per_ch_v.get():
            try:
                for i in [1, 2, 3, 4]:
                    ch_p_map[i] = float(self.ch_p_vars[i].get())
                    assert 0.0 <= ch_p_map[i] <= 10.0
            except:
                messagebox.showerror(
                    "오류", "채널별 p는 0~10"
                )
                return None

        try:
            bg_stride = int(float(self.bg_stride_v.get()))
            assert 1 <= bg_stride <= 100
        except:
            messagebox.showerror(
                "오류", "bg_stride는 1~100 정수"
            )
            return None

        # 픽셀 크기
        px_um = None
        if self.px_v.get().strip():
            try:
                px_um = float(self.px_v.get())
                assert 1e-4 <= px_um <= 100.0
            except:
                messagebox.showerror(
                    "오류", "픽셀크기(µm/px) 입력 오류"
                )
                return None

        # DPI
        try:
            full_dpi = int(float(self.full_dpi_v.get()))
            assert 50 <= full_dpi <= 1200
        except:
            full_dpi = 300
        try:
            png_dpi = int(float(self.png_dpi_v.get()))
            assert 50 <= png_dpi <= 1200
        except:
            png_dpi = 300

        # Crop 고정 크기
        fc = bool(self.fixed_crop_v.get())
        cw = chh = None
        if self.out_png_v.get() and self.crop_on_v.get() and fc:
            try:
                cw = int(float(self.crop_w_v.get()))
                chh = int(float(self.crop_h_v.get()))
                assert 32 <= cw <= 8000 and 32 <= chh <= 8000
            except:
                messagebox.showerror(
                    "오류", "crop 픽셀 크기(32~8000)"
                )
                return None

        # 부분 추출
        subset_on = bool(self.subset_on_v.get())
        ss_stage = ss_time = ss_roi = None
        if subset_on:
            try:
                ss_stage = int(self.subset_stage_v.get())
                assert ss_stage >= 0
            except:
                messagebox.showerror(
                    "오류", "Stage는 정수"
                )
                return None
            if self.tl_v.get() and self.subset_time_v.get().strip():
                try:
                    ss_time = int(self.subset_time_v.get())
                    assert ss_time >= 0
                except:
                    messagebox.showerror(
                        "오류", "Timepoint는 정수"
                    )
                    return None
            if self.subset_roi_v.get().strip():
                try:
                    ss_roi = int(self.subset_roi_v.get())
                    assert ss_roi >= 1
                except:
                    messagebox.showerror(
                        "오류", "ROI는 1 이상 정수"
                    )
                    return None

        # Auto clip
        try:
            auto_lo = float(self.auto_lo_v.get())
            auto_hi = float(self.auto_hi_v.get())
            assert 0.0 <= auto_lo < auto_hi <= 100.0
        except:
            messagebox.showerror(
                "오류",
                "Auto clip p_low/p_high는 0–100, p_low < p_high"
            )
            return None

        # 병렬 처리
        try:
            workers = int(self.proc_workers_v.get())
            assert 0 <= workers <= 64
        except:
            messagebox.showerror(
                "오류", "프로세스 개수는 0~64 정수"
            )
            return None

        cfg = dict(
            img_dir=img,
            roi_dir=roi,
            out_root=out,
            timelapse=bool(self.tl_v.get()),
            channels_to_quant=sorted(set(chs)),
            ch_color_map=ch_color_map,
            bg_scope=self.scope_v.get(),
            bg_mode=self.bgmode_v.get(),
            percentile=p,
            per_channel_p=bool(self.per_ch_v.get()),
            ch_p_map=ch_p_map,
            clip_neg=bool(self.clip_v.get()),
            bg_stride=bg_stride,
            px_um=px_um,
            out_xls=bool(self.out_xls_v.get()),
            out_tif=bool(self.out_tif_v.get()),
            out_png=bool(self.out_png_v.get()),
            tif_mask_outside=bool(self.tif_mask_v.get()),
            save_raw_crop_tif=bool(self.save_raw_crop_tif_v.get()),
            full_on=bool(self.full_on_v.get()),
            full_mask_outside=bool(self.full_mask_v.get()),
            full_sb_on=bool(self.full_sb_v.get()),
            full_sb_um=float(self.full_sb_um_v.get() or 0),
            full_sb_anchor=self.full_sb_anchor_v.get(),
            full_sb_font=int(self.full_sb_font_v.get() or 10),
            full_cmap_on=bool(self.full_cmap_on_v.get()),
            full_cmap=self.full_cmap_v.get(),
            full_cmin=self.full_cmin_v.get().strip(),
            full_cmax=self.full_cmax_v.get().strip(),
            full_cbar_on=bool(self.full_cbar_v.get()),
            full_png_dpi=full_dpi,
            crop_on=bool(self.crop_on_v.get()),
            crop_mask_outside=bool(self.crop_mask_v.get()),
            crop_sb_on=bool(self.crop_sb_v.get()),
            crop_sb_um=float(self.crop_sb_um_v.get() or 0),
            crop_sb_anchor=self.crop_sb_anchor_v.get(),
            crop_sb_font=int(self.crop_sb_font_v.get() or 10),
            crop_cmap_on=bool(self.crop_cmap_on_v.get()),
            crop_cmap=self.crop_cmap_v.get(),
            crop_cmin=self.crop_cmin_v.get().strip(),
            crop_cmax=self.crop_cmax_v.get().strip(),
            crop_cbar_on=bool(self.crop_cbar_v.get()),
            png_dpi=png_dpi,
            fixed_crop=fc,
            crop_w=cw if cw else 500,
            crop_h=chh if chh else 500,
            subset_on=subset_on,
            subset_stage=ss_stage,
            subset_time=ss_time,
            subset_roi=ss_roi,
            auto_clip_lo=auto_lo,
            auto_clip_hi=auto_hi,
            proc_parallel=bool(self.proc_parallel_v.get()),
            proc_workers=workers,
        )
        return cfg

    # ---------- 파이프라인 실행 ----------

    def _run_pipeline(self, cfg):
        try:
            img_dir = cfg["img_dir"]
            roi_dir = cfg["roi_dir"]
            out_root = ensure_dir(cfg["out_root"])
            timelapse = cfg["timelapse"]

            files = list_tifs(img_dir)
            keymap = build_keymap(files, timelapse=timelapse)
            if not keymap:
                print("해석 가능한 이미지가 없습니다.")
                return

            # 부분추출
            if cfg["subset_on"] and (cfg["subset_stage"] is not None):
                s_code = f"S{int(cfg['subset_stage']):02d}"
                if (not timelapse) or (cfg["subset_time"] is None):
                    keymap = {k: v for k, v in keymap.items() if k[0] == s_code}
                else:
                    t_code = f"t{int(cfg['subset_time']):02d}"
                    keymap = {k: v for k, v in keymap.items()
                              if (k[0] == s_code and k[1] == t_code)}
                if not keymap:
                    print(f"[부분추출] 조건(Stage={s_code})에 맞는 키가 없습니다.")
                    return

            out_xls = cfg["out_xls"]
            out_tif = cfg["out_tif"]
            out_png = cfg["out_png"]

            xls_dir = ensure_dir(os.path.join(out_root, "xls")) if out_xls else None
            tif32_dir = ensure_dir(os.path.join(out_root, "TIFF")) if out_tif else ""
            tif16_dir = ensure_dir(os.path.join(out_root, "TIFF16")) if out_tif else ""
            png_root = ensure_dir(os.path.join(out_root, "PNG")) if out_png else ""

            # 전체 스텝
            total_steps = 0
            for (s, t) in keymap.keys():
                n_roi = count_rois_fast(roi_dir, s, t, timelapse)
                total_steps += (n_roi if n_roi > 0 else 1)
            if total_steps <= 0:
                total_steps = 1
            self._set_progress_total(total_steps)
            print(f"[정보] 총 작업 단위(ROI/키): {total_steps}")

            # 태스크
            tasks = []
            for (s, t), chmap in keymap.items():
                stid = f"{s}_{t}" if (timelapse and t is not None) else s
                task = {
                    "s": s,
                    "t": t,
                    "stid": stid,
                    "chmap": chmap,
                    "chs_to_quant": cfg["channels_to_quant"],
                    "roi_dir": roi_dir,
                    "timelapse": timelapse,
                    "skip_no_roi": True,

                    "bg_scope": cfg["bg_scope"],
                    "bg_mode": cfg["bg_mode"],
                    "percentile": cfg["percentile"],
                    "per_channel_p": cfg["per_channel_p"],
                    "ch_p_map": cfg["ch_p_map"],
                    "clip_neg": cfg["clip_neg"],
                    "bg_stride": cfg["bg_stride"],

                    "px_um": cfg["px_um"],
                    "do_tif": out_tif,
                    "do_png": out_png,
                    "tif_mask_outside": cfg["tif_mask_outside"],
                    "tif32_dir": tif32_dir,
                    "tif16_dir": tif16_dir,
                    "png_root": png_root,

                    "full_on": cfg["full_on"],
                    "full_mask_outside": cfg["full_mask_outside"],
                    "full_sb_on": cfg["full_sb_on"],
                    "full_sb_um": cfg["full_sb_um"],
                    "full_sb_anchor": cfg["full_sb_anchor"],
                    "full_sb_font": cfg["full_sb_font"],
                    "full_cmap_on": cfg["full_cmap_on"],
                    "full_cmap": cfg["full_cmap"],
                    "full_cmin": cfg["full_cmin"],
                    "full_cmax": cfg["full_cmax"],
                    "full_cbar_on": cfg["full_cbar_on"],
                    "full_png_dpi": cfg["full_png_dpi"],

                    "crop_on": cfg["crop_on"],
                    "crop_mask_outside": cfg["crop_mask_outside"],
                    "crop_sb_on": cfg["crop_sb_on"],
                    "crop_sb_um": cfg["crop_sb_um"],
                    "crop_sb_anchor": cfg["crop_sb_anchor"],
                    "crop_sb_font": cfg["crop_sb_font"],
                    "crop_cmap_on": cfg["crop_cmap_on"],
                    "crop_cmap": cfg["crop_cmap"],
                    "crop_cmin": cfg["crop_cmin"],
                    "crop_cmax": cfg["crop_cmax"],
                    "crop_cbar_on": cfg["crop_cbar_on"],
                    "png_dpi": cfg["png_dpi"],
                    "fixed_crop": cfg["fixed_crop"],
                    "crop_w": cfg["crop_w"],
                    "crop_h": cfg["crop_h"],

                    "subset_on": cfg["subset_on"],
                    "subset_roi": cfg["subset_roi"],
                    "auto_lo": cfg["auto_clip_lo"],
                    "auto_hi": cfg["auto_clip_hi"],
                    "ch_color_map": cfg["ch_color_map"],
                    "save_raw_crop_tif": cfg["save_raw_crop_tif"],
                }
                tasks.append(task)

            rows_all = []
            proc_parallel = cfg["proc_parallel"]
            workers = cfg["proc_workers"]
            if workers == 0:
                workers = min(os.cpu_count() or 2, 8)

            if proc_parallel:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    fut2task = {ex.submit(_process_key_task, task): task for task in tasks}
                    for fu in as_completed(fut2task):
                        if self.cancel_requested:
                            break
                        res = fu.result()
                        rows_all.extend(res["rows"])
                        steps = int(res.get("steps", 1))
                        self._progress_step(steps, msg=f"{fut2task[fu]['stid']} 완료")
                        for line in res.get("logs", []):
                            print(line)
            else:
                for task in tasks:
                    if self.cancel_requested:
                        break
                    res = _process_key_task(task)
                    rows_all.extend(res["rows"])
                    steps = int(res.get("steps", 1))
                    self._progress_step(steps, msg=f"{task['stid']} 진행")
                    for line in res.get("logs", []):
                        print(line)

            if self.cancel_requested:
                print("[알림] 사용자가 작업을 중단했습니다.")
            else:
                if cfg["out_xls"] and rows_all:
                    try:
                        save_excel(rows_all, keymap, xls_dir)
                    except Exception as e:
                        print("[주의] XLS 저장 중 오류:", e)
                elif cfg["out_xls"] and not rows_all:
                    print("[주의] ROI가 없어 정량 테이블이 생성되지 않았습니다.")
                print("[완료] 모든 처리가 종료되었습니다.")

        except Exception as e:
            print("[치명적 오류]", e)
            traceback.print_exc()
        finally:
            self.root.after(0, self._run_finished)

    # ---------- Progress UI ----------

    def _set_progress_total(self, total):
        self.progress_total = max(1, int(total))
        self.progress_done = 0
        self.progress_var.set(0.0)
        self._update_eta()

    def _progress_step(self, inc=1, msg=None):
        def _u():
            if not self.running:
                return
            self.progress_done += int(inc)
            if self.progress_done > self.progress_total:
                self.progress_done = self.progress_total
            self.progress_var.set(
                self.progress_done / float(self.progress_total)
            )
            if msg:
                self.lab_status.config(text=msg)
            self._update_eta()
        self.root.after(0, _u)

    def _update_eta(self):
        if not self.start_ts:
            self.lab_eta.config(
                text="경과: 00:00:00 | 남은: --:--:--"
            )
            return
        done = self.progress_done
        total = self.progress_total
        now = time.time()
        elapsed = int(now - self.start_ts)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        if done > 0:
            rate = elapsed / done
            rem = int(rate * (total - done))
            rh = rem // 3600
            rm = (rem % 3600) // 60
            rs = rem % 60
            self.lab_eta.config(
                text=f"경과: {h:02d}:{m:02d}:{s:02d} | 남은: {rh:02d}:{rm:02d}:{rs:02d}"
            )
        else:
            self.lab_eta.config(
                text=f"경과: {h:02d}:{m:02d}:{s:02d} | 남은: --:--:--"
            )

    def _run_finished(self):
        self.running = False
        self.btn_cancel.configure(state=tk.DISABLED)
        self._set_all_inputs_state(tk.NORMAL)
        self._toggle_png_group()
        self._toggle_subset()
        self._toggle_full_box()
        self._toggle_crop_box()
        if self.gui_logger and self.gui_logger.log_path:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.gui_logger.log_path, "a", encoding="utf-8") as f:
                f.write(f"[END] {ts}\n")

# ===================== 엔트리 포인트 =====================

if __name__ == "__main__":
    FluorIntensityApp()

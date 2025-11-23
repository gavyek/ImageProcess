#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-based Channel/Phase Cropping — v2.8 (Padding Mode GUI + Inline Log)
------------------------------------------------------------------------
- Timelapse ON/OFF 지원
- 채널 자동 인식(_chX/_cX 우선, 없으면 마지막 숫자 토큰; timelapse면 tXX 제외)
- ROI(JSON)의 polygon을 감싸는 bbox + padding
- fixed_crop일 때: ROI 중심 기준 W×H 창 고정 + 프레임 밖이면 '패딩 모드'로 자연스럽게 채움
  (reflect / edge / symmetric / constant(값 지정))
- ROI 바깥 masking 옵션
- PNG / TIFF16(정규화) / TIFF 원본값 크롭 저장
- 색상(단색) 입히기, 감마, 상/하위 컷 퍼센트
- 포맷별 하위 폴더 저장
- 하단 Log 창(실시간) + 실행 중 버튼 잠금
"""

import os
import re
import glob
import json
import sys
import threading

import numpy as np
from tifffile import imread, imwrite

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# ----------------------- 파일 유틸 -----------------------

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

# ----------------------- 파서 -----------------------

CH_TOKEN = re.compile(r'(?i)(?:^|[_-])(ch|c)(\d{1,3})(?=$|[_-])')

def parse_stage_time(basename: str, timelapse: bool):
    """Timelapse OFF: S01_X,  ON: S01_tXX_X"""
    name = os.path.splitext(basename)[0]
    ms = re.search(r'(?i)(?:^|[_-])S(\d{1,3})(?=$|[_-])', name)
    s = f"S{int(ms.group(1)):02d}" if ms else None
    t = None
    if timelapse:
        mt = re.search(r'(?i)(?:^|[_-])t(\d{1,3})(?=$|[_-])', name)
        if mt:
            t = f"t{int(mt.group(1)):02d}"
    return s, t

def detect_channel(basename: str, timelapse: bool):
    """
    우선순위:
      1) _chX / _cX
      2) 마지막 숫자 토큰 (timelapse면 tXX 숫자는 제외)
    """
    name = os.path.splitext(basename)[0]
    m = CH_TOKEN.search(name)
    if m:
        try:
            return int(m.group(2))
        except:
            return None

    tokens = re.split(r'[_-]', name)
    nums = [tok for tok in tokens if tok.isdigit()]

    if timelapse:
        mt = re.search(r'(?i)(?:^|[_-])t(\d{1,3})(?=$|[_-])', name)
        if mt:
            # 시간 토큰 그대로 사용 (선행 0 유지)
            t_str = mt.group(1)
            nums = [n for n in nums if n != t_str]

    if nums:
        try:
            return int(nums[-1])
        except:
            return None
    return None

def build_map(files, ch_select: int, timelapse: bool):
    """반환: {(Sxx, txx or None): path}"""
    out = {}
    for p in files:
        base = os.path.basename(p)
        s, t = parse_stage_time(base, timelapse=timelapse)
        if not s:
            continue
        ch = detect_channel(base, timelapse=timelapse)
        if ch is None:
            continue
        if ch == ch_select:
            out[(s, t)] = p
    return out

# ----------------------- ROI -----------------------

def load_roi_polys(roi_dir, s, t, timelapse):
    """ROI JSON: Sxx.json / Sxx_txx.json"""
    base = f"{s}_{t}" if (timelapse and t is not None) else s
    j = os.path.join(roi_dir, f"{base}.json")
    if not os.path.exists(j):
        return None
    with open(j, "r", encoding="utf-8") as f:
        data = json.load(f)
    polys = []
    for poly in data.get("rois", []):
        P = np.asarray(poly, dtype=float)
        if P.shape[0] >= 3:
            polys.append(P)
    return polys if polys else None

def rasterize_polygon(poly, shape):
    H, W = shape
    path = mpath.Path(np.asarray(poly, dtype=float))
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.vstack((xx.ravel(), yy.ravel())).T
    return path.contains_points(pts).reshape(H, W)

# ----------------------- 그리기 -----------------------

COLOR_MAP = {
    "Grayscale": None,
    "Cyan":     (0.0, 1.0, 1.0),
    "Blue":     (0.0, 0.0, 1.0),
    "Red":      (1.0, 0.0, 0.0),
    "Yellow":   (1.0, 1.0, 0.0),
    "Green":    (0.0, 1.0, 0.0),
    "Magenta":  (1.0, 0.0, 1.0),
    "Orange":   (1.0, 0.5, 0.0),
}

def draw_scalebar(ax, img_w, img_h, bar_px, bar_um, frac_margin=0.05, lw=3):
    x_start = int(img_w * (1.0 - frac_margin) - bar_px)
    y = int(img_h * (1.0 - frac_margin))
    x_end = x_start + bar_px
    ax.plot([x_start, x_end], [y, y], color='w', linewidth=lw)
    ax.text(
        (x_start + x_end) / 2,
        y - max(10, int(0.02 * img_h)),
        f"{bar_um:.0f} µm",
        color='w',
        ha='center',
        va='bottom',
        fontsize=9,
        bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none')
    )

def save_png_image(img, out_path, dpi=300, out_px=None, scalebar_um=None, px_um=None):
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    if img.ndim == 2:
        ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.clip(img, 0, 1))

    ax.set_axis_off()
    H, W = img.shape[:2]

    if (scalebar_um is not None) and (px_um is not None):
        bar_px = int(round(float(scalebar_um) / float(px_um)))
        bar_px = max(2, min(bar_px, int(0.8 * W)))
        draw_scalebar(ax, W, H, bar_px, bar_px * float(px_um))

    fig.tight_layout(pad=0)

    if out_px:
        fig.set_size_inches(out_px[0] / dpi, out_px[1] / dpi)

    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ----------------------- Log Redirector -----------------------

class TextRedirector:
    def __init__(self, text_widget, orig_stream=None):
        self.text_widget = text_widget
        self.orig = orig_stream

    def write(self, msg):
        if self.orig:
            try:
                self.orig.write(msg)
            except Exception:
                pass

        if not msg:
            return

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

# ----------------------- 실행 로직 -----------------------

def process(cfg):
    raw_dir = cfg["raw_dir"]
    roi_dir = cfg["roi_dir"]
    out_dir = cfg["out_dir"]
    ch_select = int(cfg["channel"])
    timelapse = bool(cfg["timelapse"])

    add_scalebar = bool(cfg["add_scalebar"])
    px_um = float(cfg["px_um"]) if (cfg["px_um"] not in (None, "",)) else None
    sb_len_um = float(cfg["sb_len_um"])

    png_dpi = int(cfg["png_dpi"])
    fixed_crop = bool(cfg["fixed_crop"])
    crop_w = int(cfg["crop_w"]) if cfg["crop_w"] else None
    crop_h = int(cfg["crop_h"]) if cfg["crop_h"] else None

    subset_on = bool(cfg["subset_on"])
    subset_stage = cfg["subset_stage"]
    subset_time = cfg["subset_time"]
    subset_roi = cfg["subset_roi"]  # int or None

    color_name = cfg["color"]
    gamma = float(cfg["gamma"])
    low_cut = float(cfg["low_cut"])
    high_cut = float(cfg["high_cut"])

    mask_outside = bool(cfg["mask_outside"])
    save_png = bool(cfg["save_png"])
    save_tiff16 = bool(cfg["save_tiff16"])
    save_tiff_raw = bool(cfg["save_tiff_raw"])

    pad_ratio = 0.02  # bbox 패딩 비율
    pad_mode = cfg.get("pad_mode", "reflect")
    pad_const = float(cfg.get("pad_const", 0.0))

    os.makedirs(out_dir, exist_ok=True)
    png_dir = os.path.join(out_dir, "PNG")
    t16_dir = os.path.join(out_dir, "TIFF16")
    trw_dir = os.path.join(out_dir, "TIFF")
    for d in (png_dir, t16_dir, trw_dir):
        os.makedirs(d, exist_ok=True)

    files = list_tifs(raw_dir)
    fmap = build_map(files, ch_select, timelapse=timelapse)

    # Subset 필터링
    items = list(fmap.items())
    if subset_on:
        items = [kv for kv in items if kv[0][0] == subset_stage and (not timelapse or kv[0][1] == subset_time)]

    items.sort(key=lambda kv: natural_key(f"{kv[0][0]}_{kv[0][1] or ''}"))

    def apply_tone(img_f32):
        # low/high cut (%)
        lc = np.clip(low_cut, 0, 49.9)
        hc = np.clip(high_cut, 0, 49.9)
        if lc > 0 or hc > 0:
            lo = np.percentile(img_f32, lc)
            hi = np.percentile(img_f32, 100 - hc)
            if hi <= lo:
                hi = lo + 1e-6
            img_f32 = (img_f32 - lo) / (hi - lo)
            img_f32 = np.clip(img_f32, 0, 1)

        # gamma (출력 = 입력 ** (1/γ))
        g = max(1e-4, min(10.0, gamma))
        img_f32 = np.power(img_f32, 1.0 / g)
        return img_f32

    def colorize(gray01):
        if COLOR_MAP[color_name] is None:
            return gray01
        r, g, b = COLOR_MAP[color_name]
        out = np.dstack([gray01 * r, gray01 * g, gray01 * b])
        return out

    print(f"[INFO] 대상 파일 수: {len(items)}  (ch={ch_select}, timelapse={timelapse})")
    for (s, t), path in items:
        print(f"\n[PROCESS] {os.path.basename(path)}  stage={s} time={t}")
        img = imread(path)
        if img.ndim > 2:
            # 다채널 TIF이면 채널 차원 제거(최대값 프로젝션 등은 외부에서)
            img = img.squeeze()
        H, W = img.shape[:2]

        polys = load_roi_polys(roi_dir, s, t, timelapse)
        if not polys:
            print("  - ROI 없음 → 스킵")
            continue

        # 정규화용 float (복사본)
        raw_full = img.astype(np.float32)
        # 16-bit 정규화 기준
        vmax = raw_full.max()
        if vmax <= 0:
            vmax = 1.0
        img01_full = np.clip(raw_full / vmax, 0, 1)

        # ROI 마스크 (여러 polygon이면 union)
        mask = np.zeros((H, W), dtype=bool)
        for P in polys:
            mask |= rasterize_polygon(P, (H, W))

        # bbox + padding
        ys, xs = np.where(mask)
        if ys.size == 0:
            print("  - ROI mask 빈 값 → 스킵")
            continue

        miny, maxy = ys.min(), ys.max()
        minx, maxx = xs.min(), xs.max()

        extra = max(10, int(pad_ratio * max(H, W)))
        x0 = max(int(np.floor(minx)) - extra, 0)
        x1 = min(int(np.ceil (maxx)) + extra, W - 1)
        y0 = max(int(np.floor(miny)) - extra, 0)
        y1 = min(int(np.ceil (maxy)) + extra, H - 1)

        # --- fixed crop 처리 ---
        if fixed_crop and crop_w and crop_h:
            target_w = int(crop_w)
            target_h = int(crop_h)

            # ROI 중심
            cx = int(round((x0 + x1) / 2))
            cy = int(round((y0 + y1) / 2))
            half_w = target_w // 2
            half_h = target_h // 2

            # 부족량 계산
            left_need   = max(0, half_w - cx)
            right_need  = max(0, (cx + half_w) - (W - 1))
            top_need    = max(0, half_h - cy)
            bottom_need = max(0, (cy + half_h) - (H - 1))

            pad_cfg = ((top_need, bottom_need), (left_need, right_need))

            if pad_mode == "constant":
                img_pad = np.pad(img01_full, pad_cfg, mode="constant", constant_values=pad_const)
                raw_pad = np.pad(raw_full,   pad_cfg, mode="constant", constant_values=pad_const*vmax)
            else:
                img_pad = np.pad(img01_full, pad_cfg, mode=pad_mode)
                raw_pad = np.pad(raw_full,   pad_cfg, mode=pad_mode)

            # 좌표 이동
            cx += left_need
            cy += top_need
            H2, W2 = img_pad.shape[:2]

            x0 = max(0, cx - half_w)
            x1 = min(W2 - 1, x0 + target_w - 1)
            y0 = max(0, cy - half_h)
            y1 = min(H2 - 1, y0 + target_h - 1)

            img01 = img_pad[y0:y1+1, x0:x1+1].copy()
            raw_c = raw_pad[y0:y1+1, x0:x1+1].copy()
            # 패딩된 곳의 mask도 필요하면 사용할 수 있도록 재구성
            mask_pad = np.pad(mask.astype(np.uint8), pad_cfg, mode="constant", constant_values=0)
            mask_c = mask_pad[y0:y1+1, x0:x1+1] > 0
        else:
            img01 = img01_full[y0:y1+1, x0:x1+1].copy()
            raw_c = raw_full  [y0:y1+1, x0:x1+1].copy()
            mask_c = mask     [y0:y1+1, x0:x1+1].copy()

        # ROI 바깥 masking
        if mask_outside:
            img01 = np.where(mask_c, img01, 0.0)
            raw_c = np.where(mask_c, raw_c, 0.0)

        # 톤 & 색상
        img01 = apply_tone(img01)
        rgb = colorize(img01)

        # 파일명
        base = f"{s}_{t}" if (timelapse and t is not None) else s
        base = f"{base}_ch{ch_select}"
        out_base = base

        # 저장
        if save_png:
            Hc, Wc = img01.shape[:2]
            out_px = None
            if fixed_crop and crop_w and crop_h:
                out_px = (int(crop_w), int(crop_h))
            png_path = os.path.join(png_dir, f"{out_base}.png")
            save_png_image(
                rgb if rgb.ndim == 3 else img01,
                png_path,
                dpi=png_dpi,
                out_px=out_px,
                scalebar_um=(sb_len_um if add_scalebar else None),
                px_um=px_um
            )
            print(f"  - PNG 저장: {png_path}")

        if save_tiff16:
            t16 = np.clip(img01 * 65535.0, 0, 65535).astype(np.uint16)
            t16_path = os.path.join(t16_dir, f"{out_base}.tif")
            imwrite(t16_path, t16)
            print(f"  - TIFF16 저장: {t16_path}")

        if save_tiff_raw:
            trw_path = os.path.join(trw_dir, f"{out_base}.tif")
            imwrite(trw_path, raw_c.astype(img.dtype))
            print(f"  - TIFF(raw) 저장: {trw_path}")

    print("\n[DONE] 모든 작업이 완료되었습니다.")

# ----------------------- GUI -----------------------

def gui():
    cfg = {
        "raw_dir": None,
        "roi_dir": None,
        "out_dir": None,
        "channel": 1,
        "timelapse": False,
        "add_scalebar": False,
        "px_um": None,
        "sb_len_um": 20.0,
        "png_dpi": 300,
        "fixed_crop": True,
        "crop_w": 500,
        "crop_h": 500,
        "subset_on": False,
        "subset_stage": "",
        "subset_time": "",
        "subset_roi": "",
        "color": "Grayscale",
        "gamma": 1.0,
        "low_cut": 1.0,
        "high_cut": 1.0,
        "mask_outside": True,
        "save_png": True,
        "save_tiff16": True,
        "save_tiff_raw": True,
        "pad_mode": "reflect",
        "pad_const": 0.0,
    }

    root = tk.Tk()
    root.title("ROI Channel/Phase Cropper (v2.8 • Padding mode + Log)")
    root.resizable(False, False)

    # --- Tk 변수 ---
    raw_v = tk.StringVar()
    roi_v = tk.StringVar()
    out_v = tk.StringVar()

    ch_v = tk.StringVar(value="1")
    tl_v = tk.BooleanVar(value=False)

    scb_v = tk.BooleanVar(value=False)
    px_v = tk.StringVar(value="")
    sbl_v = tk.StringVar(value="20.0")
    dpi_v = tk.StringVar(value="300")

    fxc_v = tk.BooleanVar(value=True)
    cw_v = tk.StringVar(value="500")
    chh_v = tk.StringVar(value="500")

    subset_on = tk.BooleanVar(value=False)
    stage_v = tk.StringVar(value="")
    time_v = tk.StringVar(value="")
    roi_vv = tk.StringVar(value="")

    color_v = tk.StringVar(value="Grayscale")
    gamma_v = tk.StringVar(value="1.0")
    lowcut_v = tk.StringVar(value="1.0")
    highcut_v = tk.StringVar(value="1.0")

    mask_v = tk.BooleanVar(value=True)
    save_png = tk.BooleanVar(value=True)
    save_tiff16 = tk.BooleanVar(value=True)
    save_tiff_raw = tk.BooleanVar(value=True)

    pad_mode_v = tk.StringVar(value="reflect")
    pad_const_v = tk.StringVar(value="0.0")

    pad = {'padx': 8, 'pady': 6}

    # --- 콜백 ---
    def browse_raw():
        p = filedialog.askdirectory(title="Select Raw TIF Folder")
        if p: raw_v.set(p)

    def browse_roi():
        p = filedialog.askdirectory(title="Select ROI Folder (JSON)")
        if p: roi_v.set(p)

    def browse_out():
        p = filedialog.askdirectory(title="Select Output Folder")
        if p: out_v.set(p)

    def toggle_scalebar(*_):
        st = tk.NORMAL if scb_v.get() else tk.DISABLED
        e_px.config(state=st)
        e_sbl.config(state=st)

    def toggle_fixed(*_):
        st = tk.NORMAL if fxc_v.get() else tk.DISABLED
        e_cw.config(state=st)
        e_chh.config(state=st)

    def toggle_subset(*_):
        if subset_on.get():
            e_stage.config(state=tk.NORMAL)
            e_roi.config(state=tk.NORMAL)
            if tl_v.get():
                e_time.config(state=tk.NORMAL)
            else:
                e_time.config(state=tk.DISABLED)
        else:
            e_stage.config(state=tk.DISABLED)
            e_time.config(state=tk.DISABLED)
            e_roi.config(state=tk.DISABLED)

    def on_run():
        if not raw_v.get() or not roi_v.get() or not out_v.get():
            messagebox.showwarning("Missing", "Raw/ROI/Output folder are required.")
            return
        cfg["raw_dir"] = raw_v.get()
        cfg["roi_dir"] = roi_v.get()
        cfg["out_dir"] = out_v.get()

        cfg["channel"] = int(ch_v.get())
        cfg["timelapse"] = bool(tl_v.get())
        cfg["add_scalebar"] = bool(scb_v.get())
        cfg["px_um"] = px_v.get().strip() if scb_v.get() else None
        cfg["sb_len_um"] = float(sbl_v.get())
        cfg["png_dpi"] = int(dpi_v.get())

        cfg["fixed_crop"] = bool(fxc_v.get())
        cfg["crop_w"] = int(cw_v.get())
        cfg["crop_h"] = int(chh_v.get())

        cfg["subset_on"] = bool(subset_on.get())
        cfg["subset_stage"] = stage_v.get().strip()
        cfg["subset_time"] = time_v.get().strip()
        cfg["subset_roi"] = roi_vv.get().strip()

        cfg["color"] = color_v.get()
        cfg["gamma"] = float(gamma_v.get())
        cfg["low_cut"] = float(lowcut_v.get())
        cfg["high_cut"] = float(highcut_v.get())

        cfg["mask_outside"] = bool(mask_v.get())
        cfg["save_png"] = bool(save_png.get())
        cfg["save_tiff16"] = bool(save_tiff16.get())
        cfg["save_tiff_raw"] = bool(save_tiff_raw.get())

        cfg["pad_mode"] = pad_mode_v.get()
        try:
            cfg["pad_const"] = float(pad_const_v.get())
        except:
            cfg["pad_const"] = 0.0

        btn_run.config(state=tk.DISABLED)
        th = threading.Thread(target=lambda: (process(cfg), btn_run.config(state=tk.NORMAL)), daemon=True)
        th.start()

    # --- 레이아웃 ---
    tk.Label(root, text="Raw TIF Folder").grid(row=0, column=0, sticky="w", **pad)
    fr = tk.Frame(root); fr.grid(row=0, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=raw_v, width=52).pack(side="left")
    tk.Button(fr, text="Browse", width=8, command=browse_raw).pack(side="left", padx=6)

    tk.Label(root, text="ROI Folder").grid(row=1, column=0, sticky="w", **pad)
    fr = tk.Frame(root); fr.grid(row=1, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=roi_v, width=52).pack(side="left")
    tk.Button(fr, text="Browse", width=8, command=browse_roi).pack(side="left", padx=6)

    tk.Label(root, text="Output Folder").grid(row=2, column=0, sticky="w", **pad)
    fr = tk.Frame(root); fr.grid(row=2, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=out_v, width=52).pack(side="left")
    tk.Button(fr, text="Browse", width=8, command=browse_out).pack(side="left", padx=6)

    tk.Checkbutton(root, text="Timelapse filenames (SXX_tXX_X)", variable=tl_v, command=toggle_subset)\
        .grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 0))

    tk.Label(root, text="Channel to crop").grid(row=4, column=0, sticky="w", **pad)
    tk.Entry(root, textvariable=ch_v, width=8).grid(row=4, column=1, sticky="w", **pad)

    # 스케일바
    fr = tk.Frame(root); fr.grid(row=5, column=0, columnspan=2, sticky="w", **pad)
    tk.Checkbutton(fr, text="Add scalebar", variable=scb_v, command=toggle_scalebar).pack(side="left")
    tk.Label(fr, text="  pixel size (µm/px):").pack(side="left")
    e_px = tk.Entry(fr, textvariable=px_v, width=10, state=tk.DISABLED); e_px.pack(side="left")
    tk.Label(fr, text="  bar length (µm):").pack(side="left")
    e_sbl = tk.Entry(fr, textvariable=sbl_v, width=8, state=tk.DISABLED); e_sbl.pack(side="left")

    # PNG 옵션 + fixed crop
    fr = tk.Frame(root); fr.grid(row=6, column=0, columnspan=2, sticky="w", **pad)
    tk.Label(fr, text="PNG DPI:").pack(side="left")
    tk.Entry(fr, textvariable=dpi_v, width=6).pack(side="left", padx=(4, 12))
    tk.Checkbutton(fr, text="Fixed crop (pixels)", variable=fxc_v, command=toggle_fixed).pack(side="left")
    tk.Label(fr, text="  W×H(px): ").pack(side="left")
    e_cw = tk.Entry(fr, textvariable=cw_v, width=6); e_cw.pack(side="left")
    tk.Label(fr, text=" × ").pack(side="left")
    e_chh = tk.Entry(fr, textvariable=chh_v, width=6); e_chh.pack(side="left")

    # 부분 추출
    sub = tk.LabelFrame(root, text="Subset")
    sub.grid(row=7, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))
    tk.Checkbutton(sub, text="Enable subset", variable=subset_on, command=toggle_subset)\
        .grid(row=0, column=0, sticky="w", padx=8, pady=6)
    tk.Label(sub, text="Stage").grid(row=1, column=0, sticky="e", padx=6)
    e_stage = tk.Entry(sub, textvariable=stage_v, width=8, state=tk.DISABLED); e_stage.grid(row=1, column=1, sticky="w")
    tk.Label(sub, text="Time").grid(row=1, column=2, sticky="e", padx=6)
    e_time = tk.Entry(sub, textvariable=time_v, width=8, state=tk.DISABLED); e_time.grid(row=1, column=3, sticky="w")
    tk.Label(sub, text="ROI (unused)").grid(row=1, column=4, sticky="e", padx=6)
    e_roi = tk.Entry(sub, textvariable=roi_vv, width=8, state=tk.DISABLED); e_roi.grid(row=1, column=5, sticky="w")

    # 색상 & 톤
    newf = tk.LabelFrame(root, text="Color & Tone")
    newf.grid(row=8, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))
    tk.Label(newf, text="Color:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    tk.OptionMenu(newf, color_v, *list(COLOR_MAP.keys())).grid(row=0, column=1, sticky="w")
    tk.Label(newf, text="Gamma γ (0.0001~10)").grid(row=0, column=2, sticky="e", padx=6)
    tk.Entry(newf, textvariable=gamma_v, width=10).grid(row=0, column=3, sticky="w")
    tk.Label(newf, text="Lower cut % (0~49.9)").grid(row=1, column=0, sticky="e", padx=6)
    tk.Entry(newf, textvariable=lowcut_v, width=10).grid(row=1, column=1, sticky="w")
    tk.Label(newf, text="Upper cut % (0~49.9)").grid(row=1, column=2, sticky="e", padx=6)
    tk.Entry(newf, textvariable=highcut_v, width=10).grid(row=1, column=3, sticky="w")

    # 패딩 모드
    pf = tk.LabelFrame(root, text="Padding Mode (for fixed crop near borders)")
    pf.grid(row=9, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))
    tk.Label(pf, text="Mode:").grid(row=0, column=0, sticky="e", padx=6)
    tk.OptionMenu(pf, pad_mode_v, "reflect", "edge", "symmetric", "constant").grid(row=0, column=1, sticky="w")
    tk.Label(pf, text="Constant value (0.0~1.0 for PNG/TIFF16)").grid(row=0, column=2, sticky="e", padx=6)
    tk.Entry(pf, textvariable=pad_const_v, width=8).grid(row=0, column=3, sticky="w")

    # 마스킹 & 저장 포맷
    more = tk.LabelFrame(root, text="Masking / Save formats")
    more.grid(row=10, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))
    tk.Checkbutton(more, text="Mask outside ROI", variable=mask_v).grid(row=0, column=0, sticky="w", padx=8)
    tk.Checkbutton(more, text="Save PNG", variable=save_png).grid(row=0, column=1, sticky="w", padx=8)
    tk.Checkbutton(more, text="Save TIFF16 (normalized)", variable=save_tiff16).grid(row=0, column=2, sticky="w", padx=8)
    tk.Checkbutton(more, text="Save TIFF (raw values)", variable=save_tiff_raw).grid(row=0, column=3, sticky="w", padx=8)

    # 실행 버튼
    btn_run = tk.Button(root, text="Run", width=16, command=on_run)
    btn_run.grid(row=11, column=0, columnspan=2, pady=(6, 6))
    # 로그창
    log = scrolledtext.ScrolledText(root, width=88, height=16); log.grid(row=12, column=0, columnspan=2, padx=8, pady=(0,8))
    sys.stdout = TextRedirector(log, orig_stream=sys.stdout)
    sys.stderr = TextRedirector(log, orig_stream=sys.stderr)

    toggle_scalebar(); toggle_fixed(); toggle_subset()

    root.mainloop()

if __name__ == "__main__":
    gui()

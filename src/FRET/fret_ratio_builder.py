#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRET Ratio Builder (v1.2-stage-parallel, inline-LOG GUI)
v1.2 patch (25.11.22)
--영어모드 추가 : python src/FRET/fret_ratio_builder.py -mode EN 실행시
--channel 선택 여부 가능

"""

import os, re, glob, json, time, traceback, threading, queue, sys
import numpy as np
import pandas as pd

from tifffile import imread, imwrite
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# -------------------- Language resources --------------------
LANG_DEFAULT = "ko"
LANG_CURRENT = LANG_DEFAULT

STRINGS = {
    "ko": {
        "title_app": "FRET Ratio Builder (v1.1)",
        "lbl_img": "FRET images 경로",
        "lbl_roi": "ROI data 경로",
        "lbl_out": "출력 루트(선택)",
        "btn_browse": "찾기",
        "cb_timelapse": "Timelapse(시간 포함: SXX_TXX_X)",
        "lbl_ratio": "Ratio 방식",
        "ratio_fd": "FRET / Donor",
        "ratio_df": "Donor / FRET",
        "donor_ch": "Donor ch",
        "acceptor_ch": "Acceptor ch",
        "bg_group": "BG 옵션",
        "bg_scope": "Scope",
        "bg_mode": "Mode",
        "bg_percentile": "percentile p(%)",
        "per_channel_p": "채널별 p",
        "clip_neg": "음수 clip",
        "eps_pct": "ε percentile(denom)%",
        "px_um": "픽셀크기 (µm/px) — scalebar용",
        "out_group": "출력 그룹",
        "out_xls": "XLS",
        "out_tif": "TIF (RAT/RAT_ROI_masked)",
        "out_png": "PNG",
        "png_opts": "PNG 옵션",
        "png_full": "Full 이미지 저장(grayscale)",
        "png_crop": "Crop 이미지 저장(ROI별)",
        "mask_out": "Crop: ROI 바깥 mask",
        "scalebar": "Crop: 스케일바",
        "sb_len": " 길이(µm): ",
        "cmap": "Crop: 컬러맵",
        "cmap_label": "Colormap:",
        "show_cbar": "컬러바 표시",
        "cmin": "최소값",
        "cmax": "  최대값",
        "png_dpi": "PNG DPI:",
        "fixed_crop": "고정 crop 영역 크기 사용",
        "crop_wh": "  W×H(px): ",
        "subset_group": "부분추출 / 병렬",
        "subset_on": "부분추출 활성화",
        "subset_stage": "Stage(정수)",
        "subset_time": "Time(선택; timelapse)",
        "subset_roi": "ROI(선택)",
        "n_workers": "프로세스 수",
        "btn_run": "실행",
        "btn_quit": "종료",
        "log_label": "Log",
        "err_title": "입력 오류",
        "err_invalid_img": "유효한 FRET images 경로를 선택하세요.",
        "err_invalid_roi": "유효한 ROI 경로를 선택하세요.",
        "err_percentile": "percentile/ε는 0~10 범위로 입력하세요.",
        "err_per_channel": "채널별 p는 0~10 범위",
        "err_px": "픽셀크기(µm/px) 입력 오류",
        "err_dpi": "PNG DPI는 50~1200",
        "err_crop": "crop 픽셀 크기(32~8000)",
        "err_stage": "Stage는 정수",
        "err_time": "Timepoint는 정수",
        "err_roi": "ROI는 1 이상 정수",
        "err_workers": "병렬 프로세스 수는 1 ~ CPU 코어 이내 정수",
        "err_channels": "Donor/Acceptor 채널 번호는 1~32 사이 서로 다르게 입력하세요",
        "msg_run_start": "\n================= 실행 시작 =================",
        "msg_run_end": "================= 실행 종료 =================",
        "msg_match_none": "매칭되는 (donor=_{donor}, acceptor=_{acceptor}) 파일이 없습니다.",
        "msg_subset_none": "[부분추출] 조건(Stage={stage})에 맞는 파일이 없습니다.",
        "msg_info_stage": "[정보] 총 Stage={count} / 병렬 프로세스={workers}",
        "msg_stage_error": "[오류] Stage {stage} 처리 중 예외:",
        "msg_progress": "[진행] {done}/{total} Stage 완료: {stage}",
        "msg_save_xlsx": "[저장] xls/{name}",
        "msg_save_csv": "[저장] xls/{name}",
        "msg_no_roi_table": "[주의] ROI가 없어 정량 테이블이 생성되지 않았습니다.",
        "msg_xls_off": "[정보] XLS 출력 비활성화",
        "msg_done": "[완료] 병렬(Stage 단위) 처리 종료 — 경과 {mm:02d}:{ss:02d}",
        "stage_start": "[Stage {stage}] 시작",
        "stage_item": "  - 처리: {id}",
        "stage_no_roi": "    [경고] ROI 미발견: {stid}.json → ROI 기반 출력 스킵",
        "stage_end": "[Stage {stage}] 종료 (총 {count} time/파일)",
        "browse_img": "FRET images 경로 선택",
        "browse_roi": "ROI data 경로 선택",
        "browse_out": "출력 루트(RES 상위) 선택",
    },
    "en": {
        "title_app": "FRET Ratio Builder (v13-stage-parallel + Log)",
        "lbl_img": "FRET images folder",
        "lbl_roi": "ROI data folder",
        "lbl_out": "Output root (optional)",
        "btn_browse": "Browse",
        "cb_timelapse": "Timelapse (filename: SXX_TXX_X)",
        "lbl_ratio": "Ratio mode",
        "ratio_fd": "FRET / Donor",
        "ratio_df": "Donor / FRET",
        "donor_ch": "Donor ch",
        "acceptor_ch": "Acceptor ch",
        "bg_group": "BG options",
        "bg_scope": "Scope",
        "bg_mode": "Mode",
        "bg_percentile": "percentile p(%)",
        "per_channel_p": "per-channel p",
        "clip_neg": "clip negatives",
        "eps_pct": "ε percentile(denom)%",
        "px_um": "Pixel size (µm/px) — for scalebar",
        "out_group": "Outputs",
        "out_xls": "XLS",
        "out_tif": "TIF (RAT/RAT_ROI_masked)",
        "out_png": "PNG",
        "png_opts": "PNG options",
        "png_full": "Full image grayscale only",
        "png_crop": "Crop image ROI-only",
        "mask_out": "Crop: mask outside ROI",
        "scalebar": "Crop: scalebar",
        "sb_len": " length(µm): ",
        "cmap": "Crop: colormap",
        "cmap_label": "Colormap:",
        "show_cbar": "Show colorbar",
        "cmin": "Min",
        "cmax": "  Max",
        "png_dpi": "PNG DPI:",
        "fixed_crop": "Use fixed crop size",
        "crop_wh": "  W×H(px): ",
        "subset_group": "Subset / Parallel",
        "subset_on": "Enable subset",
        "subset_stage": "Stage(int)",
        "subset_time": "Time(optional; timelapse)",
        "subset_roi": "ROI(optional)",
        "n_workers": "Processes",
        "btn_run": "Run",
        "btn_quit": "Quit",
        "log_label": "Log",
        "err_title": "Input error",
        "err_invalid_img": "Select a valid FRET images folder.",
        "err_invalid_roi": "Select a valid ROI folder.",
        "err_percentile": "percentile/ε must be 0~10.",
        "err_per_channel": "Per-channel p must be 0~10",
        "err_px": "Pixel size (µm/px) input error",
        "err_dpi": "PNG DPI must be 50~1200",
        "err_crop": "Crop size must be 32~8000",
        "err_stage": "Stage must be an integer",
        "err_time": "Timepoint must be an integer",
        "err_roi": "ROI must be integer >= 1",
        "err_workers": "Processes must be 1..CPU cores",
        "err_channels": "Donor/Acceptor channels must be 1~32 and different",
        "msg_run_start": "\n================= Run start =================",
        "msg_run_end": "================= Run end =================",
        "msg_match_none": "No matched files for donor=_{donor}, acceptor=_{acceptor}.",
        "msg_subset_none": "[Subset] No files match Stage={stage}.",
        "msg_info_stage": "[Info] Stages={count} / workers={workers}",
        "msg_stage_error": "[Error] Stage {stage} raised an exception:",
        "msg_progress": "[Progress] {done}/{total} stages done: {stage}",
        "msg_save_xlsx": "[Saved] xls/{name}",
        "msg_save_csv": "[Saved] xls/{name}",
        "msg_no_roi_table": "[Warn] No ROI → metric table not generated.",
        "msg_xls_off": "[Info] XLS output disabled",
        "msg_done": "[Done] Parallel (per stage) finished in {mm:02d}:{ss:02d}",
        "stage_start": "[Stage {stage}] start",
        "stage_item": "  - Processing: {id}",
        "stage_no_roi": "    [Warn] ROI missing: {stid}.json → skip ROI-based outputs",
        "stage_end": "[Stage {stage}] end (total {count} time/files)",
        "browse_img": "Select FRET images folder",
        "browse_roi": "Select ROI data folder",
        "browse_out": "Select output root (RES parent)",
    },
}

def t(key: str, default=None, lang=None) -> str:
    lng = (lang or LANG_CURRENT or LANG_DEFAULT)
    if lng not in STRINGS:
        lng = LANG_DEFAULT
    if key in STRINGS[lng]:
        return STRINGS[lng][key]
    if key in STRINGS.get(LANG_DEFAULT, {}):
        return STRINGS[LANG_DEFAULT][key]
    return default if default is not None else key

def pick_lang_from_argv(argv):
    lang = LANG_DEFAULT
    for i, a in enumerate(argv):
        al = str(a).lower()
        if al in ("-mode", "--mode", "-lang", "--lang") and i + 1 < len(argv):
            nxt = str(argv[i + 1]).lower()
            if nxt.startswith("en"):
                lang = "en"
        if al in ("en", "english", "-mode=en", "--mode=en", "-lang=en", "--lang=en"):
            lang = "en"
    return lang

# -------------------- 공통 유틸 --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_tifs(folder):
    exts = ("*.tif","*.tiff","*.TIF","*.TIFF")
    files = []
    for e in exts: files.extend(glob.glob(os.path.join(folder, e)))
    uniq = {}
    for p in files:
        norm = os.path.normcase(os.path.abspath(p))
        if norm not in uniq: uniq[norm] = p
    return sorted(uniq.values(), key=natural_key)

def fmt_stage(n):  # 'S01'
    return f"S{int(n):02d}"

def fmt_time(n):   # 't00'
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

def find_json_path(roi_dir, basename, timelapse):
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    if s_num is None:
        # 파일명에 Stage가 없으면 베이스 이름 기반 시도
        base = os.path.splitext(basename)[0]
        cand = os.path.join(roi_dir, base + ".json")
        return cand
    norm = fmt_stage(s_num) + (f"_{fmt_time(t_num)}" if (timelapse and t_num is not None) else "")
    cands = [
        os.path.join(roi_dir, norm + ".json"),
        os.path.join(roi_dir, f"S{int(s_num)}" + (f"_t{int(t_num)}" if (timelapse and t_num is not None) else "") + ".json"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]

def read_tiff_with_fallback(path, page=0):
    try:
        return imread(path, key=page)
    except Exception as e:
        msg = str(e)
        if ("imagecodecs" not in msg) and ("COMPRESSION" not in msg):
            raise
        with Image.open(path) as im:
            try: im.seek(page)
            except EOFError: im.seek(0)
            return np.array(im)

def read_2d(path):
    a = read_tiff_with_fallback(path)
    if a.ndim > 2: a = a[...,0] if a.ndim==3 else a[0,...]
    return a.astype(np.float32, copy=False)

def rasterize_polygon(poly, shape):
    H,W = shape
    path = mpath.Path(np.asarray(poly, dtype=float))
    yy,xx = np.mgrid[0:H,0:W]; pts = np.vstack((xx.ravel(),yy.ravel())).T
    return path.contains_points(pts).reshape(H,W)

def load_roi_polys(roi_folder, s, t, timelapse):
    base = f"{s}_{t}" if (timelapse and t is not None) else s
    json_path = find_json_path(roi_folder, base, timelapse)
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polys=[]
    for poly in data.get("rois", []):
        P = np.asarray(poly, dtype=float)
        if P.shape[0] >= 3: polys.append(P)
    return polys if polys else None

def _vals_in_scope(img2d, scope_mask):
    return img2d.ravel() if scope_mask is None else img2d[scope_mask]

def bg_value(img2d, mode="percentile", p=1.0, scope_mask=None):
    vals = _vals_in_scope(img2d, scope_mask)
    if vals.size == 0:
        return 0.0
    if mode == "percentile":
        return float(np.percentile(vals, p))
    elif mode == "hist-mode":
        hist, bins = np.histogram(vals, bins=2048)
        if hist.sum() <= 0:
            return float(np.percentile(vals, p))
        cdf = np.cumsum(hist).astype(float); cdf /= cdf[-1]
        target = float(p)/100.0
        idx = int(np.searchsorted(cdf, target, side="left"))
        thr = bins[-1] if idx >= len(bins)-1 else 0.5*(bins[idx]+bins[idx+1])
        return float(thr)
    else:
        return 0.0

def bg_correct(img2d, mode="percentile", p=1.0, scope_mask=None, clip_neg=True):
    B = bg_value(img2d, mode=mode, p=p, scope_mask=scope_mask)
    J = img2d - B
    if clip_neg: J[J<0] = 0.0
    return J, B

def pick_epsilon(denom_vals, eps_abs=5.0, p_floor=1.0):
    if denom_vals.size==0: return float(eps_abs)
    return float(max(eps_abs, np.percentile(denom_vals, p_floor)))

def quantify_per_roi(R, polys, extra_imgs=None):
    rows=[]; H,W=R.shape
    for i,poly in enumerate(polys,1):
        m = rasterize_polygon(poly,(H,W))
        vals = R[m]; vals = vals[np.isfinite(vals)]
        if vals.size==0:
            row={"roi":i,"area_px":int(m.sum()),"ratio_mean":np.nan,"ratio_median":np.nan,"ratio_std":np.nan,"ratio_p5":np.nan,"ratio_p95":np.nan}
        else:
            row={"roi":i,"area_px":int(m.sum()),
                 "ratio_mean":float(np.mean(vals)),
                 "ratio_median":float(np.median(vals)),
                 "ratio_std":float(np.std(vals)),
                 "ratio_p5":float(np.percentile(vals,5)),
                 "ratio_p95":float(np.percentile(vals,95))}
        if extra_imgs:
            for name,img in extra_imgs.items():
                iv = img[m].astype(np.float32)
                row[f"{name}_mean"]=float(np.mean(iv)) if iv.size else np.nan
                row[f"{name}_median"]=float(np.median(iv)) if iv.size else np.nan
        rows.append(row)
    return rows

def auto_minmax(vals, p_lo=1.0, p_hi=99.0):
    vals = vals[np.isfinite(vals)]
    if vals.size==0: return 0.0, 1.0
    lo = np.percentile(vals, p_lo); hi = np.percentile(vals, p_hi)
    if hi <= lo: hi = lo + 1e-6
    return float(lo), float(hi)

def save_png_gray(img2d, out_path, vmin=None, vmax=None, dpi=300, out_px=None):
    fig,ax=plt.subplots()
    ax.set_facecolor('black'); fig.patch.set_facecolor('black')
    ax.imshow(img2d, cmap='gray', vmin=vmin, vmax=vmax); ax.set_axis_off()
    fig.tight_layout(pad=0)
    if out_px:
        fig.set_size_inches(out_px[0]/dpi, out_px[1]/dpi)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

def draw_scalebar(ax, img_w, img_h, bar_px, bar_um, frac_margin=0.05, lw=3):
    x_start = int(img_w*(1.0 - frac_margin) - bar_px)
    y = int(img_h*(1.0 - frac_margin))
    x_end = x_start + bar_px
    ax.plot([x_start, x_end], [y, y], color='w', linewidth=lw)
    ax.text((x_start+x_end)/2, y - max(10, int(0.02*img_h)), f"{bar_um:.0f} µm",
            color='w', ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

def add_short_colorbar(fig, ax, vmin, vmax, cmap='jet', label="FRET ratio"):
    bbox = ax.get_position()
    ax_h = bbox.height; ax_y0 = bbox.y0; ax_x1 = bbox.x1
    pad = 0.01; cb_w = 0.02; cb_h = ax_h * (2.0/3.0)
    cb_y0 = ax_y0 + (ax_h - cb_h)/2.0; cb_x0 = ax_x1 + pad
    cb_ax = fig.add_axes([cb_x0, cb_y0, cb_w, cb_h]); cb_ax.set_facecolor('black')
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=plt.get_cmap(cmap), norm=norm, orientation='vertical')
    cb.set_label(label, rotation=90, color='w'); cb.ax.yaxis.label.set_color('w')
    cb.set_ticks([vmin, vmax]); cb.ax.set_yticklabels([f"{vmin:.2f}", f"{vmax:.2f}"], color='w')
    cb.ax.tick_params(color='w', labelcolor='w', length=3); cb.outline.set_edgecolor('w')

def save_png_colormap(img2d, out_path, vmin=None, vmax=None, cmap='jet',
                      mask=None, scalebar=None, px_um=None,
                      show_colorbar=True, dpi=300, out_px=None):
    fig,ax=plt.subplots()
    ax.set_facecolor('black'); fig.patch.set_facecolor('black')
    shown = np.array(img2d, copy=True)
    if mask is not None:
        shown = np.ma.array(shown, mask=~mask)
        cmap_obj = plt.get_cmap(cmap).copy(); cmap_obj.set_bad(alpha=0.0)
        ax.imshow(shown, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if (scalebar is not None) and (px_um is not None):
        H,W = shown.shape[:2]
        bar_px = int(round(float(scalebar) / float(px_um)))
        bar_px = max(2, min(bar_px, int(0.8*W)))
        draw_scalebar(ax, W, H, bar_px, bar_px*float(px_um))
    if show_colorbar and (vmin is not None) and (vmax is not None):
        add_short_colorbar(fig, ax, vmin, vmax, cmap=cmap, label="FRET ratio")
    fig.tight_layout(pad=0)
    if out_px:
        fig.set_size_inches(out_px[0]/dpi, out_px[1]/dpi)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

# -------------------- Stage 워커 --------------------
def process_one_stage(stage_key, pairs_for_stage, p, paths):
    logs = []
    logs.append(t("stage_start", "[Stage {stage}] start", lang=LANG_CURRENT).format(stage=stage_key))
    (RES_ROOT, RAT32, RAT16, RROI32, RROI16, PNG_FULL, PNG_CROP) = paths

    scope=p["bg_scope"]; bgmode=p["bg_mode"]
    per_ch=bool(p["per_channel_p"]); p_glob=float(p["percentile"])
    donor_p=float(p["donor_p"]); fret_p=float(p["fret_p"])
    clip_neg=bool(p["clip_neg"]); eps_p=float(p["eps_percentile"])
    ratio_mode=p["ratio_mode"]; timelapse=bool(p["timelapse"])
    out_tif=bool(p["out_tif"]); out_png=bool(p["out_png"])
    opt_full=bool(p["save_full"]); opt_crop=bool(p["save_crop"])
    mask_out=bool(p["mask_outside"])
    opt_cmap=bool(p["apply_cmap"]); cmap_name=p["cmap_name"]; show_cbar=bool(p["show_colorbar"])
    png_dpi=int(p["png_dpi"])
    opt_sb=bool(p["add_scalebar"]); sb_um=p["scale_bar_um"]
    px_um=p["px_um"]
    out_size_crop=(p["crop_w"], p["crop_h"]) if (out_png and opt_crop and p["fixed_crop"]) else None

    rows_stage=[]
    join_st = lambda s_num, t_code: f"{s_num}_{t_code}" if (timelapse and t_code is not None) else s_num

    for (s, t_code), dpath, apath in pairs_for_stage:
        stid = join_st(s, t_code)
        logs.append(t("stage_item", "  - Processing: {id}", lang=LANG_CURRENT).format(id=stid))
        D=read_2d(dpath); A=read_2d(apath)
        polys=load_roi_polys(p["roi_dir"], s, t_code, timelapse=timelapse)
        H,W=D.shape; union=None
        if polys:
            union=np.zeros((H,W),dtype=bool)
            for P in polys: union |= rasterize_polygon(P,(H,W))
        else:
            logs.append(t("stage_no_roi", "    [Warn] ROI missing: {stid}.json ? skip ROI-based outputs", lang=LANG_CURRENT).format(stid=stid))
        scope_mask = (union if (scope=="roi_union" and union is not None) else None)

        d_p=donor_p if per_ch else p_glob
        a_p=fret_p  if per_ch else p_glob
        Dbc,Db=bg_correct(D,mode=bgmode,p=d_p,scope_mask=scope_mask,clip_neg=clip_neg)
        Abc,Ab=bg_correct(A,mode=bgmode,p=a_p,scope_mask=scope_mask,clip_neg=clip_neg)

        if ratio_mode=="FRET/Donor": numer,denom=Abc,Dbc; suffix="FoverD"
        else: numer,denom=Dbc,Abc; suffix="DoverF"

        denom_vals = denom[scope_mask] if scope_mask is not None else denom.ravel()
        eps = pick_epsilon(denom_vals, eps_abs=5.0, p_floor=p["eps_percentile"])
        R_full = (numer + eps) / (denom + eps)

        # TIF ??
        if out_tif:
            imwrite(os.path.join(RAT32, f"{stid}_ratio_{suffix}.tif"), R_full.astype(np.float32))
            vals=R_full[np.isfinite(R_full)]
            if vals.size>0:
                lo,hi=auto_minmax(vals,1.0,99.0)
                Rclip=np.clip(R_full,lo,hi); Rnorm=(Rclip-lo)/(hi-lo+1e-12)
                imwrite(os.path.join(RAT16, f"{stid}_ratio_{suffix}_preview.tif"),(Rnorm*65535).astype(np.uint16), photometric='minisblack')
            else:
                imwrite(os.path.join(RAT16, f"{stid}_ratio_{suffix}_preview.tif"), np.zeros_like(R_full,dtype=np.uint16), photometric='minisblack')

        # PNG full
        if out_png and opt_full:
            vals=R_full[np.isfinite(R_full)]
            lo,hi=auto_minmax(vals,1.0,99.0)
            save_png_gray(R_full, os.path.join(PNG_FULL,f"{stid}_ratio_{suffix}.png"), vmin=lo, vmax=hi, dpi=png_dpi, out_px=None)

        # ROI ?? + PNG crop
        if polys:
            R_roi = R_full.copy()
            if union is not None: R_roi[~union]=np.nan
            if out_tif:
                imwrite(os.path.join(RROI32, f"{stid}_ratio_{suffix}.tif"), R_roi.astype(np.float32))
                vals=R_roi[np.isfinite(R_roi)]
                if vals.size>0:
                    lo,hi=auto_minmax(vals,1.0,99.0)
                    Rclip=np.clip(R_roi,lo,hi); Rnorm=(Rclip-lo)/(hi-lo+1e-12)
                    imwrite(os.path.join(RROI16, f"{stid}_ratio_{suffix}_preview.tif"),(Rnorm*65535).astype(np.uint16), photometric='minisblack')
                else:
                    imwrite(os.path.join(RROI16, f"{stid}_ratio_{suffix}_preview.tif"), np.zeros_like(R_full,dtype=np.uint16), photometric='minisblack')

            per_roi = quantify_per_roi(R_full, polys, extra_imgs={"donor":Dbc,"yfret":Abc})
            for r in per_roi:
                r.update({"stage":s,"time":(t_code if timelapse else None),"eps":eps,"p":p["percentile"],
                          "donor_p":d_p,"fret_p":a_p,"ratio_mode":p['ratio_mode'],
                          "bg_scope":p["bg_scope"],"bg_mode":p["bg_mode"],"clip_neg":p["clip_neg"],"eps_p":p["eps_percentile"]})
            rows_stage.extend(per_roi)

            if out_png and opt_crop:
                for i, P in enumerate(polys, 1):
                    pts = np.asarray(P)
                    minx,maxx = pts[:,0].min(), pts[:,0].max()
                    miny,maxy = pts[:,1].min(), pts[:,1].max()
                    pad = max(10, int(0.05*max(W,H)))
                    x0=max(int(minx)-pad,0); x1=min(int(maxx)+pad,W-1)
                    y0=max(int(miny)-pad,0); y1=min(int(maxy)+pad,H-1)
                    crop = R_full[y0:y1+1, x0:x1+1]
                    P2 = pts.copy(); P2[:,0]-=x0; P2[:,1]-=y0
                    mask = rasterize_polygon(P2, crop.shape)

                    # vmin/vmax
                    vmin=vmax=None
                    if opt_cmap:
                        try: vmin=float(p["cmin_txt"]) if p["cmin_txt"]!="" else None
                        except: vmin=None
                        try: vmax=float(p["cmax_txt"]) if p["cmax_txt"]!="" else None
                        except: vmax=None
                        if (vmin is None) or (vmax is None) or (vmax<=vmin):
                            lo,hi=auto_minmax(crop[mask],1.0,99.0)
                            if vmin is None: vmin=lo
                            if (vmax is None) or (vmax<=vmin): vmax=hi

                    out_png_path=os.path.join(PNG_CROP,f"{stid}_roi{i}_{suffix}.png")
                    if opt_cmap:
                        used_mask = (mask if p["mask_outside"] else None)
                        save_png_colormap(crop, out_png_path, vmin=vmin, vmax=vmax, cmap=cmap_name,
                                          mask=used_mask, scalebar=(sb_um if opt_sb else None),
                                          px_um=px_um, show_colorbar=p["show_colorbar"],
                                          dpi=png_dpi, out_px=out_size_crop)
                    else:
                        crop_vis=np.array(crop,copy=True)
                        if p["mask_outside"]: crop_vis[~mask]=0.0
                        lo,hi=auto_minmax(crop_vis[np.isfinite(crop_vis)],1.0,99.0)
                        save_png_gray(crop_vis, out_png_path, vmin=lo, vmax=hi, dpi=png_dpi, out_px=out_size_crop)

    logs.append(t("stage_end", "[Stage {stage}] end (total {count} time/files)", lang=LANG_CURRENT).format(stage=stage_key, count=len(pairs_for_stage)))
    return stage_key, rows_stage, logs
CMAP_CHOICES = ["jet","turbo","viridis","plasma","magma","inferno","cividis"]

class App(tk.Tk):
    def __init__(self, lang=LANG_DEFAULT):
        super().__init__()
        self.lang = lang
        self.title(t("title_app", "FRET Ratio Builder (v13-stage-parallel + Log)", lang=self.lang))
        self.resizable(False, False)
        self.params = self._default_params()
        self._build_gui()
        self.log_queue = queue.Queue()
        self.after(60, self._poll_log)  # GUI -> 큐 폴링

    # ---------- 파라미터 ----------
    def _default_params(self):
        return {
            "img_dir": None, "roi_dir": None, "out_root": "",
            "timelapse": False,
            "ratio_mode": "Donor/FRET",
            "donor_ch": 1, "acceptor_ch": 2,
            "bg_scope": "full", "bg_mode": "percentile",
            "percentile": 1.0, "per_channel_p": False,
            "donor_p": 1.0, "fret_p": 1.0,
            "clip_neg": True, "eps_percentile": 1.0,
            "px_um": None,
            "out_xls": True, "out_tif": True, "out_png": True,
            "save_full": False, "save_crop": True,
            "mask_outside": True,
            "apply_cmap": True, "cmap_name": "jet",
            "show_colorbar": False,
            "png_dpi": 300,
            "add_scalebar": False, "scale_bar_um": 20.0,
            "cmin_txt": "", "cmax_txt": "",
            "fixed_crop": True, "crop_w": 500, "crop_h": 500,
            "subset_on": False, "subset_stage": "", "subset_time": "", "subset_roi": "",
            "n_workers": max(1, (os.cpu_count() or 4) - 1)
        }

    # ---------- GUI 구성 ----------
    def _build_gui(self):
        p = self.params
        pad={'padx':8,'pady':6}
        tr = lambda key, default=None: t(key, default, lang=self.lang)

        # Tk variables
        self.img_v=tk.StringVar(); self.roi_v=tk.StringVar(); self.out_v=tk.StringVar(value="")
        self.tl_v=tk.BooleanVar(value=p["timelapse"])
        self.mode_v=tk.StringVar(value=p["ratio_mode"])
        self.donor_ch_v=tk.StringVar(value=str(p["donor_ch"])); self.acceptor_ch_v=tk.StringVar(value=str(p["acceptor_ch"]))
        self.scope_v=tk.StringVar(value=p["bg_scope"])
        self.bgmode_v=tk.StringVar(value=p["bg_mode"])
        self.p_v=tk.StringVar(value=str(p["percentile"]))
        self.per_ch_v=tk.BooleanVar(value=p["per_channel_p"])
        self.donor_p_v=tk.StringVar(value=str(p["donor_p"]))
        self.fret_p_v=tk.StringVar(value=str(p["fret_p"]))
        self.clip_v=tk.BooleanVar(value=p["clip_neg"])
        self.eps_v=tk.StringVar(value=str(p["eps_percentile"]))
        self.px_v=tk.StringVar(value="")
        self.out_xls=tk.BooleanVar(value=p["out_xls"]); self.out_tif=tk.BooleanVar(value=p["out_tif"]); self.out_png=tk.BooleanVar(value=p["out_png"])
        self.save_full=tk.BooleanVar(value=p["save_full"]); self.save_crop=tk.BooleanVar(value=p["save_crop"])
        self.mask_out=tk.BooleanVar(value=p["mask_outside"])
        self.apply_cmap=tk.BooleanVar(value=p["apply_cmap"]); self.cmap_v=tk.StringVar(value=p["cmap_name"])
        self.show_cbar=tk.BooleanVar(value=p["show_colorbar"])
        self.png_dpi_v=tk.StringVar(value=str(p["png_dpi"]))
        self.add_sb=tk.BooleanVar(value=p["add_scalebar"]); self.sb_um_v=tk.StringVar(value=str(p["scale_bar_um"]))
        self.cmin_v=tk.StringVar(value=p["cmin_txt"]); self.cmax_v=tk.StringVar(value=p["cmax_txt"])
        self.fixed_crop=tk.BooleanVar(value=p["fixed_crop"]); self.crop_w_v=tk.StringVar(value=str(p["crop_w"])); self.crop_h_v=tk.StringVar(value=str(p["crop_h"]))
        self.subset_on=tk.BooleanVar(value=p["subset_on"]); self.subset_stage=tk.StringVar(value=""); self.subset_time=tk.StringVar(value=""); self.subset_roi=tk.StringVar(value="")
        self.n_workers_v=tk.StringVar(value=str(p["n_workers"]))

        # ??
        tk.Label(self,text=tr("lbl_img","FRET images folder")).grid(row=0,column=0,sticky="w",**pad)
        fr=tk.Frame(self); fr.grid(row=0,column=1,sticky="ew",**pad)
        tk.Entry(fr,textvariable=self.img_v,width=50).pack(side="left")
        tk.Button(fr,text=tr("btn_browse","Browse"),width=8,command=self._browse_img).pack(side="left",padx=6)

        tk.Label(self,text=tr("lbl_roi","ROI data folder")).grid(row=1,column=0,sticky="w",**pad)
        fr=tk.Frame(self); fr.grid(row=1,column=1,sticky="ew",**pad)
        tk.Entry(fr,textvariable=self.roi_v,width=50).pack(side="left")
        tk.Button(fr,text=tr("btn_browse","Browse"),width=8,command=self._browse_roi).pack(side="left",padx=6)

        tk.Label(self,text=tr("lbl_out","Output root (optional)")).grid(row=2,column=0,sticky="w",**pad)
        fr=tk.Frame(self); fr.grid(row=2,column=1,sticky="ew",**pad)
        tk.Entry(fr,textvariable=self.out_v,width=50).pack(side="left")
        tk.Button(fr,text=tr("btn_browse","Browse"),width=8,command=self._browse_out).pack(side="left",padx=6)

        tk.Checkbutton(self, text=tr("cb_timelapse","Timelapse (filename=SXX_TXX_X)"), variable=self.tl_v, command=self._toggle_subset)            .grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0,2))

        # Ratio
        fr=tk.Frame(self); fr.grid(row=4,column=1,sticky="w",**pad)
        tk.Label(self,text=tr("lbl_ratio","Ratio mode")).grid(row=4,column=0,sticky="w",**pad)
        tk.Radiobutton(fr,text=t("ratio_fd","FRET / Donor", lang=self.lang),variable=self.mode_v,value="FRET/Donor").pack(side="left",padx=4)
        tk.Radiobutton(fr,text=t("ratio_df","Donor / FRET", lang=self.lang),variable=self.mode_v,value="Donor/FRET").pack(side="left",padx=4)
        tk.Label(fr,text=t("donor_ch","Donor ch", lang=self.lang)).pack(side="left",padx=(12,4))
        tk.Entry(fr,textvariable=self.donor_ch_v,width=4).pack(side="left")
        tk.Label(fr,text=t("acceptor_ch","Acceptor ch", lang=self.lang)).pack(side="left",padx=(10,4))
        tk.Entry(fr,textvariable=self.acceptor_ch_v,width=4).pack(side="left")

        # BG
        bgf=tk.LabelFrame(self,text=t("bg_group","BG options", lang=self.lang)); bgf.grid(row=5,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
        tk.Label(bgf,text=t("bg_scope","Scope", lang=self.lang)).grid(row=0,column=0,sticky="e",padx=6)
        ttk.Combobox(bgf,textvariable=self.scope_v,values=["full","roi_union"],width=10,state="readonly").grid(row=0,column=1,sticky="w")
        tk.Label(bgf,text=t("bg_mode","Mode", lang=self.lang)).grid(row=0,column=2,sticky="e",padx=6)
        ttk.Combobox(bgf,textvariable=self.bgmode_v,values=["percentile","hist-mode"],width=12,state="readonly").grid(row=0,column=3,sticky="w")
        tk.Label(bgf,text=t("bg_percentile","percentile p(%)", lang=self.lang)).grid(row=1,column=0,sticky="e",padx=6)
        tk.Entry(bgf,textvariable=self.p_v,width=10).grid(row=1,column=1,sticky="w")
        tk.Checkbutton(bgf,text=t("per_channel_p","per-channel p", lang=self.lang),variable=self.per_ch_v).grid(row=1,column=2,sticky="w")
        tk.Label(bgf,text=t("donor_p","Donor p", lang=self.lang)).grid(row=1,column=3,sticky="e"); tk.Entry(bgf,textvariable=self.donor_p_v,width=8).grid(row=1,column=4,sticky="w")
        tk.Label(bgf,text=t("fret_p","FRET p", lang=self.lang)).grid(row=1,column=5,sticky="e"); tk.Entry(bgf,textvariable=self.fret_p_v,width=8).grid(row=1,column=6,sticky="w")
        tk.Checkbutton(bgf,text=t("clip_neg","clip negatives", lang=self.lang),variable=self.clip_v).grid(row=2,column=0,columnspan=2,sticky="w",padx=6)
        tk.Label(bgf,text=t("eps_pct","? percentile(denom)%", lang=self.lang)).grid(row=2,column=2,sticky="e",padx=6)
        tk.Entry(bgf,textvariable=self.eps_v,width=10).grid(row=2,column=3,sticky="w")

        # ??/????
        tk.Label(self,text=tr("px_um","Pixel size (?m/px) ? for scalebar")).grid(row=6,column=0,sticky="w",**pad)
        tk.Entry(self,textvariable=self.px_v,width=12).grid(row=6,column=1,sticky="w",**pad)

        # ?? ??
        outf=tk.LabelFrame(self,text=t("out_group","Outputs", lang=self.lang)); outf.grid(row=7,column=0,columnspan=2,sticky="we",padx=8,pady=(6,6))
        tk.Checkbutton(outf,text=t("out_xls","XLS", lang=self.lang),variable=self.out_xls).grid(row=0,column=0,sticky="w",padx=8)
        tk.Checkbutton(outf,text=t("out_tif","TIF (RAT/RAT_ROI_masked)", lang=self.lang),variable=self.out_tif).grid(row=0,column=1,sticky="w",padx=8)
        tk.Checkbutton(outf,text=t("out_png","PNG", lang=self.lang),variable=self.out_png,command=self._toggle_png_group).grid(row=0,column=2,sticky="w",padx=8)

        # PNG ??
        tk.Label(self,text=t("png_opts","PNG options", lang=self.lang)).grid(row=8,column=0,sticky="nw",**pad)
        self.imgopt_frame=tk.Frame(self); self.imgopt_frame.grid(row=8,column=1,sticky="w",**pad)
        tk.Checkbutton(self.imgopt_frame,text=t("png_full","Full image grayscale only", lang=self.lang),variable=self.save_full).pack(anchor="w")
        tk.Checkbutton(self.imgopt_frame,text=t("png_crop","Crop image ROI-only", lang=self.lang),variable=self.save_crop,command=self._toggle_png_group).pack(anchor="w")
        tk.Checkbutton(self.imgopt_frame,text=t("mask_out","Crop: mask outside ROI", lang=self.lang),variable=self.mask_out).pack(anchor="w")

        row_sb=tk.Frame(self.imgopt_frame); row_sb.pack(anchor="w")
        tk.Checkbutton(row_sb,text=t("scalebar","Crop: scalebar", lang=self.lang),variable=self.add_sb,command=self._toggle_scalebar).pack(side="left")
        tk.Label(row_sb,text=t("sb_len"," length(?m): ", lang=self.lang)).pack(side="left")
        self.e_sb=tk.Entry(row_sb,textvariable=self.sb_um_v,width=8,state=tk.DISABLED); self.e_sb.pack(side="left",padx=(0,6))

        row_cmap=tk.Frame(self.imgopt_frame); row_cmap.pack(anchor="w",pady=(4,0))
        tk.Checkbutton(row_cmap,text=t("cmap","Crop: colormap", lang=self.lang),variable=self.apply_cmap,command=self._toggle_cmap).pack(side="left",padx=(0,8))
        tk.Label(row_cmap,text=t("cmap_label","Colormap:", lang=self.lang)).pack(side="left")
        self.dd_cmap=tk.OptionMenu(row_cmap,self.cmap_v,*CMAP_CHOICES); self.dd_cmap.config(width=8); self.dd_cmap.pack(side="left")
        tk.Checkbutton(row_cmap,text=t("show_cbar","Show colorbar", lang=self.lang),variable=self.show_cbar).pack(side="left",padx=8)

        row_cm2=tk.Frame(self.imgopt_frame); row_cm2.pack(anchor="w")
        tk.Label(row_cm2,text=t("cmin","Min", lang=self.lang)).pack(side="left"); self.e_cmin=tk.Entry(row_cm2,textvariable=self.cmin_v,width=8,state=tk.DISABLED); self.e_cmin.pack(side="left")
        tk.Label(row_cm2,text=t("cmax","  Max", lang=self.lang)).pack(side="left"); self.e_cmax=tk.Entry(row_cm2,textvariable=self.cmax_v,width=8,state=tk.DISABLED); self.e_cmax.pack(side="left")

        row_dpi=tk.Frame(self.imgopt_frame); row_dpi.pack(anchor="w",pady=(4,0))
        tk.Label(row_dpi,text=t("png_dpi","PNG DPI:", lang=self.lang)).pack(side="left"); tk.Entry(row_dpi,textvariable=self.png_dpi_v,width=6).pack(side="left")
        row_fix=tk.Frame(self.imgopt_frame); row_fix.pack(anchor="w",pady=(4,0))
        tk.Checkbutton(row_fix,text=t("fixed_crop","Use fixed crop size", lang=self.lang),variable=self.fixed_crop,command=self._toggle_fixedcrop).pack(side="left")
        tk.Label(row_fix,text=t("crop_wh","  W?H(px): ", lang=self.lang)).pack(side="left")
        self.e_cw=tk.Entry(row_fix,textvariable=self.crop_w_v,width=6); self.e_cw.pack(side="left")
        tk.Label(row_fix,text=" ? ").pack(side="left")
        self.e_ch=tk.Entry(row_fix,textvariable=self.crop_h_v,width=6); self.e_ch.pack(side="left")

        # Subset / Parallel
        subf=tk.LabelFrame(self,text=t("subset_group","Subset / Parallel", lang=self.lang)); subf.grid(row=9,column=0,columnspan=2,sticky="we",padx=8,pady=(6,6))
        tk.Checkbutton(subf,text=t("subset_on","Enable subset", lang=self.lang),variable=self.subset_on,command=self._toggle_subset).grid(row=0,column=0,sticky="w",padx=8,pady=6)
        tk.Label(subf,text=t("subset_stage","Stage(int)", lang=self.lang)).grid(row=1,column=0,sticky="e",padx=8); self.e_stage=tk.Entry(subf,textvariable=self.subset_stage,width=10,state=tk.DISABLED); self.e_stage.grid(row=1,column=1,sticky="w")
        tk.Label(subf,text=t("subset_time","Time(optional; timelapse)", lang=self.lang)).grid(row=1,column=2,sticky="e",padx=8); self.e_time=tk.Entry(subf,textvariable=self.subset_time,width=10,state=tk.DISABLED); self.e_time.grid(row=1,column=3,sticky="w")
        tk.Label(subf,text=t("subset_roi","ROI(optional)", lang=self.lang)).grid(row=1,column=4,sticky="e",padx=8); self.e_roi=tk.Entry(subf,textvariable=self.subset_roi,width=10,state=tk.DISABLED); self.e_roi.grid(row=1,column=5,sticky="w")
        tk.Label(subf,text=t("n_workers","Processes", lang=self.lang)).grid(row=0,column=2,sticky="e",padx=8); tk.Entry(subf,textvariable=self.n_workers_v,width=8).grid(row=0,column=3,sticky="w")

        # ??/?? ??
        fr=tk.Frame(self); fr.grid(row=10,column=0,columnspan=2,pady=8)
        self.btn_run = tk.Button(fr,text=t("btn_run","Run", lang=self.lang),width=12,command=self._on_run)
        self.btn_run.pack(side="left",padx=6)
        tk.Button(fr,text=t("btn_quit","Quit", lang=self.lang),width=12,command=self.destroy).pack(side="left",padx=6)

        # ----- ?? LOG -----
        tk.Label(self,text=t("log_label","Log", lang=self.lang)).grid(row=11,column=0,sticky="w",padx=8,pady=(0,2))
        self.log_txt = scrolledtext.ScrolledText(self, wrap="word", height=12, state="disabled")
        self.log_txt.grid(row=12,column=0,columnspan=2,sticky="we",padx=8,pady=(0,8))

        # ?? ???
        self._toggle_png_group(); self._toggle_subset()
    # ---------- 파일 다이얼로그 ----------
    def _browse_img(self):
        p=filedialog.askdirectory(title=t("browse_img","Select FRET images folder", lang=self.lang))
        if p: self.img_v.set(p)
    def _browse_roi(self):
        p=filedialog.askdirectory(title=t("browse_roi","Select ROI data folder", lang=self.lang))
        if p: self.roi_v.set(p)
    def _browse_out(self):
        p=filedialog.askdirectory(title=t("browse_out","Select output root (RES parent)", lang=self.lang))
        if p: self.out_v.set(p)

    # ---------- 옵션 enable/disable ----------
    def _set_state_recursive(self, w, state):
        try: w.configure(state=state)
        except: pass
        for ch in getattr(w,'winfo_children',lambda:[])():
            self._set_state_recursive(ch, state)

    def _toggle_png_group(self, *_):
        st = tk.NORMAL if self.out_png.get() else tk.DISABLED
        self._set_state_recursive(self.imgopt_frame, st)
        self._toggle_cmap(); self._toggle_scalebar(); self._toggle_fixedcrop()

    def _toggle_cmap(self, *_):
        s = (self.out_png.get() and self.save_crop.get() and self.apply_cmap.get())
        self.dd_cmap.config(state=(tk.NORMAL if s else tk.DISABLED))
        self.e_cmin.config(state=(tk.NORMAL if s else tk.DISABLED))
        self.e_cmax.config(state=(tk.NORMAL if s else tk.DISABLED))

    def _toggle_scalebar(self, *_):
        self.e_sb.config(state=(tk.NORMAL if (self.out_png.get() and self.save_crop.get() and self.add_sb.get()) else tk.DISABLED))

    def _toggle_fixedcrop(self, *_):
        on = (self.out_png.get() and self.save_crop.get() and self.fixed_crop.get())
        self.e_cw.config(state=(tk.NORMAL if on else tk.DISABLED))
        self.e_ch.config(state=(tk.NORMAL if on else tk.DISABLED))

    def _toggle_subset(self, *_):
        on = self.subset_on.get()
        self.e_stage.config(state=(tk.NORMAL if on else tk.DISABLED))
        self.e_time.config(state=(tk.NORMAL if (on and self.tl_v.get()) else tk.DISABLED))
        self.e_roi.config(state=(tk.NORMAL if on else tk.DISABLED))

    # ---------- Log 처리 ----------
    def log(self, msg: str):
        self.log_queue.put(msg.rstrip())

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_txt.configure(state="normal")
                self.log_txt.insert("end", msg + "\n")
                self.log_txt.see("end")
                self.log_txt.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.after(60, self._poll_log)

    # ---------- 실행 ----------
    def _collect_params(self):
        img=self.img_v.get().strip(); roi=self.roi_v.get().strip()
        if not roi:
            roi=os.path.join(img,"roi")
        if not img or not os.path.isdir(img): raise ValueError(t("err_invalid_img","Select a valid FRET images folder.", lang=self.lang))
        if not os.path.isdir(roi): raise ValueError(t("err_invalid_roi","Select a valid ROI folder.", lang=self.lang))
        try:
            p=float(self.p_v.get()); assert 0.0<=p<=10.0
            eps=float(self.eps_v.get()); assert 0.0<=eps<=10.0
        except:
            raise ValueError(t("err_percentile","percentile/ε must be 0~10.", lang=self.lang))
        d_p=p; f_p=p
        if self.per_ch_v.get():
            try:
                d_p=float(self.donor_p_v.get()); f_p=float(self.fret_p_v.get())
                assert 0.0<=d_p<=10.0 and 0.0<=f_p<=10.0
            except:
                raise ValueError(t("err_per_channel","Per-channel p must be 0~10", lang=self.lang))

        px_um=None
        if self.px_v.get().strip():
            try:
                px_um=float(self.px_v.get()); assert 1e-4<=px_um<=100.0
            except:
                raise ValueError(t("err_px","Pixel size (µm/px) input error", lang=self.lang))

        try:
            dpi=int(float(self.png_dpi_v.get())); assert 50<=dpi<=1200
        except:
            raise ValueError(t("err_dpi","PNG DPI must be 50~1200", lang=self.lang))

        fc = bool(self.fixed_crop.get())
        cw=ch_=None
        if self.out_png.get() and self.save_crop.get() and fc:
            try:
                cw=int(float(self.crop_w_v.get())); ch_=int(float(self.crop_h_v.get()))
                assert 32<=cw<=8000 and 32<=ch_<=8000
            except:
                raise ValueError(t("err_crop","Crop size must be 32~8000", lang=self.lang))

        sub = {"on": bool(self.subset_on.get()), "stage": None, "time": None, "roi": None}
        if self.subset_on.get():
            try: sub["stage"]=int(self.subset_stage.get()); assert sub["stage"]>=0
            except: raise ValueError(t("err_stage","Stage must be an integer", lang=self.lang))
            if self.tl_v.get() and self.subset_time.get().strip():
                try: sub["time"]=int(self.subset_time.get()); assert sub["time"]>=0
                except: raise ValueError(t("err_time","Timepoint must be an integer", lang=self.lang))
            if self.subset_roi.get().strip():
                try: sub["roi"]=int(self.subset_roi.get()); assert sub["roi"]>=1
                except: raise ValueError(t("err_roi","ROI must be integer >= 1", lang=self.lang))

        try:
            n_workers = int(self.n_workers_v.get()); assert 1 <= n_workers <= max(1, (os.cpu_count() or 4))
        except:
            raise ValueError(t("err_workers","Processes must be 1..CPU cores", lang=self.lang))

        try:
            donor_ch = int(self.donor_ch_v.get()); acceptor_ch = int(self.acceptor_ch_v.get())
            assert 1 <= donor_ch <= 32 and 1 <= acceptor_ch <= 32 and donor_ch != acceptor_ch
        except:
            raise ValueError(t("err_channels","Donor/Acceptor channels must be 1~32 and different", lang=self.lang))

        self.params.update(
            img_dir=img, roi_dir=roi, out_root=self.out_v.get().strip(),
            timelapse=bool(self.tl_v.get()),
            ratio_mode=self.mode_v.get(), donor_ch=donor_ch, acceptor_ch=acceptor_ch,
            bg_scope=self.scope_v.get(), bg_mode=self.bgmode_v.get(),
            percentile=p, per_channel_p=bool(self.per_ch_v.get()),
            donor_p=d_p, fret_p=f_p, clip_neg=bool(self.clip_v.get()), eps_percentile=eps,
            px_um=px_um, out_xls=bool(self.out_xls.get()), out_tif=bool(self.out_tif.get()), out_png=bool(self.out_png.get()),
            save_full=bool(self.save_full.get()), save_crop=bool(self.save_crop.get()), mask_outside=bool(self.mask_out.get()),
            apply_cmap=bool(self.apply_cmap.get()), cmap_name=self.cmap_v.get(), show_colorbar=bool(self.show_cbar.get()),
            png_dpi=dpi, add_scalebar=bool(self.add_sb.get()), scale_bar_um=float(self.sb_um_v.get()) if self.add_sb.get() else None,
            cmin_txt=self.cmin_v.get().strip(), cmax_txt=self.cmax_v.get().strip(),
            fixed_crop=fc, crop_w=cw if cw else 500, crop_h=ch_ if ch_ else 500,
            subset_on=sub["on"], subset_stage=sub["stage"], subset_time=sub["time"], subset_roi=sub["roi"],
            n_workers=n_workers
        )
    def _lock_form(self, locked: bool):
        state = tk.DISABLED if locked else tk.NORMAL
        # 모든 위젯 잠그기/풀기
        for w in self.winfo_children():
            if w is self.log_txt:  # 로그창은 항상 활성
                continue
            self._set_state_recursive(w, state)
        # 실행 버튼만 상태 덮어쓰기
        self.btn_run.configure(state=(tk.DISABLED if locked else tk.NORMAL))

    def _on_run(self):
        try:
            self._collect_params()
        except ValueError as e:
            messagebox.showerror(t("err_title","Input error", lang=self.lang), str(e))
            return

        # 로그 초기 한 줄
        self.log("\n================= 실행 시작 =================")
        self._lock_form(True)

        # 백그라운드 스레드에서 실행(메인 GUI는 유지)
        threading.Thread(target=self._pipeline_thread, daemon=True).start()

    # ---------- 파이프라인(스레드) ----------
    def _pipeline_thread(self):
        p = self.params
        try:
            out_root = p["out_root"].strip() or os.path.join(p["img_dir"], "RES")
            RES_ROOT = ensure_dir(out_root)
            XLS_DIR   = ensure_dir(os.path.join(RES_ROOT, "xls")) if p["out_xls"] else None
            RAT_ROOT  = ensure_dir(os.path.join(RES_ROOT, "RAT")) if p["out_tif"] else None
            RAT32     = ensure_dir(os.path.join(RAT_ROOT, "32bit")) if p["out_tif"] else None
            RAT16     = ensure_dir(os.path.join(RAT_ROOT, "16bit")) if p["out_tif"] else None
            RROI_ROOT = ensure_dir(os.path.join(RES_ROOT, "RAT_ROI_masked")) if p["out_tif"] else None
            RROI32    = ensure_dir(os.path.join(RROI_ROOT, "32bit")) if p["out_tif"] else None
            RROI16    = ensure_dir(os.path.join(RROI_ROOT, "16bit")) if p["out_tif"] else None
            PNG_ROOT  = ensure_dir(os.path.join(RES_ROOT, "PNG_RAT")) if p["out_png"] else None
            PNG_FULL  = ensure_dir(os.path.join(PNG_ROOT, "full")) if p["out_png"] else None
            PNG_CROP  = ensure_dir(os.path.join(PNG_ROOT, "crop")) if p["out_png"] else None

            files = list_tifs(p["img_dir"])

            donors, accs = {}, {}
            timelapse = bool(p["timelapse"])
            donor_ch = int(p["donor_ch"]); acceptor_ch = int(p["acceptor_ch"])
            for path in files:
                base = os.path.basename(path)
                s_num, t_num, ch = parse_tokens(base, timelapse)
                if s_num is None or ch is None:
                    continue
                s = fmt_stage(s_num)
                t_code = fmt_time(t_num) if (timelapse and t_num is not None) else None
                key = (s, t_code)
                if ch == donor_ch: donors[key] = path
                elif ch == acceptor_ch: accs[key] = path

            pair_keys = sorted(set(donors.keys()) & set(accs.keys()), key=lambda k: (
                int(re.search(r'\d+', k[0]).group()),
                (int(re.search(r'\d+', k[1]).group()) if k[1] else -1)
            ))
            pairs = [((s,t), donors[(s,t)], accs[(s,t)]) for (s,t) in pair_keys]

            if not pairs:
                self.log(t("msg_match_none","No matched files for donor=_{donor}, acceptor=_{acceptor}.", lang=self.lang).format(donor=donor_ch, acceptor=acceptor_ch))
                return

            if p["subset_on"] and (p["subset_stage"] is not None):
                s_code = fmt_stage(p["subset_stage"])
                if (not timelapse) or p["subset_time"] is None:
                    pairs=[pp for pp in pairs if pp[0][0]==s_code]
                else:
                    t_code=fmt_time(p["subset_time"])
                    pairs=[pp for pp in pairs if (pp[0][0]==s_code and pp[0][1]==t_code)]
                if not pairs:
                    self.log(t("msg_subset_none","[Subset] No files match Stage={stage}.", lang=self.lang).format(stage=s_code))
                    return

            stage_groups = {}
            for pr in pairs:
                s = pr[0][0]
                stage_groups.setdefault(s, []).append(pr)
            ordered_stages = sorted(stage_groups.keys(), key=lambda s: int(re.search(r'\d+', s).group()))
            self.log(t("msg_info_stage","[Info] Stages={count} / workers={workers}", lang=self.lang).format(count=len(ordered_stages), workers=p["n_workers"]))

            freeze_support()
            paths = (RES_ROOT, RAT32, RAT16, RROI32, RROI16, PNG_FULL, PNG_CROP)

            all_rows = []
            t0=time.time()
            with ProcessPoolExecutor(max_workers=int(p["n_workers"])) as ex:
                futs = {ex.submit(process_one_stage, s, stage_groups[s], p, paths): s for s in ordered_stages}
                done=0; total=len(ordered_stages)
                for fut in as_completed(futs):
                    s = futs[fut]
                    try:
                        stage_key, rows, stage_logs = fut.result()
                        for line in stage_logs: self.log(line)
                        all_rows.append((stage_key, rows))
                    except Exception:
                        self.log(t("msg_stage_error","[Error] Stage {stage} raised an exception:", lang=self.lang).format(stage=s))
                        self.log(traceback.format_exc())
                    done += 1
                    self.log(t("msg_progress","[Progress] {done}/{total} stages done: {stage}", lang=self.lang).format(done=done, total=total, stage=s))

            # 병합(순서 보장)
            rows_all=[]
            for s in ordered_stages:
                for (ss, rows) in all_rows:
                    if ss==s:
                        rows_all.extend(rows); break

            # 저장
            if p["out_xls"] and rows_all:
                df=pd.DataFrame(rows_all)
                cols=["stage","time","roi","area_px","ratio_mean","ratio_median","ratio_std","ratio_p5","ratio_p95",
                      "donor_mean","donor_median","yfret_mean","yfret_median","eps","p","ratio_mode","bg_mode"]
                df=df[[c for c in cols if c in df.columns]]
                if timelapse:
                    df["time_idx"]=df["time"].str.extract(r"t(\d+)",expand=False).astype(int)
                else:
                    df["time_idx"]=0
                df["stage_idx"]=df["stage"].str.extract(r"S(\d+)",expand=False).astype(int)
                df["roi_lab"]="s"+df["stage_idx"].astype(str)+"c"+df["roi"].astype(str)
                mean_mat=df.pivot(index="time_idx",columns="roi_lab",values="ratio_mean").sort_index()
                med_mat =df.pivot(index="time_idx",columns="roi_lab",values="ratio_median").sort_index()

                def _pick_engine():
                    try:
                        import xlsxwriter; return 'xlsxwriter'
                    except Exception:
                        try:
                            import openpyxl; return 'openpyxl'
                        except Exception:
                            return None
                xlsx=os.path.join(RES_ROOT,"xls","fret_ratio_perROI.xlsx")
                csv =os.path.join(RES_ROOT,"xls","fret_ratio_perROI.csv")
                eng=_pick_engine()
                if eng:
                    with pd.ExcelWriter(xlsx, engine=eng) as w:
                        df.to_excel(w, index=False, sheet_name="per_ROI")
                        mean_mat.to_excel(w, sheet_name="ratio_mean_matrix")
                        med_mat.to_excel(w, sheet_name="ratio_median_matrix")
                    self.log(t("msg_save_xlsx","[Saved] xls/{name}", lang=self.lang).format(name=os.path.basename(xlsx)))
                df.to_csv(csv, index=False); self.log(t("msg_save_csv","[Saved] xls/{name}", lang=self.lang).format(name=os.path.basename(csv)))
            elif p["out_xls"] and not rows_all:
                self.log(t("msg_no_roi_table","[Warn] No ROI → metric table not generated.", lang=self.lang))
            else:
                self.log(t("msg_xls_off","[Info] XLS output disabled", lang=self.lang))

            dt=time.time()-t0
            mm=int(dt//60); ss=int(dt%60)
            self.log(t("msg_done","[Done] Parallel (per stage) finished in {mm:02d}:{ss:02d}", lang=self.lang).format(mm=mm, ss=ss))
        finally:
            # GUI 복구
            self._lock_form(False)
            self.log(t("msg_run_end","================= Run end =================", lang=self.lang) + "\n")

# -------------------- 엔트리 포인트 --------------------
def main():
    global LANG_CURRENT
    LANG_CURRENT = pick_lang_from_argv(sys.argv[1:])
    app = App(lang=LANG_CURRENT)
    app.mainloop()

if __name__=="__main__":
    main()

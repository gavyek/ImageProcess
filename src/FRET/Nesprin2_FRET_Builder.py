#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nesprin2_FRET Builder (v1.1, patched)
V1.1 patch(25.11.24)
- -mode EN 추가
- Log창 활성화 + 단일 실행 흐름 정리 + 파일명 저장 규칙 보완
- PATCH
  - PNG master ON일 때만 패널/크롭/풀 이미지 옵션 활성화
  - "PNG: full ratio(gray)" 라벨을 "PNG: full image"로 변경
  - 스케일 preset / 컬러바 / 스케일바 기본값 조정
"""

import os, re, glob, json, math
import numpy as np
import pandas as pd
import sys, logging, queue, threading, contextlib
from tkinter import scrolledtext
from tifffile import imread, imwrite
from PIL import Image

from scipy.ndimage import distance_transform_edt, binary_dilation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# -------------------- Language resources --------------------
LANG_DEFAULT = "ko"
LANG_CURRENT = LANG_DEFAULT

STRINGS = {
    "ko": {
        "title_app": "Nesprin2_FRET Builder (v4)",
        "lbl_img": "이미지 경로",
        "lbl_roi": "ROI 경로",
        "lbl_out": "출력 루트(선택)",
        "btn_browse": "찾기",
        "cb_timelapse": "Timelapse 파일명(SXX_tXX_X)",
        "frame_channels": "채널 선택 & Ratio",
        "lbl_donor": "Donor",
        "lbl_fret": "FRET",
        "lbl_intensity": "Intensity표시 채널",
        "ratio_fd": "FRET / Donor (YFRET/mTFP)",
        "ratio_df": "Donor / FRET (mTFP/YFRET)",
        "frame_bg": "BG 옵션",
        "bg_scope": "Scope",
        "bg_mode": "Mode",
        "bg_percentile": "percentile p(%)",
        "bg_per_ch": "채널별 p",
        "bg_donor_p": "Donor p",
        "bg_fret_p": "FRET p",
        "bg_clip_neg": "음수 clip",
        "bg_eps_pct": "ε 퍼센타일(denom)%",
        "frame_metric": "픽셀/림/애뉴러스(µm)",
        "metric_px": "픽셀크기 (µm/px)",
        "rim_preset_label": "림/애뉴러스 프리셋",
        "rim_thickness": "림 두께 (µm)",
        "rim_annulus_enable": "로컬 BG 애뉴러스 사용",
        "rim_annulus_inout": "내경/외경 (µm)",
        "rim_preset_thin": "Thin (림 0.45 µm, 애뉴러스 0.6/1.5 µm)",
        "rim_preset_medium": "Medium (림 0.67 µm, 애뉴러스 0.9/1.8 µm)",
        "rim_preset_thick": "Thick (림 1.00 µm, 애뉴러스 1.2/2.0 µm)",
        "rim_preset_custom": "사용자 정의",
        "frame_spec": "스펙트럴 보정(선택)",
        "spec_use": "스펙트럴 교정 사용",
        "spec_alpha": "α(d→F)",
        "spec_beta": "β(a→F)",
        "spec_g": "G",
        "spec_aonly": "Acceptor-only 채널(선택)",
        "frame_display": "표시/스케일",
        "display_preset": "Preset",
        "display_fret_minmax": "FRET min/max",
        "display_colormap": "컬러맵",
        "display_show_cbar": "컬러바 표시",
        "display_scalebar": "스케일바(µm)",
        "frame_output": "출력",
        "output_xls": "XLS",
        "output_tif": "TIF",
        "output_png": "PNG",
        "output_panel": "PNG: 패널(Intensity+FRET, rim-only)",
        "output_full": "PNG: full image",
        "scale_preset_015": "0-1.5",
        "scale_preset_010": "0-1.0",
        "scale_preset_custom": "Custom",
        "frame_crop": "PNG: ROI Crop",
        "crop_save": "저장",
        "crop_save_intensity": "Intensity rim-only도 저장",
        "crop_mask_outside": "ROI 바깥 마스킹",
        "crop_fixed": "고정 픽셀 크기",
        "crop_wh": "W×H(px)",
        "crop_vmin": "vmin",
        "crop_vmax": "vmax",
        "frame_subset": "부분 추출",
        "subset_enable": "활성화",
        "subset_stage": "Stage",
        "subset_time": "Time",
        "frame_qc": "QC / Outlier 옵션",
        "qc_sat_filter": "포화 픽셀 제거",
        "qc_sat_threshold": "포화 임계값",
        "qc_clip_ratio": "고비율(Ratio) 제외",
        "qc_clip_max": "최대 비율",
        "log_label": "Log",
        "btn_run": "실행",
        "btn_cancel": "취소",
        "browse_img": "이미지 폴더 선택",
        "browse_roi": "ROI 폴더 선택",
        "browse_out": "출력 루트(RES 상위) 선택",
        "err_title": "오류",
        "err_invalid_img": "유효한 이미지 경로를 선택하세요.",
        "err_invalid_roi": "유효한 ROI 경로를 선택하세요.",
        "err_channels": "채널 선택을 확인하세요.",
        "err_percentile": "percentile/ε 입력 오류",
        "err_metric": "픽셀·림 입력 오류",
        "err_annulus": "애뉴러스 내/외경(µm) 확인",
        "err_fret_minmax": "FRET min/max 확인",
        "err_scalebar": "스케일바 길이 확인",
        "err_crop": "Crop W/H(px) 확인",
        "err_qc": "QC 옵션 확인",
        "log_batch_start": "배치 작업을 시작합니다.",
        "log_batch_end": "모든 작업이 정상 종료되었습니다.",
        "log_worker_exception": "작업 도중 예외 발생: {err}",
        "log_ready": "준비 완료. 파라미터 변경 후 다시 [실행]을 누르세요.",
        "msg_warn_no_roi_table": "[주의] ROI가 없어 정량 테이블이 없습니다.",
        "msg_save_csv": "[저장] xls/{name} (CSV)",
        "msg_save_xlsx": "[저장] xls/{name} (XLSX)",
        "msg_warn_no_openpyxl": "[경고] openpyxl 미설치 → XLSX 저장 생략, CSV만 저장되었습니다.",
        "msg_info_pairs": "[정보] 총 처리 대상 쌍: {count}",
        "msg_no_pairs": "매칭되는 (donor, fret) 채널 쌍이 없습니다.",
        "msg_subset_none": "[부분추출] 조건에 맞는 쌍이 없습니다.",
        "msg_processing": "[처리] {tag} ...",
        "msg_warn_no_roi_tag": "[경고] {tag}: ROI 없음 — 건너뜀",
        "msg_done_outdir": "[완료] 출력 폴더: {dir}",
    },
    "en": {
        "title_app": "Nesprin2_FRET Builder (v4)",
        "lbl_img": "Image folder",
        "lbl_roi": "ROI folder",
        "lbl_out": "Output root (optional)",
        "btn_browse": "Browse",
        "cb_timelapse": "Timelapse filenames (SXX_tXX_X)",
        "frame_channels": "Channels & Ratio",
        "lbl_donor": "Donor",
        "lbl_fret": "FRET",
        "lbl_intensity": "Intensity channel",
        "ratio_fd": "FRET / Donor (YFRET/mTFP)",
        "ratio_df": "Donor / FRET (mTFP/YFRET)",
        "frame_bg": "BG options",
        "bg_scope": "Scope",
        "bg_mode": "Mode",
        "bg_percentile": "percentile p(%)",
        "bg_per_ch": "per-channel p",
        "bg_donor_p": "Donor p",
        "bg_fret_p": "FRET p",
        "bg_clip_neg": "clip negatives",
        "bg_eps_pct": "ε percentile(denom)%",
        "frame_metric": "Pixel/Rim/Annulus (µm)",
        "metric_px": "Pixel size (µm/px)",
        "rim_preset_label": "Rim/Annulus preset",
        "rim_thickness": "Rim thickness (µm)",
        "rim_annulus_enable": "Use local BG annulus",
        "rim_annulus_inout": "Inner/outer (µm)",
        "rim_preset_thin": "Thin (rim 0.45 µm, ann 0.6/1.5 µm)",
        "rim_preset_medium": "Medium (rim 0.67 µm, ann 0.9/1.8 µm)",
        "rim_preset_thick": "Thick (rim 1.00 µm, ann 1.2/2.0 µm)",
        "rim_preset_custom": "Custom",
        "frame_spec": "Spectral correction (optional)",
        "spec_use": "Use spectral correction",
        "spec_alpha": "α(d→F)",
        "spec_beta": "β(a→F)",
        "spec_g": "G",
        "spec_aonly": "Acceptor-only channel (optional)",
        "frame_display": "Display / Scale",
        "display_preset": "Preset",
        "display_fret_minmax": "FRET min/max",
        "display_colormap": "Colormap",
        "display_show_cbar": "Show colorbar",
        "display_scalebar": "Scalebar (µm)",
        "frame_output": "Outputs",
        "output_xls": "XLS",
        "output_tif": "TIF",
        "output_png": "PNG",
        "output_panel": "PNG: panel (Intensity+FRET, rim-only)",
        "output_full": "PNG: full image",
        "scale_preset_015": "0-1.5",
        "scale_preset_010": "0-1.0",
        "scale_preset_custom": "Custom",
        "frame_crop": "PNG: ROI Crop",
        "crop_save": "Save",
        "crop_save_intensity": "Save intensity rim-only",
        "crop_mask_outside": "Mask outside ROI",
        "crop_fixed": "Use fixed pixel size",
        "crop_wh": "W×H(px)",
        "crop_vmin": "vmin",
        "crop_vmax": "vmax",
        "frame_subset": "Subset",
        "subset_enable": "Enable",
        "subset_stage": "Stage",
        "subset_time": "Time",
        "frame_qc": "QC / Outlier options",
        "qc_sat_filter": "Remove saturated pixels",
        "qc_sat_threshold": "Saturation threshold",
        "qc_clip_ratio": "Remove high ratio",
        "qc_clip_max": "Max ratio",
        "log_label": "Log",
        "btn_run": "Run",
        "btn_cancel": "Cancel",
        "browse_img": "Select image folder",
        "browse_roi": "Select ROI folder",
        "browse_out": "Select output root (RES parent)",
        "err_title": "Error",
        "err_invalid_img": "Select a valid image folder.",
        "err_invalid_roi": "Select a valid ROI folder.",
        "err_channels": "Check channel selections.",
        "err_percentile": "percentile/ε input error",
        "err_metric": "Pixel/rim input error",
        "err_annulus": "Check annulus inner/outer (µm)",
        "err_fret_minmax": "Check FRET min/max",
        "err_scalebar": "Check scalebar length",
        "err_crop": "Check crop W/H(px)",
        "err_qc": "Check QC options",
        "log_batch_start": "Starting batch processing.",
        "log_batch_end": "All tasks finished successfully.",
        "log_worker_exception": "Exception during processing: {err}",
        "log_ready": "Ready. Modify parameters and press [Run] again.",
        "msg_warn_no_roi_table": "[Warn] No ROI → metric table is empty.",
        "msg_save_csv": "[Saved] xls/{name} (CSV)",
        "msg_save_xlsx": "[Saved] xls/{name} (XLSX)",
        "msg_warn_no_openpyxl": "[Warn] openpyxl missing → skipped XLSX, saved CSV only.",
        "msg_info_pairs": "[Info] Total pairs: {count}",
        "msg_no_pairs": "No matched (donor, fret) channel pairs.",
        "msg_subset_none": "[Subset] No pairs match the filter.",
        "msg_processing": "[Processing] {tag} ...",
        "msg_warn_no_roi_tag": "[Warn] {tag}: ROI missing — skipped",
        "msg_done_outdir": "[Done] Output folder: {dir}",
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

# ===================== 공통 유틸 =====================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_tifs(folder):
    exts = ("*.tif","*.tiff","*.TIF","*.TIFF")
    files=[]
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    uniq={}
    for p in files:
        norm=os.path.normcase(os.path.abspath(p))
        if norm not in uniq:
            uniq[norm]=p
    return sorted(uniq.values(), key=natural_key)

def fmt_stage(n):  # 'S01'
    return f"S{int(n):02d}"

def fmt_time(n):   # 't00'
    return f"t{int(n):02d}"

def parse_tokens(basename: str, timelapse: bool):
    name=os.path.splitext(basename)[0]
    # channel
    ch=None
    m_ch=re.search(r'(?:[_-](\d+)$)|(?:[_-](?:ch|c)(\d+)$)', name, flags=re.IGNORECASE)
    if m_ch:
        ch=int(next(g for g in m_ch.groups() if g is not None))
    # stage
    m_s=re.search(r'(?i)(?:^|[_-])S(\d+)(?=$|[_-])', name)
    s_num=int(m_s.group(1)) if m_s else None
    # time
    t_num=None
    if timelapse:
        m_t=re.search(r'(?i)(?:^|[_-])t(\d+)(?=$|[_-])', name)
        t_num=int(m_t.group(1)) if m_t else None
    return s_num, t_num, ch

def clean_base_for_save(basename: str, timelapse: bool):
    s_num,t_num,_=parse_tokens(basename,timelapse)
    if s_num is None:
        name=os.path.splitext(basename)[0]
        return re.sub(r'([_-])\d+$','',name)
    if timelapse and (t_num is not None):
        return f"{fmt_stage(s_num)}_{fmt_time(t_num)}"
    return fmt_stage(s_num)

def find_json_path(roi_dir, basename, timelapse):
    s_num,t_num,_=parse_tokens(basename,timelapse)
    norm=clean_base_for_save(basename,timelapse)
    candidates=[os.path.join(roi_dir, norm + ".json")]
    if s_num is not None:
        legacy=f"S{int(s_num)}"
        if timelapse and (t_num is not None):
            legacy=f"{legacy}_t{int(t_num)}"
        candidates.append(os.path.join(roi_dir, legacy + ".json"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def read_tiff_with_fallback(path, page=0):
    try:
        arr = imread(path, key=page)
        return arr
    except Exception as e:
        msg=str(e)
        if ("imagecodecs" not in msg) and ("COMPRESSION" not in msg):
            raise
        with Image.open(path) as im:
            try:
                im.seek(page)
            except EOFError:
                im.seek(0)
            return np.array(im)

def read_2d(path):
    a=read_tiff_with_fallback(path)
    if a.ndim>2:
        a=a[...,0] if a.ndim==3 else a[0,...]
    return a.astype(np.float32, copy=False)

def to_int_or_none(x):
    if x is None:
        return None
    if isinstance(x, str):
        s=x.strip()
        if s=="":
            return None
        try:
            return int(float(s))
        except:
            return None
    try:
        return int(x)
    except:
        return None

# 파일명 끝 채널 교체 (안전)
def swap_ch(path, old_ch, new_ch):
    b=os.path.basename(path); d=os.path.dirname(path)
    pat = re.compile(
        rf'([_-])(?:ch|c)?{re.escape(str(old_ch))}(\.(?:tif|tiff))$',
        flags=re.IGNORECASE
    )
    if pat.search(b):
        b2 = pat.sub(rf'\g<1>{new_ch}\g<2>', b)
        return os.path.join(d, b2)
    pat_fallback = re.compile(r'(.+?)([_-])(\d+)(\.(?:tif|tiff))$', flags=re.IGNORECASE)
    m = pat_fallback.match(b)
    if m:
        prefix, dash, digits, ext = m.groups()
        return os.path.join(d, f"{prefix}{dash}{new_ch}{ext}")
    return path

# ===================== ROI/마스크/보정 =====================

def rasterize_polygon(poly, shape):
    H,W=shape
    path=mpath.Path(np.asarray(poly, dtype=float))
    yy,xx=np.mgrid[0:H,0:W]
    pts=np.vstack((xx.ravel(),yy.ravel())).T
    return path.contains_points(pts).reshape(H,W)

def load_roi_polys(roi_folder, s, t, timelapse):
    base=f"{s}_{t}" if (timelapse and t is not None) else s
    json_path=find_json_path(roi_folder, base, timelapse)
    if not os.path.exists(json_path):
        return None
    with open(json_path,"r",encoding="utf-8") as f:
        data=json.load(f)
    polys=[]
    for poly in data.get("rois",[]):
        P=np.asarray(poly, dtype=float)
        if P.shape[0]>=3:
            polys.append(P)
    return polys if polys else None

def make_inside_rim_mask(union_mask, rim_px:int):
    if rim_px<=0:
        return union_mask.copy()
    dist_in=distance_transform_edt(union_mask)
    rim=(dist_in>0) & (dist_in<=rim_px)
    return rim

def annulus_mask_from_poly(poly, shape, inner_px:int, outer_px:int):
    H,W=shape
    base=rasterize_polygon(poly,(H,W))
    if inner_px<1:
        inner_px=1
    if outer_px<=inner_px:
        outer_px=inner_px+1
    se_out = np.ones((2*outer_px+1, 2*outer_px+1), dtype=bool)
    se_in  = np.ones((2*inner_px+1, 2*inner_px+1), dtype=bool)
    out=binary_dilation(base, structure=se_out)
    inn=binary_dilation(base, structure=se_in)
    return out & (~inn)

def _vals_in_scope(img2d, scope_mask):
    return img2d.ravel() if scope_mask is None else img2d[scope_mask]

def bg_value(img2d, mode="percentile", p=1.0, scope_mask=None):
    vals=_vals_in_scope(img2d, scope_mask)
    if vals.size==0:
        return 0.0
    vals = vals[np.isfinite(vals)]
    if vals.size==0:
        return 0.0
    if mode=="percentile":
        return float(np.percentile(vals, p))
    elif mode=="hist-mode":
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
    B=bg_value(img2d, mode=mode, p=p, scope_mask=scope_mask)
    J=img2d - B
    if clip_neg:
        J[J<0]=0.0
    return J, B

def spectral_correct(yfret, donor, acceptor_only=None, alpha=0.0, beta=0.0, g_factor=1.0):
    yf=yfret.astype(np.float32, copy=False)
    d = donor.astype(np.float32, copy=False)
    if acceptor_only is not None:
        ao = acceptor_only.astype(np.float32, copy=False)
        yf_corr = yf - alpha*d - beta*ao
    else:
        yf_corr = yf - alpha*d
    return d, yf_corr * float(g_factor)

def pick_epsilon(denom_vals, eps_abs=5.0, p_floor=1.0):
    if denom_vals.size==0:
        return float(eps_abs)
    denom_vals = denom_vals[np.isfinite(denom_vals)]
    if denom_vals.size==0:
        return float(eps_abs)
    return float(max(eps_abs, np.percentile(denom_vals, p_floor)))

def auto_minmax(vals, p_lo=1.0, p_hi=99.0):
    vals=vals[np.isfinite(vals)]
    if vals.size==0:
        return 0.0, 1.0
    lo=np.percentile(vals,p_lo); hi=np.percentile(vals,p_hi)
    if hi<=lo:
        hi=lo+1e-6
    return float(lo), float(hi)

# ===================== 그리기/저장 =====================

def draw_scalebar(ax, img_w, img_h, bar_px, bar_um, frac_margin=0.05, lw=3):
    x_start=int(img_w*(1.0 - frac_margin) - bar_px)
    y=int(img_h*(1.0 - frac_margin))
    x_end=x_start + bar_px
    ax.plot([x_start,x_end],[y,y], color='w', linewidth=lw)
    ax.text((x_start+x_end)/2, y - max(10,int(0.02*img_h)), f"{bar_um:.0f} µm",
            color='w', ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

def save_panel_intensity_ratio(int_img, ratio_img, rim_mask, out_png, px_um,
                               add_scalebar=False, sb_um=5.0,
                               cmap="turbo", vmin=0.0, vmax=0.7,
                               show_colorbar=True,
                               title_left="Intensity", title_right="FRET"):
    I=np.where(rim_mask, int_img, np.nan)
    R=np.where(rim_mask, ratio_img, np.nan)
    ivals=I[np.isfinite(I)]
    if ivals.size:
        ilo,ihi=np.percentile(ivals,1), np.percentile(ivals,99)
    else:
        ilo,ihi=0.0,1.0

    fig,axes=plt.subplots(1,2, figsize=(6,3))
    axes[0].imshow(I, vmin=ilo, vmax=ihi, cmap="gray")
    axes[0].set_title(title_left); axes[0].axis("off")
    im=axes[1].imshow(R, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(title_right); axes[1].axis("off")

    H,W=R.shape
    if add_scalebar and px_um>0:
        bar_px=int(round(sb_um/px_um)); bar_px=max(2, min(bar_px, int(0.8*W)))
        draw_scalebar(axes[0], W, H, bar_px, bar_px*px_um)
        draw_scalebar(axes[1], W, H, bar_px, bar_px*px_um)

    if show_colorbar:
        cb=fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cb.set_label("FRET ratio")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def save_png_gray(img2d, out_path, vmin=None, vmax=None, dpi=300, out_px=None):
    fig,ax=plt.subplots()
    ax.set_facecolor('black'); fig.patch.set_facecolor('black')
    ax.imshow(img2d, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    if out_px:
        fig.set_size_inches(out_px[0]/dpi, out_px[1]/dpi)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

def save_png_colormap(img2d, out_path, vmin=None, vmax=None, cmap='turbo',
                      mask=None, scalebar=None, px_um=None,
                      show_colorbar=True, dpi=300, out_px=None):
    fig,ax=plt.subplots()
    ax.set_facecolor('black'); fig.patch.set_facecolor('black')
    shown = np.array(img2d, copy=True)
    if mask is not None:
        shown = np.ma.array(shown, mask=~mask)
        cmap_obj = plt.get_cmap(cmap)
        try:
            cmap_obj = cmap_obj.copy()
        except AttributeError:
            pass
        try:
            cmap_obj.set_bad(alpha=0.0)
        except Exception:
            pass
        im=ax.imshow(shown, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    else:
        im=ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if (scalebar is not None) and (px_um is not None):
        H,W = shown.shape[:2]
        bar_px = int(round(float(scalebar) / float(px_um)))
        bar_px = max(2, min(bar_px, int(0.8*W)))
        draw_scalebar(ax, W, H, bar_px, bar_px*float(px_um))
    if show_colorbar and (vmin is not None) and (vmax is not None):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("FRET ratio")
    fig.tight_layout(pad=0)
    if out_px:
        fig.set_size_inches(out_px[0]/dpi, out_px[1]/dpi)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)

class TkTextHandler(logging.Handler):
    """logging -> Tk ScrolledText로 안전 전달"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.text_widget.after(50, self._poll)

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = "[Log format error]"
        self.queue.put(msg)

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.text_widget.configure(state="normal")
                self.text_widget.insert("end", msg + "\n")
                self.text_widget.see("end")
                self.text_widget.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.text_widget.after(50, self._poll)

class LoggerWriter:
    """print()를 logger로 리다이렉트하기 위한 file-like 객체"""
    def __init__(self, log_func):
        self.log_func = log_func
        self._buf = ""

    def write(self, msg):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip() != "":
                self.log_func(line)

    def flush(self):
        if self._buf.strip():
            self.log_func(self._buf.strip())
        self._buf = ""

# ===================== GUI =====================

CMAP_CHOICES = ["jet","turbo","viridis","plasma","magma","inferno","cividis"]
SCALE_PRESET_BASE = [
    ("0-1.5", (0.0, 1.5)),
    ("0-1.0", (0.0, 1.0)),
    ("custom", (None, None)),
]

RIM_PRESET_DEFS = [
    ("thin",   (0.45, 0.6, 1.5, None)),
    ("medium", (0.67, 0.9, 1.8, None)),
    ("thick",  (1.00, 1.2, 2.0, None)),
    ("custom", (None, None, None, None)),
]

def build_scale_presets(lang=None):
    labels = {
        "0-1.5": t("scale_preset_015", "0-1.5", lang=lang),
        "0-1.0": t("scale_preset_010", "0-1.0", lang=lang),
        "custom": t("scale_preset_custom", "Custom", lang=lang),
    }
    choices=[]; values={}
    for key, (cmin, cmax) in SCALE_PRESET_BASE:
        lbl = labels.get(key, key)
        choices.append(lbl)
        values[lbl] = (cmin, cmax)
    return choices, values

def build_rim_presets(lang=None):
    labels = {
        "thin": t("rim_preset_thin","Thin (rim 0.45 µm, ann 0.6/1.5 µm)", lang=lang),
        "medium": t("rim_preset_medium","Medium (rim 0.67 µm, ann 0.9/1.8 µm)", lang=lang),
        "thick": t("rim_preset_thick","Thick (rim 1.00 µm, ann 1.2/2.0 µm)", lang=lang),
        "custom": t("rim_preset_custom","Custom", lang=lang),
    }
    choices=[]; values={}
    for key, vals in RIM_PRESET_DEFS:
        lbl = labels.get(key, key)
        choices.append(lbl)
        values[lbl] = vals
    return choices, values



def gui_get_params(lang=None):
    lang = lang or LANG_CURRENT
    scale_choices, scale_values = build_scale_presets(lang)
    rim_choices, rim_values = build_rim_presets(lang)
    default_scale = scale_choices[0] if scale_choices else ""
    default_rim_preset = rim_choices[-1] if rim_choices else ""
    res = {
        "img_dir": None, "roi_dir": None, "out_root": "",
        "timelapse": False,
        "donor_ch": 2, "fret_ch": 3, "intensity_ch": 1,
        "ratio_mode": "FRET/Donor",
        "bg_scope": "full",
        "bg_mode": "percentile",
        "percentile": 1.0, "per_channel_p": False, "donor_p": 1.0, "fret_p": 1.0,
        "clip_neg": True, "eps_percentile": 1.0,
        "px_um": 0.223, "rim_um": 1.12,
        "annulus_on": False, "ann_in_um": 1.2, "ann_out_um": 2.5,
        "use_spectral": False, "alpha": 0.0, "beta": 0.0, "g_factor": 1.0, "aonly_ch": None,
        "scale_preset": default_scale,
        "fret_min": 0.0, "fret_max": 1.5,
        "cmap_name": "turbo", "show_colorbar": False,
        "add_scalebar": False, "scale_bar_um": 20.0,
        "out_xls": True, "out_tif": True, "out_png": True,
        "save_panel": True, "save_full": False,
        "save_crop": True, "save_crop_intensity": False, "mask_outside": True,
        "crop_fixed": True, "crop_w": 500, "crop_h": 500,
        "crop_vmin_txt": "", "crop_vmax_txt": "",
        "subset_on": False, "subset_stage": None, "subset_time": None,
        "sat_filter_on": True, "sat_threshold": 65535.0,
        "clip_ratio_on": True, "clip_ratio_max": 20.0
    }

    root=tk.Tk()
    root.title(t("title_app","Nesprin2_FRET Builder (v4)", lang=lang))
    root.resizable(False, False)

    # path vars
    img_v=tk.StringVar()
    roi_v=tk.StringVar()
    out_v=tk.StringVar(value="")
    tl_v=tk.BooleanVar(value=False)
    donor_v=tk.StringVar(value="2")
    fret_v=tk.StringVar(value="3")
    inten_v=tk.StringVar(value="1")

    # ratio
    mode_v=tk.StringVar(value="FRET/Donor")

    # BG
    scope_v=tk.StringVar(value="full")
    bgmode_v=tk.StringVar(value="percentile")
    p_v=tk.StringVar(value="1.0")
    per_ch_v=tk.BooleanVar(value=False)
    donor_p_v=tk.StringVar(value="1.0")
    fret_p_v=tk.StringVar(value="1.0")
    clip_v=tk.BooleanVar(value=True)
    eps_v=tk.StringVar(value="1.0")

    # metric
    px_v=tk.StringVar(value="0.223")
    rim_um_v=tk.StringVar(value="1.12")
    ann_on_v=tk.BooleanVar(value=False)
    ann_in_v=tk.StringVar(value="1.2")
    ann_out_v=tk.StringVar(value="2.5")
    # NEW: rim/annulus preset
    rim_preset_v = tk.StringVar(value=default_rim_preset)

    # spectral
    spec_on_v=tk.BooleanVar(value=False)
    alpha_v=tk.StringVar(value="0.0")
    beta_v=tk.StringVar(value="0.0")
    gfac_v=tk.StringVar(value="1.0")
    aonly_v=tk.StringVar(value="")

    # display
    preset_v=tk.StringVar(value=res["scale_preset"])
    cmin_v=tk.StringVar(value="0.0")
    cmax_v=tk.StringVar(value="1.5")
    cmap_v=tk.StringVar(value="turbo")
    cbar_v=tk.BooleanVar(value=False)
    sb_on_v=tk.BooleanVar(value=False)
    sb_um_v=tk.StringVar(value="20.0")

    # outputs
    out_xls=tk.BooleanVar(value=True)
    out_tif=tk.BooleanVar(value=True)
    out_png=tk.BooleanVar(value=True)
    save_panel=tk.BooleanVar(value=True)
    save_full=tk.BooleanVar(value=False)

    # crop
    save_crop=tk.BooleanVar(value=True)
    save_crop_I=tk.BooleanVar(value=False)
    mask_out=tk.BooleanVar(value=True)
    crop_fixed=tk.BooleanVar(value=True)
    crop_w_v=tk.StringVar(value="500")
    crop_h_v=tk.StringVar(value="500")
    cvmin_v=tk.StringVar(value="")
    cvmax_v=tk.StringVar(value="")

    # subset
    sub_on=tk.BooleanVar(value=False)
    sub_s=tk.StringVar(value="")
    sub_t=tk.StringVar(value="")

    # QC
    sat_on_v = tk.BooleanVar(value=True)
    sat_thr_v = tk.StringVar(value="65535")
    clip_on_v = tk.BooleanVar(value=True)
    clip_max_v = tk.StringVar(value="20.0")

    def browse_img():
        p=filedialog.askdirectory(title=t("browse_img","이미지 폴더 선택", lang=lang))
        if p:
            img_v.set(p)

    def browse_roi():
        p=filedialog.askdirectory(title=t("browse_roi","ROI 폴더 선택", lang=lang))
        if p:
            roi_v.set(p)

    def browse_out():
        p=filedialog.askdirectory(title=t("browse_out","출력 루트(선택)", lang=lang))
        if p:
            out_v.set(p)

    def on_preset_change(*_):
        cmin, cmax = scale_values.get(preset_v.get(), (None, None))
        if (cmin is not None) and (cmax is not None):
            cmin_v.set(str(cmin))
            cmax_v.set(str(cmax))

    pad={'padx':8,'pady':6}

    # 경로
    tk.Label(root,text=t("lbl_img","이미지 경로", lang=lang)).grid(row=0,column=0,sticky="w",**pad)
    fr=tk.Frame(root); fr.grid(row=0,column=1,sticky="ew",**pad)
    tk.Entry(fr,textvariable=img_v,width=52).pack(side="left")
    tk.Button(fr,text=t("btn_browse","찾기", lang=lang),width=8,command=browse_img).pack(side="left",padx=6)

    tk.Label(root,text=t("lbl_roi","ROI 경로", lang=lang)).grid(row=1,column=0,sticky="w",**pad)
    fr=tk.Frame(root); fr.grid(row=1,column=1,sticky="ew",**pad)
    tk.Entry(fr,textvariable=roi_v,width=52).pack(side="left")
    tk.Button(fr,text=t("btn_browse","찾기", lang=lang),width=8,command=browse_roi).pack(side="left",padx=6)

    tk.Label(root,text=t("lbl_out","출력 루트(선택)", lang=lang)).grid(row=2,column=0,sticky="w",**pad)
    fr=tk.Frame(root); fr.grid(row=2,column=1,sticky="ew",**pad)
    tk.Entry(fr,textvariable=out_v,width=52).pack(side="left")
    tk.Button(fr,text=t("btn_browse","찾기", lang=lang),width=8,command=browse_out).pack(side="left",padx=6)

    tk.Checkbutton(root, text=t("cb_timelapse","Timelapse 파일명(SXX_tXX_X)", lang=lang), variable=tl_v)\
        .grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(4,0))

    # 채널 + ratio
    chf=tk.LabelFrame(root,text=t("frame_channels","채널 선택 & Ratio", lang=lang))
    chf.grid(row=4,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Label(chf,text=t("lbl_donor","Donor", lang=lang)).grid(row=0,column=0,sticky="e")
    ttk.Combobox(chf,textvariable=donor_v,
                 values=["0","1","2","3","4"],width=6,state="readonly").grid(row=0,column=1,sticky="w")
    tk.Label(chf,text=t("lbl_fret","FRET", lang=lang)).grid(row=0,column=2,sticky="e")
    ttk.Combobox(chf,textvariable=fret_v,
                 values=["0","1","2","3","4"],width=6,state="readonly").grid(row=0,column=3,sticky="w")
    tk.Label(chf,text=t("lbl_intensity","Intensity표시 채널", lang=lang)).grid(row=0,column=4,sticky="e")
    ttk.Combobox(chf,textvariable=inten_v,
                 values=["0","1","2","3","4"],width=6,state="readonly").grid(row=0,column=5,sticky="w")
    fr_ratio=tk.Frame(chf); fr_ratio.grid(row=1,column=0,columnspan=6,sticky="w",pady=(4,0))
    tk.Radiobutton(fr_ratio,text=t("ratio_fd","FRET / Donor (YFRET/mTFP)", lang=lang),
                   variable=mode_v,value="FRET/Donor").pack(side="left",padx=4)
    tk.Radiobutton(fr_ratio,text=t("ratio_df","Donor / FRET (mTFP/YFRET)", lang=lang),
                   variable=mode_v,value="Donor/FRET").pack(side="left",padx=12)

    # BG
    bgf=tk.LabelFrame(root,text=t("frame_bg","BG 옵션", lang=lang))
    bgf.grid(row=5,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Label(bgf,text=t("bg_scope","Scope", lang=lang)).grid(row=0,column=0,sticky="e",padx=6,pady=4)
    ttk.Combobox(bgf,textvariable=scope_v,
                 values=["full","roi_union","annulus"],width=10,
                 state="readonly").grid(row=0,column=1,sticky="w")
    tk.Label(bgf,text=t("bg_mode","Mode", lang=lang)).grid(row=0,column=2,sticky="e",padx=6,pady=4)
    ttk.Combobox(bgf,textvariable=bgmode_v,
                 values=["percentile","hist-mode"],width=12,
                 state="readonly").grid(row=0,column=3,sticky="w")
    tk.Label(bgf,text=t("bg_percentile","percentile p(%)", lang=lang)).grid(row=1,column=0,sticky="e",padx=6,pady=4)
    tk.Entry(bgf,textvariable=p_v,width=10).grid(row=1,column=1,sticky="w")
    tk.Checkbutton(bgf,text=t("bg_per_ch","채널별 p", lang=lang),variable=per_ch_v)\
        .grid(row=1,column=2,sticky="w")
    tk.Label(bgf,text=t("bg_donor_p","Donor p", lang=lang)).grid(row=1,column=3,sticky="e")
    tk.Entry(bgf,textvariable=donor_p_v,width=8).grid(row=1,column=4,sticky="w")
    tk.Label(bgf,text=t("bg_fret_p","FRET p", lang=lang)).grid(row=1,column=5,sticky="e")
    tk.Entry(bgf,textvariable=fret_p_v,width=8).grid(row=1,column=6,sticky="w")
    tk.Checkbutton(bgf,text=t("bg_clip_neg","음수 clip", lang=lang),variable=clip_v)\
        .grid(row=2,column=0,columnspan=2,sticky="w",padx=6)
    tk.Label(bgf,text=t("bg_eps_pct","ε 퍼센타일(denom)%", lang=lang)).grid(row=2,column=2,sticky="e",padx=6)
    tk.Entry(bgf,textvariable=eps_v,width=10).grid(row=2,column=3,sticky="w")

     # ============ Metric (px, rim, annulus) WITH PRESET ============
    met=tk.LabelFrame(root, text=t("frame_metric","픽셀/림/애뉴러스(µm)", lang=lang))
    met.grid(row=6,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    # row 0: pixel size + preset combobox
    tk.Label(met,text=t("metric_px","픽셀크기 (µm/px)", lang=lang)).grid(row=0,column=0,sticky="e")
    tk.Entry(met,textvariable=px_v,width=10).grid(row=0,column=1,sticky="w")
    tk.Label(met,text=t("rim_preset_label","림/애뉴러스 프리셋", lang=lang)).grid(row=0,column=2,sticky="e")
    cb_rim_preset = ttk.Combobox(
        met, textvariable=rim_preset_v,
        values=rim_choices, width=34, state="readonly"
    )
    cb_rim_preset.grid(row=0,column=3,columnspan=3,sticky="w")

    # row 1~2: (사용자 정의 입력부)
    tk.Label(met,text=t("rim_thickness","림 두께 (µm)", lang=lang)).grid(row=1,column=0,sticky="e",padx=6)
    ent_rim_um = tk.Entry(met,textvariable=rim_um_v,width=10)
    ent_rim_um.grid(row=1,column=1,sticky="w")

    chk_ann = tk.Checkbutton(met,text=t("rim_annulus_enable","로컬 BG 애뉴러스 사용", lang=lang),variable=ann_on_v)
    chk_ann.grid(row=2,column=0,sticky="w",padx=6)

    tk.Label(met,text=t("rim_annulus_inout","내경/외경 (µm)", lang=lang)).grid(row=2,column=1,sticky="e")
    ent_ann_in  = tk.Entry(met,textvariable=ann_in_v,width=8)
    ent_ann_out = tk.Entry(met,textvariable=ann_out_v,width=8)
    ent_ann_in.grid(row=2,column=2,sticky="w")
    tk.Label(met,text="/").grid(row=2,column=3,sticky="w")
    ent_ann_out.grid(row=2,column=4,sticky="w")

    def _set_widgets_state_for_custom(is_custom: bool):
        state = "normal" if is_custom else "disabled"
        ent_rim_um.configure(state=state)
        # 프리셋이어도 애뉴러스 ON/OFF는 사용자가 항상 바꿀 수 있게 둠
        chk_ann.configure(state="normal")
        ent_ann_in.configure(state=state)
        ent_ann_out.configure(state=state)

    def apply_rim_preset(*_):
        sel = rim_preset_v.get()
        vals = rim_values.get(sel, (None, None, None, None))
        rim_um, ann_in_um, ann_out_um, ann_on = vals
        is_custom = (rim_choices and sel == rim_choices[-1])
        if not is_custom:
            # 값 자동 채움 (두께/내·외경만 세팅)
            if rim_um is not None:
                rim_um_v.set(f"{rim_um:.2f}")
            if ann_in_um is not None:
                ann_in_v.set(f"{ann_in_um:.1f}")
            if ann_out_um is not None:
                ann_out_v.set(f"{ann_out_um:.1f}")
            # ann_on 이 None이면 사용자의 현 상태를 유지 (강제 변경하지 않음)
            if ann_on is not None:
                ann_on_v.set(bool(ann_on))
            # 프리셋에서도 annulus ON/OFF 체크박스는 조작 가능
            _set_widgets_state_for_custom(False)
        else:
            # 사용자 정의: 전부 직접 입력 가능
            _set_widgets_state_for_custom(True)


    cb_rim_preset.bind("<<ComboboxSelected>>", apply_rim_preset)
    # 초기 상태 반영
    apply_rim_preset()

    # Spectral
    spf=tk.LabelFrame(root,text=t("frame_spec","스펙트럴 보정(선택)", lang=lang))
    spf.grid(row=7,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Checkbutton(spf,text=t("spec_use","스펙트럴 교정 사용", lang=lang),variable=spec_on_v)\
        .grid(row=0,column=0,sticky="w",padx=6)
    tk.Label(spf,text=t("spec_alpha","α(d→F)", lang=lang)).grid(row=0,column=1,sticky="e")
    tk.Entry(spf,textvariable=alpha_v,width=8).grid(row=0,column=2,sticky="w")
    tk.Label(spf,text=t("spec_beta","β(a→F)", lang=lang)).grid(row=0,column=3,sticky="e")
    tk.Entry(spf,textvariable=beta_v,width=8).grid(row=0,column=4,sticky="w")
    tk.Label(spf,text=t("spec_g","G", lang=lang)).grid(row=0,column=5,sticky="e")
    tk.Entry(spf,textvariable=gfac_v,width=8).grid(row=0,column=6,sticky="w")
    tk.Label(spf,text=t("spec_aonly","Acceptor-only 채널(선택)", lang=lang)).grid(row=0,column=7,sticky="e")
    ttk.Combobox(spf,textvariable=aonly_v,
                 values=["","0","1","2","3","4"],width=4,
                 state="readonly").grid(row=0,column=8,sticky="w")

    # Display
    dsf=tk.LabelFrame(root,text=t("frame_display","표시/스케일", lang=lang))
    dsf.grid(row=8,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Label(dsf,text=t("display_preset","Preset", lang=lang)).grid(row=0,column=0,sticky="e")
    dd=ttk.Combobox(dsf,textvariable=preset_v,
                    values=scale_choices,width=12,state="readonly")
    dd.grid(row=0,column=1,sticky="w")
    dd.bind("<<ComboboxSelected>>", on_preset_change)
    tk.Label(dsf,text=t("display_fret_minmax","FRET min/max", lang=lang)).grid(row=0,column=2,sticky="e")
    tk.Entry(dsf,textvariable=cmin_v,width=6).grid(row=0,column=3,sticky="w")
    tk.Label(dsf,text="~").grid(row=0,column=4,sticky="w")
    tk.Entry(dsf,textvariable=cmax_v,width=6).grid(row=0,column=5,sticky="w")
    tk.Label(dsf,text=t("display_colormap","컬러맵", lang=lang)).grid(row=0,column=6,sticky="e")
    ttk.Combobox(dsf,textvariable=cmap_v,
                 values=CMAP_CHOICES,width=10,state="readonly").grid(row=0,column=7,sticky="w")
    tk.Checkbutton(dsf,text=t("display_show_cbar","컬러바 표시", lang=lang),variable=cbar_v)\
        .grid(row=0,column=8,sticky="w",padx=6)
    tk.Checkbutton(dsf,text=t("display_scalebar","스케일바(µm)", lang=lang),variable=sb_on_v)\
        .grid(row=0,column=9,sticky="w")
    tk.Entry(dsf,textvariable=sb_um_v,width=6).grid(row=0,column=10,sticky="w")

    # Outputs
    outf=tk.LabelFrame(root,text=t("frame_output","출력", lang=lang))
    outf.grid(row=9,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Checkbutton(outf,text=t("output_xls","XLS", lang=lang),variable=out_xls)\
        .grid(row=0,column=0,sticky="w",padx=6)
    tk.Checkbutton(outf,text=t("output_tif","TIF", lang=lang),variable=out_tif)\
        .grid(row=0,column=1,sticky="w",padx=6)
    png_master_cb = tk.Checkbutton(outf,text=t("output_png","PNG", lang=lang),variable=out_png)
    png_master_cb.grid(row=0,column=2,sticky="w",padx=6)
    panel_cb = tk.Checkbutton(outf,
                              text=t("output_panel","PNG: 패널(Intensity+FRET, rim-only)", lang=lang),
                              variable=save_panel)
    panel_cb.grid(row=0,column=3,sticky="w",padx=6)
    full_cb = tk.Checkbutton(outf,
                             text=t("output_full","PNG: full image", lang=lang),
                             variable=save_full)
    full_cb.grid(row=0,column=4,sticky="w",padx=6)

    # Crop 옵션
    cropf=tk.LabelFrame(root,text=t("frame_crop","PNG: ROI Crop", lang=lang))
    cropf.grid(row=10,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Checkbutton(cropf,text=t("crop_save","저장", lang=lang),variable=save_crop)\
        .grid(row=0,column=0,sticky="w",padx=6)
    tk.Checkbutton(cropf,text=t("crop_save_intensity","Intensity rim-only도 저장", lang=lang),
                   variable=save_crop_I)\
        .grid(row=0,column=1,sticky="w",padx=6)
    tk.Checkbutton(cropf,text=t("crop_mask_outside","ROI 바깥 마스킹", lang=lang),
                   variable=mask_out)\
        .grid(row=0,column=2,sticky="w",padx=6)
    tk.Checkbutton(cropf,text=t("crop_fixed","고정 픽셀 크기", lang=lang),
                   variable=crop_fixed)\
        .grid(row=0,column=3,sticky="w",padx=6)
    tk.Label(cropf,text=t("crop_wh","W×H(px)", lang=lang)).grid(row=0,column=4,sticky="e")
    tk.Entry(cropf,textvariable=crop_w_v,width=6)\
        .grid(row=0,column=5,sticky="w")
    tk.Label(cropf,text="×").grid(row=0,column=6,sticky="w")
    tk.Entry(cropf,textvariable=crop_h_v,width=6)\
        .grid(row=0,column=7,sticky="w")
    tk.Label(cropf,text=t("crop_vmin","vmin", lang=lang)).grid(row=1,column=0,sticky="e")
    tk.Entry(cropf,textvariable=cvmin_v,width=8)\
        .grid(row=1,column=1,sticky="w")
    tk.Label(cropf,text=t("crop_vmax","vmax", lang=lang)).grid(row=1,column=2,sticky="e")
    tk.Entry(cropf,textvariable=cvmax_v,width=8)\
        .grid(row=1,column=3,sticky="w")

    # PNG master에 따른 패널/크롭/풀이미지 활성화 제어
    def on_png_toggle(*_):
        enabled = bool(out_png.get())
        state = "normal" if enabled else "disabled"
        panel_cb.configure(state=state)
        full_cb.configure(state=state)
        for child in cropf.winfo_children():
            child.configure(state=state)

    out_png.trace("w", on_png_toggle)
    on_png_toggle()

    # Subset
    subf=tk.LabelFrame(root,text=t("frame_subset","부분 추출", lang=lang))
    subf.grid(row=11,column=0,columnspan=2,sticky="we",padx=8,pady=(4,6))
    tk.Checkbutton(subf,text=t("subset_enable","활성화", lang=lang),variable=sub_on)\
        .grid(row=0,column=0,sticky="w",padx=6)
    tk.Label(subf,text=t("subset_stage","Stage", lang=lang)).grid(row=0,column=1,sticky="e")
    tk.Entry(subf,textvariable=sub_s,width=8)\
        .grid(row=0,column=2,sticky="w")
    tk.Label(subf,text=t("subset_time","Time", lang=lang)).grid(row=0,column=3,sticky="e")
    tk.Entry(subf,textvariable=sub_t,width=8)\
        .grid(row=0,column=4,sticky="w")

    # QC
    qcf = tk.LabelFrame(root, text=t("frame_qc","QC / Outlier 옵션", lang=lang))
    qcf.grid(row=12, column=0, columnspan=2,
             sticky="we", padx=8, pady=(4,6))
    tk.Checkbutton(qcf, text=t("qc_sat_filter","포화 픽셀 제거", lang=lang),
                   variable=sat_on_v)\
        .grid(row=0, column=0, sticky="w", padx=6)
    tk.Label(qcf, text=t("qc_sat_threshold","포화 임계값", lang=lang))\
        .grid(row=0, column=1, sticky="e")
    tk.Entry(qcf, textvariable=sat_thr_v, width=10)\
        .grid(row=0, column=2, sticky="w")
    tk.Checkbutton(qcf, text=t("qc_clip_ratio","고비율(Ratio) 제외", lang=lang),
                   variable=clip_on_v)\
        .grid(row=0, column=3, sticky="w", padx=(12,6))
    tk.Label(qcf, text=t("qc_clip_max","최대 비율", lang=lang))\
        .grid(row=0, column=4, sticky="e")
    tk.Entry(qcf, textvariable=clip_max_v, width=8)\
        .grid(row=0, column=5, sticky="w")

    def to_int_or_none_local(x):
        try:
            s=x.strip()
            if s=="":
                return None
            return int(float(s))
        except:
            return None

    # ---- 실행 버튼 동작
    def on_ok():
        img=img_v.get().strip()
        roi=roi_v.get().strip()
        if not img or not os.path.isdir(img):
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_invalid_img","유효한 이미지 경로를 선택하세요.", lang=lang))
            return
        if not roi or not os.path.isdir(roi):
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_invalid_roi","유효한 ROI 경로를 선택하세요.", lang=lang))
            return
        try:
            donor=int(donor_v.get())
            fret=int(fret_v.get())
            inten=int(inten_v.get())
            assert 0<=donor<=4 and 0<=fret<=4 and donor!=fret
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_channels","채널 선택을 확인하세요.", lang=lang))
            return

        res["ratio_mode"]=mode_v.get()

        # BG
        res["bg_scope"]=scope_v.get()
        res["bg_mode"]=bgmode_v.get()
        try:
            p=float(p_v.get()); assert 0.0<=p<=10.0
            d_p=float(donor_p_v.get()); f_p=float(fret_p_v.get())
            assert 0.0<=d_p<=10.0 and 0.0<=f_p<=10.0
            eps=float(eps_v.get()); assert 0.0<=eps<=10.0
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_percentile","percentile/ε 입력 오류", lang=lang))
            return

        # metric
        try:
            px=float(px_v.get()); assert 1e-4<=px<=100.0
            rim_um=float(rim_um_v.get()); assert 0.1<=rim_um<=20.0
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_metric","픽셀·림 입력 오류", lang=lang))
            return
        try:
            ann_on=bool(ann_on_v.get())
            ann_in=float(ann_in_v.get())
            ann_out=float(ann_out_v.get())
            if ann_on:
                assert 0.1<=ann_in<ann_out<=50.0
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_annulus","애뉴러스 내/외경(µm) 확인", lang=lang))
            return

        # spectral
        aonly = to_int_or_none_local(aonly_v.get())

        # display
        try:
            cmin=float(cmin_v.get())
            cmax=float(cmax_v.get())
            assert cmax>cmin
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_fret_minmax","FRET min/max 확인", lang=lang))
            return
        try:
            sb_on=bool(sb_on_v.get())
            sb_um=float(sb_um_v.get())
            if sb_on:
                assert 0.5<=sb_um<=200.0
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_scalebar","스케일바 길이 확인", lang=lang))
            return

        # crop
        try:
            cw=int(float(crop_w_v.get()))
            ch=int(float(crop_h_v.get()))
            assert 32<=cw<=8000 and 32<=ch<=8000
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_crop","Crop W/H(px) 확인", lang=lang))
            return

        # QC
        try:
            sat_on=bool(sat_on_v.get())
            sat_thr=float(sat_thr_v.get())
            clip_on=bool(clip_on_v.get())
            clip_max=float(clip_max_v.get())
            assert clip_max>0.0
        except:
            messagebox.showerror(t("err_title","오류", lang=lang), t("err_qc","QC 옵션 확인", lang=lang))
            return

        res.update(
            img_dir=img, roi_dir=roi, out_root=out_v.get().strip(),
            timelapse=bool(tl_v.get()),
            donor_ch=donor, fret_ch=fret, intensity_ch=inten,

            percentile=p, per_channel_p=bool(per_ch_v.get()),
            donor_p=d_p, fret_p=f_p, clip_neg=bool(clip_v.get()),
            eps_percentile=eps,

            px_um=px, rim_um=rim_um,
            annulus_on=ann_on, ann_in_um=ann_in, ann_out_um=ann_out,

            use_spectral=bool(spec_on_v.get()),
            alpha=float(alpha_v.get()), beta=float(beta_v.get()),
            g_factor=float(gfac_v.get()), aonly_ch=aonly,

            scale_preset=preset_v.get(),
            fret_min=cmin, fret_max=cmax,
            cmap_name=cmap_v.get(), show_colorbar=bool(cbar_v.get()),
            add_scalebar=sb_on, scale_bar_um=sb_um,

            out_xls=bool(out_xls.get()),
            out_tif=bool(out_tif.get()),
            out_png=bool(out_png.get()),
            save_panel=bool(save_panel.get()),
            save_full=bool(save_full.get()),

            save_crop=bool(save_crop.get()),
            save_crop_intensity=bool(save_crop_I.get()),
            mask_outside=bool(mask_out.get()),
            crop_fixed=bool(crop_fixed.get()),
            crop_w=cw, crop_h=ch,
            crop_vmin_txt=cvmin_v.get().strip(),
            crop_vmax_txt=cvmax_v.get().strip(),

            subset_on=bool(sub_on.get()),
            subset_stage=(to_int_or_none_local(sub_s.get())
                          if sub_on.get() else None),
            subset_time=(to_int_or_none_local(sub_t.get())
                         if sub_on.get() else None),

            sat_filter_on=sat_on,
            sat_threshold=sat_thr,
            clip_ratio_on=clip_on,
            clip_ratio_max=clip_max
        )

        # ---- 실행 중 위젯 잠그기 (Log 프레임 제외)
        def _lock_widgets(lock=True):
            for w in root.winfo_children():
                try:
                    if w is logf:
                        continue
                    w.configure(state=("disabled" if lock else "normal"))
                except:
                    pass

        _lock_widgets(True)

        logger = logging.getLogger("GUI_STATUS")
        logger.setLevel(logging.INFO)
        out_stream = LoggerWriter(logger.info)
        err_stream = LoggerWriter(logger.error)

        def _worker():
            try:
                logger.info(t("log_batch_start","배치 시작…", lang=lang))
                with contextlib.redirect_stdout(out_stream), \
                     contextlib.redirect_stderr(err_stream):
                    run_pipeline(res)
                logger.info(t("log_batch_end","모든 작업이 정상 종료되었습니다.", lang=lang))
            except Exception as e:
                logger.error(t("log_worker_exception","작업 중 예외 발생: {err}", lang=lang).format(err=e))
            finally:
                _lock_widgets(False)
                logger.info(t("log_ready","준비 완료. 파라미터 변경 후 다시 [실행]할 수 있습니다.", lang=lang))

        threading.Thread(target=_worker, daemon=True).start()

    # --- Log 창
    root.grid_columnconfigure(1, weight=1)
    logf = tk.LabelFrame(root, text=t("log_label","Log", lang=lang))
    logf.grid(row=14, column=0, columnspan=2,
              sticky="nsew", padx=8, pady=(0,8))
    txt = scrolledtext.ScrolledText(logf, wrap="word", height=14)
    txt.configure(state="disabled")
    txt.pack(fill="both", expand=True)

    logger = logging.getLogger("GUI_STATUS")
    logger.setLevel(logging.INFO)
    handler = TkTextHandler(txt)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                           datefmt="%H:%M:%S"))
    if not any(isinstance(h, TkTextHandler) for h in logger.handlers):
        logger.addHandler(handler)

    fr=tk.Frame(root)
    fr.grid(row=13,column=0,columnspan=2,pady=10)
    tk.Button(fr,text=t("btn_run","실행", lang=lang),width=12,command=on_ok)\
        .pack(side="left",padx=6)
    tk.Button(fr,text=t("btn_cancel","취소", lang=lang),width=12,
              command=lambda:(res.update(img_dir=None),
                              root.destroy()))\
        .pack(side="left",padx=6)

    root.mainloop()
    if res.get("img_dir") is None:
        raise SystemExit("Cancelled")
    return res

# ===================== 빌드/정량 =====================

def build_pairs_by_channel(files, timelapse, donor_ch, fret_ch):
    key2paths={}
    for p in files:
        base=os.path.basename(p)
        s_num,t_num,ch=parse_tokens(base,timelapse)
        if s_num is None or ch is None:
            continue
        s=fmt_stage(s_num)
        t=fmt_time(t_num) if (timelapse and t_num is not None) else None
        key=(s,t)
        d=key2paths.setdefault(key,{})
        d[ch]=p
    keys=[]
    for key,chmap in key2paths.items():
        if (donor_ch in chmap) and (fret_ch in chmap):
            keys.append(key)
    keys.sort(key=lambda k: (
        int(re.search(r'\d+', k[0]).group()),
        int(re.search(r'\d+', k[1]).group()) if k[1] else -1
    ))
    pairs=[ (key, key2paths[key][donor_ch], key2paths[key][fret_ch]) for key in keys ]
    return pairs, key2paths

def save_xls(rows_all, xls_dir, timelapse):
    df=pd.DataFrame(rows_all)
    if df.empty:
        print(t("msg_warn_no_roi_table","[주의] ROI가 없어 정량 테이블이 없습니다.", lang=LANG_CURRENT))
        return
    keep=["stage","time","roi","area_px",
          "ratio_mode",
          "ratio_mean","ratio_median","ratio_std","ratio_p5","ratio_p95",
          "ratio_FoverD_mean","ratio_DoverF_mean",
          "donor_mean","fret_mean",
          "eps","p","donor_p","fret_p","bg_scope","bg_mode","clip_neg",
          "sat_filter_on","sat_threshold","clip_ratio_on","clip_ratio_max"]
    df=df[[c for c in keep if c in df.columns]].copy()

    df["stage_idx"]=df["stage"].str.extract(r"S(\d+)",expand=False).astype(int)
    if timelapse:
        df["time_idx"]=df["time"].str.extract(r"t(\d+)",expand=False).astype(int)
    else:
        df["time_idx"]=0
    df["roi_lab"]="s"+df["stage_idx"].astype(str)+"c"+df["roi"].astype(str)

    mean_mat=df.pivot(index="time_idx",columns="roi_lab",
                      values="ratio_mean").sort_index()
    med_mat=df.pivot(index="time_idx",columns="roi_lab",
                     values="ratio_median").sort_index()

    xlsx=os.path.join(xls_dir,"nesprin2_fret_perROI.xlsx")
    csv =os.path.join(xls_dir,"nesprin2_fret_perROI.csv")

    df.to_csv(csv, index=False)
    print(t("msg_save_csv","[저장] xls/{name} (CSV)", lang=LANG_CURRENT).format(name=os.path.basename(csv)))

    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="per_ROI")
            mean_mat.to_excel(w, sheet_name="ratio_mean_matrix")
            med_mat.to_excel(w, sheet_name="ratio_median_matrix")
        print(t("msg_save_xlsx","[저장] xls/{name} (XLSX)", lang=LANG_CURRENT).format(name=os.path.basename(xlsx)))
    except ModuleNotFoundError:
        print(t("msg_warn_no_openpyxl","[경고] openpyxl 미설치 → XLSX 저장 생략, CSV만 저장되었습니다.", lang=LANG_CURRENT))

def make_name(s, t, timelapse: bool):
    return f"{s}_{t}" if (timelapse and t is not None) else s

def run_pipeline(p):
    img_dir=p["img_dir"]; roi_dir=p["roi_dir"]
    out_root=p["out_root"].strip() or os.path.join(img_dir,"RES_Nesprin2")
    timelapse=bool(p["timelapse"])

    donor_ch=int(p["donor_ch"])
    fret_ch=int(p["fret_ch"])
    intensity_ch=int(p["intensity_ch"])

    files=list_tifs(img_dir)
    pairs, key2paths=build_pairs_by_channel(files, timelapse, donor_ch, fret_ch)
    print(t("msg_info_pairs","[정보] 총 처리 대상 쌍: {count}", lang=LANG_CURRENT).format(count=len(pairs)))

    if not pairs:
        print(t("msg_no_pairs","매칭되는 (donor, fret) 채널 쌍이 없습니다.", lang=LANG_CURRENT))
        return

    # subset
    if p["subset_on"] and (p["subset_stage"] is not None):
        s_code=f"S{int(p['subset_stage']):02d}"
        t_given = (p.get("subset_time") is not None) if timelapse else False
        if not t_given:
            pairs=[pp for pp in pairs if pp[0][0]==s_code]
        else:
            t_code=f"t{int(p['subset_time']):02d}"
            pairs=[pp for pp in pairs
                   if (pp[0][0]==s_code and pp[0][1]==t_code)]
        if not pairs:
            print(t("msg_subset_none","[부분추출] 조건에 맞는 쌍이 없습니다.", lang=LANG_CURRENT))
            return

    res_root=ensure_dir(out_root)
    xls_dir =ensure_dir(os.path.join(res_root,"xls")) if p["out_xls"] else None

    tif_root=ensure_dir(os.path.join(res_root,"TIF")) if p["out_tif"] else None
    tif32_full =ensure_dir(os.path.join(tif_root,"ratio32_full")) if p["out_tif"] else None
    tif32_rim  =ensure_dir(os.path.join(tif_root,"ratio32_rim")) if p["out_tif"] else None

    png_root=ensure_dir(os.path.join(res_root,"PNG")) if p["out_png"] else None
    png_full_ratio = ensure_dir(os.path.join(png_root,"FULL_RATIO")) \
        if p["out_png"] and p["save_full"] else None
    png_full_int   = ensure_dir(os.path.join(png_root,"FULL_INT")) \
        if p["out_png"] and p["save_full"] else None
    png_panel=ensure_dir(os.path.join(png_root,"panel")) \
        if p["out_png"] else None
    png_crop_ratio = ensure_dir(os.path.join(png_root,"CROP_RATIO")) \
        if p["out_png"] and p["save_crop"] else None
    png_crop_int   = ensure_dir(os.path.join(png_root,"CROP_INT")) \
        if p["out_png"] and p["save_crop"] else None
    png_crop_int_no = ensure_dir(os.path.join(png_crop_int,"no_rim")) \
        if png_crop_int else None
    png_crop_int_r  = ensure_dir(os.path.join(png_crop_int,"rim")) \
        if png_crop_int else None

    rows_all=[]
    px_um=float(p["px_um"])
    rim_px=max(1, int(round(float(p["rim_um"])/px_um)))
    ann_on=bool(p["annulus_on"])
    ann_in_px=max(1, int(round(float(p["ann_in_um"])/px_um))) if ann_on else 0
    ann_out_px=max(ann_in_px+1,
                   int(round(float(p["ann_out_um"])/px_um))) if ann_on else 0

    scope = p["bg_scope"]
    bgmode = p["bg_mode"]
    p_glob = float(p["percentile"])
    per_ch = bool(p["per_channel_p"])
    donor_p=float(p["donor_p"])
    fret_p=float(p["fret_p"])
    clip_neg=bool(p["clip_neg"])
    eps_p = float(p["eps_percentile"])

    sat_on = bool(p["sat_filter_on"])
    sat_thr = float(p["sat_threshold"])
    clip_on = bool(p["clip_ratio_on"])
    clip_max = float(p["clip_ratio_max"])

    for (key, dpath, apath) in pairs:
        s, t_code = key
        tag = make_name(s, t_code, timelapse)
        print(t("msg_processing","[처리] {tag} ...", lang=LANG_CURRENT).format(tag=tag))
        D=read_2d(dpath)
        A=read_2d(apath)

        # saturation
        if sat_on:
            mask_sat = (D>=sat_thr) | (A>=sat_thr)
            if np.any(mask_sat):
                D = D.astype(np.float32, copy=True)
                A = A.astype(np.float32, copy=True)
                D[mask_sat] = np.nan
                A[mask_sat] = np.nan

        # intensity channel
        cand_int = swap_ch(dpath, donor_ch, intensity_ch)
        if not os.path.exists(cand_int):
            cand_int = swap_ch(apath, fret_ch, intensity_ch)
        I = read_2d(cand_int) if os.path.exists(cand_int) else D

        # acceptor-only
        Aonly=None
        aonly = to_int_or_none(p.get("aonly_ch", None))
        if aonly is not None:
            cand = swap_ch(dpath, donor_ch, aonly)
            if not os.path.exists(cand):
                cand = swap_ch(apath, fret_ch, aonly)
            if os.path.exists(cand):
                Aonly=read_2d(cand)

        polys=load_roi_polys(roi_dir, s, t_code, timelapse)
        H,W=D.shape
        if not polys:
            print(t("msg_warn_no_roi_tag","[경고] {tag}: ROI 없음 — 건너뜀", lang=LANG_CURRENT).format(tag=tag))
            continue

        union=np.zeros((H,W),dtype=bool)
        for P in polys:
            union |= rasterize_polygon(P,(H,W))

        # BG scope mask
        if scope=="full":
            scope_mask=None
        elif scope=="roi_union":
            scope_mask=union
        elif scope=="annulus":
            scope_mask=union
        else:
            scope_mask=union

        d_p = donor_p if per_ch else p_glob
        a_p = fret_p  if per_ch else p_glob

        Dbc,_=bg_correct(D, mode=bgmode,
                         p=d_p, scope_mask=scope_mask,
                         clip_neg=clip_neg)
        Abc,_=bg_correct(A, mode=bgmode,
                         p=a_p, scope_mask=scope_mask,
                         clip_neg=clip_neg)
        Aonly_bc=None
        if Aonly is not None:
            Aonly_bc,_=bg_correct(Aonly, mode=bgmode,
                                  p=p_glob, scope_mask=scope_mask,
                                  clip_neg=clip_neg)

        if bool(p["use_spectral"]):
            Dcorr, Acorr = spectral_correct(
                Abc, Dbc, acceptor_only=Aonly_bc,
                alpha=float(p["alpha"]),
                beta=float(p["beta"]),
                g_factor=float(p["g_factor"])
            )
        else:
            Dcorr, Acorr = Dbc, Abc

        denom_vals_for_eps = Dcorr[union] if p["ratio_mode"]=="FRET/Donor" else Acorr[union]
        eps = pick_epsilon(denom_vals_for_eps,
                           eps_abs=5.0, p_floor=eps_p)

        if p["ratio_mode"]=="FRET/Donor":
            numer, denom = Acorr, Dcorr
            suffix="FoverD"
            numer_alt, denom_alt = Dcorr, Acorr
            suffix_alt="DoverF"
        else:
            numer, denom = Dcorr, Acorr
            suffix="DoverF"
            numer_alt, denom_alt = Acorr, Dcorr
            suffix_alt="FoverD"

        R_full = (numer + eps) / (denom + eps)
        R_alt  = (numer_alt + eps) / (denom_alt + eps)

        if clip_on:
            R_full = np.where(R_full>clip_max, np.nan, R_full)
            R_alt  = np.where(R_alt >clip_max, np.nan, R_alt)

        rim_mask = make_inside_rim_mask(union, rim_px)

        for i,P in enumerate(polys, start=1):
            roi_full_mask = rasterize_polygon(P,(H,W))
            roi_mask = roi_full_mask & rim_mask

            R_roi = R_full.copy()
            R_roi_alt = R_alt.copy()

            if (scope=="annulus") or ann_on:
                ann = annulus_mask_from_poly(
                    P,(H,W),inner_px=ann_in_px,
                    outer_px=ann_out_px
                )
                bg_n = np.nanmedian(numer[ann]) if np.isfinite(numer[ann]).any() else 0.0
                bg_d = np.nanmedian(denom[ann]) if np.isfinite(denom[ann]).any() else 0.0
                bg_n_alt = np.nanmedian(numer_alt[ann]) if np.isfinite(numer_alt[ann]).any() else 0.0
                bg_d_alt = np.nanmedian(denom_alt[ann]) if np.isfinite(denom_alt[ann]).any() else 0.0

                numer_eff = np.maximum(numer - bg_n, 0.0) if clip_neg else (numer - bg_n)
                denom_eff = np.maximum(denom - bg_d, 0.0) if clip_neg else (denom - bg_d)
                numer_eff_alt = np.maximum(numer_alt - bg_n_alt, 0.0) if clip_neg else (numer_alt - bg_n_alt)
                denom_eff_alt = np.maximum(denom_alt - bg_d_alt, 0.0) if clip_neg else (denom_alt - bg_d_alt)

                R_roi = (numer_eff + eps) / (denom_eff + eps)
                R_roi_alt = (numer_eff_alt + eps) / (denom_eff_alt + eps)

                if clip_on:
                    R_roi = np.where(R_roi>clip_max, np.nan, R_roi)
                    R_roi_alt = np.where(R_roi_alt>clip_max, np.nan, R_roi_alt)

            vals = R_roi[roi_mask]; vals = vals[np.isfinite(vals)]
            vals_alt = R_roi_alt[roi_mask]; vals_alt = vals_alt[np.isfinite(vals_alt)]

            if vals.size==0:
                row={
                    "stage":s,"time":(t if timelapse else None),
                    "roi":i,"area_px":int(roi_mask.sum()),
                    "ratio_mean":np.nan,"ratio_median":np.nan,
                    "ratio_std":np.nan,"ratio_p5":np.nan,"ratio_p95":np.nan,
                    "ratio_FoverD_mean":(float(np.nanmean(vals_alt))
                        if p["ratio_mode"]=="DoverF"
                        else float(np.nanmean(vals))),
                    "ratio_DoverF_mean":(float(np.nanmean(vals))
                        if p["ratio_mode"]=="DoverF"
                        else float(np.nanmean(vals_alt))),
                    "donor_mean":np.nan,"fret_mean":np.nan,
                    "eps":eps,"p":p_glob,"donor_p":d_p,"fret_p":a_p,
                    "ratio_mode":p["ratio_mode"],
                    "bg_scope":scope,"bg_mode":bgmode,"clip_neg":clip_neg,
                    "sat_filter_on":sat_on,"sat_threshold":sat_thr,
                    "clip_ratio_on":clip_on,"clip_ratio_max":clip_max
                }
            else:
                row={
                    "stage":s,"time":(t if timelapse else None),
                    "roi":i,"area_px":int(roi_mask.sum()),
                    "ratio_mean":float(np.mean(vals)),
                    "ratio_median":float(np.median(vals)),
                    "ratio_std":float(np.std(vals)),
                    "ratio_p5":float(np.percentile(vals,5)),
                    "ratio_p95":float(np.percentile(vals,95)),
                    "ratio_FoverD_mean":(float(np.nanmean(vals_alt))
                        if p["ratio_mode"]=="DoverF"
                        else float(np.nanmean(vals))),
                    "ratio_DoverF_mean":(float(np.nanmean(vals))
                        if p["ratio_mode"]=="DoverF"
                        else float(np.nanmean(vals_alt))),
                    "donor_mean":float(np.nanmean(Dcorr[roi_mask])),
                    "fret_mean":float(np.nanmean(Acorr[roi_mask])),
                    "eps":eps,"p":p_glob,"donor_p":d_p,"fret_p":a_p,
                    "ratio_mode":p["ratio_mode"],
                    "bg_scope":scope,"bg_mode":bgmode,"clip_neg":clip_neg,
                    "sat_filter_on":sat_on,"sat_threshold":sat_thr,
                    "clip_ratio_on":clip_on,"clip_ratio_max":clip_max
                }
            rows_all.append(row)

            # ROI crop
            if p["out_png"] and p["save_crop"] and png_crop_ratio:
                pts = np.asarray(P)
                minx,maxx = pts[:,0].min(), pts[:,0].max()
                miny,maxy = pts[:,1].min(), pts[:,1].max()
                pad = max(10, int(0.05*max(W,H)))
                x0=max(int(minx)-pad,0); x1=min(int(maxx)+pad,W-1)
                y0=max(int(miny)-pad,0); y1=min(int(maxy)+pad,H-1)

                cropR = R_roi[y0:y1+1, x0:x1+1]
                cropI = I[y0:y1+1,  x0:x1+1]

                P2 = pts.copy()
                P2[:,0]-=x0; P2[:,1]-=y0
                crop_roi_full = rasterize_polygon(P2, cropR.shape)
                crop_rim_mask = crop_roi_full & rim_mask[y0:y1+1, x0:x1+1]

                try:
                    vmin=float(p["crop_vmin_txt"]) if p["crop_vmin_txt"]!="" else None
                except:
                    vmin=None
                try:
                    vmax=float(p["crop_vmax_txt"]) if p["crop_vmax_txt"]!="" else None
                except:
                    vmax=None
                if (vmin is None) or (vmax is None) or (vmax<=vmin):
                    if np.any(crop_rim_mask):
                        lo,hi = auto_minmax(cropR[crop_rim_mask], 1.0, 99.0)
                    else:
                        lo,hi = auto_minmax(cropR, 1.0, 99.0)
                    if vmin is None:
                        vmin=lo
                    if (vmax is None) or (vmax<=vmin):
                        vmax=hi

                out_size = (int(p["crop_w"]), int(p["crop_h"])) \
                    if p["crop_fixed"] else None

                ratio_path = os.path.join(
                    png_crop_ratio,
                    f"{tag}_roi{i}_{suffix}_rim.png"
                )
                save_png_colormap(
                    cropR, ratio_path,
                    vmin=vmin, vmax=vmax,
                    cmap=p["cmap_name"],
                    mask=crop_rim_mask,
                    scalebar=(p["add_scalebar"]
                              and p["scale_bar_um"])
                             and p["scale_bar_um"] or None,
                    px_um=px_um,
                    show_colorbar=bool(p["show_colorbar"]),
                    dpi=300, out_px=out_size
                )

                ivals = cropI[np.isfinite(cropI)]
                if ivals.size>0:
                    ilo, ihi = np.percentile(ivals,1), np.percentile(ivals,99)
                else:
                    ilo, ihi = 0.0, 1.0
                if png_crop_int_no:
                    int_path_no = os.path.join(
                        png_crop_int_no,
                        f"{tag}_roi{i}_INT_crop_full.png"
                    )
                    save_png_gray(
                        cropI, int_path_no,
                        vmin=ilo, vmax=ihi,
                        dpi=300, out_px=out_size
                    )

                if bool(p["save_crop_intensity"]) and png_crop_int_r:
                    I_vis = np.array(cropI, copy=True)
                    I_vis[~crop_rim_mask] = np.nan
                    ivals2 = I_vis[np.isfinite(I_vis)]
                    if ivals2.size>0:
                        ilo2,ihi2 = np.percentile(ivals2,1), np.percentile(ivals2,99)
                    else:
                        ilo2,ihi2 = 0.0,1.0
                    int_path_r = os.path.join(
                        png_crop_int_r,
                        f"{tag}_roi{i}_INT_rim.png"
                    )
                    save_png_gray(
                        I_vis, int_path_r,
                        vmin=ilo2, vmax=ihi2,
                        dpi=300, out_px=out_size
                    )

        # TIF
        if p["out_tif"]:
            imwrite(os.path.join(
                tif32_full,
                f"{tag}_ratio_full_{suffix}.tif"
            ), R_full.astype(np.float32))
            R_rim = np.where(rim_mask, R_full, np.nan)
            imwrite(os.path.join(
                tif32_rim,
                f"{tag}_ratio_rim_{suffix}.tif"
            ), R_rim.astype(np.float32))

        # full PNG (full image)
        if p["out_png"] and p["save_full"] and png_full_ratio:
            vals=R_full[np.isfinite(R_full)]
            lo,hi=auto_minmax(vals,1.0,99.0)
            save_png_gray(
                R_full,
                os.path.join(
                    png_full_ratio,
                    f"{tag}_ratio_full_{suffix}.png"
                ),
                vmin=lo, vmax=hi,
                dpi=300, out_px=None
            )
            ivals_full = I[np.isfinite(I)]
            if ivals_full.size>0:
                ilo_f, ihi_f = np.percentile(ivals_full,1), np.percentile(ivals_full,99)
            else:
                ilo_f, ihi_f = 0.0, 1.0
            if png_full_int:
                save_png_gray(
                    I,
                    os.path.join(
                        png_full_int,
                        f"{tag}_INT_full.png"
                    ),
                    vmin=ilo_f, vmax=ihi_f,
                    dpi=300, out_px=None
                )

        # 패널 PNG
        if p["out_png"] and p["save_panel"] and png_panel:
            save_panel_intensity_ratio(
                I, R_full, rim_mask,
                out_png=os.path.join(
                    png_panel,
                    f"{tag}_panel_{suffix}.png"
                ),
                px_um=float(p["px_um"]),
                add_scalebar=bool(p["add_scalebar"]),
                sb_um=float(p["scale_bar_um"]),
                cmap=p["cmap_name"],
                vmin=float(p["fret_min"]),
                vmax=float(p["fret_max"]),
                show_colorbar=bool(p["show_colorbar"]),
                title_left="Intensity",
                title_right="FRET"
            )

    if p["out_xls"] and xls_dir:
        save_xls(rows_all, xls_dir, timelapse=timelapse)

    print(t("msg_done_outdir","[완료] 출력 폴더: {dir}", lang=LANG_CURRENT).format(dir=res_root))

def main():
    global LANG_CURRENT
    LANG_CURRENT = pick_lang_from_argv(sys.argv[1:])
    gui_get_params(lang=LANG_CURRENT)

if __name__=="__main__":
    main()

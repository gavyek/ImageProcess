#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-based Channel/Phase Cropping — v1
(Timelapse + 포맷별 하위폴더 + Log + GUI 유지)

- Timelapse ON/OFF 지원:
  • OFF : S01_X.tif       → stage=S01, time=None, ch=X
  • ON  : S01_tXX_X.tif   → stage=S01, time=tXX, ch=X
- 부분 추출 UI: Timelapse ON일 때만 Time 입력칸 활성화
- ROI 바깥 Masking 옵션(검정)
- PNG / TIFF16(정규화) / TIFF(정규화 X, 원본값 crop) 각각 저장 옵션
- PNG/형광 색입히기(단색), 하위/상위 % 클리핑 → [0..1] → 감마(출력=입력**(1/γ))
- 포맷별 하위폴더: PNG/, TIFF16/, TIFF/
- 하단 Log 창: print 출력 및 진행상황 표시
- 실행 중: 실행 버튼 비활성화, 완료 후 다시 활성화
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

    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
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

# ----------------------- GUI & 실행 -----------------------

def gui():
    # 설정값 컨테이너 (실행 시 검증 후 업데이트)
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
        "subset_stage": None,
        "subset_time": None,
        "subset_roi": None,
        "color": "Grayscale",
        "gamma": 1.0,
        "low_cut": 1.0,
        "high_cut": 1.0,
        "mask_outside": True,
        "save_png": True,
        "save_tiff16": True,
        "save_tiff_raw": True,
    }

    root = tk.Tk()
    root.title("ROI Channel/Phase Cropper (v2.7 + Log)")
    root.resizable(False, False)

    # --- Tk 변수들 ---
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

    pad = {'padx': 8, 'pady': 6}

    # --- 폴더 선택 콜백 ---

    def browse_raw():
        p = filedialog.askdirectory(title="Raw TIF 폴더 선택")
        if p:
            raw_v.set(p)

    def browse_roi():
        p = filedialog.askdirectory(title="ROI 폴더 선택")
        if p:
            roi_v.set(p)

    def browse_out():
        p = filedialog.askdirectory(title="출력 폴더 선택")
        if p:
            out_v.set(p)

    # --- 토글 콜백 ---

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

    # --- UI 구성 ---

    tk.Label(root, text="Raw TIF 폴더").grid(row=0, column=0, sticky="w", **pad)
    fr = tk.Frame(root)
    fr.grid(row=0, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=raw_v, width=52).pack(side="left")
    tk.Button(fr, text="찾기", width=8, command=browse_raw).pack(side="left", padx=6)

    tk.Label(root, text="ROI 폴더").grid(row=1, column=0, sticky="w", **pad)
    fr = tk.Frame(root)
    fr.grid(row=1, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=roi_v, width=52).pack(side="left")
    tk.Button(fr, text="찾기", width=8, command=browse_roi).pack(side="left", padx=6)

    tk.Label(root, text="출력 폴더").grid(row=2, column=0, sticky="w", **pad)
    fr = tk.Frame(root)
    fr.grid(row=2, column=1, sticky="ew", **pad)
    tk.Entry(fr, textvariable=out_v, width=52).pack(side="left")
    tk.Button(fr, text="찾기", width=8, command=browse_out).pack(side="left", padx=6)

    tk.Checkbutton(
        root,
        text="Timelapse(시간축 있음: SXX_tXX_X)",
        variable=tl_v,
        command=toggle_subset
    ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 0))

    tk.Label(root, text="크롭할 채널 번호").grid(row=4, column=0, sticky="w", **pad)
    tk.Entry(root, textvariable=ch_v, width=8).grid(row=4, column=1, sticky="w", **pad)

    # 스케일바
    fr = tk.Frame(root)
    fr.grid(row=5, column=0, columnspan=2, sticky="w", **pad)
    tk.Checkbutton(fr, text="스케일바 추가", variable=scb_v, command=toggle_scalebar)\
        .pack(side="left")
    tk.Label(fr, text="  픽셀크기(µm/px):").pack(side="left")
    e_px = tk.Entry(fr, textvariable=px_v, width=10, state=tk.DISABLED)
    e_px.pack(side="left")
    tk.Label(fr, text="  바 길이(µm):").pack(side="left")
    e_sbl = tk.Entry(fr, textvariable=sbl_v, width=8, state=tk.DISABLED)
    e_sbl.pack(side="left")

    # PNG 옵션
    fr = tk.Frame(root)
    fr.grid(row=6, column=0, columnspan=2, sticky="w", **pad)
    tk.Label(fr, text="PNG DPI:").pack(side="left")
    tk.Entry(fr, textvariable=dpi_v, width=6).pack(side="left", padx=(4, 12))
    tk.Checkbutton(fr, text="고정 crop 픽셀", variable=fxc_v, command=toggle_fixed)\
        .pack(side="left")
    tk.Label(fr, text="  W×H(px): ").pack(side="left")
    e_cw = tk.Entry(fr, textvariable=cw_v, width=6)
    e_cw.pack(side="left")
    tk.Label(fr, text=" × ").pack(side="left")
    e_chh = tk.Entry(fr, textvariable=chh_v, width=6)
    e_chh.pack(side="left")

    # 부분 추출
    sub = tk.LabelFrame(root, text="부분 추출")
    sub.grid(row=7, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))
    tk.Checkbutton(
        sub, text="부분 추출 활성화", variable=subset_on, command=toggle_subset
    ).grid(row=0, column=0, sticky="w", padx=8, pady=6)

    tk.Label(sub, text="Stage(필수)").grid(row=1, column=0, sticky="e", padx=6)
    e_stage = tk.Entry(sub, textvariable=stage_v, width=8, state=tk.DISABLED)
    e_stage.grid(row=1, column=1, sticky="w")

    tk.Label(sub, text="Time(timelapse)").grid(row=1, column=2, sticky="e", padx=6)
    e_time = tk.Entry(sub, textvariable=time_v, width=8, state=tk.DISABLED)
    e_time.grid(row=1, column=3, sticky="w")

    tk.Label(sub, text="ROI(선택)").grid(row=1, column=4, sticky="e", padx=6)
    e_roi = tk.Entry(sub, textvariable=roi_vv, width=8, state=tk.DISABLED)
    e_roi.grid(row=1, column=5, sticky="w")

    # 색상 & 톤
    newf = tk.LabelFrame(root, text="색상 & 톤 조정")
    newf.grid(row=8, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))

    tk.Label(newf, text="색상:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    tk.OptionMenu(newf, color_v, *list(COLOR_MAP.keys()))\
        .grid(row=0, column=1, sticky="w")

    tk.Label(newf, text="감마 γ (0.0001~10)").grid(row=0, column=2, sticky="e", padx=6)
    e_gamma = tk.Entry(newf, textvariable=gamma_v, width=10)
    e_gamma.grid(row=0, column=3, sticky="w")

    tk.Label(newf, text="하위 컷 % (0~49.9)").grid(row=1, column=0, sticky="e", padx=6)
    e_low = tk.Entry(newf, textvariable=lowcut_v, width=10)
    e_low.grid(row=1, column=1, sticky="w")

    tk.Label(newf, text="상위 컷 % (0~49.9)").grid(row=1, column=2, sticky="e", padx=6)
    e_high = tk.Entry(newf, textvariable=highcut_v, width=10)
    e_high.grid(row=1, column=3, sticky="w")

    # 마스킹 & 저장 포맷
    more = tk.LabelFrame(root, text="마스킹 / 저장 포맷")
    more.grid(row=9, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 6))

    tk.Checkbutton(more, text="ROI 바깥 Masking(검정)", variable=mask_v)\
        .grid(row=0, column=0, sticky="w", padx=8)
    tk.Checkbutton(more, text="PNG 저장", variable=save_png)\
        .grid(row=0, column=1, sticky="w", padx=8)
    tk.Checkbutton(more, text="TIFF16 저장(정규화)", variable=save_tiff16)\
        .grid(row=0, column=2, sticky="w", padx=8)
    tk.Checkbutton(more, text="TIFF 저장(원본값 crop)", variable=save_tiff_raw)\
        .grid(row=0, column=3, sticky="w", padx=8)

    # 실행 / 종료 버튼
    fb = tk.Frame(root)
    fb.grid(row=10, column=0, columnspan=2, pady=10)
    btn_run = tk.Button(fb, text="실행", width=12)
    btn_run.pack(side="left", padx=6)
    btn_quit = tk.Button(fb, text="종료", width=12, command=root.destroy)
    btn_quit.pack(side="left", padx=6)

    # Log 영역
    tk.Label(root, text="Log").grid(row=11, column=0, sticky="nw", padx=8, pady=(0, 0))
    log_text = scrolledtext.ScrolledText(root, height=10, width=90)
    log_text.grid(row=11, column=1, sticky="we", padx=8, pady=(0, 8))

    # stdout 리다이렉트
    sys.stdout = TextRedirector(log_text, sys.stdout)

    running = {"flag": False}

    # --- 실행 로직 ---

    def validate_and_update_cfg():
        # 폴더
        raw = raw_v.get().strip()
        roi = roi_v.get().strip()
        out = out_v.get().strip()

        if not raw or not os.path.isdir(raw):
            messagebox.showerror("오류", "유효한 Raw 폴더를 선택하세요.")
            return False
        if not roi or not os.path.isdir(roi):
            messagebox.showerror("오류", "유효한 ROI 폴더를 선택하세요.")
            return False
        if not out or not os.path.isdir(out):
            messagebox.showerror("오류", "유효한 출력 폴더를 선택하세요.")
            return False

        # 채널
        try:
            ch = int(ch_v.get())
            assert 0 <= ch <= 99
        except:
            messagebox.showerror("오류", "채널 번호를 정수(0~99)로 입력")
            return False

        # PNG DPI
        try:
            dpi = int(float(dpi_v.get()))
            assert 50 <= dpi <= 1200
        except:
            messagebox.showerror("오류", "PNG DPI(50~1200) 범위로 입력")
            return False

        # 스케일바
        px_um = None
        sb_len = None
        if scb_v.get():
            try:
                px_um = float(px_v.get())
                assert 1e-4 <= px_um <= 100.0
            except:
                messagebox.showerror("오류", "픽셀크기(µm/px) 입력 오류")
                return False
            try:
                sb_len = float(sbl_v.get())
                assert 0.1 <= sb_len <= 1e6
            except:
                messagebox.showerror("오류", "스케일바 길이(µm) 입력 오류")
                return False

        # 고정 crop
        out_px = None
        if fxc_v.get():
            try:
                cw = int(float(cw_v.get()))
                chh = int(float(chh_v.get()))
                assert 32 <= cw <= 8000 and 32 <= chh <= 8000
                out_px = (cw, chh)
            except:
                messagebox.showerror(
                    "오류", "고정 crop 픽셀 크기(32~8000) 범위로 입력"
                )
                return False
        else:
            cw = chh = None

        # 부분 추출
        ss_on = bool(subset_on.get())
        s_val = None
        t_val = None
        r_val = None

        if ss_on:
            stxt = stage_v.get().strip()
            if not stxt:
                messagebox.showerror(
                    "오류", "부분 추출 ON일 때 Stage는 필수입니다."
                )
                return False
            try:
                s_val = int(stxt)
                assert s_val >= 0
            except:
                messagebox.showerror(
                    "오류", "Stage는 정수(예: 6 → S06)"
                )
                return False

            if tl_v.get():
                ttxt = time_v.get().strip()
                if ttxt:
                    try:
                        t_val = int(ttxt)
                        assert t_val >= 0
                    except:
                        messagebox.showerror(
                            "오류", "Timepoint는 정수(예: 3 → t03)"
                        )
                        return False

            rtxt = roi_vv.get().strip()
            if rtxt:
                try:
                    r_val = int(rtxt)
                    assert r_val >= 1
                except:
                    messagebox.showerror(
                        "오류", "ROI는 1 이상의 정수"
                    )
                    return False

        # 톤 컨트롤
        try:
            gamma = float(gamma_v.get())
            assert 1e-4 <= gamma <= 10.0
        except:
            messagebox.showerror("오류", "감마는 0.0001~10.0 범위")
            return False

        try:
            low_cut = float(lowcut_v.get())
            assert 0.0 <= low_cut < 50.0
        except:
            messagebox.showerror("오류", "하위 컷%는 0~49.9")
            return False

        try:
            high_cut = float(highcut_v.get())
            assert 0.0 <= high_cut < 50.0
        except:
            messagebox.showerror("오류", "상위 컷%는 0~49.9")
            return False

        cfg.update(
            raw_dir=raw,
            roi_dir=roi,
            out_dir=out,
            channel=ch,
            timelapse=bool(tl_v.get()),
            add_scalebar=bool(scb_v.get()),
            px_um=px_um,
            sb_len_um=sb_len,
            png_dpi=dpi,
            fixed_crop=bool(fxc_v.get()),
            crop_w=out_px[0] if out_px else cw,
            crop_h=out_px[1] if out_px else chh,
            subset_on=ss_on,
            subset_stage=s_val,
            subset_time=t_val,
            subset_roi=r_val,
            color=color_v.get(),
            gamma=gamma,
            low_cut=low_cut,
            high_cut=high_cut,
            mask_outside=bool(mask_v.get()),
            save_png=bool(save_png.get()),
            save_tiff16=bool(save_tiff16.get()),
            save_tiff_raw=bool(save_tiff_raw.get()),
        )
        return True

    def run_crop(p):
        try:
            raw_dir = p["raw_dir"]
            roi_dir = p["roi_dir"]
            out_root = p["out_dir"]

            ch_select = int(p["channel"])
            timelapse = bool(p["timelapse"])

            add_sb = bool(p["add_scalebar"])
            px_um = p["px_um"]
            sb_len = p["sb_len_um"]
            png_dpi = int(p["png_dpi"])

            fixed_crop = bool(p["fixed_crop"])
            crop_w = p["crop_w"]
            crop_h = p["crop_h"]
            out_px = (crop_w, crop_h) if fixed_crop and crop_w and crop_h else None

            color_sel = p["color"]
            gamma = float(p["gamma"])
            low_cut = float(p["low_cut"])
            high_cut = float(p["high_cut"])

            mask_outside = bool(p["mask_outside"])
            do_png = bool(p["save_png"])
            do_tif16 = bool(p["save_tiff16"])
            do_tif_raw = bool(p["save_tiff_raw"])

            ss_on = bool(p["subset_on"])
            ss_stage = p["subset_stage"]
            ss_time = p["subset_time"]
            ss_roi = p["subset_roi"]

            color_rgb = (
                np.array(COLOR_MAP[color_sel], dtype=np.float32)
                if COLOR_MAP[color_sel] is not None
                else None
            )

            files = list_tifs(raw_dir)
            fmap = build_map(files, ch_select, timelapse=timelapse)

            if not fmap:
                print(f"[알림] 채널 ch={ch_select}에 해당하는 파일을 찾지 못했습니다.")
                return

            # 부분 추출 필터
            if ss_on and (ss_stage is not None):
                s_code = f"S{int(ss_stage):02d}"
                if (not timelapse) or (ss_time is None):
                    fmap = {k: v for k, v in fmap.items() if k[0] == s_code}
                else:
                    t_code = f"t{int(ss_time):02d}"
                    fmap = {
                        k: v
                        for k, v in fmap.items()
                        if (k[0] == s_code and k[1] == t_code)
                    }
                if not fmap:
                    print(
                        f"[부분추출] 조건(Stage={s_code}, "
                        f"Time={'ALL' if (not timelapse or ss_time is None) else t_code}) 일치 없음."
                    )
                    return

            # 출력 폴더
            png_dir = os.path.join(out_root, "PNG")
            tif16_dir = os.path.join(out_root, "TIFF16")
            tif_dir = os.path.join(out_root, "TIFF")

            if do_png:
                os.makedirs(png_dir, exist_ok=True)
            if do_tif16:
                os.makedirs(tif16_dir, exist_ok=True)
            if do_tif_raw:
                os.makedirs(tif_dir, exist_ok=True)

            print(
                f"[정보] 대상 이미지: {len(fmap)}개, "
                f"채널 ch={ch_select}, 출력 루트: {out_root}"
            )

            pad_ratio = 0.05  # ROI 주변 여유 5% (최소 10px)

            for (s, t), ipath in sorted(fmap.items()):
                raw_full = imread(ipath)
                if raw_full.ndim > 2:
                    # 첫 채널만 사용 (multi-channel TIF 대응)
                    raw_full = raw_full[0, ...] if raw_full.ndim == 3 else raw_full[..., 0]

                img = raw_full.astype(np.float32, copy=False)
                H, W = img.shape

                polys = load_roi_polys(roi_dir, s, t, timelapse=timelapse)
                keytag = f"{s}{('_' + t) if (timelapse and t) else ''}"

                if not polys:
                    print(f"[스킵] ROI 미존재: {keytag}")
                    continue

                roi_indices = list(range(1, len(polys) + 1))
                if ss_on and (ss_roi is not None):
                    k = int(ss_roi)
                    if 1 <= k <= len(polys):
                        roi_indices = [k]
                    else:
                        print(f"[부분추출 경고] {keytag}: ROI {k} 없음 → 스킵")
                        roi_indices = []

                for i in roi_indices:
                    P = np.asarray(polys[i - 1])
                    minx, maxx = P[:, 0].min(), P[:, 0].max()
                    miny, maxy = P[:, 1].min(), P[:, 1].max()

                    pad = max(10, int(pad_ratio * max(W, H)))
                    x0 = max(int(minx) - pad, 0)
                    x1 = min(int(maxx) + pad, W - 1)
                    y0 = max(int(miny) - pad, 0)
                    y1 = min(int(maxy) + pad, H - 1)

                    crop_f32 = img[y0:y1 + 1, x0:x1 + 1].copy()
                    crop_raw = raw_full[y0:y1 + 1, x0:x1 + 1].copy()

                    # ROI 좌표를 crop 기준으로 이동
                    P2 = P.copy()
                    P2[:, 0] -= x0
                    P2[:, 1] -= y0
                    local_mask = rasterize_polygon(P2, crop_f32.shape)

                    # ----- 정규화 (PNG/TIFF16) -----
                    vals = crop_f32[np.isfinite(crop_f32)]
                    if vals.size == 0:
                        print(f"[경고] 유효 픽셀 없음: {keytag}_roi{i}")
                        continue

                    lo = np.percentile(vals, low_cut)
                    hi = np.percentile(vals, 100.0 - high_cut)

                    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
                        lo = float(np.nanmin(vals))
                        hi = float(np.nanmax(vals))

                    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
                        print(f"[경고] 정규화 실패: {keytag}_roi{i}")
                        continue

                    norm = np.clip((crop_f32 - lo) / (hi - lo), 0.0, 1.0)

                    # ROI 외부 마스킹
                    if mask_outside:
                        norm = norm * local_mask.astype(np.float32)

                    # 감마 적용
                    norm_gamma = np.power(norm, 1.0 / float(gamma))

                    # ----- PNG 저장 -----
                    if do_png:
                        if color_rgb is None:
                            out_img = norm_gamma
                        else:
                            rgb = color_rgb[None, None, :]
                            out_img = np.clip(norm_gamma[..., None] * rgb, 0.0, 1.0)

                        out_png = os.path.join(
                            png_dir, f"{keytag}_roi{i}_ch{ch_select}.png"
                        )
                        save_png_image(
                            out_img,
                            out_png,
                            dpi=png_dpi,
                            out_px=out_px,
                            scalebar_um=(sb_len if add_sb else None),
                            px_um=px_um,
                        )
                        print(f"[PNG]   {os.path.basename(out_png)}")

                    # ----- TIFF16 (정규화) 저장 -----
                    if do_tif16:
                        out16 = (np.clip(norm_gamma, 0, 1) * 65535).astype(np.uint16)
                        t16_path = os.path.join(
                            tif16_dir, f"{keytag}_roi{i}_ch{ch_select}.tif"
                        )
                        imwrite(t16_path, out16, photometric='minisblack')
                        print(f"[TIFF16] {os.path.basename(t16_path)}")

                    # ----- TIFF (원본값 crop) 저장 -----
                    if do_tif_raw:
                        raw_out = crop_raw.copy()
                        if mask_outside:
                            raw_out[~local_mask] = 0
                        t_path = os.path.join(
                            tif_dir, f"{keytag}_roi{i}_ch{ch_select}.tif"
                        )
                        imwrite(t_path, raw_out)
                        print(f"[TIFF]  {os.path.basename(t_path)}")

            print("\n[완료] 모든 크롭 저장 완료.")

        except Exception as e:
            import traceback
            print("\n[오류] 실행 중 예외 발생:", e)
            traceback.print_exc()

        finally:
            # 버튼 원복
            def _done():
                running["flag"] = False
                try:
                    btn_run.config(state=tk.NORMAL)
                except Exception:
                    pass
            try:
                root.after(0, _done)
            except Exception:
                pass

    def start_run():
        if running["flag"]:
            print("[알림] 이미 실행 중입니다.")
            return
        # Log 초기화
        log_text.delete("1.0", tk.END)
        if not validate_and_update_cfg():
            return
        running["flag"] = True
        btn_run.config(state=tk.DISABLED)
        print("[정보] ROI 크롭을 시작합니다...\n")
        # 설정 snapshot 전달
        p = dict(cfg)
        th = threading.Thread(target=run_crop, args=(p,), daemon=True)
        th.start()

    btn_run.config(command=start_run)

    # 초기 상태
    toggle_scalebar()
    toggle_fixed()
    toggle_subset()

    root.mainloop()

def main():
    gui()

if __name__ == "__main__":
    main()

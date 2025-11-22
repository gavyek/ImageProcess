#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphology Builder (V1.5) — Timelapse 토글 + 파일명 파서 통일 + Pillow 폴백
- ROI 다각형 기반 형태 파라미터 계산
- 파일명 규칙:
    • Timelapse ON  : SXX_TXX_X.tif  (예: S2_t00_1.tif → S02_t00)
    • Timelapse OFF : SXX_X.tif      (예: S2_1.tif    → S02)
  채널 표기: "_X", "_chX", "_cX" 모두 지원
- GUI:
    * 이미지 폴더 / ROI 폴더 / 픽셀크기(µm/px) / 채널번호 / [채널 미표기 포함] / [Timelapse]
    * [저장] Full 오버레이 이미지
    * [저장] Crop 이미지
        - [옵션] Crop 시 ROI 바깥 마스킹(0)
        - [옵션] Scale bar 추가 (기본 100 µm, 입력 가능 / Crop 저장 ON일 때만 활성화)
- 출력: RES_MOR/xls + RES_MOR/PNG(overlay_full, overlay_crop)
- ROI 규칙: <ROI_DIR>/SXX[_tXX].json  (표준 우선, 레거시(SX[_tX].json)도 자동 인식)
"""

import os, re, glob, json, math
import numpy as np
import pandas as pd

from tifffile import imread
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import tkinter as tk
from tkinter import filedialog, messagebox

# ===================== 공통 유틸 & 파서 =====================

def ensure_out(folder, name):
    out = os.path.join(folder, name); os.makedirs(out, exist_ok=True); return out

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_tifs(folder):
    exts=("*.tif","*.tiff","*.TIF","*.TIFF"); files=[]
    for ext in exts: files.extend(glob.glob(os.path.join(folder, ext)))
    uniq={}
    for p in files:
        norm=os.path.normcase(os.path.abspath(p))
        if norm not in uniq: uniq[norm]=p
    return sorted(uniq.values(), key=natural_key)

def fmt_stage(n):  # 'S01'
    return f"S{int(n):02d}"

def fmt_time(n):   # 't00'
    return f"t{int(n):02d}"

def parse_tokens(basename: str, timelapse: bool):
    """
    timelapse=True  : stage, time, channel 추출
    timelapse=False : stage, channel 추출 (time=None)
    channel: 끝 토큰(_X) 또는 _chX/_cX
    return: (stage_num or None, time_num or None, channel or None)
    """
    name = os.path.splitext(basename)[0]

    # channel
    ch = None
    m_ch = re.search(r'(?:[_-](\d+)$)|(?:[_-](?:ch|c)(\d+)$)', name, flags=re.IGNORECASE)
    if m_ch:
        g = next((g for g in m_ch.groups() if g is not None), None)
        if g is not None:
            try: ch = int(g)
            except: ch = None

    # stage
    m_s = re.search(r'(?i)S(\d+)', name)
    s_num = int(m_s.group(1)) if m_s else None

    # time
    t_num = None
    if timelapse:
        m_t = re.search(r'(?i)t(\d+)', name)
        t_num = int(m_t.group(1)) if m_t else None

    return s_num, t_num, ch

def clean_base_for_save(basename: str, timelapse: bool):
    s_num, t_num, _ = parse_tokens(basename, timelapse)
    if s_num is None:
        return os.path.splitext(basename)[0]
    if timelapse and (t_num is not None):
        return f"{fmt_stage(s_num)}_{fmt_time(t_num)}"
    return fmt_stage(s_num)

def find_json_path(roi_dir, basename_or_base, timelapse):
    """
    표준(S01[_t00].json) 우선, 없으면 레거시(S1[_t0].json) 탐색.
    basename_or_base: 원래 파일명 또는 'Sxx[_txx]' 베이스
    """
    base = os.path.splitext(basename_or_base)[0]
    # base가 'S..[_t..]'가 아니면 재구성
    s_num, t_num, _ = parse_tokens(base, timelapse)
    standard = clean_base_for_save(base, timelapse)
    candidates = [os.path.join(roi_dir, standard + ".json")]
    if s_num is not None:
        legacy = f"S{int(s_num)}"
        if timelapse and (t_num is not None):
            legacy = f"{legacy}_t{int(t_num)}"
        candidates.append(os.path.join(roi_dir, legacy + ".json"))
    for p in candidates:
        if os.path.exists(p): return p
    return candidates[0]

# ===================== TIFF 로딩(폴백 포함) =====================

def read_tiff_with_fallback(path, page=0):
    try:
        return imread(path, key=page)
    except Exception:
        with Image.open(path) as im:
            try: im.seek(page)
            except EOFError: im.seek(0)
            return np.array(im)

def read_2d(path):
    arr = read_tiff_with_fallback(path)
    if arr.ndim>2: arr = arr[...,0] if arr.ndim==3 else arr[0,...]
    return arr.astype(np.float32, copy=False)

# ===================== 채널 판별 =====================

def detect_channel(base):
    """
    파일명에서 채널 번호(int) 추출.
    예: *_1.tif, *_ch1.tif, *_c1.tiff → 1
    """
    name = os.path.splitext(base)[0]
    m = re.search(r'(?:[_-](\d+)$)|(?:[_-](?:ch|c)(\d+)$)', name, flags=re.IGNORECASE)
    if m:
        g = next((g for g in m.groups() if g is not None), None)
        if g:
            try: return int(g)
            except: return None
    return None

# ===================== ROI/도형 유틸 =====================

def load_roi_polys(roi_folder, s_code, t_code, timelapse):
    """
    s_code='S01', t_code='t00' or None
    """
    base = f"{s_code}_{t_code}" if (timelapse and t_code is not None) else s_code
    roi_json = find_json_path(roi_folder, base, timelapse)
    if not os.path.exists(roi_json): return None
    with open(roi_json,"r",encoding="utf-8") as f: data=json.load(f)
    polys=[]
    for poly in data.get("rois",[]):
        P=np.asarray(poly, dtype=float)
        if P.shape[0]>=3: polys.append(P)
    return polys if polys else None

def rasterize_polygon(poly, shape):
    H,W=shape
    path=mpath.Path(np.asarray(poly, dtype=float))
    yy,xx=np.mgrid[0:H,0:W]; pts=np.vstack((xx.ravel(),yy.ravel())).T
    return path.contains_points(pts).reshape(H,W)

def polygon_perimeter(poly):
    P=np.asarray(poly, dtype=float)
    dif=P[(np.arange(len(P))+1)%len(P)]-P
    seg=np.sqrt((dif**2).sum(axis=1))
    return float(seg.sum())

def shoelace_area(poly):
    P=np.asarray(poly, dtype=float)
    x,y=P[:,0],P[:,1]
    return float(0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))))

def convex_hull(points):
    pts=np.unique(points, axis=0)
    pts=pts[np.lexsort((pts[:,1], pts[:,0]))]
    if len(pts)<=1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0: lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0: upper.pop()
        upper.append(tuple(p))
    hull=np.array(lower[:-1]+upper[:-1], dtype=float)
    return hull

def second_moments(mask):
    ys, xs = np.nonzero(mask)
    if xs.size==0: return (np.nan, np.nan), np.array([[np.nan,np.nan],[np.nan,np.nan]])
    xc = xs.mean(); yc = ys.mean()
    x = xs - xc; y = ys - yc
    cov = np.cov(np.vstack([x,y]))
    return (yc, xc), cov

def major_minor_axes_um(mask, px_um):
    (yc,xc), cov = second_moments(mask)
    if not np.isfinite(cov).all(): return np.nan, np.nan, np.nan, np.nan
    w,v = np.linalg.eigh(cov)
    lam1, lam2 = w[1], w[0]
    angle = math.degrees(math.atan2(v[1,1], v[0,1]))
    a = 4.0*math.sqrt(max(lam1,0.0))
    b = 4.0*math.sqrt(max(lam2,0.0))
    return a*px_um, b*px_um, angle, (yc,xc)

def morphology_from_polygon(poly, shape, px_um):
    H,W = shape
    mask = rasterize_polygon(poly, (H,W))
    area_px = float(mask.sum())
    if area_px==0:
        return {
            "area_px":0,"area_um2":0,"perimeter_px":np.nan,"perimeter_um":np.nan,
            "circularity":np.nan,"roundness":np.nan,"solidity":np.nan,
            "major_um":np.nan,"minor_um":np.nan,"aspect_ratio":np.nan,"orientation_deg":np.nan,
            "centroid_x":np.nan,"centroid_y":np.nan
        }
    area_um2 = area_px*(px_um**2)
    perimeter_px = float(polygon_perimeter(poly))
    perimeter_um = perimeter_px*px_um
    hull = convex_hull(np.asarray(poly, dtype=float))
    if hull.shape[0]>=3:
        area_hull_px = shoelace_area(hull)
        solidity = float(area_px/area_hull_px) if area_hull_px>0 else np.nan
    else:
        solidity = np.nan
    major_um, minor_um, orientation_deg, (cy, cx) = major_minor_axes_um(mask, px_um)
    aspect_ratio = float(major_um/minor_um) if (np.isfinite(major_um) and np.isfinite(minor_um) and minor_um>0) else np.nan
    circularity = float(4.0*math.pi*area_px/(perimeter_px**2)) if perimeter_px>0 else np.nan
    roundness = float(4.0*area_um2/(math.pi*(major_um**2))) if (np.isfinite(major_um) and major_um>0) else np.nan
    return {
        "area_px":area_px,"area_um2":area_um2,
        "perimeter_px":perimeter_px,"perimeter_um":perimeter_um,
        "circularity":circularity,"roundness":roundness,"solidity":solidity,
        "major_um":major_um,"minor_um":minor_um,"aspect_ratio":aspect_ratio,"orientation_deg":orientation_deg,
        "centroid_x":float(cx),"centroid_y":float(cy)
    }

# ===================== 스케일바 =====================

def draw_scalebar(ax, img_w, img_h, bar_px, bar_um, frac_margin=0.05, lw=3):
    x_start = int(img_w*(1.0 - frac_margin) - bar_px)
    y = int(img_h*(1.0 - frac_margin))
    x_end = x_start + bar_px
    ax.plot([x_start, x_end], [y, y], color='w', linewidth=lw)
    ax.text((x_start+x_end)/2, y - max(10, int(0.02*img_h)), f"{bar_um:.0f} µm",
            color='w', ha='center', va='bottom', fontsize=9,
            bbox=dict(facecolor='black', alpha=0.4, pad=1, edgecolor='none'))

# ===================== GUI =====================

def gui_get_params():
    result={
        "img_dir":None,"roi_dir":None,"px_um":None,
        "sel_ch":1,"include_no_channel":False,
        "timelapse":False,  # 기본 해제
        "save_full":True,"save_crop":True,"mask_outside_roi":False,
        "add_scalebar":False,"scale_bar_um":100.0
    }
    root=tk.Tk(); root.title("Morphology Builder 설정 (v1.5)"); root.resizable(False,False)
    v_img=tk.StringVar(); v_roi=tk.StringVar(); v_px=tk.StringVar(value="")
    v_ch=tk.StringVar(value="1")
    v_inc=tk.BooleanVar(value=False)
    v_tl=tk.BooleanVar(value=False)  # Timelapse
    v_full=tk.BooleanVar(value=True)
    v_crop=tk.BooleanVar(value=True)
    v_mask=tk.BooleanVar(value=False)
    v_scalebar=tk.BooleanVar(value=False)
    v_scalebar_len=tk.StringVar(value="100")

    def b_img():
        p=filedialog.askdirectory(title="이미지 폴더 선택")
        if p: v_img.set(p)
    def b_roi():
        p=filedialog.askdirectory(title="ROI 폴더 선택")
        if p: v_roi.set(p)

    def on_toggle_crop():
        if v_crop.get():
            cb_mask.config(state=tk.NORMAL)
            cb_scalebar.config(state=tk.NORMAL)
            e_scalebar.config(state=(tk.NORMAL if v_scalebar.get() else tk.DISABLED))
        else:
            v_mask.set(False); cb_mask.config(state=tk.DISABLED)
            v_scalebar.set(False); cb_scalebar.config(state=tk.DISABLED)
            e_scalebar.config(state=tk.DISABLED)

    def on_toggle_scalebar():
        e_scalebar.config(state=(tk.NORMAL if (v_scalebar.get() and v_crop.get()) else tk.DISABLED))

    def on_ok():
        img=v_img.get().strip(); roi=v_roi.get().strip(); px=v_px.get().strip(); ch=v_ch.get().strip()
        if not img or not os.path.isdir(img): messagebox.showerror("오류","유효한 이미지 폴더를 선택하세요."); return
        if not roi or not os.path.isdir(roi): messagebox.showerror("오류","유효한 ROI 폴더를 선택하세요."); return
        try:
            px_um=float(px); assert 1e-4<=px_um<=100.0
        except Exception:
            messagebox.showerror("오류","픽셀크기(µm/px)를 실수로 입력하세요. 예: 0.108"); return
        try:
            sel_ch=int(ch); assert 1<=sel_ch<=32
        except Exception:
            messagebox.showerror("오류","채널 번호는 1~32 사이의 정수로 입력하세요."); return

        sb_um = None
        if v_crop.get() and v_scalebar.get():
            try:
                sb_um=float(v_scalebar_len.get()); assert 0.1<=sb_um<=1e6
            except Exception:
                messagebox.showerror("오류","Scale bar 길이를 µm 단위로 입력하세요. 예: 100"); return

        result.update(
            img_dir=img, roi_dir=roi, px_um=px_um, sel_ch=sel_ch,
            include_no_channel=bool(v_inc.get()),
            timelapse=bool(v_tl.get()),
            save_full=bool(v_full.get()),
            save_crop=bool(v_crop.get()),
            mask_outside_roi=bool(v_mask.get()) if v_crop.get() else False,
            add_scalebar=bool(v_scalebar.get()) if v_crop.get() else False,
            scale_bar_um=sb_um if (v_crop.get() and v_scalebar.get()) else None
        )
        root.destroy()

    def on_cancel():
        root.destroy(); raise SystemExit("작업이 취소되었습니다.")

    pad={'padx':8,'pady':6}
    tk.Label(root,text="이미지 폴더").grid(row=0,column=0,sticky="w",**pad)
    f1=tk.Frame(root); f1.grid(row=0,column=1,sticky="ew",**pad)
    tk.Entry(f1,textvariable=v_img,width=56).pack(side="left")
    tk.Button(f1,text="찾기",width=8,command=b_img).pack(side="left",padx=6)

    tk.Label(root,text="ROI 폴더").grid(row=1,column=0,sticky="w",**pad)
    f2=tk.Frame(root); f2.grid(row=1,column=1,sticky="ew",**pad)
    tk.Entry(f2,textvariable=v_roi,width=56).pack(side="left")
    tk.Button(f2,text="찾기",width=8,command=b_roi).pack(side="left",padx=6)

    tk.Label(root,text="픽셀크기 (µm/px)").grid(row=2,column=0,sticky="w",**pad)
    tk.Entry(root,textvariable=v_px,width=12).grid(row=2,column=1,sticky="w",**pad)

    tk.Label(root,text="채널 번호 (정수)").grid(row=3,column=0,sticky="w",**pad)
    tk.Entry(root,textvariable=v_ch,width=12).grid(row=3,column=1,sticky="w",**pad)

    tk.Checkbutton(root,text="채널 미표기 파일도 사용(선택 채널로 간주)",variable=v_inc)\
        .grid(row=4,column=0,columnspan=2,sticky="w",**pad)

    tk.Checkbutton(root,text="Timelapse(시간축 있음: 파일명 SXX_TXX_X)",variable=v_tl)\
        .grid(row=5,column=0,columnspan=2,sticky="w",**pad)

    # 저장 옵션
    tk.Label(root,text="이미지 저장 옵션").grid(row=6,column=0,sticky="w",**pad)
    fopt=tk.Frame(root); fopt.grid(row=6,column=1,sticky="w",**pad)
    tk.Checkbutton(fopt,text="Full 오버레이 저장",variable=v_full).pack(anchor="w")
    tk.Checkbutton(fopt,text="Crop 저장",variable=v_crop,command=on_toggle_crop).pack(anchor="w")

    cb_mask = tk.Checkbutton(fopt,text="Crop 시 ROI 바깥 마스킹(0)",variable=v_mask)
    cb_mask.pack(anchor="w")

    cb_scalebar = tk.Checkbutton(fopt,text="Scale bar 추가",variable=v_scalebar,command=on_toggle_scalebar)
    cb_scalebar.pack(anchor="w")
    row_sb = tk.Frame(fopt); row_sb.pack(anchor="w")
    tk.Label(row_sb,text="길이 (µm): ").pack(side="left")
    e_scalebar = tk.Entry(row_sb,textvariable=v_scalebar_len,width=10,state=tk.DISABLED)
    e_scalebar.pack(side="left")

    on_toggle_crop()

    f3=tk.Frame(root); f3.grid(row=7,column=0,columnspan=2,pady=10)
    tk.Button(f3,text="실행",width=12,command=on_ok).pack(side="left",padx=6)
    tk.Button(f3,text="취소",width=12,command=on_cancel).pack(side="left",padx=6)
    root.mainloop()
    return result

# ===================== Main =====================

def main():
    params=gui_get_params()
    img_dir=params["img_dir"]; roi_dir=params["roi_dir"]; px_um=float(params["px_um"])
    sel_ch=int(params["sel_ch"]); include_no_channel=bool(params["include_no_channel"])
    timelapse=bool(params["timelapse"])
    save_full=bool(params["save_full"]); save_crop=bool(params["save_crop"])
    mask_outside=bool(params["mask_outside_roi"])
    add_scalebar=bool(params["add_scalebar"]); scale_bar_um=params["scale_bar_um"]

    files_all=list_tifs(img_dir)
    if not files_all:
        print("이미지(.tif/.tiff)가 없습니다."); return

    # 채널 필터(+ timelapse용 파싱 미리 수행)
    files=[]; skipped_noch=0; skipped_mismatch=0
    meta = {}
    for p in files_all:
        base=os.path.basename(p)
        s_num, t_num, ch = parse_tokens(base, timelapse)
        # 채널 없으면 옵션에 따라 포함
        if ch is None:
            if include_no_channel:
                files.append(p); meta[p]=(s_num, t_num, None)
            else:
                skipped_noch+=1
            continue
        if ch==sel_ch:
            files.append(p); meta[p]=(s_num, t_num, ch)
        else:
            skipped_mismatch+=1

    print(f"[정보] 총 파일: {len(files_all)} | 사용: {len(files)} | 채널미표기: {skipped_noch} | 다른채널: {skipped_mismatch} | sel_ch={sel_ch} | include_no_channel={include_no_channel}")
    if not files:
        print("[중단] 사용할 파일이 없습니다. 채널 번호/파일명 표기를 확인하세요.")
        return

    # 출력 폴더
    res_root=ensure_out(img_dir,"RES_MOR")
    out_xls=ensure_out(res_root,"xls")
    png_root=ensure_out(res_root,"PNG")
    png_full=ensure_out(png_root,"overlay_full")
    png_crop=ensure_out(png_root,"overlay_crop")

    rows=[]
    for img_path in files:
        base=os.path.basename(img_path)
        s_num, t_num, _ = meta.get(img_path, parse_tokens(base, timelapse))
        if s_num is None:
            print(f"[스킵] 스테이지 파싱 실패: {base}")
            continue

        S = fmt_stage(s_num)
        t_code = fmt_time(t_num) if (timelapse and t_num is not None) else None

        img=read_2d(img_path)
        polys=load_roi_polys(roi_dir, S, t_code, timelapse)
        if not polys:
            base_for_log = f"{S}_{t_code}" if (timelapse and t_code is not None) else S
            print(f"[경고] ROI 미발견: {base_for_log}.json")
            continue

        H,W=img.shape
        # Full overlay 저장
        if save_full:
            fig,ax=plt.subplots(figsize=(8,8*H/W))
            ax.imshow(img, cmap="gray"); ax.set_axis_off()
            for i,poly in enumerate(polys,1):
                P=np.asarray(poly)
                ax.plot(np.r_[P[:,0],P[0,0]], np.r_[P[:,1],P[0,1]], lw=1.5, color='cyan')
                cx = P[:,0].mean(); cy=P[:,1].mean()
                ax.text(cx,cy, str(i), fontsize=10, ha="center", va="center",
                        bbox=dict(facecolor='black', alpha=0.3, pad=1, edgecolor='none'), color='w')
            tag = f"{S}_{t_code}" if (timelapse and t_code is not None) else S
            full_png=os.path.join(png_full, f"{tag}_overlay_ch{sel_ch}.png")
            fig.tight_layout(pad=0); fig.savefig(full_png, dpi=200); plt.close(fig)

        # 측정 + crop 저장
        for i,poly in enumerate(polys,1):
            met=morphology_from_polygon(poly, (H,W), px_um)
            met.update({
                "stage":S, "time":(t_code if timelapse else None),
                "roi":i,"px_um":px_um,"img":base,"channel":sel_ch
            })
            rows.append(met)

            if not save_crop:
                continue

            # crop bbox
            P=np.asarray(poly); minx,maxx=P[:,0].min(),P[:,0].max(); miny,maxy=P[:,1].min(),P[:,1].max()
            pad= max(10, int(0.05*max(W,H)))
            x0=max(int(minx)-pad,0); x1=min(int(maxx)+pad,W-1)
            y0=max(int(miny)-pad,0); y1=min(int(maxy)+pad,H-1)

            crop = img[y0:y1+1, x0:x1+1]
            P2=P.copy(); P2[:,0]-=x0; P2[:,1]-=y0

            if mask_outside:
                mask_full = rasterize_polygon(P2, crop.shape)
                crop = crop * mask_full.astype(crop.dtype)

            fig,ax=plt.subplots(figsize=(5,5*(y1-y0+1)/(x1-x0+1)))
            ax.imshow(crop, cmap="gray"); ax.set_axis_off()
            ax.plot(np.r_[P2[:,0],P2[0,0]], np.r_[P2[:,1],P2[0,1]], lw=1.5, color='cyan')
            ax.set_title(f"{S}{'' if (not timelapse or t_code is None) else '_'+t_code}  ROI#{i}  ch{sel_ch}  AR={met['aspect_ratio']:.2f}  Circ={met['circularity']:.3f}", fontsize=9)

            if add_scalebar and (scale_bar_um is not None):
                crop_h, crop_w = crop.shape
                bar_px = int(round(float(scale_bar_um) / px_um))
                max_bar = int(0.8 * crop_w)
                if bar_px > max_bar and max_bar > 1: bar_px = max_bar
                if bar_px < 2: bar_px = 2
                shown_um = bar_px * px_um
                draw_scalebar(ax, crop_w, crop_h, bar_px, shown_um, frac_margin=0.05, lw=3)

            tag = f"{S}_{t_code}" if (timelapse and t_code is not None) else S
            crop_png=os.path.join(png_crop, f"{tag}_roi{i}_ch{sel_ch}.png")
            fig.tight_layout(pad=0.1); fig.savefig(crop_png, dpi=220); plt.close(fig)

    if not rows:
        print("[주의] 결과가 없습니다. 파일명 규칙·채널 표기·ROI json을 확인하세요."); return

    cols = ["stage","time","roi","img","channel","px_um",
            "area_px","area_um2","perimeter_px","perimeter_um",
            "major_um","minor_um","aspect_ratio","orientation_deg",
            "circularity","roundness","solidity",
            "centroid_x","centroid_y"]
    df=pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns: df[c]=np.nan
    df=df[cols].sort_values(["stage","time","roi"], na_position="last").reset_index(drop=True)

    xlsx=os.path.join(out_xls,"morphology_perROI.xlsx")
    csv =os.path.join(out_xls,"morphology_perROI.csv")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="per_ROI")
    df.to_csv(csv, index=False)
    print(f"[저장] {xlsx}")
    print(f"[저장] {csv}")

if __name__=="__main__":
    main()

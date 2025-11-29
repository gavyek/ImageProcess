#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FA Analyzer v1 (Final Release)
- Integrated Log & Status Window
- Subset Extraction (Stage/Cell Filtering)
- Zoom Control (Padding)
- Custom Export Styles (ROI Line, Colormap, Resolution)
"""

import os
import glob
import json
import traceback
import re
import numpy as np
import pandas as pd
from pathlib import Path
import datetime

# GUI Framework
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Image Processing
from tifffile import imread
from skimage import morphology, measure
from skimage.draw import polygon

# Plotting
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import path as mpl_path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# Helper: MATLAB v7.3 HDF5 Parser (Optional)
# =========================================================
try:
    import h5py
except ImportError:
    h5py = None

def extract_matlab_boundaries(mat_path):
    if h5py is None: return None
    boundaries = []
    try:
        with h5py.File(mat_path, 'r') as f:
            if 'bdokcc' not in f: return None
            cell_refs = f['bdokcc']
            for i in range(len(cell_refs)):
                try:
                    ref = cell_refs[i][0]
                    fa_refs_obj = f[ref]
                    for j in range(len(fa_refs_obj)):
                        try:
                            fa_ref = fa_refs_obj[j][0]
                            coord_data = f[fa_ref][:]
                            poly = np.array(coord_data)
                            if poly.shape[0] == 2 and poly.shape[1] > 2: poly = poly.T
                            if poly.shape[1] == 2 and poly.shape[0] > 2: boundaries.append(poly)
                        except: continue
                except: continue
    except: return None
    return boundaries

def find_matching_mat(mat_dir, s_tag):
    if not os.path.exists(mat_dir): return None
    p = os.path.join(mat_dir, f"{s_tag}.mat")
    if os.path.exists(p): return p
    p = os.path.join(mat_dir, f"BNDb_{s_tag}.mat")
    if os.path.exists(p): return p
    try:
        num = int(re.search(r'\d+', s_tag).group())
        candidates = glob.glob(os.path.join(mat_dir, "*.mat"))
        for cand in candidates:
            base = os.path.basename(cand)
            if f"s{num}.mat" in base or f"s{num}_" in base: return cand
    except: pass
    return None

# =========================================================
# Core Logic: FA Segmentation & Intensity
# =========================================================
def analyze_fa_crop(image_crop, roi_mask_crop, config, global_stats):
    img_float = image_crop.astype(np.float32)
    
    # Handle stats unpacking
    bg_val_passed = None
    if len(global_stats) == 3:
        m, s, bg_val_passed = global_stats
    else:
        m, s = global_stats
    
    # Background Calculation
    if bg_val_passed is not None:
        bg_val = bg_val_passed
    else:
        bg_val = np.percentile(img_float, 1.0)

    alpha = config['alpha']
    threshold_val = m + alpha * s
    
    bw = img_float > threshold_val
    bw = bw & roi_mask_crop
    
    min_px = config['min_px']
    if min_px > 0:
        bw = morphology.remove_small_objects(bw, min_size=min_px)
        
    close_rad = config['close_radius']
    if close_rad > 0:
        selem = morphology.disk(close_rad)
        bw = morphology.binary_closing(bw, selem)
        
    labeled_img = measure.label(bw)
    props = measure.regionprops(labeled_img, intensity_image=img_float)
    
    max_px = config['max_px']
    subtract_bg = config.get('subtract_bg', True)
    
    results = {'OK': [], 'Large': [], 'Small': []}
    
    for prop in props:
        area = prop.area
        contours = measure.find_contours(labeled_img == prop.label, 0.5)
        if not contours: continue
        contour = contours[0]
        
        category = 'OK'
        if area < min_px: category = 'Small'
        elif area > max_px: category = 'Large'
        
        mean_raw = prop.mean_intensity
        mean_corr = max(0, mean_raw - bg_val) if subtract_bg else mean_raw
        
        int_den_raw = mean_raw * area
        int_den_corr = mean_corr * area
        
        fa_data = {
            'label': prop.label,
            'area': area,
            'contour': contour,
            'centroid': prop.centroid,
            'mean_int_raw': mean_raw,
            'mean_int_corr': mean_corr,
            'int_den_raw': int_den_raw,   
            'int_den_corr': int_den_corr, 
            'bg_level': bg_val
        }
        results[category].append(fa_data)
        
    return results, threshold_val, bw, labeled_img

# =========================================================
# Export Helper
# =========================================================
def draw_scalebar(ax, img_w, img_h, bar_px, bar_um, 
                  show_text=True, font_size=10, color='white'):
    """Draws a scale bar at bottom right"""
    margin_x = int(img_w * 0.05)
    margin_y = int(img_h * 0.05)
    
    # Position: Bottom Right
    x_end = img_w - margin_x
    x_start = x_end - bar_px
    y = img_h - margin_y
    
    # Draw Line
    ax.plot([x_start, x_end], [y, y], color=color, linewidth=3)
    
    # Draw Text
    if show_text:
        text_y = y - max(10, int(0.02 * img_h)) # Slightly above line
        ax.text(
            (x_start + x_end) / 2,
            text_y,
            f"{int(bar_um)} µm",
            color=color,
            ha='center',
            va='bottom',
            fontsize=font_size,
            fontweight='bold'
        )

def save_crop_colormap(img_crop, mask, roi_poly_crop, out_path, 
                       cmap_name='jet', show_cbar=True, mode='FA Only',
                       vmin=None, vmax=None,
                       sb_on=False, sb_len_um=20, sb_text=True, sb_font=10, px_size=0.112,
                       out_w=500, out_h=500, out_dpi=600,
                       roi_lw=0.5, roi_color='gray'):
    
    # Calculate Figure Size in Inches for requested Pixels
    fig_w_in = out_w / out_dpi
    fig_h_in = out_h / out_dpi
    
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=out_dpi)
    fig.patch.set_facecolor('black')
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    
    masked_data = np.ma.array(img_crop, mask=~mask)
    
    # Auto Scale
    if vmin is None or vmax is None:
        valid_data = img_crop[mask]
        if valid_data.size > 0:
            auto_vmin, auto_vmax = np.percentile(valid_data, 1), np.percentile(valid_data, 99)
        else:
            auto_vmin, auto_vmax = 0, 1
        
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax

    # Set Colormap
    cmap_lower = cmap_name.lower()
    if cmap_lower in ['blue', 'cyan', 'green', 'yellow', 'red', 'magenta']:
        cmap = LinearSegmentedColormap.from_list(f"custom_{cmap_lower}", ["black", cmap_lower])
    elif cmap_lower == 'grayscale':
        cmap = plt.get_cmap('gray').copy()
    else:
        try:
            cmap = plt.get_cmap(cmap_name).copy()
        except:
            cmap = plt.get_cmap('jet').copy()
        
    cmap.set_bad(color='black') 
    
    # Display Image (Centered)
    im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Overlay ROI with custom style
    ax.plot(roi_poly_crop[:, 0], roi_poly_crop[:, 1], 
            linestyle='--', linewidth=roi_lw, color=roi_color, alpha=0.8)

    # Scale Bar
    if sb_on and px_size > 0:
        bar_px = sb_len_um / px_size
        h, w = img_crop.shape
        if bar_px < w:
            draw_scalebar(ax, w, h, bar_px, sb_len_um, show_text=sb_text, font_size=sb_font)

    ax.axis('off')
    
    if show_cbar:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="3%", height="40%", loc='center right', borderpad=1)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)
        cbar.outline.set_edgecolor('white')

    fig.savefig(out_path, facecolor='black', edgecolor='none', dpi=out_dpi)
    plt.close(fig)

# =========================================================
# Main GUI Class
# =========================================================
class FAAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FA Analyzer v1 (Final)")
        self.root.geometry("1450x980")
        
        self.root.bind('<s>', lambda e: self._save_current_view())
        self.root.bind('<S>', lambda e: self._save_current_view())
        
        self.img_dir = tk.StringVar()
        self.roi_dir = tk.StringVar()
        self.mat_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.channel_var = tk.StringVar(value="1")
        
        self.pixel_size_mode = tk.StringVar(value="0.112")
        self.pixel_size_custom = tk.StringVar(value="")
        
        self.alpha_var = tk.DoubleVar(value=2.0)
        self.min_area_um_var = tk.DoubleVar(value=1.5)
        self.max_area_um_var = tk.DoubleVar(value=30.0)
        self.close_rad_var = tk.IntVar(value=1)
        self.subtract_bg_var = tk.BooleanVar(value=True) 
        
        self.auto_zoom_var = tk.BooleanVar(value=True)
        self.show_mat_var = tk.BooleanVar(value=True)
        self.enable_matlab_var = tk.BooleanVar(value=False)
        
        self.save_ok_only_var = tk.BooleanVar(value=False)

        self.global_params = {
            'alpha': 2.0,
            'min_area_um': 1.5,
            'max_area_um': 30.0,
            'close_radius': 1,
            'subtract_bg': True
        }
        
        self.file_list = []
        self.current_idx = -1
        self.current_rois = []
        self.current_mat_polys = None
        self.cell_settings = {} 
        self.selected_cell_idx = None
        
        self.current_img = None
        self.current_stats = (0, 0, 0) 
        
        self._setup_ui()
        
    def _setup_ui(self):
        top_frame = ttk.LabelFrame(self.root, text="Configuration")
        top_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        grid_opts = {'padx': 5, 'pady': 2, 'sticky': 'ew'}
        
        ttk.Label(top_frame, text="Image Folder:").grid(row=0, column=0, sticky='e')
        ttk.Entry(top_frame, textvariable=self.img_dir, width=60).grid(row=0, column=1, **grid_opts)
        ttk.Button(top_frame, text="Browse", command=lambda: self._browse_dir(self.img_dir, is_img=True)).grid(row=0, column=2)
        
        ttk.Label(top_frame, text="ROI Folder:").grid(row=1, column=0, sticky='e')
        ttk.Entry(top_frame, textvariable=self.roi_dir, width=60).grid(row=1, column=1, **grid_opts)
        ttk.Button(top_frame, text="Browse", command=lambda: self._browse_dir(self.roi_dir)).grid(row=1, column=2)
        
        ttk.Label(top_frame, text="Output Folder:").grid(row=2, column=0, sticky='e')
        ttk.Entry(top_frame, textvariable=self.out_dir, width=60).grid(row=2, column=1, **grid_opts)
        ttk.Button(top_frame, text="Browse", command=lambda: self._browse_dir(self.out_dir)).grid(row=2, column=2)
        
        self.chk_matlab = ttk.Checkbutton(top_frame, text="Enable Legacy MATLAB Import", variable=self.enable_matlab_var, command=self._toggle_matlab_ui)
        self.chk_matlab.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        self.lbl_mat = ttk.Label(top_frame, text="MATLAB Data (Optional):")
        self.ent_mat = ttk.Entry(top_frame, textvariable=self.mat_dir, width=60)
        self.btn_mat = ttk.Button(top_frame, text="Browse", command=lambda: self._browse_dir(self.mat_dir))
        
        param_frame = ttk.Frame(top_frame)
        param_frame.grid(row=5, column=0, columnspan=3, sticky='ew', pady=5)
        
        ttk.Label(param_frame, text="Target Ch:").pack(side="left", padx=(0,5))
        ttk.Entry(param_frame, textvariable=self.channel_var, width=5).pack(side="left", padx=(0,20))
        
        ttk.Label(param_frame, text="Pixel Size:").pack(side="left", padx=(0,5))
        self.combo_px = ttk.Combobox(param_frame, textvariable=self.pixel_size_mode, 
                                     values=["0.112", "0.223", "custom"], state="readonly", width=10)
        self.combo_px.pack(side="left", padx=5)
        self.combo_px.bind("<<ComboboxSelected>>", self._on_px_mode_change)
        
        self.entry_px_custom = ttk.Entry(param_frame, textvariable=self.pixel_size_custom, width=10, state="disabled")
        self.entry_px_custom.pack(side="left", padx=5)
        
        ttk.Button(param_frame, text="Load Files", command=self._load_file_list).pack(side="left", padx=20)

        # Main Panes
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True, padx=10, pady=5)
        
        left_frame = ttk.Frame(main_pane, width=200)
        self.listbox = tk.Listbox(left_frame, selectmode=tk.SINGLE)
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind('<<ListboxSelect>>', self._on_select_file)
        
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        main_pane.add(left_frame, weight=1)
        
        center_frame = ttk.Frame(main_pane)
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        toolbar = NavigationToolbar2Tk(self.canvas, center_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        main_pane.add(center_frame, weight=4)
        
        # Right Panel
        right_frame = ttk.LabelFrame(main_pane, text="Parameters")
        
        mode_frame = ttk.Frame(right_frame)
        mode_frame.pack(fill='x', padx=10, pady=5)
        self.lbl_mode = ttk.Label(mode_frame, text="[Global Settings Mode]", foreground="black", font=("Arial", 10, "bold"))
        self.lbl_mode.pack(anchor='w')
        
        ttk.Checkbutton(mode_frame, text="Auto-Zoom to Selected Cell", variable=self.auto_zoom_var, command=self._update_plot).pack(anchor='w', pady=(2,0))
        self.chk_show_mat = ttk.Checkbutton(mode_frame, text="Show Legacy MATLAB Data", variable=self.show_mat_var, command=self._update_plot)
        
        def add_control(label, var, from_, to_, res):
            frame = ttk.Frame(right_frame)
            frame.pack(fill='x', padx=10, pady=5)
            ttk.Label(frame, text=label).pack(anchor='w')
            inner = ttk.Frame(frame)
            inner.pack(fill='x')
            ent = ttk.Entry(inner, textvariable=var, width=6)
            ent.pack(side='left', padx=(0, 5))
            ent.bind('<Return>', lambda e: self._on_param_update(None))
            scl = tk.Scale(inner, variable=var, from_=from_, to=to_, resolution=res, orient="horizontal", showvalue=0)
            scl.pack(side='left', fill='x', expand=True)
            scl.config(command=self._on_param_update) 
            return ent, scl

        add_control("Threshold Alpha", self.alpha_var, 0.1, 10.0, 0.1)
        add_control("Min Area (um^2)", self.min_area_um_var, 0.1, 10.0, 0.1)
        add_control("Max Area (um^2)", self.max_area_um_var, 10.0, 500.0, 1.0)
        add_control("Closing Radius (px)", self.close_rad_var, 0, 5, 1)

        int_frame = ttk.LabelFrame(right_frame, text="Intensity")
        int_frame.pack(fill='x', padx=10, pady=10)
        ttk.Checkbutton(int_frame, text="Subtract Background (Bottom 1%)", variable=self.subtract_bg_var, command=self._on_param_update).pack(anchor='w', padx=5, pady=5)

        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(pady=10, anchor='w', padx=10)
        ttk.Label(legend_frame, text="■ OK (Green)", foreground="green").pack(anchor='w')
        ttk.Label(legend_frame, text="■ Large (Red)", foreground="red").pack(anchor='w')
        ttk.Label(legend_frame, text="■ Small (Blue)", foreground="blue").pack(anchor='w')
        self.lbl_legend_mat = ttk.Label(legend_frame, text="-- MATLAB (Magenta)", foreground="magenta")

        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        save_opt_frame = ttk.Frame(right_frame)
        save_opt_frame.pack(fill='x', padx=10, pady=5)
        ttk.Checkbutton(save_opt_frame, text="Save 'OK' Category Only", variable=self.save_ok_only_var).pack(anchor='w')

        ttk.Button(right_frame, text="PROCESS CURRENT FILE ONLY", command=self._run_single_process).pack(fill='x', padx=10, pady=5)
        ttk.Button(right_frame, text="Quick Snapshot (Auto-Save) [Key: S]", command=self._save_current_view).pack(fill='x', padx=10, pady=5)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        ttk.Label(right_frame, text="Batch Tools").pack(anchor='w', padx=10)
        
        btn_merge = ttk.Button(right_frame, text="Merge All CSVs -> Excel Report", command=self._merge_all_csvs_report)
        btn_merge.pack(fill='x', padx=10, pady=5)
        
        btn_export_crops = ttk.Button(right_frame, text="Export All Crop Images (Enhanced)", command=self._open_export_dialog)
        btn_export_crops.pack(fill='x', padx=10, pady=5)
        
        main_pane.add(right_frame, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        self.statusbar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side="bottom", fill="x")
        
        self._toggle_matlab_ui()

    def _toggle_matlab_ui(self):
        if self.enable_matlab_var.get():
            self.lbl_mat.grid(row=4, column=0, sticky='e')
            self.ent_mat.grid(row=4, column=1, padx=5, pady=2, sticky='ew')
            self.btn_mat.grid(row=4, column=2)
            self.chk_show_mat.pack(anchor='w', pady=(2,0))
            self.lbl_legend_mat.pack(anchor='w')
        else:
            self.lbl_mat.grid_remove()
            self.ent_mat.grid_remove()
            self.btn_mat.grid_remove()
            self.chk_show_mat.pack_forget()
            self.lbl_legend_mat.pack_forget()

    def _browse_dir(self, var, is_img=False):
        d = filedialog.askdirectory()
        if d:
            var.set(d)
            if is_img and not self.out_dir.get():
                self.out_dir.set(os.path.join(d, "BND_FA"))

    def _on_px_mode_change(self, event):
        if self.pixel_size_mode.get() == "custom":
            self.entry_px_custom.config(state="normal")
        else:
            self.entry_px_custom.config(state="disabled")

    def _get_pixel_size(self):
        mode = self.pixel_size_mode.get()
        if mode == "custom":
            try: return float(self.pixel_size_custom.get())
            except: return 0.112
        return float(mode)

    def _gui_to_params(self):
        try:
            return {
                'alpha': self.alpha_var.get(),
                'min_area_um': self.min_area_um_var.get(),
                'max_area_um': self.max_area_um_var.get(),
                'close_radius': self.close_rad_var.get(),
                'subtract_bg': self.subtract_bg_var.get()
            }
        except: return None

    def _params_to_gui(self, params):
        self.alpha_var.set(params['alpha'])
        self.min_area_um_var.set(params['min_area_um'])
        self.max_area_um_var.set(params['max_area_um'])
        self.close_rad_var.set(params['close_radius'])
        self.subtract_bg_var.set(params.get('subtract_bg', True))

    def _convert_um_to_px_config(self, params):
        px_size = self._get_pixel_size()
        return {
            'alpha': params['alpha'],
            'min_px': params['min_area_um'] / (px_size**2),
            'max_px': params['max_area_um'] / (px_size**2),
            'close_radius': params['close_radius'],
            'subtract_bg': params.get('subtract_bg', True)
        }

    def _load_file_list(self):
        img_dir = self.img_dir.get()
        roi_dir = self.roi_dir.get()
        target_ch = self.channel_var.get()
        
        if not os.path.isdir(img_dir) or not os.path.isdir(roi_dir):
            messagebox.showerror("Error", "Check input directories.")
            return

        self.file_list = []
        self.listbox.delete(0, tk.END)
        
        all_imgs = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        count = 0
        for img_path in all_imgs:
            fname = os.path.basename(img_path)
            if f"_{target_ch}.tif" in fname or f"_{target_ch}.TIF" in fname:
                try:
                    parts = fname.split('_')
                    s_tag = parts[0]
                    json_name = f"{s_tag}.json"
                    json_path = os.path.join(roi_dir, json_name)
                    if os.path.exists(json_path):
                        self.file_list.append((img_path, json_path, s_tag))
                        self.listbox.insert(tk.END, f"{s_tag}")
                        count += 1
                except: pass
        
        if count == 0:
            messagebox.showinfo("Result", f"No matching pairs found for Ch {target_ch}.")
        else:
            self.status_var.set(f"Loaded {count} files.")
            self.listbox.select_set(0)
            self._on_select_file(None)

    def _restore_settings_from_individual_csv(self, s_tag):
        out_root = self.out_dir.get()
        if not out_root: return
        
        indiv_path = os.path.join(out_root, "individual_results", f"{s_tag}_results.csv")
        if not os.path.exists(indiv_path): return
        
        try:
            df = pd.read_csv(indiv_path)
            if df.empty: return
            
            restored_count = 0
            unique_cells = df.drop_duplicates(subset=['Cell_ID'])
            
            for _, row in unique_cells.iterrows():
                cell_idx = int(row['Cell_ID']) - 1
                if 'Used_Alpha' in row:
                    alpha = row['Used_Alpha']
                    min_a = row.get('Min_Area_Setting', 1.5)
                    max_a = row.get('Max_Area_Setting', 30.0)
                    close_r = row.get('Close_Radius_Setting', 1)
                    sub_bg = row.get('Subtract_BG_Setting', True)
                    
                    self.cell_settings[cell_idx] = {
                        'alpha': float(alpha),
                        'min_area_um': float(min_a),
                        'max_area_um': float(max_a),
                        'close_radius': int(close_r),
                        'subtract_bg': bool(sub_bg)
                    }
                    restored_count += 1
            
            if restored_count > 0:
                self.status_var.set(f"Restored settings for {restored_count} cells from {s_tag}_results.csv")
                
        except Exception as e:
            print(f"Error loading CSV settings: {e}")

    def _on_select_file(self, event):
        sel = self.listbox.curselection()
        if not sel: return
        new_idx = sel[0]
        
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            
            img_path, json_path, s_tag = self.file_list[self.current_idx]
            try:
                self.current_img = imread(img_path, aszarr=False) 
                if self.current_img.ndim > 2: self.current_img = self.current_img[:,:,0]
                
                img_float = self.current_img.astype(np.float32)
                
                sample = img_float[::10, ::10]
                glob_bg = np.percentile(sample, 1.0)
                self.current_stats = (np.nanmean(img_float), np.nanstd(img_float), glob_bg)
                
                with open(json_path, 'r') as f:
                    roi_data = json.load(f)
                rois = []
                for item in roi_data.get('rois', []):
                    pts = item if isinstance(item, list) else item.get('rois', item)
                    if isinstance(item, list): pts = item
                    if pts: rois.append(np.array(pts))
                self.current_rois = rois
                
            except Exception as e:
                print(f"Error loading image: {e}")
                self.current_img = None
                return

            self.cell_settings = {}
            self.selected_cell_idx = None
            self.current_mat_polys = None
            self.lbl_mode.config(text="[Global Settings Mode]", foreground="black")
            self._params_to_gui(self.global_params)
            
            self._restore_settings_from_individual_csv(s_tag)
            
            if self.enable_matlab_var.get():
                mat_dir = self.mat_dir.get()
                if mat_dir and os.path.isdir(mat_dir):
                    mat_path = find_matching_mat(mat_dir, s_tag)
                    if mat_path:
                        self.current_mat_polys = extract_matlab_boundaries(mat_path)
        
        self._update_plot()

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax: return
        if not self.current_rois: return
        
        click_point = (event.xdata, event.ydata)
        clicked_idx = None
        for i, poly in enumerate(self.current_rois):
            path = mpl_path.Path(poly)
            if path.contains_point(click_point):
                clicked_idx = i
                break
        
        self.selected_cell_idx = clicked_idx
        
        if clicked_idx is not None:
            self.lbl_mode.config(text=f"Cell #{clicked_idx+1} Selected", foreground="blue")
            if clicked_idx in self.cell_settings:
                params = self.cell_settings[clicked_idx]
            else:
                params = self.global_params.copy()
            self._params_to_gui(params)
        else:
            self.lbl_mode.config(text="[Global Settings Mode]", foreground="black")
            self._params_to_gui(self.global_params)
            
        self._update_plot()

    def _on_param_update(self, event=None):
        if self.current_idx < 0: return
        current_gui_params = self._gui_to_params()
        if not current_gui_params: return

        if self.selected_cell_idx is not None:
            self.cell_settings[self.selected_cell_idx] = current_gui_params
        else:
            self.global_params = current_gui_params
        self._update_plot()

    def _update_plot(self):
        if self.current_img is None: return
        
        img = self.current_img
        rois = self.current_rois
        _, _, s_tag = self.file_list[self.current_idx]
        
        self.ax.clear()
        self.ax.imshow(img, cmap='gray', interpolation='nearest') 
        
        if self.enable_matlab_var.get() and self.show_mat_var.get() and self.current_mat_polys:
            for poly in self.current_mat_polys:
                self.ax.plot(poly[:, 1], poly[:, 0], linewidth=1.0, color='magenta', linestyle='--')

        # Auto-Zoom
        if self.selected_cell_idx is not None and self.auto_zoom_var.get():
            roi = rois[self.selected_cell_idx]
            xs = roi[:, 0]; ys = roi[:, 1]
            min_x, max_x = xs.min(), xs.max()
            min_y, max_y = ys.min(), ys.max()
            pad_x = (max_x - min_x) * 0.2 + 20
            pad_y = (max_y - min_y) * 0.2 + 20
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(max_y + pad_y, min_y - pad_y) 
        
        for i, roi_poly in enumerate(rois):
            if i in self.cell_settings:
                params = self.cell_settings[i]
            else:
                params = self.global_params
            
            config = self._convert_um_to_px_config(params)
            
            xs = roi_poly[:, 0]
            ys = roi_poly[:, 1]
            x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
            y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
            
            pad = 5
            x_min = max(0, x_min - pad)
            x_max = min(img.shape[1], x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(img.shape[0], y_max + pad)
            
            img_crop = img[y_min:y_max, x_min:x_max]
            poly_crop = roi_poly.copy()
            poly_crop[:, 0] -= x_min
            poly_crop[:, 1] -= y_min
            mask_crop = np.zeros(img_crop.shape, dtype=bool)
            rr, cc = polygon(poly_crop[:,1], poly_crop[:,0], img_crop.shape)
            mask_crop[rr, cc] = True
            
            res, th_val, _, _ = analyze_fa_crop(img_crop, mask_crop, config, self.current_stats)
            
            # Shift Results back to Global Frame
            for cat in res:
                for item in res[cat]:
                    item['contour'][:, 0] += y_min
                    item['contour'][:, 1] += x_min
            
            edge_c = 'cyan' if i == self.selected_cell_idx else 'yellow'
            line_w = 2.5 if i == self.selected_cell_idx else 1.0
            
            patch = mpatches.Polygon(roi_poly, closed=True, edgecolor=edge_c, facecolor='none', linewidth=line_w, linestyle='-')
            self.ax.add_patch(patch)
            self.ax.text(roi_poly[:,0].mean(), roi_poly[:,1].mean(), str(i+1), color=edge_c, fontweight='bold')
            
            colors = {'OK': 'lime', 'Large': 'red', 'Small': 'blue'}
            for cat, items in res.items():
                c = colors[cat]
                for item in items:
                    cnt = item['contour']
                    self.ax.plot(cnt[:, 1], cnt[:, 0], linewidth=0.8, color=c)

        mode_str = "Global" if self.selected_cell_idx is None else f"Cell #{self.selected_cell_idx+1}"
        self.ax.set_title(f"{s_tag} | Mode: {mode_str}")
        self.ax.axis('off')
        self.canvas.draw()

    def _save_current_view(self):
        if self.current_idx < 0: return
        fpath = filedialog.asksaveasfilename(defaultextension=".png")
        if fpath:
            self.fig.savefig(fpath)

    def _run_single_process(self):
        if self.current_img is None: return
        
        out_root = self.out_dir.get()
        indiv_dir = os.path.join(out_root, "individual_results")
        if not os.path.exists(indiv_dir): os.makedirs(indiv_dir)
        img_out = os.path.join(out_root, "fig")
        if not os.path.exists(img_out): os.makedirs(img_out)
        
        px_size = self._get_pixel_size()
        
        img_path, json_path, s_tag = self.file_list[self.current_idx]
        img = self.current_img
        rois = self.current_rois
        
        try:
            fig = Figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            file_rows = []
            save_ok_only = self.save_ok_only_var.get()
            
            for i, roi_poly in enumerate(rois):
                if i in self.cell_settings:
                    params = self.cell_settings[i]
                else:
                    params = self.global_params

                config = self._convert_um_to_px_config(params)
                
                xs = roi_poly[:, 0]; ys = roi_poly[:, 1]
                x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
                y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
                pad = 5
                x_min = max(0, x_min - pad); x_max = min(img.shape[1], x_max + pad)
                y_min = max(0, y_min - pad); y_max = min(img.shape[0], y_max + pad)
                img_crop = img[y_min:y_max, x_min:x_max]
                poly_crop = roi_poly.copy()
                poly_crop[:, 0] -= x_min
                poly_crop[:, 1] -= y_min
                mask_crop = np.zeros(img_crop.shape, dtype=bool)
                rr, cc = polygon(poly_crop[:,1], poly_crop[:,0], img_crop.shape)
                mask_crop[rr, cc] = True
                
                res, th_val, _, _ = analyze_fa_crop(img_crop, mask_crop, config, self.current_stats)
                
                # Draw
                for cat in res:
                    for item in res[cat]:
                        item['contour'][:, 0] += y_min
                        item['contour'][:, 1] += x_min
                
                patch = mpatches.Polygon(roi_poly, closed=True, edgecolor='yellow', facecolor='none', linewidth=1, linestyle='--')
                ax.add_patch(patch)
                colors = {'OK': 'lime', 'Large': 'red', 'Small': 'blue'}
                for cat, items in res.items():
                    c = colors[cat]
                    for item in items:
                        cnt = item['contour']
                        ax.plot(cnt[:, 1], cnt[:, 0], linewidth=0.8, color=c)
                        
                        if save_ok_only and cat != 'OK': continue
                        
                        row_data = {
                            'File': s_tag,
                            'Cell_ID': i+1,
                            'Category': cat,
                            'Area_px': item['area'],
                            'Area_um2': item['area'] * (px_size**2),
                            'Mean_Intensity_Raw': item['mean_int_raw'],
                            'Mean_Intensity_Corr': item['mean_int_corr'],
                            'Int_Density_Raw': item['int_den_raw'],
                            'Int_Density_Corr': item['int_den_corr'],
                            'Background_Level': item['bg_level'],
                            'Used_Alpha': params['alpha'],
                            'Global_Threshold': th_val,
                            'Min_Area_Setting': params['min_area_um'],
                            'Max_Area_Setting': params['max_area_um'],
                            'Close_Radius_Setting': params['close_radius'],
                            'Subtract_BG_Setting': params.get('subtract_bg', True)
                        }
                        file_rows.append(row_data)
            
            fig.savefig(os.path.join(img_out, f"{s_tag}_FA.png"), dpi=150)
            plt.close(fig)
            
            if file_rows:
                csv_path = os.path.join(indiv_dir, f"{s_tag}_results.csv")
                pd.DataFrame(file_rows).to_csv(csv_path, index=False)
                self.status_var.set(f"Saved: {s_tag}_results.csv")
                messagebox.showinfo("Success", f"Saved {s_tag}_results.csv")
            else:
                self.status_var.set("No FAs found to save.")
                
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", str(e))

    def _merge_all_csvs_report(self):
        out_root = self.out_dir.get()
        if not out_root: return
        
        indiv_dir = os.path.join(out_root, "individual_results")
        if not os.path.exists(indiv_dir):
            messagebox.showinfo("Info", "No individual_results folder found.")
            return
            
        all_csvs = glob.glob(os.path.join(indiv_dir, "*_results.csv"))
        if not all_csvs:
            messagebox.showinfo("Info", "No CSV files found to merge.")
            return
        
        try:
            import openpyxl
        except ImportError:
            messagebox.showerror("Error", "Install 'openpyxl' (pip install openpyxl)")
            return

        try:
            dfs = []
            for f in all_csvs:
                dfs.append(pd.read_csv(f))
            
            master_df = pd.concat(dfs, ignore_index=True)
            master_path = os.path.join(out_root, "FA_Results_Master.xlsx")
            
            target_col = 'Mean_Intensity_Corr' if 'Mean_Intensity_Corr' in master_df.columns else 'Mean_Intensity'
            
            with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
                master_df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                cell_summary = master_df.groupby(['File', 'Cell_ID']).agg({
                    'Area_um2': ['count', 'mean', 'sum'],
                    target_col: 'mean'
                }).reset_index()
                
                cell_summary.columns = ['File', 'Cell_ID', 'FA_Count', 'Mean_Area_um2', 'Total_Area_um2', 'Mean_Intensity_Avg']
                cell_summary.to_excel(writer, sheet_name='Cell_Summary', index=False)
                
                master_df['FA_Index'] = master_df.groupby(['File', 'Cell_ID']).cumcount() + 1
                master_df['UniqueID'] = master_df['File'] + "_c" + master_df['Cell_ID'].astype(str)
                
                mat_int = master_df.pivot_table(index='UniqueID', columns='FA_Index', values=target_col)
                mat_int.to_excel(writer, sheet_name='Matrix_Mean_Int')
                
                mat_area = master_df.pivot_table(index='UniqueID', columns='FA_Index', values='Area_um2')
                mat_area.to_excel(writer, sheet_name='Matrix_Area_um2')

            messagebox.showinfo("Success", f"Generated MATLAB-style report:\n{master_path}")
            os.startfile(master_path)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _open_export_dialog(self):
        if not self.file_list: return
        
        dlg = tk.Toplevel(self.root)
        dlg.title("Advanced Export Options")
        dlg.geometry("380x850") # Taller for log area
        
        # 1. Mode
        mode_frame = ttk.LabelFrame(dlg, text="Export Mode")
        mode_frame.pack(fill='x', padx=10, pady=5)
        mode_var = tk.StringVar(value="FA Only")
        ttk.Combobox(mode_frame, textvariable=mode_var, values=["FA Only", "Whole Cell"], state="readonly").pack(fill='x', padx=5, pady=5)
        
        # 2. Subset Extraction (NEW)
        sub_frame = ttk.LabelFrame(dlg, text="Subset Extraction")
        sub_frame.pack(fill='x', padx=10, pady=5)
        
        sub_on_var = tk.BooleanVar(value=False)
        chk_sub = ttk.Checkbutton(sub_frame, text="Enable Subset", variable=sub_on_var)
        chk_sub.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Label(sub_frame, text="Stage No:").grid(row=0, column=1, sticky='e')
        sub_stage_var = tk.StringVar()
        e_stage = ttk.Entry(sub_frame, textvariable=sub_stage_var, width=5, state='disabled')
        e_stage.grid(row=0, column=2, sticky='w', padx=2)
        
        ttk.Label(sub_frame, text="Cell No:").grid(row=0, column=3, sticky='e')
        sub_cell_var = tk.StringVar()
        e_cell = ttk.Entry(sub_frame, textvariable=sub_cell_var, width=5, state='disabled')
        e_cell.grid(row=0, column=4, sticky='w', padx=2)

        def toggle_subset():
            st = 'normal' if sub_on_var.get() else 'disabled'
            e_stage.config(state=st)
            e_cell.config(state=st)
        sub_on_var.trace('w', lambda *a: toggle_subset())

        # 3. Resolution
        res_frame = ttk.LabelFrame(dlg, text="Resolution & Zoom")
        res_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(res_frame, text="Width (px):").grid(row=0, column=0, sticky='e', padx=5)
        w_var = tk.IntVar(value=500)
        ttk.Entry(res_frame, textvariable=w_var, width=6).grid(row=0, column=1, sticky='w')
        
        ttk.Label(res_frame, text="Height (px):").grid(row=0, column=2, sticky='e', padx=5)
        h_var = tk.IntVar(value=500)
        ttk.Entry(res_frame, textvariable=h_var, width=6).grid(row=0, column=3, sticky='w')
        
        ttk.Label(res_frame, text="DPI:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        dpi_var = tk.IntVar(value=600)
        ttk.Entry(res_frame, textvariable=dpi_var, width=6).grid(row=1, column=1, sticky='w')
        
        ttk.Label(res_frame, text="Crop Padding (px):").grid(row=2, column=0, columnspan=2, sticky='e', padx=5, pady=5)
        pad_var = tk.IntVar(value=100)
        ttk.Entry(res_frame, textvariable=pad_var, width=6).grid(row=2, column=2, sticky='w')
        
        # 4. Filter Options
        filt_frame = ttk.LabelFrame(dlg, text="Filter Overlay")
        filt_frame.pack(fill='x', padx=10, pady=5)
        ok_only_var = tk.BooleanVar(value=True) 
        ttk.Checkbutton(filt_frame, text="Show 'OK' Category Only (Exclude Red/Blue)", variable=ok_only_var).pack(anchor='w', padx=5, pady=5)
        
        # 5. Scale Bar Settings
        sb_frame = ttk.LabelFrame(dlg, text="Scale Bar Settings")
        sb_frame.pack(fill='x', padx=10, pady=5)
        
        sb_on_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sb_frame, text="Enable Scale Bar", variable=sb_on_var).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Label(sb_frame, text="Length (µm):").grid(row=0, column=1, sticky='e')
        sb_len_var = tk.DoubleVar(value=20.0)
        ttk.Entry(sb_frame, textvariable=sb_len_var, width=5).grid(row=0, column=2, sticky='w', padx=2)
        
        sb_text_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sb_frame, text="Show Text", variable=sb_text_var).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Label(sb_frame, text="Font Size:").grid(row=1, column=1, sticky='e')
        sb_font_var = tk.IntVar(value=10)
        ttk.Entry(sb_frame, textvariable=sb_font_var, width=5).grid(row=1, column=2, sticky='w', padx=2)
        
        # 6. Colormap Settings
        cm_frame = ttk.LabelFrame(dlg, text="Colormap Settings")
        cm_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(cm_frame, text="Map:").grid(row=0, column=0, sticky='e', padx=5)
        cmap_var = tk.StringVar(value="green")
        cmap_choices = ["jet", "viridis", "inferno", "plasma", "gray", 
                        "blue", "cyan", "green", "yellow", "red", "magenta", "grayscale"]
        ttk.Combobox(cm_frame, textvariable=cmap_var, values=cmap_choices, state="readonly", width=10).grid(row=0, column=1, sticky='w', pady=2)
        
        cbar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cm_frame, text="Show Bar", variable=cbar_var).grid(row=0, column=2, sticky='w', padx=5)
        
        rescale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cm_frame, text="Rescale Intensity", variable=rescale_var).grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(cm_frame, text="Min:").grid(row=2, column=0, sticky='e')
        min_var = tk.StringVar(value="")
        e_min = ttk.Entry(cm_frame, textvariable=min_var, width=6)
        e_min.grid(row=2, column=1, sticky='w', padx=2)
        
        ttk.Label(cm_frame, text="Max:").grid(row=2, column=2, sticky='e')
        max_var = tk.StringVar(value="")
        e_max = ttk.Entry(cm_frame, textvariable=max_var, width=6)
        e_max.grid(row=2, column=3, sticky='w', padx=2)
        
        def toggle_rescale():
            st = 'normal' if rescale_var.get() else 'disabled'
            e_min.config(state=st)
            e_max.config(state=st)
        rescale_var.trace('w', lambda *args: toggle_rescale())
        toggle_rescale()

        # 7. ROI Boundary Settings
        roi_frame = ttk.LabelFrame(dlg, text="ROI Boundary Settings")
        roi_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(roi_frame, text="Width:").grid(row=0, column=0, sticky='e', padx=5)
        roi_lw_var = tk.DoubleVar(value=0.5)
        ttk.Entry(roi_frame, textvariable=roi_lw_var, width=5).grid(row=0, column=1, sticky='w')

        ttk.Label(roi_frame, text="Color:").grid(row=0, column=2, sticky='e', padx=5)
        roi_color_var = tk.StringVar(value="gray")
        color_choices = ['gray', 'white', 'yellow', 'cyan', 'magenta', 'red', 'lime', 'blue', 'black']
        ttk.Combobox(roi_frame, textvariable=roi_color_var, values=color_choices, state="readonly", width=8).grid(row=0, column=3, sticky='w')
        
        # 8. Log Area (NEW)
        log_frame = ttk.LabelFrame(dlg, text="Log / Status")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        log_text = tk.Text(log_frame, height=8, state='disabled', bg='#f0f0f0', font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
        log_text.configure(yscrollcommand=scrollbar.set)
        log_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        def log_msg(msg):
            log_text.config(state='normal')
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            log_text.config(state='disabled')
            dlg.update()

        def run_export_wrapper():
            # Parse Min/Max
            v_min = None
            v_max = None
            if rescale_var.get():
                try: v_min = float(min_var.get())
                except: v_min = None
                try: v_max = float(max_var.get())
                except: v_max = None
            
            # Parse Subset
            sub_stage = None
            sub_cell = None
            if sub_on_var.get():
                s_val = sub_stage_var.get().strip()
                c_val = sub_cell_var.get().strip()
                if s_val:
                    try: sub_stage = int(s_val)
                    except: log_msg("[Error] Stage No must be integer"); return
                if c_val:
                    try: sub_cell = int(c_val)
                    except: log_msg("[Error] Cell No must be integer"); return

            # Call export function with logger callback (don't close dialog)
            self._export_all_crops(
                mode=mode_var.get(), 
                cmap=cmap_var.get(), 
                show_cbar=cbar_var.get(),
                vmin=v_min, vmax=v_max,
                sb_on=sb_on_var.get(),
                sb_len=sb_len_var.get(),
                sb_text=sb_text_var.get(),
                sb_font=sb_font_var.get(),
                out_w=w_var.get(),
                out_h=h_var.get(),
                out_dpi=dpi_var.get(),
                show_ok_only=ok_only_var.get(),
                padding=pad_var.get(),
                roi_lw=roi_lw_var.get(),
                roi_color=roi_color_var.get(),
                subset_on=sub_on_var.get(),
                subset_stage=sub_stage,
                subset_cell=sub_cell,
                logger=log_msg
            )
            
        ttk.Button(dlg, text="START EXPORT", command=run_export_wrapper).pack(fill='x', padx=10, pady=5)
        # Close button since it stays open
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(fill='x', padx=10, pady=5)

    def _export_all_crops(self, mode='FA Only', cmap='jet', show_cbar=True, vmin=None, vmax=None,
                          sb_on=False, sb_len=20, sb_text=True, sb_font=10,
                          out_w=500, out_h=500, out_dpi=600, show_ok_only=False, padding=20,
                          roi_lw=0.5, roi_color='gray',
                          subset_on=False, subset_stage=None, subset_cell=None,
                          logger=None):
        
        out_root = self.out_dir.get()
        crop_dir = os.path.join(out_root, "crop_images")
        if not os.path.exists(crop_dir): os.makedirs(crop_dir)
        
        count = 0
        px_size = self._get_pixel_size()
        
        if logger: logger("Initializing Export...")
        
        try:
            for idx, (img_path, json_path, s_tag) in enumerate(self.file_list):
                # Subset Filter: Stage
                if subset_on and subset_stage is not None:
                    # Extract number from s_tag (e.g. S01 -> 1, s2 -> 2)
                    m = re.search(r'\d+', s_tag)
                    if m:
                        s_num = int(m.group())
                        if s_num != subset_stage:
                            continue
                    else:
                        continue # Skip if no number found
                
                if logger: logger(f"Processing {s_tag}...")
                
                # Load Image
                if idx == self.current_idx and self.current_img is not None:
                    img = self.current_img
                    stats = self.current_stats
                else:
                    img = imread(img_path)
                    if img.ndim > 2: img = img[:,:,0]
                    img_float = img.astype(np.float32)
                    sample = img_float[::10, ::10]
                    bg_val = np.percentile(sample, 1.0)
                    stats = (np.nanmean(img_float), np.nanstd(img_float), bg_val)
                
                with open(json_path, 'r') as f: roi_data = json.load(f)
                rois = []
                for item in roi_data.get('rois', []):
                    pts = item if isinstance(item, list) else item.get('rois', item)
                    if isinstance(item, list): pts = item
                    if pts: rois.append(np.array(pts))
                
                # Load Settings from CSV if exists
                csv_path = os.path.join(out_root, "individual_results", f"{s_tag}_results.csv")
                file_settings = {}
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        u_cells = df.drop_duplicates(subset=['Cell_ID'])
                        for _, r in u_cells.iterrows():
                            cid = int(r['Cell_ID']) - 1
                            file_settings[cid] = {
                                'alpha': float(r['Used_Alpha']),
                                'min_area_um': float(r['Min_Area_Setting']),
                                'max_area_um': float(r['Max_Area_Setting']),
                                'close_radius': int(r['Close_Radius_Setting']),
                                'subtract_bg': bool(r.get('Subtract_BG_Setting', True))
                            }
                    except: pass
                
                for i, roi_poly in enumerate(rois):
                    # Subset Filter: Cell
                    if subset_on and subset_cell is not None:
                        if (i + 1) != subset_cell:
                            continue

                    params = file_settings.get(i, self.global_params)
                    config = self._convert_um_to_px_config(params)
                    
                    xs = roi_poly[:, 0]; ys = roi_poly[:, 1]
                    x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
                    y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
                    
                    # Use User Padding
                    pad = padding
                    x_min = max(0, x_min - pad); x_max = min(img.shape[1], x_max + pad)
                    y_min = max(0, y_min - pad); y_max = min(img.shape[0], y_max + pad)
                    
                    img_crop = img[y_min:y_max, x_min:x_max]
                    poly_crop = roi_poly.copy()
                    poly_crop[:, 0] -= x_min
                    poly_crop[:, 1] -= y_min
                    mask_crop = np.zeros(img_crop.shape, dtype=bool)
                    rr, cc = polygon(poly_crop[:,1], poly_crop[:,0], img_crop.shape)
                    mask_crop[rr, cc] = True
                    
                    res, _, bw, labeled_img = analyze_fa_crop(img_crop, mask_crop, config, stats)
                    
                    # Logic for Final Mask
                    if show_ok_only and mode == 'FA Only':
                        # Only show pixels belonging to 'OK' labels
                        final_mask = np.zeros_like(bw)
                        for item in res['OK']:
                            final_mask[labeled_img == item['label']] = True
                    elif mode == "FA Only":
                        final_mask = bw 
                    else:
                        final_mask = mask_crop
                        
                    fname = f"{s_tag}_Cell_{i+1}_{mode.replace(' ','')}.png"
                    out_path = os.path.join(crop_dir, fname)
                    
                    save_crop_colormap(
                        img_crop, final_mask, poly_crop, out_path, 
                        cmap_name=cmap, show_cbar=show_cbar, mode=mode,
                        vmin=vmin, vmax=vmax,
                        sb_on=sb_on, sb_len_um=sb_len, sb_text=sb_text, sb_font=sb_font,
                        px_size=px_size,
                        out_w=out_w, out_h=out_h, out_dpi=out_dpi,
                        roi_lw=roi_lw, roi_color=roi_color
                    )
                    count += 1
                    if logger: logger(f"  -> Saved Cell #{i+1}")
            
            if logger: logger(f"[Success] Exported {count} images.")
            
        except Exception as e:
            traceback.print_exc()
            if logger: logger(f"[Error] {str(e)}")
        finally:
            if logger: logger("Done.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FAAnalyzerApp(root)
    root.mainloop()
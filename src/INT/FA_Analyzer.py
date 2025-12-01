#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FA Analyzer v1.1 (Batch Process Added)
- Default Export Option: 'Show OK Only' is now ON.
- Removed 'Quick Snapshot' button.
- Added 'PROCESS ALL FILES (Batch)' button with Progress Log.
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
        self.root.title("FA Analyzer v1.1 (Batch)")
        self.root.geometry("1450x980")
        
        # Shortcuts removed as Quick Snapshot is gone
        # self.root.bind('<s>', lambda e: self._save_current_view())
        
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
        
        self.save_ok_only_var = tk.BooleanVar(value=True)

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
        
        # Performance Optimization: Caching
        self.analysis_cache = {} # Key: (cell_idx, param_tuple), Value: result
        
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
            
            # OPTIMIZATION: 
            # 1. 'command' (drag) -> _on_slider_interaction (Live update ONLY for single cell)
            # 2. 'ButtonRelease-1' -> _on_param_update (Final update for Global Mode)
            scl.config(command=self._on_slider_interaction)
            scl.bind("<ButtonRelease-1>", self._on_param_update)
            
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
        # CHANGED: Replaced Quick Snapshot with Process All
        ttk.Button(right_frame, text="PROCESS ALL FILES (Batch)", command=self._run_batch_process).pack(fill='x', padx=10, pady=5)
        
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
        
        # Clear cache on new file load
        self.analysis_cache = {}
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

    def _on_slider_interaction(self, val):
        """
        Called continuously while dragging slider.
        - Global Mode: Do NOTHING (wait for release to avoid lag).
        - Cell Mode: Update LIVE (fast enough due to caching).
        """
        if self.selected_cell_idx is not None:
            self._on_param_update()

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
            
            # CACHING LOGIC
            # Create a hashable key for the current state
            # We need: cell_index, and all parameters that affect analysis
            param_key = (
                params['alpha'],
                params['min_area_um'],
                params['max_area_um'],
                params['close_radius'],
                params.get('subtract_bg', True),
                self.current_stats # Stats also affect result (bg_val)
            )
            
            cache_key = (i, param_key)
            
            if cache_key in self.analysis_cache:
                res, th_val, _, _ = self.analysis_cache[cache_key]
            else:
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
                
                res, th_val, bw, labeled_img = analyze_fa_crop(img_crop, mask_crop, config, self.current_stats)
                
                # Shift Results back to Global Frame immediately for caching consistency
                for cat in res:
                    for item in res[cat]:
                        item['contour'][:, 0] += y_min
                        item['contour'][:, 1] += x_min
                
                # Store in cache
                self.analysis_cache[cache_key] = (res, th_val, bw, labeled_img)

            # NOTE: We already shifted contours in the calculation block above (or retrieved shifted ones)
            # So we don't need to shift them again here.
            
            edge_c = 'cyan' if i == self.selected_cell_idx else 'yellow'
            line_w = 2.5 if i == self.selected_cell_idx else 1.0
            
            patch = mpatches.Polygon(roi_poly, closed=True, edgecolor=edge_c, facecolor='none', linewidth=1, linestyle='-')
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

    def _run_batch_process(self):
        if not self.file_list:
            messagebox.showinfo("Info", "No files loaded.")
            return
            
        if not messagebox.askyesno("Confirm Batch", "Process ALL files with current Global Settings?\n(Individual cell settings will be ignored in batch mode)"):
            return

        out_root = self.out_dir.get()
        indiv_dir = os.path.join(out_root, "individual_results")
        if not os.path.exists(indiv_dir): os.makedirs(indiv_dir)
        
        px_size = self._get_pixel_size()
        save_ok_only = self.save_ok_only_var.get()
        params = self.global_params
        config = self._convert_um_to_px_config(params)
        
        # Progress Window
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Batch Processing...")
        prog_win.geometry("400x300")
        
        log_text = tk.Text(prog_win, height=15, state='disabled', bg='#f0f0f0', font=("Consolas", 9))
        log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        def log(msg):
            log_text.config(state='normal')
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            log_text.config(state='disabled')
            prog_win.update()
            
        count = 0
        try:
            for idx, (img_path, json_path, s_tag) in enumerate(self.file_list):
                log(f"Processing {s_tag}...")
                
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
                
                file_rows = []
                for i, roi_poly in enumerate(rois):
                    # In batch mode, we use global config for all cells
                    # unless we want to load from existing CSV? 
                    # Usually batch overwrites or creates new. Let's use global.
                    
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
                    
                    res, th_val, _, _ = analyze_fa_crop(img_crop, mask_crop, config, stats)
                    
                    for cat, items in res.items():
                        if save_ok_only and cat != 'OK': continue
                        for item in items:
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
                
                if file_rows:
                    csv_path = os.path.join(indiv_dir, f"{s_tag}_results.csv")
                    pd.DataFrame(file_rows).to_csv(csv_path, index=False)
                    count += 1
            
            log(f"Done. Processed {count} files.")
            messagebox.showinfo("Batch Complete", f"Processed {count} files.\nResults saved in: {indiv_dir}")
            
        except Exception as e:
            traceback.print_exc()
            log(f"Error: {e}")

    def _merge_all_csvs_report(self):
        out_root = self.out_dir.get()
        indiv_dir = os.path.join(out_root, "individual_results")
        if not os.path.exists(indiv_dir):
            messagebox.showinfo("Info", "No individual results found.")
            return
            
        csvs = glob.glob(os.path.join(indiv_dir, "*.csv"))
        if not csvs:
            messagebox.showinfo("Info", "No CSV files found.")
            return
            
        dfs = []
        for c in csvs:
            try:
                dfs.append(pd.read_csv(c))
            except: pass
            
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            
            # Summary Pivot
            summary = full_df.groupby(['File', 'Category']).size().unstack(fill_value=0)
            if 'OK' not in summary.columns: summary['OK'] = 0
            if 'Large' not in summary.columns: summary['Large'] = 0
            if 'Small' not in summary.columns: summary['Small'] = 0
            
            summary['Total_Count'] = summary['OK'] + summary['Large'] + summary['Small']
            
            # Save Excel
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_xls = os.path.join(out_root, f"Batch_Summary_{timestamp}.xlsx")
            
            with pd.ExcelWriter(out_xls) as writer:
                summary.to_excel(writer, sheet_name='Summary_Counts')
                full_df.to_excel(writer, sheet_name='All_Data', index=False)
                
            messagebox.showinfo("Success", f"Report generated:\n{out_xls}")
        else:
            messagebox.showinfo("Info", "No data to merge.")

    def _open_export_dialog(self):
        ExportDialog(self.root, self.file_list, self.out_dir.get(), self.current_idx, self.current_img, self.current_rois, self.current_stats, self.global_params, self._get_pixel_size())

# =========================================================
# Export Dialog Class (Enhanced)
# =========================================================
class ExportDialog:
    def __init__(self, parent, file_list, out_root, current_idx, current_img, current_rois, current_stats, global_params, px_size):
        self.top = tk.Toplevel(parent)
        self.top.title("Export Crop Images")
        self.top.geometry("500x600")
        
        self.file_list = file_list
        self.out_root = out_root
        self.current_idx = current_idx
        self.current_img = current_img
        self.current_rois = current_rois
        self.current_stats = current_stats
        self.global_params = global_params
        self.px_size = px_size
        
        # Options
        self.mode_var = tk.StringVar(value="FA Only") # FA Only, Overlay, Mask
        self.cmap_var = tk.StringVar(value="jet")
        self.sb_var = tk.BooleanVar(value=True)
        self.sb_len_var = tk.DoubleVar(value=10.0)
        self.dpi_var = tk.IntVar(value=300)
        self.fmt_var = tk.StringVar(value="png")
        
        ttk.Label(self.top, text="Export Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        f = ttk.Frame(self.top)
        f.pack(fill='x', padx=20, pady=5)
        ttk.Label(f, text="Mode:").pack(side='left')
        ttk.Combobox(f, textvariable=self.mode_var, values=["FA Only", "Grayscale Raw"], state="readonly").pack(side='left', padx=10)
        
        f = ttk.Frame(self.top)
        f.pack(fill='x', padx=20, pady=5)
        ttk.Label(f, text="Colormap:").pack(side='left')
        ttk.Combobox(f, textvariable=self.cmap_var, values=["jet", "viridis", "magma", "gray", "hot"], state="readonly").pack(side='left', padx=10)
        
        f = ttk.Frame(self.top)
        f.pack(fill='x', padx=20, pady=5)
        ttk.Checkbutton(f, text="Add Scale Bar", variable=self.sb_var).pack(side='left')
        ttk.Label(f, text="Length (um):").pack(side='left', padx=(20, 5))
        ttk.Entry(f, textvariable=self.sb_len_var, width=5).pack(side='left')
        
        f = ttk.Frame(self.top)
        f.pack(fill='x', padx=20, pady=5)
        ttk.Label(f, text="DPI:").pack(side='left')
        ttk.Entry(f, textvariable=self.dpi_var, width=5).pack(side='left', padx=10)
        
        ttk.Button(self.top, text="Export Current Image Crops", command=self._export_current).pack(fill='x', padx=20, pady=10)
        ttk.Button(self.top, text="Export ALL Images (Batch)", command=self._export_all).pack(fill='x', padx=20, pady=5)
        
    def _export_current(self):
        if self.current_img is None: return
        self._process_export([ (self.current_img, self.current_rois, self.file_list[self.current_idx][2], self.current_stats) ])
        
    def _export_all(self):
        if not messagebox.askyesno("Confirm", "Export crops for ALL files? This may take time."): return
        
        tasks = []
        # We need to load images for batch
        # This might be slow, so we do it one by one in process loop
        # But here we just pass the list of files to process
        self._process_export(None) # None triggers batch load mode
        
    def _process_export(self, tasks):
        # tasks is list of (img, rois, s_tag, stats) or None
        
        out_dir = os.path.join(self.out_root, "crops_export")
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        cmap = self.cmap_var.get()
        sb_on = self.sb_var.get()
        sb_len = self.sb_len_var.get()
        dpi = self.dpi_var.get()
        
        # Helper to process one file
        def process_one(img, rois, s_tag, stats):
            file_dir = os.path.join(out_dir, s_tag)
            if not os.path.exists(file_dir): os.makedirs(file_dir)
            
            # Global Config for now
            config = {
                'alpha': self.global_params['alpha'],
                'min_px': self.global_params['min_area_um'] / (self.px_size**2),
                'max_px': self.global_params['max_area_um'] / (self.px_size**2),
                'close_radius': self.global_params['close_radius'],
                'subtract_bg': self.global_params.get('subtract_bg', True)
            }
            
            for i, roi_poly in enumerate(rois):
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
                
                # Analyze to get mask
                _, _, bw, _ = analyze_fa_crop(img_crop, mask_crop, config, stats)
                
                # Save
                fname = f"Cell_{i+1}.png"
                path = os.path.join(file_dir, fname)
                
                save_crop_colormap(img_crop, bw, poly_crop, path, 
                                   cmap_name=cmap, show_cbar=True, 
                                   sb_on=sb_on, sb_len_um=sb_len, px_size=self.px_size,
                                   out_dpi=dpi)
                                   
        # Execution
        if tasks:
            for t in tasks: process_one(*t)
            messagebox.showinfo("Done", "Export complete.")
        else:
            # Batch Mode
            prog = tk.Toplevel(self.top)
            l = tk.Label(prog, text="Exporting...")
            l.pack()
            
            for img_path, json_path, s_tag in self.file_list:
                l.config(text=f"Exporting {s_tag}...")
                prog.update()
                
                try:
                    img = imread(img_path)
                    if img.ndim > 2: img = img[:,:,0]
                    img = img.astype(np.float32)
                    
                    with open(json_path, 'r') as f: roi_data = json.load(f)
                    rois = []
                    for item in roi_data.get('rois', []):
                        pts = item if isinstance(item, list) else item.get('rois', item)
                        if isinstance(item, list): pts = item
                        if pts: rois.append(np.array(pts))
                        
                    # Stats
                    sample = img[::10, ::10]
                    bg = np.percentile(sample, 1.0)
                    stats = (np.nanmean(img), np.nanstd(img), bg)
                    
                    process_one(img, rois, s_tag, stats)
                    
                except: pass
            
            prog.destroy()
            messagebox.showinfo("Done", "Batch Export complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FAAnalyzerApp(root)
    root.mainloop()

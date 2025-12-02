import os
import sys
import warnings

# [중요] 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# [중요] OpenMP 라이브러리 충돌 방지 및 스레드 제한
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # 데드락 방지 핵심

# [핵심] 모델 캐시 경로를 현재 폴더로 변경
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.abspath("./.cellpose_cache")

# [핵심] PyTorch 우선 로드 및 워밍업
try:
    import torch
    # GPU 벤치마크 비활성화 (멈춤 방지)
    torch.backends.cudnn.benchmark = False
    # 워밍업: 강제로 GPU 연산을 시도해본다.
    if torch.cuda.is_available():
        try:
            dummy = torch.zeros(1).cuda()
            del dummy
            GPU_OK = True
        except Exception:
            GPU_OK = False
    else:
        GPU_OK = False
except ImportError:
    GPU_OK = False
    pass

import glob
import re
import json
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import cv2

# [중요] OpenCV 멀티스레딩 비활성화
cv2.setNumThreads(0)

# Cellpose 라이브러리 체크
try:
    import cellpose
    from cellpose import models, io
    
    print(f"[System] Cellpose loaded from: {cellpose.__file__}")
    
    if hasattr(models, 'CellposeModel'):
        ModelClass = models.CellposeModel
        print("[System] Using 'models.CellposeModel' class.")
    elif hasattr(models, 'Cellpose'):
        ModelClass = models.Cellpose
        print("[System] Using 'models.Cellpose' class.")
    else:
        raise ImportError("모델 클래스를 찾을 수 없습니다.")
    
    CELLPOSE_AVAILABLE = True
    
except (ImportError, AttributeError, Exception) as e:
    CELLPOSE_AVAILABLE = False
    print(f"[치명적 오류] {e}")

class CellposeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cellpose Auto Segmentation (Fix v3.9 - CPU Optimize)")
        self.root.geometry("750x700")
        
        self.img_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.model_type = tk.StringVar(value="cyto3") 
        self.diameter = tk.DoubleVar(value=30.0) 
        self.channel_filter = tk.StringVar(value="_4") 
        self.use_gpu = tk.BooleanVar(value=True)
        self.is_running = False

        self._setup_ui()

        if not CELLPOSE_AVAILABLE:
            self.log("[치명적 오류] Cellpose 라이브러리 로드 실패.")
        else:
            self.log(f"[준비] Cellpose 라이브러리 로드 완료. (Ver: {cellpose.version})")
            
            # GPU 상태 로그
            if GPU_OK:
                gpu_name = torch.cuda.get_device_name(0)
                self.log(f"[시스템] GPU 정상 작동 확인됨: {gpu_name}")
            else:
                self.log(f"[시스템] GPU를 사용할 수 없습니다. (CPU 모드)")
                self.log(f"   -> 분석 속도가 매우 느릴 수 있습니다.")
                self.use_gpu.set(False) # 강제 해제
                
            self.log("이미지 폴더를 선택하고 '분석 시작'을 누르세요.")
    
    def _setup_ui(self):
        # 1. 경로 설정
        frame_path = tk.LabelFrame(self.root, text="경로 설정", padx=10, pady=10)
        frame_path.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_path, text="이미지 폴더:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_path, textvariable=self.img_dir, width=50).grid(row=0, column=1, padx=5)
        tk.Button(frame_path, text="찾기", command=self._browse_img_dir).grid(row=0, column=2)

        tk.Label(frame_path, text="저장 폴더:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame_path, textvariable=self.out_dir, width=50).grid(row=1, column=1, padx=5)
        tk.Button(frame_path, text="찾기", command=self._browse_out_dir).grid(row=1, column=2)

        # 2. 파라미터 설정
        frame_param = tk.LabelFrame(self.root, text="분석 파라미터", padx=10, pady=10)
        frame_param.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_param, text="모델 타입:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_param, textvariable=self.model_type, width=15).grid(row=0, column=1, sticky="w", padx=5)
        
        tk.Label(frame_param, text="세포 지름(px):").grid(row=1, column=0, sticky="w")
        tk.Entry(frame_param, textvariable=self.diameter, width=15).grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(frame_param, text="(0=자동, 권장:30~100)").grid(row=1, column=2, sticky="w")

        tk.Label(frame_param, text="파일명 필터:").grid(row=2, column=0, sticky="w")
        tk.Entry(frame_param, textvariable=self.channel_filter, width=15).grid(row=2, column=1, sticky="w", padx=5)
        tk.Label(frame_param, text="(예: _4 -> S01_4.TIF만 분석)").grid(row=2, column=2, sticky="w")

        tk.Checkbutton(frame_param, text="GPU 사용", variable=self.use_gpu).grid(row=0, column=3, rowspan=3, padx=20)

        # 3. 실행 버튼
        self.btn_run = tk.Button(self.root, text="분석 시작 (Run)", command=self.start_analysis, height=2, bg="#dddddd")
        self.btn_run.pack(fill="x", padx=10, pady=10)

        # 4. 로그 창
        frame_log = tk.LabelFrame(self.root, text="로그", padx=10, pady=10)
        frame_log.pack(fill="both", expand=True, padx=10, pady=5)
        self.txt_log = scrolledtext.ScrolledText(frame_log, height=10)
        self.txt_log.pack(fill="both", expand=True)

    def log(self, msg):
        print(msg) 
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)

    def _browse_img_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.img_dir.set(p)
            self.out_dir.set(os.path.join(p, "roi"))

    def _browse_out_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.out_dir.set(p)

    def start_analysis(self):
        if self.is_running: return
        self.is_running = True
        self.btn_run.config(state=tk.DISABLED, text="분석 중...")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _read_image_safe(self, filepath):
        try:
            stream = open(filepath, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
            stream.close()
            if img is None:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            return img
        except Exception as e:
            print(f"Safe read failed: {e}")
            return None

    def _run_thread(self):
        try:
            folder = self.img_dir.get()
            save_folder = self.out_dir.get()
            model_name = self.model_type.get()
            diam = self.diameter.get()
            use_gpu = self.use_gpu.get()
            filter_str = self.channel_filter.get().strip()

            self._ensure_dir(save_folder)
            
            if not CELLPOSE_AVAILABLE: return

            self.log(f">> 모델 초기화 중 ({model_name})...")
            
            # [수정] GPU 강제 체크 로직 및 배치 사이즈 조절
            if use_gpu and GPU_OK:
                self.log(f">> GPU 모드로 동작합니다.")
                BATCH_SIZE = 4
            else:
                self.log(f">> [경고] GPU를 찾을 수 없어 CPU 모드로 동작합니다.")
                self.log(f">> CPU 분석은 매우 느립니다. (이미지당 1~5분 소요)")
                self.log(f">> 멈춘 것이 아니니 인내심을 갖고 기다려주세요.")
                use_gpu = False 
                BATCH_SIZE = 1 # CPU는 1장씩 처리하는게 덜 멈춤
            
            model = ModelClass(gpu=use_gpu, model_type=model_name)
            self.log(f">> 모델 로드 완료!")
            
            files = self._list_tifs(folder)
            target_files = [f for f in files if filter_str in os.path.basename(f)] if filter_str else files

            if len(target_files) == 0:
                self.log("[경고] 분석할 파일이 없습니다.")
                return
            
            self.log(f">> 총 {len(target_files)}개 파일 분석 대기.")

            for idx, filepath in enumerate(target_files):
                filename = os.path.basename(filepath)
                save_name = self._parse_tokens(filename) 
                json_path = os.path.join(save_folder, f"{save_name}.json")

                self.log(f"[{idx+1}/{len(target_files)}] 분석 시작: {filename}")
                
                try:
                    img = self._read_image_safe(filepath)
                    if img is None:
                        self.log(f"   [읽기 실패] 파일을 열 수 없습니다.")
                        continue
                    if img.max() <= img.min():
                        self.log("   [스킵] 빈 이미지.")
                        continue
                        
                    self.log(f"   Shape: {img.shape}")

                except Exception as e:
                    self.log(f"   [처리 에러] {e}")
                    continue

                try:
                    self.log(f"   -> Cellpose 추론 중... (Diam={diam}, Batch={BATCH_SIZE})")
                    start_t = time.time()
                    
                    masks, flows, styles = model.eval(img, diameter=diam, batch_size=BATCH_SIZE)
                    
                    end_t = time.time()
                    self.log(f"   -> 추론 완료 ({end_t - start_t:.2f}초)")
                    
                except Exception as e:
                    self.log(f"   [추론 에러] {e}")
                    continue

                rois = self._masks_to_polygons(masks)
                
                if len(rois) == 0:
                    self.log("   -> [결과] 0개 세포 (발견 실패)")
                else:
                    data = {
                        "name": save_name,
                        "image_shape": {"height": int(img.shape[0]), "width": int(img.shape[1])},
                        "rois": rois,
                        "generated_by": f"Cellpose_{model_name}"
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self.log(f"   -> ROI {len(rois)}개 저장 완료.")

            self.log("\n[완료] 모든 작업이 끝났습니다!")
            messagebox.showinfo("완료", "분석 완료")

        except Exception as e:
            self.log(f"[에러 발생] {e}")
            print(f"Error details: {e}")
        finally:
            self.is_running = False
            self.btn_run.config(state=tk.NORMAL, text="분석 시작 (Run)")

    def _ensure_dir(self, path):
        if not os.path.exists(path): os.makedirs(path)

    def _natural_key(self, s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    def _list_tifs(self, folder):
        exts = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files, key=self._natural_key)

    def _parse_tokens(self, basename):
        name = os.path.splitext(basename)[0]
        s_match = re.search(r'(?i)S(\d+)', name)
        t_match = re.search(r'(?i)t(\d+)', name)
        s_str = f"S{int(s_match.group(1)):02d}" if s_match else ""
        t_str = f"t{int(t_match.group(1)):02d}" if t_match else ""
        if s_str and t_str: return f"{s_str}_{t_str}"
        elif s_str: return s_str
        else: return name

    def _masks_to_polygons(self, masks):
        polygons = []
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]
        for label in unique_labels:
            binary_mask = (masks == label).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 20: continue
                poly = cnt.squeeze().astype(float)
                if poly.ndim == 2 and poly.shape[0] >= 3:
                    polygons.append(poly.tolist())
        return polygons

if __name__ == "__main__":
    root = tk.Tk()
    app = CellposeApp(root)
    root.mainloop()
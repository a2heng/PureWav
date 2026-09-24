import os
import sys
import json
import time
import random
import string
import datetime
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import onnxruntime as ort
import soundfile as sf
from typing import Tuple, List
import threading
import queue
from fractions import Fraction
import av
from version import VERSION

# 支持的音频和视频格式
AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma']
VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# 配置文件路径
CONFIG_FILE = "audio_denoise_config.json"

# 深色主题配色 (对齐 wav_browser 的 VS Code 暗色风格)
BG = "#1e1e1e"       # 窗口底
PANEL = "#252526"    # 输入框/列表底
CARD = "#2d2d30"     # 卡片/表头
BORDER = "#3c3c3c"
FG = "#d4d4d4"
MUTED = "#9d9d9d"
ACC = "#0e639c"      # 主色
ACC_H = "#1177bb"


def _fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"

class AudioDenoiseApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"音频文件降噪·PureWav·v{VERSION}")
        self.geometry("900x700")
        self.minsize(720, 560)
        self.config = self.load_config()
        self._setup_theme()
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 初始化处理状态
        self.processing = False
        self.current_file = None
        self.files_to_process = []
        self.process_queue = queue.Queue()
        self.progress_value = 0
        
        # 确定模型文件路径（兼容打包后环境）
        if hasattr(sys, '_MEIPASS'):
            self.model_path = os.path.join(sys._MEIPASS, "v6_erb_skip_proj_batch.onnx")
        else:
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "v6_erb_skip_proj_batch.onnx")
        
        # 设置图标（如果有）
        if hasattr(sys, '_MEIPASS'):
            # 当程序被打包后，_MEIPASS 指向临时解压目录
            icon_path = os.path.join(sys._MEIPASS, "audio_icon.ico")
        else:
            # 开发环境下使用当前目录
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_icon.ico")
        
        # 窗口图标: 仅 Windows 用 .ico (X11 的 Tk 不支持 .ico, 会抛 TclError)
        if sys.platform == "win32" and os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        
        # 启动队列处理线程
        self.start_queue_processor()
    
    def start_queue_processor(self):
        """启动队列处理线程"""
        self.queue_thread = threading.Thread(target=self.process_queue_items, daemon=True)
        self.queue_thread.start()
    
    def load_config(self):
        """加载配置文件"""
        default_config = {
            "output_dir": os.path.expanduser("~/Desktop")
        }
        
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        return default_config
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def _setup_theme(self):
        """深色卡片式主题 (clam 主题上自绘, 无额外依赖)。"""
        self.configure(bg=BG)
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL,
                     bordercolor=BORDER, focuscolor=ACC)
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("TButton", background="#3a3d3e", foreground=FG, borderwidth=0,
                     focusthickness=0, padding=(10, 6))
        st.map("TButton",
               background=[("active", "#4a4d4e"), ("pressed", ACC), ("disabled", "#2a2a2a")],
               foreground=[("disabled", MUTED)])
        st.configure("Accent.TButton", background=ACC, foreground="#ffffff")
        st.map("Accent.TButton", background=[("active", ACC_H), ("pressed", "#0a4d78"),
                                             ("disabled", "#2a2a2a")])
        st.configure("TEntry", fieldbackground=PANEL, foreground=FG, insertcolor=FG,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=4)
        st.configure("TLabelframe", background=BG, bordercolor=BORDER, relief="solid", borderwidth=1)
        st.configure("TLabelframe.Label", background=BG, foreground=MUTED)
        st.configure("Drop.TLabel", background=PANEL, foreground=MUTED, relief="solid",
                     borderwidth=1, bordercolor=BORDER)
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=FG,
                     bordercolor=BORDER, rowheight=24)
        st.map("Treeview", background=[("selected", ACC)], foreground=[("selected", "#ffffff")])
        st.configure("Treeview.Heading", background=CARD, foreground=MUTED,
                     relief="flat", padding=(6, 4))
        st.map("Treeview.Heading", background=[("active", "#3a3d3e")])
        st.configure("TProgressbar", background=ACC, troughcolor=PANEL, bordercolor=BORDER,
                     lightcolor=ACC, darkcolor=ACC, thickness=10)
        st.configure("Vertical.TScrollbar", background=CARD, troughcolor=BG,
                     bordercolor=BG, arrowcolor=MUTED)
        st.configure("Status.TLabel", background=CARD, foreground=MUTED, padding=(10, 4))

    def create_widgets(self):
        """创建UI组件 (深色卡片式)。"""
        # 顶部标题
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=14, pady=(12, 4))
        ttk.Label(header, text="PureWav", font=("Segoe UI", 16, "bold"),
                  foreground=ACC).pack(side=tk.LEFT)
        ttk.Label(header, text=f"音频 / 视频降噪   v{VERSION}",
                  style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0), pady=(6, 0))

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # 输出设置卡片
        cfg = ttk.LabelFrame(main_frame, text=" 输出设置 ")
        cfg.pack(fill=tk.X, pady=(0, 8))
        cfg.columnconfigure(1, weight=1)
        ttk.Label(cfg, text="输出目录").grid(row=0, column=0, padx=(10, 6), pady=10, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value=self.config.get("output_dir", ""))
        ttk.Entry(cfg, textvariable=self.output_dir_var).grid(row=0, column=1, padx=6,
                                                              pady=10, sticky=tk.EW)
        ttk.Button(cfg, text="浏览", command=self.browse_output_dir).grid(row=0, column=2,
                                                                        padx=6, pady=10)
        ttk.Button(cfg, text="保存配置", command=self.save_app_config).grid(row=0, column=3,
                                                                         padx=(6, 10), pady=10)

        # 文件列表卡片
        files = ttk.LabelFrame(main_frame, text=" 文件列表 ")
        files.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        files.columnconfigure(0, weight=1)
        files.rowconfigure(1, weight=1)

        self.drop_label = ttk.Label(files, text="⬇   拖放音频 / 视频文件或文件夹到这里",
                                    anchor=tk.CENTER, padding=14, style="Drop.TLabel")
        self.drop_label.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=(10, 6))
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        self.file_tree = ttk.Treeview(files, columns=("name", "size"), show="headings",
                                      selectmode="extended")
        self.file_tree.heading("name", text="文件名")
        self.file_tree.heading("size", text="大小")
        self.file_tree.column("name", anchor=tk.W, width=460)
        self.file_tree.column("size", anchor=tk.E, width=90, stretch=False)
        self.file_tree.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 8))
        vsb = ttk.Scrollbar(files, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=vsb.set)
        vsb.grid(row=1, column=1, sticky="ns", padx=(0, 10), pady=(0, 8))
        self.file_tree.bind("<Double-1>", lambda e: self.show_spectrum())

        fbar = ttk.Frame(files)
        fbar.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=(0, 10))
        for text, cmd in (("添加文件", self.add_files), ("添加文件夹", self.add_folder),
                          ("移除选中", self.remove_file), ("清空", self.clear_files)):
            ttk.Button(fbar, text=text, command=cmd).pack(side=tk.LEFT, padx=(0, 6))

        # 操作栏
        actions = ttk.Frame(main_frame)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="频谱可视化", command=self.show_spectrum).pack(side=tk.LEFT)
        self.process_btn = ttk.Button(actions, text="▶  开始处理", style="Accent.TButton",
                                      command=self.start_processing)
        self.process_btn.pack(side=tk.RIGHT)

        # 进度条
        pf = ttk.Frame(main_frame)
        pf.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(pf, text="进度", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(pf, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True)

        # 日志卡片
        log_frame = ttk.LabelFrame(main_frame, text=" 处理日志 ")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, bg=PANEL, fg=FG,
                                insertbackground=FG, relief=tk.FLAT, highlightthickness=0,
                                padx=10, pady=8, height=6)
        log_sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.log_text.config(state=tk.DISABLED)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel",
                  anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def save_app_config(self):
        """保存应用配置"""
        self.config["output_dir"] = self.output_dir_var.get()
        
        if self.save_config():
            messagebox.showinfo("成功", "配置已保存")
        else:
            messagebox.showerror("错误", "保存配置失败")
    
    def on_drop(self, event):
        """处理文件拖放事件"""
        files = self.tk.splitlist(event.data)
        self.add_files_to_list(files)
    
    def add_files(self):
        """添加文件"""
        filetypes = (
            ("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a *.wma"),
            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("所有文件", "*.*")
        )
        
        files = filedialog.askopenfilenames(filetypes=filetypes)
        if files:
            self.add_files_to_list(files)
    
    def add_folder(self):
        """添加文件夹"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            # 递归遍历文件夹中的所有文件
            all_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
            
            self.add_files_to_list(all_files)
    
    def _add_one(self, path):
        """加入列表 (以完整路径作为唯一 iid, 重复添加自动跳过)。"""
        if self.file_tree.exists(path):
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        self.file_tree.insert("", tk.END, iid=path,
                              values=(os.path.basename(path), _fmt_size(size)))

    def _all_paths(self):
        return list(self.file_tree.get_children())

    def _selected_paths(self):
        return list(self.file_tree.selection())

    def add_files_to_list(self, paths):
        """添加文件到列表，支持文件和文件夹"""
        for path in paths:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self.is_supported_file(file_path) and os.path.exists(file_path):
                            self._add_one(file_path)
            elif os.path.isfile(path) and self.is_supported_file(path):
                self._add_one(path)

    def is_supported_file(self, path):
        """检查文件是否是支持的格式"""
        ext = os.path.splitext(path)[1].lower()
        return ext in AUDIO_FORMATS or ext in VIDEO_FORMATS

    def remove_file(self):
        """移除选中的文件"""
        for iid in self.file_tree.selection():
            self.file_tree.delete(iid)

    def clear_files(self):
        """清空文件列表"""
        self.file_tree.delete(*self.file_tree.get_children())
    
    def show_spectrum(self):
        """频谱窗口: 默认显示降噪后; 按住「对比」看原图, 松手回降噪后 (两张图预计算, 切换无延迟)。"""
        paths = self._selected_paths() or self._all_paths()
        if not paths:
            messagebox.showwarning("警告", "请先添加并选择一个文件")
            return
        path = paths[0]

        try:
            import matplotlib
            matplotlib.rcParams["font.sans-serif"] = [
                "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                "WenQuanYi Micro Hei", "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.pyplot as plt
            import spectrum_viz as sv
        except Exception as e:
            messagebox.showerror("错误", f"频谱可视化需要 matplotlib: {e}")
            return

        self.status_var.set(f"计算频谱: {os.path.basename(path)} ...")
        self.update()

        # 单次 IO: PyAV 解码到内存 + 内存推理, 不落任何临时文件
        try:
            noisy = decode_mono48k(path)
            if len(noisy) == 0:
                raise RuntimeError("文件没有音频流")
            enhanced = denoise_array(noisy, self.model_path)
        except Exception as e:
            self.status_var.set("就绪")
            messagebox.showerror("错误", f"准备频谱数据失败: {e}")
            return

        # 对齐长度, 两张同形图 → 切换只改 mesh 数据, 无重算
        n = max(len(noisy), len(enhanced))
        noisy = np.pad(noisy, (0, n - len(noisy)))
        enhanced = np.pad(enhanced, (0, n - len(enhanced)))
        disp_n, edges, times = sv.prepare(noisy)
        disp_e, _, _ = sv.prepare(enhanced)
        T = min(disp_n.shape[1], disp_e.shape[1])
        disp_n, disp_e = disp_n[:, :T], disp_e[:, :T]
        times = times[:T]
        x_edges = np.concatenate([times, [times[-1] + sv.HOP / sv.FS]]) if T else np.array([0.0, sv.HOP / sv.FS])

        name = os.path.basename(path)

        # 波形包络 (±1) —— 原图 / 降噪后, 预计算两张, 切换零延迟
        tw, wlo_n, whi_n = sv.waveform_envelope(noisy)
        _, wlo_e, whi_e = sv.waveform_envelope(enhanced)

        win = tk.Toplevel(self)
        win.title(f"频谱 - {name}")
        win.geometry("1500x800")
        win.minsize(820, 440)

        # 左窄控制列 (定宽不随窗口伸缩) + 右巨型可视化 (吃掉全部剩余宽度)
        win.columnconfigure(0, weight=0, minsize=150)
        win.columnconfigure(1, weight=1)
        win.rowconfigure(0, weight=1)

        side = ttk.Frame(win, width=150)
        side.grid(row=0, column=0, sticky="nsew", padx=(6, 2), pady=6)
        side.grid_propagate(False)
        body = ttk.Frame(win)
        body.grid(row=0, column=1, sticky="nsew", padx=(2, 6), pady=6)

        ttk.Label(side, text=name, wraplength=140,
                  font=("", 8), justify=tk.LEFT).pack(anchor="w")
        cmp_btn = ttk.Button(side, text="对比\n按住看原图")
        cmp_btn.pack(fill=tk.X, pady=(8, 4))
        state_lbl = ttk.Label(side, text="降噪后", font=("", 8), foreground="#0a7a3a")
        state_lbl.pack(anchor="w")

        # 右: 上波形 (±1) / 下频谱, 尽量铺满
        fig = plt.figure(figsize=(12, 6), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.08,
                              left=0.03, right=0.995, top=0.99, bottom=0.035)
        ax_w = fig.add_subplot(gs[0])
        ax_s = fig.add_subplot(gs[1], sharex=ax_w)

        poly_n = ax_w.fill_between(tw, wlo_n, whi_n, linewidth=0,
                                   color="#d62728", visible=False)
        poly_e = ax_w.fill_between(tw, wlo_e, whi_e, linewidth=0,
                                   color="#1f77b4", visible=True)
        ax_w.set_ylim(-1.0, 1.0)
        ax_w.set_yticks([-1, 0, 1])
        ax_w.set_ylabel("Wave", fontsize=6)
        ax_w.tick_params(direction="in", labelsize=6, length=2, pad=-9)

        im = ax_s.pcolormesh(x_edges, edges, disp_e, shading="flat",
                             cmap=sv.CMAP, vmin=sv.VMIN, vmax=sv.VMAX)
        ax_s.axhline(sv.BAND_LO_HZ, color="white", linewidth=0.6, alpha=0.5)
        ax_s.set_ylim([0, sv.BAND_HI_HZ])
        ax_s.set_ylabel("Hz", fontsize=6)
        ax_s.set_xlabel("Time (s)", fontsize=6)
        ax_s.tick_params(direction="in", labelsize=6, length=2, pad=-9)
        ax_s.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax_s.yaxis.set_major_locator(plt.MaxNLocator(5))
        for a in (ax_w, ax_s):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)

        canvas = FigureCanvasTkAgg(fig, master=body)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _show(disp, orig):
            im.set_array(disp.ravel())
            poly_n.set_visible(orig)
            poly_e.set_visible(not orig)
            state_lbl.config(text="原图" if orig else "降噪后",
                             foreground="#c0392b" if orig else "#0a7a3a")
            canvas.draw_idle()

        cmp_btn.bind("<ButtonPress-1>", lambda e: _show(disp_n, True))
        cmp_btn.bind("<ButtonRelease-1>", lambda e: _show(disp_e, False))
        self.status_var.set("就绪")

    def start_processing(self):
        """开始处理文件"""
        if self.processing:
            return
        
        files = self._all_paths()
        if not files:
            messagebox.showwarning("警告", "请先添加要处理的文件")
            return
        
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showwarning("警告", "请设置输出目录")
            return
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                messagebox.showerror("错误", f"创建输出目录失败: {str(e)}")
                return
        
        # 将元组转换为列表
        self.files_to_process = list(files)
        self.processing = True
        self.status_var.set("处理中...")
        self.process_btn.config(state=tk.DISABLED)  # 禁用开始处理按钮
        
        # 重置进度条
        self.progress_var.set(0)
        self.progress_bar.update()
        
        # 将文件添加到处理队列
        for file_path in self.files_to_process:
            self.process_queue.put(file_path)
    
    def process_queue_items(self):
        """处理队列中的文件"""
        while True:
            try:
                file_path = self.process_queue.get(timeout=1)
                self.process_file(file_path)
                self.process_queue.task_done()
            except queue.Empty:
                continue
    
    def process_file(self, file_path):
        """处理单个文件"""
        try:
            # 更新当前处理文件
            self.current_file = file_path
            self.log_message(f"开始处理: {os.path.basename(file_path)}")
            
            # 创建进度回调函数
            def progress_callback(progress):
                self.progress_var.set(int(progress * 100))
                self.update()
            
            # 处理文件
            success, output_path = process_media_file(
                file_path, 
                self.output_dir_var.get(), 
                self.model_path,
                progress_callback
            )
            
            if success:
                self.log_message(f"处理成功! 输出文件: {os.path.basename(output_path)}")
            else:
                self.log_message(f"处理失败: {output_path}")
            
            # 从待处理列表中移除
            if file_path in self.files_to_process:
                self.files_to_process.remove(file_path)
            
            # 检查是否所有文件都已处理
            if not self.files_to_process:
                self.after(100, self.finish_processing)
        except Exception as e:
            self.log_message(f"处理过程中发生错误: {str(e)}")
            if file_path in self.files_to_process:
                self.files_to_process.remove(file_path)
            
            if not self.files_to_process:
                self.after(100, self.finish_processing)
    
    def finish_processing(self):
        """完成所有文件处理"""
        self.processing = False
        self.status_var.set("处理完成")
        self.process_btn.config(state=tk.NORMAL)  # 启用开始处理按钮
        self.progress_var.set(0)  # 重置进度条
    
    def log_message(self, message):
        """记录日志消息"""
        # 修改时间戳格式，添加毫秒级精度
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # [:3] 保留三位小数表示毫秒
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def on_close(self):
        """关闭应用"""
        if self.processing:
            if not messagebox.askyesno("确认", "处理仍在进行中，确定要退出吗？"):
                return
        
        # 保存配置
        self.config["output_dir"] = self.output_dir_var.get()
        self.save_config()
        
        self.destroy()

# 以下是处理函数的实现

# ── 媒体后端 (PyAV: 纯 Python wheel, 自带 FFmpeg 库, 跨平台, 无需外部 exe) ──

_MODEL_SESSIONS = {}


def load_session(model_path: str):
    """按模型路径缓存 ONNX 会话 (避免每次处理/频谱都重新加载)。"""
    sess = _MODEL_SESSIONS.get(model_path)
    if sess is None:
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'],
                                    sess_options=ort.SessionOptions())
        _MODEL_SESSIONS[model_path] = sess
    return sess


def decode_mono48k(path: str, sr: int = 48000) -> np.ndarray:
    """任意音/视频 → 48k 单声道 float32 (PyAV 解码到内存, 不落盘)。"""
    chunks = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == 'audio'), None)
        if stream is None:
            raise RuntimeError('文件没有音频流')
        resampler = av.audio.resampler.AudioResampler(format='flt', layout='mono', rate=sr)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


def probe_samplerate(path: str, default: int = 48000) -> int:
    """探测文件采样率 (音频用 soundfile, 视频回退 PyAV)。"""
    try:
        return int(sf.info(path).samplerate)
    except Exception:
        pass
    try:
        with av.open(path) as c:
            st = next((s for s in c.streams if s.type == 'audio'), None)
            if st is not None and st.rate:
                return int(st.rate)
    except Exception:
        pass
    return default


def _stft(signal, window, n_fft=960, hop=480):
    frames = []
    for i in range(0, len(signal) - len(window) + 1, hop):
        frames.append(np.fft.rfft(signal[i:i + len(window)] * window, n=n_fft))
    if not frames:
        return np.zeros((1, n_fft // 2 + 1), dtype=np.complex64)
    return np.stack(frames)


def _istft(spec, window, hop=480, n_fft=960):
    n_frames = spec.shape[0]
    out_len = (n_frames - 1) * hop + len(window)
    output = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(spec[i], n=n_fft).astype(np.float32) * window
        start = i * hop
        output[start:start + len(window)] += frame
        win_sum[start:start + len(window)] += window ** 2
    return output / np.maximum(win_sum, 1e-8)


def _denoise_chunk(audio_chunk, session, window):
    """一段 float32 (48k mono) → 降噪后 float32 (STFT → ONNX → ISTFT)。"""
    spec = _stft(audio_chunk, window)
    spec_ri = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
    spec_ri = spec_ri.reshape(1, spec_ri.shape[0], spec_ri.shape[1], 2)
    spec_ri = np.transpose(spec_ri, (0, 3, 1, 2))
    enhanced_ri = session.run(None, {'spec': spec_ri})[0]
    enhanced_ri = np.transpose(enhanced_ri, (0, 2, 3, 1))
    enhanced_spec = (enhanced_ri[0, :, :, 0] + 1j * enhanced_ri[0, :, :, 1]).astype(np.complex64)
    enhanced = _istft(enhanced_spec, window)[:len(audio_chunk)]
    return np.clip(enhanced, -1.0, 1.0)


def denoise_array(audio: np.ndarray, model_path: str, chunk_duration: float = 30.0,
                  progress_callback=None) -> np.ndarray:
    """整段 float32 (48k mono) → 降噪后 float32 (内存, 不落盘)。"""
    session = load_session(model_path)
    window = np.hanning(960).astype(np.float32)
    chunk = max(1, int(48000 * chunk_duration))
    total = len(audio)
    parts = []
    for i in range(0, total, chunk):
        parts.append(_denoise_chunk(audio[i:i + chunk], session, window))
        if progress_callback:
            progress_callback(min(1.0, (i + chunk) / max(1, total)))
    if not parts:
        return audio.astype(np.float32).copy()
    return np.concatenate(parts).astype(np.float32)[:total]


def process_audio_file(audio_path: str, output_path: str, model_path: str, progress_callback=None,
                       chunk_duration: float = 30.0, output_sr: int = None) -> bool:
    """流式处理音频: PyAV 解码 → ONNX 推理 → WAV 写出 (全程无临时文件)。"""
    session = load_session(model_path)
    window = np.hanning(960).astype(np.float32)
    chunk = max(1, int(48000 * chunk_duration))
    try:
        with av.open(audio_path) as container:
            stream = next((s for s in container.streams if s.type == 'audio'), None)
            if stream is None:
                raise RuntimeError('文件没有音频流')
            total = None
            if container.duration:
                total = int(container.duration / av.time_base * 48000)
            resampler = av.audio.resampler.AudioResampler(format='flt', layout='mono', rate=48000)
            processed = 0

            with sf.SoundFile(output_path, 'w', samplerate=48000, channels=1,
                              subtype='PCM_16') as out:
                buf = np.zeros(0, dtype=np.float32)

                def emit(seg):
                    nonlocal processed
                    out.write(_denoise_chunk(seg, session, window))
                    processed += len(seg)
                    if progress_callback:
                        progress_callback(min(1.0, processed / total) if total else 0.0)

                for frame in container.decode(stream):
                    for rf in resampler.resample(frame):
                        buf = np.concatenate([buf, rf.to_ndarray().reshape(-1).astype(np.float32)])
                    while len(buf) >= chunk:
                        emit(buf[:chunk])
                        buf = buf[chunk:].copy()
                for rf in resampler.resample(None):
                    buf = np.concatenate([buf, rf.to_ndarray().reshape(-1).astype(np.float32)])
                if len(buf):
                    emit(buf)

        if output_sr and output_sr != 48000:
            from scipy.signal import resample_poly
            from math import gcd
            data, sr = sf.read(output_path, dtype='float32')
            g = gcd(sr, output_sr)
            data = resample_poly(data, output_sr // g, sr // g).astype(np.float32)
            sf.write(output_path, data, output_sr, subtype='PCM_16')
        return True
    except Exception as e:
        print(f"处理音频文件失败: {e}")
        return False


def replace_video_audio(video_path: str, audio: np.ndarray, output_path: str,
                        sr: int = 48000) -> bool:
    """把降噪后音频 (numpy) 封装回视频: 视频流直接拷贝, 音频重编码 AAC (无临时文件)。"""
    try:
        inp = av.open(video_path)
        out = av.open(output_path, 'w')
        try:
            in_v = inp.streams.video[0]
            out_v = out.add_stream_from_template(in_v)
            out_a = out.add_stream('aac', rate=sr)
            out_a.layout = 'mono'
            out_a.bit_rate = 192000
            frame_size = out_a.codec_context.frame_size or 1024
            n = len(audio)
            for i in range(0, n, frame_size):
                seg = audio[i:i + frame_size]
                if len(seg) < frame_size:
                    seg = np.pad(seg, (0, frame_size - len(seg)))
                af = av.AudioFrame.from_ndarray(seg.reshape(1, -1).astype('float32'),
                                                format='flt', layout='mono')
                af.sample_rate = sr
                af.pts = i
                af.time_base = Fraction(1, sr)
                for pkt in out_a.encode(af):
                    out.mux(pkt)
            for pkt in out_a.encode(None):
                out.mux(pkt)
            for pkt in inp.demux(in_v):
                if pkt.dts is None:
                    continue
                pkt.stream = out_v
                out.mux(pkt)
        finally:
            out.close()
            inp.close()
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"封装视频失败: {e}")
        return False

def process_media_file(input_path: str, output_dir: str, model_path: str, progress_callback=None, output_sr: int = None) -> Tuple[bool, str]:
    """处理媒体文件 (音频: 流式解码→降噪→写 WAV; 视频: 解码到内存→降噪→流拷贝封装)。"""
    try:
        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext in VIDEO_FORMATS
        is_audio = ext in AUDIO_FORMATS
        if not is_video and not is_audio:
            return False, "不支持的文件格式"

        random_hex = ''.join(random.choices(string.hexdigits, k=8)).lower()
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_ext = '.mp4' if is_video else '.wav'
        output_path = os.path.join(output_dir, f"{base_name}_降噪_{random_hex}{output_ext}")

        if is_video:
            audio = decode_mono48k(input_path)
            if len(audio) == 0:
                return False, "视频没有音频流"
            enhanced = denoise_array(audio, model_path, progress_callback=progress_callback)
            if output_sr and output_sr != 48000:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(48000, output_sr)
                enhanced = resample_poly(enhanced, output_sr // g, 48000 // g).astype(np.float32)
                audio_sr = output_sr
            else:
                audio_sr = 48000
            if not replace_video_audio(input_path, enhanced, output_path, sr=audio_sr):
                return False, "替换视频音频失败"
        else:
            if not process_audio_file(input_path, output_path, model_path,
                                      progress_callback, output_sr=output_sr):
                return False, "处理音频失败"

        return True, output_path
    except Exception as e:
        print(f"[DEBUG] 处理过程中发生异常: {str(e)}")
        return False, f"处理失败: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="PureWav - audio denoiser")
        parser.add_argument("input", nargs="?", help="input audio file")
        parser.add_argument("-o", "--output", help="output wav path")
        parser.add_argument("-m", "--model", default=None, help="ONNX model path")
        parser.add_argument("-r", "--sr", type=int, default=None, help="output sample rate (default: same as input)")
        args = parser.parse_args()

        if args.input:
            model_path = args.model
            if not model_path:
                if hasattr(sys, '_MEIPASS'):
                    model_path = os.path.join(sys._MEIPASS, "v6_erb_skip_proj_batch.onnx")
                else:
                    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "v6_erb_skip_proj_batch.onnx")

            if args.output:
                output_path = args.output
            else:
                base, ext = os.path.splitext(args.input)
                output_path = f"{base}_denoised.wav"

            output_sr = args.sr or probe_samplerate(args.input)

            print(f"Input:  {args.input}")
            print(f"Output: {output_path} ({output_sr}Hz)")

            def cli_progress(p):
                print(f"\r  [{p*100:5.1f}%]", end="", flush=True)

            success = process_audio_file(
                args.input, output_path, model_path, cli_progress, output_sr=output_sr
            )
            if success:
                print(f"\n  -> {output_path}")
            else:
                print(f"\n  Error: processing failed")
                sys.exit(1)
        else:
            parser.print_help()
    else:
        app = AudioDenoiseApp()
        app.mainloop()
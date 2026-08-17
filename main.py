import os
import sys
import json
import time
import random
import string
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import onnxruntime as ort
import subprocess
import soundfile as sf
from typing import Tuple, List
import threading
import queue
from version import VERSION

# 支持的音频和视频格式
AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma']
VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# 配置文件路径
CONFIG_FILE = "audio_denoise_config.json"

class AudioDenoiseApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"音频文件降噪·PureWav·v{VERSION}")
        self.geometry("800x600")
        self.config = self.load_config()
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
        
        if os.path.exists(icon_path):
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
    
    def create_widgets(self):
        """创建UI组件"""
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 配置区域
        config_frame = ttk.LabelFrame(main_frame, text="配置设置")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 输出目录
        ttk.Label(config_frame, text="输出目录:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value=self.config.get("output_dir", ""))
        output_dir_entry = ttk.Entry(config_frame, textvariable=self.output_dir_var, width=50)
        output_dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        browse_btn = ttk.Button(config_frame, text="浏览", command=self.browse_output_dir)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # 保存配置按钮
        save_config_btn = ttk.Button(config_frame, text="保存配置", command=self.save_app_config)
        save_config_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # 文件拖放区域
        drop_frame = ttk.LabelFrame(main_frame, text="拖放文件或文件夹到此处")
        drop_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.drop_label = ttk.Label(drop_frame, text="拖放音频/视频文件或文件夹到这里", 
                                   relief=tk.SUNKEN, anchor=tk.CENTER, padding=20)
        self.drop_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 注册拖放事件
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        # 文件列表
        self.file_listbox = tk.Listbox(drop_frame, height=5)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        add_file_btn = ttk.Button(btn_frame, text="添加文件", command=self.add_files)
        add_file_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        add_folder_btn = ttk.Button(btn_frame, text="添加文件夹", command=self.add_folder)
        add_folder_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        remove_btn = ttk.Button(btn_frame, text="移除文件", command=self.remove_file)
        remove_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_btn = ttk.Button(btn_frame, text="清空列表", command=self.clear_files)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 处理按钮
        self.process_btn = ttk.Button(btn_frame, text="开始处理", command=self.start_processing)
        self.process_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 进度条区域
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="降噪进度:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.log_text.config(state=tk.DISABLED)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
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
    
    def add_files_to_list(self, paths):
        """添加文件到列表，支持文件和文件夹"""
        for path in paths:
            # 如果是文件夹，递归添加其中的文件
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if self.is_supported_file(file_path) and os.path.exists(file_path):
                            self.file_listbox.insert(tk.END, file_path)
            # 如果是文件，直接添加
            elif os.path.isfile(path) and self.is_supported_file(path):
                self.file_listbox.insert(tk.END, path)
    
    def is_supported_file(self, path):
        """检查文件是否是支持的格式"""
        ext = os.path.splitext(path)[1].lower()
        return ext in AUDIO_FORMATS or ext in VIDEO_FORMATS
    
    def remove_file(self):
        """移除选中的文件"""
        selection = self.file_listbox.curselection()
        if selection:
            self.file_listbox.delete(selection[0])
    
    def clear_files(self):
        """清空文件列表"""
        self.file_listbox.delete(0, tk.END)
    
    def start_processing(self):
        """开始处理文件"""
        if self.processing:
            return
        
        files = self.file_listbox.get(0, tk.END)
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

# 在文件顶部添加此函数
def get_ffmpeg_path():
    """获取ffmpeg可执行文件的路径"""
    if hasattr(sys, '_MEIPASS'):
        # 打包后环境
        return os.path.join(sys._MEIPASS, 'ffmpeg.exe')
    else:
        # 开发环境
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg.exe')

# 修改extract_audio函数
def extract_audio(input_path: str, temp_audio_path: str) -> bool:
    """从视频中提取音频"""
    try:
        print(f"[DEBUG] extract_audio: 开始从 {input_path} 提取音频")
        ffmpeg_path = get_ffmpeg_path()
        print(f"[DEBUG] extract_audio: 使用ffmpeg路径: {ffmpeg_path}")
        
        cmd = [
            ffmpeg_path, '-i', input_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '48000', '-ac', '1', temp_audio_path, '-y'
        ]
        print(f"[DEBUG] extract_audio: 执行命令: {' '.join(cmd)}")
        
        # 添加startupinfo参数来隐藏黑框
        si = None
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            print(f"[DEBUG] extract_audio: 使用Windows隐藏窗口模式")
        
        # 捕获输出以进行调试
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si)
        print(f"[DEBUG] extract_audio: 命令执行成功，返回码: 0")
        
        # 检查输出文件是否存在且不为空
        if os.path.exists(temp_audio_path):
            file_size = os.path.getsize(temp_audio_path)
            print(f"[DEBUG] extract_audio: 输出文件存在，大小: {file_size} 字节")
            if file_size > 0:
                return True
            else:
                print(f"[DEBUG] extract_audio: 警告: 输出文件存在但为空")
                return False
        else:
            print(f"[DEBUG] extract_audio: 错误: 输出文件不存在: {temp_audio_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[DEBUG] extract_audio: 错误: ffmpeg执行失败，返回码: {e.returncode}")
        print(f"[DEBUG] extract_audio: 错误输出: {e.stderr.decode('utf-8', errors='ignore')}")
        return False
    except Exception as e:
        print(f"[DEBUG] extract_audio: 异常: {str(e)}")
        return False

def process_audio_file(audio_path: str, output_path: str, model_path: str, progress_callback=None,
                       chunk_duration: float = 30.0) -> bool:
    """
    分块处理音频文件：numpy STFT → ONNX 推理 → numpy ISTFT
    """
    NFFT = 960
    HOP = 480
    WIN = 960
    window = np.hanning(WIN).astype(np.float32)

    def stft(signal):
        frames = []
        for i in range(0, len(signal) - WIN + 1, HOP):
            frame = signal[i:i + WIN] * window
            spec = np.fft.rfft(frame, n=NFFT)
            frames.append(spec)
        if not frames:
            return np.zeros((1, NFFT // 2 + 1), dtype=np.complex64)
        return np.stack(frames)

    def istft(spec):
        n_frames, n_freq = spec.shape
        out_len = (n_frames - 1) * HOP + WIN
        output = np.zeros(out_len, dtype=np.float32)
        win_sum = np.zeros(out_len, dtype=np.float32)
        for i in range(n_frames):
            frame = np.fft.irfft(spec[i], n=NFFT).astype(np.float32) * window
            start = i * HOP
            output[start:start + WIN] += frame
            win_sum[start:start + WIN] += window ** 2
        win_sum = np.maximum(win_sum, 1e-8)
        return output / win_sum

    try:
        ffmpeg_path = get_ffmpeg_path()
        temp_standard_path = os.path.splitext(audio_path)[0] + "_standard.wav"

        cmd = [
            ffmpeg_path, '-i', audio_path, '-ar', '48000', '-ac', '1',
            '-acodec', 'pcm_s16le', temp_standard_path, '-y'
        ]
        si = None
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si)

        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'],
                                       sess_options=ort.SessionOptions())
        chunk_size = int(48000 * chunk_duration)

        with sf.SoundFile(temp_standard_path, 'r') as infile, \
             sf.SoundFile(output_path, 'w', samplerate=infile.samplerate,
                          channels=1, subtype='PCM_16') as outfile:

            total_frames = len(infile)
            processed = 0

            while True:
                audio_chunk = infile.read(chunk_size, dtype='float32')
                if len(audio_chunk) == 0:
                    break

                spec = stft(audio_chunk)
                spec_ri = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
                spec_ri = spec_ri.reshape(1, spec_ri.shape[0], spec_ri.shape[1], 2)
                spec_ri = np.transpose(spec_ri, (0, 3, 1, 2))

                enhanced_ri = session.run(None, {'spec': spec_ri})[0]
                enhanced_ri = np.transpose(enhanced_ri, (0, 2, 3, 1))
                enhanced_spec = (enhanced_ri[0, :, :, 0] + 1j * enhanced_ri[0, :, :, 1]).astype(np.complex64)

                enhanced_audio = istft(enhanced_spec)
                enhanced_audio = enhanced_audio[:len(audio_chunk)]
                enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
                outfile.write(enhanced_audio)

                processed += len(audio_chunk)
                if progress_callback:
                    progress_callback(processed / total_frames)

        os.remove(temp_standard_path)
        return True

    except Exception as e:
        print(f"处理音频文件失败: {str(e)}")
        try:
            if 'temp_standard_path' in locals() and os.path.exists(temp_standard_path):
                os.remove(temp_standard_path)
        except:
            pass
        return False


def replace_video_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """替换视频中的音频"""
    try:
        print(f"[DEBUG] replace_video_audio: 开始替换音频，视频: {video_path}, 音频: {audio_path}, 输出: {output_path}")
        
        # 检查输入文件是否存在
        if not os.path.exists(video_path):
            print(f"[DEBUG] replace_video_audio: 错误: 视频文件不存在: {video_path}")
            return False
        if not os.path.exists(audio_path):
            print(f"[DEBUG] replace_video_audio: 错误: 音频文件不存在: {audio_path}")
            return False
        
        print(f"[DEBUG] replace_video_audio: 输入文件检查通过")
        ffmpeg_path = get_ffmpeg_path()
        print(f"[DEBUG] replace_video_audio: 使用ffmpeg路径: {ffmpeg_path}")
        
        # 修改ffmpeg命令，使用更兼容的参数
        cmd = [
            ffmpeg_path, '-i', video_path, '-i', audio_path, 
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', 
            '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0', 
            output_path, '-y'
        ]
        print(f"[DEBUG] replace_video_audio: 执行命令: {' '.join(cmd)}")
        
        # 添加startupinfo参数来隐藏黑框
        si = None
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            print(f"[DEBUG] replace_video_audio: 使用Windows隐藏窗口模式")
        
        # 捕获输出以进行调试
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=si)
        
        print(f"[DEBUG] replace_video_audio: 命令执行完成，返回码: {result.returncode}")
        
        if result.returncode != 0:
            print(f"[DEBUG] replace_video_audio: 错误: ffmpeg执行失败")
            print(f"[DEBUG] replace_video_audio: 标准输出: {result.stdout.decode('utf-8', errors='ignore')}")
            print(f"[DEBUG] replace_video_audio: 错误输出: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
        
        # 检查输出文件是否存在且不为空
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"[DEBUG] replace_video_audio: 输出文件存在，大小: {file_size} 字节")
            if file_size > 0:
                # 检查原始视频大小
                orig_size = os.path.getsize(video_path)
                print(f"[DEBUG] replace_video_audio: 原始视频大小: {orig_size} 字节")
                return True
            else:
                print(f"[DEBUG] replace_video_audio: 警告: 输出文件存在但为空")
                return False
        else:
            print(f"[DEBUG] replace_video_audio: 错误: 输出文件不存在: {output_path}")
            return False
            
    except Exception as e:
        print(f"[DEBUG] replace_video_audio: 异常: {str(e)}")
        return False

def process_media_file(input_path: str, output_dir: str, model_path: str, progress_callback=None) -> Tuple[bool, str]:
    """处理媒体文件"""
    try:
        # 检查文件类型
        ext = os.path.splitext(input_path)[1].lower()
        is_video = ext in VIDEO_FORMATS
        is_audio = ext in AUDIO_FORMATS
        
        if not is_video and not is_audio:
            return False, "不支持的文件格式"
        
        # 生成输出路径
        random_hex = ''.join(random.choices(string.hexdigits, k=8)).lower()
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_ext = '.mp4' if is_video else '.wav'
        output_path = os.path.join(output_dir, f"{base_name}_降噪_{random_hex}{output_ext}")
        
        # 添加调试日志
        print(f"[DEBUG] 处理文件: {input_path}")
        print(f"[DEBUG] 是视频文件: {is_video}, 是音频文件: {is_audio}")
        print(f"[DEBUG] 输出路径: {output_path}")
        
        if is_video:
            # 提取音频
            temp_audio_path = os.path.join(output_dir, f"temp_{random_hex}.wav")
            print(f"[DEBUG] 开始提取音频到: {temp_audio_path}")
            if not extract_audio(input_path, temp_audio_path):
                print(f"[DEBUG] 提取音频失败")
                return False, "提取音频失败"
            
            # 检查提取的音频文件是否存在且不为空
            if not os.path.exists(temp_audio_path):
                print(f"[DEBUG] 错误: 提取的音频文件不存在: {temp_audio_path}")
                return False, f"提取的音频文件不存在: {temp_audio_path}"
            if os.path.getsize(temp_audio_path) == 0:
                print(f"[DEBUG] 错误: 提取的音频文件为空: {temp_audio_path}")
                return False, f"提取的音频文件为空: {temp_audio_path}"
            print(f"[DEBUG] 音频提取成功，文件大小: {os.path.getsize(temp_audio_path)} 字节")
            
            # 处理音频
            temp_processed_audio_path = os.path.join(output_dir, f"temp_processed_{random_hex}.wav")
            print(f"[DEBUG] 开始处理音频: {temp_audio_path} -> {temp_processed_audio_path}")
            if not process_audio_file(temp_audio_path, temp_processed_audio_path, model_path, progress_callback):
                print(f"[DEBUG] 处理音频失败")
                os.remove(temp_audio_path)
                return False, "处理音频失败"
            
            # 检查处理后的音频文件是否存在且不为空
            if not os.path.exists(temp_processed_audio_path):
                print(f"[DEBUG] 错误: 处理后的音频文件不存在: {temp_processed_audio_path}")
                os.remove(temp_audio_path)
                return False, f"处理后的音频文件不存在: {temp_processed_audio_path}"
            if os.path.getsize(temp_processed_audio_path) == 0:
                print(f"[DEBUG] 错误: 处理后的音频文件为空: {temp_processed_audio_path}")
                os.remove(temp_audio_path)
                os.remove(temp_processed_audio_path)
                return False, f"处理后的音频文件为空: {temp_processed_audio_path}"
            print(f"[DEBUG] 音频处理成功，文件大小: {os.path.getsize(temp_processed_audio_path)} 字节")
            
            # 替换视频音频
            print(f"[DEBUG] 开始替换视频音频: {input_path} + {temp_processed_audio_path} -> {output_path}")
            if not replace_video_audio(input_path, temp_processed_audio_path, output_path):
                print(f"[DEBUG] 替换视频音频失败")
                os.remove(temp_audio_path)
                os.remove(temp_processed_audio_path)
                return False, "替换视频音频失败"
            
            # 检查输出视频文件是否存在
            if not os.path.exists(output_path):
                print(f"[DEBUG] 错误: 输出视频文件不存在: {output_path}")
                os.remove(temp_audio_path)
                os.remove(temp_processed_audio_path)
                return False, f"输出视频文件不存在: {output_path}"
            print(f"[DEBUG] 视频音频替换成功，输出文件大小: {os.path.getsize(output_path)} 字节")
            
            # 清理临时文件
            os.remove(temp_audio_path)
            os.remove(temp_processed_audio_path)
            print(f"[DEBUG] 清理临时文件成功")
        else:
            # 处理音频文件
            print(f"[DEBUG] 开始处理音频文件: {input_path} -> {output_path}")
            if not process_audio_file(input_path, output_path, model_path, progress_callback):
                print(f"[DEBUG] 处理音频文件失败")
                return False, "处理音频失败"
            print(f"[DEBUG] 音频文件处理成功，输出文件大小: {os.path.getsize(output_path)} 字节")
        
        return True, output_path
    except Exception as e:
        print(f"[DEBUG] 处理过程中发生异常: {str(e)}")
        return False, f"处理失败: {str(e)}"

if __name__ == "__main__":
    app = AudioDenoiseApp()
    app.mainloop()
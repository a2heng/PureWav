"""PureWav 频谱可视化 —— 参照 PureVoxModel research_platform 的 Streamlit 频谱页。

参数对齐 web 频谱页:
    fs=48000, n_fft=960, hop=480, hann 窗, cmap=inferno, vmin=-80dB, vmax=0dB

两条硬要求:
    1. **全频轴 + 20–24 kHz 按 1 kHz 分带**: 0–20 kHz 逐 STFT bin 线性展示;
       20–24 kHz 压缩成 4 个 1 kHz 带 (取带内均值) —— 混合轴, 高频细节不被 80 个
       50 Hz bin 拉长。
    2. **完整音频不切分时间**: 整段一次性 STFT, 时间轴是整条音频, 不分段拼接。

本模块只依赖 numpy / matplotlib (matplotlib 经 librosa 已随应用安装); 不在 import 期
强制加载 matplotlib, 避免无 GUI 环境下拖累。
"""

import numpy as np

FS = 48000
N_FFT = 960
HOP = 480
BAND_LO_HZ = 20000       # 混合轴分界: 低于此逐 bin 线性
BAND_HI_HZ = 24000       # 最高频 (nyquist)
BAND_WIDTH_HZ = 1000     # 20–24 kHz 按 1 kHz 分带
VMIN, VMAX = -90.0, -10.0  # dB 色标上下界 (对齐 audioscope 标准)
CMAP = "inferno"


def compute_spectrogram(audio, fs=FS, n_fft=N_FFT, hop=HOP):
    """整段 STFT → (mag_db (F,T), freqs (F,), times (T,))。

    分块计算以控内存, 但时间轴是整条音频 (不做任何分段拼接)。
    """
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    n_freq = n_fft // 2 + 1
    if len(audio) < n_fft:
        return (np.full((n_freq, 1), VMIN, dtype=np.float32),
                np.arange(n_freq) * fs / n_fft, np.zeros(1, dtype=np.float32))

    window = np.hanning(n_fft).astype(np.float32)
    # 窗相干增益归一: 满幅单音峰值 → 0 dB (同 audioscope power_scale)
    half_win_sum = float(window.sum()) * 0.5
    power_scale = 1.0 / max(half_win_sum * half_win_sum, 1e-12)
    n_frames = 1 + (len(audio) - n_fft) // hop
    mag_db = np.empty((n_freq, n_frames), dtype=np.float32)

    block = 2048  # 每次算多少帧, 控峰值内存
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        idx = start + np.arange(stop - start)
        frames = np.stack([audio[i * hop:i * hop + n_fft] for i in idx])
        spec = np.fft.rfft(frames * window, n=n_fft, axis=1)      # (b, F)
        power = (np.abs(spec) ** 2) * power_scale
        mag_db[:, start:stop] = (10.0 * np.log10(power + 1e-14)).T
    # DC / Nyquist 置零 (同 audioscope)
    mag_db[0, :] = VMIN
    mag_db[-1, :] = VMIN

    freqs = np.arange(n_freq) * fs / n_fft
    times = np.arange(n_frames) * hop / fs
    return mag_db, freqs, times


def hybrid_axis(mag_db, freqs):
    """全频 + 20–24 kHz 按 1 kHz 分带 → (display (R,T), y_edges (R+1,))。

    0–20 kHz: 逐 bin (每 50 Hz 一行); 20–24 kHz: 按 1 kHz 分 4 带取带内均值。
    y_edges 为每行的频率上下界 (非均匀), 直接喂 pcolormesh 的 Y 轴。
    """
    bin_hz = float(freqs[1] - freqs[0]) if len(freqs) > 1 else FS / N_FFT
    n_freq = mag_db.shape[0]
    f_max = float(freqs[-1])                                   # 24000
    lo = min(int(round(BAND_LO_HZ / bin_hz)), n_freq)          # 400 (20 kHz 对应 bin)
    lin = mag_db[:lo]                                          # 线性段
    lin_edges = np.arange(lo + 1) * bin_hz                     # 0 .. 20 kHz

    if lo < n_freq:
        n_bands = max(1, int(round((f_max - BAND_LO_HZ) / BAND_WIDTH_HZ)))  # 4
        groups = np.array_split(np.arange(lo, n_freq), n_bands)
        # 带内取功率均值再转 dB (同 audioscope computeBandMeanPower), 而非 dB 均值
        hi = np.stack([10.0 * np.log10(np.mean(10.0 ** (mag_db[g] / 10.0), axis=0) + 1e-14)
                       for g in groups])
        hi_edges = np.linspace(BAND_LO_HZ, f_max, n_bands + 1)  # 20k,21k,22k,23k,24k
        display = np.concatenate([lin, hi], axis=0)
        edges = np.concatenate([lin_edges, hi_edges[1:]])
    else:
        display, edges = lin, lin_edges
    return display, edges


def prepare(audio, fs=FS):
    """音频 → (display (R,T), y_edges, times) —— 供绘图直接用。"""
    mag_db, freqs, times = compute_spectrogram(audio, fs)
    display, edges = hybrid_axis(mag_db, freqs)
    return display, edges, times


def waveform_envelope(audio, fs=FS, max_points=2000):
    """音频 → (t, lo, hi): 分段 min/max 包络, 用于 ±1 满幅波形。

    把整段音频等分成约 max_points 段, 每段取最小/最大值, 既保留瞬态又极少点数。
    """
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    step = max(1, n // max_points)
    trim = (n // step) * step
    seg = audio[:trim].reshape(-1, step)
    lo = seg.min(axis=1)
    hi = seg.max(axis=1)
    t = (np.arange(seg.shape[0]) + 0.5) * step / fs
    return t, lo, hi


def plot_spectrogram(audio, ax, title="Spectrogram", fs=FS, vmin=VMIN, vmax=VMAX):
    """把整段音频的频谱画到给定 matplotlib Axes (混合轴)。"""
    display, edges, times = prepare(audio, fs)
    x_edges = np.concatenate([times, [times[-1] + HOP / fs]]) if len(times) else np.array([0.0, HOP / fs])
    im = ax.pcolormesh(x_edges, edges, display, shading="flat", cmap=CMAP,
                       vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim([0, BAND_HI_HZ])
    # 20 kHz 分界处画一条细线, 提示高频段已按 1 kHz 分带
    ax.axhline(BAND_LO_HZ, color="white", linewidth=0.6, alpha=0.5)
    return im


def new_figure(audio, title="Spectrogram", fs=FS, figsize=(12, 4)):
    """独立 Figure (给 Tk 嵌入用; 后端由嵌入方决定, 此处不强制)。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = plot_spectrogram(audio, ax, title=title, fs=fs)
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    return fig

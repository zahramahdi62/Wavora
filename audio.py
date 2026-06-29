"""
Real-Time Audio Equalizer

    AudioStream         -> Microphone and speaker connection (audio input/output)
    FFTProcessor        -> Frequency domain transformation (FFT), windowing, reconstruction (IFFT + Overlap-Add)
    Equalizer           -> Applying gain to any frequency band
    PresetManager       -> Save/restore presets as JSON
    SpectrumVisualizer  -> Live spectrum display with matplotlib in Tkinter
    EqualizerGUI        -> Layout of the user interface and connecting all the parts together
    main()              -> The starting point of the program

 Why these defaults (DSP parameter choices)

SAMPLE_RATE = 44100 Hz
    Standard CD-quality rate, covers full 20 Hz–20 kHz hearing range (Nyquist = 22.05 kHz), supported natively by virtually every audio interface
    
BLOCK_SIZE = 1024 samples
    At 44.1 kHz this is ~23 ms of audio — small enough for low perceived latency, large enough that FFT bin resolution is usable (43 Hz/bin) and CPU overhead per callback stays low
    
FFT_SIZE = 2048 (zero-padded from 1024)
    Zero-padding doubles frequency resolution (~21.5 Hz/bin) without needing a longer analysis window, improving accuracy of low-band gain mapping (Sub Bass 20–60 Hz only spans ~2 bins at 1024-point resolution — too coarse)

Window function Hann 
   Smooths block edges to suppress spectral leakage; required because we're chopping a continuous signal into finite blocks (rectangular truncation creates false frequencies)

Overlap 50% (hop size = 512)
    Standard for Hann windows — using Overlap-Add (OLA) reconstruction, 50% overlap with Hann satisfies the constant-overlap-add (COLA) condition, avoiding amplitude ripple in reconstructed audio

"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Callable, Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal.windows import hann

# All constants in one place, to avoid "magic numbers" scattered throughout the code

SAMPLE_RATE: int = 44_100          # Sampling rate (Hz)
BLOCK_SIZE: int = 1024             # Size of each audio input/output block
HOP_SIZE: int = BLOCK_SIZE // 2    # Step forward between blocks
FFT_SIZE: int = 2048               #FFT size
CHANNELS: int = 1                  # Single-channel for DSP simplicity

GAIN_MIN_DB: float = -12.0
GAIN_MAX_DB: float = 12.0

#Definition of equalizer frequency bands: Name -> (Low frequency Hz, High frequency Hz)

BAND_DEFINITIONS: dict[str, tuple[float, float]] = {
    "Sub Bass":  (20.0, 60.0),
    "Bass":      (60.0, 250.0),
    "Mid":       (250.0, 2_000.0),
    "Upper Mid": (2_000.0, 4_000.0),
    "Treble":    (4_000.0, 20_000.0),
}

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

"""Converting audio blocks between time domain and frequency domain with Overlap-Add reconstruction."""
class FFTProcessor:

    def __init__(self, block_size: int = BLOCK_SIZE, fft_size: int = FFT_SIZE,
                 hop_size: int = HOP_SIZE) -> None:
        self.block_size = block_size
        self.fft_size = fft_size
        self.hop_size = hop_size

        
        self.window = hann(block_size, sym=False).astype(np.float32)

        self._ola_buffer = np.zeros(fft_size, dtype=np.float32)

        self.last_magnitude: np.ndarray = np.zeros(fft_size // 2 + 1, dtype=np.float32)

        self.freq_bins: np.ndarray = np.fft.rfftfreq(fft_size, d=1.0 / SAMPLE_RATE)
        
    """It transforms a time block into the frequency domain."""
    def forward(self, time_block: np.ndarray) -> np.ndarray:
        windowed = time_block * self.window
        padded = np.zeros(self.fft_size, dtype=np.float32)
        padded[: self.block_size] = windowed

        spectrum = np.fft.rfft(padded)
        self.last_magnitude = np.abs(spectrum)
        return spectrum

    """ It returns the frequency spectrum (previously gained by the Equalizer) to the time domain and reconstructs it continuously using the Overlap-Add technique."""
    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        time_block = np.fft.irfft(spectrum, n=self.fft_size).astype(np.float32)
        self._ola_buffer += time_block
        output_chunk = self._ola_buffer[: self.hop_size].copy()
        self._ola_buffer = np.roll(self._ola_buffer, -self.hop_size)
        self._ola_buffer[-self.hop_size:] = 0.0
        return output_chunk
    
"""The logic of "inter-frequency matching" (which FFT band belongs to which band) and the conversion of the dB slider value to a linear gain coefficient are completely independent of the FFT and Audio."""
class Equalizer:
    
    """ Mapping FFT bins to frequency bands and applying independent gain to each band."""
    def __init__(self, freq_bins: np.ndarray) -> None:
        self.freq_bins = freq_bins
        self.band_names = list(BAND_DEFINITIONS.keys())
        self._band_masks: dict[str, np.ndarray] = {
            name: (freq_bins >= low) & (freq_bins < high)
            for name, (low, high) in BAND_DEFINITIONS.items()
        }
        self._lock = threading.Lock()
        self._gains_db: dict[str, float] = {name: 0.0 for name in self.band_names}
        self._gain_array = np.ones_like(freq_bins, dtype=np.float32)
        self._rebuild_gain_array()

    """" Updates the gain value of a band (called by the GUI slider) """
    def set_band_gain_db(self, band_name: str, gain_db: float) -> None:
        gain_db = float(np.clip(gain_db, GAIN_MIN_DB, GAIN_MAX_DB))
        with self._lock:
            self._gains_db[band_name] = gain_db
            self._rebuild_gain_array()

    def get_band_gain_db(self, band_name: str) -> float:
        with self._lock:
            return self._gains_db[band_name]

    def get_all_gains_db(self) -> dict[str, float]:
        with self._lock:
            return dict(self._gains_db)

    """ To load a preset -- sets all bands at once."""
    def set_all_gains_db(self, gains: dict[str, float]) -> None:
        with self._lock:
            for name, value in gains.items():
                if name in self._gains_db:
                    self._gains_db[name] = float(np.clip(value, GAIN_MIN_DB, GAIN_MAX_DB))
            self._rebuild_gain_array()

    """
    Convert decibels to linear coefficients for each FFT interval:
    gain_linear = 10 ^ (dB / 20)
    This is the standard formula for converting decibels of amplitude to linear coefficients.
    """
    def _rebuild_gain_array(self) -> None:
        gain_array = np.ones_like(self.freq_bins, dtype=np.float32)
        for name in self.band_names:
            db = self._gains_db[name]
            linear_gain = 10.0 ** (db / 20.0)
            gain_array[self._band_masks[name]] = linear_gain
        self._gain_array = gain_array

    """ Applying gain to the frequency spectrum. """
    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        with self._lock:
            gain_array = self._gain_array
        return spectrum * gain_array

# This class reads the microphone, passes the block to an external callback function for processing, and sends the result to the speaker. Device errors (no microphone/speaker, connection lost) are also handled here.
class AudioStream:
    """ Manage two-way audio streaming (microphone input + speaker output) with sounddevice."""

    def __init__(self, process_block: Callable[[np.ndarray], np.ndarray],
                 on_status_change: Callable[[str], None]) -> None:
        self._process_block = process_block
        self._on_status_change = on_status_change
        self._stream: Optional[sd.Stream] = None
        self._is_running = False

    @staticmethod
    def has_input_device() -> bool:
        try:
            devices = sd.query_devices()
            return any(d["max_input_channels"] > 0 for d in devices)
        except Exception:
            return False

    @staticmethod
    def has_output_device() -> bool:
        try:
            devices = sd.query_devices()
            return any(d["max_output_channels"] > 0 for d in devices)
        except Exception:
            return False

    def _callback(self, indata: np.ndarray, outdata: np.ndarray,
                    frames: int, time_info, status: sd.CallbackFlags) -> None:
        if status:
            self._on_status_change(f"هشدار استریم صوتی: {status}")

        mono_in = indata[:, 0]
        try:
            processed = self._process_block(mono_in)
        except Exception as exc:
            self._on_status_change(f"خطا در پردازش بلاک صوتی: {exc}")
            processed = np.zeros(frames, dtype=np.float32)

        processed = np.clip(processed, -1.0, 1.0)
        outdata[:, 0] = processed

    def start(self) -> None:
        if self._is_running:
            return
        if not self.has_input_device():
            raise RuntimeError("No microphone found.")
        if not self.has_output_device():
            raise RuntimeError("No speakers found.")

        try:
            self._stream = sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=HOP_SIZE,       
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._is_running = True
            self._on_status_change("Recording from microphone")
        except sd.PortAudioError as exc:
            self._is_running = False
            raise RuntimeError(f"Audio device error: {exc}") from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except sd.PortAudioError:
                pass
        self._stream = None
        self._is_running = False
        self._on_status_change("It stopped.")

    @property
    def is_running(self) -> bool:
        return self._is_running


# PRESET MANAGER
class PresetManager:
    
    """Load and save gain presets as a JSON file."""
    BUILTIN_PRESETS: dict[str, dict[str, float]] = {
        "Flat":      {"Sub Bass": 0, "Bass": 0, "Mid": 0, "Upper Mid": 0, "Treble": 0},
        "Rock":      {"Sub Bass": 4, "Bass": 3, "Mid": -2, "Upper Mid": 2, "Treble": 4},
        "Pop":       {"Sub Bass": 1, "Bass": 2, "Mid": 1, "Upper Mid": 3, "Treble": 2},
        "Jazz":      {"Sub Bass": 2, "Bass": 1, "Mid": 0, "Upper Mid": 1, "Treble": 2},
        "Classical": {"Sub Bass": 0, "Bass": 0, "Mid": 0, "Upper Mid": -1, "Treble": 1},
        "Voice":     {"Sub Bass": -6, "Bass": -3, "Mid": 5, "Upper Mid": 4, "Treble": -1},
    }

    def __init__(self, presets_dir: str = PRESETS_DIR) -> None:
        self.presets_dir = presets_dir
        os.makedirs(self.presets_dir, exist_ok=True)
        self._ensure_builtin_files_exist()

    def _ensure_builtin_files_exist(self) -> None:
        for name, gains in self.BUILTIN_PRESETS.items():
            path = self._path_for(name)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(gains, f, indent=2, ensure_ascii=False)

    def _path_for(self, name: str) -> str:
        safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-"))
        return os.path.join(self.presets_dir, f"{safe_name}.json")

    def load(self, name: str) -> dict[str, float]:
        path = self._path_for(name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, gains: dict[str, float]) -> None:
        path = self._path_for(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gains, f, indent=2, ensure_ascii=False)

    def list_presets(self) -> list[str]:
        names = []
        for fname in sorted(os.listdir(self.presets_dir)):
            if fname.endswith(".json"):
                names.append(os.path.splitext(fname)[0])
        return names


# This class only reads the latest spectrum data and updates the graph every few milliseconds (with a Tkinter timer) -- without completely redrawing the axes (which causes flicker and slowness).
class SpectrumVisualizer:
    
    """Live frequency spectrum display with logarithmic X-axis, grid and optional Peak-Hold."""
    def __init__(self, parent: tk.Widget, freq_bins: np.ndarray) -> None:
        self.freq_bins = freq_bins
        self._peak_hold = np.zeros_like(freq_bins)
        self._smoothed = np.zeros_like(freq_bins)
        self._smoothing_factor = 0.6  

        self.figure = plt.Figure(figsize=(6, 3.2), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.ax.set_xscale("log")
        self.ax.set_xlim(20, SAMPLE_RATE / 2)
        self.ax.set_ylim(0, 50)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Domain")
        self.ax.set_title("Live frequency spectrum")
        self.ax.grid(True, which="both", linestyle="--", alpha=0.4)

        (self._line,) = self.ax.plot(freq_bins, np.zeros_like(freq_bins),
                                      color="#1f77b4", linewidth=1.2, label="Spectrum")
        (self._peak_line,) = self.ax.plot(freq_bins, np.zeros_like(freq_bins),
                                           color="#d62728", linewidth=0.8,
                                           alpha=0.6, label="Peak Hold")
        self.ax.legend(loc="upper right", fontsize=8)
        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    """Updates the graph with the latest spectrum data (called on the GUI thread)."""
    def update(self, magnitude: np.ndarray) -> None:
        self._smoothed = (self._smoothing_factor * self._smoothed +
                           (1 - self._smoothing_factor) * magnitude)

        self._peak_hold = np.maximum(self._peak_hold * 0.95, magnitude)

        self._line.set_ydata(self._smoothed)
        self._peak_line.set_ydata(self._peak_hold)
        self.canvas.draw_idle()


# Eliminate sound noise
class NoiseReducer:
    def __init__(self, n_bins: int, reduction_strength: float = 1.0):
        self.noise_profile = np.zeros(n_bins, dtype=np.float32)
        self.reduction_strength = reduction_strength  
        self._calibrating = False
        self._calib_samples = []

    def start_calibration(self):
        self._calibrating = True
        self._calib_samples = []

    def feed_calibration(self, magnitude: np.ndarray):
        self._calib_samples.append(magnitude)

    def finish_calibration(self):
        if self._calib_samples:
            self.noise_profile = np.mean(self._calib_samples, axis=0)
        self._calibrating = False

    def apply(self, spectrum: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        
        noise_est = self.noise_profile * self.reduction_strength + 1e-8
        gain_mask = np.clip(1.0 - (noise_est / (magnitude + 1e-8)), 0.0, 1.0)
        return spectrum * gain_mask


# All widget creation, layout and event binding in one place. This class itself has no DSP logic; it just holds the AudioStream / Equalizer / SpectrumVisualizer / PresetManager and binds user events to them.
class EqualizerGUI:
    
    """Main Tkinter window: Start/Stop buttons, sliders, presets, and spectrum graph."""
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Real-Time Audio Equalizer")
        self.root.geometry("780x640")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fft_processor = FFTProcessor()
        self.noise_reducer = NoiseReducer(self.fft_processor.freq_bins.shape[0])
        self.equalizer = Equalizer(self.fft_processor.freq_bins)
        self.preset_manager = PresetManager()
        self.audio_stream = AudioStream(
            process_block=self._process_audio_block,
            on_status_change=self._set_status_threadsafe,
        )

        self._build_widgets()

        self._schedule_visualizer_update()

    # UI
    def _build_widgets(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        self.start_button = ttk.Button(top_frame, text="▶ start", command=self._on_start)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(top_frame, text="■ Stop", command=self._on_stop,
                                       state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.status_label = ttk.Label(top_frame, text="Ready. Click Start to begin.")
        self.status_label.pack(side="left", padx=15)

        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.pack(fill="both", expand=True)
        self.visualizer = SpectrumVisualizer(plot_frame, self.fft_processor.freq_bins)

        sliders_frame = ttk.LabelFrame(self.root, text="Frequency bands (dB)", padding=10)
        sliders_frame.pack(fill="x", padx=10, pady=5)

        self._sliders: dict[str, ttk.Scale] = {}
        for col, (band_name, (low, high)) in enumerate(BAND_DEFINITIONS.items()):
            band_frame = ttk.Frame(sliders_frame)
            band_frame.grid(row=0, column=col, padx=10)

            label_text = f"{band_name}\n({low:.0f}-{high:.0f} Hz)"
            ttk.Label(band_frame, text=label_text, anchor="center").pack()

            value_label = ttk.Label(band_frame, text="0.0 dB")
            value_label.pack()

            slider = ttk.Scale(
                band_frame, from_=GAIN_MAX_DB, to=GAIN_MIN_DB, orient="vertical",
                length=160,
                command=lambda val, name=band_name, lbl=value_label: self._on_slider_change(name, val, lbl),
            )
            slider.set(0.0)
            slider.pack()
            self._sliders[band_name] = slider

        preset_frame = ttk.LabelFrame(self.root, text="Presets", padding=10)
        preset_frame.pack(fill="x", padx=10, pady=5)
        for name in self.preset_manager.list_presets():
            ttk.Button(preset_frame, text=name,
                       command=lambda n=name: self._apply_preset(n)).pack(side="left", padx=4)
        ttk.Button(preset_frame, text="Save current status...",
                   command=self._save_current_as_preset).pack(side="left", padx=10)

    # callbacks
    def _on_slider_change(self, band_name: str, value: str, label: ttk.Label) -> None:
        gain_db = float(value)
        self.equalizer.set_band_gain_db(band_name, gain_db)
        label.config(text=f"{gain_db:.1f} dB")

    def _apply_preset(self, name: str) -> None:
        try:
            gains = self.preset_manager.load(name)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            messagebox.showerror("Error loading preset", str(exc))
            return
        self.equalizer.set_all_gains_db(gains)
        for band_name, slider in self._sliders.items():
            slider.set(gains.get(band_name, 0.0))

    def _save_current_as_preset(self) -> None:
        name = tk.simpledialog.askstring("ave Preset", "Enter the name of the preset:")
        if not name:
            return
        self.preset_manager.save(name, self.equalizer.get_all_gains_db())
        messagebox.showinfo("Saved", f"preset «{name}» Saved.")

    def _on_start(self) -> None:
        try:
            self.audio_stream.start()
        except RuntimeError as exc:
            messagebox.showerror("Audio device error", str(exc))
            return
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

    def _on_stop(self) -> None:
        self.audio_stream.stop()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def _on_close(self) -> None:
        self.audio_stream.stop()
        self.root.destroy()

    # audio thread
    """
    This function runs on the real-time audio thread (not the GUI thread).
    The complete processing chain is: FFT -> Apply Gain -> IFFT.
    """
    def _process_audio_block(self, mono_in: np.ndarray) -> np.ndarray:
        padded_input = np.zeros(BLOCK_SIZE, dtype=np.float32)
        padded_input[: len(mono_in)] = mono_in
          
        spectrum = self.fft_processor.forward(padded_input)
        spectrum = self.noise_reducer.apply(spectrum, self.fft_processor.last_magnitude)
        shaped_spectrum = self.equalizer.apply(spectrum)
        output_block = self.fft_processor.inverse(shaped_spectrum)
        return output_block

    # status
    """Since this function might be called from the audio thread, we delegate the UI update to the main thread."""
    def _set_status_threadsafe(self, message: str) -> None:
        self.root.after(0, lambda: self.status_label.config(text=message))

    # visualizer
    """Reads the latest spectrum and updates the graph every ~33ms (~30fps)."""
    def _schedule_visualizer_update(self) -> None:
        magnitude = self.fft_processor.last_magnitude
        self.visualizer.update(magnitude)
        self.root.after(33, self._schedule_visualizer_update)


# MAIN
def main() -> None:
    import tkinter.simpledialog  # noqa: F401 -- Ensure import before use

    if not AudioStream.has_input_device() or not AudioStream.has_output_device():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Audio device not found",
            "No valid microphone or speakers were found on this system.\n"
            "Please connect the audio device and try again.",
        )
        return

    root = tk.Tk()
    app = EqualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
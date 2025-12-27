import streamlit as st
import numpy as np
import wave
import os
import matplotlib.pyplot as plt
from io import BytesIO

# ==========================================
# 1. CORE LOGIC & STRATEGIES
# ==========================================

def stream_audio_file(file_obj, chunk_duration_ms=50):
    """
    Generator simulating a live stream.
    Accepts a file-like object (uploaded file) or path.
    """
    try:
        with wave.open(file_obj, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            chunk_size = int(sample_rate * (chunk_duration_ms / 1000))
            current_time = 0.0
            
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0: break
                
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                if n_channels > 1:
                    audio_chunk = audio_chunk.reshape(-1, n_channels).mean(axis=1)
                
                if len(audio_chunk) < chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)))
                
                yield audio_chunk, sample_rate, current_time
                current_time += (chunk_duration_ms / 1000)
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
        return

class BaseStrategy:
    def process(self, chunk, sr, current_time):
        raise NotImplementedError

class SilenceStrategy(BaseStrategy):
    """
    Tracks RMS energy. If energy stays below threshold for X seconds
    (after the warmup period), it triggers a drop.
    """
    def __init__(self, warmup=2.0, silence_thresh=1.5, energy_floor=500):
        self.warmup = warmup
        self.limit = silence_thresh
        self.floor = energy_floor
        self.counter = 0.0

    def process(self, chunk, sr, current_time):
        if current_time < self.warmup:
            self.counter = 0.0
            return False, None

        rms = np.sqrt(np.mean(chunk.astype(float)**2))
        chunk_sec = len(chunk) / sr

        if rms < self.floor:
            self.counter += chunk_sec
        else:
            self.counter = 0.0
            
        if self.counter >= self.limit:
            return True, f"Silence Timeout ({self.limit}s)"
        return False, None

class BeepStrategy(BaseStrategy):
    """
    Uses FFT to find the dominant frequency. If the loudest frequency
    is within the target range for a minimum duration, it triggers.
    """
    def __init__(self, target_freq=800, tolerance=100, min_dur=0.15, min_amp=2000):
        self.freq_min = target_freq - tolerance
        self.freq_max = target_freq + tolerance
        self.min_dur = min_dur
        self.min_amp = min_amp
        self.counter = 0.0

    def process(self, chunk, sr, current_time):
        # FFT Analysis
        spectrum = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), 1/sr)
        mags = np.abs(spectrum)
        
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        peak_amp = mags[peak_idx]
        
        chunk_sec = len(chunk) / sr
        
        # Check if loudest sound is our beep
        is_match = (peak_amp > self.min_amp) and (self.freq_min <= peak_freq <= self.freq_max)
        
        if is_match:
            self.counter += chunk_sec
        else:
            self.counter = 0.0
            
        if self.counter >= self.min_dur:
            return True, f"Beep Detected ({peak_freq:.0f}Hz)"
        return False, None

class CombinedStrategy(BaseStrategy):
    """
    Runs both Silence and Beep strategies in parallel.
    Prioritizes Beep detection (Fast Lane).
    Falls back to Silence detection (Slow Lane).
    """
    def __init__(self, warmup, silence_thresh, target_freq, tolerance, min_dur):
        self.silence_engine = SilenceStrategy(warmup, silence_thresh)
        self.beep_engine = BeepStrategy(target_freq, tolerance, min_dur)

    def process(self, chunk, sr, current_time):
        # 1. Check Beep (Priority)
        drop_beep, reason_beep = self.beep_engine.process(chunk, sr, current_time)
        if drop_beep:
            return True, reason_beep
            
        # 2. Check Silence (Fallback)
        drop_silence, reason_silence = self.silence_engine.process(chunk, sr, current_time)
        if drop_silence:
            return True, reason_silence
            
        return False, None

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================

st.set_page_config(page_title="Voicemail Drop Simulator", layout="wide")

st.title("📞 Smart Voicemail Drop Simulator")
st.markdown("""
This tool simulates streaming audio processing to determine the exact millisecond to leave a compliant voicemail.
""")

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("1. Select Strategy")
    strategy_mode = st.radio(
        "Detection Logic",
        ["Combined (Recommended)", "Silence Only", "Beep Only"]
    )
    
    st.divider()
    
    st.header("2. Hyperparameters")
    
    # Dynamic sliders based on strategy
    if strategy_mode in ["Combined (Recommended)", "Silence Only"]:
        st.subheader("Silence Settings")
        silence_thresh = st.slider("Silence Threshold (sec)", 1.0, 5.0, 2.5, help="How long to wait after talking stops")
        warmup_time = st.slider("Warmup / Safe Start (sec)", 0.0, 5.0, 2.0, help="Ignore silence during this start period")
    else:
        # Default values if hidden
        silence_thresh = 2.5
        warmup_time = 2.0

    if strategy_mode in ["Combined (Recommended)", "Beep Only"]:
        st.subheader("Beep Settings")
        target_freq = st.slider("Target Frequency (Hz)", 400, 2000, 1000)
        tolerance = st.slider("Frequency Tolerance (+/- Hz)", 10, 300, 100)
        min_dur = st.slider("Min Beep Duration (sec)", 0.05, 0.5, 0.1, step=0.05)
    else:
        target_freq = 1000
        tolerance = 100
        min_dur = 0.1

    st.divider()
    
    st.header("3. Audio Source")
    
    # -- PRE-LOADED FILES LOGIC --
    # Put your local .wav files in a folder named 'samples' or just listing them here
    # For this code to work out-of-the-box, I'll allow manual path entry or upload
    input_method = st.radio("Input Method", ["Upload File", "Select Pre-loaded"])
    
    audio_file = None
    file_name_display = ""
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload .wav", type=["wav"])
        if uploaded_file:
            audio_file = uploaded_file
            file_name_display = uploaded_file.name
            
    else:
        # SCAN LOCAL DIRECTORY FOR WAV FILES
        # Ensure you have some .wav files in the same folder as this script
        local_files = [f for f in os.listdir('.') if f.endswith('.wav')]
        if not local_files:
            st.warning("No .wav files found in current directory.")
        else:
            selected_file = st.selectbox("Choose a file", local_files)
            if selected_file:
                audio_file = open(selected_file, 'rb')
                file_name_display = selected_file

# --- MAIN AREA ---

# 1. Strategy Description Box
st.info(f"**Current Strategy: {strategy_mode}**")
if strategy_mode == "Silence Only":
    st.markdown(f"> Waits for the user to be silent for **{silence_thresh}s**. Ignores the first **{warmup_time}s** (Warmup).")
elif strategy_mode == "Beep Only":
    st.markdown(f"> Scans specifically for a tone around **{target_freq}Hz** (+/- {tolerance}Hz) lasting > **{min_dur}s**.")
else:
    st.markdown(f"> **Hybrid Approach:** Prioritizes detecting a **{target_freq}Hz** beep. If no beep is found, falls back to waiting for **{silence_thresh}s** of silence.")

# 2. Run Simulation
if audio_file:
    # Read full file for visualization plotting
    # We need to rewind the file pointer if it's an opened file
    if hasattr(audio_file, 'seek'):
        audio_file.seek(0)
    
    # Use scipy/wave to read full data for plot
    try:
        import scipy.io.wavfile as wav
        # We need BytesIO if it's an uploaded file object
        if hasattr(audio_file, 'read'):
            audio_bytes = audio_file.read()
            sr, audio_data = wav.read(BytesIO(audio_bytes))
            # Reset pointer for streaming later
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)
            else:
                # If we read it all, we might need to re-wrap for streaming
                audio_file = BytesIO(audio_bytes)
        else:
            sr, audio_data = wav.read(audio_file)
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        duration = len(audio_data) / sr
        
        st.audio(audio_file) # Standard Player
        
        if st.button("🚀 Run Simulation", type="primary"):
            
            # Init Engine
            if strategy_mode == "Silence Only":
                engine = SilenceStrategy(warmup_time, silence_thresh)
            elif strategy_mode == "Beep Only":
                engine = BeepStrategy(target_freq, tolerance, min_dur)
            else:
                engine = CombinedStrategy(warmup_time, silence_thresh, target_freq, tolerance, min_dur)
            
            # --- RUN STREAMING LOOP ---
            # Reset file pointer again for the simulation
            if hasattr(audio_file, 'seek'): audio_file.seek(0)
            
            stream = stream_audio_file(audio_file, chunk_duration_ms=50)
            
            drop_time = None
            drop_reason = "No Drop Triggered"
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for chunk, stream_sr, cur_time in stream:
                # Update progress
                prog = min(cur_time / duration, 1.0)
                progress_bar.progress(prog)
                
                # Check Logic
                should_drop, reason = engine.process(chunk, stream_sr, cur_time)
                
                if should_drop:
                    drop_time = cur_time
                    drop_reason = reason
                    break
            
            progress_bar.progress(100)
            
            # --- VISUALIZATION ---
            st.divider()
            
            col_res, col_plot = st.columns([1, 2])
            
            with col_res:
                if drop_time:
                    st.success("✅ **Voicemail Dropped!**")
                    st.metric("Timestamp", f"{drop_time:.2f}s")
                    st.info(f"**Reason:**\n{drop_reason}")
                else:
                    st.error("❌ Call Ended Without Drop")
                    st.write("Criteria were not met before file ended.")

            with col_plot:
                st.subheader("Waveform Analysis")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                times = np.linspace(0, duration, len(audio_data))
                ax.plot(times, audio_data, color='lightgray', label="Audio Signal")
                
                # 1. Highlight Warmup Zone
                if strategy_mode != "Beep Only":
                    ax.axvspan(0, warmup_time, color='yellow', alpha=0.2, label="Warmup (Ignored)")
                
                # 2. Highlight Drop Line
                if drop_time:
                    ax.axvline(drop_time, color='red', linewidth=2, linestyle='--', label="DROP Point")
                    # Green zone after drop
                    ax.axvspan(drop_time, duration, color='green', alpha=0.1, label="Message Playing")
                
                ax.set_yticks([])
                ax.set_xlim(0, duration)
                ax.set_xlabel("Time (seconds)")
                ax.legend(loc='upper right')
                
                # Clean plot style
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not process file. Ensure it is a valid WAV. Error: {e}")

else:
    st.info("👈 Please upload a file or select one from the sidebar to begin.")
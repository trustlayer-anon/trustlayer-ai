import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import random
import hashlib
try:
    from pydub import AudioSegment
    pydub_available = True
except ImportError:
    pydub_available = False
    print("[Warning] pydub not installed; only WAV supported.")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def select_source_segment(audio, fs, seed, duration_sec=5):
    length = len(audio)
    random.seed(seed)
    attempts = 0
    while attempts < 20:
        start = random.randint(0, length - int(duration_sec * fs))
        segment = audio[start:start + int(duration_sec * fs)]
        if np.std(segment) > 0.01 * np.max(np.abs(audio)):
            stretched = np.interp(
                np.linspace(0, len(segment), length),
                np.arange(len(segment)),
                segment
            )
            return stretched, False
        attempts += 1
    
    t = np.linspace(0, length / fs, length)
    pattern_id = random.randint(0, 2)
    if pattern_id == 0:
        fallback = np.sin(2 * np.pi * (21000 + 1500 * t / (length/fs)) * t)
    elif pattern_id == 1:
        fallback = np.sin(2 * np.pi * (24000 - 2000 * t / (length/fs)) * t)
    else:
        fallback = np.random.randn(length) * np.sin(2 * np.pi * 22500 * t)
    print(f"[TrustLayer Audio] Fallback pattern {pattern_id} used")
    return fallback * 0.1, True

def encode_audio(input_path, message, output_path):
    if pydub_available:
        try:
            audio_seg = AudioSegment.from_file(input_path)
            fs = audio_seg.frame_rate
            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.channels > 1:
                samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)
            audio = samples / 32768.0
        except Exception as e:
            print(f"[Error] pydub load failed: {e}")
            sys.exit(1)
    else:
        try:
            fs, data = wavfile.read(input_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            audio = data.astype(float) / np.abs(data).max()
        except Exception as e:
            print(f"[Error] WAV load failed: {e}")
            sys.exit(1)
    
    seed = int(hashlib.sha256(message.encode() if message else b'default').hexdigest(), 16) % (2**32)
    source, _ = select_source_segment(audio, fs, seed)
    
    lowcut_u, highcut_u = 21000, 24000
    if highcut_u <= fs / 2:
        b_u, a_u = butter_bandpass(lowcut_u, highcut_u, fs)
        source_u = lfilter(b_u, a_u, source)
    
    lowcut_i, highcut_i = 10, 18
    b_i, a_i = butter_bandpass(lowcut_i, highcut_i, fs)
    source_i = lfilter(b_i, a_i, source)
    
    embedded = audio.copy()
    if 'source_u' in locals():
        embedded += source_u * 0.05
    embedded += source_i * 0.05
    embedded = np.clip(embedded, -1, 1)
    
    wavfile.write(output_path, fs, (embedded * 32767).astype(np.int16))
    print(f"[TrustLayer Audio] Stamped saved: {output_path}")

def decode_audio(input_path):
    if pydub_available:
        try:
            audio_seg = AudioSegment.from_file(input_path)
            fs = audio_seg.frame_rate
            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.channels > 1:
                samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)
            audio = samples / 32768.0
        except Exception as e:
            print(f"[Error] pydub load failed: {e}")
            sys.exit(1)
    else:
        try:
            fs, data = wavfile.read(input_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            audio = data.astype(float) / 32767
        except Exception as e:
            print(f"[Error] WAV load failed: {e}")
            sys.exit(1)
    
    extracted_u = np.zeros_like(audio)
    lowcut_u, highcut_u = 21000, 24000
    if highcut_u <= fs / 2:
        b_u, a_u = butter_bandpass(lowcut_u, highcut_u, fs)
        extracted_u = lfilter(b_u, a_u, audio)
    
    lowcut_i, highcut_i = 10, 18
    b_i, a_i = butter_bandpass(lowcut_i, highcut_i, fs)
    extracted_i = lfilter(b_i, a_i, audio)
    
    extracted = (extracted_u + extracted_i) * 50
    extracted = np.clip(extracted, -1, 1)
    
    wavfile.write("reveal.wav", fs, (extracted * 32767).astype(np.int16))
    print("[TrustLayer Audio] Reveal saved as reveal.wav")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ["encode", "decode"]:
        print("Usage: python trustlayer_audio.py encode <input> <output> [message_seed]")
        print("       python trustlayer_audio.py decode <input>")
        sys.exit(1)
    
    if sys.argv[1] == "encode":
        message = sys.argv[4] if len(sys.argv) > 4 else ""
        encode_audio(sys.argv[2], message, sys.argv[3])
    else:
        decode_audio(sys.argv[2])
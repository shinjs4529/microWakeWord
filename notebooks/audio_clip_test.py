from microwakeword.inference import Model
from scipy.io import wavfile
import soundfile as sf
from absl import logging
import numpy as np
import os

m = Model(os.path.join("trained_models/wakeword", "tflite_stream_state_internal", "stream_state_internal.tflite"), stride=3)
#m = Model(os.path.join("trained_models/wakeword", "tflite_stream_state_internal_quant", "stream_state_internal_quant.tflite"), stride=3)
#m = Model(os.path.join("trained_models/wakeword", "tflite_non_stream", "non_stream_main.tflite"), stride=1)

filePath = "wavFileTest/ci_6.wav"
wav, sr = sf.read(filePath)
assert sr == 16_000, "16 kHz wav only"
if wav.ndim == 2:                                    # 스테레오면 평균
    wav = wav.mean(axis=1).astype(wav.dtype)
    print("1st if")
if wav.dtype != np.int16:
    wav = np.clip(wav*32767, -32768, 32767).astype(np.int16)

probs = m.predict_clip(wav, 20)      # ← list of probabilities (per 10 ms frame)
#print("probs:", probs)
print("max prob:", max(probs))

"""
Using sound matching to find game starts.
Not relaible
"""

import os, subprocess, tempfile, time
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

VIDEO_PATH   = "full_vod_check.mp4"
TEMPLATE_WAV = "match_found_template.wav"
SR           = 16_000         
BLOCK_SEC    = 60              
THRESHOLD    = 0.50          
MERGE_SEC    = 0.8           

def extract_pcm_mono(src, out_wav, sr=SR):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-i", src, "-vn",
         "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
         out_wav],
        check=True)

def norm_z(x: np.ndarray):
    x = x.astype(np.float32)
    x -= x.mean()
    n = np.linalg.norm(x)
    return x / n if n else x

def main():
    wall_start = time.time()

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    t0 = time.time()
    extract_pcm_mono(VIDEO_PATH, tmp_wav)
    t1 = time.time()
    print(f"Audio extraction   : {t1 - t0:6.2f} s")

    t_audio, _ = sf.read(TEMPLATE_WAV, dtype="float32")
    if t_audio.ndim == 2:                  
        t_audio = t_audio.mean(axis=1)
    t_audio = norm_z(t_audio)             
    m = len(t_audio)

    block_len = int(BLOCK_SEC * SR)
    overlap   = m - 1
    hop       = 1 / SR

    peaks_sec = []
    offset_samples = 0
    t2 = time.time()

    with sf.SoundFile(tmp_wav) as f:
        total_blocks = int(np.ceil(len(f) / block_len))
        for block in tqdm(f.blocks(blocksize=block_len,
                           overlap=overlap,
                           dtype="float32"),
                  total=total_blocks,
                  desc="Scanning audio"):
            if block.ndim == 2:               
                block = block.mean(axis=1)
            y = norm_z(block)
            corr = fftconvolve(y, t_audio[::-1], mode="valid")


            locs = np.where(corr >= THRESHOLD)[0]
            for idx in locs:
                sec = (offset_samples + idx) * hop
                if not peaks_sec or sec - peaks_sec[-1] > MERGE_SEC:
                    peaks_sec.append(sec)

            offset_samples += len(block) - overlap

    t3 = time.time()
    print(f"Correlation pass   : {t3 - t2:6.2f} s")

    print("\nMATCH‑FOUND cue timestamps:")
    for ts in peaks_sec:
        h, m_, s = int(ts // 3600), int(ts % 3600 // 60), ts % 60
        print(f"  {h:02d}:{m_:02d}:{s:05.2f}")

    os.remove(tmp_wav)
    print(f"\nTotal elapsed      : {time.time() - wall_start:6.2f} s")

if __name__ == "__main__":
    main()

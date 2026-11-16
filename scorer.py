#!/usr/bin/env python3
"""Technical best-of selector. No neural model, no extra GPU memory.

Scores a candidate WAV on:
  * duration match to the source (when known),
  * clipping (1 s windows whose peak is at full scale),
  * silence (1 s windows below -45 dBFS),
  * loudness deviation from about -14 dBFS RMS.

This is a sanity/quality floor, not a perceptual judge. It deliberately does
not try to replace SongEval/TuneJury.
"""
import math
import os
import subprocess
import wave

try:
    import audioop  # Python <= 3.12
except ImportError:  # pragma: no cover - Python 3.13+
    audioop = None


def _duration(path):
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError):
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=30,
            )
            return float(proc.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return 0.0


def _chunk_metrics(chunk, width):
    if audioop is not None:
        peak = audioop.max(chunk, width)
        rms = audioop.rms(chunk, width)
        return peak / 32768.0, rms / 32768.0
    # Fallback without audioop: 16-bit little-endian only.
    import array
    samples = array.array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0, 0.0
    if width != 2:
        return max(abs(v) for v in samples) / 32768.0, 0.0
    peak = max(max(samples), -min(samples)) / 32768.0
    if len(samples) % 2 == 0:
        samples = samples[0::2]
    rms = math.sqrt(sum(v * v for v in samples) / max(1, len(samples))) / 32768.0
    return peak, rms


def metrics(path):
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        duration = frames / rate if rate else 0.0
        chunk_frames = max(1, rate)  # 1 second
        total = clipped = silent = 0
        rms_sum = 0.0
        peak = 0.0
        while True:
            chunk = handle.readframes(chunk_frames)
            if not chunk:
                break
            peak_chunk, rms_chunk = _chunk_metrics(chunk, width)
            total += 1
            peak = max(peak, peak_chunk)
            rms_sum += rms_chunk
            if peak_chunk >= 0.999:
                clipped += 1
            if rms_chunk <= 10 ** (-45.0 / 20.0):
                silent += 1
        rms = rms_sum / total if total else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-9))
    return {
        "duration_s": round(duration, 3),
        "sample_rate": rate,
        "channels": channels,
        "peak": round(peak, 6),
        "rms_db": round(rms_db, 2),
        "clipped_ratio": round(clipped / total, 4) if total else 1.0,
        "silent_ratio": round(silent / total, 4) if total else 1.0,
    }


def score(path, target_duration=None):
    """Return 0..1; higher is better. Raises only for unreadable files."""
    if not path or not os.path.isfile(path):
        raise ValueError("candidate file not found")
    info = metrics(path)
    duration = info["duration_s"]
    value = 1.0
    if target_duration and target_duration > 0:
        value -= 0.4 * min(1.0, abs(duration - target_duration) / target_duration)
    value -= 0.8 * info["clipped_ratio"]
    value -= 0.5 * info["silent_ratio"]
    value -= 0.05 * min(1.0, abs(info["rms_db"] + 14.0) / 20.0)
    if duration < 20:
        value -= 0.25
    return max(0.0, min(1.0, value))

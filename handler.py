#!/usr/bin/env python3
"""RunPod handler for YuE2: generation + real audio remix (SheetSage2 -> ABC -> YuE2).

Tasks (input.task):
  generate    lyrics/style (+ optional abc), lora_id/genre, seed, best_of
  remix       audio_base64 | audio_url | abc, genre, lyrics, style,
              instrumental?, seed, best_of
  transcribe  audio_base64 | audio_url -> ABC only

Remix is a real audio remix: the source audio is converted to WAV, SheetSage2
produces ABC, and YuE2 renders that score with cot=melody.  Passing abc directly
skips transcription (useful for repeat takes).

best_of generates N candidates with seed..seed+N-1 and selects one with a
technical selector (no extra model, no TuneJury).  It scores duration match,
clipping, silence and loudness; it is not a perceptual judge.

Output: base64 MP3 (default) or WAV, plus ABC/timing metadata.  The resident
audiocpp_server is never exposed; it listens on 127.0.0.1:8080.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse

import requests

BASE = os.environ.get("AUDIOCPP_BASE_URL", "http://127.0.0.1:8080")
CONFIG = os.environ.get("YUE2_CONFIG", "/app/server.json")
GENRES_PATH = os.environ.get("YUE2_GENRES", "/app/genres.json")
BITRATE = os.environ.get("YUE2_MP3_BITRATE", "192k")
MAX_AUDIO_BYTES = int(os.environ.get("YUE2_MAX_AUDIO_BYTES", str(80 * 1024 * 1024)))
MAX_MP3_BYTES = int(os.environ.get("YUE2_MAX_MP3_BYTES", str(18 * 1024 * 1024)))
MAX_WAV_BYTES = int(os.environ.get("YUE2_MAX_WAV_BYTES", str(80 * 1024 * 1024)))
MIN_SOURCE_SECONDS = float(os.environ.get("YUE2_MIN_SOURCE_SECONDS", "10"))
MAX_SOURCE_SECONDS = float(os.environ.get("YUE2_MAX_SOURCE_SECONDS", "300"))
MAX_LYRICS = 20000
MAX_ABC = 50000
ALLOWED_OPTIONS = {
    "guidance_scale": (int, float),
    "num_inference_steps": int,
    "semantic_temperature": (int, float),
    "semantic_top_p": (int, float),
    "semantic_top_k": int,
    "abc_temperature": (int, float),
    "abc_top_p": (int, float),
    "abc_top_k": int,
}
START_TS = time.time()
_server = None
_ready_at = None
_genres = None


def log(*args):
    print(*args, flush=True)


def genres():
    global _genres
    if _genres is None:
        with open(GENRES_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        data.pop("comment", None)
        _genres = data
    return _genres


def start_server():
    global _server, _ready_at
    t0 = time.time()
    _server = subprocess.Popen(
        ["/usr/local/bin/audiocpp_server", "--config", CONFIG, "--log"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    while time.time() < t0 + 1200:
        if _server.poll() is not None:
            raise RuntimeError(f"audiocpp_server exited early rc={_server.returncode}")
        try:
            if requests.get(f"{BASE}/health", timeout=5).ok:
                models = requests.get(f"{BASE}/v1/models", timeout=5)
                log(f"[yue2] server up in {time.time() - t0:.1f}s: {models.text[:300]}")
                _ready_at = time.time()
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("audiocpp_server not ready within 1200s")


def resolve_model(key):
    """Genre key or explicit model id -> (model_id, spec). Client never passes a path."""
    table = genres()
    if key in (None, "", "base", "yue2"):
        return "yue2-base", table["base"]
    if key in table:
        return table[key]["model_id"], table[key]
    by_id = {}
    for spec in table.values():
        by_id.setdefault(spec["model_id"], spec)
    if key in by_id:
        return key, by_id[key]
    raise ValueError(
        f"Unknown genre/lora_id '{key}'. Use one of {sorted(table)} or a model id."
    )


def _seed(inp):
    value = inp.get("seed", 12345)
    if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < 2**63):
        raise ValueError("seed must be an integer in [0, 2^63).")
    return value


def _best_of(inp):
    value = inp.get("best_of", 1)
    if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 8):
        raise ValueError("best_of must be an integer in [1, 8].")
    return value


def _selector(inp):
    value = inp.get("selector", "technical")
    if value != "technical":
        raise ValueError("selector must be 'technical' (no extra model installed).")
    return value


def _output_format(inp):
    value = inp.get("output_format", "mp3")
    if value not in ("mp3", "wav"):
        raise ValueError("output_format must be mp3 or wav.")
    return value


def _text(inp, key, limit):
    value = inp.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    if len(value) > limit:
        raise ValueError(f"{key} too long (max {limit} chars).")
    return value


def _options(inp):
    result = {}
    for key, value in (inp.get("options") or {}).items():
        if key not in ALLOWED_OPTIONS:
            raise ValueError(f"Unsupported option '{key}'. Allowed: {sorted(ALLOWED_OPTIONS)}")
        if isinstance(value, bool) or not isinstance(value, ALLOWED_OPTIONS[key]):
            raise ValueError(f"Option '{key}' has wrong type.")
        result[key] = value
    for key in ALLOWED_OPTIONS:
        if key in inp:
            result[key] = inp[key]
    return result


def _style(inp):
    style = _text(inp, "style", 4000).strip()
    if not style:
        raise ValueError("style is required.")
    return style


def _source(inp):
    audio_b64 = inp.get("audio_base64")
    audio_url = inp.get("audio_url")
    abc = _text(inp, "abc", MAX_ABC)
    supplied = [name for name, value in (
        ("audio_base64", audio_b64), ("audio_url", audio_url), ("abc", abc)
    ) if value]
    if len(supplied) != 1:
        raise ValueError("Provide exactly one of audio_base64, audio_url, abc.")
    return {"audio_base64": audio_b64, "audio_url": audio_url, "abc": abc or None,
            "kind": supplied[0]}


def parse_generate(inp):
    lyrics = _text(inp, "lyrics", MAX_LYRICS)
    style = _style(inp)
    if not lyrics.strip() and not inp.get("instrumental"):
        raise ValueError("lyrics is required for a vocal generation (or set instrumental=true).")
    if inp.get("instrumental"):
        lyrics = "[instrumental]"
    model_id, spec = resolve_model(inp.get("lora_id") or inp.get("genre"))
    options = {**spec.get("options", {}), **_options(inp)}
    cot = inp.get("cot", spec.get("cot", "off"))
    if cot not in ("off", "melody", "full"):
        raise ValueError("cot must be off|melody|full.")
    abc = _text(inp, "abc", MAX_ABC)
    if abc:
        options["abc"] = abc
        if cot == "off":
            cot = "melody"
    return {
        "task": "generate", "model_id": model_id, "lyrics": lyrics, "style": style,
        "cot": cot, "seed": _seed(inp), "best_of": _best_of(inp),
        "selector": _selector(inp), "options": options, "output_format": _output_format(inp),
        "genre": inp.get("genre"), "abc": abc or None, "source": None,
        "source_duration_s": None,
    }


def parse_remix(inp):
    instrumental = bool(inp.get("instrumental", False))
    genre = inp.get("genre", "base")
    variant = inp.get("variant") or "inst"
    if instrumental:
        if variant not in ("inst", "inst-nar", "nar"):
            raise ValueError("variant must be inst|inst-nar|nar.")
        model_id, spec = resolve_model(variant)
        lyrics = "[instrumental]"
    else:
        lyrics = _text(inp, "lyrics", MAX_LYRICS)
        if not lyrics.strip():
            raise ValueError("lyrics is required when instrumental=false.")
        model_id, spec = resolve_model(genre)
    style = _style(inp)
    source = _source(inp)
    options = {**spec.get("options", {}), **_options(inp)}
    return {
        "task": "remix", "model_id": model_id, "lyrics": lyrics, "style": style,
        "cot": "melody", "seed": _seed(inp), "best_of": _best_of(inp),
        "selector": _selector(inp), "options": options, "output_format": _output_format(inp),
        "genre": genre, "abc": source["abc"], "source": source,
        "source_duration_s": None,
    }


def parse_transcribe(inp):
    source = _source(inp)
    if source["kind"] == "abc":
        raise ValueError("transcribe expects audio_base64 or audio_url, not abc.")
    return {"task": "transcribe", "source": source}


def _download_audio(source, workdir):
    path = os.path.join(workdir, "input.bin")
    if source["kind"] == "audio_base64":
        data = source["audio_base64"]
        if data.startswith("data:"):
            data = data.split(",", 1)[-1]
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception as error:
            raise ValueError(f"invalid audio_base64: {error}")
        if len(raw) > MAX_AUDIO_BYTES:
            raise ValueError("audio too large.")
        open(path, "wb").write(raw)
        return path

    url = source["audio_url"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("audio_url must be http(s).")
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        size = 0
        with open(path, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                size += len(chunk)
                if size > MAX_AUDIO_BYTES:
                    raise ValueError("audio too large.")
                handle.write(chunk)
    return path


def _probe_duration(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise ValueError(f"cannot read audio: {proc.stderr[:300]}")
    try:
        return float(proc.stdout.strip())
    except ValueError as error:
        raise ValueError(f"cannot read duration: {proc.stdout!r}") from error


def prepare_audio(source, workdir):
    raw = _download_audio(source, workdir)
    duration = _probe_duration(raw)
    if duration < MIN_SOURCE_SECONDS:
        raise ValueError(f"source is {duration:.1f}s; minimum is {MIN_SOURCE_SECONDS:.0f}s.")
    if duration > MAX_SOURCE_SECONDS:
        raise ValueError(f"source is {duration:.1f}s; maximum is {MAX_SOURCE_SECONDS:.0f}s.")
    wav = os.path.join(workdir, "source.wav")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", raw, "-vn", "-ac", "2", "-ar", "48000", wav],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg conversion failed: {proc.stderr[:500]}")
    return wav, duration


def transcribe(wav_path):
    started = time.time()
    response = requests.post(
        f"{BASE}/v1/tasks/run",
        json={"model": "sheetsage2", "request": {"audio": wav_path}},
        timeout=900,
    )
    if not response.ok:
        raise RuntimeError(f"SheetSage2 HTTP {response.status_code}: {response.text[:1500]}")
    payload = response.json()
    result = payload.get("result", payload)
    artifacts = result.get("artifacts") or payload.get("artifacts") or []
    abc = None
    for artifact in artifacts:
        if artifact.get("id") == "score" or artifact.get("kind") == "score" \
                or artifact.get("mime") == "text/vnd.abc":
            value = artifact.get("payload") or artifact.get("text")
            if not value:
                continue
            try:
                abc = base64.b64decode(value).decode("utf-8")
            except Exception:
                abc = value if isinstance(value, str) else None
            break
    if not abc:
        raise RuntimeError(f"SheetSage2 returned no ABC. Keys: {list(result)[:20]}")
    return abc, round(time.time() - started, 2), result.get("timing") or {}


def render_one(req, seed):
    options = {"style": req["style"], "cot": req["cot"], **req["options"]}
    if req.get("abc"):
        options["abc"] = req["abc"]
    response = requests.post(
        f"{BASE}/v1/tasks/run",
        json={"model": req["model_id"],
              "request": {"lyrics": req["lyrics"], "seed": seed, "options": options}},
        timeout=3500,
    )
    if not response.ok:
        raise RuntimeError(f"YuE2 HTTP {response.status_code}: {response.text[:1500]}")
    payload = response.json()
    result = payload.get("result", payload)
    audio = result.get("audio")
    if not audio:
        raise RuntimeError(f"YuE2 returned no audio. Keys: {list(result)[:20]}")
    return result


def wav_to_mp3(wav_bytes):
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "in.wav")
        target = os.path.join(directory, "out.mp3")
        open(source, "wb").write(wav_bytes)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", source, "-codec:a", "libmp3lame",
             "-b:a", BITRATE, target],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MP3 encode failed: {proc.stderr[:1000]}")
        return open(target, "rb").read()


def run_best_of(req):
    import scorer
    cand = []
    errors = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(req["best_of"]):
            seed = req["seed"] + index
            try:
                result = render_one(req, seed)
                wav = base64.b64decode(result["audio"])
                path = os.path.join(directory, f"candidate_{index}.wav")
                open(path, "wb").write(wav)
                score = scorer.score(path, target_duration=req.get("source_duration_s"))
                cand.append({"index": index, "seed": seed, "path": path,
                             "result": result, "score": score})
            except Exception as error:
                errors.append({"index": index, "seed": seed, "error": str(error)})
        if not cand:
            raise RuntimeError(f"all candidates failed: {errors[:3]}")
        best = max(cand, key=lambda item: (item["score"], -item["seed"]))
        raw = open(best["path"], "rb").read()
        if req["output_format"] == "wav":
            output = raw
            limit = MAX_WAV_BYTES
        else:
            output = wav_to_mp3(raw)
            limit = MAX_MP3_BYTES
        if len(output) > limit:
            raise RuntimeError(f"output too large ({len(output)} > {limit} bytes).")
        return best, cand, output, errors


def handler(job):
    started = time.time()
    inp = (job or {}).get("input") or {}
    try:
        task = inp.get("task")
        if task is None:
            task = "remix" if (inp.get("audio_base64") or inp.get("audio_url")) else "generate"
        if task == "generate":
            req = parse_generate(inp)
        elif task == "remix":
            req = parse_remix(inp)
        elif task == "transcribe":
            req = parse_transcribe(inp)
        else:
            raise ValueError(f"Unsupported task '{task}'. Use generate|remix|transcribe.")

        with tempfile.TemporaryDirectory() as directory:
            transcription = None
            if task == "transcribe":
                wav, duration = prepare_audio(req["source"], directory)
                abc, seconds, timing = transcribe(wav)
                return {"task": "transcribe", "abc": abc,
                        "source_duration_s": round(duration, 3),
                        "transcription_s": seconds, "audiocpp_timing": timing,
                        "handler_seconds": round(time.time() - started, 2)}
            if task == "remix" and req.get("source") and req["source"]["kind"] != "abc":
                wav, duration = prepare_audio(req["source"], directory)
                req["source_duration_s"] = duration
                abc, seconds, timing = transcribe(wav)
                req["abc"] = abc
                transcription = {"seconds": seconds, "duration_s": round(duration, 3),
                                 "timing": timing}

        best, cand, output, errors = run_best_of(req)
        result = best["result"]
        response = {
            "task": task,
            "format": req["output_format"],
            "audio_base64": base64.b64encode(output).decode(),
            "bytes": len(output),
            "bitrate": BITRATE if req["output_format"] == "mp3" else None,
            "seed": best["seed"],
            "best_seed": best["seed"],
            "best_score": round(best["score"], 6),
            "best_of": req["best_of"],
            "selector": req["selector"],
            "candidates": [{"index": item["index"], "seed": item["seed"],
                            "score": round(item["score"], 6)} for item in cand],
            "candidate_errors": errors,
            "genre": req.get("genre"),
            "model_id": req["model_id"],
            "cot": req["cot"],
            "style": req["style"],
            "abc": req.get("abc") if task == "remix" else None,
            "transcription": transcription,
            "sample_rate": result.get("sample_rate"),
            "channels": result.get("channels"),
            "duration_s": (result.get("timing") or {}).get("audio_duration_ms", 0) / 1000 or None,
            "audiocpp_timing": result.get("timing"),
            "handler_seconds": round(time.time() - started, 2),
            "worker_ready_seconds": round(_ready_at - START_TS, 1) if _ready_at else None,
        }
        return response
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        log(f"[yue2] {type(error).__name__}: {error}")
        return {"error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    start_server()
    import runpod
    runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda n: 1})

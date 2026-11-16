#!/usr/bin/env bash
set -euo pipefail
BASE="https://raw.githubusercontent.com/FastAnt/yue2-remix-worker/main"
mkdir -p /app /models/SheetSage2-GGUF

for f in handler.py scorer.py genres.json server.json; do
  curl -fsSL "$BASE/$f" -o "/app/$f"
done

if [ ! -s /models/SheetSage2-GGUF/sheetsage2-orig.gguf ]; then
  curl -L --fail --retry 5 --retry-delay 5 \
    -o /tmp/sheetsage2-orig.gguf \
    "https://huggingface.co/audio-cpp/SheetSage2-GGUF/resolve/6dd648b0acf1ba9e662d8509a3eabcb201911c2c/sheetsage2-orig.gguf"
  echo "52bb5846c452037d39931aa8050885b6c751b9c7afcc8ef6d6d3067d241731a4  /tmp/sheetsage2-orig.gguf" | sha256sum -c -
  mv /tmp/sheetsage2-orig.gguf /models/SheetSage2-GGUF/sheetsage2-orig.gguf
fi

test -s /models/Yue2-3B-GGUF/yue2-3b-bf16.gguf
test -s /models/Yue2-3B-GGUF/yue2-vae-f32.gguf
test -s /models/Yue2-LoRA/ar_lora_inst_v3abc.safetensors
test -s /models/Yue2-LoRA/nar_lora_joint_v4.safetensors

exec python3 -u /app/handler.py

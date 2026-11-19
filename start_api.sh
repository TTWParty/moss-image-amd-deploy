#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MOSS_MODEL_DIR:-$SCRIPT_DIR/MOSS-Image-SFT}"
PORT="${MOSS_PORT:-8000}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRIPT_DIR/.inductor_cache}"
export MOSS_MODEL_DIR="$MODEL_DIR"

# 1. 自动补齐推理与 API 依赖（不覆盖预装 ROCm PyTorch）
python -c "from transformers import Qwen3_5Model; from diffusers import AutoencoderKLFlux2; import fastapi, uvicorn" 2>/dev/null || \
  pip install -U --upgrade-strategy only-if-needed "diffusers>=0.38" "transformers>=5.3" "accelerate>=1.10" "safetensors>=0.6" "Pillow>=10" fastapi uvicorn

# 2. 若本地无权重则自动通过魔搭下载
if [ ! -f "$MODEL_DIR/model_index.json" ]; then
  echo "未检测到本地模型，开始从魔搭下载 openmoss/MOSS-Image-SFT 到 $MODEL_DIR ..."
  modelscope download --model openmoss/MOSS-Image-SFT --local_dir "$MODEL_DIR"
fi

# 3. 启动 OpenAI 兼容 FastAPI 服务
echo "启动 MOSS-Image API（端口 $PORT，精度=${MOSS_PRECISION:-int8}）..."
uvicorn api_server:app --app-dir "$SCRIPT_DIR" --host 0.0.0.0 --port "$PORT"

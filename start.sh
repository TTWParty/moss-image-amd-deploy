#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MOSS_MODEL_DIR:-$SCRIPT_DIR/MOSS-Image-SFT}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRIPT_DIR/.inductor_cache}"

# 1. 自动补齐依赖（不覆盖预装 ROCm PyTorch）
python -c "from transformers import Qwen3_5Model; from diffusers import AutoencoderKLFlux2" 2>/dev/null || \
  pip install -U --upgrade-strategy only-if-needed "diffusers>=0.38" "transformers>=5.3" "accelerate>=1.10" "safetensors>=0.6" "Pillow>=10"

# 2. 校验 ROCm GPU 可用
python -c "import torch; assert torch.cuda.is_available(), 'GPU 不可用，请确认已保留 ROCm 版 PyTorch'; print('PyTorch:', torch.__version__)"

# 3. 若本地无权重则自动通过魔搭下载 MOSS-Image-SFT
if [ ! -f "$MODEL_DIR/model_index.json" ]; then
  echo "未检测到本地模型，开始从魔搭下载 openmoss/MOSS-Image-SFT 到 $MODEL_DIR ..."
  modelscope download --model openmoss/MOSS-Image-SFT --local_dir "$MODEL_DIR"
fi

# 4. 启动交互式出图（默认 INT8-ConvRot + Spectrum）
MOSS_MODEL_DIR="$MODEL_DIR" python "$SCRIPT_DIR/chat_moss.py" "$@"

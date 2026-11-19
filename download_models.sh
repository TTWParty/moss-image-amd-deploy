#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${MOSS_MODEL_DIR:-}" ]; then
  MODEL_ROOT="$(dirname "$MOSS_MODEL_DIR")"
elif [ -d "/mnt/workspace/moss-image/MOSS-Image-SFT" ]; then
  MODEL_ROOT="/mnt/workspace/moss-image"
else
  MODEL_ROOT="$SCRIPT_DIR"
fi

EXTRA_MODELS_DIR="${MOSS_EXTRA_MODELS_DIR:-$MODEL_ROOT/extra_models}"
LORA_DIR="${MOSS_LORA_DIR:-$MODEL_ROOT/loras}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TARGET="${1:-all}"

mkdir -p "$EXTRA_MODELS_DIR" "$LORA_DIR"

dl_file() {
  local url="$1"
  local dst="$2"
  local label="$3"
  if [ -s "$dst" ]; then
    echo "✓ [已存在，跳过] $label ($dst)"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  local tmp="${dst}.part"
  echo "↓ [下载中] $label"
  echo "  来源: $url"
  curl -fSL -C - --retry 5 --retry-delay 2 --connect-timeout 15 "$url" -o "$tmp"
  mv -f "$tmp" "$dst"
  echo "✓ [完成] $label"
}

download_loras() {
  echo "=== 1/2 下载 4步 / 8步 Turbo LoRA（共 ~3.6 GB，FP32 未合并格式）==="
  dl_file \
    "${HF_ENDPOINT}/akhaliq/MOSS-Image-SFT-4-steps/resolve/main/lora.safetensors" \
    "${LORA_DIR}/sft-4step.safetensors" \
    "SFT 4步 Turbo LoRA (akhaliq, 1.19 GB)"
  dl_file \
    "${HF_ENDPOINT}/lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora.safetensors" \
    "${LORA_DIR}/sft-8step.safetensors" \
    "SFT 8步 DMD LoRA (lodestones, 1.19 GB)"
  dl_file \
    "${HF_ENDPOINT}/lodestones/Kromoss/resolve/main/8-steps-dmd2-lora.safetensors" \
    "${LORA_DIR}/kromoss-8step-dmd2.safetensors" \
    "Kromoss 8步 DMD2-GAN LoRA (lodestones, 1.19 GB)"
}

download_kromoss() {
  echo "=== 2/2 下载 Kromoss 全量重训底模 Transformer（共 ~10.2 GB，复用现有 Qwen3.5-9B + VAE）==="
  local kdir="${EXTRA_MODELS_DIR}/kromoss/transformer"
  for f in \
    "config.json" \
    "diffusion_pytorch_model.safetensors.index.json" \
    "pipeline_moss_image.py" \
    "diffusion_pytorch_model-00001-of-00003.safetensors" \
    "diffusion_pytorch_model-00002-of-00003.safetensors" \
    "diffusion_pytorch_model-00003-of-00003.safetensors"; do
    dl_file \
      "${HF_ENDPOINT}/lodestones/Kromoss/resolve/main/transformer/${f}" \
      "${kdir}/${f}" \
      "Kromoss transformer/${f}"
  done
}

echo "下载目标目录:"
echo "  底模目录 (MOSS_EXTRA_MODELS_DIR): $EXTRA_MODELS_DIR"
echo "  LoRA 目录 (MOSS_LORA_DIR)       : $LORA_DIR"
echo "  镜像源 (HF_ENDPOINT)            : $HF_ENDPOINT"
echo ""

case "$TARGET" in
  all)
    download_loras
    download_kromoss
    ;;
  loras)
    download_loras
    ;;
  kromoss)
    dl_file \
      "${HF_ENDPOINT}/lodestones/Kromoss/resolve/main/8-steps-dmd2-lora.safetensors" \
      "${LORA_DIR}/kromoss-8step-dmd2.safetensors" \
      "Kromoss 8步 DMD2-GAN LoRA (lodestones, 1.19 GB)"
    download_kromoss
    ;;
  *)
    echo "用法: bash download_models.sh [all|loras|kromoss]"
    exit 1
    ;;
esac

echo ""
echo "全部下载完成！"
echo "如果 Web UI 服务已在运行，可直接点页面右上角「扫描新模型」热加载；或重启 start_api.sh 自动加载。"

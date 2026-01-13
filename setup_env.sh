#!/bin/bash

set -e

DEFAULT_WORKSPACE="/workspace"
WORKSPACE_DIR="${1:-$DEFAULT_WORKSPACE}"
echo "Using workspace directory: $WORKSPACE_DIR"
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Error: Directory $WORKSPACE_DIR does not exist."
    echo "Please provide a valid workspace directory as the first argument."
    exit 1
fi

export HF_HOME="$WORKSPACE_DIR/hf_cache"
export HF_HUB_CACHE="$WORKSPACE_DIR/hf_cache/hub"
export TRANSFORMERS_CACHE="$WORKSPACE_DIR/hf_cache/transformers"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Optional: login to Hugging Face for gated models (non-interactive).
# Usage:
#   export HF_TOKEN=...
#   ./setup_env.sh /workspace
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is set; attempting non-interactive Hugging Face login..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli login --token "$HF_TOKEN" || true
    elif command -v hf >/dev/null 2>&1; then
        hf auth login --token "$HF_TOKEN" || true
    else
        echo "Warning: neither 'huggingface-cli' nor 'hf' found; skipping HF login."
    fi
fi

export PIP_BREAK_SYSTEM_PACKAGES=1
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory."
fi

if [ -d "$WORKSPACE_DIR/sglang" ]; then
    cd "$WORKSPACE_DIR/sglang"
    pip install -e "python[diffusion]"
else
    echo "Error: $WORKSPACE_DIR/sglang not found."
    exit 1
fi

pip install -v git+https://github.com/huggingface/diffusers.git
pip install -v git+https://github.com/vipshop/cache-dit.git
pip install -v git+https://github.com/ModelTC/LightX2V.git --no-deps
echo "Installing vllm-omni..."
pip install vllm==0.12.0 --no-deps
pip install git+https://github.com/vllm-project/vllm-omni.git@ef01223c42be10ee260b9f6e5ec31894cd09d86e --no-deps
pip install cbor2 blake3 cachetools py-cpuinfo lm-format-enforcer omegaconf

echo "Building SGLang AOT kernels..."
export CCACHE_DIR="$WORKSPACE_DIR/sglang/.ccache"
export TORCH_CUDA_ARCH_LIST="9.0"
cd "$WORKSPACE_DIR/sglang/sgl-kernel"
find python/sgl_kernel -maxdepth 2 -name "common_ops*.so" -delete
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
SGLANG_BUILD_DIR="$WORKSPACE_DIR/sglang_build"
cmake -S "$WORKSPACE_DIR/sglang/sgl-kernel" -B "$SGLANG_BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE"
cmake --build "$SGLANG_BUILD_DIR" --target common_ops_sm90_build -j$(nproc)
if compgen -G "$SGLANG_BUILD_DIR/flash_ops.*.so" > /dev/null; then
    cp "$SGLANG_BUILD_DIR"/flash_ops.*.so python/sgl_kernel/
fi
if compgen -G "$SGLANG_BUILD_DIR/flashmla_ops.*.so" > /dev/null; then
    cp "$SGLANG_BUILD_DIR"/flashmla_ops.*.so python/sgl_kernel/
fi
if compgen -G "$SGLANG_BUILD_DIR/spatial_ops.*.so" > /dev/null; then
    cp "$SGLANG_BUILD_DIR"/spatial_ops.*.so python/sgl_kernel/
fi

mkdir -p python/sgl_kernel/sm90
if compgen -G "$SGLANG_BUILD_DIR/sm90/*" > /dev/null; then
    cp "$SGLANG_BUILD_DIR"/sm90/* python/sgl_kernel/sm90/
fi

export PYTHONPATH="$WORKSPACE_DIR/sglang/sgl-kernel/python:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "Installing FlashAttention (LightX2V dependency)..."
if [ -d "$WORKSPACE_DIR/flash-attention/hopper" ]; then
    cd "$WORKSPACE_DIR/flash-attention/hopper"
    python setup.py install
    cd "$WORKSPACE_DIR/flash-attention"
    python setup.py install
    export PYTHONPATH="$WORKSPACE_DIR/flash-attention/hopper:$PYTHONPATH"
else
    echo "Warning: $WORKSPACE_DIR/flash-attention not found. Skipping FlashAttention installation."
fi
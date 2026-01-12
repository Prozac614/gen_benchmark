# Generation Benchmark

A minimal framework for benchmarking image and video generation performance across different inference engines (Diffusers, SGLang, LightX2V, Fal.ai).

## Installation

```bash
pip install -r requirements.txt
```

If you run SGLang/LightX2V, start from the docker image: `lmsysorg/sglang:dev`

```bash
export PIP_BREAK_SYSTEM_PACKAGES=1 # solve externally-managed-environment error
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers

# Install SGLang Diffusion and LightX2v
cd /workspace/sglang
pip install -e "python[diffusion]"
pip install -v git+https://github.com/huggingface/diffusers.git 
pip install -v git+https://github.com/vipshop/cache-dit.git
pip install -v git+https://github.com/ModelTC/LightX2V.git --no-deps

# Build SGLang AOT kernels faster with ccache
export CCACHE_DIR=/workspace/sglang/.ccache
export TORCH_CUDA_ARCH_LIST="9.0"
cd /workspace/sglang/sgl-kernel
find python/sgl_kernel -maxdepth 2 -name "common_ops*.so" -delete
TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -S /workspace/sglang/sgl-kernel -B /workspace/sglang_build \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE"
cmake --build /workspace/sglang_build --target common_ops_sm90_build -j$(nproc)
cp /workspace/sglang_build/flash_ops.*.so python/sgl_kernel/
cp /workspace/sglang_build/flashmla_ops.*.so python/sgl_kernel/
cp /workspace/sglang_build/spatial_ops.*.so python/sgl_kernel/
cp /workspace/sglang_build/sm90/* python/sgl_kernel/sm90/
export PYTHONPATH=/workspace/sglang/sgl-kernel/python:$PYTHONPATH

# FlashAttention (LightX2V dependency)
cd /workspace/flash-attention/hopper
python setup.py install
```

## Quick Start

Run all targets defined in a config file:

```bash
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/benchmark_all_local.yaml
```

Run specific targets:

```bash
python main.py --config model_configs/benchmark_all_local.yaml --targets diffusers_qwen_2512
```

```bash
python main.py --config model_configs/benchmark_sglang_all.yaml
```

## Configuration

Configs are in `model_configs/`. A typical config looks like:

```yaml
includes:
  - _fragments/cases_common.yaml # Shared test cases

targets:
  my_target_name:
    engine_class: "DiffusersEngine" # or SGLangEngine, LightX2VEngine, FalEngine
    type: "local"
    task: "text-to-image"
    case_ids: ["case_id_1"]
    params:
      model: "path/to/model"
```
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

#  vllm-omni
pip install vllm==0.12.0 --no-deps
pip install git+https://github.com/vllm-project/vllm-omni.git@ef01223c42be10ee260b9f6e5ec31894cd09d86e  --no-deps
Successfully installed aiofiles-24.1.0 antlr4-python3-runtime-4.9.3 audioread-3.1.0 brotli-1.2.0 cache-dit-1.1.8 ffmpy-1.0.0 gradio-5.50.0 gradio-client-1.14.0 groovy-0.1.2 joblib-1.5.3 lazy_loader-0.4 librosa-0.11.0 llvmlite-0.46.0 numba-0.63.1 numpy-2.3.5 omegaconf-2.3.0 pooch-1.8.2 pydantic-2.12.3 pydantic-core-2.41.4 pydub-0.25.1 resampy-0.4.3 ruff-0.14.11 safehttpx-0.1.7 scikit-learn-1.8.0 semantic-version-2.10.0 shellingham-1.5.4 soxr-1.0.0 threadpoolctl-3.6.0 tomlkit-0.13.3 typer-0.21.1 vllm-omni-0.12.0rc1 websockets-15.0.1
pip install cbor2
pip install blake3
pip install cachetools
pip install py-cpuinfo
pip install lm-format-enforcer
```

## Quick Start

Run all targets defined in a config file:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/benchmark_all.yaml
```

Run specific targets:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/benchmark_all.yaml --targets sglang_qwen_2512
python main.py --config model_configs/benchmark_all.yaml --targets lightx2v_qwen_2512
python main.py --config model_configs/benchmark_all.yaml --targets vllm_omni_qwen_2512
python main.py --config model_configs/benchmark_all.yaml --targets diffusers_qwen_2512
```

```bash
python main.py --config model_configs/benchmark_sglang.yaml
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
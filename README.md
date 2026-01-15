# Generation Benchmark

A minimal framework for benchmarking image and video generation performance across different inference engines (Diffusers, SGLang, LightX2V, Fal.ai).

## Installation

```bash
pip install -r requirements.txt
```

If you run SGLang/LightX2V, start from the docker image: `lmsysorg/sglang:dev`

```bash
# Setup dependencies inside the container.
# Pass your workspace root (where repos like sglang/LightX2V live). Default is /workspace.
bash setup_env.sh /workspace
```

## Quick Start

Run all targets defined in a config file:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/all.yaml
```

Run specific targets:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/all.yaml --targets sglang_qwen_2512
python main.py --config model_configs/all.yaml --targets lightx2v_qwen_2512
python main.py --config model_configs/all.yaml --targets vllm_omni_qwen_2512
python main.py --config model_configs/all.yaml --targets diffusers_qwen_2512
```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python main.py --config model_configs/sglang.yaml
python main.py --config model_configs/lightx2v.yaml
python main.py --config model_configs/vllm.yaml
python main.py --config model_configs/diffusers.yaml
```

## Configuration

Configs are in `model_configs/`.

We split shared test cases and per-engine targets:

- `model_configs/cases.yaml`: shared `defaults` + all `cases`
- `model_configs/targets_*.yaml`: engine-specific `targets` (e.g. `targets_sglang.yaml`, `targets_vllm.yaml`)
- `model_configs/{all,sglang,lightx2v,vllm,diffusers}.yaml`: entrypoints that include `cases.yaml` + one or more `targets_*.yaml`

A typical entrypoint config looks like:

```yaml
includes:
  - cases.yaml
  - targets_sglang.yaml

targets: {}
```
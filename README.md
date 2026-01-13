# Generation Benchmark

A minimal framework for benchmarking image and video generation performance across different inference engines (Diffusers, SGLang, LightX2V, Fal.ai).

## Installation

```bash
pip install -r requirements.txt
```

If you run SGLang/LightX2V, start from the docker image: `lmsysorg/sglang:dev`

``` bash
bash setup.sh /workpace
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
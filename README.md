Here is the English translation of the README. Per your request, emojis have been removed, and the status sections use standard Markdown checkboxes.

---

# Generation Benchmark Framework

Automated Benchmarking Framework for Inference Acceleration

## Project Intent

This project aims to build an automated benchmarking framework for horizontal comparison, designed to evaluate the performance of different inference acceleration frameworks on image and video generation tasks. By providing a unified interface and test cases, it allows for objective comparisons of key metrics such as latency and VRAM usage across various frameworks.

### Core Features

* **Multi-Framework Comparison**: Supports comparison across multiple mainstream inference acceleration frameworks.
* **Multi-Task Support**: Supports Image Generation (Text-to-Image, Image-to-Image) and Video Generation (Text-to-Video, Image-to-Video).
* **Performance Monitoring**: Automatically monitors generation latency and GPU VRAM usage.
* **Unified Interface**: Simplifies the integration and testing of different frameworks through a unified engine interface.

### Comparison Dimensions

* **Latency**: Total time consumed for the generation task.
* **VRAM Usage**: Peak GPU memory usage (Local engines only).
* **Output Quality**: The generated image/video files.

## Supported Frameworks

This project plans to support the following inference acceleration frameworks:

| Framework | Status | Description |
| --- | --- | --- |
| **Diffusers** | Implemented | Hugging Face Diffusers library, supports local inference. |
| **Fal.ai** | Implemented | Fal.ai cloud API service. |
| **SGLang Diffusion** | Planned | SGLang optimized diffusion model inference engine. |
| **LightX2D** | Planned | LightX2D video generation acceleration framework. |
| **vLLM Omni** | Planned | vLLM Omni multimodal inference engine. |
| **Replicate** | Planned | Replicate cloud API service. |

## Installation

### Basic Dependencies

```bash
pip install -r requirements.txt

```

## Usage

### Basic Usage

Run the benchmark for a specific model configuration:

```bash
python main.py --config model_configs/black-forest-labs_FLUX.1-dev.yaml

```

### Running Specific Targets

If multiple targets are defined in the configuration file, you can run only specific targets:

```bash
python main.py --config model_configs/black-forest-labs_FLUX.1-dev.yaml --targets diffusers_flux1_dev fal_flux1_dev

```

### Command Line Arguments

* `--config` (Required): Path to the configuration file. Defaults to `config.yaml`.
* `--targets` (Optional): Filter for specific targets to run. Accepts multiple names. If not provided, all targets in the configuration file will be executed.

### Examples

**Example 1: Run all targets for FLUX.1-dev**

```bash
python main.py --config model_configs/black-forest-labs_FLUX.1-dev.yaml

```

**Example 2: Run only the Diffusers engine**

```bash
python main.py --config model_configs/black-forest-labs_FLUX.1-dev.yaml --targets diffusers_flux1_dev

```

**Example 3: Run a video generation model**

```bash
python main.py --config model_configs/Wan-AI_Wan2.1-T2V-1.3B-Diffusers.yaml

```

## Configuration

Configuration files use the YAML format and contain two main sections: `cases` (test cases) and `targets` (evaluation targets).

### Configuration Structure

```yaml
cases:
  - id: "test_case_1"
    type: "image"  # or "video"
    prompt: "Your prompt here"
    # ... other parameters

targets:
  - name: "engine_name"
    engine_class: "EngineClassName"
    type: "local"  # or "remote"
    task: "image"  # or "video"
    # ... engine specific configuration

```

### Cases Configuration

Each test case includes the following fields:

* `id` (Required): Unique identifier for the test case.
* `type` (Required): Task type, `"image"` or `"video"`.
* `prompt` (Required): Generation prompt.
* `num_inference_steps` (Optional): Number of inference steps, default is 30.
* `height` (Optional): Height of the generated output, default is 512.
* `width` (Optional): Width of the generated output, default is 512.
* `num_frames` (Optional, Video only): Number of video frames, default is 25.
* `image_path` (Optional): Input image path (for Image-to-Image or Image-to-Video tasks). Supports local paths or URLs.

### Targets Configuration

Each evaluation target includes the following fields:

* `name` (Required): Unique name for the target.
* `engine_class` (Required): Engine class name, e.g., `"DiffusersEngine"`, `"FalEngine"`.
* `type` (Required): Engine type, `"local"` (local inference) or `"remote"` (remote API).
* `task` (Required): Task type, `"image"` or `"video"`.
* `output_dir` (Optional): Output directory, defaults to `"outputs"`.

## Project Structure

```text
gen_benchmark/
├── main.py                 # Main entry point
├── monitor.py              # GPU monitoring module
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── engines/                # Engine implementation directory
│   ├── __init__.py
│   ├── base.py             # Base engine abstract class
│   ├── diffusers_engine.py # Diffusers engine implementation
│   └── fal_engine.py       # Fal.ai engine implementation
└── model_configs/          # Model configuration directory
    ├── black-forest-labs_FLUX.1-dev.yaml
    ├── black-forest-labs_FLUX.1-schnell.yaml
    ├── black-forest-labs_FLUX.2-dev.yaml
    ├── Qwen_Qwen-Image.yaml
    ├── Qwen_Qwen-Image-Edit.yaml
    ├── Tongyi-MAI_Z-Image-Turbo.yaml
    ├── Wan-AI_Wan2.1-T2V-1.3B-Diffusers.yaml
    ├── Wan-AI_Wan2.1-T2V-14B.yaml
    ├── Wan-AI_Wan2.2-I2V-A14B-Diffusers.yaml
    └── Wan-AI_Wan2.2-TI2V-5B-Diffusers.yaml

```

## Development Status

### Implemented Features

* [x] DiffusersEngine - Support for image and video generation
* [x] FalEngine - Remote API support for image and video generation
* [x] GPU Monitoring - Automatic VRAM monitoring for local engines
* [x] Unified Engine Interface - Facilitates extension of new engines
* [x] Multi-Model Config Support - Independent configuration files for each model
* [x] Performance Metrics Collection - Latency and VRAM usage statistics

### Planned Features

* [ ] SGLangEngine - SGLang Diffusion engine integration
* [ ] LightX2VEngine - LightX2D video generation engine integration
* [ ] VLLMOmniEngine - vLLM Omni multimodal engine integration
* [ ] ReplicateEngine - Replicate API engine integration
* [ ] Result Visualization - Performance comparison charts and reports
* [ ] Batch Testing - Support for batch running multiple model configurations

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
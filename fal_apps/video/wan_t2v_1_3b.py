"""Wan2.1-T2V-1.3B fal serverless app."""

import sys
sys.path.insert(0, "/root/gen_benchmark")

from fal_apps.base import BaseVideoApp


class WanT2V1_3BApp(BaseVideoApp):
    """Wan-AI/Wan2.1-T2V-1.3B-Diffusers video generation app."""

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    app_name = "wan-t2v-1-3b"
    machine_type = "GPU-A100"

    requirements = [
        "torch==2.6.0",
        "accelerate==1.6.0",
        "transformers==4.51.3",
        "diffusers>=0.30.0",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
        "imageio>=2.31.0",
        "imageio-ffmpeg>=0.4.9",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

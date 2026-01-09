"""Z-Image-Turbo fal serverless app."""

import sys
sys.path.insert(0, "/root/gen_benchmark")

from fal_apps.base import BaseImageApp


class ZImageTurboApp(BaseImageApp):
    """Tongyi-MAI/Z-Image-Turbo image generation app."""

    model_id = "Tongyi-MAI/Z-Image-Turbo"
    app_name = "z-image-turbo"
    machine_type = "GPU-A100"

    requirements = [
        "torch==2.6.0",
        "accelerate==1.6.0",
        "transformers==4.51.3",
        "diffusers>=0.30.0",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu124",
    ]

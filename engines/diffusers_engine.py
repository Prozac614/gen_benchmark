import os
import time
from datetime import datetime
from typing import Dict, Any

import inspect
import torch
from diffusers.utils import load_image

from .base import BaseEngine


class DiffusersEngine(BaseEngine):
    """Engine adapter for Hugging Face Diffusers library."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.task = config.get("task", "image")
        self.params = config.get("params") or {}
        
        if self.task in ["image", "text-to-image", "image-to-image"]:
            self.task_type = "image"
        elif self.task in ["video", "text-to-video", "image-to-video"]:
            self.task_type = "video"
        else:
            print(f"Warning: Unknown task '{self.task}', defaulting to 'image' logic.")
            self.task_type = "image"

    def load(self) -> None:
        model_path = self.params["model"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        cpu_offload = bool(self.params.get("cpu_offload", False))
        
        if self.task_type == "image":
            from diffusers import DiffusionPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image
            try:
                if self.task == "image-to-image":
                    self.pipe = AutoPipelineForImage2Image.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                    )
                elif self.task == "text-to-image":
                    self.pipe = AutoPipelineForText2Image.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                    )
                else:
                    self.pipe = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                    )
            except Exception as e:
                print(f"  (AutoPipeline load failed: {e}, falling back to DiffusionPipeline)")
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                )
            
        elif self.task_type == "video":
            try:
                if self.task == "image-to-video":
                    try:
                        from diffusers import WanImageToVideoPipeline
                        self.pipe = WanImageToVideoPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                    except ImportError:
                        from diffusers import WanPipeline
                        self.pipe = WanPipeline.from_pretrained(
                            model_path,
                            torch_dtype=dtype,
                        )
                else:
                    from diffusers import WanPipeline
                    self.pipe = WanPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                    )
            except ImportError:
                from diffusers import DiffusionPipeline
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                )

        if cpu_offload:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
                raise RuntimeError(
                    f"{type(self.pipe).__name__} does not support enable_model_cpu_offload(), "
                    "but target.params.cpu_offload=true was requested."
                )
        else:
            try:
                self.pipe.to("cuda")
            except torch.OutOfMemoryError as e:
                raise torch.OutOfMemoryError(
                    f"CUDA OOM while moving {type(self.pipe).__name__} to GPU. "
                    "For apples-to-apples benchmarking, prefer running this target alone "
                    "(e.g. `--targets diffusers_flux2_dev`) to maximize free VRAM. "
                    "If you must run it, set target.params.cpu_offload=true (slower / not comparable)."
                ) from e

    def _adapt_pipe_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipe is None:
            return kwargs

        try:
            sig = inspect.signature(self.pipe.__call__)
        except Exception:
            return kwargs

        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs

        accepted = set(params.keys())
        out = dict(kwargs)

        if "image" in out and "image" not in accepted:
            img = out.pop("image")
            for cand in (
                "init_image",
                "input_image",
                "conditioning_image",
                "first_frame_image",
                "first_frame",
            ):
                if cand in accepted:
                    out[cand] = img
                    break
            else:
                raise TypeError(
                    f"{type(self.pipe).__name__} does not accept an image input argument. "
                    "This looks like an image-to-video case, but the loaded pipeline does not support it "
                    "(or uses an unexpected argument name)."
                )

        # Drop unsupported kwargs
        return {k: v for k, v in out.items() if k in accepted}

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image or video based on the test case."""
        prompt = case["prompt"]
        case_params = case.get("params") or {}
        num_inference_steps = case_params.get("num_inference_steps", 30)
        height = case_params.get("height", 512)
        width = case_params.get("width", 512)
        seed = case_params.get("seed", 42)
        
        image_path = case_params.get("image_path")
        image = None
        if image_path:
            image = load_image(image_path)
        
        base_output_dir = self.config.get("output_dir", "outputs")
        output_dir = os.path.join(base_output_dir, self.name, case["id"])
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        pipe_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "generator": generator,
        }

        # Common knobs (pipelines vary; we adapt/filter below)
        if case_params.get("guidance_scale") is not None:
            pipe_kwargs["guidance_scale"] = float(case_params["guidance_scale"])
            # Some pipelines (e.g. Qwen) use true_cfg_scale.
            pipe_kwargs["true_cfg_scale"] = float(case_params["guidance_scale"])
        if case_params.get("negative_prompt") is not None:
            pipe_kwargs["negative_prompt"] = case_params.get("negative_prompt")
        
        if self.task_type == "video":
            pipe_kwargs["num_frames"] = case_params.get("num_frames", 25)
            
        if image is not None:
            pipe_kwargs["image"] = image
            
        start_time = time.perf_counter()
        output = self.pipe(**self._adapt_pipe_kwargs(pipe_kwargs))
        end_time = time.perf_counter()
        latency = end_time - start_time
        
        if self.task_type == "image":
            output_filename = f"output_{timestamp}.png"
            output_path = os.path.join(output_dir, output_filename)
            output.images[0].save(output_path)
        elif self.task_type == "video":
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            import imageio
            import numpy as np
            frames = output.frames[0]
            if frames.dtype == np.float32 or frames.dtype == np.float16 or frames.dtype == float:
                frames = (frames * 255).astype(np.uint8)
            imageio.mimwrite(output_path, frames, fps=8, quality=8)
        
        return {
            "e2e_latency": latency,
            "steady_state_seconds": latency,
            "output_path": output_path,
        }

    def unload(self) -> None:
        """Unload the pipeline and free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

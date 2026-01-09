import os
import time
from datetime import datetime
from typing import Dict, Any

import torch
from diffusers.utils import load_image

from .base import BaseEngine


class DiffusersEngine(BaseEngine):
    """Engine adapter for Hugging Face Diffusers library."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.task = config.get("task", "image")

    def load(self) -> None:
        """Load the appropriate diffusers pipeline based on task type."""
        model_path = self.config["model"]
        
        if self.task == "image":
            # Try to load the appropriate pipeline based on model
            from diffusers import DiffusionPipeline
            
            # Use AutoPipeline or DiffusionPipeline for flexibility
            try:
                from diffusers import AutoPipelineForText2Image
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    model_path,
                    dtype=torch.float16,
                )
            except Exception:
                # Fallback to generic DiffusionPipeline
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    dtype=torch.float16,
                )
            self.pipe.to("cuda")
            
        elif self.task == "video":
            # Load Wan pipeline for video generation
            # Note: The actual pipeline class may vary based on diffusers version
            try:
                from diffusers import WanPipeline
                self.pipe = WanPipeline.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                )
            except ImportError:
                # Fallback: try CogVideoX or other video pipelines
                from diffusers import DiffusionPipeline
                self.pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16,
                )
            self.pipe.to("cuda")
        
        # Enable memory optimizations if needed
        if hasattr(self.pipe, "enable_model_cpu_offload"):
            # Uncomment if running into OOM issues
            # self.pipe.enable_model_cpu_offload()
            pass

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image or video based on the test case."""
        prompt = case["prompt"]
        num_inference_steps = case.get("num_inference_steps", 30)
        height = case.get("height", 512)
        width = case.get("width", 512)
        
        # Load image if image_path is provided (for image editing tasks)
        image_path = case.get("image_path")
        image = None
        if image_path:
            image = load_image(image_path)
        
        # Create output directory
        base_output_dir = self.config.get("output_dir", "outputs")
        output_dir = os.path.join(base_output_dir, self.name, case["id"])
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        start_time = time.time()
        
        if self.task == "image":
            # Image generation
            pipe_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
            }
            if image is not None:
                pipe_kwargs["image"] = image
            
            output = self.pipe(**pipe_kwargs)
            
            # Save image
            output_filename = f"output_{timestamp}.png"
            output_path = os.path.join(output_dir, output_filename)
            output.images[0].save(output_path)
            
        elif self.task == "video":
            # Video generation
            num_frames = case.get("num_frames", 25)
            
            pipe_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "num_frames": num_frames,
            }
            if image is not None:
                pipe_kwargs["image"] = image
            
            output = self.pipe(**pipe_kwargs)
            
            # Save video
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Export frames to video
            import imageio
            frames = output.frames[0]  # Get first batch
            imageio.mimwrite(output_path, frames, fps=8, quality=8)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return {
            "latency": latency,
            "output_path": output_path,
        }

    def unload(self) -> None:
        """Unload the pipeline and free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

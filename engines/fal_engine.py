import os
import time
import logging
import requests
from datetime import datetime
from typing import Dict, Any

import fal_client

from .base import BaseEngine


class FalEngine(BaseEngine):
    """Engine adapter for fal.ai API endpoints."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = config.get("task", "image")
        self.fal_model = config["fal_model"]
        self.fal_model_i2v = config.get("fal_model_i2v")

    def load(self) -> None:
        """No operation for API-based engine."""
        pass

    # Optional parameters supported for video generation APIs
    VIDEO_OPTIONAL_PARAMS = [
        "num_inference_steps",
        "negative_prompt",
        "num_frames",
        "frames_per_second",
        "resolution",
        "aspect_ratio",
        "enable_safety_checker",
        "enable_output_safety_checker",
        "enable_prompt_expansion",
        "acceleration",
        "guidance_scale",
        "guidance_scale_2",
        "shift",
        "interpolator_model",
        "num_interpolated_frames",
        "adjust_fps_for_interpolation",
        "video_quality",
        "video_write_mode",
    ]

    def _get_endpoint(self, case: Dict[str, Any]) -> str:
        """Determine the appropriate fal.ai endpoint based on case."""
        if self.task == "video" and case.get("image_url") and self.fal_model_i2v:
            return self.fal_model_i2v
        return self.fal_model

    def _build_arguments(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build arguments dict for fal.ai API call."""
        prompt = case["prompt"]
        
        # Initialize arguments with prompt
        arguments = {"prompt": prompt}
        
        # Handle image size: use resolution if provided, otherwise use width/height
        if case.get("resolution"):
            # Use resolution and aspect_ratio for video APIs (e.g., fal-ai/wan)
            pass  # resolution will be added via VIDEO_OPTIONAL_PARAMS loop
        else:
            # Use traditional width/height for image APIs
            height = case.get("height", 512)
            width = case.get("width", 512)
            arguments["image_size"] = {"width": width, "height": height}
        
        # Add seed if provided
        if case.get("seed") is not None:
            arguments["seed"] = case["seed"]
        
        # Add image_url if image_url exists
        if case.get("image_url"):
            arguments["image_url"] = case["image_url"]
        
        # Add negative_prompt with default empty string
        arguments["negative_prompt"] = case.get("negative_prompt", "")
        
        # Add all optional video parameters if they exist in case (exclude negative_prompt as it's handled separately with default value)
        for param in self.VIDEO_OPTIONAL_PARAMS:
            if param != "negative_prompt" and param in case and case[param] is not None:
                arguments[param] = case[param]
        
        return arguments

    def _download_file(self, url: str, output_path: str) -> None:
        """Download file from URL to local path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image or video using fal.ai API."""
        # Determine endpoint
        endpoint = self._get_endpoint(case)
        
        # Build arguments
        arguments = self._build_arguments(case)
        
        # Create output directory
        base_output_dir = self.config.get("output_dir", "outputs")
        output_dir = os.path.join(base_output_dir, self.name, case["id"])
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Call fal.ai API
        inference_time = None
        def on_queue_update(status):
            nonlocal inference_time
            if hasattr(status, 'metrics') and status.metrics:
                inference_time = status.metrics.get("inference_time")

        start_time = time.time()
        result = fal_client.subscribe(
            endpoint,
            arguments=arguments,
            on_queue_update=on_queue_update
        )
        end_time = time.time()
        
        # Calculate end-to-end latency
        e2e_latency = end_time - start_time
        
        
        # Extract URL and download
        if self.task == "image":
            result_url = result["images"][0]["url"]
            output_filename = f"output_{timestamp}.png"
        else:  # video
            result_url = result["video"]["url"]
            output_filename = f"output_{timestamp}.mp4"
        
        output_path = os.path.join(output_dir, output_filename)
        self._download_file(result_url, output_path)
        
        return {
            "e2e_latency": e2e_latency,
            "inference_time": inference_time if inference_time is not None else "N/A",
            "output_path": output_path,
        }

    def unload(self) -> None:
        """No operation for API-based engine."""
        pass

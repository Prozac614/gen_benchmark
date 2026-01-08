import os
import time
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

    def _get_endpoint(self, case: Dict[str, Any]) -> str:
        """Determine the appropriate fal.ai endpoint based on case."""
        if self.task == "video" and case.get("image_path") and self.fal_model_i2v:
            return self.fal_model_i2v
        return self.fal_model

    def _build_arguments(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build arguments dict for fal.ai API call."""
        prompt = case["prompt"]
        height = case.get("height", 512)
        width = case.get("width", 512)
        
        arguments = {
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
        }
        
        # Add image_url if image_path exists
        if case.get("image_path"):
            arguments["image_url"] = case["image_path"]
        
        # Add num_inference_steps if exists
        if case.get("num_inference_steps"):
            arguments["num_inference_steps"] = case["num_inference_steps"]
        
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
        start_time = time.time()
        handler = fal_client.submit(endpoint, arguments=arguments)
        result = handler.get()
        end_time = time.time()
        
        latency = end_time - start_time
        
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
            "latency": latency,
            "output_path": output_path,
        }

    def unload(self) -> None:
        """No operation for API-based engine."""
        pass

"""Engine for calling self-deployed fal serverless endpoints."""

import os
import time
import requests
from datetime import datetime
from typing import Dict, Any

import fal_client

from .base import BaseEngine


class FalServerlessEngine(BaseEngine):
    """Engine adapter for self-deployed fal.ai serverless endpoints."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = config.get("task", "image")
        self.fal_app_id = config["fal_app_id"]
        self.fal_endpoint = config.get("fal_endpoint", "/")

    def load(self) -> None:
        """No operation for API-based engine."""
        pass

    def _build_arguments(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build arguments dict for fal.ai API call."""
        prompt = case["prompt"]

        if self.task == "image":
            arguments = {
                "prompt": prompt,
                "image_size": {
                    "width": case.get("width", 1024),
                    "height": case.get("height", 1024),
                },
                "num_inference_steps": case.get("num_inference_steps", 30),
                "enable_safety_checker": case.get("enable_safety_checker", True),
            }
            if case.get("image_path"):
                arguments["image_url"] = case["image_path"]
        else:  # video
            arguments = {
                "prompt": prompt,
                "height": case.get("height", 480),
                "width": case.get("width", 832),
                "num_frames": case.get("num_frames", 81),
                "num_inference_steps": case.get("num_inference_steps", 30),
                "enable_safety_checker": case.get("enable_safety_checker", True),
            }
            if case.get("image_path"):
                arguments["image_url"] = case["image_path"]

        return arguments

    def _download_file(self, url: str, output_path: str) -> None:
        """Download file from URL to local path."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image or video using self-deployed fal serverless endpoint."""
        # Build full endpoint
        endpoint = f"{self.fal_app_id}{self.fal_endpoint}"

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

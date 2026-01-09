import os
import time
from datetime import datetime
from typing import Dict, Any

import replicate

from .base import BaseEngine


class ReplicateEngine(BaseEngine):
    """Engine adapter for Replicate API endpoints."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task = config.get("task", "image")
        self.replicate_model = config["replicate_model"]

    def load(self) -> None:
        """No operation for API-based engine."""
        pass

    def _build_input(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Build input dict for Replicate API call."""
        prompt = case["prompt"]
        height = case.get("height", 512)
        width = case.get("width", 512)
        seed = case.get("seed", 42)
        
        input_dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "seed": seed,
        }
        
        # Add num_inference_steps if exists
        if case.get("num_inference_steps"):
            input_dict["num_inference_steps"] = case["num_inference_steps"]
        
        # Add num_frames for video tasks if exists
        if self.task == "video" and case.get("num_frames"):
            input_dict["num_frames"] = case["num_frames"]
        
        # Add image if image_path exists
        if case.get("image_path"):
            image_path = case["image_path"]
            # Check if it's a URL or local file
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # URL: pass as string
                input_dict["image"] = image_path
            else:
                # Local file: open file object
                input_dict["image"] = open(image_path, "rb")
        
        return input_dict

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image or video using Replicate API."""
        # Build input dict
        input_dict = self._build_input(case)
        
        # Create output directory
        base_output_dir = self.config.get("output_dir", "outputs")
        output_dir = os.path.join(base_output_dir, self.name, case["id"])
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Call Replicate API
        start_time = time.time()
        output = replicate.run(self.replicate_model, input=input_dict)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Handle output based on task
        if self.task == "image":
            # Get first file output if it's a list
            if isinstance(output, list):
                file_output = output[0]
            else:
                file_output = output
            
            # Read bytes and save
            image_bytes = file_output.read()
            output_filename = f"output_{timestamp}.png"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
        else:  # video
            # Get first file output if it's a list
            if isinstance(output, list):
                file_output = output[0]
            else:
                file_output = output
            
            # Read bytes and save
            video_bytes = file_output.read()
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "wb") as f:
                f.write(video_bytes)
        
        return {
            "latency": latency,
            "output_path": output_path,
        }

    def unload(self) -> None:
        """No operation for API-based engine."""
        pass

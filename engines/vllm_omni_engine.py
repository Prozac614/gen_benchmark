import os
import time
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image, export_to_video

from .base import BaseEngine

class VllmOmniEngine(BaseEngine):
    @staticmethod
    def _normalize_payload_to_frames(payload: Any) -> list[np.ndarray]:
        """
        Normalize vLLM-Omni diffusion payloads into a list of uint8 HWC RGB frames.
        vLLM-Omni can return:
        - PIL.Image.Image
        - list[PIL.Image.Image]
        - np.ndarray (HWC/CHW) or list[np.ndarray]
        - torch.Tensor (CHW, HWC, TCHW, THWC)
        - dict-like with keys such as frames/images/video
        """

        def _maybe_unwrap_dict(x: Any) -> Any:
            if isinstance(x, dict):
                for k in ("frames", "images", "video", "videos", "output", "result"):
                    if k in x and x[k] is not None:
                        return x[k]
            return x

        def _flatten_one_level(seq: list[Any]) -> list[Any]:
            out: list[Any] = []
            for item in seq:
                if isinstance(item, (list, tuple)):
                    out.extend(list(item))
                else:
                    out.append(item)
            return out

        def _to_uint8_hwc_rgb(arr: np.ndarray) -> np.ndarray:
            a = arr
            if not isinstance(a, np.ndarray):
                a = np.asarray(a)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            elif a.ndim == 3:
                # CHW -> HWC
                if a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
                    a = np.transpose(a, (1, 2, 0))
                # drop alpha
                if a.shape[-1] == 4:
                    a = a[..., :3]
                if a.shape[-1] == 1:
                    a = np.repeat(a, 3, axis=-1)
                if a.shape[-1] != 3:
                    raise ValueError(f"Frame has unexpected channel count: shape={a.shape}")
            else:
                raise ValueError(f"Frame must be 2D/3D array. Got shape={a.shape}")

            if a.dtype == np.uint8:
                return a
            a = a.astype(np.float32, copy=False)
            # Heuristic: float in [0,1] or [0,255]
            if np.nanmax(a) <= 1.5:
                a = a * 255.0
            a = np.clip(a, 0.0, 255.0).astype(np.uint8)
            return a

        def _from_torch_tensor(t: torch.Tensor) -> list[np.ndarray]:
            tt = t.detach()
            if tt.is_cuda:
                tt = tt.cpu()
            if tt.ndim == 4:
                # Assume time/batch dimension first
                return [_to_uint8_hwc_rgb(tt[i].numpy()) for i in range(int(tt.shape[0]))]
            if tt.ndim == 3:
                return [_to_uint8_hwc_rgb(tt.numpy())]
            if tt.ndim == 2:
                return [_to_uint8_hwc_rgb(tt.numpy())]
            raise ValueError(f"Unsupported torch tensor frame shape: {tuple(tt.shape)}")

        x = _maybe_unwrap_dict(payload)
        if isinstance(x, Image.Image):
            return [_to_uint8_hwc_rgb(np.array(x.convert("RGB")))]
        if isinstance(x, torch.Tensor):
            return _from_torch_tensor(x)
        if isinstance(x, np.ndarray):
            # If it's a stacked video tensor: THWC or TCHW
            if x.ndim == 4:
                return [_to_uint8_hwc_rgb(x[i]) for i in range(int(x.shape[0]))]
            return [_to_uint8_hwc_rgb(x)]
        if isinstance(x, (list, tuple)):
            items = _flatten_one_level(list(x))
            frames: list[np.ndarray] = []
            for item in items:
                frames.extend(VllmOmniEngine._normalize_payload_to_frames(item))
            return frames

        # Fall back: try numpy conversion
        arr = np.asarray(x)
        if arr.ndim in (2, 3, 4):
            if arr.ndim == 4:
                return [_to_uint8_hwc_rgb(arr[i]) for i in range(int(arr.shape[0]))]
            return [_to_uint8_hwc_rgb(arr)]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.omni = None
        self.params = config.get("params") or {}
        self.task = config.get("task", "text-to-video")

    def load(self) -> None:
        env_vars = (self.params.get("env") or {}) if isinstance(self.params, dict) else {}
        if env_vars:
            if not isinstance(env_vars, dict):
                raise TypeError(f"target.params.env must be a dict, got {type(env_vars)}")
            for k, v in env_vars.items():
                os.environ[str(k)] = str(v)
            if "DIFFUSION_ATTENTION_BACKEND" in env_vars:
                print(f"DIFFUSION_ATTENTION_BACKEND={os.environ.get('DIFFUSION_ATTENTION_BACKEND')}")

        try:
            from vllm_omni.entrypoints.omni import Omni
        except ImportError:
            raise ImportError("Please install vllm-omni to use VllmOmniEngine.")

        model_name = self.params.get("model")
        cache_backend = self.params.get("cache_backend")
        cache_config = self.params.get("cache_config")
        
        print(f"Loading vLLM-Omni model: {model_name}")
        if cache_backend:
            print(f"Using cache backend: {cache_backend}")
            
        self.omni = Omni(
            model=model_name,
            cache_backend=cache_backend,
            cache_config=cache_config,
        )

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run generation via vLLM-Omni.
        - vLLM-Omni's public `Omni.generate()` always calls `self.close()` in a `finally`,
          which adds shutdown overhead to wall-clock time and makes it hard to reuse the
          loaded model across multiple benchmark cases.
        """
        if self.omni is None:
            raise RuntimeError("Engine not loaded")

        prompt = case["prompt"]
        case_params = case.get("params") or {}
        
        # Default to 1024x1024 as recommended in docs for Qwen-Image
        height = case_params.get("height", 1024)
        width = case_params.get("width", 1024)
        num_frames = case_params.get("num_frames", 81)
        num_inference_steps = case_params.get("num_inference_steps", 50)
        guidance_scale = case_params.get("cfg_scale") or case_params.get("guidance_scale", 4.0)
        fps = case_params.get("fps", 16)
        seed = case_params.get("seed", 42)
        
        image_path = case_params.get("image_path")
        pil_image = None
        if image_path:
            if not os.path.isabs(image_path) and not image_path.startswith("/"):
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                image_path = os.path.join(repo_root, image_path)
            try:
                pil_image = load_image(image_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Failed to load image_path={image_path!r}: {e}") from e

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        omni_params_kwargs = {
            "pil_image": pil_image,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        sampling_params_list = []
        for stage_id, stage in enumerate(self.omni.stage_list):
            stage_type = getattr(stage, "stage_type", "llm")
            if stage_type == "diffusion":
                default_dict = self.omni.default_sampling_params_list[stage_id]
                sampling_params_list.append({**default_dict, **omni_params_kwargs})
            else:
                sampling_params_list.append(self.omni.default_sampling_params_list[stage_id])

        start_time = time.perf_counter()
        outputs = self.omni._run_generation(prompt, sampling_params_list, use_tqdm=False)
        latency = time.perf_counter() - start_time

        output_base_dir = self.config.get("output_dir", "outputs")
        case_dir = os.path.join(output_base_dir, case["id"])
        os.makedirs(case_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        def _unwrap_payload(obj):
            # vllm_omni.outputs.OmniRequestOutput (both pipeline and diffusion modes)
            if hasattr(obj, "request_output") and obj.request_output is not None:
                return _unwrap_payload(obj.request_output)
            if hasattr(obj, "images") and isinstance(obj.images, list) and obj.images:
                return obj.images
            return obj

        if not outputs:
            raise RuntimeError("vLLM-Omni returned empty outputs list")

        payload = _unwrap_payload(outputs[0])
        frames: list[Any]
        if isinstance(payload, Image.Image):
            frames = [payload]
        elif isinstance(payload, list):
            frames = payload
        else:
            frames = [payload]

        # Saving
        if not frames:
            raise RuntimeError("No frames/images found in vLLM-Omni output payload")

        is_video = ("video" in self.task) or (len(frames) > 1)
        if is_video:
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(case_dir, output_filename)

            video_frames = self._normalize_payload_to_frames(frames)
            if not video_frames:
                raise RuntimeError("No frames/images found after normalizing vLLM-Omni payload")
            try:
                export_to_video(video_frames, output_path, fps=fps)
            except ValueError:
                import imageio
                imageio.mimwrite(output_path, video_frames, fps=fps, quality=8)
        else:
            output_filename = f"output_{timestamp}.png"
            output_path = os.path.join(case_dir, output_filename)
            img_arrs = self._normalize_payload_to_frames(frames[0])
            if not img_arrs:
                raise RuntimeError("No image found after normalizing vLLM-Omni payload")
            Image.fromarray(img_arrs[0]).save(output_path)

        return {
            "e2e_latency": latency,
            "steady_state_seconds": latency, 
            "output_path": output_path,
        }

    def unload(self) -> None:
        if self.omni is not None:
            try:
                # Keep consistent with vLLM-Omni's lifecycle.
                self.omni.close()
            except Exception:
                pass
            del self.omni
            self.omni = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

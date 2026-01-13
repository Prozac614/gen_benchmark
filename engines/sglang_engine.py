import sys
import os
import torch
import time
from typing import Dict, Any
import statistics
import tempfile
import shutil
import json

from sglang.multimodal_gen import DiffGenerator
from .base import BaseEngine    

class SGLangEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.generator = None
        self.params = config.get("params") or {}
        self.model = self.params["model"]
        self.num_gpus = self.params.get("num_gpus", 1)
        self.server_args = self.params.get("server_args") or {}

    def load(self) -> None:
        if self.generator is None:
            env_vars = self.params.get("env", {})
            for k, v in env_vars.items():
                os.environ[str(k)] = str(v)
                print(f"Set env var: {k}={v}")

            server_args = dict(self.server_args)
            server_args.setdefault("model_path", self.model)
            server_args.setdefault("num_gpus", self.num_gpus)
            self.generator = DiffGenerator.from_pretrained(server_args=server_args, local_mode=True)

    def _parse_sglang_metrics(self, perf_log_dir: str) -> Dict[str, Any]:
        log_file = os.path.join(perf_log_dir, "performance.log")
        if not os.path.exists(log_file):
            return {}
        
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
            
            if not lines:
                return {}
                
            last_line = lines[-1]
            data = json.loads(last_line)
            
            total_ms = data.get("total_duration_ms", 0)
            return {
                "latency_seconds": total_ms / 1000.0,
                "denoise_steps_ms": data.get("denoise_steps_ms", [])
            }
        except Exception as e:
            print(f"Error parsing sglang metrics: {e}")
            return {}

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self.generator is None:
            raise RuntimeError("Engine not loaded")

        prompt = case["prompt"]
        case_params = case.get("params") or {}

        def _project_root() -> str:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def _resolve_local_path(p: str | None):
            if not p or not isinstance(p, str):
                return p
            # Allow comma-separated list for multi-image edit models.
            if "," in p:
                parts = [x.strip() for x in p.split(",") if x.strip()]
                out = []
                for part in parts:
                    if os.path.isabs(part) or part.startswith("/"):
                        out.append(part)
                    else:
                        out.append(os.path.join(_project_root(), part))
                return ",".join(out)
            if os.path.isabs(p) or p.startswith("/"):
                return p
            return os.path.join(_project_root(), p)
        
        output_dir = os.path.join(
            self.config.get("output_dir", "outputs"),
            case["id"]
        )
        os.makedirs(output_dir, exist_ok=True)
        
        sampling_params = dict(
            prompt=prompt,
            return_frames=False, # We just want to save output
            output_path=output_dir,
            save_output=True,
            num_inference_steps=case_params.get("num_inference_steps", 30),
            height=case_params.get("height", 512),
            width=case_params.get("width", 512),
            num_frames=case_params.get("num_frames", 25),
            guidance_scale=case_params.get("guidance_scale", 5.0),
        )

        if "image_path" in case_params and case_params["image_path"] is not None:
            sampling_params["image_path"] = _resolve_local_path(case_params["image_path"])
        elif "image_urls" in case_params and case_params["image_urls"] is not None:
            # Backward-compatible: allow passing a URL or a local path here.
            image_urls = case_params["image_urls"]
            if isinstance(image_urls, str):
                sampling_params["image_path"] = _resolve_local_path(image_urls)
            else:
                sampling_params["image_path"] = image_urls

        if "negative_prompt" in case_params and case_params["negative_prompt"] is not None:
            # Some SGLang pipelines reject empty-string negative_prompt during verification.
            neg = case_params["negative_prompt"]
            if isinstance(neg, str) and neg == "":
                neg = " "
            sampling_params["negative_prompt"] = neg
        
        if "seed" in case_params:
            sampling_params["seed"] = case_params["seed"]

        perf_log_dir = tempfile.mkdtemp()
        os.environ["SGLANG_PERF_LOG_DIR"] = perf_log_dir
        os.environ["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"

        try:
            repeats = int(case_params.get("repeats", 5))
            warmup_runs = int(case_params.get("warmup_runs", 0)) # Default 0 as requested "remove warmup"

            if warmup_runs > 0:
                warmup_params = dict(sampling_params)
                warmup_params["save_output"] = False
                for _ in range(warmup_runs):
                    self.generator.generate(sampling_params_kwargs=warmup_params)

            # Execute measured runs
            latencies: list[float] = []
            
            for i in range(repeats):
                current_params = dict(sampling_params)
                current_params["save_output"] = (i == repeats - 1)
                
                t_start = time.perf_counter()
                self.generator.generate(sampling_params_kwargs=current_params)
                t_end = time.perf_counter()
                
                metrics = self._parse_sglang_metrics(perf_log_dir)
                if metrics:
                    latencies.append(metrics["latency_seconds"])
                else:
                    latencies.append(t_end - t_start)
            
            # Process results
            if len(latencies) >= 3:
                # Remove min and max
                sorted_lats = sorted(latencies)
                valid_lats = sorted_lats[1:-1]
                avg_latency = statistics.mean(valid_lats)
            else:
                avg_latency = statistics.mean(latencies)

            return {
                "e2e_latency": avg_latency, 
                "steady_state_seconds": avg_latency,
                "steady_state_runs": latencies,
                "output_path": output_dir
            }
        finally:
            if os.path.exists(perf_log_dir):
                shutil.rmtree(perf_log_dir)


    def unload(self) -> None:
        if self.generator:
            try:
                self.generator.shutdown()
            except Exception as e:
                print(f"Error shutting down SGLang generator: {e}")
            del self.generator
            self.generator = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        time.sleep(5)

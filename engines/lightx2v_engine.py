import os
import time
import io
import re
from typing import Dict, Any

import hashlib
import torch
import gc
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base import BaseEngine

_LIGHTX2V_SET_INPUT_INFO_PATCHED = False

def _patch_lightx2v_set_input_info_once() -> None:
    global _LIGHTX2V_SET_INPUT_INFO_PATCHED
    if _LIGHTX2V_SET_INPUT_INFO_PATCHED:
        return
    try:
        import lightx2v.utils.input_info as input_info_mod  # type: ignore
    except Exception:
        return

    orig = getattr(input_info_mod, "set_input_info", None)
    if orig is None:
        return

    all_keys = getattr(input_info_mod, "ALL_INPUT_INFO_KEYS", set())

    def patched_set_input_info(args):
        info = orig(args)
        for k in all_keys:
            if hasattr(args, k):
                try:
                    setattr(info, k, getattr(args, k))
                except Exception:
                    pass
        return info

    input_info_mod.set_input_info = patched_set_input_info
    _LIGHTX2V_SET_INPUT_INFO_PATCHED = True

def _resolve_hf_cache_snapshot_path(model_path: str) -> str:
    if not model_path:
        return model_path
    p = model_path.rstrip("/")
    if os.path.exists(p):
        return p

    if p.endswith("/snapshots/current"):
        snapshots_dir = p[: -len("/current")]
        if os.path.isdir(snapshots_dir):
            try:
                entries = [
                    os.path.join(snapshots_dir, d)
                    for d in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, d))
                ]
                if entries:
                    entries.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    return entries[0]
            except Exception:
                return model_path

    return model_path

class LightX2VEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.params = config.get("params") or {}
        # Match LightX2V CLI naming for clarity.
        self.model_path = self.params["model_path"]
        self.model_cls = self.params.get("model_cls", "wan2.1")
        self.task = self.params.get("task", "t2v")
        self.config_json = self.params.get("config_json")
        self.dit_original_ckpt = self.params.get("dit_original_ckpt")
        self.low_noise_original_ckpt = self.params.get("low_noise_original_ckpt")
        self.high_noise_original_ckpt = self.params.get("high_noise_original_ckpt")
        self.offload = self.params.get("offload") 
        self.compile = bool(self.params.get("compile", False))
        self.parallel = self.params.get("parallel") 
        # LightX2V internal profiler:
        # 0 disable, 1 enable Level1_Log, 2 enable Level1_Log + Level2_Log
        self.profiling_debug_level = self.params.get("profiling_debug_level", 2)

    def load(self) -> None:
        if self.pipe is not None:
            return

        if os.getenv("PROFILING_DEBUG_LEVEL") is None:
            os.environ["PROFILING_DEBUG_LEVEL"] = str(int(self.profiling_debug_level))

        from lightx2v import LightX2VPipeline 
        _patch_lightx2v_set_input_info_once()

        resolved = _resolve_hf_cache_snapshot_path(self.model_path)
        if resolved != self.model_path:
            print(f"Resolved LightX2V model_path: {self.model_path!r} -> {resolved!r}")
            self.model_path = resolved

        if not os.path.exists(self.model_path):
            raise ValueError(
                "LightX2VEngine requires target.params.model_path to be a LOCAL directory path "
                "that contains model files (typically including `config.json`). "
                f"Got: {self.model_path!r}"
            )
        
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            found = False
            for subdir in ["transformer", "unet", "text_encoder"]:
                if os.path.exists(os.path.join(self.model_path, subdir, "config.json")):
                    found = True
                    break
            
            if not found:
                print(f"WARNING: config.json not found at {config_path}. Proceeding, but LightX2V might fail if it strictly requires it.")
        
        self.pipe = LightX2VPipeline(
            model_path=self.model_path,
            model_cls=self.model_cls,
            task=self.task,
        )

        if self.dit_original_ckpt:
            self.pipe.dit_original_ckpt = self.dit_original_ckpt
        if self.low_noise_original_ckpt:
            self.pipe.low_noise_original_ckpt = self.low_noise_original_ckpt
        if self.high_noise_original_ckpt:
            self.pipe.high_noise_original_ckpt = self.high_noise_original_ckpt

        if isinstance(self.offload, dict):
            self.pipe.enable_offload(**self.offload)
        if self.compile:
            self.pipe.enable_compile()
        if isinstance(self.parallel, dict): 
            self.pipe.enable_parallel(**self.parallel)

        if self.config_json:
            mc = str(self.model_cls or "")
            if mc.startswith("wan") or mc in ("wan2.1", "wan2.2"):
                self.pipe.set_infer_config_json(self.config_json)
                from lightx2v.utils.set_config import set_config, print_config

                cfg = set_config(self.pipe)
                try:
                    root_entries = os.listdir(cfg["model_path"])
                    has_root_safetensors = any(p.endswith(".safetensors") for p in root_entries)
                    transformer_dir = os.path.join(cfg["model_path"], "transformer")
                    has_transformer_safetensors = os.path.isdir(transformer_dir) and any(
                        p.endswith(".safetensors") for p in os.listdir(transformer_dir)
                    )
                    if (
                        not cfg.get("dit_original_ckpt")
                        and not cfg.get("low_noise_original_ckpt")
                        and not cfg.get("high_noise_original_ckpt")
                        and (not has_root_safetensors)
                        and has_transformer_safetensors
                    ):
                        cfg["dit_original_ckpt"] = transformer_dir
                except Exception:
                    pass
                if "dim" not in cfg:
                    for k in ("text_dim", "hidden_size", "model_dim"):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            cfg["dim"] = v
                            break
                if "num_heads" not in cfg:
                    for k in ("num_attention_heads", "n_heads"):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            cfg["num_heads"] = v
                            break
                # Wan weights/infer also expect in_dim/out_dim for patchify/unpatchify.
                if "in_dim" not in cfg:
                    for k in ("in_channels", "num_channels_latents"):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            cfg["in_dim"] = v
                            break
                if "out_dim" not in cfg:
                    for k in ("out_channels", "num_channels_latents"):
                        v = cfg.get(k)
                        if isinstance(v, int) and v > 0:
                            cfg["out_dim"] = v
                            break
                print_config(cfg)
                try:
                    self.pipe.runner = self.pipe._init_runner(cfg)
                except KeyError as e:
                    if str(e).strip("'") in ("patch_embedding.weight",):
                        raise RuntimeError(
                            "LightX2V(Wan) failed because required weight key "
                            f"{e} was not found. This typically means the checkpoint files were not loaded. "
                            "If you're using a Diffusers snapshot, weights are usually in "
                            "`<model_path>/transformer/*.safetensors` — set `dit_original_ckpt` to that "
                            "directory (or let the engine auto-detect it)."
                        ) from e
                    raise
            else:
                self.pipe.create_generator(config_json=self.config_json)

    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipe is None:
            raise RuntimeError("Engine not loaded")

        case_params = case.get("params") or {}
        prompt = case["prompt"]

        # Decide output suffix from task (LightX2V supports both image + video tasks).
        ext = "png" if self.task in ("t2i", "i2i") else "mp4"
        run_id = self.config.get("_run_id") or time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.config.get("output_dir", "outputs"),
            case["id"],
            f"output_{run_id}.{ext}",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not self.config_json:
            self.pipe.create_generator(
                attn_mode=case_params.get("attn_mode", "sage_attn2"),
                infer_steps=case_params.get("num_inference_steps", 50),
                height=case_params.get("height", 480),
                width=case_params.get("width", 832),
                num_frames=case_params.get("num_frames", 81),
                guidance_scale=case_params.get("guidance_scale", 5.0),
                sample_shift=case_params.get("sample_shift", 5.0),
                fps=case_params.get("fps", 16),
                aspect_ratio=case_params.get("aspect_ratio", "16:9"),
                boundary=case_params.get("boundary", 0.9),
                boundary_step_index=case_params.get("boundary_step_index", 2),
                denoising_step_list=case_params.get(
                    "denoising_step_list", [1000, 750, 500, 250]
                ),
            )

        start = time.perf_counter()
        log_buffer = io.StringIO()
        sink_id = None
        if hasattr(logger, "add"):
            sink_id = logger.add(log_buffer, format="{message}")
        def _project_root() -> str:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def _fallback_local_asset_for_url(url: str) -> str | None:
            try:
                base = os.path.basename(url.split("?", 1)[0])
                if not base:
                    return None
                cand = os.path.join(_project_root(), "figs", base)
                return cand if os.path.exists(cand) else None
            except Exception:
                return None

        def _download_with_retries(url: str) -> str:
            import requests
            import tempfile

            # Cache downloads under outputs so repeated runs don't keep hitting HF.
            cache_dir = os.path.join(self.config.get("output_dir", "outputs"), "_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Use a stable filename per URL basename.
            base = os.path.basename(url.split("?", 1)[0]) or "input.png"
            cached_path = os.path.join(cache_dir, base)
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
                return cached_path

            headers = {
                "User-Agent": "gen_benchmark/1.0 (LightX2VEngine)",
            }
            last_err: Exception | None = None
            for attempt in range(1, 6):
                try:
                    resp = requests.get(url, stream=True, headers=headers, timeout=60)
                    if resp.status_code == 429:
                        raise RuntimeError(f"429 Too Many Requests for url: {url}")
                    resp.raise_for_status()

                    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(base)[-1] or ".png")
                    os.close(fd)
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    # Atomic-ish move into cache
                    try:
                        os.replace(tmp_path, cached_path)
                        return cached_path
                    except Exception:
                        return tmp_path
                except Exception as e:
                    last_err = e
                    # Exponential-ish backoff
                    time.sleep(min(30, 2 ** attempt))
            raise RuntimeError(f"Failed to download image after retries: {last_err}") from last_err

        def _project_root() -> str:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        def _resolve_local_path(p: str | None) -> str | None:
            if not p or not isinstance(p, str):
                return None
            # LightX2V i2i can be "a.png,b.png" style; normalize each.
            if "," in p:
                parts = [x.strip() for x in p.split(",") if x.strip()]
                resolved_parts: list[str] = []
                for part in parts:
                    if os.path.isabs(part) or part.startswith("/"):
                        resolved_parts.append(part)
                    else:
                        resolved_parts.append(os.path.join(_project_root(), part))
                return ",".join(resolved_parts)
            if os.path.isabs(p) or p.startswith("/"):
                return p
            return os.path.join(_project_root(), p)

        def _preprocess_i2i_image_if_needed(image_path: str | None) -> str | None:
            """
            LightX2V Qwen i2i tends to respect the *input image* aspect ratio.
            input_image_preprocess: "none" (default) | "pad" | "pad_blur"
            """
            if not image_path:
                return None
            if self.task != "i2i":
                return image_path
            if str(self.model_cls or "") != "qwen_image":
                return image_path

            mode = str(case_params.get("input_image_preprocess") or "none").strip().lower()
            if mode in ("", "none", "off", "false", "0"):
                return image_path

            cs = case_params.get("custom_shape")
            if not (isinstance(cs, (list, tuple)) and len(cs) == 2):
                return image_path
            try:
                th, tw = int(cs[0]), int(cs[1])
            except Exception:
                return image_path
            if th <= 0 or tw <= 0:
                return image_path

            try:
                from PIL import Image, ImageFilter
            except Exception:
                return image_path

            def _process_one(p: str) -> str:
                if not p or not os.path.exists(p):
                    return p
                try:
                    st = os.stat(p)
                    key = f"{p}:{st.st_mtime_ns}:{mode}:{th}x{tw}"
                    hh = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                    cache_dir = os.path.join(
                        self.config.get("output_dir", "outputs"), "_cache", "preprocessed_inputs"
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    out_path = os.path.join(cache_dir, f"{hh}.png")
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                        return out_path

                    im = Image.open(p).convert("RGB")
                    w0, h0 = im.size
                    if w0 <= 0 or h0 <= 0:
                        return p

                    # Fit inside target, keep aspect ratio
                    scale = min(tw / w0, th / h0)
                    nw, nh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
                    fg = im.resize((nw, nh), resample=Image.BICUBIC)

                    if mode == "pad_blur":
                        bg = im.resize((tw, th), resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(radius=24))
                    else:
                        # plain pad
                        bg = Image.new("RGB", (tw, th), (0, 0, 0))

                    x0 = (tw - nw) // 2
                    y0 = (th - nh) // 2
                    bg.paste(fg, (x0, y0))
                    bg.save(out_path)
                    return out_path
                except Exception:
                    return p

            # Support comma-separated multi-image inputs
            if "," in image_path:
                parts = [x.strip() for x in image_path.split(",") if x.strip()]
                return ",".join(_process_one(p) for p in parts)
            return _process_one(image_path)

        image_path = _resolve_local_path(case_params.get("image_path"))
        if not image_path:
            image_urls = case_params.get("image_urls")
            if image_urls:
                if isinstance(image_urls, list):
                    image_urls = image_urls[0] # Take first one for now
                
                if isinstance(image_urls, str):
                    if image_urls.startswith("http"):
                        try:
                            logger.info(f"Downloading image from {image_urls}")
                            image_path = _download_with_retries(image_urls)
                        except Exception as e:
                            logger.error(f"Failed to download image: {e}")
                            fallback = _fallback_local_asset_for_url(image_urls)
                            if fallback:
                                logger.info(f"Using local fallback image: {fallback}")
                                image_path = fallback
                            else:
                                image_path = None
                    else:
                        # Assume local path
                        image_path = image_urls

        image_path = _preprocess_i2i_image_if_needed(image_path)

        if self.task in ("i2i", "i2v") and not image_path:
            raise RuntimeError(
                f"LightX2V {self.task} requires an input image, but none was available. "
            )

        try:
            if self.config_json:
                if "num_inference_steps" in case_params:
                    steps = int(case_params["num_inference_steps"])
                    self.pipe.infer_steps = steps
                if "height" in case_params:
                    self.pipe.target_height = int(case_params["height"])
                if "width" in case_params:
                    self.pipe.target_width = int(case_params["width"])
                if "num_frames" in case_params:
                    self.pipe.target_video_length = int(case_params["num_frames"])
                if "guidance_scale" in case_params:
                    self.pipe.sample_guide_scale = float(case_params["guidance_scale"])
                    self.pipe.enable_cfg = bool(self.pipe.sample_guide_scale != 1.0)
                if "sample_shift" in case_params:
                    self.pipe.sample_shift = float(case_params["sample_shift"])
                    if "fps" in case_params:
                        self.pipe.fps = int(case_params["fps"])
                if "aspect_ratio" in case_params:
                    ar = str(case_params["aspect_ratio"])
                    self.pipe.aspect_ratio = ar
                if str(self.model_cls or "") == "qwen_image":
                    cs = case_params.get("custom_shape")
                    if isinstance(cs, (list, tuple)) and len(cs) == 2:
                        self.pipe.custom_shape = [int(cs[0]), int(cs[1])]
                    else:
                        h = case_params.get("height")
                        w = case_params.get("width")
                        if isinstance(h, int) and isinstance(w, int):
                            self.pipe.custom_shape = [h, w]

            # lightx2v.utils.input_info.set_input_info() expects pipeline.target_shape to exist.
            if not hasattr(self.pipe, "target_shape") or getattr(self.pipe, "target_shape", None) in (None, ""):
                ts = None
                # Prefer custom_shape when present (Qwen uses it for exact sizes).
                cs = getattr(self.pipe, "custom_shape", None)
                if isinstance(cs, (list, tuple)) and len(cs) == 2:
                    try:
                        ts = [int(cs[0]), int(cs[1])]
                    except Exception:
                        ts = None
                if ts is None:
                    try:
                        h = int(getattr(self.pipe, "target_height", 0) or 0)
                        w = int(getattr(self.pipe, "target_width", 0) or 0)
                        if h > 0 and w > 0:
                            ts = [h, w]
                    except Exception:
                        ts = None
                if ts is None:
                    # Last resort: take from case params.
                    try:
                        h = int(case_params.get("height") or 0)
                        w = int(case_params.get("width") or 0)
                        if h > 0 and w > 0:
                            ts = [h, w]
                    except Exception:
                        ts = None
                if ts is not None:
                    try:
                        setattr(self.pipe, "target_shape", ts)
                    except Exception:
                        pass

            self.pipe.generate(
                seed=case_params.get("seed", 42),
                prompt=prompt,
                negative_prompt=case_params.get("negative_prompt", ""),
                image_path=image_path,
                last_frame_path=case_params.get("last_frame_path"),
                audio_path=case_params.get("audio_path"),
                save_result_path=output_path,
            )
        finally:
            if sink_id is not None:
                logger.remove(sink_id)

        e2e = time.perf_counter() - start
        
        logs = log_buffer.getvalue()
        inference_time = "N/A"
        run_pipeline_match = re.search(
            r"Level1_Log\s+RUN pipeline cost ([\d\.]+) seconds", logs, re.IGNORECASE
        )
        # LightX2V's profiler prints: "Level1_Log Total Cost cost X.XXXXXX seconds"
        total_cost_match = re.search(
            r"Level1_Log\s+Total Cost cost ([\d\.]+) seconds", logs, re.IGNORECASE
        )
        if total_cost_match is None:
            total_cost_match = re.search(
                r"Total Cost.*?([\d\.]+)\s*(?:s|sec|secs|seconds)\b",
                logs,
                re.IGNORECASE,
            )
        
        if run_pipeline_match:
             inference_time = float(run_pipeline_match.group(1))
        elif total_cost_match:
             inference_time = float(total_cost_match.group(1))
        return {
            "e2e_latency": e2e,
            "inference_time": inference_time, # Backward compatibility
            "steady_state_seconds": inference_time if isinstance(inference_time, float) else e2e,
            "output_path": output_path,
        }

    def unload(self) -> None:
        if self.pipe is not None:
            try:
                if hasattr(self.pipe, "runner"):
                    try:
                        runner = getattr(self.pipe, "runner", None)
                        if runner is not None:
                            for attr in (
                                "model",
                                "dit",
                                "transformer",
                                "unet",
                                "vae",
                                "text_encoder",
                                "image_encoder",
                                "clip",
                                "tokenizer",
                            ):
                                if hasattr(runner, attr):
                                    try:
                                        setattr(runner, attr, None)
                                    except Exception:
                                        pass
                        setattr(self.pipe, "runner", None)
                    except Exception:
                        pass
            finally:
                del self.pipe
                self.pipe = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        time.sleep(2)

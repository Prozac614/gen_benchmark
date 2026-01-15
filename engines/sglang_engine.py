import sys
import os
import torch
import time
from typing import Dict, Any
import statistics
import tempfile
import shutil
import json
import shlex
import subprocess
import signal

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
        self._pending_cli_verify = None

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

    def _format_equivalent_cli(self, *, sampling_params: Dict[str, Any]) -> str:
        def q(v: Any) -> str:
            return shlex.quote(str(v))

        def dq(v: Any) -> str:
            s = str(v)
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return f"\"{s}\""

        env_vars = self.params.get("env", {}) or {}
        env_prefix = " ".join(f"{k}={q(v)}" for k, v in env_vars.items())

        cmd: list[str] = ["sglang", "generate", f"--model-path={self.model}"]

        server_args = dict(self.server_args or {})
        for k in ("log_level", "warmup", "dit_layerwise_offload", "ulysses_degree"):
            if k in server_args:
                cmd.append(f"--{k.replace('_','-')}={server_args[k]}")

        # Sampling args
        if "prompt" in sampling_params:
            cmd.append(f"--prompt={dq(sampling_params['prompt'])}")
        if "negative_prompt" in sampling_params:
            cmd.append(f"--negative-prompt={dq(sampling_params['negative_prompt'])}")
        if "image_path" in sampling_params:
            cmd.append(f"--image-path={q(sampling_params['image_path'])}")
        if "width" in sampling_params:
            cmd.append(f"--width={sampling_params['width']}")
        if "height" in sampling_params:
            cmd.append(f"--height={sampling_params['height']}")
        if "num_inference_steps" in sampling_params:
            cmd.append(f"--num-inference-steps={sampling_params['num_inference_steps']}")
        if "num_frames" in sampling_params:
            cmd.append(f"--num-frames={sampling_params['num_frames']}")
        if "guidance_scale" in sampling_params:
            cmd.append(f"--guidance-scale={sampling_params['guidance_scale']}")
        if "seed" in sampling_params:
            cmd.append(f"--seed={sampling_params['seed']}")

        if sampling_params.get("save_output"):
            cmd.append("--save-output")
        if "output_path" in sampling_params:
            cmd.append(f"--output-path={q(sampling_params['output_path'])}")

        # Tokens are already safely formatted (numbers/bools or explicitly quoted),
        # so just join for readability (avoid turning our double quotes into single-quoted tokens).
        rendered = " ".join(str(x) for x in cmd)
        return (env_prefix + " " + rendered).strip() if env_prefix else rendered

    def _build_equivalent_cli_argv(self, *, sampling_params: Dict[str, Any]) -> list[str]:
        cmd: list[str] = ["sglang", "generate", f"--model-path={self.model}"]

        server_args = dict(self.server_args or {})
        for k in ("log_level", "warmup", "dit_layerwise_offload", "ulysses_degree"):
            if k in server_args:
                cmd.append(f"--{k.replace('_','-')}={server_args[k]}")

        if "prompt" in sampling_params:
            cmd.append(f"--prompt={sampling_params['prompt']}")
        if "negative_prompt" in sampling_params:
            cmd.append(f"--negative-prompt={sampling_params['negative_prompt']}")
        if "image_path" in sampling_params:
            cmd.append(f"--image-path={sampling_params['image_path']}")
        if "width" in sampling_params:
            cmd.append(f"--width={sampling_params['width']}")
        if "height" in sampling_params:
            cmd.append(f"--height={sampling_params['height']}")
        if "num_inference_steps" in sampling_params:
            cmd.append(f"--num-inference-steps={sampling_params['num_inference_steps']}")
        if "num_frames" in sampling_params:
            cmd.append(f"--num-frames={sampling_params['num_frames']}")
        if "guidance_scale" in sampling_params:
            cmd.append(f"--guidance-scale={sampling_params['guidance_scale']}")
        if "seed" in sampling_params:
            cmd.append(f"--seed={sampling_params['seed']}")
        if sampling_params.get("save_output"):
            cmd.append("--save-output")
        if "output_path" in sampling_params:
            cmd.append(f"--output-path={sampling_params['output_path']}")
        return cmd

    def _build_cli_verify_argv(self, *, sampling_params: Dict[str, Any]) -> list[str]:
        argv = self._build_equivalent_cli_argv(sampling_params=sampling_params)

        out: list[str] = []
        saw_warmup = False
        for tok in argv:
            if tok.startswith("--log-level="):
                continue
            if tok.startswith("--warmup="):
                out.append("--warmup=True")
                saw_warmup = True
                continue
            out.append(tok)
        if not saw_warmup:
            out.insert(3, "--warmup=True")
        return out

    def _format_cli_from_argv(self, argv: list[str]) -> str:
        """
        Render a CLI command string from argv, keeping prompt-like args double-quoted.
        """
        def dq(v: str) -> str:
            s = str(v)
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return f"\"{s}\""

        rendered: list[str] = []
        for tok in argv:
            if tok.startswith("--prompt="):
                rendered.append("--prompt=" + dq(tok[len("--prompt="):]))
            elif tok.startswith("--negative-prompt="):
                rendered.append("--negative-prompt=" + dq(tok[len("--negative-prompt="):]))
            else:
                rendered.append(shlex.quote(tok))
        return " ".join(rendered)

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
            # Keep URLs as-is (so image_path can be a remote URL like the CLI example).
            if p.startswith("http://") or p.startswith("https://"):
                return p
            # Allow comma-separated list for multi-image edit models.
            if "," in p:
                parts = [x.strip() for x in p.split(",") if x.strip()]
                out = []
                for part in parts:
                    if part.startswith("http://") or part.startswith("https://"):
                        out.append(part)
                        continue
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

        if bool(self.params.get("print_cli_command", True)):
            try:
                green = "\033[32m"
                reset = "\033[0m"
                sep = f"{green}{'=' * 60}{reset}"
                print("\n" + sep)
                print(f"{green}Equivalent CLI command{reset}")
                print(self._format_equivalent_cli(sampling_params=sampling_params))
                print(sep + "\n")
            except Exception as e:
                print(f"Warning: failed to format equivalent CLI command: {e}")

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

            if bool(self.params.get("verify_cli_command", False)):
                verify_params = dict(sampling_params)
                verify_out = os.path.join(output_dir, "__cli_verify")
                os.makedirs(verify_out, exist_ok=True)
                verify_params["output_path"] = verify_out
                verify_params["save_output"] = True

                argv = self._build_cli_verify_argv(sampling_params=verify_params)
                cli_str = self._format_cli_from_argv(argv)

                env = dict(os.environ)
                for k, v in (self.params.get("env", {}) or {}).items():
                    env[str(k)] = str(v)

                self._pending_cli_verify = {
                    "cli_str": cli_str,
                    "argv": argv,
                    "env": env,
                    "verify_out": verify_out,
                    "strict": bool(self.params.get("verify_cli_strict", False)),
                }

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
        def _ps_snapshot() -> list[tuple[int, int, str]]:
            """
            Return [(pid, ppid, cmd), ...] best-effort.
            We intentionally avoid extra deps (psutil) since this repo is lightweight.
            """
            try:
                out = subprocess.check_output(
                    ["ps", "-eo", "pid=,ppid=,cmd="],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return []
            rows: list[tuple[int, int, str]] = []
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                # pid ppid cmd...
                parts = line.split(None, 2)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except ValueError:
                    continue
                cmd = parts[2] if len(parts) >= 3 else ""
                rows.append((pid, ppid, cmd))
            return rows

        def _descendants_of(root_pid: int) -> list[tuple[int, str]]:
            rows = _ps_snapshot()
            children: dict[int, list[tuple[int, str]]] = {}
            for pid, ppid, cmd in rows:
                children.setdefault(ppid, []).append((pid, cmd))

            out: list[tuple[int, str]] = []
            stack = [root_pid]
            seen = {root_pid}
            while stack:
                cur = stack.pop()
                for pid, cmd in children.get(cur, []):
                    if pid in seen:
                        continue
                    seen.add(pid)
                    out.append((pid, cmd))
                    stack.append(pid)
            return out

        def _kill_pids(pids: list[int], sig: int) -> None:
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except Exception:
                    # Best-effort; don't fail unload() because one pid is protected.
                    pass

        def _cleanup_leaked_sglang_children() -> None:
            """
            Best-effort cleanup:
            - sglang local server often spawns worker/scheduler subprocesses.
            - On exceptions, those can survive and keep VRAM allocated.
            We only target descendants of *this* benchmark process and only those
            that look sglang-related to avoid collateral damage.
            """
            me = os.getpid()
            descendants = _descendants_of(me)
            # Heuristic: match known sglang diffusion processes/commands.
            suspect = [
                pid
                for pid, cmd in descendants
                if ("sglang" in cmd)
                or ("sgl-diffusion" in cmd)
                or ("sglang-diffusion" in cmd)
                or ("multimodal_gen" in cmd)
                or ("launch_server" in cmd)
            ]
            if not suspect:
                return

            # Try graceful stop first, then force-kill.
            _kill_pids(suspect, signal.SIGTERM)
            time.sleep(1.0)
            _kill_pids(suspect, signal.SIGKILL)

        if self.generator:
            try:
                self.generator.shutdown()
            except Exception as e:
                print(f"Error shutting down SGLang generator: {e}")
            del self.generator
            self.generator = None
        
        import gc
        gc.collect()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        
        time.sleep(5)
        # Extra safeguard: if shutdown() didn't fully reap subprocesses, kill leaked workers.
        try:
            _cleanup_leaked_sglang_children()
        except Exception as e:
            print(f"Warning: failed to cleanup leaked sglang subprocesses: {e}")

        # Execute deferred CLI verification after shutdown to avoid GPU OOM.
        if self._pending_cli_verify:
            green = "\033[32m"
            reset = "\033[0m"
            sep = f"{green}{'=' * 60}{reset}"
            try:
                print("\n" + sep)
                print(f"{green}CLI verify {reset}")
                print(self._pending_cli_verify["cli_str"])
                print(sep)
                subprocess.run(self._pending_cli_verify["argv"], env=self._pending_cli_verify["env"], check=True)
                print(f"{green}CLI verify: OK (output -> {self._pending_cli_verify['verify_out']}){reset}\n")
            except Exception as e:
                msg = f"CLI verify failed: {e}"
                if self._pending_cli_verify.get("strict"):
                    raise RuntimeError(msg) from e
                print(f"Warning: {msg}")
            finally:
                self._pending_cli_verify = None

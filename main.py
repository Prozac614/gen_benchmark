#!/usr/bin/env python3
"""
Benchmark runner for comparing generation performance across frameworks.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import re

from dotenv import load_dotenv

import yaml
import pandas as pd

import engines
from monitor import GPUMonitor


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge for dict values; override wins."""
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def _deep_merge_recursive(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, bv in b.items():
            av = out.get(k)
            out[k] = _deep_merge_recursive(av, bv)
        return out
    return b


def _merge_item_with_params(default_item: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    merged = _deep_merge_dicts(default_item, item)
    default_params = (default_item or {}).get("params") or {}
    item_params = (item or {}).get("params") or {}
    merged["params"] = _deep_merge_dicts(default_params, item_params)
    return merged


def _normalize_list_or_map(
    value: Any, *, key_field: str, defaults: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Accept either:
    - list[dict]
    - dict[name|id -> dict]
    and return list[dict] with `key_field` set when input is a dict.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        out: List[Dict[str, Any]] = []
        for k, v in value.items():
            if not isinstance(v, dict):
                raise TypeError(f"Expected mapping values to be dict, got {type(v)} for {k!r}")
            item = dict(v)
            item[key_field] = k
            out.append(_merge_item_with_params(defaults, item))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            if not isinstance(v, dict):
                raise TypeError(f"Expected list items to be dict, got {type(v)}")
            out.append(_merge_item_with_params(defaults, v))
        return out
    raise TypeError(f"Expected list or dict, got {type(value)}")


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config schema.

    Supported conveniences:
    - `defaults.case`: merged into every case (case fields win)
    - `defaults.target`: merged into every target (target fields win)
    - Ensure `params` dict exists for every case/target
    """
    defaults = config.get("defaults") or {}
    case_defaults = defaults.get("case") or {}
    target_defaults = defaults.get("target") or {}

    norm_cases = _normalize_list_or_map(config.get("cases"), key_field="id", defaults=case_defaults)
    norm_targets = _normalize_list_or_map(config.get("targets"), key_field="name", defaults=target_defaults)

    for c in norm_cases:
        c.setdefault("params", {})
    for t in norm_targets:
        t.setdefault("params", {})

    config = dict(config)
    config["cases"] = norm_cases
    config["targets"] = norm_targets
    return config


def _merge_configs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two raw (unnormalized) configs.

    Rules:
    - `defaults`: deep-merge recursively
    - `cases`: merge by `id` (last wins, shallow merge + shallow params merge)
    - `targets`: merge by `name` (or mapping key) (last wins, shallow merge + shallow params merge)
    - other top-level keys: last wins
    """
    a = dict(a or {})
    b = dict(b or {})

    out: Dict[str, Any] = dict(a)

    # defaults
    out["defaults"] = _deep_merge_recursive(a.get("defaults") or {}, b.get("defaults") or {})

    # cases
    a_cases = _normalize_list_or_map(a.get("cases"), key_field="id", defaults={})
    b_cases = _normalize_list_or_map(b.get("cases"), key_field="id", defaults={})
    cases_by_id: Dict[str, Dict[str, Any]] = {c.get("id"): c for c in a_cases if c.get("id")}
    for c in b_cases:
        cid = c.get("id")
        if not cid:
            raise ValueError("Each case must have an 'id'")
        if cid in cases_by_id:
            cases_by_id[cid] = _merge_item_with_params(cases_by_id[cid], c)
        else:
            cases_by_id[cid] = c
    out["cases"] = list(cases_by_id.values())

    # targets
    a_targets = _normalize_list_or_map(a.get("targets"), key_field="name", defaults={})
    b_targets = _normalize_list_or_map(b.get("targets"), key_field="name", defaults={})
    targets_by_name: Dict[str, Dict[str, Any]] = {t.get("name"): t for t in a_targets if t.get("name")}
    for t in b_targets:
        tn = t.get("name")
        if not tn:
            raise ValueError("Each target must have a 'name' (or be declared as a mapping key)")
        if tn in targets_by_name:
            targets_by_name[tn] = _merge_item_with_params(targets_by_name[tn], t)
        else:
            targets_by_name[tn] = t
    out["targets"] = list(targets_by_name.values())

    # other keys: last wins (excluding ones we handled)
    for k, v in b.items():
        if k in ("defaults", "cases", "targets", "includes"):
            continue
        out[k] = v

    return out


def _load_raw_with_includes(config_path: str, seen: set[str]) -> Dict[str, Any]:
    ap = os.path.abspath(config_path)
    if ap in seen:
        raise ValueError(f"Config include cycle detected at: {config_path}")
    seen.add(ap)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict, got {type(raw)} in {config_path}")

    base_dir = os.path.dirname(ap)
    includes = raw.get("includes") or []
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise TypeError(f"`includes` must be a list[str] or str, got {type(includes)} in {config_path}")

    merged: Dict[str, Any] = {}
    for inc in includes:
        if not isinstance(inc, str) or not inc.strip():
            raise TypeError(f"`includes` entries must be non-empty strings, got {inc!r} in {config_path}")
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        merged = _merge_configs(merged, _load_raw_with_includes(inc_path, seen))

    # Merge current file last (it overrides included fragments)
    raw_no_includes = dict(raw)
    raw_no_includes.pop("includes", None)
    merged = _merge_configs(merged, raw_no_includes)
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file (supports `includes:`)."""
    raw = _load_raw_with_includes(config_path, seen=set())
    return normalize_config(raw)


def get_engine_class(engine_class_name: str):
    """Dynamically get engine class by name."""
    return getattr(engines, engine_class_name)

def _slugify(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    # Replace path separators first (HF ids are like org/model)
    s = s.replace("/", "__")
    # Keep simple filesystem-friendly characters
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("_") or "unknown"


def _infer_framework(target: Dict[str, Any]) -> str:
    # Prefer engine class name (more reliable than target name)
    cls = (target.get("engine_class") or "").lower()
    if "sglang" in cls:
        return "sglang"
    if "lightx2v" in cls:
        return "lightx2v"
    # Fallback: target name prefix
    name = target.get("name", "")
    if isinstance(name, str) and "_" in name:
        return name.split("_", 1)[0]
    return "unknown"


def _infer_model_id(target: Dict[str, Any], framework: str) -> str:
    params = target.get("params") or {}
    # SGLang: HF id usually in params["model"]
    if "model" in params and isinstance(params.get("model"), str):
        return params["model"]
    # LightX2V: typically local model_path
    if "model_path" in params and isinstance(params.get("model_path"), str):
        mp = params["model_path"].rstrip("/")
        # Prefer extracting a readable HF name from local cache paths like:
        # .../hub/models--ORG--NAME/snapshots/<rev>
        m = re.search(r"models--([^/]+)--([^/]+)", mp)
        if m:
            org, name = m.group(1), m.group(2)
            return f"{org}/{name}"
        # Otherwise fall back to the directory name.
        return os.path.basename(mp) or mp
    # Fallback to target name without framework prefix
    name = target.get("name", "unknown")
    if isinstance(name, str) and name.startswith(framework + "_"):
        return name[len(framework) + 1 :]
    return str(name)


def _case_dir_from_output_path(output_path: str) -> str:
    if not output_path or output_path == "N/A":
        return ""
    # For LightX2V output_path is a file; for SGLang it's usually a directory.
    return output_path if os.path.isdir(output_path) else os.path.dirname(output_path)


def _select_cases_for_target(cases: List[Dict[str, Any]], target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Select which cases to run for a target.

    Priority:
      1) target["case_ids"] if provided (list[str] or str)
      2) fallback: match by task (backward-compatible behavior)
    """
    target_task = target.get("task", "image")

    case_ids = target.get("case_ids")
    if case_ids is not None:
        # Accept either list[str] or a single string (or comma-separated string).
        if isinstance(case_ids, str):
            # allow "a,b,c" or "a b c"
            parts = re.split(r"[,\s]+", case_ids.strip())
            wanted = [p for p in parts if p]
        elif isinstance(case_ids, list):
            wanted = []
            for x in case_ids:
                if isinstance(x, str) and x.strip():
                    wanted.append(x.strip())
                else:
                    raise TypeError(f"target.case_ids must be list[str], got item {x!r} ({type(x)})")
        else:
            raise TypeError(f"target.case_ids must be list[str] or str, got {type(case_ids)}")

        by_id = {c.get("id"): c for c in cases}
        selected: List[Dict[str, Any]] = []
        missing: List[str] = []
        for cid in wanted:
            c = by_id.get(cid)
            if c is None:
                missing.append(cid)
            else:
                selected.append(c)
        if missing:
            print(
                f"WARNING: Target {target.get('name','<unknown>')!r} requested missing case_ids: {', '.join(missing)}"
            )
        return selected

    # Backward compatibility: run all cases with matching task type.
    return [c for c in cases if c.get("task") == target_task]


def run_benchmark(config_path: str, target_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Run the benchmark suite.

    Args:
        config_path: Path to config.yaml
        target_names: Optional list of target names to filter. If None or empty, runs all targets.

    Returns:
        DataFrame with benchmark results
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    cases = config["cases"]
    targets = config["targets"]
    
    # Filter targets if target_names is provided
    if target_names is not None and len(target_names) > 0:
        available_target_names = [t["name"] for t in targets]
        targets = [t for t in targets if t["name"] in target_names]
        missing_targets = [name for name in target_names if name not in available_target_names]
        if missing_targets:
            print(f"WARNING: The following target names were not found in config: {', '.join(missing_targets)}")
    
    results: List[Dict[str, Any]] = []
    output_base_dir: Optional[str] = None
    frameworks_seen: set[str] = set()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for target in targets:
        target_name = target["name"]
        engine_class_name = target["engine_class"]
        target_type = target.get("type", "local")
        target_task = target.get("task", "image")
        target_output_dir = target.get("output_dir", "outputs")
        if output_base_dir is None:
            output_base_dir = target_output_dir

        framework = _infer_framework(target)
        model_id = _infer_model_id(target, framework)
        model_slug = _slugify(model_id)
        frameworks_seen.add(framework)

        target_output_dir = os.path.join(output_base_dir, framework, model_slug)
        target["output_dir"] = target_output_dir
        target["_run_id"] = run_id
        
        
        print(f"\n{'='*60}")
        print(f"Loading Engine: {target_name}")
        print(f"  Class: {engine_class_name}")
        print(f"  Type: {target_type}")
        print(f"  Task: {target_task}")
        print(f"{'='*60}")
        
        try:
            # Instantiate engine
            EngineClass = get_engine_class(engine_class_name)
            engine = EngineClass(target)
            
            # Load model
            print("Loading model...")
            engine.load()
            print("Model loaded successfully.")
            
            # Select cases (per-target case_ids if present; else fallback by task type)
            matching_cases = _select_cases_for_target(cases, target)
            
            if not matching_cases:
                print(f"  No matching cases for task type '{target_task}', skipping.")
                engine.unload()
                continue
            
            # Run each matching case
            for case in matching_cases:
                case_id = case["id"]
                print(f"\n  Running Case: {case_id}")
                print(f"    Prompt: {case['prompt'][:50]}...")
                
                # Determine if GPU monitoring should be enabled
                is_local = target_type == "local"
                
                # Run generation with monitoring
                with GPUMonitor(enabled=is_local) as monitor:
                    metrics = engine.generate(case)
                
                # Record results
                # Handle both fal engine (e2e_latency, inference_time) and other engines (latency)
                if "e2e_latency" in metrics:
                    e2e_latency_seconds = round(metrics["e2e_latency"], 4) if isinstance(metrics["e2e_latency"], (int, float)) else metrics["e2e_latency"]
                    inference_time = metrics.get("inference_time", "N/A")
                    inference_time_seconds = round(inference_time, 4) if isinstance(inference_time, (int, float)) else inference_time
                else:
                    # Backward compatibility for other engines
                    e2e_latency_seconds = round(metrics["latency"], 4) if isinstance(metrics["latency"], (int, float)) else metrics["latency"]
                    inference_time_seconds = "N/A"
                
                result = {
                    "engine": target_name,
                    "case_id": case_id,
                    "task": target_task,
                    "e2e_latency_seconds": e2e_latency_seconds,
                    "steady_state_seconds": round(metrics.get("steady_state_seconds"), 4)
                    if isinstance(metrics.get("steady_state_seconds"), (int, float))
                    else (metrics.get("steady_state_seconds", "N/A")),
                    "inference_time_seconds": inference_time_seconds,
                    "peak_vram_mb": round(monitor.peak_vram_mb, 2) if is_local else "N/A",
                    "output_path": metrics.get("output_path", "N/A"),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                
                # Save individual metrics
                case_dir = _case_dir_from_output_path(result["output_path"])
                if not case_dir:
                    # Fallback: use target_output_dir/case_id
                    case_dir = os.path.join(target_output_dir, case_id)
                os.makedirs(case_dir, exist_ok=True)
                # Keep a timestamped copy to avoid overwriting across runs, plus a convenient "latest".
                metrics_ts_path = os.path.join(case_dir, f"metrics_{run_id}.json")
                with open(metrics_ts_path, "w") as f:
                    json.dump(result, f, indent=2)
                with open(os.path.join(case_dir, "metrics.json"), "w") as f:
                    json.dump(result, f, indent=2)
                
                e2e_display = f"{result['e2e_latency_seconds']:.2f}s" if isinstance(result['e2e_latency_seconds'], (int, float)) else str(result['e2e_latency_seconds'])
                inference_display = f"{result['inference_time_seconds']:.2f}s" if isinstance(result['inference_time_seconds'], (int, float)) else str(result['inference_time_seconds'])
                print(f"    E2E Latency: {e2e_display}")
                steady_display = (
                    f"{result['steady_state_seconds']:.2f}s"
                    if isinstance(result["steady_state_seconds"], (int, float))
                    else str(result["steady_state_seconds"])
                )
                print(f"    Steady State: {steady_display}")
                print(f"    Inference Time: {inference_display}")
                if is_local:
                    print(f"    Peak VRAM: {result['peak_vram_mb']:.2f} MB")
                print(f"    Output: {result['output_path']}")
            
            # Unload model
            print(f"\nUnloading engine: {target_name}")
            engine.unload()
            
        except Exception as e:
            print(f"ERROR: Failed to run {target_name}: {e}")
            raise  # Fast fail as per plan
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    if output_base_dir is None:
        output_base_dir = "outputs"
    # Save CSV under outputs/<framework>/ when running a single framework; otherwise under outputs/.
    if len(frameworks_seen) == 1:
        only_fw = next(iter(frameworks_seen))
        return df, os.path.join(output_base_dir, only_fw)
    return df, output_base_dir


def main():
    """Main entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Benchmark runner for generation frameworks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help="Filter targets by name. Can specify multiple names. If not provided, runs all targets.",
    )
    args = parser.parse_args()
    
    print("="*60)
    print("  Generation Benchmark Runner")
    print("="*60)
    print(f"Config: {args.config}")
    print()
    
    # Run benchmark
    df, output_dir = run_benchmark(args.config, target_names=args.targets)
    
    # Print results
    print("\n" + "="*60)
    print("  BENCHMARK RESULTS")
    print("="*60 + "\n")
    
    if not df.empty:
        print(df.to_string(index=False))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("No results collected.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

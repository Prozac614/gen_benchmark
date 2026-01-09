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

from dotenv import load_dotenv

import yaml
import pandas as pd

import engines
from monitor import GPUMonitor


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_engine_class(engine_class_name: str):
    """Dynamically get engine class by name."""
    return getattr(engines, engine_class_name)


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
    first_target_output_dir = None
    
    # Iterate over targets
    for target in targets:
        target_name = target["name"]
        engine_class_name = target["engine_class"]
        target_type = target.get("type", "local")
        target_task = target.get("task", "image")
        target_output_dir = target.get("output_dir", "outputs")
        
        # Record first target's output_dir for CSV saving
        if first_target_output_dir is None:
            first_target_output_dir = target_output_dir
        
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
            
            # Filter cases by task type
            matching_cases = [c for c in cases if c.get("type") == target_task]
            
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
                    inference_time_seconds = round(metrics["inference_time"], 4) if isinstance(metrics["inference_time"], (int, float)) else metrics["inference_time"]
                else:
                    # Backward compatibility for other engines
                    e2e_latency_seconds = round(metrics["latency"], 4) if isinstance(metrics["latency"], (int, float)) else metrics["latency"]
                    inference_time_seconds = "N/A"
                
                result = {
                    "engine": target_name,
                    "case_id": case_id,
                    "task": target_task,
                    "e2e_latency_seconds": e2e_latency_seconds,
                    "inference_time_seconds": inference_time_seconds,
                    "peak_vram_mb": round(monitor.peak_vram_mb, 2) if is_local else "N/A",
                    "output_path": metrics.get("output_path", "N/A"),
                    "timestamp": datetime.now().isoformat(),
                }
                results.append(result)
                
                # Save individual metrics
                metrics_path = os.path.join(
                    target_output_dir, target_name, case_id, "metrics.json"
                )
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump(result, f, indent=2)
                
                e2e_display = f"{result['e2e_latency_seconds']:.2f}s" if isinstance(result['e2e_latency_seconds'], (int, float)) else str(result['e2e_latency_seconds'])
                inference_display = f"{result['inference_time_seconds']:.2f}s" if isinstance(result['inference_time_seconds'], (int, float)) else str(result['inference_time_seconds'])
                print(f"    E2E Latency: {e2e_display}")
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
    
    # Use first target's output_dir for CSV saving (already set in loop)
    if first_target_output_dir is None:
        first_target_output_dir = "outputs"
    
    return df, first_target_output_dir


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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

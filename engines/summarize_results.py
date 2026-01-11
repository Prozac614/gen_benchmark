#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List


def _iter_metrics_files(outputs_dir: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(outputs_dir):
        for fn in files:
            if fn == "metrics.json":
                out.append(os.path.join(root, fn))
    return sorted(out)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _md_escape(s: Any) -> str:
    s = "" if s is None else str(s)
    return s.replace("|", "\\|").replace("\n", " ")


def _to_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_md_escape(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="outputs dir (default: outputs)")
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    ap.add_argument("--csv", default=None, help="write CSV to path (optional)")
    args = ap.parse_args()

    outputs_dir = args.outputs
    if not os.path.isdir(outputs_dir):
        raise SystemExit(f"outputs dir not found: {outputs_dir}")

    files = _iter_metrics_files(outputs_dir)
    rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            rows.append(_read_json(p))
        except Exception:
            continue

    for r in rows:
        runs = r.get("steady_state_runs")
        runs_list: List[float] = []
        if isinstance(runs, list):
            for x in runs:
                if isinstance(x, (int, float)):
                    runs_list.append(float(x))
        r["steady_state_runs"] = json.dumps(runs_list) if runs_list else ""
        if runs_list:
            r["steady_state_p50"] = round(statistics.median(runs_list), 4)
            
            if len(runs_list) >= 3:
                sorted_runs = sorted(runs_list)
                valid_runs = sorted_runs[1:-1]
                r["steady_state_trimmed_mean"] = round(statistics.mean(valid_runs), 4)
            else:
                r["steady_state_trimmed_mean"] = round(statistics.mean(runs_list), 4)
        else:
            ss = r.get("steady_state_seconds")
            r["steady_state_p50"] = round(ss, 4) if isinstance(ss, (int, float)) else ss if ss is not None else ""
            r["steady_state_trimmed_mean"] = r["steady_state_p50"]

    cols = [
        "engine",
        "case_id",
        "task",
        "e2e_latency_seconds",
        "steady_state_seconds",
        "steady_state_trimmed_mean",
        "steady_state_p50",
        "steady_state_runs",
        "inference_time_seconds",
        "peak_vram_mb",
        "output_path",
        "timestamp",
    ]

    if args.csv:
        import csv

        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in cols})

    if args.format == "csv":
        import csv
        import sys

        w = csv.DictWriter(sys.stdout, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
        return 0

    print(_to_markdown_table(rows, cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


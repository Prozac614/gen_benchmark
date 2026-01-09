#!/usr/bin/env python3
"""Deployment script for fal serverless apps."""

import argparse
import subprocess
import sys

# App registry: app_name -> module_path:ClassName
APP_REGISTRY = {
    "z-image-turbo": "fal_apps.image.z_image_turbo:ZImageTurboApp",
    "wan-t2v-1-3b": "fal_apps.video.wan_t2v_1_3b:WanT2V1_3BApp",
}


def list_apps():
    """List all available apps."""
    print("Available apps:")
    for name, path in APP_REGISTRY.items():
        print(f"  - {name}: {path}")


def deploy_app(app_name: str):
    """Deploy a specific app."""
    if app_name not in APP_REGISTRY:
        print(f"Error: Unknown app '{app_name}'")
        print("Use --list to see available apps.")
        sys.exit(1)

    app_path = APP_REGISTRY[app_name]
    cmd = ["fal", "deploy", app_path]

    print(f"Deploying {app_name}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd="/root/gen_benchmark")
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Deploy fal serverless apps")
    parser.add_argument("--app", type=str, help="App name to deploy")
    parser.add_argument("--list", action="store_true", help="List available apps")

    args = parser.parse_args()

    if args.list:
        list_apps()
    elif args.app:
        deploy_app(args.app)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

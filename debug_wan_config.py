
import os
import torch
from transformers import AutoConfig

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"Config for {model_id}:")
    print(f"  dim: {getattr(config, 'dim', 'N/A')}")
    print(f"  num_attention_heads: {getattr(config, 'num_attention_heads', 'N/A')}")
    print(f"  attention_head_dim: {getattr(config, 'attention_head_dim', 'N/A')}")
    print(f"  text_dim: {getattr(config, 'text_dim', 'N/A')}")
except Exception as e:
    print(f"Error loading config: {e}")

# Check vllm_omni if possible
try:
    import vllm_omni
    print(f"vllm_omni version: {getattr(vllm_omni, '__version__', 'unknown')}")
    print(f"vllm_omni path: {vllm_omni.__file__}")
except ImportError:
    print("vllm_omni not found")

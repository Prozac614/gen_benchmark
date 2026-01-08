import time
import threading
from typing import Optional


class GPUMonitor:
    """Context manager for monitoring GPU memory usage during generation."""

    def __init__(self, enabled: bool = True, device_index: int = 0):
        """
        Initialize the GPU monitor.

        Args:
            enabled: Whether to enable monitoring (disable for remote APIs)
            device_index: GPU device index to monitor
        """
        self.enabled = enabled
        self.device_index = device_index
        self.peak_vram_mb: float = 0.0
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.handle = None

    def __enter__(self) -> "GPUMonitor":
        """Start monitoring GPU memory."""
        if not self.enabled:
            return self

        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            
            self.running = True
            self.peak_vram_mb = 0.0
            
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"Warning: GPU monitoring disabled due to error: {e}")
            self.enabled = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring and cleanup."""
        self.running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.enabled and self.handle is not None:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _monitor_loop(self) -> None:
        """Background loop to sample GPU memory usage."""
        import pynvml
        
        while self.running:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                current_vram_mb = info.used / (1024 * 1024)
                self.peak_vram_mb = max(self.peak_vram_mb, current_vram_mb)
            except Exception:
                pass
            
            time.sleep(0.05)  # Sample every 50ms

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEngine(ABC):
    """Abstract base class for all benchmark engines."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the engine with configuration.

        Args:
            config: Target configuration from YAML
        """
        self.config = config
        self.name = config.get("name", "unnamed_engine")

    def load(self) -> None:
        """Load the model. Override in subclass if needed."""
        pass

    @abstractmethod
    def generate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generation for a test case.

        Args:
            case: Test case configuration from YAML

        Returns:
            Dict containing:
                - e2e_latency: float (seconds) - end-to-end total latency (for fal engine)
                - inference_time: float (seconds) or "N/A" - pure inference time from API (for fal engine)
                - latency: float (seconds) - general latency (for other engines)
                - output_path: str (path to generated output)
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Unload the model and cleanup. Override in subclass if needed."""
        pass

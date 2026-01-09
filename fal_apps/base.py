"""Base classes and Pydantic models for fal serverless apps."""

from typing import Literal

import fal
from fal.toolkit.image import ImageSizeInput, Image, ImageSize, get_image_size
from fal.toolkit.image.safety_checker import postprocess_images
from fal.toolkit.video import Video
from fastapi import Response
from pydantic import Field, BaseModel


# =============================================================================
# Image Task Models
# =============================================================================

class ImageInput(BaseModel):
    """Input model for image generation."""

    prompt: str = Field(
        title="Prompt",
        description="The prompt to generate an image from.",
        examples=[
            "A futuristic cyberpunk city at night, neon lights reflecting on wet streets"
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to use.",
    )
    image_size: ImageSizeInput = Field(
        default=ImageSize(width=1024, height=1024),
        description="The size of the generated image.",
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of inference steps to perform.",
        ge=1,
        le=100,
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility.",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="CFG scale.",
        ge=0.0,
        le=20.0,
    )
    num_images: int = Field(
        default=1,
        description="The number of images to generate.",
        ge=1,
        le=4,
    )
    output_format: Literal["jpeg", "png"] = Field(
        default="png",
        description="The format of the generated image.",
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker.",
    )
    image_url: str | None = Field(
        default=None,
        description="Input image URL for image-to-image tasks.",
    )


class ImageOutput(BaseModel):
    """Output model for image generation."""

    images: list[Image] = Field(description="The generated image files info.")
    seed: int = Field(description="Seed used for generation.")
    has_nsfw_concepts: list[bool] = Field(description="NSFW detection results.")
    prompt: str = Field(description="The prompt used.")


# =============================================================================
# Video Task Models
# =============================================================================

class VideoInput(BaseModel):
    """Input model for video generation."""

    prompt: str = Field(
        title="Prompt",
        description="The prompt to generate a video from.",
        examples=[
            "A serene mountain landscape with flowing clouds"
        ],
    )
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to use.",
    )
    height: int = Field(
        default=480,
        description="Video height.",
        ge=256,
        le=1080,
    )
    width: int = Field(
        default=832,
        description="Video width.",
        ge=256,
        le=1920,
    )
    num_frames: int = Field(
        default=81,
        description="Number of frames to generate.",
        ge=1,
        le=256,
    )
    num_inference_steps: int = Field(
        default=30,
        description="The number of inference steps.",
        ge=1,
        le=100,
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility.",
    )
    guidance_scale: float = Field(
        default=6.0,
        description="CFG scale.",
        ge=0.0,
        le=20.0,
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="Enable safety checker.",
    )
    image_url: str | None = Field(
        default=None,
        description="Input image URL for image-to-video tasks.",
    )


class VideoOutput(BaseModel):
    """Output model for video generation."""

    video: Video = Field(description="The generated video file info.")
    seed: int = Field(description="Seed used for generation.")
    prompt: str = Field(description="The prompt used.")


# =============================================================================
# Base App Classes
# =============================================================================

class BaseImageApp(fal.App):
    """Base class for image generation apps."""

    # Subclass must override
    model_id: str = None

    # Default configuration
    keep_alive = 60
    min_concurrency = 0
    max_concurrency = 2

    def setup(self):
        """Load the pipeline."""
        import torch
        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to("cuda")

    @fal.endpoint("/")
    async def generate(self, input: ImageInput, response: Response) -> ImageOutput:
        """Generate images."""
        import torch

        # Preprocess input
        image_size = get_image_size(input.image_size)
        seed = input.seed or torch.seed()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Build pipeline arguments
        pipe_kwargs = {
            "prompt": input.prompt,
            "negative_prompt": input.negative_prompt,
            "num_inference_steps": input.num_inference_steps,
            "guidance_scale": input.guidance_scale,
            "height": image_size.height,
            "width": image_size.width,
            "num_images_per_prompt": input.num_images,
            "generator": generator,
        }

        # Add input image for I2I tasks
        if input.image_url:
            from diffusers.utils import load_image
            pipe_kwargs["image"] = load_image(input.image_url)

        # Generate images
        output = self.pipe(**pipe_kwargs)
        images = output.images

        # Apply safety checking
        postprocessed = postprocess_images(
            images,
            input.enable_safety_checker,
        )

        return ImageOutput(
            images=[
                Image.from_pil(image, input.output_format)
                for image in postprocessed["images"]
            ],
            seed=seed,
            has_nsfw_concepts=postprocessed["has_nsfw_concepts"],
            prompt=input.prompt,
        )


class BaseVideoApp(fal.App):
    """Base class for video generation apps."""

    # Subclass must override
    model_id: str = None

    # Default configuration
    keep_alive = 60
    min_concurrency = 0
    max_concurrency = 2

    def setup(self):
        """Load the pipeline."""
        import torch
        from diffusers import WanPipeline

        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @fal.endpoint("/")
    async def generate(self, input: VideoInput, response: Response) -> VideoOutput:
        """Generate videos."""
        import torch
        import tempfile
        import imageio

        # Setup seed
        seed = input.seed or torch.seed()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Build pipeline arguments
        pipe_kwargs = {
            "prompt": input.prompt,
            "negative_prompt": input.negative_prompt,
            "num_inference_steps": input.num_inference_steps,
            "guidance_scale": input.guidance_scale,
            "height": input.height,
            "width": input.width,
            "num_frames": input.num_frames,
            "generator": generator,
        }

        # Add input image for I2V tasks
        if input.image_url:
            from diffusers.utils import load_image
            pipe_kwargs["image"] = load_image(input.image_url)

        # Generate video
        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]  # Get first batch

        # Save to temporary file and upload
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            imageio.mimwrite(f.name, frames, fps=16, quality=8)
            video = Video.from_path(f.name)

        return VideoOutput(
            video=video,
            seed=seed,
            prompt=input.prompt,
        )

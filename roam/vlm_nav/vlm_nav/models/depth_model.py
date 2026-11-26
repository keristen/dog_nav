"""Monocular depth with Depth Anything V2 through Hugging Face transformers (pip install vlm_nav[depth])."""

import numpy as np
import torch
from PIL import Image
import os


class DepthEstimator:
    """Relative depth for an RGB image: larger values are farther away. Units are arbitrary."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf", device: str = None):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Read the local Hugging Face cache first. This avoids a network check on
        # every run after the model has been downloaded once.
        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
        try:
            self.processor = AutoImageProcessor.from_pretrained(model, local_files_only=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(model, local_files_only=True)
        except (OSError, ValueError):
            if offline:
                raise RuntimeError(
                    f"Depth model '{model}' is not available in the local Hugging Face cache "
                    "and HF_HUB_OFFLINE is enabled."
                )
            print(f"Depth model '{model}' is not cached; downloading it from Hugging Face...")
            self.processor = AutoImageProcessor.from_pretrained(model)
            self.model = AutoModelForDepthEstimation.from_pretrained(model)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # The relative models predict inverse depth (disparity), at the image size after post-processing.
        disparity = self.processor.post_process_depth_estimation(outputs, target_sizes=[image.size[::-1]])[0]
        disparity = disparity["predicted_depth"].float().cpu().numpy()
        disparity = disparity / max(float(disparity.max()), 1e-6)
        return 1.0 / np.clip(disparity, 1e-3, None)

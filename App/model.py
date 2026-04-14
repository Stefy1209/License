import os
import numpy as np
import torch
from typing import Optional
from depth_anything_3.api import DepthAnything3

# Help PyTorch find CUDA libraries in Conda environments
if "CONDA_PREFIX" in os.environ:
    conda_lib = os.path.join(os.environ["CONDA_PREFIX"], "lib")
    os.environ["LD_LIBRARY_PATH"] = conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def load_model(model_id: str, device: str) -> torch.nn.Module:
    """Load DepthAnything3 and optionally compile it."""
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

    model = DepthAnything3.from_pretrained(model_id).to(device)
    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    return model


@torch.inference_mode()
def estimate_depth(
    rgb_frame: np.ndarray,
    model: torch.nn.Module,
    intrinsics: Optional[np.ndarray],
    device: str,
) -> np.ndarray:
    """Run inference; returns a float32 depth map at model resolution."""
    ctx = torch.amp.autocast("cuda") if device == "cuda" else torch.no_grad()
    with ctx:
        prediction = model.inference(
            image=[rgb_frame],
            intrinsics=[intrinsics] if intrinsics is not None else None,
        )
    return prediction.depth[0]

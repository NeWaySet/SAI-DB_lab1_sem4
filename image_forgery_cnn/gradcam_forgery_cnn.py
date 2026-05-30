from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_forgery_cnn import build_model, build_transforms, load_image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(image_tensor).squeeze()
        probability = float(torch.sigmoid(logit).detach().cpu().item())
        logit.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        cam = (cam - cam_min) / max(cam_max - cam_min, 1e-8)
        return cam, probability

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Grad-CAM visualization for forgery detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam_overlay.png")
    return parser.parse_args()


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, output_path: Path, probability: float) -> None:
    original = original.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    original_np = np.asarray(original).astype(np.float32) / 255.0
    color_map = plt.get_cmap("jet")(heatmap)[..., :3]
    overlay = np.clip(0.58 * original_np + 0.42 * color_map, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Input image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay, P(forged)={probability:.3f}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 224))
    input_mode = str(config.get("input_mode", "rgb"))
    ela_quality = int(config.get("ela_quality", 90))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = build_transforms(image_size, train=False)
    visual_image = Image.open(image_path).convert("RGB")
    model_image = load_image(image_path, input_mode=input_mode, ela_quality=ela_quality)
    tensor = transform(model_image).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.features[-1])
    try:
        heatmap, probability = gradcam(tensor)
    finally:
        gradcam.close()

    overlay_heatmap(visual_image, heatmap, output_path, probability)
    print(f"Saved Grad-CAM to: {output_path}")
    print(f"Probability of forged/tampered class: {probability:.4f}")


if __name__ == "__main__":
    main()

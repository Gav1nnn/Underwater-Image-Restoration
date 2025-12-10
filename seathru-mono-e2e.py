# seathru-mono-e2e.py
import argparse
import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

import deps.monodepth2.networks as networks
from sea_thru_pipeline import run_seathru_pipeline


###############################################################
# Utility Functions
###############################################################
# 图片加载与预处理
def load_and_resize_image(image_path, target_size):
    """Load image and resize while keeping aspect ratio."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # keep aspect ratio
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return img, img_resized


def pil_to_tensor(pil_img):
    """Convert PIL image to normalized tensor (0–1)."""
    transform = transforms.ToTensor()
    return transform(pil_img).unsqueeze(0)


def tensor_to_np(t):
    """Tensor -> numpy image 0–255 uint8."""
    img = t.squeeze().detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    return np.transpose(img, (1, 2, 0))


###############################################################
# Monodepth2 Depth Prediction
###############################################################

def load_monodepth_model(model_name, device):
    """Load encoder and depth decoder."""
    print(f"[INFO] Loading monodepth2 model: {model_name}")

    # <-- 修改后的路径（适配你当前的根目录 models 文件夹）
    model_path = os.path.join("models", model_name)

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, feed_width, feed_height



def predict_depth(img_resized, encoder, decoder, feed_w, feed_h, device,
                  depth_scale, depth_offset):
    """Run monodepth2 and output depth aligned to resized image."""

    # Resize to feed resolution:
    img_resized_feed = img_resized.resize((feed_w, feed_h), Image.LANCZOS)
    input_tensor = pil_to_tensor(img_resized_feed).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp,
            (img_resized.height, img_resized.width),
            mode="bilinear",
            align_corners=False
        )

    # convert disparity → depth (simple conversion)
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    depth = 1.0 / (disp_resized_np + 1e-7)

    # scale/offset adjustment
    depth = depth * depth_scale + depth_offset

    return depth.astype(np.float32)


###############################################################
# Main Pipeline
###############################################################

def process_image_with_seathru(
    image_path,
    model_name,
    output_path,
    size=1024,
    depth_scale=10.0,
    depth_offset=2.0,
    no_cuda=False,
    save_depth=False,
    save_intermediate=False
):
    """Full Sea-Thru pipeline."""

    t0 = time.time()

    # ------------------------------------------
    # Device
    # ------------------------------------------
    device = torch.device("cpu" if no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------
    # Load image
    # ------------------------------------------
    original_img, img_resized = load_and_resize_image(image_path, size)
    print(f"[INFO] Loaded and resized image to: {img_resized.size}")

    # ------------------------------------------
    # Load monodepth model
    # ------------------------------------------
    encoder, decoder, feed_w, feed_h = load_monodepth_model(model_name, device)

    # ------------------------------------------
    # Predict depth map
    # ------------------------------------------
    depth = predict_depth(
        img_resized,
        encoder,
        decoder,
        feed_w,
        feed_h,
        device,
        depth_scale,
        depth_offset
    )

    if save_depth:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        depth_img = (depth_norm * 255).astype(np.uint8)
        Image.fromarray(depth_img).save("depth_debug.png")
        print("[INFO] Saved depth_debug.png")

    # ------------------------------------------
    # Run Sea-Thru pipeline
    # ------------------------------------------
    img_resized_np = np.array(img_resized)
    recovered = run_seathru_pipeline(
        img_resized_np,
        depth,
        save_intermediate=save_intermediate
    )

    # ------------------------------------------
    # Resize recovered → original
    # ------------------------------------------
    recovered_img = Image.fromarray(recovered.astype(np.uint8))
    final = recovered_img.resize(original_img.size, Image.LANCZOS)

    # ------------------------------------------
    # Save output
    # ------------------------------------------
    final.save(output_path)
    print(f"[SUCCESS] Output saved → {output_path}")
    print(f"[TIME] Total = {time.time() - t0:.2f}s")


###############################################################
# Command Line Interface
###############################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output", type=str, default="out.png")

    # Sizes
    parser.add_argument("--size", type=int, default=1024)

    # Depth controls
    parser.add_argument("--depth-scale", type=float, default=10.0)
    parser.add_argument("--depth-offset", type=float, default=2.0)

    # Flags
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")

    args = parser.parse_args()

    process_image_with_seathru(
        image_path=args.image,
        model_name=args.model_name,
        output_path=args.output,
        size=args.size,
        depth_scale=args.depth_scale,
        depth_offset=args.depth_offset,
        no_cuda=args.no_cuda,
        save_depth=args.save_depth,
        save_intermediate=args.save_intermediate
    )

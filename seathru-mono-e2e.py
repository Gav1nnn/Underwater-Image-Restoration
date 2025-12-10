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
# 隐藏所有的UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# ----------------------------
# 工具函数
# ----------------------------
# 图片加载与预处理
def load_and_resize_image(image_path, target_size):
    """加载图像并按比例缩放至目标大小 (最长边为 target_size)。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 保持宽高比
    if w > h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    return img, img_resized


def pil_to_tensor(pil_img):
    """将 PIL 图像转换为归一化的 PyTorch Tensor (0–1)，并添加 Batch 维度。"""
    transform = transforms.ToTensor()
    return transform(pil_img).unsqueeze(0)


def tensor_to_np(t):
    """将 Tensor 转换回 0–255 uint8 的 NumPy 图像格式。"""
    img = t.squeeze().detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    return np.transpose(img, (1, 2, 0))


# ----------------------------
# Monodepth2 深度预测
# ----------------------------

def load_monodepth_model(model_name, device):
    """Load encoder and depth decoder."""
    print(f"[INFO] Loading monodepth2 model: {model_name}")

    # # 模型路径
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
    """运行 Monodepth2 模型并输出与缩放图像对齐的深度图。"""

    # 将缩放后的图像进一步缩放到模型所需的输入分辨率
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

    # 将视差 (disp) 转换为深度 (depth)
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    depth = 1.0 / (disp_resized_np + 1e-7)

    # 应用用户指定的深度比例缩放和偏移 (用于调整预测深度与实际深度的关系)
    depth = depth * depth_scale + depth_offset
    # 返回浮点数深度图
    return depth.astype(np.float32)


# ----------------------------
# Main Pipeline (主管道)
# ----------------------------

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
    """Monodepth2 + Sea-Thru 完整的端到端管道。"""

    t0 = time.time()

    # ------------------------------------------
    # Device 设备选择
    # ------------------------------------------
    if no_cuda:
        # 用户强制要求不用 GPU
        device = torch.device("cpu")

    else:
        # 优先 CUDA → 再 MPS → 最后 CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # ------------------------------------------
    # Load image 加载图片
    # ------------------------------------------
    original_img, img_resized = load_and_resize_image(image_path, size)
    print(f"[INFO] Loaded and resized image to: {img_resized.size}")

    # ------------------------------------------
    # Load monodepth model 加载深度模型
    # ------------------------------------------
    encoder, decoder, feed_w, feed_h = load_monodepth_model(model_name, device)

    # ------------------------------------------
    # Predict depth map 预测深度图
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
        # 将深度图归一化到 0-255 范围以便保存为图像
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        depth_img = (depth_norm * 255).astype(np.uint8)
        Image.fromarray(depth_img).save("depth_debug.png")
        print("[INFO] Saved depth_debug.png")

    # ------------------------------------------
    # Run Sea-Thru pipeline 运行Sea-Thru
    # ------------------------------------------
    img_resized_np = np.array(img_resized)
    recovered = run_seathru_pipeline(
        img_resized_np,
        depth,
        save_intermediate=save_intermediate
    )

    # ------------------------------------------
    # Resize recovered → original 回复道原始尺寸
    # ------------------------------------------
    recovered_img = Image.fromarray(recovered.astype(np.uint8))
    final = recovered_img.resize(original_img.size, Image.LANCZOS)

    # ------------------------------------------
    # Save output 保存输出
    # ------------------------------------------
    final.save(output_path)
    print(f"[SUCCESS] Output saved → {output_path}")
    print(f"[TIME] Total = {time.time() - t0:.2f}s")


# ----------------------------
# 命令行接口
# ----------------------------

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

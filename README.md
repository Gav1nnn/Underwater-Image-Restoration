# 🌊 Sea-Thru + Monodepth2：水下图像端到端复原系统

本项目实现了基于 **Sea-Thru 物理模型** 与 **Monodepth2 单目深度估计网络** 的 **端到端水下图像复原流程**。

你只需要输入一张水下照片，系统即可自动完成：

- 单目深度估计
- 水下光衰减建模
- Backscatter 去除
- 颜色真实恢复
- 光照补偿

最终生成一张能够“还原到真实水下场景”的高清图像。

------

# 📁 项目结构

```
MyIR/
│── seathru-mono-e2e.py          # 主入口，完成完整的 E2E 处理流程
│── sea_thru_pipeline.py         # Sea-Thru 算法核心流程
│── deps/
│     └── monodepth2/            # Monodepth2 官方实现
│── models/
│     └── mono_1024x320/         # 预训练模型：encoder.pth / depth.pth
│── input/                       # 输入目录（需手动创建）
│── output/                      # 输出目录（需手动创建）
└── README.md
```

------

# 🔧 环境配置

支持 **Windows / macOS（Intel + M1/M2/M3/M4 Apple Silicon）**

推荐使用 **conda** 创建环境。

------

## 1）创建虚拟环境

```bash
uv venv --python 3.9 seathru
source seathru-mac/bin/activate
```

------

## 2）安装依赖

### 基础依赖：

```bash
uv pip install numpy scipy scikit-image pillow matplotlib rawpy scikit-learn pyyaml joblib
```

### PyTorch（这里是mac上的CPU/MPS版本）

```bash
uv pip install torch torchvision torchaudio
```

如果你是 Windows 且 CUDA 不兼容，强制使用 CPU：

```
--no-cuda
```

### macOS Apple Silicon 提示

PyTorch 会自动使用 MPS（Metal），无需额外操作。

------

# ▶️ 使用方法（CLI）

进入项目目录：

```
cd MyIR
```

执行处理命令：

```
python seathru-mono-e2e.py \
  --image input/example.jpg \
  --model-name mono_1024x320 \
  --output output/result.png \
  --no-cuda
```

------

# ⚙️ 参数说明

| 参数                  | 说明                              |
| --------------------- | --------------------------------- |
| `--image`             | 输入水下图像路径                  |
| `--model-name`        | 选择模型目录名称（models 下）     |
| `--output`            | 输出恢复图像路径                  |
| `--size`              | 图像处理的最大分辨率（默认 1024） |
| `--depth-scale`       | 深度图整体缩放                    |
| `--depth-offset`      | 深度补偿偏移                      |
| `--no-cuda`           | 强制 CPU 模式                     |
| `--save-depth`        | 保存深度图调试文件                |
| `--save-intermediate` | 保存中间结果（便于可视化）        |

运行成功后，你将在 `output/` 中获得最终恢复图像。

------

# 🧠 算法流程说明

整个 Sea-Thru + Monodepth2 端到端流程如下：

1. **图像读取并等比缩放**
2. **Monodepth2 深度预测**
3. **估计水体散射参数（β、D）**
4. **Backscatter 去除**
5. **光照衰减反演（Signal Restoration）**
6. **色彩校正与重建**
7. **恢复图像缩放回原尺寸**
8. **输出最终结果**

主函数入口位于：

```py
process_image_with_seathru()
```

前端无需了解算法细节，仅需调用输出结果的函数即可。

------

# 🔍 调试输出（可选）

加入参数：

```
--save-intermediate --save-depth
```

将生成：

- `debug_input.png`
- `debug_backscatter.png`
- `debug_illum.png`
- `debug_betaD_r.png`
- `debug_recovered.png`
- `depth_debug.png`

非常适合研究、展示和论文说明。

------

# ⚠️ 常见问题

### 1）FileNotFoundError: input.jpg

请确保图像在 `input/` 或给出绝对路径。

### 2）CUDA 错误

在 Windows 上常见，使用：

```
--no-cuda
```

### 3）MPS 警告（macOS）

正常，无需处理。

------
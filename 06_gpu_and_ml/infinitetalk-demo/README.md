# InfiniteTalk on Modal

这是一个在 [Modal](https://modal.com) 上运行 [InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) 的完整示例。
InfiniteTalk 是一个高质量的音频驱动人像视频生成模型 (Talking Head Generation)。

## ✨ 特性

- **全自动环境构建**：自动配置 PyTorch 2.5, CUDA 12.4, Flash Attention 等复杂依赖。
- **模型持久化**：使用 Modal Volume (`infinitetalk-models`) 缓存 20GB+ 的模型权重，无需每次下载。
- **高性能推理**：自动申请 **A100-80GB** 显卡处理大型 Wan2.1 14B 模型。
- **命令行工具**：内置 CLI 工具，支持快速验证和生成。
- **Web UI**：提供 Gradio 界面进行交互式生成 (WIP)。

## 🛠️ 准备工作

1. **注册 Modal 账号**: [modal.com](https://modal.com)
2. **设置 HuggingFace Token**:
   在 Modal 后台 Secrets 中创建一个名为 `huggingface-secret` 的 Secret，包含 key `HF_TOKEN`。
   这是下载 InfiniteTalk 模型所必需的。

## 🚀 快速开始

### 1. 首次运行（自动下载模型）

首次运行时，脚本会自动检测并下载所需模型（约 25GB）。建议先跑一次下载流程：

```bash
modal run infinitetalk_demo.py --download-only
```

### 2. 生成视频（CLI 模式）

使用自带的测试素材（或者你自己的图片/音频）生成视频：

```bash
modal run infinitetalk_demo.py \
    --image-path ./test_face.png \
    --audio-path ./en-Alice_woman_4s.wav \
    --output-path ./output.mp4
```

### 3. 使用参数优化生成

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image-path` | - | 输入人像图片路径 (PNG/JPG) |
| `--audio-path` | - | 输入语音音频路径 (WAV/MP3) |
| `--output-path` | `/tmp/...` | 输出视频保存路径 |
| `--resolution` | `480` | 分辨率 (目前支持 480，未来支持 720) |
| `--sample-steps` | `40` | 采样步数。**测试建议设为 10-20 以大幅加速**，正式生成建议 30-50。 |
| `--download-only` | `False` | 仅下载模型不推理 |

**极速测试命令（约 3-5 分钟）**：

```bash
modal run infinitetalk_demo.py \
    --image-path ./test_face.png \
    --audio-path ./en-Alice_woman_4s.wav \
    --sample-steps 10 \
    --resolution 480
```

## ⚠️ 常见问题

### 1. 运行超时 (Timeout)
Wan2.1 14B 模型非常大，编译和推理需要时间。如果遇到超时：
- 我们已将超时设置为 2 小时。
- 请确保本地网络稳定，或使用 `caffeinate -i ...` 防止电脑休眠断开连接。

### 2. 显存不足 (OOM)
默认配置使用 **A100-80GB**。不要尝试在 A10G 或 T4 上运行此模型，显存不足均会失败。

### 3. 生成速度慢
- 首次运行需要 JIT 编译，会慢几分钟。后续运行会变快。
- 减少 `--sample-steps` 可以线性提升速度（牺牲一定画质）。

## 🔧 开发调试

- **实时日志**：脚本已配置实时流式日志，你可以看到具体的进度条。
- **本地调试**：`local_entrypoint` 负责参数解析和上传文件，推理逻辑在远程 `InfiniteTalkModel` 类中。

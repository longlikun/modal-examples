#!/usr/bin/env python3
# infinitetalk_demo.py
#
# 运行方式:
#   1. 首次运行下载模型: modal run infinitetalk_demo.py
#   2. 部署 Web 服务: modal deploy infinitetalk_demo.py
#
# 前置条件:
#   - 在 Modal 平台上设置一个名为 "huggingface-secret" 的 Secret
#   - 值为你的 Hugging Face 读取令牌 (HF_TOKEN=hf_xxx)
#   - Token 获取地址: https://huggingface.co/settings/tokens

import modal

# 1. 定义镜像 (Image)
# -----------------
# 使用 CUDA 基础镜像，安装所有必要的系统和 Python 依赖项

REPO_DIR = "/root/InfiniteTalk"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",  # 升级 CUDA 版本
        add_python="3.10",
    )
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("pip==24.0")
    # 先安装 PyTorch 2.5+ (xfuser 需要 torch.distributed.tensor.experimental)
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # 克隆 InfiniteTalk 仓库
    .run_commands(
        f"cd /root && git clone https://github.com/MeiGen-AI/InfiniteTalk.git",
    )
    # 安装 xformers (与 PyTorch 2.5 兼容)
    .pip_install(
        "xformers==0.0.28.post3",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    # 安装项目依赖 (从 requirements.txt 手动提取)
    .pip_install(
        "opencv-python>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.44.0,<4.46.0",  # 4.44.x 不会默认使用 SDPA
        "tokenizers",  # 让 pip 自动解析兼容版本
        "accelerate>=1.1.1",
        "tqdm",
        "imageio",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "scikit-image",
        "loguru",
        "gradio>=5.0.0",
        "numpy>=1.23.5,<2",
        "xfuser>=0.4.1",
        "pyloudnorm",
        "optimum-quanto==0.2.6",
        "scenedetect",
        "moviepy==1.0.3",
        "decord",
    )
    # 安装 flash_attn 及其他依赖
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        "misaki[en]",
        "psutil",
        "librosa",
        "safetensors",
        "huggingface-hub",
        "einops",
    )
    # 必须安装 flash_attn (PyTorch 2.5 需要从源码编译，耗时约 5-10 分钟)
    .pip_install("flash_attn>=2.6.3")
)

app = modal.App(name="infinitetalk-demo", image=image)

# 2. 定义持久化存储 (Volume)
# -----------------------
WEIGHTS_DIR = "/models"
MODELS_VOLUME = modal.Volume.from_name("infinitetalk-models", create_if_missing=True)


# 3. 模型下载函数
# ---------------------
@app.function(
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,  # 允许 60 分钟下载
)
def download_models():
    """下载 InfiniteTalk 所需的所有模型"""
    import subprocess
    import os
    import shutil
    from pathlib import Path

    models = {
        "Wan2.1-I2V-14B-480P": {
            "repo": "Wan-AI/Wan2.1-I2V-14B-480P",
            # 必须检查 T5 编码器文件 (约 9GB) 和 VAE
            "validate": ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json"]
        },
        "chinese-wav2vec2-base": {
            "repo": "TencentGameMate/chinese-wav2vec2-base",
            "validate": ["config.json", "preprocessor_config.json"]
        },
        "InfiniteTalk": {
            "repo": "MeiGen-AI/InfiniteTalk",
            "validate": ["single/infinitetalk.safetensors", "multi/infinitetalk.safetensors"]
        },
    }

    for local_name, config in models.items():
        hub_name = config["repo"]
        validate_files = config["validate"]
        local_path = os.path.join(WEIGHTS_DIR, local_name)
        
        # 验证模型通过
        is_valid = True
        if not os.path.exists(local_path):
            is_valid = False
        else:
            for v_file in validate_files:
                v_path = os.path.join(local_path, v_file)
                if not os.path.exists(v_path):
                    print(f"⚠️  Missing validation file: {v_file}")
                    is_valid = False
                    break
        
        if is_valid:
            print(f"✓ Model {local_name} verified.")
            continue

        # 如果验证失败，清理目录并重新下载
        if os.path.exists(local_path):
            print(f"↻ Validation failed for {local_name}. Cleaning up and re-downloading...")
            shutil.rmtree(local_path)
            
        print(f"⬇ Downloading {hub_name} to {local_path}...")
        try:
            subprocess.run(
                [
                    "huggingface-cli",
                    "download",
                    hub_name,
                    "--local-dir",
                    local_path,
                    "--local-dir-use-symlinks",
                    "False",
                ],
                check=True,
            )
            print(f"✓ Successfully downloaded {local_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {local_name}: {e}")
            raise

    # 处理 README 中的特殊下载命令 (model.safetensors 从 PR)
    base_path = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
    pr_file_path = os.path.join(base_path, "model.safetensors")
    
    # 额外检查文件大小 (简单验证是否为 LFS 指针)
    needs_download = True
    if os.path.exists(pr_file_path):
        size = os.path.getsize(pr_file_path)
        if size > 1024 * 1024:  # 大于 1MB
            needs_download = False
            print("✓ chinese-wav2vec2-base model.safetensors verified.")
    
    if needs_download:
        print("⬇ Downloading special file for chinese-wav2vec2-base...")
        try:
            subprocess.run(
                [
                    "huggingface-cli",
                    "download",
                    "TencentGameMate/chinese-wav2vec2-base",
                    "model.safetensors",
                    "--revision",
                    "refs/pr/1",
                    "--local-dir",
                    base_path,
                    "--local-dir-use-symlinks",
                    "False",
                ],
                check=True,
            )
            print("✓ Successfully downloaded model.safetensors")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download model.safetensors: {e}")
            raise

    # 提交 Volume 变更
    MODELS_VOLUME.commit()
    print("✓ Model download complete and volume committed.")


# 4. 推理函数 (无 Gradio，纯 API)
# -----------------
@app.cls(
    gpu="A100-80GB",  # 升级到 80GB 显存; 也可以尝试 size="80GB" 但字符串更通用
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    timeout=7200,  # 增加到 2 小时
    scaledown_window=300,  # 5分钟后关闭空闲容器
)
class InfiniteTalkModel:
    @modal.enter()
    def load_model(self):
        """容器启动时加载模型"""
        import sys
        import os

        # 修复 transformers SDPA 兼容性问题
        # InfiniteTalk 的 wav2vec2 需要 output_attentions=True，这与 SDPA 不兼容
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        os.environ["ATTN_BACKEND"] = "eager"

        print(f"📁 Repo directory: {REPO_DIR}")
        os.chdir(REPO_DIR)
        sys.path.insert(0, REPO_DIR)

        # 验证模型文件存在
        model_paths = [
            os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
            os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
            os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
        ]
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path not found: {path}. Run 'modal run infinitetalk_demo.py' first to download models.")
        print("✓ All model paths verified")

        # 设置模型路径
        self.ckpt_dir = os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P")
        self.wav2vec_dir = os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base")
        self.infinitetalk_dir = os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors")

        print("✓ Model paths configured")

    @modal.method()
    def generate_video(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        resolution: str = "480",  # "480" 或 "720"
        sample_steps: int = 40,
        motion_frame: int = 9,
    ) -> bytes:
        """
        生成数字人视频

        参数:
        - image_bytes: 输入图片的字节流
        - audio_bytes: 输入音频的字节流 (支持 wav, mp3)
        - resolution: 分辨率 ("480" 或 "720")
        - sample_steps: 采样步数 (默认 40)
        - motion_frame: 运动帧数 (默认 9)

        返回:
        - 生成的视频字节流 (MP4)
        """
        import subprocess
        import json
        import tempfile
        import os
        from pathlib import Path

        print(f"🎬 Starting video generation...")
        print(f"   Resolution: {resolution}P, Steps: {sample_steps}")

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 保存输入文件
            image_path = temp_path / "input_image.png"
            audio_path = temp_path / "input_audio.wav"
            image_path.write_bytes(image_bytes)
            audio_path.write_bytes(audio_bytes)

            # 创建输入 JSON (按 InfiniteTalk 期望的格式)
            # 参考: examples/single_example_image.json
            input_data = {
                "prompt": "A person is speaking naturally with clear lip movements and natural expressions.",
                "cond_video": str(image_path),  # 图片路径
                "cond_audio": {
                    "person1": str(audio_path)  # 音频路径
                }
            }
            json_path = temp_path / "input.json"
            with open(json_path, "w") as f:
                json.dump(input_data, f)

            # 输出路径前缀 (不创建文件夹，因为脚本会把这个当前缀并自动添加 .mp4 后缀)
            output_prefix = temp_path / "output"

            # 构建推理命令
            cmd = [
                "python",
                "generate_infinitetalk.py",
                "--ckpt_dir", self.ckpt_dir,
                "--wav2vec_dir", self.wav2vec_dir,
                "--infinitetalk_dir", self.infinitetalk_dir,
                "--input_json", str(json_path),
                "--size", f"infinitetalk-{resolution}",
                "--sample_steps", str(sample_steps),
                "--mode", "streaming",
                "--motion_frame", str(motion_frame),
                "--num_persistent_param_in_dit", "0",  # 低显存模式
                "--save_file", str(output_prefix),
            ]

            print(f"📝 Running command: {' '.join(cmd)}")

            # 执行推理 (不捕获输出，直接打印到标准输出以便实时查看进度)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=REPO_DIR,
                bufsize=1,
                universal_newlines=True,
            )

            # 实时打印输出
            logs = []
            for line in process.stdout:
                print(line, end="")
                logs.append(line)
            
            return_code = process.wait()

            if return_code != 0:
                stderr_output = "".join(logs)
                print(f"❌ Error: Process exited with code {return_code}")
                # 尝试从日志中提取更有用的错误信息
                raise RuntimeError(f"Video generation failed with code {return_code}. Check logs above.")

            print(f"✓ Generation complete")

            # 查找输出视频
            # 脚本生成的视频应该是 output.mp4
            output_video = temp_path / "output.mp4"
            
            if not output_video.exists():
                # 尝试查找任何 mp4
                video_files = list(temp_path.glob("*.mp4"))
                if not video_files:
                    raise FileNotFoundError(f"No output video found at {output_video} or anywhere in temp dir")
                output_video = video_files[0]
            print(f"📹 Output video: {output_video}")

            return output_video.read_bytes()


# 5. Gradio Web 应用 (可选)
# -----------------
@app.cls(
    gpu="A100-40GB",
    volumes={WEIGHTS_DIR: MODELS_VOLUME},
    timeout=1800,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=1)
class GradioApp:
    @modal.enter()
    def build_app(self):
        import sys
        import os
        import gradio as gr

        print(f"📁 Setting up Gradio app...")
        os.chdir(REPO_DIR)
        sys.path.insert(0, REPO_DIR)

        # 验证模型文件存在
        model_paths = [
            os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
            os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
            os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
        ]
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path not found: {path}")
        print("✓ All model paths verified")

        try:
            # 导入 InfiniteTalk 的 app.py
            from app import parse_args, build_demo

            print("🔧 Patching sys.argv for argument parsing...")
            sys.argv = [
                "app.py",
                "--ckpt_dir", os.path.join(WEIGHTS_DIR, "Wan2.1-I2V-14B-480P"),
                "--wav2vec_dir", os.path.join(WEIGHTS_DIR, "chinese-wav2vec2-base"),
                "--infinitetalk_dir", os.path.join(WEIGHTS_DIR, "InfiniteTalk", "single", "infinitetalk.safetensors"),
                "--num_persistent_param_in_dit", "0",
                "--motion_frame", "9",
            ]

            print(f"📝 Parsing arguments...")
            args = parse_args()

            print("🏗️  Building Gradio demo (this may take a few minutes)...")
            demo = build_demo(args)
            print("✓ Gradio demo built successfully")

            self.demo = demo

        except Exception as e:
            print(f"✗ Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    @modal.asgi_app()
    def serve(self):
        """提供 Gradio ASGI 应用"""
        return self.demo


# 6. 本地入口点 (命令行测试)
# ---------------------
@app.local_entrypoint()
def main(
    image_path: str = "",
    audio_path: str = "",
    output_path: str = "/tmp/infinitetalk_output.mp4",
    resolution: str = "480",
    sample_steps: int = 40,  # 新增控制步数参数
    download_only: bool = False,
):
    """
    InfiniteTalk 命令行入口

    用法:
      # 快速测试 (低质量)
      modal run infinitetalk_demo.py --image-path ./face.png --audio-path ./speech.wav --sample-steps 10
    """
    from pathlib import Path

    # 仅下载模型
    if download_only or (not image_path and not audio_path):
        print("📦 Downloading models...")
        download_models.remote()
        print("\n✅ Models downloaded successfully!")
        print("\n🚀 Next steps:")
        print("   modal run infinitetalk_demo.py --image-path ./face.png --audio-path ./speech.wav")
        return

    # 验证输入文件
    image_file = Path(image_path)
    audio_file = Path(audio_path)

    if not image_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return

    if not audio_file.exists():
        print(f"❌ Error: Audio file not found: {audio_path}")
        return

    print(f"🖼️  Image: {image_path}")
    print(f"🎵 Audio: {audio_path}")
    print(f"📐 Resolution: {resolution}P")
    print(f"⚡ Steps: {sample_steps}")

    # 读取输入文件
    print("\n📤 Reading input files...")
    image_bytes = image_file.read_bytes()
    audio_bytes = audio_file.read_bytes()
    print(f"   Image size: {len(image_bytes) / 1024:.1f} KB")
    print(f"   Audio size: {len(audio_bytes) / 1024:.1f} KB")

    # 调用远程推理
    print("\n🚀 Starting video generation (this may take 5-15 minutes)...")
    model = InfiniteTalkModel()
    video_bytes = model.generate_video.remote(
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        resolution=resolution,
        sample_steps=sample_steps,
    )

    # 保存输出
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(video_bytes)

    print(f"\n✅ Video generated successfully!")
    print(f"📹 Output: {output_path}")
    print(f"💾 Size: {len(video_bytes) / 1024 / 1024:.2f} MB")
#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Seochan99. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SD3.5 LoRA Inference Script

Generate images using trained LoRA adapters with Stable Diffusion 3.5.
Supports single/batch generation, various sampling parameters, and output formats.

Author: Seochan99
"""

import argparse
import os
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusion3Pipeline
import gc


def parse_args():
    """Parse command line arguments for SD3.5 LoRA inference."""
    parser = argparse.ArgumentParser(
        description="Generate images using SD3.5 with trained LoRA adapters"
    )

    # ═══════════════════════════════════════════════════════════
    # Model and LoRA Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--model_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Path to base SD3.5 model or HuggingFace model ID",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA weights directory",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scaling factor (0.0 = no effect, 1.0 = full effect)",
    )

    # ═══════════════════════════════════════════════════════════
    # Generation Parameters
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted, bad anatomy",
        help="Negative prompt to avoid unwanted elements",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of denoising steps (more steps = better quality, slower)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (higher = more prompt adherence)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation",
    )

    # ═══════════════════════════════════════════════════════════
    # Output Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="png",
        choices=["png", "jpg", "webp"],
        help="Output image format",
    )
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="Save prompt information in filename",
    )

    # ═══════════════════════════════════════════════════════════
    # System Configuration
    # ═══════════════════════════════════════════════════════════
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model precision (auto recommended)",
    )
    parser.add_argument(
        "--enable_memory_efficient_attention",
        action="store_true",
        help="Enable memory efficient attention for lower VRAM usage",
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Offload models to CPU when not in use (saves VRAM)",
    )

    return parser.parse_args()


def setup_device_and_dtype(args):
    """Determine optimal device and dtype configuration."""
    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Dtype selection
    if args.dtype == "auto":
        if device == "cuda":
            # Use bfloat16 for modern GPUs, float16 for older ones
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere and newer
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif device == "mps":
            dtype = torch.float16  # MPS doesn't support bfloat16
        else:
            dtype = torch.float32  # CPU
    else:
        dtype = getattr(torch, args.dtype)

    print(f"🔧 Using device: {device}, dtype: {dtype}")
    return device, dtype


def load_pipeline(args, device, dtype):
    """Load and configure the SD3.5 pipeline with LoRA weights."""
    print("📦 Loading Stable Diffusion 3.5 pipeline...")

    # Load base pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None if not args.enable_cpu_offload else "auto",
    )

    # Apply memory optimizations
    if args.enable_memory_efficient_attention:
        pipeline.enable_attention_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        print("✅ Memory efficient attention enabled")

    if args.enable_cpu_offload:
        pipeline.enable_model_cpu_offload()
        print("✅ CPU offloading enabled")
    else:
        pipeline = pipeline.to(device)

    # Load LoRA weights
    print(f"🎯 Loading LoRA weights from: {args.lora_path}")
    try:
        pipeline.load_lora_weights(args.lora_path)
        print(f"✅ LoRA weights loaded successfully (scale: {args.lora_scale})")
    except Exception as e:
        print(f"❌ Failed to load LoRA weights: {e}")
        print("💡 Make sure the LoRA path contains pytorch_lora_weights.safetensors")
        raise

    return pipeline


def generate_images(pipeline, args):
    """Generate images using the loaded pipeline and LoRA weights."""
    print(f"🎨 Generating {args.num_images} image(s)...")
    print(f"📝 Prompt: {args.prompt}")

    # Setup generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
        print(f"🎲 Using seed: {args.seed}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    generated_images = []

    for i in range(args.num_images):
        print(f"🖼️ Generating image {i+1}/{args.num_images}...")

        try:
            # Generate image
            with torch.no_grad():
                output = pipeline(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    cross_attention_kwargs={"scale": args.lora_scale},
                )

                image = output.images[0]
                generated_images.append(image)

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if args.save_prompt:
                    # Create safe filename from prompt
                    safe_prompt = "".join(
                        c
                        for c in args.prompt[:50]
                        if c.isalnum() or c in (" ", "-", "_")
                    ).rstrip()
                    safe_prompt = safe_prompt.replace(" ", "_")
                    filename = (
                        f"{timestamp}_{i+1:03d}_{safe_prompt}.{args.output_format}"
                    )
                else:
                    filename = f"{timestamp}_{i+1:03d}.{args.output_format}"

                filepath = os.path.join(args.output_dir, filename)

                # Save with appropriate quality settings
                if args.output_format == "jpg":
                    image.save(filepath, "JPEG", quality=95, optimize=True)
                elif args.output_format == "webp":
                    image.save(filepath, "WEBP", quality=90, method=6)
                else:  # png
                    image.save(filepath, "PNG", optimize=True)

                print(f"✅ Saved: {filepath}")

        except Exception as e:
            print(f"❌ Failed to generate image {i+1}: {e}")
            continue

        # Clear cache between generations to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return generated_images


def main():
    """Main inference function."""
    args = parse_args()

    print("🚀 SD3.5 LoRA Inference")
    print("=" * 50)

    # Setup device and precision
    device, dtype = setup_device_and_dtype(args)

    # Load pipeline with LoRA
    pipeline = load_pipeline(args, device, dtype)

    # Generate images
    images = generate_images(pipeline, args)

    print("\n🎉 Generation complete!")
    print(f"📂 Generated {len(images)} image(s) in: {args.output_dir}")

    # Print generation summary
    print("\n📋 Generation Summary:")
    print(f"  Model: {args.model_path}")
    print(f"  LoRA: {args.lora_path} (scale: {args.lora_scale})")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Guidance: {args.guidance_scale}")
    print(f"  Seed: {args.seed}")


if __name__ == "__main__":
    main()

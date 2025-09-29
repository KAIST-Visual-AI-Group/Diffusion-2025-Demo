# Project Structure

This document outlines the complete structure of the SD3.5 LoRA Training project.

```
stable-diffusion-3.5-text2image-lora/
├── 📄 README.md                          # Main documentation
├── 📄 LICENSE                            # Apache 2.0 license
├── 📄 requirements.txt                   # Python dependencies
├── 📄 STRUCTURE.md                       # This file
│
├── 🐍 train_text_to_image_lora_sd35.py  # Main training script
├── 🎨 inference.py                       # Image generation script
│
├── 📁 scripts/                           # Training and setup scripts
│   ├── 🔧 setup.sh                      # Environment setup script
│   ├── 🚀 train_basic.sh                # Basic training script
│   ├── 🔥 train_advanced.sh             # Advanced training script
│   └── 🎨 inference.sh                  # Easy inference script
│
├── 📁 examples/                          # Example data and templates
│   └── 📁 dataset/                      # Example dataset structure
│       ├── 📄 README.md                 # Dataset documentation
│       ├── 📄 metadata.jsonl            # Example metadata file
│       └── 📁 images/                   # Directory for training images
│           └── 📄 .gitkeep              # Placeholder for git
│
└── 📁 outputs/                          # Training outputs (created during training)
    ├── 📁 sd35-lora-basic/             # Basic training results
    ├── 📁 sd35-lora-advanced/          # Advanced training results
    └── 📁 validation_images/            # Generated validation images
```

## 📁 Directory Descriptions

### Root Files

-   **README.md**: Complete project documentation with setup and usage instructions
-   **LICENSE**: Apache 2.0 open source license
-   **requirements.txt**: All Python package dependencies
-   **train_text_to_image_lora_sd35.py**: The main training script with full SD3.5 LoRA implementation
-   **inference.py**: Comprehensive image generation script with trained LoRA adapters

### Scripts Directory (`scripts/`)

Contains ready-to-use bash scripts for different scenarios:

-   **setup.sh**: Automated environment setup (dependencies, accelerate config, directories)
-   **train_basic.sh**: Simple training script with sensible defaults
-   **train_advanced.sh**: Full-featured training with all optimizations enabled
-   **inference.sh**: Easy-to-use script for generating images with trained LoRA models

### Examples Directory (`examples/`)

Provides templates and examples for users:

-   **dataset/**: Example dataset structure showing the required format
-   **metadata.jsonl**: Sample metadata file with diverse image descriptions
-   **images/**: Placeholder directory where users add their training images

### Outputs Directory (`outputs/`)

Created automatically during training, contains:

-   **Model checkpoints**: Saved at regular intervals
-   **LoRA weights**: Final trained adapters
-   **Validation images**: Generated during training for monitoring
-   **Training logs**: TensorBoard logs and metrics

## 🚀 Quick Navigation

| Want to...                        | Go to...                           |
| --------------------------------- | ---------------------------------- |
| **Understand the project**        | `README.md`                        |
| **Start training immediately**    | `scripts/train_basic.sh`           |
| **See example dataset format**    | `examples/dataset/`                |
| **Customize training parameters** | `train_text_to_image_lora_sd35.py` |
| **Set up environment**            | `scripts/setup.sh`                 |
| **Advanced training features**    | `scripts/train_advanced.sh`        |
| **Generate images with LoRA**     | `scripts/inference.sh`             |
| **Custom inference parameters**   | `inference.py`                     |

## 📋 Getting Started Workflow

1. **Setup**: Run `bash scripts/setup.sh`
2. **Prepare Data**: Add images to `examples/dataset/images/`
3. **Train**: Run `bash scripts/train_basic.sh`
4. **Generate**: Run `bash scripts/inference.sh`
5. **Monitor**: Check `outputs/` for results and `generated_images/` for outputs
6. **Iterate**: Adjust parameters and re-train as needed

## 🔄 File Creation During Usage

When you run the training:

```
outputs/
├── sd35-lora-{config}/
│   ├── pytorch_lora_weights.safetensors   # Trained LoRA weights
│   ├── adapter_config.json                # LoRA configuration
│   ├── logs/                              # TensorBoard logs
│   ├── validation_images/                 # Generated validation images
│   └── checkpoint-{step}/                 # Training checkpoints
```

This structure ensures easy navigation, clear organization, and smooth user experience from setup to training completion.

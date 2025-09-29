#!/bin/bash

# Basic SD3 LoRA Training Script
# This script provides a simple starting point for training LoRA adapters

echo "🚀 Starting SD3 LoRA Training (Basic Configuration)"
echo "=================================================="

# Check if accelerate is configured
if ! accelerate env >/dev/null 2>&1; then
    echo "⚠️ Accelerate not configured. Running accelerate config..."
    accelerate config
fi

# Set default values (can be overridden by environment variables)
MODEL_NAME=${MODEL_NAME:-"stabilityai/stable-diffusion-3-medium-diffusers"}
DATASET_DIR=${DATASET_DIR:-"./examples/dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/sd3-lora-basic"}
RESOLUTION=${RESOLUTION:-512}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EPOCHS=${EPOCHS:-10}
RANK=${RANK:-64}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
GPU_IDS=${GPU_IDS:-0}

echo "📋 Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: $RESOLUTION"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  LoRA Rank: $RANK"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
accelerate launch --gpu_ids "$GPU_IDS" train_text_to_image_lora_sd35.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --dataset_name "linoyts/rubber_ducks" \
  --output_dir "$OUTPUT_DIR" \
  --resolution $RESOLUTION \
  --train_batch_size $BATCH_SIZE \
  --num_train_epochs $EPOCHS \
  --rank $RANK \
  --learning_rate $LEARNING_RATE \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --validation_prompt "photo of a TOK duck, sitting by a chair in Paris, enjoying the autumn leaves" \
  --validation_epochs 2 \
  --num_validation_images 4 \
  --checkpointing_steps 500 \
  --seed 42 \
  --report_to tensorboard

echo ""
echo "✅ Training completed! Check your results in: $OUTPUT_DIR"
echo "📊 View training logs with: tensorboard --logdir $OUTPUT_DIR/logs" 
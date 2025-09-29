#!/bin/bash

# Advanced SD3.5 LoRA Training Script
# This script includes all advanced features and optimizations

echo "🔥 Starting SD3.5 LoRA Training (Advanced Configuration)"
echo "======================================================"

# Check if accelerate is configured
if ! accelerate env >/dev/null 2>&1; then
    echo "⚠️ Accelerate not configured. Running accelerate config..."
    accelerate config
fi

# Advanced configuration with all features
MODEL_NAME=${MODEL_NAME:-"stabilityai/stable-diffusion-3.5-medium"}
DATASET_DIR=${DATASET_DIR:-"./examples/dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/sd35-lora-advanced"}
RESOLUTION=${RESOLUTION:-1024}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EPOCHS=${EPOCHS:-20}
RANK=${RANK:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
TEXT_ENCODER_LR=${TEXT_ENCODER_LR:-5e-5}
VALIDATION_PROMPT=${VALIDATION_PROMPT:-"a cyberpunk cityscape at night, neon lights, futuristic architecture"}

echo "📋 Advanced Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: $RESOLUTION"
echo "  Batch Size: $BATCH_SIZE (Grad Accum: $GRAD_ACCUM)"
echo "  Epochs: $EPOCHS"
echo "  LoRA Rank: $RANK"
echo "  LR: $LEARNING_RATE (Text Encoder: $TEXT_ENCODER_LR)"
echo "  Validation: $VALIDATION_PROMPT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Advanced training with all features
accelerate launch train_text_to_image_lora_sd35.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --resolution $RESOLUTION \
  --train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_train_epochs $EPOCHS \
  --rank $RANK \
  --learning_rate $LEARNING_RATE \
  --text_encoder_lr $TEXT_ENCODER_LR \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --max_grad_norm 1.0 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --train_text_encoder \
  --validation_prompt "$VALIDATION_PROMPT" \
  --validation_epochs 5 \
  --num_validation_images 4 \
  --checkpointing_steps 1000 \
  --dataloader_num_workers 4 \
  --weighting_scheme logit_normal \
  --logit_mean 0.0 \
  --logit_std 1.0 \
  --precondition_outputs 1 \
  --seed 42 \
  --report_to tensorboard

echo ""
echo "🎉 Advanced training completed!"
echo "📂 Results saved in: $OUTPUT_DIR"
echo "📊 View training logs: tensorboard --logdir $OUTPUT_DIR/logs"
echo "🖼️ Validation images: $OUTPUT_DIR/validation_images/" 
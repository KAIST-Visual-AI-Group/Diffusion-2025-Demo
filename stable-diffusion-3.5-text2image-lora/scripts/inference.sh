#!/bin/bash

# SD3.5 LoRA Inference Script
# Easy image generation with trained LoRA adapters

echo "🎨 SD3.5 LoRA Image Generation"
echo "=============================="

# Check if inference.py exists
if [ ! -f "inference.py" ]; then
    echo "❌ inference.py not found in current directory"
    echo "💡 Please run this script from the project root"
    exit 1
fi

# Default values (can be overridden by environment variables)
LORA_PATH=${LORA_PATH:-"./outputs/sd35-lora-basic"}
PROMPT=${PROMPT:-"a beautiful landscape with mountains and a lake, professional photography"}
NUM_IMAGES=${NUM_IMAGES:-4}
OUTPUT_DIR=${OUTPUT_DIR:-"./generated_images"}
STEPS=${STEPS:-28}
GUIDANCE=${GUIDANCE:-7.0}
SEED=${SEED:-42}
RESOLUTION=${RESOLUTION:-1024}

echo "📋 Generation Configuration:"
echo "  LoRA Path: $LORA_PATH"
echo "  Prompt: $PROMPT"
echo "  Images: $NUM_IMAGES"
echo "  Output: $OUTPUT_DIR"
echo "  Steps: $STEPS"
echo "  Guidance: $GUIDANCE"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Seed: $SEED"
echo ""

# Check if LoRA weights exist
if [ ! -d "$LORA_PATH" ]; then
    echo "❌ LoRA weights not found at: $LORA_PATH"
    echo "💡 Please train a model first or specify correct path:"
    echo "   LORA_PATH=\"./path/to/your/lora\" bash scripts/inference.sh"
    exit 1
fi

# Check for LoRA weights file
if [ ! -f "$LORA_PATH/pytorch_lora_weights.safetensors" ]; then
    echo "❌ LoRA weights file not found: $LORA_PATH/pytorch_lora_weights.safetensors"
    echo "💡 Make sure training completed successfully"
    exit 1
fi

echo "🚀 Starting image generation..."

# Run inference
python inference.py \
    --lora_path "$LORA_PATH" \
    --prompt "$PROMPT" \
    --num_images $NUM_IMAGES \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_steps $STEPS \
    --guidance_scale $GUIDANCE \
    --seed $SEED \
    --height $RESOLUTION \
    --width $RESOLUTION \
    --save_prompt \
    --enable_memory_efficient_attention

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 Generation completed successfully!"
    echo "📂 Check your images in: $OUTPUT_DIR"
    echo ""
    echo "💡 To generate different images:"
    echo "   PROMPT=\"your custom prompt\" bash scripts/inference.sh"
    echo "   NUM_IMAGES=8 STEPS=50 bash scripts/inference.sh"
else
    echo ""
    echo "❌ Generation failed with exit code: $exit_code"
    echo "💡 Check the error messages above for troubleshooting"
fi 
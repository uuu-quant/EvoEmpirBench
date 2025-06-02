#!/bin/bash

# Configuration
MODEL=${1:-"gpt-4"}
DIFFICULTIES=("easy" "medium" "hard")
LEVELS_COUNT=${2:-3}
MEMORY_DIR="./data/memory/${MODEL}"
API_KEY=""  # Set your API key in environment variable
BASE_URL=""  # Set your API URL in environment variable

# Set API keys from environment if available
if [ -n "$OPENAI_API_KEY" ]; then
    API_KEY="$OPENAI_API_KEY"
fi

if [ -n "$OPENAI_API_BASE" ]; then
    BASE_URL="$OPENAI_API_BASE"
fi

# Create directories
mkdir -p "$MEMORY_DIR"

# Function to train on a single difficulty
train_difficulty() {
    DIFFICULTY=$1
    echo "Training on $DIFFICULTY with model $MODEL"
    
    # Run training
    python -m nips2025.game2.scripts.train_agent \
        --model "$MODEL" \
        --difficulty "$DIFFICULTY" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --levels_count "$LEVELS_COUNT" \
        --memory_dir "$MEMORY_DIR" \
        --with_memory \
        --max_episode_steps 100
}

# Train on each difficulty in sequence
for DIFFICULTY in "${DIFFICULTIES[@]}"; do
    train_difficulty "$DIFFICULTY"
done

# Optimize knowledge after all training
echo "Optimizing truth knowledge..."
python -m nips2025.game2.scripts.optimize_truth \
    --model "$MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --memory_dir "$MEMORY_DIR" \
    --force

echo "Training completed. Memory stored in $MEMORY_DIR" 
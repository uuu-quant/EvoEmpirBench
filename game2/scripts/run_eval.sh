#!/bin/bash

# Configuration
MODEL=${1:-"gpt-4"}
DIFFICULTIES=("easy" "medium" "hard")
LEVELS_COUNT=${2:-5}
RESULTS_DIR="./results/${MODEL}"
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
mkdir -p "$RESULTS_DIR"

# Function to evaluate a single difficulty
evaluate_difficulty() {
    DIFFICULTY=$1
    echo "Evaluating $DIFFICULTY with model $MODEL"
    
    # First run - Without truth knowledge
    echo "Running evaluation without truth knowledge..."
    python -m nips2025.game2.scripts.evaluate_agent \
        --model "$MODEL" \
        --difficulty "$DIFFICULTY" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --levels_count "$LEVELS_COUNT" \
        --results_dir "$RESULTS_DIR" \
        --no_truth_knowledge \
        --run_id "${DIFFICULTY}_no_truth"
    
    # Second run - With truth knowledge
    echo "Running evaluation with truth knowledge..."
    python -m nips2025.game2.scripts.evaluate_agent \
        --model "$MODEL" \
        --difficulty "$DIFFICULTY" \
        --api_key "$API_KEY" \
        --base_url "$BASE_URL" \
        --levels_count "$LEVELS_COUNT" \
        --results_dir "$RESULTS_DIR" \
        --truth_knowledge_path "$RESULTS_DIR/truth_knowledge.json" \
        --run_id "${DIFFICULTY}_with_truth"
}

# Run evaluations for each difficulty
for DIFFICULTY in "${DIFFICULTIES[@]}"; do
    evaluate_difficulty "$DIFFICULTY"
done

# Run analysis
echo "Running analysis..."
python -m nips2025.game2.scripts.analyze_results \
    --model "$MODEL" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/analysis"

echo "Evaluation completed. Results in $RESULTS_DIR" 
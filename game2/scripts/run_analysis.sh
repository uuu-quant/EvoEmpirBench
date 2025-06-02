#!/bin/bash

# Configuration
MODEL=${1:-"gpt-4"}
RESULTS_DIR="./results/${MODEL}"
OUTPUT_DIR="$RESULTS_DIR/analysis"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
echo "Running analysis for model $MODEL"
python -m nips2025.game2.scripts.analyze_results \
    --model "$MODEL" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

# Generate charts
echo "Generating charts..."
python -m nips2025.game2.scripts.generate_charts \
    --model "$MODEL" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

# Generate report
echo "Generating report..."
python -m nips2025.game2.scripts.generate_report \
    --model "$MODEL" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis completed. Results in $OUTPUT_DIR" 
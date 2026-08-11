#!/bin/bash
# 运行Game1结果分析脚本

# 定义默认参数
MODEL=""
RESULTS_DIR="../../results"
OUTPUT_DIR=""
COMPARE_MODELS=""

# 处理命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --compare_models)
      COMPARE_MODELS="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 构建命令行参数
ARGS=()

if [ -n "$MODEL" ]; then
  ARGS+=("--model" "$MODEL")
fi

ARGS+=("--results_dir" "$RESULTS_DIR")

if [ -n "$OUTPUT_DIR" ]; then
  ARGS+=("--output_dir" "$OUTPUT_DIR")
else
  # 默认输出目录为结果目录下的analysis子目录
  OUTPUT_DIR="${RESULTS_DIR}/analysis"
  ARGS+=("--output_dir" "$OUTPUT_DIR")
fi

if [ -n "$COMPARE_MODELS" ]; then
  ARGS+=("--compare_models" "$COMPARE_MODELS")
fi

# 打印执行信息
echo "开始分析游戏1结果..."
if [ -n "$MODEL" ]; then
  echo "模型: $MODEL"
elif [ -n "$COMPARE_MODELS" ]; then
  echo "比较模型: $COMPARE_MODELS"
fi
echo "结果目录: $RESULTS_DIR"
echo "输出目录: $OUTPUT_DIR"

# 确保当前目录是脚本所在目录
cd "$(dirname "$0")"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行Python脚本
python -m analyze_results "${ARGS[@]}"

echo "分析完成! 结果保存在 $OUTPUT_DIR" 
#!/bin/bash
# 运行Game1训练脚本

# 定义默认参数
MODEL="gpt-4"
MODE="Level 2"
MAPS_COUNT=5
RESULTS_DIR="../../results"
MAPS_DIR="../maps"
API_KEY=""
BASE_URL=""
MEMORY_DIR=""
TRAINING_ITERATIONS=3

# 处理命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL="$2"
      shift
      shift
      ;;
    --mode)
      MODE="$2"
      shift
      shift
      ;;
    --maps_count)
      MAPS_COUNT="$2"
      shift
      shift
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift
      shift
      ;;
    --maps_dir)
      MAPS_DIR="$2"
      shift
      shift
      ;;
    --api_key)
      API_KEY="$2"
      shift
      shift
      ;;
    --base_url)
      BASE_URL="$2"
      shift
      shift
      ;;
    --memory_dir)
      MEMORY_DIR="$2"
      shift
      shift
      ;;
    --iterations)
      TRAINING_ITERATIONS="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 检查环境变量
if [ -z "$API_KEY" ] && [ -n "$OPENAI_API_KEY" ]; then
  API_KEY="$OPENAI_API_KEY"
fi

if [ -z "$BASE_URL" ] && [ -n "$OPENAI_API_BASE" ]; then
  BASE_URL="$OPENAI_API_BASE"
fi

# 如果内存目录未指定，使用结果目录下的memory子目录
if [ -z "$MEMORY_DIR" ]; then
  MEMORY_DIR="${RESULTS_DIR}/memory"
fi

# 构建命令行参数
ARGS=()
ARGS+=("--model" "$MODEL")
ARGS+=("--mode" "$MODE")
ARGS+=("--maps_count" "$MAPS_COUNT")
ARGS+=("--results_dir" "$RESULTS_DIR")
ARGS+=("--maps_dir" "$MAPS_DIR")
ARGS+=("--memory_dir" "$MEMORY_DIR")
ARGS+=("--iterations" "$TRAINING_ITERATIONS")

if [ -n "$API_KEY" ]; then
  ARGS+=("--api_key" "$API_KEY")
fi

if [ -n "$BASE_URL" ]; then
  ARGS+=("--base_url" "$BASE_URL")
fi

# 打印执行信息
echo "开始训练游戏1代理..."
echo "模型: $MODEL"
echo "模式: $MODE"
echo "地图数量: $MAPS_COUNT"
echo "训练迭代次数: $TRAINING_ITERATIONS"
echo "记忆目录: $MEMORY_DIR"

# 确保当前目录是脚本所在目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$MEMORY_DIR"

# 运行Python脚本
echo "运行训练脚本..."
python -m train_agent "${ARGS[@]}"

echo "训练完成!"

# 评估训练后的效果
echo "评估训练后的效果..."
./run_eval.sh --model "$MODEL" --mode "$MODE" --results_dir "$RESULTS_DIR" --maps_dir "$MAPS_DIR" --with_truth --truth_path "${MEMORY_DIR}/truth_knowledge.json"

echo "训练和评估完成!" 
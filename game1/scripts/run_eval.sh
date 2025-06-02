#!/bin/bash
# 运行Game1评估脚本

# 定义默认参数
MODEL="gpt-4"
MODE="Level 2"
MAPS_COUNT=10
RESULTS_DIR="../../results"
MAPS_DIR="../maps"
API_KEY=""
BASE_URL=""
WITH_TRUTH=false
TRUTH_PATH=""
FULL_VISION=false

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
    --maps)
      MAPS="$2"
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
    --with_truth)
      WITH_TRUTH=true
      shift
      ;;
    --truth_path)
      TRUTH_PATH="$2"
      shift
      shift
      ;;
    --full_vision)
      FULL_VISION=true
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

# 构建命令行参数
ARGS=()
ARGS+=("--model" "$MODEL")
ARGS+=("--mode" "$MODE")

if [ -n "$MAPS" ]; then
  ARGS+=("--maps" "$MAPS")
else
  ARGS+=("--maps_count" "$MAPS_COUNT")
fi

ARGS+=("--results_dir" "$RESULTS_DIR")
ARGS+=("--maps_dir" "$MAPS_DIR")

if [ -n "$API_KEY" ]; then
  ARGS+=("--api_key" "$API_KEY")
fi

if [ -n "$BASE_URL" ]; then
  ARGS+=("--base_url" "$BASE_URL")
fi

if [ "$WITH_TRUTH" = true ]; then
  ARGS+=("--with_truth")
  if [ -n "$TRUTH_PATH" ]; then
    ARGS+=("--truth_path" "$TRUTH_PATH")
  fi
fi

if [ "$FULL_VISION" = true ]; then
  ARGS+=("--full_vision")
fi

# 添加随机种子以确保可重复性
ARGS+=("--random_seed" "42")

# 打印执行信息
echo "开始评估游戏1代理性能..."
echo "模型: $MODEL"
echo "模式: $MODE"
echo "地图数量: $MAPS_COUNT"
echo "使用真理知识: $WITH_TRUTH"
echo "完全视野模式: $FULL_VISION"

# 确保当前目录是脚本所在目录
cd "$(dirname "$0")"

# 运行Python脚本
python -m evaluate_agent "${ARGS[@]}"

echo "评估完成!" 
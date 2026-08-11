#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
SCRIPT_PATH="${PROJECT_DIR}/scripts/evaluate_agent.py"

# 默认参数
API_TYPE="deepseek"  # 可选: deepseek 或 openai
MODEL="deepseek-reasoner"  # 根据API_TYPE确定默认值
API_KEY=""  # 需要在运行时提供或从环境变量获取
BASE_URL=""  # 仅当API_TYPE=openai时使用
NUM_MAPS=30
MAX_STEPS=100
GAME_MODE=""  # 留空表示评估所有模式
RESUME=true   # 默认启用续传模式，跳过已评估的内容
START_MAP_INDEX=0  # 默认从第一张地图开始评估

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --api_type)
      API_TYPE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --api_key)
      API_KEY="$2"
      shift 2
      ;;
    --base_url)
      BASE_URL="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --num_maps)
      NUM_MAPS="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --mode)
      GAME_MODE="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --start_map_index)
      START_MAP_INDEX="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  --api_type        API类型 (deepseek 或 openai，默认: deepseek)"
      echo "  --model           模型名称 (默认: deepseek-reasoner 或 gpt-4o，取决于API类型)"
      echo "  --api_key         API密钥"
      echo "  --base_url        API基础URL (仅当api_type=openai时使用)"
      echo "  --save_dir        结果保存目录 (默认: ./outputs/results/{model}/game1)"
      echo "  --num_maps        每个难度评估的地图数量"
      echo "  --max_steps       每张地图的最大步数"
      echo "  --mode            仅评估指定模式 (Level 1/Level 2/Level 3)"
      echo "  --resume          是否启用续传模式 (true/false，默认: true)"
      echo "  --start_map_index 从指定索引的地图开始评估 (0-29，默认: 0)"
      echo "  --help            显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 根据API类型设置默认模型
if [ "$API_TYPE" = "openai" ] && [ "$MODEL" = "deepseek-reasoner" ]; then
  MODEL="gpt-4o"
fi

# 设置保存目录（使用模型名称作为子目录）
if [ -z "$SAVE_DIR" ]; then
  SAVE_DIR="${PROJECT_DIR}/outputs/results/${MODEL}/game1"
fi

# 创建保存目录
mkdir -p "${SAVE_DIR}"

# 构建命令行参数
ARGS=""

# 添加API密钥参数（如果提供）
if [ -n "$API_KEY" ]; then
  ARGS="$ARGS --api_key $API_KEY"
fi

# 添加其他参数
ARGS="$ARGS --api_type $API_TYPE --model $MODEL"

# 添加API基础URL（如果API类型是openai且提供了URL）
if [ "$API_TYPE" = "openai" ] && [ -n "$BASE_URL" ]; then
  ARGS="$ARGS --base_url $BASE_URL"
fi

# 添加其他参数
ARGS="$ARGS --results_dir $SAVE_DIR --num_maps $NUM_MAPS --max_steps $MAX_STEPS"

# 添加续传模式
if [ "$RESUME" = "true" ]; then
  ARGS="$ARGS --resume"
fi

# 添加起始地图索引
if [ "$START_MAP_INDEX" -gt 0 ]; then
  ARGS="$ARGS --start_map_index $START_MAP_INDEX"
fi

# 添加模式参数（如果提供）
if [ -n "$GAME_MODE" ]; then
  ARGS="$ARGS --mode \"$GAME_MODE\""
fi

# 打印评估信息
echo "开始AI代理评估..."
echo "API类型: $API_TYPE"
echo "模型: $MODEL"
echo "结果保存目录: $SAVE_DIR"
echo "地图数量: $NUM_MAPS"
echo "最大步数: $MAX_STEPS"
echo "续传模式: $RESUME"
echo "起始地图索引: $START_MAP_INDEX"
if [ -n "$GAME_MODE" ]; then
  echo "评估模式: $GAME_MODE"
else
  echo "评估模式: 全部"
fi

# 运行评估
eval "python ${SCRIPT_PATH} ${ARGS} > \"${SAVE_DIR}/evaluation.log\" 2>&1"

echo "评估完成！"
echo "查看评估日志: ${SAVE_DIR}/evaluation.log"

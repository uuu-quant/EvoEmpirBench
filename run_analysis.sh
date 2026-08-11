#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
SCRIPT_PATH="${PROJECT_DIR}/scripts/analyze_results.py"

# 默认参数
MODELS=""
RESULTS_DIR="${PROJECT_DIR}/outputs/results"
GAME="both"
SUMMARY=false
NORMALIZE=false
BASE_MODEL="human"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)
      shift
      MODELS=""
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        MODELS="${MODELS} $1"
        shift
      done
      ;;
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --game)
      GAME="$2"
      shift 2
      ;;
    --summary)
      SUMMARY=true
      shift
      ;;
    --normalize)
      NORMALIZE=true
      shift
      ;;
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --help)
      echo "用法: $0 [选项]"
      echo "选项:"
      echo "  --models       要分析的模型列表，多个模型用空格分隔"
      echo "  --results_dir  结果保存目录 (默认: ./outputs/results)"
      echo "  --game         要分析的游戏 (game1, game2, both)"
      echo "  --summary      只生成摘要表格"
      echo "  --normalize    生成以指定模型为基准的标准化表格"
      echo "  --base_model   标准化的基准模型名称 (默认: human)"
      echo "  --help         显示此帮助信息"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      exit 1
      ;;
  esac
done

# 构建命令行参数
ARGS="--results_dir ${RESULTS_DIR} --game ${GAME}"

# 添加模型参数（如果提供）
if [ -n "$MODELS" ]; then
  ARGS="$ARGS --models ${MODELS}"
fi

# 添加摘要参数（如果启用）
if [ "$SUMMARY" = true ]; then
  ARGS="$ARGS --summary"
fi

# 添加标准化参数（如果启用）
if [ "$NORMALIZE" = true ]; then
  ARGS="$ARGS --normalize --base_model ${BASE_MODEL}"
fi

# 输出分析信息
echo "开始分析AI代理评估结果..."
echo "结果目录: $RESULTS_DIR"
echo "游戏类型: $GAME"
if [ -n "$MODELS" ]; then
  echo "分析模型: $MODELS"
else
  echo "分析所有可用模型"
fi
if [ "$SUMMARY" = true ]; then
  echo "生成模式: 仅摘要表格"
fi
if [ "$NORMALIZE" = true ]; then
  echo "生成模式: 标准化表格 (基准模型: ${BASE_MODEL})"
fi

# 运行分析脚本
python ${SCRIPT_PATH} ${ARGS}

echo "分析完成！" 

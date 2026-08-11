# EvoEmpirBench: Dynamic Spatial Reasoning Benchmark

## Abstract

Most existing spatial reasoning benchmarks focus on static or globally observable environments, failing to capture the challenges of long-horizon reasoning and memory utilization under partial observability and dynamic changes. We introduce two dynamic spatial benchmarks—locally observable maze navigation and match-2 elimination—that systematically evaluate models' abilities in spatial understanding and adaptive planning when local perception, environment feedback, and global objectives are tightly coupled. Each action triggers structural changes in the environment, requiring continuous update of cognition and strategy. We further propose a subjective experience-based memory mechanism for cross-task experience transfer and validation. Experiments show that our benchmarks reveal key limitations of mainstream models in dynamic spatial reasoning and long-term memory, providing a comprehensive platform for future methodological advances.

## Repository Structure

```
/
├── common/         # Common utilities and classes
├── game1/          # Maze Navigation game
│   ├── agent/      # Agent implementations
│   ├── game/       # Game core logic
│   ├── memory/     # Agent memory storage
│   └── scripts/    # Evaluation scripts
└── game2/          # Match-2 Puzzle game
```

## Game 1: Maze Navigation

### Game Overview

Maze Navigation is a grid-based environment where agents need to navigate through mazes, find goal positions, collect coins, and avoid monsters and obstacles. The game has three difficulty levels, introducing more complex mechanics such as monsters, destructible obstacles, and special items as difficulty increases.

### Features

- **Multiple Difficulty Levels**: From simple path planning to complex resource management and tactical decision-making
- **Partial Observability**: Agents can only see a limited area of the environment around them
- **Resource Management**: Collect items and coins, manage lives
- **Dynamic Environment**: Moving monsters and interactive obstacles
- **Learning Agents**: Support for GPT-based learning agents with memory and reflection mechanisms

### Installation and Setup

1. Ensure Python 3.8+ is installed
2. Install dependencies:
   ```
   pip install openai numpy matplotlib pandas
   ```
3. Set API key (for GPT integration):
   ```
   export OPENAI_API_KEY=your_api_key
   ```

### Evaluating Agents

Use the evaluation script to test agent performance:

```bash
cd /game1/scripts
./run_eval.sh --model gpt-4 --mode "Level 2" --maps_count 10
```

Optional parameters:
- `--model`: Specify the model to use (default: gpt-4)
- `--mode`: Game difficulty level (Level 1, Level 2, Level 3)
- `--maps_count`: Number of maps to evaluate
- `--with_truth`: Use truth knowledge
- `--full_vision`: Enable full vision mode (disables partial observability)

### Analyzing Results

Analyze agent performance after evaluation:

```bash
cd /game1/scripts
./run_analysis.sh --model gpt-4
```

Compare performance across multiple models:

```bash
./run_analysis.sh --compare_models "gpt-3.5-turbo,gpt-4,claude-3"
```

### Training Agents

Train agents with memory capabilities:

```bash
cd /game1/scripts
./run_training.sh --model gpt-4 --mode "Level 2" --iterations 3
```

## Game 2: Match-2 Elimination

A grid-based puzzle game where agents need to match and eliminate pairs of same-colored blocks, managing resources and planning moves strategically to achieve target goals within limited steps.

## Agent-ExpVer Framework

We present Agent-ExpVer, a three-agent framework for:
- Environment interaction
- Experience synthesis
- Adaptive truth management

This framework drives effective online learning and markedly improves agent reasoning and interactivity in dynamic spatial environments.

## Conclusion

We introduce EvoEmpirBench, a benchmark for spatial and high-level reasoning in dynamic, interactive environments, featuring Maze Navigation and Match-2 tasks. We also present Agent-ExpVer, a three-agent framework for environment interaction, experience synthesis, and adaptive truth management; experiments show it drives effective online learning and markedly improves agent reasoning and interactivity.

### Limitations
Performance remains tied to model capacity—smaller models lag, and even top systems fall short of human baselines.

### Future Work
We will boost Agent-ExpVer's adaptability (especially for lightweight models), expand EvoEmpirBench with tasks on temporal reasoning and multi-agent collaboration, and develop advanced mechanisms for truth induction and experience management.

## File Descriptions

### Common Modules
- `common/gpt_client.py` - Unified interface for interacting with LLM APIs
- `common/utils.py` - Common utility functions

### Game 1 Core Components
- `game1/game/environment.py` - Maze navigation environment implementation
- `game1/game/map_generator.py` - Maze map generator
- `game1/game/obstacles.py` - Obstacle and interactive object classes
- `game1/game/config.py` - Game configuration constants
- `game1/game/game_runner.py` - Game runner

### Game 1 Agents
- `game1/agent/learning_agent.py` - GPT-based learning agent
- `game1/agent/agent_interface.py` - Agent interface definitions
- `game1/agent/reflection_agent.py` - Agent with reflection capabilities
- `game1/agent/map_processor.py` - Map and state processing utilities

### Game 1 Evaluation Scripts
- `game1/scripts/evaluate_agent.py` - Agent evaluation script
- `game1/scripts/analyze_results.py` - Results analysis script
- `game1/scripts/run_eval.sh` - Shell script to run evaluations
- `game1/scripts/run_analysis.sh` - Shell script to run analysis
- `game1/scripts/run_training.sh` - Shell script to run training

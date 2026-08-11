# Reproducibility Notes

This cleaned copy is intended for source-code review and reproduction from code-generated or retained level assets.

## Included Inputs

- `data/levels/maze_eval`: default evaluation maps for the maze-navigation benchmark.
- `data/levels/maze_train`: maps used by the learning/training loop.
- `data/levels/match_game`: match-style game levels used by `src/match_game/general_game.py`.

These files are lightweight and useful for quickly running the code without regenerating every level.

Maze level scales follow the original code design: Level 1 is 7x7, Level 2 is 9x9, and Level 3 is 11x11.

## Generated Outputs

Runtime artifacts are written under `outputs/` and ignored by git:

- `outputs/results`: model outputs, logs, metrics, plots, and reports.
- `outputs/memory`: subjective memory and promoted truth knowledge.
- `outputs/agent_sessions`: session traces from learning runs.
- `outputs/memory_validation`, `outputs/truth_optimization`, `outputs/memory_promotion`: diagnostic logs.
- `outputs/collected_data`, `outputs/processed_data`: optional collected/processed datasets.

## Regenerating Maps

Maze evaluation maps:

```bash
python scripts/generate_maps.py --count 30 --save_dir data/levels/maze_eval
```

Training maps:

```bash
python scripts/generate_training_maps.py --count 10 --save_dir data/levels/maze_train
```

Match-style levels are regenerated automatically by `src/match_game/general_game.py` if a difficulty folder or level file is missing.

## Minimal Verification

Run this before sharing the code:

```bash
python scripts/check_project.py
```

Expected outcome:

- required source and level directories exist;
- retained level JSON files can be parsed;
- generated output folders are absent;
- no `sk-...` style API key pattern appears in source/docs.

## API-dependent Runs

LLM evaluation requires either OpenAI-compatible or DeepSeek credentials. Use `.env.example` as a template:

```bash
cp .env.example .env
```

Then set provider-specific environment variables locally. Do not commit `.env`.

## Evaluation Notes

For comparable results, use the retained level assets, the same model/API provider, the same prompt and decoding settings, the same map subset, and the same memory initialization policy.

EvoEmpirBench evaluates live LLM agents through API calls, so minor differences may occur across endpoint versions or repeated runs. When reporting new results, include the model name, provider, map split, number of maps, max-step budget, and whether Agent-ExpVer starts from an empty or existing memory directory.

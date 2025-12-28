# Repository Guidelines

## Project Structure & Module Organization
The core package lives under `gomoku/`, with gameplay, search, and training logic in files like `gomoku_env.py`, `mcts.py`, `zero_mcts.py`, `policy.py`, `player.py`, and `trainer.py`. The GUI entry points are `gomoku/run_gui.py` and `gomoku/gui.py` with usage notes in `gomoku/README_GUI.md`. Pretrained checkpoints are stored under `gomoku/models/` and `gomoku/continue_model/`. Evaluation scripts live in `gomoku/evaluation/`, and TensorBoard logs are typically written to `gomoku/runs/`. Repository metadata and dependencies are defined in `pyproject.toml` (with `uv.lock` tracking resolved versions).

## Build, Test, and Development Commands
- `python gomoku/run_gui.py` runs the local GUI application.
- `uvicorn gomoku.app:app --reload` starts the FastAPI server in `gomoku/app.py`.
- `python gomoku/trainer.py` runs training (requires PyTorch, Ray, and a suitable device).
- `python gomoku/evaluation/quick_battle_test.py` executes a lightweight evaluation script.
- Optional dependency setup: `uv sync` (if you use uv) or `python -m pip install -e .`.

## Coding Style & Naming Conventions
Follow standard Python conventions (PEP 8). Use 4-space indentation, `snake_case` for functions/variables/files, and `PascalCase` for classes. Keep module names short and descriptive (e.g., `zero_mcts.py`). Add type hints where practical in new or edited code.

## Testing Guidelines
There is no unified test runner configured. Use the evaluation scripts in `gomoku/evaluation/` as functional checks; most files are named with a `_test.py` suffix but are run directly (e.g., `python gomoku/evaluation/lightweight_eval.py`). When adding new evaluation utilities, follow this pattern and keep runtime reasonable.

## Commit & Pull Request Guidelines
Recent commits use a `type: description` format (e.g., `feature: gui`). Keep messages short and action-oriented. For PRs, include a concise description, a list of commands or scripts you ran (or “not run” with reasoning), and note any new or updated model checkpoints. Include screenshots for GUI changes and link relevant issues when applicable.

## Configuration & Assets
Model checkpoints are large; prefer referencing paths under `gomoku/models/` in scripts and avoid hardcoding absolute paths. If you add new assets, document the expected location and naming pattern in the relevant script or README.

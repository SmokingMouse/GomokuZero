---
title: GomokuZero
emoji: ♟️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---


# GomokuZero

GomokuZero 是一个基于 AlphaZero 思路的五子棋项目，包含 MCTS/Zero MCTS、训练与评估脚本、
本地 GUI，以及可用于 Web 前端的 FastAPI WebSocket 服务。

## 特性

- 标准 Gomoku 环境与规则实现（9x9 默认，可扩展）。
- 纯 MCTS 与 Zero MCTS（策略网络 + 价值网络）。
- 本地 GUI（Pygame）支持人机对战与 AI 对战。
- FastAPI WebSocket 后端，支持浏览器对战与验证集管理接口。
- 训练与自对弈管线（Ray + PyTorch）。
- Rust 版 MCTS 实验实现（`zero_mcts_rs/`）。

## 目录结构

- `gomoku/`：核心实现（环境、MCTS、策略网络、训练与评估）。
- `models/`：预训练权重与继续训练模型。
- `runs/`：TensorBoard 日志输出。
- `web/`：前端工程（Next.js 静态导出）。
- `zero_mcts_rs/`：Rust 实验版 MCTS。
- `WEB_DEPLOY.md`：Web 部署说明。

## 安装

Python 3.12+。

```bash
# 使用 uv
uv sync

# 或使用 pip
python -m pip install -e .
```

## 快速开始

本地 GUI（Pygame）：

```bash
python gomoku/gui.py
```

FastAPI 后端（WebSocket + 验证集管理）：

```bash
uvicorn gomoku.app:app --reload
```

常用环境变量：

- `GOMOKU_MODEL_PATH`：模型权重路径（默认 `models/gomoku_zero_9_lab_4/policy_step_30000.pth`）
- `GOMOKU_MCTS_ITERS`：MCTS 迭代次数（默认 `400`）
- `GOMOKU_MCTS_PUCT`：PUCT 常数（默认 `2.0`）
- `GOMOKU_AI_WORKERS`：AI 线程数（默认 `2`）
- `GOMOKU_CORS_ORIGINS`：CORS 允许域（默认 `*`，逗号分隔）

Web 前端说明见 `web/README.md`，部署指引见 `WEB_DEPLOY.md`。

## 训练

```bash
python gomoku/trainer.py
```

训练依赖 PyTorch、Ray，且会占用较多算力。可按需调整 `gomoku/trainer.py` 中的参数。

## 评估/验证

运行轻量的 MCTS 案例集：

```bash
python gomoku/evaluation/mcts_cases.py
```

## Rust MCTS（可选）

`zero_mcts_rs/` 提供实验性的 Rust MCTS。若需要在 Python 中使用，请参考目录内的 `pyproject.toml`
与 `Cargo.toml`，通过 maturin 构建安装。

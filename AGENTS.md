# 仓库协作指南

## 项目结构与模块组织
核心包位于 `gomoku/`，对弈、搜索与训练逻辑主要在 `gomoku_env.py`、`mcts.py`、`zero_mcts.py`、`policy.py`、`player.py`、`trainer.py`。GUI 入口为 `gomoku/run_gui.py` 与 `gomoku/gui.py`，使用说明见 `gomoku/README_GUI.md`。预训练权重存放在 `gomoku/models/` 与 `gomoku/continue_model/`。评估脚本位于 `gomoku/evaluation/`，TensorBoard 日志通常写入 `gomoku/runs/`。仓库元数据与依赖定义在 `pyproject.toml`（`uv.lock` 记录解析版本）。

## 构建、测试与开发命令
- `python gomoku/run_gui.py` 启动本地 GUI。
- `uvicorn gomoku.app:app --reload` 启动 FastAPI 服务（`gomoku/app.py`）。
- `python gomoku/trainer.py` 启动训练（依赖 PyTorch、Ray 及合适设备）。
- 可选依赖安装：`uv sync`（使用 uv）或 `python -m pip install -e .`。

## 代码风格与命名规范
遵循 PEP 8。使用 4 空格缩进，函数/变量/文件名使用 `snake_case`，类名使用 `PascalCase`。模块名简短清晰（如 `zero_mcts.py`）。新写或修改的代码尽量补充类型标注。

## 测试规范
无统一测试 runner。使用 `gomoku/evaluation/` 下脚本做功能验证，多数文件以 `_test.py` 结尾但直接运行（如 `python gomoku/evaluation/lightweight_eval.py`）。新增评估工具需保持运行时间合理。

## 提交与 PR 规范
提交信息使用 `type: description` 格式（如 `feature: gui`），简短明确。PR 需包含：简要说明、运行过的命令（或说明未运行原因）、新增/更新模型权重说明。GUI 变更需附截图，必要时关联 issue。

## 配置与资产
模型权重大，脚本中优先引用 `gomoku/models/` 下路径，避免硬编码绝对路径。新增资产需在相关脚本或 README 中说明位置与命名规则。

## 新增原则
当需要“按文档实现特性”时，必须先在文档中生成方案，再开始实际实现。

## Skills
这些技能在启动时从多个本地来源发现。每条包含名称、描述与文件路径，便于打开查看完整说明。
- skill-creator: 用于创建/更新 Codex 技能（扩展知识、流程或工具集成）。(file: /home/smokingmouse/.codex/skills/.system/skill-creator/SKILL.md)
- skill-installer: 从预设列表或 GitHub 仓库安装技能（含私有仓库）。(file: /home/smokingmouse/.codex/skills/.system/skill-installer/SKILL.md)
- Discovery: 可用技能列在项目文档或运行时 "## Skills" 区域（名称 + 描述 + 文件路径），这些是权威来源；技能主体在磁盘路径中。
- Trigger rules: 若用户点名技能（`$SkillName` 或明文）或任务明显匹配技能描述，该回合必须使用技能。多次点名需全部使用；未再次提及则不延续到下一回合。
- Missing/blocked: 若点名技能不存在或路径不可读，简要说明并继续采用替代方案。
- How to use a skill（逐步披露）：
  1) 决定使用技能后，先打开其 `SKILL.md`，只读满足流程的最小部分。
  2) 若 `SKILL.md` 指向 `references/` 等目录，只加载必要文件，不要全量读取。
  3) 若存在 `scripts/`，优先运行或修改脚本，避免手写长代码块。
  4) 若存在 `assets/` 或模板，优先复用而非重建。
- Description as trigger: `SKILL.md` 中的 YAML `description` 是主触发信号；不确定时先做简短澄清。
- Coordination and sequencing:
  - 若多个技能适用，选择覆盖需求的最小集合并声明使用顺序。
  - 说明正在使用的技能及原因（一句话）。若跳过明显技能，也要说明原因。
- Context hygiene:
  - 控制上下文规模，长内容先总结；只加载必要文件。
  - 避免深层嵌套引用，优先 `SKILL.md` 直接链接的一跳文件。
  - 存在多个变体时只选相关引用并说明选择。
- Safety and fallback: 若技能无法正常使用（缺文件/指令不清），说明问题并继续替代方案。

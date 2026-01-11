# Web + 对战技术方案（大纲）

## 1. 目标
- 提供 Web UI 进行对弈与 AI vs AI 观战。
- 支持多种 MCTS 后端（Python / Rust）并保证能力可对齐。
- 便于扩展：新模型、新评估模式、新 UI 视图。

## 2. 范围
- 范围内：Web UI、API 服务、对局编排、模型加载、基础指标。
- 范围外：训练链路的系统性改造、分布式训练基础设施。

## 3. 当前状态
- Web 前端：`web/` 下的 Next.js。
- API 服务：`gomoku/app.py`（FastAPI）。
- 核心对弈与 MCTS：`gomoku/`，Rust MCTS 在 `zero_mcts_rs/`。

## 4. 目标架构
- 前端（Next.js） -> 后端（FastAPI） -> 引擎（Python/Rust MCTS）。
- 本地单进程开发，生产可扩展为多 worker。

## 5. 组件职责
### 5.1 前端（web/）
- 棋盘展示、走子记录、对局控制。
- 模式：人机、机机、复盘。
- 实时更新：轮询或 WebSocket（待定）。

### 5.2 后端（gomoku/app.py）
- 会话管理与对局编排。
- MCTS 引擎选择（python/rust）。
- 模型加载与缓存（可扩展）。
- 对局指标接口（时延、步数、胜负）。

### 5.3 引擎层
- Python：`gomoku/light_zero_mcts.py`、`gomoku/zero_mcts.py`。
- Rust：`zero_mcts_rs`（Python 绑定）。
- 统一策略推理接口（obs -> logits/value）。

## 6. API 设计（草案）
- `POST /api/match/start`
  - body: { board_size, engine, model, iterations, temperature, seed }
  - response: { match_id }
- `POST /api/match/step`
  - body: { match_id, action? }
  - response: { board, current_player, done, winner, last_action }
- `GET /api/match/state`
  - response: { board, moves, done, winner }
- `POST /api/match/stop`

## 7. 数据流
- 前端发起对局 -> 后端创建 env + MCTS 状态。
- 每步：引擎落子或接受玩家落子。
- 后端回传棋盘、状态与元信息。

## 8. 对局模式
- 人机对战。
- AI vs AI（Python vs Rust 或同引擎）。
- 自博弈用于评估统计（不一定有 UI）。
- 支持预设棋局起局（来自棋谱面板）。

## 9. 模型管理
- 默认模型目录：`gomoku/continue_model/`。
- 扩展：模型注册（名称/版本/元数据）。
- 加载策略：对局开始加载，复用缓存。

## 10. 性能与对齐
- 目标：相同迭代数下胜率接近。
- 指标：单步时延、总耗时、胜负比例。
- 可选：保存每步 pi 以做对齐分析。

## 11. 可观测性
- 对局开始/结束日志：耗时、步数、胜负。
- 可选：对局记录用于回放。

## 12. 部署
- 本地：`uvicorn gomoku.app:app --reload`。
- Web：`web/` 开发服务器或静态部署（待定）。
- 生产：API + 静态 Web 分离部署。

## 13. 测试与验证
- 手动：AI vs AI 对比胜负与时延。
- 脚本：复用 `gomoku/evaluation/`。
- 训练流程：定期加载验证集，记录 top1/topK 命中率并写入 TensorBoard。
- 训练采样：自博弈完成后按比例分叉，从中局重新开始一盘生成额外样本。

## 14. 新增功能：棋谱绘制与验证集
- 功能目标：
  - 在 Web 上绘制棋谱（逐步落子、悔棋、标注关键点）。
  - 支持保存当前局面与“最优行动点（正解）”。
  - 形成验证集，供后续训练与评估使用。
- 实施方案（落地版）：
  - 前端新增“棋谱绘制”面板，支持模式切换：落黑、落白、擦除、标注最佳点。
  - 支持撤销/清空，保存时将棋盘编码为 2D 数组（0/1/2）。
  - 最优点支持多选，保存时统一转为 action 索引数组。
  - 提供可配置 API 地址（默认由 ws 地址推导或环境变量）。
  - 保存成功后返回样本 id，前端显示反馈。
  - 绘制交互：左键落黑、右键落白；根据棋子数量自动推断当前落子方。
- 数据结构（草案）：
  - `state`: 当前局面（与训练观测一致）。
  - `best_action`: 标注的最优动作，保存为 action 索引数组（可多选）。
  - `meta`: 备注、作者、来源、时间、版本、难度等。
- 存储位置（建议）：
  - `gomoku/validation_sets/` 目录下按日期或主题分类。
  - JSONL 或 NPZ 格式（待定）。
- API 草案：
  - `POST /api/validation/save`
    - body: { state, best_action, meta }
  - `GET /api/validation/list`
  - `GET /api/validation/get?id=...`
  - `POST /api/validation/update`
    - body: { id, state?, best_action?, meta?, board_size? }
  - `POST /api/validation/delete`
    - body: { id }
- 补充实现点：
  - `state` 结构统一为对象：`{ board, current_player, last_action }`，兼容旧格式。
  - 前端提供验证集列表与回放，支持加载到棋谱面板继续编辑并更新。
  - 增加“难度”字段（如 1~5），用于训练评估与筛选。
  - 支持删除样本，避免验证集污染。
  - 列表支持按难度过滤。

## 15. 新增功能：AI vs AI（Web）
- 目标：在 Web 端直接发起 AI vs AI 对局，并实时回放。
- 实现方案：
  - WebSocket 增加 `action=ai_vs_ai`，服务端创建会话并自动循环落子。
  - 每步推送 `game_state`，前端复用棋盘渲染。
  - 可选：支持 `delay_ms` 控制节奏。
- 前端交互：
  - 增加对局模式切换（Human vs AI / AI vs AI）。
  - AI vs AI 模式下隐藏落子交互，仅展示对局进展。
  - 支持暂停/继续/停止控制，便于观察与复盘。
- 预设起局：
  - 对局开始时可携带 `preset`（棋盘 + 当前落子方）。
- 单步执行：
  - 提供 `action=ai_step`，在暂停或停止后手动执行一步。
- 控制协议（草案）：
  - `action=ai_pause`：暂停自动落子。
  - `action=ai_resume`：继续自动落子。
  - `action=ai_stop`：停止自动落子并保持棋盘状态。

## 16. 待讨论问题
- WebSocket 还是轮询？
- 是否统一随机种子以便跨引擎可复现？
- 棋谱格式与回放标准如何定义？

## 17. 里程碑（可编辑）
- M1：对局 API + Web 基础 UI。
- M2：AI vs AI 模式 + 对局统计视图。
- M3：Rust 引擎切换开关。
- M4：棋谱绘制 + 验证集保存。

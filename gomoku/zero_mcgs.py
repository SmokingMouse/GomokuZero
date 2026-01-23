import math
import torch
import numpy as np
from gomoku.gomoku_env import GomokuEnv
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gomoku.batched_inference import BatchPolicyRunner


class ZeroMCGSNode:
    def __init__(self, prior_prob: float = 0.0, raw_value: float = 0.0):
        # 神经网络输出的原始估值 V (U)
        self.raw_value = raw_value

        # 缓存的 Q 值，初始化为 U，随子节点更新而变动
        self.q_value = raw_value

        # 节点被访问的总次数 (Node Visits)
        # 注意：在 MCGS 中，PUCT 不用这个，只用于统计和显示
        self.visit_count = 0

        # 核心数据结构：边 (Edges)
        # 格式：action -> { 'node': ZeroMCGSNode, 'n': edge_visits, 'p': prior_prob }
        self.children = {}

    def recompute_q(self):
        """
        [MCGS 核心]：根据子节点当前的 Q 值，重新计算自己的 Q 值。
        公式：Q(s) = ( U(s) + sum( N(s,a) * -Q(s') ) ) / ( 1 + sum(N(s,a)) )
        注意：因为是零和博弈，子节点的 Q 是对手的视角，所以要取反 (-child.q)。
        """
        total_edge_visits = 0
        weighted_q_sum = 0.0

        for info in self.children.values():
            child_node = info["node"]
            edge_visits = info["n"]

            if edge_visits > 0:
                total_edge_visits += edge_visits
                # 直接读取子节点缓存的 Q 值 (O(1))
                # 视角转换：对手的好就是我的坏
                weighted_q_sum += edge_visits * (-child_node.q_value)

        # 分母 +1 是为了包含 raw_value (U) 的那一次虚拟计数
        self.q_value = (self.raw_value + weighted_q_sum) / (1 + total_edge_visits)


class ZeroMCGS:
    def __init__(
        self,
        policy: torch.nn.Module,
        puct: float = 2.0,
        device: str = "cpu",
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        policy_runner: Optional["BatchPolicyRunner"] = None,
    ):
        self.policy = policy
        self.puct = puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # 全局置换表 (Transposition Table)
        # Key: Board Hash, Value: ZeroMCGSNode
        self.node_table = {}
        self.root: ZeroMCGSNode | None = None
        print(f"Initialized ZeroMCGS on device: {device}")

    def reset(self):
        """清空树/图，开始新的一局"""
        self.node_table.clear()
        self.root = None

    def step(self, action):
        """
        在对局中推进一步。
        尝试复用现有的子图作为新的 Root。
        """
        pass
        if self.root is not None and action in self.root.children:
            # 移动 Root 指针到子节点
            self.root = self.root.children[action]["node"]
            # 注意：在 Graph 中不能简单断开 parent 引用，也不能随意清空 table
            # 为了节省内存，工业级实现通常会在这里做 LRU 清理，或者清理不可达节点
            # 这里简单起见，不做清理
        else:
            # 如果动作不在搜索树内（比如对方下了一步意料之外的棋），重置
            self.root = None
            # 可选：self.node_table.clear()

    def _policy_eval(self, obs: np.ndarray):
        if self.policy_runner is not None:
            return self.policy_runner.predict(obs)

        # 优化4: 确保 tensor 在正确的 device
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            return self.policy(torch_x)

    def run(
        self,
        env: GomokuEnv,
        iterations: int = 800,
        temperature: float = 1.0,
        use_dirichlet: bool = True,
    ):
        """执行 MCTS 搜索"""

        # 1. 确保 Root 存在且已初始化
        board_hash = env.get_hash()
        if self.root is None:
            if board_hash in self.node_table:
                self.root = self.node_table[board_hash]
            else:
                # 创建临时的 Dummy Root，稍后在 expand 中填充值
                self.root = ZeroMCGSNode()
                self.node_table[board_hash] = self.root

        # 如果 Root 是刚创建的空节点（没有 children 且 raw_value 可能未准），先扩展它
        if not self.root.children and self.root.visit_count == 0:
            # 判断是否终局（防止在终局状态 crash）
            if not env.done:
                self._expand(self.root, env)
                if use_dirichlet:
                    self._add_dirichlet_noise(self.root)

        # 2. 搜索循环
        for _ in range(iterations):
            node = self.root
            scratch_env = env.clone()
            search_path = []  # 记录路径 [(node, action), ...]

            # --- A. Selection (下探) ---
            while node.children:
                action, next_node = self._select_child(node)
                search_path.append((node, action))

                scratch_env.step(action)
                node = next_node

                # 如果遇到终局，停止下探
                if scratch_env.done:
                    break

            # --- B. Evaluation & Expansion (扩展) ---
            # 此时 node 是叶子节点（或者刚到达的终局节点）

            # 情况 1: 游戏结束
            if scratch_env.done:
                # 计算终局价值。
                # winner: 黑色=1, 白色=-1, 平局=0
                # 当前视角 value: 如果 winner == scratch_env.current_player，则是 1
                # 也就是：value = winner * current_player_color
                # (假设 env.current_player 返回 1 或 -1)

                # 注意：scratch_env.step() 之后，current_player 已经切换到了对手
                # 刚才落子的是 'last_player'。
                # 胜负是相对于刚才落子的人的。
                # 假设我们总是存储 "当前节点行动者" 的优势。
                # 终局节点没有行动者了，它的价值对于上一手落子的人来说是明确的。

                # 简单处理：
                # 如果 winner != 0, 说明上一手导致了分出胜负。
                # 站在 node 的视角（如果还能下），它已经输了（因为对手连成了5子）。
                # 所以 node.raw_value 应该是 -1.0 (必败/已死)。
                # 或者更直观的：Backprop 时，我们拿到的 value 是相对于 node 的。
                if scratch_env.winner != 0:
                    value = -1.0  # 无论谁到了这个局面，都被判负（因为对方刚连完）
                    # 也有实现是存 1.0 (代表上一步的人赢了)，取决于 update 逻辑。
                    # 这里采用：Q 代表当前局面好坏。局面已死，Q=-1。
                else:
                    value = 0.0  # 平局

                # 更新叶子节点的固有价值
                node.raw_value = value
                node.q_value = value

            # 情况 2: 普通节点，且未扩展过 (children为空)
            elif not node.children:
                # 扩展，计算 U，并初始化 children
                value = self._expand(node, scratch_env)

            # 情况 3: 节点已存在且已扩展 (Graph 中汇聚到了旧节点)
            else:
                # 直接复用该节点已有的 Q 值
                value = node.q_value

            # --- C. Backpropagation (回溯) ---
            # 沿着路径更新 Edge Visits，并触发 Recompute
            self._backpropagate(search_path)

        # 3. 返回最终决策
        return self.select_action_with_temperature(temperature)

    def _select_child(self, node):
        best_score = -float("inf")
        best_action = -1
        best_child = None

        # [MCGS] 使用 Root 的总边缘访问次数
        # sum(N(s, b))
        parent_visits = sum(info["n"] for info in node.children.values())

        # 加上 epsilon 防止除零 (虽然通常 parent_visits > 0)
        sqrt_parent_visits = math.sqrt(parent_visits + 1e-8)

        for action, info in node.children.items():
            child_node = info["node"]
            edge_visits = info["n"]
            prior_prob = info["p"]

            # [MCGS] 递归 Q 值
            # 站在 node 视角，child 的 Q 是对手的收益，所以取反
            q_value = -child_node.q_value

            # [MCGS] 探索项分母使用 Edge Visits
            u_value = self.puct * prior_prob * sqrt_parent_visits / (1 + edge_visits)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child_node

        return best_action, best_child

    def _expand(self, node: ZeroMCGSNode, env: GomokuEnv):
        # 1. 神经网络推理
        obs = env._get_observation()  # 需适配你的 env
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        self.policy.eval()
        with torch.no_grad():
            policy_logits, value_tensor = self.policy(torch_x)

        value = value_tensor.item()
        node.raw_value = value
        node.q_value = value  # 初始 Q = U

        # 2. 处理动作概率
        valid_actions = env.get_valid_actions()
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        valid_probs = policy_probs[valid_actions]
        prob_sum = valid_probs.sum()
        if prob_sum > 0:
            valid_probs /= prob_sum  # 重新归一A化
        else:
            # 极罕见情况：所有合法动作概率由于 mask 为 0，均分
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)

        # 3. 创建/链接子节点
        for action, prob in zip(valid_actions, valid_probs):
            # 虚拟走一步，计算哈希
            # 注意：这里需要 env 支持轻量级 clone 和 step
            # 如果 clone 成本高，可以只计算 hash 增量
            next_env = env.clone()
            next_env.step(action)
            child_hash = next_env.get_hash()

            # [MCGS] 查表：如果节点已存在，直接链接
            if child_hash in self.node_table:
                child = self.node_table[child_hash]
            else:
                child = ZeroMCGSNode(raw_value=0.0)  # 先占位，下次访问再 expand
                self.node_table[child_hash] = child

            # [MCGS] 存边信息
            node.children[action] = {
                "node": child,
                "n": 0,  # Edge Visits 初始为 0
                "p": prob,
            }

        return value

    def _backpropagate(self, search_path):
        """
        [MCGS] 路径回溯更新 (On-path Update)
        从深到浅，更新 edge_n，并触发 recompute_q
        """
        # reversed: 从叶子节点的父节点 -> Root
        for parent, action in reversed(search_path):
            # 1. 更新边缘访问次数
            parent.children[action]["n"] += 1

            # 2. 更新节点总访问次数 (仅用于统计/调试)
            parent.visit_count += 1

            # 3. [核心] 触发 Q 值重新计算
            # 因为我的子节点 (children[action]['node']) 的状态变了
            # (可能是子节点的 Q 变了，也可能是这条边的 n 变了)
            # 所以我要利用缓存的子节点 Q 值，更新我自己的 Q
            parent.recompute_q()

    def _add_dirichlet_noise(self, node):
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            node.children[action]["p"] = (1 - self.dirichlet_epsilon) * node.children[
                action
            ]["p"] + self.dirichlet_epsilon * noise[i]

    def select_action_with_temperature(self, temperature=1.0):
        """
        根据 Root 的 Edge Visits 生成最终策略
        """
        if self.root is None or not self.root.children:
            return -1, np.zeros(1)  # Should not happen

        # 提取 (action, edge_visits)
        visits = [(action, info["n"]) for action, info in self.root.children.items()]

        actions, counts = zip(*visits)
        counts = np.array(counts).astype(float)

        if temperature == 0:
            # 贪婪选择
            best_idx = np.argmax(counts)
            action = actions[best_idx]
            # Pi 依然返回 one-hot 或者 归一化的 counts
            pi_probs = counts / counts.sum()
        else:
            # 带温度采样
            scaled_counts = counts ** (1.0 / temperature)
            probs = scaled_counts / np.sum(scaled_counts)
            action = np.random.choice(actions, p=probs)
            pi_probs = probs  # 这里通常返回访问分布作为训练目标

        # 构造完整的 pi 向量 (size = board_w * board_h)
        board_size = getattr(self.policy, "board_size", 9)
        board_area = board_size**2
        full_pi = np.zeros(board_area)
        for act, prob in zip(actions, pi_probs):
            full_pi[act] = prob

        return action, full_pi

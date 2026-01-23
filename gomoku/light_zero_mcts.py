import math
import random
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from gomoku.batched_inference import BatchPolicyRunner


# 优化1: 极简节点类，移除 self.env
class LightNode:
    __slots__ = [
        "visits",
        "value_sum",
        "children",
        "parent",
        "prior_prob",
        "is_noise_added",
        "virtual_loss",
    ]

    def __init__(self, prior_prob=0.0, parent=None):
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}  # {action: LightNode}
        self.parent = parent
        self.prior_prob = float(prior_prob)

        self.is_noise_added = False
        self.virtual_loss = 0

    @property
    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class LightZeroMCTS:
    def __init__(
        self,
        policy,
        puct=2,
        device="cpu",
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        policy_runner: Optional["BatchPolicyRunner"] = None,
        eval_batch_size: int = 1,
        virtual_loss: float = 1.0,
        max_children: int = 0,
    ):
        self.policy = policy
        self.puct = puct
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root = None
        self.policy_runner = policy_runner
        self.eval_batch_size = max(1, eval_batch_size)
        self.virtual_loss = float(virtual_loss)
        self.max_children = max(0, int(max_children))

        self.transposition_table = {}
        self._debug_nodes = 0
        self._debug_children = 0

    def reset_debug_stats(self) -> None:
        self._debug_nodes = 0
        self._debug_children = 0

    def debug_stats(self) -> dict:
        if not self.transposition_table:
            return {"nodes": 0, "children": 0, "avg_children": 0.0}
        nodes = len(self.transposition_table)
        children = sum(len(node.children) for node in self.transposition_table.values())
        avg_children = children / nodes if nodes else 0.0
        return {"nodes": nodes, "children": children, "avg_children": avg_children}

    def step(self, action):
        """
        当真实游戏发生一步动作时，调用此方法移动 MCTS 的根节点。
        这样就实现了“树的复用”。
        """
        # pass
        # return
        if self.root is not None and action in self.root.children:
            # 1. 核心逻辑：将根节点指向对应的子节点
            self.root = self.root.children[action]
            self.root.parent = None  # 断开与旧父节点的联系，帮助垃圾回收
        else:
            # 2. 如果这步棋不在之前的搜索树里（比如对手下了一步极其冷门的棋）
            self.root = None

    def run(self, env, iterations=800, use_dirichlet=True):
        if self.root is None:
            board_hash = env.board.tobytes()

            if board_hash in self.transposition_table:
                self.root = self.transposition_table[board_hash]
                self.root.parent = None  # 断开与旧父节点的联系，帮助垃圾回收
            else:
                self.root = LightNode(prior_prob=1.0)
                self._expand(self.root, env)
                self.transposition_table[board_hash] = self.root

        if use_dirichlet and env.move_size <= 1:
            self._add_dirichlet_noise(self.root, env)

        if self.eval_batch_size <= 1:
            for _ in range(iterations):
                node = self.root

                # 优化2: 只在每次模拟开始时 clone 一次环境
                # 这个 scratch_env 会随着 Select 过程动态变化
                scratch_env = env.clone()

                # --- SELECT ---
                while node.children:
                    # 已经是内部节点，选择下一步
                    action, node = self._select_child(node)
                    scratch_env.step(action)  # 模拟环境跟进

                leaf_hash = scratch_env.board.tobytes()

                if leaf_hash not in self.transposition_table:
                    self.transposition_table[leaf_hash] = node

                # --- EXPAND & EVALUATE ---
                if scratch_env.done:
                    value = -1.0
                    if scratch_env.winner == 0:
                        value = 0.0
                else:
                    value = self._expand(node, scratch_env)

                # --- BACKPROPAGATION ---
                self._backpropagate(node, value)
                # if env.board.sum() != origin_sum:
                #     print("完了！原环境被污染了！")
        else:
            batch_size = self.eval_batch_size
            for start in range(0, iterations, batch_size):
                current_batch = min(batch_size, iterations - start)
                pending = []
                terminals = []

                for _ in range(current_batch):
                    node = self.root
                    scratch_env = env.clone()
                    path = []

                    while node.children:
                        path.append(node)
                        action, node = self._select_child(node)
                        scratch_env.step(action)

                    path.append(node)
                    self._apply_virtual_loss(path)

                    leaf_hash = scratch_env.board.tobytes()
                    if leaf_hash not in self.transposition_table:
                        self.transposition_table[leaf_hash] = node

                    if scratch_env.done:
                        value = -1.0
                        if scratch_env.winner == 0:
                            value = 0.0
                        terminals.append((path, value))
                    else:
                        obs = scratch_env._get_observation()
                        pending.append((path, node, scratch_env, obs))

                if pending:
                    obs_batch = np.stack([item[3] for item in pending], axis=0)
                    policy_logits, values = self._policy_eval_batch(obs_batch)
                    policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.view(-1).cpu().numpy()

                    for idx, (path, node, scratch_env, _) in enumerate(pending):
                        if not node.children:
                            self._expand_from_probs(
                                node, scratch_env, policy_probs[idx]
                            )
                        self._backpropagate_with_virtual_loss(
                            path, float(values[idx])
                        )

                for path, value in terminals:
                    self._backpropagate_with_virtual_loss(path, value)

        return self._best_action(env)

    def _select_child(self, node):
        # 优化3: 向量化计算 PUCT (虽然 Python 循环也够快，但逻辑要清晰)
        best_score = -float("inf")
        best_action = -1
        best_child = None

        # 预计算 sqrt(N)
        sqrt_parent_visits = math.sqrt(node.visits + node.virtual_loss)

        for action, child in node.children.items():
            # AlphaZero 标准 PUCT 公式
            # Q 是站在父节点视角的，所以不仅不用负号，甚至需要注意 value 的正负定义
            # 假设存储的 value_sum 是从 child 视角看的胜率，那么 Q_parent = -Q_child
            q_value = -(child.value_sum / (child.visits + child.virtual_loss)) if (child.visits + child.virtual_loss) > 0 else 0.0

            u_value = (
                self.puct
                * child.prior_prob
                * sqrt_parent_visits
                / (1 + child.visits + child.virtual_loss)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, env):
        # 神经网络推理
        obs = env._get_observation()
        policy_logits, value = self._policy_eval(obs)

        # Masking
        valid_actions = env.get_valid_actions()
        # 优化5: 避免复杂的 tensor mask 操作，直接在 python 层过滤
        # 虽然 tensor 操作快，但来回传数据有开销。对于 9x9 棋盘，直接取出来处理可能更快
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        self._expand_children(node, valid_actions, policy_probs)

        return value.item()

    def _expand_from_probs(self, node, env, policy_probs: np.ndarray) -> None:
        valid_actions = env.get_valid_actions()
        self._expand_children(node, valid_actions, policy_probs)

    def _expand_children(self, node, valid_actions, policy_probs: np.ndarray) -> None:
        actions = valid_actions
        if self.max_children > 0 and len(actions) > self.max_children:
            probs = np.array([policy_probs[action] for action in actions])
            top_idx = np.argpartition(probs, -self.max_children)[-self.max_children:]
            top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]
            actions = [actions[i] for i in top_idx]

        policy_sum = 0.0
        for action in actions:
            prob = float(policy_probs[action])
            node.children[action] = LightNode(prior_prob=prob, parent=node)
            policy_sum += prob

        if policy_sum > 0:
            for child in node.children.values():
                child.prior_prob /= policy_sum

    def _policy_eval(self, obs: np.ndarray):
        if self.policy_runner is not None:
            return self.policy_runner.predict(obs)

        # 优化4: 确保 tensor 在正确的 device
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            return self.policy(torch_x)

    def _policy_eval_batch(self, obs_batch: np.ndarray):
        torch_x = torch.from_numpy(obs_batch).float().to(self.device)
        with torch.no_grad():
            return self.policy(torch_x)

    def _backpropagate(self, node, value):
        # 沿途回溯更新
        curr = node
        while curr:
            curr.visits += 1
            curr.value_sum += float(value)
            value = -value  # 切换视角：对手赢 = 我输
            curr = curr.parent

    def _apply_virtual_loss(self, path):
        if self.virtual_loss <= 0:
            return
        for node in path:
            node.virtual_loss += self.virtual_loss

    def _backpropagate_with_virtual_loss(self, path, value):
        if self.virtual_loss > 0:
            for node in path:
                node.virtual_loss -= self.virtual_loss
        for node in path:
            node.visits += 1
            node.value_sum += float(value)
            value = -value

    def _add_dirichlet_noise(self, node, env):
        if node.is_noise_added:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior_prob = (
                1 - self.dirichlet_epsilon
            ) * child.prior_prob + self.dirichlet_epsilon * noise[i]
        node.is_noise_added = True

    def _best_action(self, env):
        return self.select_action_with_temperature(0, 1)

    def select_action_with_temperature(
        self, temperature=1.0, top_k=None, forbidden_actions=None
    ):
        """
        根据 MCTS 的访问次数选择动作，并返回用于训练的概率分布 pi。
        """
        if self.root is None or not self.root.children:
            # 防御性编程：如果没有搜索过，随机返回一个合法动作（需要外部传入 valid_actions）
            # 但通常 run 之后肯定有 children
            return -1, []

        forbidden_set = set(forbidden_actions or [])
        # 1. 提取 (action, visits) 数据
        # 注意：LightNode 的 children 是字典 {action_id: node}
        visits = [(action, node.visits) for action, node in self.root.children.items()]
        filtered_visits = [
            (action, count) for action, count in visits if action not in forbidden_set
        ]
        if filtered_visits:
            visits = filtered_visits

        # 2. 生成动作 (Action Selection)
        if temperature == 0:
            # --- 贪婪模式 (Greedy) ---
            # 直接选访问次数最多的，常用于评估和比赛
            # 打乱顺序是为了打破相同访问次数的平局 (Tie-breaking)
            random.shuffle(visits)
            action = max(visits, key=lambda x: x[1])[0]
        else:
            # --- 随机模式 (Stochastic) ---
            # 用于自我博弈训练，增加多样性
            actions, counts = zip(*visits)
            counts = np.array(counts)

            # 使用温度调节分布： count^(1/temp)
            # 温度越高越均匀，温度越低越尖锐
            if top_k is not None and top_k < len(counts):
                # 如果指定了 top_k，只保留访问量前 k 的动作
                indices = np.argsort(counts)[-top_k:]
                actions = np.array(actions)[indices]
                counts = counts[indices]

            scaled_counts = counts ** (1 / temperature)

            # 归一化为概率
            probs = scaled_counts / np.sum(scaled_counts)

            # 根据概率采样
            action = np.random.choice(actions, p=probs)

        # 3. 生成训练目标 (Training Target - Pi)
        # 注意：AlphaZero 标准做法是，训练目标使用【未加温度】的原始访问频率
        # 即：N / sum(N)。这代表了 MCTS 搜索出的真实“胜率分布”。

        # 获取棋盘大小 (从 policy 中获取，或者默认为 9/15)
        board_size = getattr(self.policy, "board_size", 9)
        pi = np.zeros(board_size * board_size)

        total_visits = sum(node.visits for node in self.root.children.values())

        if total_visits > 0:
            for act, node in self.root.children.items():
                pi[act] = node.visits / total_visits
            if forbidden_set:
                for act in forbidden_set:
                    if 0 <= act < len(pi):
                        pi[act] = 0.0
                pi_sum = pi.sum()
                if pi_sum > 0:
                    pi /= pi_sum

        return action, pi

    def metrics(self):
        """
        返回一些 MCTS 统计数据，供调试和分析用。
        """
        if self.root is None:
            return {"root_visits": 0, "root_q_value": 0.0, "num_children": 0}

        import sys

        return {
            "root_visits": self.root.visits,
            "root_q_value": self.root.q_value,
            "num_children": len(self.root.children),
            "transposition_table_size": len(self.transposition_table),
            "memory_usage_mb": (
                sum(sys.getsizeof(node) for node in self.transposition_table.values())
                / 1024
            ),
        }

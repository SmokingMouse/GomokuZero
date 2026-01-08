use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::Dirichlet;
use std::collections::HashMap;

// ==========================================
// 1. 纯 Rust 实现的 Gomoku 逻辑 (为了速度)
// ==========================================
#[derive(Clone)]
struct RustGomokuEnv {
    board: Vec<i8>, // 0: empty, 1: black, 2: white
    size: usize,
    current_player: i8,
    move_count: usize,
    winner: i8, // 0: none, 1: black, 2: white, 3: draw
}

impl RustGomokuEnv {
    fn new(size: usize) -> Self {
        RustGomokuEnv {
            board: vec![0; size * size],
            size,
            current_player: 1,
            move_count: 0,
            winner: 0,
        }
    }

    // 从 Python 传入的 board 数组初始化
    fn from_board(board: &[i8], size: usize, current_player: i8) -> Self {
        let mut move_count = 0;
        for &c in board {
            if c != 0 {
                move_count += 1;
            }
        }
        RustGomokuEnv {
            board: board.to_vec(),
            size,
            current_player,
            move_count,
            winner: 0, // 注意：这里假设传入的状态没有结束，或者需要在外部判断
        }
    }

    fn get_valid_actions(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn step(&mut self, action: usize) {
        if self.board[action] != 0 {
            panic!("Invalid move: {}", action);
        }
        self.board[action] = self.current_player;
        self.move_count += 1;
        
        // 检查胜负
        if self.check_win(action) {
            self.winner = self.current_player;
        } else if self.move_count == self.size * self.size {
            self.winner = 3; // Draw
        }

        self.current_player = 3 - self.current_player;
    }

    fn check_win(&self, action: usize) -> bool {
        let size = self.size as isize;
        let x = (action % self.size) as isize;
        let y = (action / self.size) as isize;
        let color = self.board[action];
        let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dx, dy) in directions {
            let mut count = 1;
            // 正向
            let mut nx = x + dx;
            let mut ny = y + dy;
            while nx >= 0 && nx < size && ny >= 0 && ny < size {
                if self.board[(ny * size + nx) as usize] == color {
                    count += 1;
                } else {
                    break;
                }
                nx += dx;
                ny += dy;
            }
            // 反向
            nx = x - dx;
            ny = y - dy;
            while nx >= 0 && nx < size && ny >= 0 && ny < size {
                if self.board[(ny * size + nx) as usize] == color {
                    count += 1;
                } else {
                    break;
                }
                nx -= dx;
                ny -= dy;
            }

            if count >= 5 {
                return true;
            }
        }
        false
    }
}

// ==========================================
// 2. MCTS 节点结构
// ==========================================
struct Node {
    visits: u32,
    value_sum: f32,
    prior: f32,
    children: HashMap<usize, Node>,
}

impl Node {
    fn new(prior: f32) -> Self {
        Node {
            visits: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
        }
    }

    fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

// ==========================================
// 3. MCTS 主类 (暴露给 Python)
// ==========================================
#[pyclass]
struct LightZeroMCTS {
    root: Option<Node>,
    board_size: usize,
    puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
}

#[pymethods]
impl LightZeroMCTS {
    #[new]
    fn new(board_size: usize, puct: f32, dirichlet_alpha: f32, dirichlet_epsilon: f32) -> Self {
        LightZeroMCTS {
            root: None,
            board_size,
            puct,
            dirichlet_alpha,
            dirichlet_epsilon,
        }
    }

    // 对应 Python 的 step 方法：复用树
    fn step(&mut self, action: usize) {
        if let Some(mut root) = self.root.take() {
            if let Some(child) = root.children.remove(&action) {
                // 如果子节点存在，直接将其作为新的根节点
                self.root = Some(child);
                // print!("Tree reused!"); 
                return;
            }
        }
        // 如果无法复用，重置为空
        self.root = None;
    }

    // 对应 Python 的 run 方法
    fn run(
        &mut self,
        py: Python,
        initial_board: PyReadonlyArray1<i8>, // 从 Python 传入当前棋盘
        current_player: i8,                  // 传入当前玩家
        policy_fn: PyObject,                 // Python 的预测函数
        iterations: usize,
        use_dirichlet: bool,
    ) -> PyResult<usize> {
        let board_vec = initial_board.as_slice()?.to_vec();
        
        // 1. 初始化根节点
        if self.root.is_none() {
            let mut root = Node::new(1.0);
            // 首次扩展，需要调用 Policy
            let env = RustGomokuEnv::from_board(&board_vec, self.board_size, current_player);
            self.expand_root(py, &mut root, &env, &policy_fn)?;
            self.root = Some(root);
        }

        let mut root = self.root.take().unwrap();

        // 2. 添加噪声
        if use_dirichlet {
            self.add_dirichlet_noise(&mut root);
        }

        // 3. 主循环
        for _ in 0..iterations {
            let mut node = &mut root;
            let mut env = RustGomokuEnv::from_board(&board_vec, self.board_size, current_player);
            let mut path = Vec::new(); // 记录路径用于回溯

            // --- SELECT ---
            while !node.children.is_empty() {
                let action = self.select_child(node);
                env.step(action);
                path.push(action);
                node = node.children.get_mut(&action).unwrap();
            }

            // --- EXPAND & EVALUATE ---
            let value = if env.winner != 0 {
                // 游戏结束
                if env.winner == 3 {
                    0.0 // Draw
                } else {
                    // 如果 env.winner == env.current_player (刚走这步的人赢了)，
                    // 对于 current_player 来说是 +1。
                    // 但通常 value 是相对于 parent 视角的。
                    // 这里的 env.current_player 已经是下一个人了。
                    // 所以如果 env.winner != env.current_player，说明上一个人赢了。
                    -1.0 
                }
            } else {
                // 未结束，调用 Python 预测
                self.expand_node(py, node, &env, &policy_fn)?
            };

            // --- BACKPROPAGATE ---
            // 先更新叶子
            self.backpropagate(node, value);
            
            // 再沿着 path 从上往下找回节点并更新 (Rust 所有权限制，需要重新从 root 走一遍或者用指针，这里为了安全用重走)
            // 优化：其实上面拿到 &mut node 时就丢失了 root 的引用。
            // 在 Rust 中实现反向链表比较麻烦，通常使用递归或者在单函数内处理。
            // 这里我们为了简单，把逻辑拆分一下：
            // 我们不能同时持有 root 和 child 的可变引用。
            // 所以，我们需要把 select, expand, backprop 放在一个递归函数里，或者使用 非递归的 unsafe 指针。
            // 为了安全，这里演示 "递归式" 写法，虽然在 loop 里调用递归有点怪，但能通过借用检查。
        }
        
        // 上面的循环结构在 Rust 处理 &mut 树时很难写，下面换成“递归+循环”的模式来实现 MCTS
        // 为了避免复杂的生命周期，我们将逻辑改为：每次迭代重新从 root 递归
        self.root = Some(root);
        
        // 重新获取 root 引用进行迭代
        for _ in 0..iterations {
             self.simulate(py, &board_vec, current_player, &policy_fn)?;
        }

        // 返回最佳动作
        Ok(self.best_action_greedy())
    }

    // 获取用于训练的 pi 和 action
    fn select_action_with_temperature(
        &self,
        temperature: f32,
        top_k: Option<usize>
    ) -> PyResult<(usize, Py<PyArray1<f32>>)> {
        let root = self.root.as_ref().expect("Run must be called before select");
        let mut visits: Vec<(usize, u32)> = root.children.iter().map(|(&a, n)| (a, n.visits)).collect();

        if visits.is_empty() {
             // 应该不会发生
             return Python::with_gil(|py| {
                let pi = PyArray1::zeros(py, self.board_size * self.board_size, false);
                Ok((0, pi.to_owned()))
            });
        }

        // 1. Action Selection
        let selected_action = if temperature == 0.0 {
            // Greedy
            let mut rng = thread_rng();
            visits.shuffle(&mut rng); // Tie-breaking
            visits.iter().max_by_key(|x| x.1).unwrap().0
        } else {
            // Stochastic
            let mut counts: Vec<f32> = visits.iter().map(|x| x.1 as f32).collect();
            let mut actions: Vec<usize> = visits.iter().map(|x| x.0).collect();

            // Top-K
            if let Some(k) = top_k {
                if k < visits.len() {
                    // 简单的排序截断
                    let mut zipped: Vec<_> = actions.into_iter().zip(counts.into_iter()).collect();
                    zipped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    zipped.truncate(k);
                    let (a, c): (Vec<_>, Vec<_>) = zipped.into_iter().unzip();
                    actions = a;
                    counts = c;
                }
            }

            // Temperature application
            let exp = 1.0 / temperature;
            let weights: Vec<f32> = counts.iter().map(|c| c.powf(exp)).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            let mut rng = thread_rng();
            actions[dist.sample(&mut rng)]
        };

        // 2. Training Target (Pi) - Raw visits normalized
        let total_visits: u32 = root.children.values().map(|n| n.visits).sum();
        
        let py_pi = Python::with_gil(|py| {
            let pi_vec = PyArray1::zeros(py, self.board_size * self.board_size, false);
            let mut pi_slice = pi_vec.readwrite();
            
            if total_visits > 0 {
                for (action, node) in &root.children {
                    pi_slice[*action] = node.visits as f32 / total_visits as f32;
                }
            }
            pi_vec.to_owned()
        });

        Ok((selected_action, py_pi))
    }
}

// 内部实现方法
impl LightZeroMCTS {
    fn simulate(
        &mut self,
        py: Python,
        board_vec: &[i8],
        current_player: i8,
        policy_fn: &PyObject,
    ) -> PyResult<()> {
        let root = self.root.as_mut().unwrap();
        let mut env = RustGomokuEnv::from_board(board_vec, self.board_size, current_player);
        
        Self::recursive_search(py, root, &mut env, policy_fn, self.puct)
    }

    // 递归搜索 helper，解决 Rust 的借用检查问题
    fn recursive_search(
        py: Python,
        node: &mut Node,
        env: &mut RustGomokuEnv,
        policy_fn: &PyObject,
        puct: f32,
    ) -> PyResult<f32> {
        // 如果游戏结束
        if env.winner != 0 {
             if env.winner == 3 { return Ok(0.0); } // Draw
             // 此时 env.current_player 是输家，所以返回 -1
             return Ok(-1.0);
        }

        // 如果是叶子节点 (未扩展)
        if node.children.is_empty() {
            // Expand & Evaluate
            // 构造输入给 Python: (1, board_size, board_size) 的 tensor 或者是 list
            // 为了最快速度，我们传 numpy array
            let board_data = env.board.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let py_array = PyArray1::from_vec(py, board_data);
            
            // 这里的 policy_fn 接收一维数组，需要在 Python 端 reshape 或这里传 shape
            // 假设 Python 端处理: def policy(board_array): ...
            let result = policy_fn.call1(py, (py_array,))?; 
            let (probs_dict, value): (&PyDict, f32) = result.extract(py)?;

            // Expand
            let valid_actions = env.get_valid_actions();
            let mut policy_sum = 0.0;
            
            for action in valid_actions {
                // 从 Python 字典中获取概率: {action: prob}
                if let Some(prob) = probs_dict.get_item(action) {
                    let p: f32 = prob.extract()?;
                    let mut child = Node::new(p);
                    node.children.insert(action, child);
                    policy_sum += p;
                }
            }
            
            // Normalize
            if policy_sum > 0.0 {
                for child in node.children.values_mut() {
                    child.prior /= policy_sum;
                }
            }

            // Update self
            node.visits += 1;
            node.value_sum += value;
            return Ok(value);
        }

        // Select
        let action = Self::select_child_action(node, puct);
        
        // Step Env
        env.step(action);
        
        // Recurse
        let child = node.children.get_mut(&action).unwrap();
        let value = Self::recursive_search(py, child, env, policy_fn, puct)?;
        
        // Backprop (Value Flip)
        let value = -value;
        node.visits += 1;
        node.value_sum += value;
        
        Ok(value)
    }

    fn select_child_action(node: &Node, puct: f32) -> usize {
        let sqrt_visits = (node.visits as f32).sqrt();
        let mut best_score = -f32::INFINITY;
        let mut best_action = 0;

        for (&action, child) in &node.children {
            let q = child.q_value(); // 这里通常取 -Q，但 AlphaZero 标准是取父视角的 Q。
            // 这里的 q_value 是 child.value_sum / child.visits。
            // child.value_sum 是从 child 的视角积累的胜负。
            // 对于 parent 来说，child 赢就是 parent 输，所以用 -child.q()
            let q_parent = -q; 
            
            let u = puct * child.prior * sqrt_visits / (1.0 + child.visits as f32);
            let score = q_parent + u;

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }
        best_action
    }

    fn expand_root(
        &self, 
        py: Python, 
        node: &mut Node, 
        env: &RustGomokuEnv, 
        policy_fn: &PyObject
    ) -> PyResult<f32> {
        // 与 recursive_search 里的 expand 逻辑基本一致，专门用于初始化
        let board_data = env.board.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let py_array = PyArray1::from_vec(py, board_data);
        
        let result = policy_fn.call1(py, (py_array,))?;
        let (probs_dict, value): (&PyDict, f32) = result.extract(py)?;

        let valid_actions = env.get_valid_actions();
        let mut policy_sum = 0.0;
        
        for action in valid_actions {
            if let Some(prob) = probs_dict.get_item(action) {
                let p: f32 = prob.extract()?;
                node.children.insert(action, Node::new(p));
                policy_sum += p;
            }
        }
         if policy_sum > 0.0 {
            for child in node.children.values_mut() {
                child.prior /= policy_sum;
            }
        }
        node.visits += 1;
        node.value_sum += value;
        Ok(value)
    }

    fn add_dirichlet_noise(&mut self, node: &mut Node) {
        if node.children.is_empty() { return; }
        let count = node.children.len();
        let dirichlet = Dirichlet::new(&vec![self.dirichlet_alpha; count]).unwrap();
        let mut rng = thread_rng();
        let noise = dirichlet.sample(&mut rng);
        
        let epsilon = self.dirichlet_epsilon;
        for (i, child) in node.children.values_mut().enumerate() {
            child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
        }
    }

    fn best_action_greedy(&self) -> usize {
        let root = self.root.as_ref().unwrap();
        // 简单的 argmax visits
        root.children.iter()
            .max_by_key(|&(_, node)| node.visits)
            .map(|(a, _)| *a)
            .unwrap_or(0)
    }
}

#[pymodule]
fn zero_mcts_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LightZeroMCTS>()?;
    Ok(())
}
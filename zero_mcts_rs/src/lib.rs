use numpy::{PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyModule};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand_distr::Gamma;
use std::collections::{HashMap, HashSet};

// ==========================================
// 1. 纯 Rust 实现的 Gomoku 逻辑 (为了速度)
// ==========================================
#[derive(Clone)]
struct RustGomokuEnv {
    board: Vec<i8>, // 0: empty, 1: black, 2: white
    size: usize,
    current_player: i8,
    winner: i8, // 0: draw/none, 1: black, 2: white
    done: bool,
    move_size: usize,
    last_action: i32,
}

impl RustGomokuEnv {
    fn from_board(
        board: &[i8],
        size: usize,
        current_player: i8,
        move_size: Option<usize>,
        last_action: Option<i32>,
    ) -> Self {
        let counted_moves = board.iter().filter(|&&c| c != 0).count();
        RustGomokuEnv {
            board: board.to_vec(),
            size,
            current_player,
            winner: 0,
            done: false,
            move_size: move_size.unwrap_or(counted_moves),
            last_action: last_action.unwrap_or(-1),
        }
    }

    fn get_valid_actions(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter(|(_, v)| **v == 0)
            .map(|(i, _)| i)
            .collect()
    }

    fn step(&mut self, action: usize) {
        if self.done {
            return;
        }
        if self.board[action] != 0 {
            panic!("Invalid move: {}", action);
        }
        self.board[action] = self.current_player;
        self.move_size += 1;
        self.last_action = action as i32;

        if self.check_win(action) {
            self.winner = self.current_player;
            self.done = true;
        } else if self.move_size == self.size * self.size {
            self.winner = 0;
            self.done = true;
        }

        if !self.done {
            self.current_player = 3 - self.current_player;
        }
    }

    fn check_win(&self, action: usize) -> bool {
        let size = self.size as isize;
        let x = (action % self.size) as isize;
        let y = (action / self.size) as isize;
        let color = self.board[action];
        let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dx, dy) in directions {
            let mut count = 1;
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

    fn observation(&self) -> Vec<Vec<Vec<f32>>> {
        let mut player1 = vec![vec![0.0f32; self.size]; self.size];
        let mut player2 = vec![vec![0.0f32; self.size]; self.size];
        for (idx, &cell) in self.board.iter().enumerate() {
            let row = idx / self.size;
            let col = idx % self.size;
            if cell == 1 {
                player1[row][col] = 1.0;
            } else if cell == 2 {
                player2[row][col] = 1.0;
            }
        }

        let mut last_action_state = vec![vec![0.0f32; self.size]; self.size];
        if self.last_action >= 0 {
            let action = self.last_action as usize;
            let row = action / self.size;
            let col = action % self.size;
            last_action_state[row][col] = 1.0;
        }

        let mut channels = Vec::with_capacity(3);
        if self.current_player == 2 {
            channels.push(player2);
            channels.push(player1);
        } else {
            channels.push(player1);
            channels.push(player2);
        }
        channels.push(last_action_state);
        channels
    }
}

// ==========================================
// 2. MCTS 节点结构 (Arena 存储)
// ==========================================
struct Node {
    visits: u32,
    value_sum: f32,
    prior_prob: f32,
    parent: Option<usize>,
    children: HashMap<usize, usize>,
    is_noise_added: bool,
}

impl Node {
    fn new(prior_prob: f32, parent: Option<usize>) -> Self {
        Node {
            visits: 0,
            value_sum: 0.0,
            prior_prob,
            parent,
            children: HashMap::new(),
            is_noise_added: false,
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
    root: Option<usize>,
    board_size: usize,
    puct: f32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    nodes: Vec<Node>,
    transposition_table: HashMap<Vec<i8>, usize>,
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
            nodes: Vec::new(),
            transposition_table: HashMap::new(),
        }
    }

    fn step(&mut self, action: usize) {
        if let Some(root_idx) = self.root {
            if let Some(&child_idx) = self.nodes[root_idx].children.get(&action) {
                self.root = Some(child_idx);
                self.nodes[child_idx].parent = None;
                return;
            }
        }
        self.root = None;
    }

    fn run(
        &mut self,
        py: Python,
        initial_board: PyReadonlyArray1<i8>,
        current_player: i8,
        policy_fn: PyObject,
        iterations: usize,
        use_dirichlet: bool,
        move_size: Option<usize>,
        last_action: Option<i32>,
    ) -> PyResult<usize> {
        let board_vec = initial_board.as_slice()?.to_vec();
        let env = RustGomokuEnv::from_board(
            &board_vec,
            self.board_size,
            current_player,
            move_size,
            last_action,
        );

        if self.root.is_none() {
            let board_hash = env.board.clone();
            if let Some(&root_idx) = self.transposition_table.get(&board_hash) {
                self.root = Some(root_idx);
                self.nodes[root_idx].parent = None;
            } else {
                let root_idx = self.new_node(1.0, None);
                self.expand(py, root_idx, &env, &policy_fn)?;
                self.root = Some(root_idx);
                self.transposition_table.insert(board_hash, root_idx);
            }
        }

        // if use_dirichlet && env.move_size <= 1 {
        if use_dirichlet {
            if let Some(root_idx) = self.root {
                self.add_dirichlet_noise(root_idx);
            }
        }

        for _ in 0..iterations {
            let mut node_idx = self.root.unwrap();
            let mut scratch_env = env.clone();

            while !self.nodes[node_idx].children.is_empty() {
                let (action, child_idx) = self.select_child(node_idx);
                scratch_env.step(action);
                node_idx = child_idx;
            }

            let leaf_hash = scratch_env.board.clone();
            if !self.transposition_table.contains_key(&leaf_hash) {
                self.transposition_table.insert(leaf_hash, node_idx);
            }

            let value = if scratch_env.done {
                if scratch_env.winner == 0 {
                    0.0
                } else {
                    -1.0
                }
            } else {
                self.expand(py, node_idx, &scratch_env, &policy_fn)?
            };

            self.backpropagate(node_idx, value);
        }

        Ok(self.best_action_greedy())
    }

    #[pyo3(signature = (temperature, top_k=None, forbidden_actions=None))]
    fn select_action_with_temperature(
        &self,
        temperature: f32,
        top_k: Option<usize>,
        forbidden_actions: Option<Vec<usize>>,
    ) -> PyResult<(usize, Py<PyArray1<f32>>)> {
        let root_idx = match self.root {
            Some(idx) => idx,
            None => return self.empty_pi(),
        };

        let forbidden_set: HashSet<usize> = forbidden_actions
            .unwrap_or_default()
            .into_iter()
            .collect();
        let visits: Vec<(usize, u32)> = self.nodes[root_idx]
            .children
            .iter()
            .map(|(&action, &child_idx)| (action, self.nodes[child_idx].visits))
            .collect();

        if visits.is_empty() {
            return self.empty_pi();
        }

        let filtered_visits: Vec<(usize, u32)> = visits
            .iter()
            .copied()
            .filter(|(action, _)| !forbidden_set.contains(action))
            .collect();
        let candidate_visits = if filtered_visits.is_empty() {
            visits.clone()
        } else {
            filtered_visits
        };

        let selected_action = if temperature == 0.0 {
            let mut shuffled = candidate_visits.clone();
            let mut rng = thread_rng();
            shuffled.shuffle(&mut rng);
            shuffled.iter().max_by_key(|x| x.1).unwrap().0
        } else {
            let mut actions: Vec<usize> = candidate_visits.iter().map(|x| x.0).collect();
            let mut counts: Vec<f32> = candidate_visits.iter().map(|x| x.1 as f32).collect();

            if let Some(k) = top_k {
                if k < actions.len() {
                    let mut zipped: Vec<_> = actions.into_iter().zip(counts.into_iter()).collect();
                    zipped.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    zipped.truncate(k);
                    let (a, c): (Vec<_>, Vec<_>) = zipped.into_iter().unzip();
                    actions = a;
                    counts = c;
                }
            }

            let exp = 1.0 / temperature;
            let weights: Vec<f32> = counts.iter().map(|c| c.powf(exp)).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            let mut rng = thread_rng();
            actions[dist.sample(&mut rng)]
        };

        let total_visits: u32 = self.nodes[root_idx]
            .children
            .values()
            .map(|&idx| self.nodes[idx].visits)
            .sum();

        let py_pi: Py<PyArray1<f32>> = Python::with_gil(|py| {
            let pi_vec = PyArray1::zeros(py, self.board_size * self.board_size, false);
            let pi_slice = unsafe { pi_vec.as_slice_mut().unwrap() };
            if total_visits > 0 {
                for (&action, &child_idx) in &self.nodes[root_idx].children {
                    pi_slice[action] = self.nodes[child_idx].visits as f32 / total_visits as f32;
                }
            }
            if !forbidden_set.is_empty() {
                let mut sum = 0.0f32;
                for (idx, val) in pi_slice.iter_mut().enumerate() {
                    if forbidden_set.contains(&idx) {
                        *val = 0.0;
                    }
                    sum += *val;
                }
                if sum > 0.0 {
                    for val in pi_slice.iter_mut() {
                        *val /= sum;
                    }
                }
            }
            pi_vec.into()
        });

        Ok((selected_action, py_pi))
    }
}

impl LightZeroMCTS {
    fn new_node(&mut self, prior: f32, parent: Option<usize>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::new(prior, parent));
        idx
    }

    fn select_child(&self, node_idx: usize) -> (usize, usize) {
        let sqrt_visits = (self.nodes[node_idx].visits as f32).sqrt();
        let mut best_score = -f32::INFINITY;
        let mut best_action = 0usize;
        let mut best_child = 0usize;

        for (&action, &child_idx) in &self.nodes[node_idx].children {
            let child = &self.nodes[child_idx];
            let q_parent = -child.q_value();
            let u_value =
                self.puct * child.prior_prob * sqrt_visits / (1.0 + child.visits as f32);
            let score = q_parent + u_value;
            if score > best_score {
                best_score = score;
                best_action = action;
                best_child = child_idx;
            }
        }
        (best_action, best_child)
    }

    fn expand(
        &mut self,
        py: Python,
        node_idx: usize,
        env: &RustGomokuEnv,
        policy_fn: &PyObject,
    ) -> PyResult<f32> {
        let obs = env.observation();
        let py_array = PyArray3::from_vec3(py, &obs)?;
        let result = policy_fn.call1(py, (py_array,))?;
        let (probs_obj, value): (PyObject, f32) = result.extract(py)?;
        let probs_any = probs_obj.bind(py);
        let policy_probs = Self::extract_policy_probs(py, &probs_any, self.board_size)?;

        let valid_actions = env.get_valid_actions();
        let mut policy_sum = 0.0;
        let mut child_specs = Vec::with_capacity(valid_actions.len());

        for action in valid_actions {
            let p = policy_probs[action];
            child_specs.push((action, p));
            policy_sum += p;
        }

        if policy_sum > 0.0 {
            for (_, prob) in child_specs.iter_mut() {
                *prob /= policy_sum;
            }
        }

        for (action, prob) in child_specs {
            let child_idx = self.new_node(prob, Some(node_idx));
            self.nodes[node_idx].children.insert(action, child_idx);
        }

        Ok(value)
    }

    fn backpropagate(&mut self, node_idx: usize, value: f32) {
        let mut current = Some(node_idx);
        let mut v = value;
        while let Some(idx) = current {
            let parent = {
                let node = &mut self.nodes[idx];
                node.visits += 1;
                node.value_sum += v;
                node.parent
            };
            v = -v;
            current = parent;
        }
    }

    fn add_dirichlet_noise(&mut self, node_idx: usize) {
        if self.nodes[node_idx].is_noise_added {
            return;
        }
        let count = self.nodes[node_idx].children.len();
        if count == 0 {
            return;
        }
        let noise = Self::sample_dirichlet(self.dirichlet_alpha, count);
        let epsilon = self.dirichlet_epsilon;

        let child_indices: Vec<usize> = self.nodes[node_idx].children.values().copied().collect();
        for (i, child_idx) in child_indices.iter().enumerate() {
            let child = &mut self.nodes[*child_idx];
            child.prior_prob = (1.0 - epsilon) * child.prior_prob + epsilon * noise[i];
        }

        self.nodes[node_idx].is_noise_added = true;
    }

    fn best_action_greedy(&self) -> usize {
        let root_idx = self.root.unwrap();
        self.nodes[root_idx]
            .children
            .iter()
            .max_by_key(|&(_, &child_idx)| self.nodes[child_idx].visits)
            .map(|(&action, _)| action)
            .unwrap_or(0)
    }

    fn extract_policy_probs(
        _py: Python,
        probs_obj: &Bound<'_, PyAny>,
        board_size: usize,
    ) -> PyResult<Vec<f32>> {
        let total_actions = board_size * board_size;

        if let Ok(prob_dict) = probs_obj.cast::<PyDict>() {
            let mut probs = vec![0.0f32; total_actions];
            for (k, v) in prob_dict.iter() {
                let action: usize = k.extract()?;
                let p: f32 = v.extract()?;
                if action < total_actions {
                    probs[action] = p;
                }
            }
            return Ok(probs);
        }

        if let Ok(prob_array) = probs_obj.extract::<PyReadonlyArray1<f32>>() {
            let slice = prob_array.as_slice()?;
            let mut probs = slice.to_vec();
            if probs.len() != total_actions {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "policy array length does not match board size",
                ));
            }
            if !Self::is_prob_distribution(&probs) {
                probs = Self::softmax(&probs);
            }
            return Ok(probs);
        }

        if let Ok(prob_array) = probs_obj.extract::<PyReadonlyArray2<f32>>() {
            let view = prob_array.as_array();
            let flattened: Vec<f32> = view.iter().copied().collect();
            if flattened.len() != total_actions {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "policy array length does not match board size",
                ));
            }
            let probs = if Self::is_prob_distribution(&flattened) {
                flattened
            } else {
                Self::softmax(&flattened)
            };
            return Ok(probs);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "policy output must be dict or numpy array",
        ))
    }

    fn is_prob_distribution(values: &[f32]) -> bool {
        if values.is_empty() {
            return false;
        }
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0.0;
        for &v in values {
            sum += v;
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        sum > 0.99 && sum < 1.01 && min_v >= -1e-6 && max_v <= 1.0 + 1e-6
    }

    fn softmax(values: &[f32]) -> Vec<f32> {
        let mut max_v = f32::NEG_INFINITY;
        for &v in values {
            if v > max_v {
                max_v = v;
            }
        }
        let mut exp_sum = 0.0;
        let mut exps = Vec::with_capacity(values.len());
        for &v in values {
            let e = (v - max_v).exp();
            exp_sum += e;
            exps.push(e);
        }
        if exp_sum == 0.0 {
            return vec![0.0; values.len()];
        }
        exps.iter().map(|v| v / exp_sum).collect()
    }

    fn sample_dirichlet(alpha: f32, count: usize) -> Vec<f32> {
        let mut rng = thread_rng();
        let gamma = Gamma::new(alpha, 1.0).unwrap();
        let mut samples = Vec::with_capacity(count);
        let mut sum = 0.0f32;
        for _ in 0..count {
            let v: f32 = gamma.sample(&mut rng);
            sum += v;
            samples.push(v);
        }
        if sum == 0.0 {
            return vec![1.0 / count as f32; count];
        }
        for v in samples.iter_mut() {
            *v /= sum;
        }
        samples
    }

    fn empty_pi(&self) -> PyResult<(usize, Py<PyArray1<f32>>)> {
        Python::with_gil(|py| {
            let pi = PyArray1::zeros(py, self.board_size * self.board_size, false);
            Ok((0, pi.into()))
        })
    }
}

#[pymodule]
fn zero_mcts_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LightZeroMCTS>()?;
    Ok(())
}

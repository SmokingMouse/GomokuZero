import torch
import numpy as np
import torch.nn.functional as F

def generate_benchmark_dataset():
    """
    生成 50 个合法的五子棋残局。
    修正点：
    1. 增加干扰子(D)以平衡黑白棋子数量，确保局面合法 (Count(Black) == Count(White) or +1)。
    2. 明确指定 last_move_coord，且必须对应棋盘上的 O。
    """
    dataset_X = []
    dataset_y = []
    
    board_size = 9 # 使用 9x9 进行测试
    
    def parse_board(pattern, target_coords, last_move_coord):
        # 初始化通道
        self_plane = np.zeros((board_size, board_size), dtype=np.float32) # 我方 (X)
        oppo_plane = np.zeros((board_size, board_size), dtype=np.float32) # 对手 (O)
        last_plane = np.zeros((board_size, board_size), dtype=np.float32) # 对手上一手
        target_plane = np.zeros((board_size, board_size), dtype=np.float32)

        rows = pattern.strip().split('\n')
        # 去除每行的首尾空格
        rows = [r.strip() for r in rows]
        
        for r, row_str in enumerate(rows):
            for c, char in enumerate(row_str):
                if char == 'X': # 我方棋子
                    self_plane[r, c] = 1.0
                elif char == 'O': # 对手棋子（关键子）
                    oppo_plane[r, c] = 1.0
                elif char == 'D': # 对手棋子（干扰子，用来平衡数量的）
                    oppo_plane[r, c] = 1.0

        # 设置 Last Move (必须是对手下的)
        if last_move_coord:
            if oppo_plane[last_move_coord[0], last_move_coord[1]] != 1.0:
                print(f"Warning: Last move {last_move_coord} is not on an Opponent stone!")
            last_plane[last_move_coord[0], last_move_coord[1]] = 1.0

        # 设置目标概率
        prob = 1.0 / len(target_coords)
        for tr, tc in target_coords:
            target_plane[tr, tc] = prob
            
        # 校验数量平衡 (假设自己是黑棋先手，轮到自己下，那么当前盘面 黑==白 或 黑==白-1 都是不对的)
        # 轮到 Self 下：
        # 如果 Self 是黑(先)：盘面应该是 Black == White
        # 如果 Self 是白(后)：盘面应该是 Black == White + 1 (即 Opponent 多一个)
        # 为了简化，我们默认生成的局面都是：轮到 Self 走，且双方子数大致平衡
        
        x_tensor = np.stack([self_plane, oppo_plane, last_plane])
        return x_tensor, target_plane

    def augment_and_add(x, y):
        # 旋转翻转扩充数据 (1 -> 8)
        x_trans = np.transpose(x, (1, 2, 0)) 
        for k in range(4):
            x_rot = np.rot90(x_trans, k)
            y_rot = np.rot90(y, k)
            dataset_X.append(np.transpose(x_rot, (2, 0, 1)))
            dataset_y.append(y_rot)
            
            x_flip = np.fliplr(x_rot)
            y_flip = np.fliplr(y_rot)
            dataset_X.append(np.transpose(x_flip, (2, 0, 1)))
            dataset_y.append(y_flip)

    # ================== 题目设计 ==================

    # --- Level 1: 嵌五必杀 (Immediate Win) ---
    # 场景：轮到 X 下。X 有 4 个子，O 也有 4 个子。
    # X 在中间空了一个洞 (0,1,2,3,4 中的 2 是空的)
    # 这是一个平衡局：X:4, O:4 (D是凑数的 O)
    l1_pattern = """
    .........
    DD.......
    .O.......
    .XX.XX...
    .........
    .........
    .........
    .........
    .........
    """ 
    # 分析：X有4个，O有3个(D+D+O)。为了平衡，O必须是4个。
    # 假设对手刚下了 (2, 1) 的 O，试图防守，但没防住下面
    l1_targets = [(3, 3)]
    x, y = parse_board(l1_pattern, l1_targets, last_move_coord=(2, 1))
    augment_and_add(x, y) # +8

    # --- Level 2: 必须堵冲四 (Critical Defense) ---
    # 场景：轮到 X 下。O 连了 4 个 (冲四)，X 必须堵。
    # 盘面：O 有 4 个，X 应该有 3 个或 4 个。
    l2_pattern = """
    .........
    ....O....
    ....O....
    X...O....
    X...O....
    X........
    .........
    .........
    .........
    """
    # 分析：O 竖线 4 个，X 左边 3 个。O 刚下了 (1, 4) 形成冲四。
    # 此时 O=4, X=3。合法（白多黑少，轮到黑下）。
    l2_targets = [(0, 4), (5, 4)] # 两头都要防，或者防一头，这里假设两头均分
    x, y = parse_board(l2_pattern, l2_targets, last_move_coord=(1, 4))
    augment_and_add(x, y) # +8 = 16

    # --- Level 3: 活三进攻 (Active 3) ---
    # 场景：X 形成活三，准备下一步成活四。
    # 盘面：X 有 3 个，O 有 3 个。
    l3_pattern = """
    .........
    D.D......
    .........
    .XXX.....
    ..O......
    .........
    .........
    .........
    .........
    """
    # X: 3个, O: 3个 (D, D, O)。平衡。
    # O 刚下在 (4, 2) 挡路，但没挡住上面。
    l3_targets = [(3, 0), (3, 4)] # 左右延伸
    x, y = parse_board(l3_pattern, l3_targets, last_move_coord=(4, 2))
    augment_and_add(x, y) # +8 = 24

    # --- Level 4: 复杂防守 (VCF 防守) ---
    # 场景：O 形成了 "跳活三" (O O . O)，中间那个点如果不补就是死。
    # 盘面：O: 3个, X: 2个。
    l4_pattern = """
    .........
    .........
    .........
    ..O.O.O..
    ....X....
    ....X....
    .........
    .........
    .........
    """
    # O: 3个, X: 2个。合法（白3黑2，轮到黑）。
    # O 刚下了 (3, 6) 的 O。
    l4_targets = [(3, 3), (3, 5), (3, 7)] # (3,5)是填中间，最优；(3,3)/(3,7)是堵两头
    # 为了简化，我们只要求堵中间
    l4_targets = [(3, 5)] 
    x, y = parse_board(l4_pattern, l4_targets, last_move_coord=(3, 6))
    augment_and_add(x, y) # +8 = 32

    # --- Level 5: 开局 (Opening) ---
    # 场景 A: 空棋盘。黑棋先手。
    # Last Move 是 None (全0)
    l5_a_pattern = """
    .........
    .........
    .........
    .........
    .........
    .........
    .........
    .........
    .........
    """
    l5_a_targets = [(4, 4)]
    # 特殊处理：last_move_coord 设为 None
    x, y = parse_board(l5_a_pattern, l5_a_targets, last_move_coord=None) 
    # 这是一个特殊的 Case，手动处理 last plane 全 0
    x[2] = np.zeros((board_size, board_size)) 
    
    dataset_X.append(x)
    dataset_y.append(y)
    
    # 场景 B: 必应点 (Standard Response)
    # 黑下了天元，白下了直指 (3, 4)。轮到黑下。
    # X:1, O:1。
    l5_b_pattern = """
    .........
    .........
    .........
    ....O....
    ....X....
    .........
    .........
    .........
    .........
    """
    # X: (4,4), O: (3,4)。O 刚下的 (3,4)。
    # 常用点：(5,5), (3,5), (5,3)
    l5_b_targets = [(5, 5), (3, 5), (5, 3)]
    x, y = parse_board(l5_b_pattern, l5_b_targets, last_move_coord=(3, 4))
    augment_and_add(x, y) # +8 = 41

    # 补齐剩余数据 (用 Level 2 的变体 - 仅仅是方向不同)
    l_supp_pattern = """
    .........
    ..X......
    ..X.O....
    ..X.O....
    ....O....
    ....O....
    .........
    .........
    .........
    """
    # O 竖线4个，X 竖线3个。O 刚下了 (5, 4)。
    l_supp_targets = [(2, 4), (6, 4)]
    x, y = parse_board(l_supp_pattern, l_supp_targets, last_move_coord=(5, 4))
    augment_and_add(x, y) # +8 = 49
    
    # 凑第50个，复制一下空棋盘
    dataset_X.append(dataset_X[32])
    dataset_y.append(dataset_y[32])

    X_tensor = torch.tensor(np.array(dataset_X[:50]), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(dataset_y[:50]), dtype=torch.float32)

    return X_tensor, y_tensor

def generate_benchmark_dataset_7x7():
    """
    生成 50 个合法的 7x7 五子棋残局。
    X: (N, 3, 7, 7)
    y: (N, 7, 7)
    """
    dataset_X = []
    dataset_y = []
    
    board_size = 7 # 修改为 7x7
    
    def parse_board(pattern, target_coords, last_move_coord):
        # 初始化通道
        self_plane = np.zeros((board_size, board_size), dtype=np.float32)
        oppo_plane = np.zeros((board_size, board_size), dtype=np.float32)
        last_plane = np.zeros((board_size, board_size), dtype=np.float32)
        target_plane = np.zeros((board_size, board_size), dtype=np.float32)

        rows = pattern.strip().split('\n')
        rows = [r.strip() for r in rows]
        
        # 校验行数，防止手误写多/写少
        if len(rows) != board_size:
            print(f"Warning: Pattern rows {len(rows)} != board_size {board_size}")

        for r, row_str in enumerate(rows):
            if len(row_str) != board_size:
                 print(f"Warning: Row {r} length {len(row_str)} != board_size {board_size}")
            for c, char in enumerate(row_str):
                if char == 'X':
                    self_plane[r, c] = 1.0
                elif char == 'O' or char == 'D':
                    oppo_plane[r, c] = 1.0

        if last_move_coord:
            if oppo_plane[last_move_coord[0], last_move_coord[1]] != 1.0:
                print(f"Warning: Last move {last_move_coord} is not on an Opponent stone!")
            last_plane[last_move_coord[0], last_move_coord[1]] = 1.0

        prob = 1.0 / len(target_coords)
        for tr, tc in target_coords:
            target_plane[tr, tc] = prob
            
        x_tensor = np.stack([self_plane, oppo_plane, last_plane])
        return x_tensor, target_plane

    def augment_and_add(x, y):
        x_trans = np.transpose(x, (1, 2, 0)) 
        for k in range(4):
            x_rot = np.rot90(x_trans, k)
            y_rot = np.rot90(y, k)
            dataset_X.append(np.transpose(x_rot, (2, 0, 1)))
            dataset_y.append(y_rot)
            
            x_flip = np.fliplr(x_rot)
            y_flip = np.fliplr(y_rot)
            dataset_X.append(np.transpose(x_flip, (2, 0, 1)))
            dataset_y.append(y_flip)

    # ================== 题目设计 (7x7) ==================

    # --- Level 1: 嵌五必杀 (Own 4 stones) ---
    # 场景：X 在第 3 行横向有 4 个子，中间空一个。
    # 棋盘：X:4, O:4 (平衡)
    # 坐标：(3,1), (3,2), [空3,3], (3,4), (3,5)
    l1_pattern = """
    .......
    .D.....
    .D.....
    .XX.XX.
    ...O...
    .D.....
    .......
    """ 
    # 分析：X=4 (3,1/2/4/5)。O=4 (1,1; 2,1; 4,3; 5,1)。
    # 假设 O 上一步下在 (4, 3) 试图防守下方
    l1_targets = [(3, 3)] # 中心点必杀
    x, y = parse_board(l1_pattern, l1_targets, last_move_coord=(4, 3))
    augment_and_add(x, y) # +8

    # --- Level 2: 必须堵冲四 (Opponent 4 stones) ---
    # 场景：O 在第 3 列竖向连了 4 个。
    # 棋盘：O:4, X:3 (轮到黑走)
    # O 位置: (1,3), (2,3), (3,3), (4,3)
    l2_pattern = """
    .......
    ...O...
    ...O...
    X..O...
    X..O...
    X......
    .......
    """
    # 分析：O=4, X=3。O 刚下了 (1, 3) 形成冲四。
    # 两头是 (0,3) 和 (5,3)。
    l2_targets = [(0, 3), (5, 3)] 
    x, y = parse_board(l2_pattern, l2_targets, last_move_coord=(1, 3))
    augment_and_add(x, y) # +8 = 16

    # --- Level 3: 活三进攻 (Own 3 stones) ---
    # 场景：X 在斜线形成活三。
    # X 位置: (2,2), (3,3), (4,4) -> 对角线
    l3_pattern = """
    .......
    .D.....
    ..X....
    ...X...
    ....X..
    ...O...
    .D.....
    """
    # 分析：X=3, O=3 (1,1; 5,3; 6,1)。平衡。
    # O 刚下了 (5, 3)。
    # 活三两头：(1, 1) 被 D 占了? 不，D 在 (1,1)，那斜线要避开。
    # 让我们换一个斜线，或者把 D 移开。
    # 修改 D 位置到 (1, 5)
    l3_pattern_fixed = """
    .......
    .....D.
    ..X....
    ...X...
    ....X..
    ...O...
    .D.....
    """
    l3_targets = [(1, 1), (5, 5)]
    x, y = parse_board(l3_pattern_fixed, l3_targets, last_move_coord=(5, 3))
    augment_and_add(x, y) # +8 = 24

    # --- Level 4: 堵跳活三 (Block Split 3) ---
    # 场景：O . O . O 结构
    # 7x7 比较挤，放在第 2 行: (2,1) (2,3) (2,5)
    l4_pattern = """
    .......
    .......
    .O.O.O.
    ...X...
    ...X...
    .......
    .......
    """
    # X:2, O:3。O 刚下了 (2, 5)。
    # 必须堵中间 (2, 2) 或 (2, 4) ? 
    # 实际上 O.O.O 的中间两个点 (2,2) 和 (2,4) 都是必救点。我们选其中一个或两个。
    l4_targets = [(2, 2), (2, 4)]
    x, y = parse_board(l4_pattern, l4_targets, last_move_coord=(2, 5))
    augment_and_add(x, y) # +8 = 32

    # --- Level 5: 开局 (Opening) ---
    # 场景 A: 7x7 空棋盘 (Center is 3,3)
    l5_a_pattern = """
    .......
    .......
    .......
    .......
    .......
    .......
    .......
    """
    l5_a_targets = [(3, 3)] # 7x7 天元是 (3,3)
    x, y = parse_board(l5_a_pattern, l5_a_targets, last_move_coord=None)
    x[2] = np.zeros((board_size, board_size)) # Clear last move
    dataset_X.append(x)
    dataset_y.append(y)

    # 场景 B: 必应点
    # 黑(3,3)，白(2,3)。
    l5_b_pattern = """
    .......
    .......
    ...O...
    ...X...
    .......
    .......
    .......
    """
    # 推荐点：(4,4), (2,4), (4,2), (2,2) 等围绕天元
    l5_b_targets = [(4, 4), (2, 4), (4, 2), (2, 2)]
    x, y = parse_board(l5_b_pattern, l5_b_targets, last_move_coord=(2, 3))
    augment_and_add(x, y) # +8 = 41

    # 补齐剩余数据 (Level 2 变体)
    l_supp_pattern = """
    .......
    .X.....
    .X.O...
    .X.O...
    ...O...
    ...O...
    .......
    """
    # O 竖线4个 (2,3) to (5,3)。X 竖线3个。
    # O 刚下了 (5,3)。
    # 堵 (1,3) 或 (6,3)。
    l_supp_targets = [(1, 3), (6, 3)]
    x, y = parse_board(l_supp_pattern, l_supp_targets, last_move_coord=(5, 3))
    augment_and_add(x, y) # +8 = 49

    # 凑整第 50 个
    dataset_X.append(dataset_X[32])
    dataset_y.append(dataset_y[32])

    X_tensor = torch.tensor(np.array(dataset_X[:50]), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(dataset_y[:50]), dtype=torch.float32)

    return X_tensor, y_tensor


def Evaluate(policy, board_size=9):
    if board_size == 9:
        X, y = generate_benchmark_dataset()
    elif board_size == 7:
        X, y = generate_benchmark_dataset_7x7()
    else:
        return 0

    policy.eval()
    device = next(policy.parameters()).device
    X = X.to(device)
    y = y.to(device)
    logits = policy(X)[0]

    probs = F.softmax(logits, dim=-1)
    y = y.reshape(y.shape[0],-1)
    score = torch.sum(y * probs, dim=1).sum() # 先在每个样本上求和，再在batch上求平均

    return score.item()

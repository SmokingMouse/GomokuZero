# 注意：这里 import 的名字就是你文件夹的名字（除非你在 Cargo.toml 改了）
import zero_mcts_rs

# 调用默认生成的测试函数
result = zero_mcts_rs.sum_as_string(5, 10)
print(f"Rust 算出来的结果: {result}")
print(f"类型是: {type(result)}")
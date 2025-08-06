# python player.py --itermax 400 --model1 gomoku_zero_resnet_play30/policy_step_4000.pth --model2 gomoku_zero_freqency/policy_step_2500.pth --pure_mcts_iter=0
# python player.py --itermax 100 --model1 gomoku_zero_coslrscheduler/policy_step_19500.pth --model2 gomoku_zero_freqency/policy_step_2500.pth --pure_mcts_iter=0
# python player.py --itermax 400 --model1 gomoku_zero_coslrscheduler/policy_step_19500.pth --model2 gomoku_zero_coslrscheduler/policy_step_5000.pth  --pure_mcts_iter=0
# python player.py --itermax 400 --model1 gomoku_zero_coslrscheduler/policy_step_5000.pth --model2 gomoku_zero_multisteplr/policy_step_15000.pth 
python player.py --itermax 400 --model2 gomoku_zero_resnet_play30/policy_step_4000.pth --model1 gomoku_zero_multisteplr/policy_step_15000.pth --pure_mcts_iter=4000
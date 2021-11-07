import torch

# Smart Warehouse configuration
warehouse_cap_conveyor = 10
warehouse_cap_order = 25
warehouse_default_c = [0, 1, 2, 2]
warehouse_num_r = 3
warehouse_ratio_item = [[6, 1, 1, 2],
                        [3, 3, 1, 3],
                        [1, 6, 1, 2],
                        [1, 3, 3, 3],
                        [1, 1, 6, 2],
                        [3, 1, 3, 3]]
warehouse_ratio_duration = 50
warehouse_tick_per_in = 2

# Smart warehouse anomaly configuration
anomaly_duration = 200
anomaly_after = 100
anomaly_mtth = 100

# Environment Configuration

# Simulation Configuration
sim_count = 1
sim_targets = ['A-RL-RL', 'RL-RL', 'A-RL', 'RL', 'Default']
sim_total_ticks = 1000
sim_tqdm_on = True

# Pytorch configuration
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL configuration
rl_batch_size = 128
rl_episodes = 50
rl_epsilon_start = 0.9
rl_epsilon_end = 0.05
rl_epsilon_decay = 200
rl_gamma = 0.999
rl_hidden_layer = 512
rl_learning_rate = 0.001
rl_memory_size = 10000

# Calculations, Assertions
warehouse_num_types = len(warehouse_default_c)
rl_input_size = warehouse_num_r * warehouse_cap_conveyor + warehouse_cap_order + 1
rl_output_size = warehouse_num_r * (1 + warehouse_num_r)
assert min(warehouse_default_c) >= 0
assert max(warehouse_default_c) < warehouse_num_types
for i in range(len(warehouse_ratio_item)):
    assert len(warehouse_ratio_item[i]) == warehouse_num_types
    assert warehouse_ratio_duration % sum(warehouse_ratio_item[i]) == 0

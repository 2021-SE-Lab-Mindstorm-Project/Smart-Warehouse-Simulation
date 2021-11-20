import torch

cap_conveyor = 5
cap_wait = 5

reward_order = 100
reward_trash = 50
reward_wait = 1

order_total = 100
order_delay = 0

anomaly_mtbf = 5
anomaly_duration = 50

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim_targets = [
    'AD-RL',
    'ORL',
    'Random'
]
sim_count = 50

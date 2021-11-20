import torch

cap_conveyor = 5
cap_wait = 5

reward_order = 25
reward_trash = 100
reward_wait = 1

order_total = 20
order_delay = 0

anomaly_mtbf = 2
anomaly_duration = 10

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim_targets = [
    'AD-RL',
    'ORL',
    'Random'
]
sim_count = 50

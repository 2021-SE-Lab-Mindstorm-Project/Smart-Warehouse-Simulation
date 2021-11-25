import torch

cap_conveyor = 100
cap_wait = 10

reward_order = 30
reward_trash = 70
reward_wait = 1

order_total = 100000
order_delay = 0

anomaly_mtbf = 10000000
anomaly_duration = 1000

cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim_targets = [
    'AAAA',
    'ORL',
    'Random'
]
sim_count = 100

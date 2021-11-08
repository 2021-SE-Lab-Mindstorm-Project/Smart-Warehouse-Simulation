import math
import random
from collections import deque, namedtuple

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

import anomaly
import config
import items
import warehouse

Transition = namedtuple('Transition', ('state', 'tactic', 'reward', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=config.rl_memory_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size, is_anomaly_aware=False, hidden_layer=config.rl_hidden_layer, path=''):
        super(DQN, self).__init__()
        self.is_anomaly_aware = is_anomaly_aware
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, self.output_size)
        )

        self.steps = -1
        self.memory = Memory()
        self.optimizer = optim.RMSprop(self.parameters())

        if path != '':
            self.load_state_dict(torch.load(path))
            self.eval()

    def select_tactic(self, state, available=None):
        if available is None:
            available = [True] * self.output_size

        sorted_result = self.model(state).sort()
        i = 0
        while i < self.output_size:
            if available[sorted_result.indices[0][i]]:
                return sorted_result.indices[0][i]
            i += 1

    def select_train_tactic(self, state, available=None):
        if available is None:
            available = [True] * self.output_size

        self.steps += 1
        sample = random.random()
        eps_threshold = config.rl_epsilon_end + (config.rl_epsilon_start - config.rl_epsilon_end) * math.exp(
            -1. * self.steps / config.rl_epsilon_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                self.select_tactic(state, available)

        available_range = []
        for i in range(self.output_size):
            if available[i]:
                available_range.append(i)
        return torch.LongTensor([[random.choice(available_range)]]).to(config.cuda_device)

    def optimize_model(self):
        if len(self.memory) < config.rl_batch_size:
            return

        transitions = self.memory.sample(config.rl_batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.cuda_device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)

        selected_tactics = self.model(state_batch).gather(1, tactic_batch)
        next_state_values = torch.zeros(config.rl_batch_size, device=config.cuda_device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
        expected_values = (next_state_values * config.rl_gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def push_optimize(self, state, tactic_tensor, reward, next_state):
        if next_state is None:
            self.memory.push(torch.FloatTensor([state]).to(config.cuda_device), tactic_tensor,
                             torch.FloatTensor([reward]).to(config.cuda_device), None)
        else:
            self.memory.push(torch.FloatTensor([state]).to(config.cuda_device), tactic_tensor,
                             torch.FloatTensor([reward]).to(config.cuda_device),
                             torch.FloatTensor([next_state]).to(config.cuda_device))

        self.optimize_model()


def train_one_step(rl_model, tick, target, item_list, orders, reward, anomalies=None):
    if rl_model.is_anomaly_aware:
        if anomalies is not None:
            current_anomaly = anomaly.get_anomaly(tick, anomalies)
        else:
            current_anomaly = [0, 0]
        state = [*current_anomaly, *target.model_state()]

    else:
        state = target.model_state()

    state_tensor = torch.FloatTensor([state]).to(config.cuda_device)

    tactic = rl_model.select_train_tactic(state_tensor, target.available_tactic(True))

    c_tactic = int(torch.round(tactic / (config.warehouse_num_r + 1)))
    r_tactic = int(tactic % (config.warehouse_num_r + 1))

    old_num_orders = len(target.orders)
    target, c, s = warehouse.run_warehouse(tick, target, c_tactic, r_tactic, item_list, orders, anomalies)
    if len(target.orders) == old_num_orders:
        reward += 100

    reward -= len(target.orders)

    # if target.is_ended():
    #     reward -= 1000000
    #     self.push_optimize(state, tactic, reward, None)
    # else:
    #     self.push_optimize(state, tactic, reward, [tick + 1, *target.model_state()])

    if rl_model.is_anomaly_aware:
        if anomalies is not None:
            current_anomaly = anomaly.get_anomaly(tick + 1, anomalies)
        else:
            current_anomaly = [0, 0]
        next_state = [*current_anomaly, *target.model_state()]

    else:
        next_state = target.model_state()

    rl_model.push_optimize(state, tactic, reward, next_state)

    return target, reward


def train_rl(rl_model):
    train_tqdm = tqdm(range(config.rl_episodes))
    with plt.ion():
        for _ in train_tqdm:
            target = warehouse.Warehouse()
            reward = 0
            item_list = items.generate_item_list(False)
            orders = items.generate_item_list(True)
            if rl_model.is_anomaly_aware:
                anomalies = anomaly.generate_anomaly()
            else:
                anomalies = None

            for tick in range(config.sim_total_ticks):
                target, reward = train_one_step(rl_model, tick, target, item_list, orders, reward, anomalies)

                # if target.is_ended():
                #     break

            train_tqdm.set_description('reward: %d' % reward)

    if rl_model.is_anomaly_aware:
        torch.save(rl_model.state_dict(), 'model/a_rl.pth')
    else:
        torch.save(rl_model.state_dict(), 'model/rl.pth')


def train_one_step_rl_rl(c_model, r_model, tick, target, item_list, orders, reward, anomalies=None):
    reward = 0
    if c_model.is_anomaly_aware:
        current_anomaly = anomaly.get_anomaly(tick, anomalies)

        c_state = [*current_anomaly, *target.model_state('c')]
        r_state = [*current_anomaly, *target.model_state('r')]

    else:
        c_state = target.model_state('c')
        r_state = target.model_state('r')

    c_state_tensor = torch.FloatTensor([c_state]).to(config.cuda_device)
    r_state_tensor = torch.FloatTensor([r_state]).to(config.cuda_device)

    c_av, r_av = target.available_tactic()
    c_tactic = c_model.select_train_tactic(c_state_tensor, c_av)
    r_tactic = r_model.select_train_tactic(r_state_tensor, r_av)

    target, c, s = warehouse.run_warehouse(tick, target, c_tactic, r_tactic, item_list, orders, anomalies)
    if c:
        reward += config.warehouse_cap_conveyor
    if s:
        reward += config.warehouse_cap_order

    reward -= len(target.c_items)
    reward -= len(target.orders)

    # if target.is_ended():
    #     reward -= 1000000
    #     self.push_optimize(state, tactic, reward, None)
    # else:
    #     self.push_optimize(state, tactic, reward, [tick + 1, *target.model_state()])

    if c_model.is_anomaly_aware:
        current_anomaly = anomaly.get_anomaly(tick, anomalies)

        c_next_state = [*current_anomaly, *target.model_state('c')]
        r_next_state = [*current_anomaly, *target.model_state('r')]

    else:
        c_next_state = target.model_state('c')
        r_next_state = target.model_state('r')

    c_model.push_optimize(c_state, c_tactic, reward, c_next_state)
    r_model.push_optimize(r_state, r_tactic, reward, r_next_state)

    return target, reward


def train_rl_rl(c_model, r_model):
    train_tqdm = range(config.rl_episodes)
    with plt.ion():
        for i in train_tqdm:
            target = warehouse.Warehouse()
            reward = 0
            item_list = items.generate_item_list(False)
            orders = items.generate_item_list(True)
            if c_model.is_anomaly_aware:
                anomalies = anomaly.generate_anomaly()
            else:
                anomalies = None

            total = 0
            for tick in range(config.sim_total_ticks):
                target, reward = train_one_step_rl_rl(c_model, r_model, tick, target, item_list, orders, reward,
                                                      anomalies)
                total += reward

                # if target.is_ended():
                #     break

            print('episode: %d reward: %d processed: %d average: %.2f' % (i, total, target.processed_order,
                                                                          target.average_time))

    if c_model.is_anomaly_aware:
        torch.save(c_model.state_dict(), 'model/a_rl_c.pth')
        torch.save(r_model.state_dict(), 'model/a_rl_r.pth')
    else:
        torch.save(c_model.state_dict(), 'model/rl_c.pth')
        torch.save(r_model.state_dict(), 'model/rl_r.pth')


if __name__ == "__main__":
    # RL
    # rl_model = DQN(config.rl_input_size, config.rl_output_size).to(config.cuda_device)
    # train_rl(rl_model)
    #
    # rl_model = DQN(config.rl_input_size + 1, config.rl_output_size, True).to(config.cuda_device)
    # train_rl(rl_model)

    # RL-RL
    c_model = DQN(config.warehouse_num_r + 1, config.warehouse_num_r + 1).to(config.cuda_device)
    r_model = DQN(config.warehouse_num_r, config.warehouse_num_r + 1).to(config.cuda_device)
    train_rl_rl(c_model, r_model)

    # c_model = DQN(config.warehouse_num_r + 3, config.warehouse_num_r + 1, True).to(config.cuda_device)
    # r_model = DQN(config.warehouse_num_r + 2, config.warehouse_num_r + 1, True).to(config.cuda_device)
    # train_rl_rl(c_model, r_model)

import math
import random
from collections import deque, namedtuple

import torch
from torch import nn, optim

import config
import smart_warehouse

Transition = namedtuple('Transition', ('state', 'tactic', 'reward', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = deque(maxlen=10000)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, anomaly_aware, path=''):
        super(DQN, self).__init__()
        self.anomaly_aware = anomaly_aware
        self.input_size = 11 if anomaly_aware else 10
        self.output_size = 3
        self.hidden_size = 512

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

        self.steps = -1
        self.memory = Memory()
        self.optimizer = optim.RMSprop(self.parameters())

        if path != '':
            self.load_state_dict(torch.load(path, map_location=config.cuda_device))
            self.eval()

    def select_tactic(self, state, available):
        state_tensor = torch.FloatTensor([state]).to(config.cuda_device)

        result = self.model(state_tensor).view(self.output_size)
        result_sort = result.sort(descending=True)

        for i in range(self.output_size):
            selected = result_sort.indices[i]
            if available[selected]:
                return torch.LongTensor([[selected]]).to(config.cuda_device)

    def select_train_tactic(self, state, available):
        sample = random.random()
        eps_threshold = 0.05 + 0.85 * math.exp(-1. * self.steps / 200)
        self.steps += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.select_tactic(state, available)

        candidate = []
        for i in range(len(available)):
            if available[i]:
                candidate.append(i)
        return torch.LongTensor([[random.choice(candidate)]]).to(config.cuda_device)

    def optimize_model(self):
        if len(self.memory) < 128:
            return

        transitions = self.memory.sample(128)
        batch = Transition(*zip(*transitions))

        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        tactic_batch = torch.cat(batch.tactic)
        reward_batch = torch.cat(batch.reward)

        selected_tactics = self.model(state_batch).gather(1, tactic_batch)
        next_state_values = self.model(next_state_batch).min(1)[0].detach()
        expected_values = (next_state_values * 0.99) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_tactics, expected_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def push_optimize(self, state, tactic_tensor, reward, next_state):
        self.memory.push(torch.FloatTensor([state]).to(config.cuda_device), tactic_tensor,
                         torch.FloatTensor([reward]).to(config.cuda_device),
                         torch.FloatTensor([next_state]).to(config.cuda_device))
        self.optimize_model()


def train(anomaly_aware):
    model = DQN(anomaly_aware).to(config.cuda_device)
    for i in range(150):
        orders = smart_warehouse.generate_order_list()
        anomaly = smart_warehouse.generate_anomaly(is_real=True) if anomaly_aware else [[], [], []]

        target = smart_warehouse.Warehouse(orders, anomaly, anomaly_aware)

        tick = 0
        while tick < config.order_total or target.get_order() > 0:
            decided = False
            decision = 3
            old_state = target.get_state()
            if target.need_decision():
                decided = True
                available = target.available()
                decision = model.select_train_tactic([tick, *old_state], available)

            reward = target.run(tick, decision)
            tick += 1

            new_state = target.get_state()
            if decided:
                model.push_optimize([tick, *old_state], decision, reward, [tick + 1, *new_state])

        print('episode: %d reward: %d processed: %d' % (i, target.reward, tick))

    path = 'model/a_rl.pth' if anomaly_aware else 'model/rl.pth'
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    train(False)
    train(True)

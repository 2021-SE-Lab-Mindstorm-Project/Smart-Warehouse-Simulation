import csv
import math
import random

import config
import reinforcement


def generate_anomaly(is_real=True, name=None):
    anomalies = []
    for i in range(3):
        if not is_real or i == 0 or i == 2:
            anomaly = []
            count = 1
            tick = 1
            while tick < config.order_total * 100:
                prob = 1 - math.e ** (-count / config.anomaly_mtbf ** 2)
                if random.random() < prob:
                    anomaly.append(tick)
                    count = 0

                count += 1
                tick += 1

            anomaly.sort()
            anomalies.append(anomaly)
        else:
            anomalies.append([])

    if name is not None:
        with open('log/anomaly/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(3):
                for anomaly in anomalies[i]:
                    csv_writer.writerow([anomaly, i])

    return anomalies


def generate_order_list(name=None):
    ans = []
    for i in range(config.order_total):
        ans.append(i % 4 + 1)

    if name is not None:
        with open('log/order/' + name + '.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for i in range(len(ans)):
                csv_writer.writerow([i + config.order_delay, ans[i]])

    return ans


def repr_list(conveyor):
    ans = 0
    for i, item in enumerate(conveyor):
        ans += item * (5 ** (config.cap_conveyor - i - 1))
    return ans


class Warehouse:
    def __init__(self, orders, anomaly, anomaly_aware):
        self.anomaly_aware = anomaly_aware

        self.c = []
        self.r = [[], [], []]
        self.s = []

        self.r_wait = [0] * 3
        self.s_wait = 0

        self.o_r = [0] * 4
        self.o_s = [0] * 4

        self.orders = orders
        self.anomaly = anomaly
        self.count = [0] * 3
        self.current_anomaly = [-1] * 3
        self.stuck = [0] * 3

        self.reward = 0

    def __str__(self):
        ans = 'C:' + str(self.c) + '\n'
        for i in range(3):
            ans += 'R' + str(i) + ':' + str(self.r[i])
            if self.current_anomaly[i] != -1:
                ans += ' stuck'
            ans += '\n'
        ans += 'S:' + str(self.s) + '\n'
        ans += 'O:' + str(self.get_order(False)) + ' sum: ' + str(self.get_order(True)) + '\n'
        return ans

    def __repr__(self):
        return str(self)

    def need_decision(self):
        if len(self.c) == 0:
            return False

        num_true = 0
        for ans in self.available():
            if ans:
                num_true += 1

        return num_true > 1

    def available(self, i=None):
        if i is not None:
            ans = len(self.r[i]) < config.cap_conveyor
            if not self.anomaly_aware:
                return ans
            return ans and self.current_anomaly[i] == -1

        ans = []
        for i in range(3):
            single_ans = len(self.r[i]) < config.cap_conveyor
            if not self.anomaly_aware:
                ans.append(single_ans)
            else:
                ans.append(single_ans and self.current_anomaly[i] == -1)
        return ans

    def get_available(self):
        available = self.available()
        ans = []
        for i, avail in enumerate(available):
            if avail:
                ans.append(i)
        return ans

    def get_inventory(self, item):
        ans = self.c.count(item)
        for i in range(3):
            ans += self.r[i].count(item)
        ans += self.s.count(item)
        return ans

    def get_order(self, is_sum=True):
        orders = []
        for i in range(4):
            orders.append(self.o_r[i] + self.o_s[i])

        if is_sum:
            return sum(orders)
        return orders

    def get_state(self):
        ans = []
        for i in range(3):
            ans.append(repr_list(self.r[i]))
        ans.append(repr_list(self.s))
        ans.extend(self.get_order(False))

        if self.anomaly_aware:
            anomaly_number = 0
            for i, anomaly in enumerate(self.current_anomaly):
                if anomaly != -1:
                    anomaly_number += (2 ** i)
            ans.append(anomaly_number)

        return ans

    def run(self, tick, decision):
        reward = 0

        c_moving = 0
        r_moving = [0] * 3

        # Move c to r
        if decision != 3 and self.available(decision) and len(self.c) != 0:
            c_moving = self.c.pop(0)

        # Move r to s
        for i in [1, 0, 2]:
            if self.current_anomaly[i] == -1 and len(self.r[i]) != 0 and len(self.s) < config.cap_conveyor:
                target_item = self.r[i][0]
                if self.o_r[target_item - 1] != 0:
                    r_moving[i] = target_item
                    self.r[i].pop(0)
                    self.o_r[target_item - 1] -= 1
                    self.o_s[target_item - 1] += 1
                    self.count[i] += 1
                    self.r_wait[i] = 0
                elif self.r_wait[i] > config.cap_wait:
                # elif tick > config.order_total + config.order_delay + 1:
                    r_moving[i] = target_item
                    self.r[i].pop(0)
                    self.r_wait[i] = 0
                else:
                    self.r_wait[i] += 1

        # Make anomaly
        for i in [0, 2]:
            if self.current_anomaly[i] == -1 and len(self.anomaly[i]) != 0 and self.count[i] == self.anomaly[i][0]:
                self.current_anomaly[i] = tick
                self.anomaly[i].pop(0)

                self.stuck[i] = r_moving[i]
                self.o_s[r_moving[i] - 1] -= 1
                self.o_r[r_moving[i] - 1] += 1
                r_moving[i] = 0

        # Solve anomaly
        for i in [0, 2]:
            if self.current_anomaly[i] != -1 and self.current_anomaly[i] + config.anomaly_duration < tick:
                self.current_anomaly[i] = -1

                target_item = self.stuck[i]
                r_moving[i] = target_item
                if self.o_r[target_item - 1] != 0:
                    self.o_r[target_item - 1] -= 1
                    self.o_s[target_item - 1] += 1

        # s
        if len(self.s) != 0:
            target_item = self.s[0]
            if self.o_s[target_item - 1] != 0:
                self.o_s[target_item - 1] -= 1
                self.s.pop(0)
                reward += config.reward_order
                self.s_wait = 0
            # elif tick > config.order_total + config.order_delay:
            elif self.s_wait > config.cap_wait:
                self.s.pop(0)
                reward -= config.reward_trash
                self.s_wait = 0
            else:
                self.s_wait += 1

        # Move Items
        if c_moving != 0:
            self.r[decision].append(c_moving)
        for i in [1, 0, 2]:
            if r_moving[i] != 0:
                self.s.append(r_moving[i])

        # Reward Check
        reward -= self.get_order() * config.reward_wait

        # Add Item
        for i in range(1, 5):
            if self.get_inventory(i) < self.get_order(False)[i - 1]:
                for j in range(5):
                    self.c.append(i)

        # Add Order
        if config.order_delay <= tick < config.order_total + config.order_delay:
            target_order = self.orders[tick - config.order_delay]
            self.o_r[target_order - 1] += 1

        self.reward += reward
        return reward


def decision_making_rl(warehouse, model, tick):
    return model.select_tactic([tick, *warehouse.get_state()], warehouse.available())


def run(name, dm_type, orders, anomalies):
    with open('log/warehouse/' + name + '_' + dm_type + '.txt', 'w', newline='') as log_file:
        with open('log/dm/' + name + '_' + dm_type + '.csv', 'w', newline='') as dm_file:
            dm_writer = csv.writer(dm_file)

            if dm_type == 'ORL':
                rl_model = reinforcement.DQN(False, path='model/rl.pth').to(config.cuda_device)
            if dm_type == 'AD-RL':
                rl_model = reinforcement.DQN(True, path='model/a_rl.pth').to(config.cuda_device)

            warehouse = Warehouse(orders, anomalies, True if dm_type == 'AD-RL' else False)
            tick = 0

            while tick < config.order_delay + config.order_total or warehouse.get_order() != 0:
                decided = False
                decision = 3
                if dm_type == 'ORL' or dm_type == 'AD-RL':
                    if warehouse.need_decision():
                        decided = True
                        old_state = warehouse.get_state()
                        decision = decision_making_rl(warehouse, rl_model, tick)
                    else:
                        avail = warehouse.get_available()
                        if len(avail) != 0:
                            decision = avail[0]

                else:
                    candidate = warehouse.get_available()
                    if len(candidate) != 0:
                        decision = random.choice(candidate)

                dm_writer.writerow([tick, int(decision)])
                reward = warehouse.run(tick, decision)

                if dm_type == 'ORL' and decided:
                    rl_model.push_optimize([tick, *old_state], decision, reward, [tick + 1, *warehouse.get_state()])

                log_file.write('tick: ' + str(tick))
                log_file.write(str(warehouse))
                log_file.write('\n')

                tick += 1

    return warehouse.reward, tick

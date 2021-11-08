import copy
import random

import torch

import anomaly
import config
import reinforcement_learning


class Warehouse:
    def __init__(self):
        self.c_items = []
        self.r_items = [[] for _ in range(config.warehouse_num_r)]
        self.s_items = []
        self.orders = []
        self.processed_time = 0
        self.processed_order = 0
        self.average_time = 0

    def __str__(self):
        str_state = 'classification: ' + str(self.c_items) + '\n'
        for i in range(config.warehouse_num_r):
            str_state += 'repository' + str(i) + ': ' + str(self.r_items[i]) + '\n'
        str_state += 'shipment: ' + str(self.s_items) + '\n'
        str_state += 'order: ' + str(self.orders)
        return str_state

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        new_warehouse = Warehouse()
        new_warehouse.c_items = copy.deepcopy(self.c_items)
        new_warehouse.r_items = copy.deepcopy(self.r_items)
        new_warehouse.s_items = copy.deepcopy(self.s_items)
        new_warehouse.orders = copy.deepcopy(self.orders)
        new_warehouse.processed_time = self.processed_time

        return new_warehouse

    def process(self, tick, c_decision, r_decision, item_in=-1, order_in=None, current_anomaly=None):
        is_c_processed = False
        is_s_processed = False

        if order_in is not None:
            self.orders.append(order_in)

        if current_anomaly is not None:
            current_anomaly = [current_anomaly[0], 0, current_anomaly[1]]

        c_move = -1
        if c_decision != config.warehouse_num_r:
            if len(self.c_items) > 0:
                if self.c_items[0] != -1:
                    if len(self.r_items[c_decision]) < config.warehouse_cap_conveyor:
                        c_move = self.c_items.pop(0)

        r_move = -1
        if r_decision != config.warehouse_num_r:
            if len(self.r_items[r_decision]) > 0:
                if self.r_items[r_decision][0] != -1:
                    if len(self.s_items) < config.warehouse_cap_conveyor:
                        if current_anomaly is None or current_anomaly[r_decision] != 1:
                            r_move = self.r_items[r_decision].pop(0)

        if len(self.s_items) > 0:
            target_item = self.s_items[0]
            if target_item != -1:
                for order in self.orders:
                    if order.item == target_item:
                        self.s_items.pop(0)
                        self.processed_time += tick - order.tick
                        self.processed_order += 1
                        self.average_time = self.processed_time / self.processed_order
                        self.orders.remove(order)
                        is_s_processed = True
                        break

        if item_in != -1:
            self.c_items.append(item_in)

        if c_move != -1:
            self.r_items[c_decision].append(c_move)
            is_c_processed = True

        if r_move != -1:
            self.s_items.append(r_move)

        return is_c_processed, is_s_processed

    def is_ended(self):
        if len(self.c_items) > config.warehouse_cap_conveyor:
            return True

        for i in range(config.warehouse_num_r):
            if len(self.r_items[i]) > config.warehouse_cap_conveyor:
                return True

        if len(self.s_items) > config.warehouse_cap_conveyor:
            return True

        if len(self.orders) > config.warehouse_cap_order:
            return True

        return False

    def model_state(self, state_type=''):
        state = []
        if state_type != 'r':
            if len(self.c_items) > 0:
                state.append(self.c_items[0])
            else:
                state.append(-1)

        for i in range(config.warehouse_num_r):
            # state.extend(self.r_items[i])
            # if len(self.r_items[i]) < config.warehouse_cap_conveyor:
            #     for _ in range(config.warehouse_cap_conveyor - len(self.r_items[i])):
            #         state.append(-1)
            if len(self.r_items[i]) == 0:
                state.append(-1)
            else:
                state.append(self.r_items[i][0])

        # if len(self.orders) < config.warehouse_cap_order:
        #     for order in self.orders:
        #         state.append(order.item)
        #     for _ in range(config.warehouse_cap_order - len(self.orders)):
        #         state.append(-1)
        # else:
        #     for order in self.orders[:config.warehouse_cap_order]:
        #         state.append(order.item)

        return state

    def available_tactic(self, is_together=False):
        c_tactic = []
        r_tactic = []
        for repo in self.r_items:
            if len(repo) < config.warehouse_cap_conveyor:
                c_tactic.append(True)
            else:
                c_tactic.append(False)

            if len(repo) != 0:
                r_tactic.append(True)
            else:
                r_tactic.append(False)

        c_tactic.append(True)
        r_tactic.append(True)

        if is_together:
            whole = []
            for i in range(config.rl_output_size):
                if c_tactic[i // config.warehouse_num_r] and r_tactic[i % config.warehouse_num_r]:
                    whole.append(True)
                else:
                    whole.append(False)

            return whole

        return c_tactic, r_tactic


def decision_making_default(warehouse):
    c_decision = config.warehouse_num_r
    if len(warehouse.c_items) != 0:
        c_decision = config.warehouse_default_c[warehouse.c_items[0]]

    r_tactic = config.warehouse_num_r
    min_index = config.sim_total_ticks + 1
    for i in range(config.warehouse_num_r):
        if len(warehouse.r_items[i]) > 0:
            target_item = warehouse.r_items[i][0]
            j = 0
            while j < min_index and j < len(warehouse.orders):
                if warehouse.orders[j].item == target_item:
                    r_tactic = i
                    min_index = j
                    break
                j += 1

    return c_decision, r_tactic


def decision_making_rl(warehouse, rl_model, is_anomaly_aware=False, current_anomaly=None):
    if current_anomaly is None:
        current_anomaly = [0, 0]

    if is_anomaly_aware:
        state_tensor = torch.FloatTensor([[*current_anomaly, *warehouse.model_state()]]).to(config.cuda_device)
    else:
        state_tensor = torch.FloatTensor([warehouse.model_state()]).to(config.cuda_device)

    result_tactic = rl_model.select_tactic(state_tensor, warehouse.available_tactic(True))

    c_tactic = int(torch.round(result_tactic / (config.warehouse_num_r + 1)))
    r_tactic = int(result_tactic % (config.warehouse_num_r + 1))

    return c_tactic, r_tactic


def decision_making_rl_rl(warehouse, c_model, r_model, is_anomaly_aware=False, current_anomaly=None):
    if current_anomaly is None:
        current_anomaly = [0, 0]

    if is_anomaly_aware:
        c_state_tensor = torch.FloatTensor([[*current_anomaly, *warehouse.model_state('c')]]).to(config.cuda_device)
        r_state_tensor = torch.FloatTensor([[*current_anomaly, *warehouse.model_state('r')]]).to(config.cuda_device)
    else:
        c_state_tensor = torch.FloatTensor([warehouse.model_state('c')]).to(config.cuda_device)
        r_state_tensor = torch.FloatTensor([warehouse.model_state('r')]).to(config.cuda_device)

    c_av, r_av = warehouse.available_tactic()
    c_result_tensor = c_model.select_tactic(c_state_tensor, c_av)
    r_result_tensor = r_model.select_tactic(r_state_tensor, r_av)

    return c_result_tensor, r_result_tensor


def run_warehouse(tick, warehouse, c_decision, r_decision, item_list, order_list, anomalies=None):
    if tick % config.warehouse_tick_per_in == 0:
        index = tick // config.warehouse_tick_per_in
        order_in = order_list[index]
        item_in = item_list[index]
    else:
        order_in = None
        item_in = -1

    current_anomaly = anomaly.get_anomaly(tick, anomalies)

    c, s = warehouse.process(tick, c_decision, r_decision, item_in, order_in, current_anomaly)

    return warehouse, c, s


def run(dm_type, item_list, order_list, anomalies=None):
    warehouse = Warehouse()
    rl_model = None
    c_model = None
    r_model = None

    if dm_type == 'RL-ONE':
        rl_model = reinforcement_learning.DQN(config.rl_input_size, config.rl_output_size, path='model/rl.pth').to(
            config.cuda_device)

    elif dm_type == 'AD-RL-ONE':
        rl_model = reinforcement_learning.DQN(config.rl_input_size + 1, config.rl_output_size, True,
                                              path='model/a_rl.pth').to(config.cuda_device)

    elif dm_type == 'RL':
        c_model = reinforcement_learning.DQN(config.warehouse_num_r + 1, config.warehouse_num_r + 1,
                                             path='model/rl_c.pth').to(config.cuda_device)
        r_model = reinforcement_learning.DQN(config.warehouse_num_r, config.warehouse_num_r + 1,
                                             path='model/rl_r.pth').to(config.cuda_device)

    elif dm_type == 'AD-RL':
        c_model = reinforcement_learning.DQN(config.warehouse_num_r + 3, config.warehouse_num_r + 1, True,
                                             path='model/a_rl_c.pth').to(config.cuda_device)
        r_model = reinforcement_learning.DQN(config.warehouse_num_r + 2, config.warehouse_num_r + 1, True,
                                             path='model/a_rl_r.pth').to(config.cuda_device)

    for tick in range(config.sim_total_ticks):
        current_anomaly = anomaly.get_anomaly(tick, anomalies)

        if dm_type == 'RL-ONE':
            c_decision, r_decision = decision_making_rl(warehouse, rl_model)
        elif dm_type == 'AD-RL-ONE':
            c_decision, r_decision = decision_making_rl(warehouse, rl_model, True, current_anomaly)
        elif dm_type == 'RL':
            c_decision, r_decision = decision_making_rl_rl(warehouse, c_model, r_model)
        elif dm_type == 'AD-RL':
            c_decision, r_decision = decision_making_rl_rl(warehouse, c_model, r_model, True, current_anomaly)
        elif dm_type == 'Random':
            c_decision = random.randrange(3)
            r_decision = random.randrange(3)
        else:
            c_decision, r_decision = decision_making_default(warehouse)

        warehouse, c, s = run_warehouse(tick, warehouse, c_decision, r_decision, item_list, order_list, anomalies)

        # print(tick)
        # print(warehouse)
        # print()

        # if warehouse.is_ended():
        #     return warehouse.processed_order, warehouse.average_time, tick

    return warehouse.processed_order, warehouse.average_time, config.sim_total_ticks

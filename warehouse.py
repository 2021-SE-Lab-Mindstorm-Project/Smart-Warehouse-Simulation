import copy

import torch

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
        if order_in is not None:
            self.orders.append(order_in)

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
                        if current_anomaly is None or current_anomaly.way != r_decision:
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
                        break

        if item_in != -1:
            self.c_items.append(item_in)

        if c_move != -1:
            self.r_items[c_decision].append(c_move)

        if r_move != -1:
            self.s_items.append(r_move)

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
            state.extend(self.r_items[i])
            if len(self.r_items[i]) < config.warehouse_cap_conveyor:
                for _ in range(config.warehouse_cap_conveyor - len(self.r_items[i])):
                    state.append(-1)

        if len(self.orders) < config.warehouse_cap_order:
            for order in self.orders:
                state.append(order.item)
            for _ in range(config.warehouse_cap_order - len(self.orders)):
                state.append(-1)
        else:
            for order in self.orders[:config.warehouse_cap_order]:
                state.append(order.item)

        return state


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


def decision_making_rl(warehouse, rl_model, is_anomaly_aware=False, current_anomaly=1):
    if is_anomaly_aware:
        state_tensor = torch.FloatTensor([[current_anomaly, *warehouse.model_state()]]).to(config.cuda_device)
    else:
        state_tensor = torch.FloatTensor([warehouse.model_state()]).to(config.cuda_device)

    result_tensor = rl_model.model(state_tensor).data.max(1)[1].view(1, 1)

    c_tactic = int(torch.round(result_tensor / (config.warehouse_num_r + 1)))
    r_tactic = int(result_tensor % (config.warehouse_num_r + 1))

    return c_tactic, r_tactic


def decision_making_rl_rl(warehouse, c_model, r_model, is_anomaly_aware=False, current_anomaly=1):
    if is_anomaly_aware:
        c_state_tensor = torch.FloatTensor([[current_anomaly, *warehouse.model_state('c')]]).to(config.cuda_device)
        r_state_tensor = torch.FloatTensor([[current_anomaly, *warehouse.model_state('r')]]).to(config.cuda_device)
    else:
        c_state_tensor = torch.FloatTensor([warehouse.model_state('c')]).to(config.cuda_device)
        r_state_tensor = torch.FloatTensor([warehouse.model_state('r')]).to(config.cuda_device)

    c_result_tensor = c_model.model(c_state_tensor).data.max(1)[1].view(1, 1)
    r_result_tensor = r_model.model(r_state_tensor).data.max(1)[1].view(1, 1)

    return c_result_tensor, r_result_tensor


def run_warehouse(tick, warehouse, c_decision, r_decision, item_list, order_list, anomalies=None):
    if tick % config.warehouse_tick_per_in == 0:
        index = tick // config.warehouse_tick_per_in
        order_in = order_list[index]
        item_in = item_list[index]
    else:
        order_in = None
        item_in = -1

    current_anomaly = None
    if anomalies is not None:
        for single_anomaly in anomalies:
            if single_anomaly.valid(tick):
                current_anomaly = single_anomaly

    warehouse.process(tick, c_decision, r_decision, item_in, order_in, current_anomaly)

    return warehouse


def run(dm_type, item_list, order_list, anomalies=None):
    warehouse = Warehouse()
    rl_model = None
    c_model = None
    r_model = None

    if dm_type == 'RL':
        rl_model = reinforcement_learning.DQN(config.rl_input_size, config.rl_output_size, path='model/rl.pth').to(
            config.cuda_device)

    elif dm_type == 'A-RL':
        rl_model = reinforcement_learning.DQN(config.rl_input_size + 1, config.rl_output_size, True,
                                              path='model/a_rl.pth').to(config.cuda_device)

    elif dm_type == 'RL-RL':
        c_model = reinforcement_learning.DQN(config.rl_input_size, config.warehouse_num_r, path='model/rl_c.pth').to(
            config.cuda_device)
        r_model = reinforcement_learning.DQN(config.rl_input_size - 1, config.warehouse_num_r,
                                             path='model/rl_r.pth').to(config.cuda_device)

    elif dm_type == 'A-RL-RL':
        c_model = reinforcement_learning.DQN(config.rl_input_size + 1, config.warehouse_num_r, True,
                                             path='model/a_rl_c.pth').to(config.cuda_device)
        r_model = reinforcement_learning.DQN(config.rl_input_size, config.warehouse_num_r, True,
                                             path='model/a_rl_r.pth').to(config.cuda_device)

    for tick in range(config.sim_total_ticks):
        current_anomaly = 1
        if dm_type == 'A-RL' or dm_type == 'A-RL-RL':
            if anomalies is not None:
                for single_anomaly in anomalies:
                    if single_anomaly.valid(tick):
                        current_anomaly = single_anomaly.way

        if dm_type == 'RL':
            c_decision, r_decision = decision_making_rl(warehouse, rl_model)
        elif dm_type == 'A-RL':
            c_decision, r_decision = decision_making_rl(warehouse, rl_model, True, current_anomaly)
        elif dm_type == 'RL-RL':
            c_decision, r_decision = decision_making_rl_rl(warehouse, c_model, r_model)
        elif dm_type == 'A-RL-RL':
            c_decision, r_decision = decision_making_rl_rl(warehouse, c_model, r_model, True, current_anomaly)
        else:
            c_decision, r_decision = decision_making_default(warehouse)

        warehouse = run_warehouse(tick, warehouse, c_decision, r_decision, item_list, order_list, anomalies)

        # print(tick)
        # print(warehouse)
        # print()

        # if warehouse.is_ended():
        #     return warehouse.processed_order, warehouse.average_time, tick

    return warehouse.processed_order, warehouse.average_time, config.sim_total_ticks

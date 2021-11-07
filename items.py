import csv
import random

import config


class Order:
    def __init__(self, item, tick):
        assert 0 <= item < config.warehouse_num_types
        self.item = item
        self.tick = tick

    def __str__(self):
        return 'item ' + str(self.item) + ' at ' + str(self.tick)

    def __repr__(self):
        return str(self)


def get_item(index):
    item_sample = []
    while len(item_sample) < config.warehouse_ratio_duration:
        for i in range(len(config.warehouse_ratio_item[index])):
            for _ in range(config.warehouse_ratio_item[index][i]):
                item_sample.append(i)

    return random.sample(item_sample, config.warehouse_ratio_duration)


def generate_item_list(is_order, name=''):
    items = []
    item_sample = []
    index = 0
    for i in range(0, config.sim_total_ticks, config.warehouse_tick_per_in):
        if len(item_sample) == 0:
            item_sample = get_item(index)
            index += 1
            index %= len(config.warehouse_ratio_item)

        if is_order:
            items.append(Order(item_sample.pop(), i))
        else:
            items.append(item_sample.pop())

    if name != '':
        if is_order:
            file_name = 'log/order/' + name + '.csv'
        else:
            file_name = 'log/item/' + name + '.csv'

        with open(file_name, 'w', newline='') as item_file:
            item_writer = csv.writer(item_file)
            for i in range(0, config.sim_total_ticks, config.warehouse_tick_per_in):
                if is_order:
                    item_writer.writerow([i, items[i // config.warehouse_tick_per_in].item])
                else:
                    item_writer.writerow([i, items[i // config.warehouse_tick_per_in]])

    return items

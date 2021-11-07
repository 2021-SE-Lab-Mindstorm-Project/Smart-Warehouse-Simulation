import datetime

import anomaly
import config
import items
import warehouse

if __name__ == "__main__":
    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        item_list = items.generate_item_list(False, experiment_name)
        order_list = items.generate_item_list(True, experiment_name)
        anomalies = anomaly.generate_anomaly(name=experiment_name)

        # for target in config.sim_targets:
        #     processed, average, ended = warehouse.run(target, item_list, order_list)
        #     print(target)
        #     print('ended: %d' % ended)
        #     print('total processed: %d' % processed)
        #     print('average processing: %.2f tick' % average)
        #     print()

        for target in config.sim_targets:
            processed, average, ended = warehouse.run(target, item_list, order_list, anomalies)
            print(target)
            print('ended: %d' % ended)
            print('total processed: %d' % processed)
            print('average processing: %.2f tick' % average)
            print()

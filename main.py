import datetime

import config

if __name__ == "__main__":
    for i in range(config.sim_count):
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

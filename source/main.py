import argparse
import json

import evaluate
from source.train import train

from huggingface_hub import login
# login("PERSONAL_ACCESS_TOKEN")  # Replace with your actual token, use environment variable or cli login


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['config_path'] = config_path
    task = config['task']
    assert task in ["train", "eval"], f"Task needs to be either 'train' or 'eval' but is {task}"

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()
    config_path = args.config
    config = parse_config(config_path)

    if config['task'] == "eval":
        evaluate.eval(config)
    elif config['task'] == "train":
        train(config)

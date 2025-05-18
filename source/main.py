import json

from huggingface_hub import login

login("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")

import evaluate
from source.train import train

dataset_path = "/mnt/2tb-1/louis/data/ImageDataset/ann_new.jsonl"
dataset_orig_file = "/mnt/2tb-1/louis/data/ImageDataset/annotations.json"

output_dir = "/home/louis/workspace/Anomaly_Det_VLM/out"
dataset_id = "gravis-excavation"

# os.environ["HF_HOME"] = "/mnt/2tb-2/VLM_models"
# os.environ["MODELSCOPE_CACHE"] = "/mnt/2tb-2/VLM_models"


# model_id = "deepseek-ai/deepseek-vl2-tiny"
# model_id = "deepseek-ai/deepseek-vl2-small"
# model_id = "deepseek-ai/deepseek-vl2"

# model_id = "Qwen/Qwen2-VL-2B-Instruct"
# model_id = "Qwen/Qwen2-VL-7B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-72B-Instruct"


# model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"   # todo
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"  # todo

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"
# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['config_path'] = config_path
    task = config['task']
    assert task in ["train", "eval"], f"Task needs to be either 'train' or 'eval' but is {task}"
    model_id = config['model_id']
    dataset_path = config['dataset_path']
    prompt = config['prompt']
    temperature = config['temperature']
    max_new_tokens = config['max_new_tokens']

    return config


if __name__ == "__main__":
    # config_path = "/home/louis/workspace/Anomaly_Det_VLM/configs/training/DeepSeek/deepseek-vl2-tiny_train.json"
    config_path = "/home/louis/workspace/Anomaly_Det_VLM/configs/deepseek_test.json"

    config = parse_config(config_path)

    if config['task'] == "eval":
        evaluate.eval(config)
    elif config['task'] == "train":
        train(config)
#
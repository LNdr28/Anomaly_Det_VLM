import json
import os

import torch
from swift import Swift, Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.llm import get_model_tokenizer, get_template, DatasetMeta, register_dataset, load_dataset, \
    EncodePreprocessor, PtEngine, InferRequest, RequestConfig

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
model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"   # todo
# model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"  # todo

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "meta-llama/Llama-3.3-70B-Instruct"
# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"


def infer():

    max_new_tokens = 2048
    temperature = 0.3
    # Perform inference using the native PyTorch engine

    # device_map = {"0": "cuda:1"}

    if "meta-llama/Llama-3.2" in model_id:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True, torch_dtype=torch.float)
    else:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True,)


    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

    # messages = [{"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging."},
    #             {"role": "user", "content": "<image>This is an image of a trench that has been dug by an excavator. Does the trench contain any objects that could hinder excavation? Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list: []"}]
    #
    # images = ["/home/louis/workspace/Anomaly_Det_VLM/custom/wire1.png"]


    messages = [{"role": "system", "content": "You are a professional airplane classification system."},
                {"role": "user", "content": "<image>What airplane is this? Choose from these possibilities: Airbus220, Airbus321, Airbus330, Airbus350, Boeing737, Boeing747, Boeing777, Boeing787."}]

    images = ["/home/louis/workspace/Anomaly_Det_VLM/custom/Toronto_23MAY27162713_0.png"]

    infer_request = InferRequest(messages=messages, images=images)

    resp_list = engine.infer([infer_request], request_config)
    print(f'response: {resp_list[0].choices[0].message.content}')


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

    config_path = "/home/louis/workspace/Anomaly_Det_VLM/configs/Qwen2.5-VL-32B-Instruct_conf.json"
    config = parse_config(config_path)

    # dataset_meta = DatasetMeta(
    #     dataset_path=dataset_path,  # Your dataset file
    # )
    # register_dataset(dataset_meta)
    # dataset = load_dataset(dataset_path)

    # infer()
    # train(model_id, dataset_path)

    evaluate.eval(config)

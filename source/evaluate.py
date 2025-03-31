import json
import os
from pathlib import Path
from datetime import datetime

import torch
from swift.llm import PtEngine, RequestConfig, register_dataset, load_dataset, DatasetMeta, InferRequest
from tqdm import tqdm

from source.utils import process_images


def eval(config):
    tmp_folder = config.get('tmp_folder', (Path(config['config_path']).parent / "tmp").mkdir(exist_ok=True))

    dataset_path = config['dataset_path']
    model_id = config['model_id']
    prompt = config['prompt']
    if "CONF_VAL" in prompt:
        assert "confidence_threshold" in config.keys(), "Confidence threshold is used in the prompt but not given in the config!"
        confidence_threshold = config["confidence_threshold"]
        prompt = prompt.replace("CONF_VAL", str(confidence_threshold))

    img_type = config.get('img_type', "default")

    data_list = []

    annotations = json.load(open(dataset_path, "r"))
    for item in annotations:
        img_id = item["image_id"]
        trench = item["trench"]
        anomalies = item["anomalies_present"]
        img_path = str(Path(dataset_path).parent / img_id)
        image_path = process_images(img_path, item["trench"], tmp_folder, img_type)

        messages = [{"role": "system",
                     "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging."},
                    {"role": "user", "content": "<image>"+prompt},
                    {"role": "assistant", "content": anomalies}]

        # images = [str(Path(dataset_path).parent / img_id)]
        images = [image_path]


        data_list.append([messages, images])

    max_new_tokens = config['max_new_tokens']
    temperature = config['temperature']

    if "meta-llama/Llama-3.2" in model_id:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True, torch_dtype=torch.float)
    else:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True)

    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

    total_images = 0
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0

    for data in tqdm(data_list, desc="Evaluating: "):
        messages = data[0]
        images = data[1]
        infer_request = InferRequest(messages=messages[:-1], images=images)
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        print(f"response: {response}; GT is {messages[2]['content']}")


        pred_anomaly = 0 if (response in ['[]', '[ ]', 'No', 'no']) else 1
        gt_anomaly = 0 if (messages[2]['content'] in ['[]', '[ ]', []]) else 1

        print(f"pred: {pred_anomaly}; gt: {gt_anomaly}")

        total_images += 1
        if gt_anomaly and pred_anomaly:
            true_pos += 1
        elif not (gt_anomaly or pred_anomaly):
            true_neg += 1
        elif gt_anomaly and not pred_anomaly:
            false_neg += 1
        elif not gt_anomaly and pred_anomaly:
            false_pos += 1

        os.remove(images[0])

    correct = true_neg + true_pos
    accuracy = (correct / total_images) * 100
    print(f"Accuracy: {accuracy}; TP: {true_pos}; TN: {true_neg}; FP: {false_pos}; FN: {false_neg}")

    now = datetime.now()
    config['timestamp'] = now.strftime("%d.%m.%Y;%H:%M:%S")

    # out_log = Path(output_dir) / (model_id.replace('/', '_') + now.strftime("%d.%m.%Y;%H:%M:%S") + ".log")
    config['results'] = {
        "Accuracy": accuracy,
        "TP": true_pos,
        "TN": true_neg,
        "FP": false_pos,
        "FN": false_neg
    }
    config_path = config['config_path']
    del config['config_path']
    out_log = str(Path(config_path).parent / Path(config_path).name[:-5]) + '_' + now.strftime("%d.%m.%Y;%H:%M:%S") + ".log"
    with open(out_log, 'w') as out_f:
        json.dump(config, out_f, indent=4)
        # out_f.write(f"Evaluation of model {model_id}\nModel params: ...\nAccuracy: {accuracy}; TP: {true_pos}; TN:
        # {true_neg}; FP: {false_pos}; FN: {false_neg}\nPrompt: <image>This is an image of a trench that has been dug
        # by an excavator. You are a professional anomaly detection and classification tool that detects objects that
        # could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools,
        # large stones and wooden planks. Provide only the english names of the objects that you detect in the trench
        # as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or
        # a whole excavator, you ignore them and return an empty list ’[]’.")
    os.rmdir(tmp_folder)

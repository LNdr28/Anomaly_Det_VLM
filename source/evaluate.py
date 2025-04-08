import json
import os
from pathlib import Path
from datetime import datetime

import torch
from swift.llm import PtEngine, RequestConfig, register_dataset, load_dataset, DatasetMeta, InferRequest
from tqdm import tqdm

from source.utils import process_images


def eval(config):
    tmp_folder = config.get('tmp_folder', (Path(config['config_path']).parent / "tmp"))
    tmp_folder.mkdir(exist_ok=True)

    dataset_path = config['dataset_path']
    model_id = config['model_id']
    prompt = config['prompt']
    if "CONF_VAL" in prompt:
        assert "confidence_threshold" in config.keys(), "Confidence threshold is used in the prompt but not given in the config!"
        confidence_threshold = config["confidence_threshold"]
        prompt = prompt.replace("CONF_VAL", str(confidence_threshold))

    img_type = config.get('img_type', "default")
    exclude_stones = config.get('exclude_stones', False)

    data_list = []

    annotations = json.load(open(dataset_path, "r"))

    for item in tqdm(annotations, desc="Preprocessing data: "):
        img_id = item["image_id"]
        trench = item["trench"]
        anomalies = item["anomalies_present"]
        img_path = str(Path(dataset_path).parent / img_id)
        image_path = process_images(img_path, trench, tmp_folder, img_type)

        messages = [{"role": "system",
                     "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging."},
                    {"role": "user", "content": "<image>"+prompt},
                    {"role": "assistant", "content": anomalies}]

        # images = [str(Path(dataset_path).parent / img_id)]
        images = [image_path]


        data_list.append([messages, images])
    if "meta-llama/Llama-3.2" in model_id:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True, torch_dtype=torch.float)
    else:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True)

    request_config = RequestConfig(max_tokens=config['max_new_tokens'], temperature=config['temperature'], top_k=config['top_k'])

    total_images = 0
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0

    missed_table = {}
    all_table = {}

    for data in tqdm(data_list, desc="Evaluating: "):
        messages = data[0]
        images = data[1]
        gt = messages[2]['content']
        infer_request = InferRequest(messages=messages[:-1], images=images)
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        print(f"response: {response}; GT is {gt}")
        for obj in gt:
            if obj not in all_table.keys():
                all_table[obj] = 1
            else:
                all_table[obj] += 1


        pred_anomaly = 0 if (response in ['[]', '[ ]', 'No', 'no']) else 1
        if exclude_stones:
            gt_anomaly = 0 if (gt in ['[]', '[ ]', [], '[stone]', '["stone"]', ["stone"], '[large stone]', '["large stone"]', ['large stone'], []]) else 1
        else:
            gt_anomaly = 0 if (gt in ['[]', '[ ]', []]) else 1

        print(f"pred: {pred_anomaly}; gt: {gt_anomaly}")

        total_images += 1
        if gt_anomaly and pred_anomaly:
            true_pos += 1
        elif not (gt_anomaly or pred_anomaly):
            true_neg += 1
        elif gt_anomaly and not pred_anomaly:
            false_neg += 1
            for obj in gt:
                if obj not in missed_table.keys():
                    missed_table[obj] = 1
                else:
                    missed_table[obj] += 1
        elif not gt_anomaly and pred_anomaly:
            false_pos += 1

        os.remove(images[0])

    correct = true_neg + true_pos
    accuracy = (correct / total_images) * 100
    f1 = 2*true_pos / (2*true_pos + false_pos + false_neg)
    print(f"Accuracy: {accuracy}; F1: {f1}; TP: {true_pos}; TN: {true_neg}; FP: {false_pos}; FN: {false_neg}")
    print(20 * '-')
    print(f"All GT anomalies: \n{all_table}\n Missed anomalies (FN): \n{missed_table}")

    now = datetime.now()
    config['timestamp'] = now.strftime("%d.%m.%Y;%H:%M:%S")

    # out_log = Path(output_dir) / (model_id.replace('/', '_') + now.strftime("%d.%m.%Y;%H:%M:%S") + ".log")
    config['results'] = {
        "Accuracy": accuracy,
        "F1": f1,
        "TP": true_pos,
        "TN": true_neg,
        "FP": false_pos,
        "FN": false_neg,
        "All_Anomalies": all_table,
        "Missed_Anomalies": missed_table
    }
    config_path = config['config_path']
    del config['config_path']
    out_log = str(Path(config_path).parent / Path(config_path).name[:-5]) + '_' + now.strftime("%d.%m.%Y;%H:%M:%S") + ".log"
    with open(out_log, 'w') as out_f:
        json.dump(config, out_f, indent=4)
    os.rmdir(tmp_folder)




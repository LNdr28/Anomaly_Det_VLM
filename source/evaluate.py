import json
import os
from pathlib import Path
from datetime import datetime

import torch
from swift.llm import PtEngine, RequestConfig,  InferRequest
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
    exclude_stones = config.get('ignore_stones', False)

    data_list = []

    annotations = json.load(open(dataset_path, "r"))

    use_context = config.get('context', False)
    context_type = config.get('context_type', "split")

    if use_context is not False and context_type == "split":
        print('Using (small?) context.')
        context_base = Path(os.getcwd()).parent / "context"
        context_bucket = str(context_base / "bucket.png")
        context_images = [context_bucket]
        context_messages = [
                    {"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging. You will be first presented with some example images of the trench and the bucket. Then you will be asked to detect anomalies in the trench."}, {"role": "assistant", "content": ""},
                    {"role": "user", "content": "<image> This image shows the trench without any anomalies"}, {"role": "assistant", "content": ""},
                    ]

        if use_context == "full" or (use_context and use_context != "small"):
            print('Using full context.')
            context_pipe_plank = str(context_base / "pipe_plank.png")
            context_stone = str(context_base / "stone.png")
            context_images.extend([context_pipe_plank, context_stone])

            context_messages.extend([
                        {"role": "user", "content": "<image> This image shows the trench with some objects. Thus the correct output would be [plank, pipe]."}, {"role": "assistant", "content": "[plank, pipe]"},
                        {"role": "user", "content": "<image> This image shows the trench a stone. The stone is small enough to fit in the bucket, so the correct output would be []."}, {"role": "assistant", "content": "[]"}
            ])
    else:
        context_messages = [{"role": "system",
                     "content": "You are a professional anomaly detection and classification tool that detects objects that prevent an excavator from digging."}]


    for item in tqdm(annotations, desc="Preprocessing data: "):
        img_id = item["image_id"]
        trench = item["trench"]
        anomalies = item["anomalies_present"]
        img_path = str(Path(dataset_path).parent / img_id)
        image_path = process_images(img_path, trench, tmp_folder, img_type)

        if context_type == 'single-message':
            if use_context == 'full':
                raise Exception('not implemented yet')
            context_base = Path(os.getcwd()).parent / "context"
            context_bucket = str(context_base / "bucket.png")
            images = [context_bucket, image_path]

            messages = context_messages + [{"role": "user", "content": "<image> This image shows the trench with the excavator's bucket. Use the bucket size to check if stones are too big to fit and should count as anomalies in the following image. <image>"+prompt},
                    {"role": "assistant", "content": anomalies}]

        else:
            messages = context_messages + [{"role": "user",
                                            "content": "<image>" + prompt},
                                           {"role": "assistant", "content": anomalies}]
            if config.get('context', False):
                images = context_images + [image_path]
            else:
                images = [image_path]

        data_list.append([messages, images])

    adapter = config.get('adapter', None)
    adapter = [adapter] if adapter else None

    if "meta-llama/Llama-3.2" in model_id:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True, adapters=adapter, torch_dtype=torch.float)
    else:
        engine = PtEngine(model_id, max_batch_size=2, use_hf=True, adapters=adapter)

    request_config = RequestConfig(max_tokens=config.get('max_new_tokens', 128), temperature=config['temperature'], top_k=config['top_k'])

    total_images = 0
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0

    class_true_pos = 0
    class_true_neg = 0
    class_false_neg = 0
    class_false_pos = 0

    missed_table = {}
    all_table = {}

    false_pred_images = {}

    with tqdm(data_list, desc="Evaluating: ", leave=True, dynamic_ncols=True) as pbar:
        for data in pbar:
            messages = data[0]
            images = data[1]
            gt = messages[-1]['content']
            infer_request = InferRequest(messages=messages[:-1], images=images)
            resp_list = engine.infer([infer_request], request_config)
            response = resp_list[0].choices[0].message.content

            msg = f"-------- response: {response}; GT is {gt}"
            pbar.set_postfix_str(msg)

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

            total_images += 1
            if gt_anomaly and pred_anomaly:
                true_pos += 1
            elif not (gt_anomaly or pred_anomaly):
                true_neg += 1
                class_true_neg += 1
            elif gt_anomaly and not pred_anomaly:
                false_pred_images[images[0].split("/")[-1]] = f'FN: {gt}'
                false_neg += 1
                class_false_neg += 1
                for obj in gt:
                    if obj not in missed_table.keys():
                        missed_table[obj] = 1
                    else:
                        missed_table[obj] += 1
            elif not gt_anomaly and pred_anomaly:
                false_pred_images[images[0].split("/")[-1]] = f'FP: {response}'
                false_pos += 1

            response = response.replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(' ', '')
            if response in ['cable', 'wire', 'pipe']:
                response = 'cable'
            if 'stone' in response:
                response = 'stone'
            gt = str(gt).replace('[', '').replace(']', '').replace('"', '').replace("'", '').replace(' ', '').split(',')
            if 'largestone' in gt or 'smallstone' in gt:
                gt.append('stone')
            if 'cable' in gt or 'wire' in gt or 'pipe' in gt or 'cable(buried)' in gt:
                gt.append('cable')
            if 'barrierplank' in gt or 'barrier_plank' in gt or 'barrier plank' in gt:
                gt.append('plank')
            if pred_anomaly and response in gt:
                class_true_pos += 1
            elif pred_anomaly:
                class_false_pos += 1

            os.remove(images[-1])

    correct = true_neg + true_pos
    accuracy = (correct / total_images) * 100
    class_accuracy = ((class_true_pos+class_true_neg) / total_images) * 100
    f1 = 2*true_pos / (2*true_pos + false_pos + false_neg)
    print(f"Accuracy: {accuracy}; F1: {f1}; Class Accuracy: {class_accuracy}; TP: {true_pos}; TN: {true_neg}; FP: {false_pos}; FN: {false_neg}")
    print(20 * '-')
    print(f"All GT anomalies: \n{all_table}\n Missed anomalies (FN): \n{missed_table}")

    now = datetime.now()
    config['timestamp'] = now.strftime("%d.%m.%Y;%H:%M:%S")

    config['results'] = {
        "Accuracy": accuracy,
        "F1": f1,
        "TP": true_pos,
        "TN": true_neg,
        "FP": false_pos,
        "FN": false_neg,
        "All_Anomalies": all_table,
        "Missed_Anomalies": missed_table,
    }

    if config.get('list_wrong', False):
        print(f"Images with false predictions: \n{false_pred_images}")
        config['results']['wrong_det'] = false_pred_images

    config_path = config['config_path']
    del config['config_path']
    out_log = str(Path(config_path).parent / Path(config_path).name[:-5]) + '_' + now.strftime("%d.%m.%Y;%H:%M:%S") + ".log"
    with open(out_log, 'w') as out_f:
        json.dump(config, out_f, indent=4)
    os.rmdir(tmp_folder)

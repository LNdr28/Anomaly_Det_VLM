import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from source.utils import process_images


def convert_dataset(in_annotations, prompt, output_dir=None, img_type='default', ignore_stones=False, dataset_type='old', downscale=False):

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(in_annotations).parent
    out = ""

    if dataset_type == 'old':
        annotations = json.load(open(in_annotations, "r"))
        for item in annotations:
            img_id = item["image_id"]
            img_path = str(Path(in_annotations).parent / img_id)
            trench = item["trench"]
            anomalies = item["anomalies_present"]

            if ignore_stones:
                anomalies = [a for a in anomalies if "stone" not in a]

            trench["image_id"] = img_id

            image_path = process_images(img_path, trench, output_dir, img_type)

            line = '{"messages": [{"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging."}, {"role": "user", "content": "<image>' + prompt + '"}, {"role": "assistant", "content": "' + str(anomalies) + '"}], "images": ["' + image_path + '"], "label": true}'

            out += line + "\n"
    elif dataset_type == 'new':
        for video in Path(in_annotations).iterdir():
            if not video.is_dir():
                print(f'Encountered non-directory item: {video}, skipping.')
                continue
            for anomaly_dir in video.iterdir():
                if not anomaly_dir.is_dir():
                    continue
                anomaly = anomaly_dir.name
                if anomaly == 'large_stone' and ignore_stones:
                    print(f'Ignoring large stones anomaly in {video.name}')
                    continue
                if anomaly == 'original' or ('stone' in anomaly and ignore_stones):
                    anomalies = []
                else:
                    anomalies = [anomaly]
                for img_path in anomaly_dir.glob("*"):
                    if downscale:
                        downscaled_img_path = output_dir / img_path.name
                        image = Image.open(img_path).convert("RGB")
                        new_size = (int(image.width //downscale), int(image.height //downscale))
                        downscaled_image = image.resize(new_size)
                        cv2.imwrite(downscaled_img_path, np.array(downscaled_image))
                        path = downscaled_img_path
                    else:
                        path = img_path

                    line = '{"messages": [{"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging."}, {"role": "user", "content": "<image>' + prompt + '"}, {"role": "assistant", "content": "' + str(anomalies) + '"}], "images": ["' + str(path) + '"], "label": true}'
                    out += line + "\n"

    else:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))

    with open(output_dir / "ann.jsonl", "w") as f:
        f.write(out)

    return output_dir / "ann.jsonl"

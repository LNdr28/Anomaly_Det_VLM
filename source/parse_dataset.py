import argparse
import json
from pathlib import Path

from source.utils import process_images


def convert_dataset(in_annotations, prompt, output_dir=None, img_type='default', ignore_stones=False, dataset_type='old'):

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
            xmin = trench["xmin"]
            ymin = trench["ymin"]
            xmax = trench["xmax"]
            ymax = trench["ymax"]

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
                if anomaly == 'original' or 'stone' in anomaly and ignore_stones:
                    anomalies = []
                else:
                    anomalies = [anomaly]
                for img_path in anomaly_dir.glob("*"):
                    line = '{"messages": [{"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging."}, {"role": "user", "content": "<image>' + prompt + '"}, {"role": "assistant", "content": "' + str(anomalies) + '"}], "images": ["' + str(img_path) + '"], "label": true}'
                    out += line + "\n"

    else:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))

    with open(output_dir / "ann.jsonl", "w") as f:
        f.write(out)

    return output_dir / "ann.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-annotations")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    convert_dataset(args.in_annotations, 'This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list: []. It is more important to not miss an anomaly than detecting a false positive!')

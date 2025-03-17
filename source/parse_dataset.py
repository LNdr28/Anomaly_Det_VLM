import argparse
import json
from pathlib import Path


def convert_dataset(args):

    if args.out_dir is not None:
        output_dir = Path(args.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.in_annotations
    out = ""

    annotations = json.load(open(args.in_annotations, "r"))
    for item in annotations:
        img_id = item["image_id"]
        trench = item["trench"]
        anomalies = item["anomalies_present"]
        trench["image_id"] = img_id
        xmin = trench["xmin"]
        ymin = trench["ymin"]
        xmax = trench["xmax"]
        ymax = trench["ymax"]

        prompt = "{'messages': [{'role': 'system', 'content': 'You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging.'}, {'role': 'user', 'content': '<image>This is an image of a trench that has been dug by an excavator. Does the trench contain any objects that could hinder excavation? Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list: [].'}, {'role': 'assistant', 'content': " + str(anomalies) + "}], 'images': ['" + img_id+ "']}"

        out += prompt + "\n"

    with open(Path(output_dir).parent / "ann_new.jsonl", "w") as f:
        f.write(out)
    # json.dump(out, open(Path(output_dir).parent / "ann_new.json", "w"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-annotations")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    convert_dataset(args)

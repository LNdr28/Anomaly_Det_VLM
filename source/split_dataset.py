import json
import random


def split_dataset():
    annotation_file = "/mnt/2tb-1/louis/data/ImageDataset/annotations_fixed.json"
    annotations = json.load(open(annotation_file, "r"))

    class_table: dict = {}

    for annotation in annotations:
        img_id = annotation["image_id"]
        anomalies = img_id.split(".")[0].split("_")[-1]
        anomalies = "None" if anomalies[0] == "0" else anomalies
        if anomalies not in class_table:
            class_table[anomalies] = 1
        else:
            class_table[anomalies] += 1

    print("Class table:")
    for k, v in class_table.items():
        print(f"{k}: {v}")

    out_train = "/mnt/2tb-1/louis/data/ImageDataset/train.json"
    out_val = "/mnt/2tb-1/louis/data/ImageDataset/val.json"
    out_train_list = []
    out_val_list = []
    val_size = 0.1
    val_nums = {}
    for k, v in class_table.items():
        val_nums[k] = int(v * val_size)
    print("Validation numbers:")
    for k, v in val_nums.items():
        print(f"{k}: {v}")

    random.shuffle(annotations)

    for annotation in annotations:
        img_id = annotation["image_id"]
        anomalies = img_id.split(".")[0].split("_")[-1]
        anomalies = "None" if anomalies[0] == "0" else anomalies
        if anomalies in val_nums.keys() and val_nums[anomalies] > 0:
            val_nums[anomalies] -= 1
            out_val_list.append(annotation)
        else:
            out_train_list.append(annotation)

    with open(out_train, "w") as f:
        json.dump(out_train_list, f, indent=4)
    with open(out_val, "w") as f:
        json.dump(out_val_list, f, indent=4)


if __name__ == "__main__":
    split_dataset()

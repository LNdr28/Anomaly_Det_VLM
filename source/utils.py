import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageEnhance
from PIL import Image


def load_image(img_path, bbox, img_type, saturation=1.0, contrast=1.0, sharpness=1.0, cropped=False, black=False):

    image = Image.open(img_path).convert("RGB")

    if saturation != 1.0:
        color_enhancer = ImageEnhance.Color(image)
        image = color_enhancer.enhance(saturation)

    if contrast != 1.0:
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(contrast)

    if sharpness != 1.0:
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(sharpness)

    if img_type == "crop":
        out_image = image.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))
    elif img_type == "black":
        out_image = Image.new("RGB", image.size, (0, 0, 0))
        out_image.paste(image.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])), (bbox["xmin"], bbox["ymin"]))
    elif img_type == "default":
        out_image = image
    else:
        raise Exception(f"Image type {img_type} unknown")

    return out_image


def get_bbox(img_path, input_path):
    with open(os.path.join(input_path, "annotations.json"), "r") as f:
        data = json.load(f)

    img_id = os.path.basename(img_path)

    for item in data:
        if item["image_id"] == img_id:
            bbox = item["trench"]
            return bbox

    return None


def process_images(img_path, bbox, tmp_folder, img_type, saturation=1.0, contrast=1.0, sharpness=1.0):

    filename = Path(img_path).name
    if bbox is None:
        print(f"No trench in {img_path}, skipping.")
        image = load_image(img_path, bbox, "default", saturation=saturation, contrast=contrast, sharpness=sharpness)
        image_path = os.path.join(tmp_folder, filename)
        cv2.imwrite(image_path, np.array(image))
        return image_path

    # Load the image
    image = load_image(img_path, bbox, img_type, saturation=saturation, contrast=contrast, sharpness=sharpness)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_path = os.path.join(tmp_folder, filename)

    cv2.imwrite(image_path, image)

    return image_path

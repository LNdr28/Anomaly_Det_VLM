# Anomaly Detection with Vision-Language Models

This repository contains code for anomaly detection in excavation trenches using Vision-Language Models (VLMs). The system is designed to identify objects in trenches that may prevent an excavator from digging, such as pipes, cables, large stones, and wooden planks.  
The code is based on [ms-swift](https://github.com/modelscope/ms-swift).


## Supported Models

The following Vision-Language Models are explicitly supported:

### DeepSeek Models
- `deepseek-ai/deepseek-vl2-tiny`
- `deepseek-ai/deepseek-vl2-small`
- `deepseek-ai/deepseek-vl2`

### Qwen Models
- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen2-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-32B-Instruct`
- `Qwen/Qwen2.5-VL-72B-Instruct`

### Llama Models
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `meta-llama/Llama-3.2-90B-Vision-Instruct`

For other models check the [ms-swift model zoo](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html) or [custom model documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-model.html).

## Installation
To install using pip:
```
pip install -r requirements.txt
```

As of now ms-swift still contains a version mismatch and requires three fixes in the environment.
- ``<your_env>/site-packages/swift/llm/template/template/qwen.py`` line 310: Replace with  
`` grid_thw = [torch.tensor(b[f'{media_type}_grid_thw']) for b in batch if b.get(f'{media_type}_grid_thw') is not None] ``
- ``<your_env>/site-packages/swift/llm/template/base.py`` 
  - line 1062: ``pixel_values = [torch.tensor(b['pixel_values']) for b in batch if b.get('pixel_values') is not None]``
  - line 1066: ``image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]``  ????This necessary????

Due to the large model size it is recommended to set a custom cache location for modelscope:
```
MODELSCOPE_CACHE=<your_path>
```
For best compatibility it is recommended to train and evaluate the models on A100 GPUs or higher. RTX 30xx GPUs or higher RTX models work as well with restrictions due to support of bf16.  

The provided requirements work for both Qwen and Llama models, but Deepseek models are not compatible since they require different versions. The following versions are known to work with Deepseek:

```
Cuda version 11.8
torch==2.0.1
transformers==4.41.2
trl==0.14.0
Install requirements with --no-deps
```

## Configuration File Structure

The system uses JSON configuration files to define parameters for training and evaluation. These files should be placed in the `configs/` directory.

### Required Parameters

- `task`: Specifies the task to perform, must be either `"train"` or `"eval"`
- `model_id`: The ID of the model to use (from the supported models list)
- `dataset_path`: Path to the dataset annotations file (old dataset, default) or dataset root (new dataset)
- `prompt`: The prompt template to use for training and evaluation
- `confidence_threshold`: Confidence threshold value (required if "CONF_VAL" is in the prompt)

### Optional Parameters
- `tmp_folder`: Temporary folder for storing intermediate files for evaluation; Output folder for training (default: config filename without extension)
- `dataset_type`: old or new dataset. Default: "old"
- `img_type`: Image processing type. Full image(default), crop(cropped to trench), black(background blacked out). Default: "default"
- `ignore_stones`: Whether to ignore stones as anomalies (default: false)
- `temperature`: Temperature parameter for model generation (typically set to 0 for deterministic outputs)
- `max_new_tokens`: Maximum number of tokens to generate in the response (default: 128)
- `top_k`: Top-k parameter for sampling during generation (default: None / no sampling)
- `top_p`: Top-p parameter for sampling during generation (default: None / no sampling)
- `repetition_penalty`: Set >1 to penalize repetition in the answer (default: None / no penalty)
- `num_beams`: Set for beam search (default: 1)
- `n`: Number of generated outputs (default: 1)
- `best_of`: Only returns best of `best_of` generated outputs. Needs `n`=1. (default: None / one generation)

#### Training Specific
- `epochs`: Number of training epochs (default: 1)
- `lora_rank`: Rank for LoRA fine-tuning (default: 8)
- `lora_alpha`: Alpha parameter for LoRA fine-tuning (default: 32)
- `per_device_batch_size`: Batch size per device (default: 1)
- `downscale`: Downscale images by this factor (default: false, no downscaling). Sometimes required for training Qwen models with high-res images.

#### Evaluation Specific
- `adapter`: Path to the fine-tuned adapter to use for evaluation (default: None)
- `list_wrong`: Whether to list images with wrong predictions (default: false)
- `context`: Whether to use context images for evaluation (false, "small", or "full"). (default: false)
- `context_type`: Type of context to use ("split" or "single-message") (default: "split")

### Example Configuration

```json
{
  "task": "train",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "dataset_path": "/path/to/dataset/annotations.json",
  "prompt": "This is an image of a trench that has been dug by an excavator. You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging. Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list '[]'.",
  "temperature": 0,
  "max_new_tokens": 128,
  "top_k": null,
  "img_type": "default",
  "epochs": 3,
  "lora_rank": 8,
  "lora_alpha": 32,
  "per_device_batch_size": 1,
  "ignore_stones": false,
  "downscale": false
}
```

## Training

The training process fine-tunes a Vision-Language Model using LoRA (Low-Rank Adaptation) to detect anomalies in excavation trenches.

To train a model:

1. Create a configuration file in the `configs/` directory with the task set to `"train"`
2. Run the main script with the path to your configuration file:

```
python source/main.py /path/to/your/config.json
```

The training process includes:
- Converting the dataset to the required format
- Setting up the model with LoRA fine-tuning
- Splitting the dataset into training and validation sets
- Training the model with the specified parameters

The fine-tuned model checkpoints will be saved in the specified output/tmp directory.

## Evaluation

Evaluation measures the performance of a (trained) model on a test dataset.

To evaluate a model:

1. Create a configuration file with the task set to `"eval"`
2. If desired specify the adapter path to the checkpoint of your fine-tuned model
3. Run the main script with the path to your configuration file:

```
python source.main.py /path/to/your/config.json
```

The evaluation process:
- Processes images from the test dataset
- Optionally uses context images to guide the model
- Computes metrics including accuracy, F1 score, and classification accuracy
- Records true positives, true negatives, false positives, and false negatives
- Outputs a log file with the evaluation results

## Testing on Individual Images

For testing the model on individual images, use the `test.py` script:

```
python -m source.test --img /path/to/your/image.jpg
```

By default, the script uses the Qwen2.5-VL-3B-Instruct model with a predefined adapter. You can modify the script to use different models or adapters.

## Project Structure

- `source/`: Main source code directory
  - `main.py`: Entry point for training and evaluation
  - `train.py`: Training pipeline
  - `evaluate.py`: Evaluation pipeline
  - `test.py`: Script for testing individual images
  - `parse_dataset.py`: Dataset conversion utilities
  - `utils.py`: Utility functions for image processing

- `configs/`: Directory for configuration files

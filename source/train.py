import shutil
from pathlib import Path

import torch

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from source.parse_dataset import convert_dataset


def train(config):
    logger = get_logger()
    output_dir = config.get('tmp_folder', (Path(config['config_path']).parent / Path(config['config_path']).name.split('.j')[0]))
    output_dir.mkdir(exist_ok=False)

    dataset_path = convert_dataset(config['dataset_path'], config['prompt'], output_dir=(output_dir/'dataset'), img_type=config['img_type'], ignore_stones=config.get('ignore_stones', False), dataset_type=config.get('dataset_type', 'old'), downscale=config.get('downscale', False))

    model_id = config['model_id']

    epochs = config.get('epochs', 1)

    system = 'You are a helpful assistant.'

    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.1
    num_proc = 4

    lora_rank = config.get('lora_rank', 8)
    lora_alpha = config.get('lora_alpha', 32)
    per_device_batch_size = config.get('per_device_batch_size', 1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        metric_for_best_model='loss',
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )

    logger.info(f'output_dir: {output_dir}')

    if "meta-llama/Llama-3.2" in model_id:
        model, tokenizer = get_model_tokenizer(model_id, use_hf=True, torch_dtype=torch.float)
    else:
        model, tokenizer = get_model_tokenizer(model_id, use_hf=True)

    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    train_dataset, val_dataset = load_dataset(str(dataset_path), split_dataset_ratio=split_dataset_ratio, num_proc=num_proc, seed=data_seed)
    logger.info(f'train_dataset[0]: {train_dataset[0]}')
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

    shutil.rmtree(output_dir/'dataset')

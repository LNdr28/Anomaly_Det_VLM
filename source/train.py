import os

import torch
from swift import Swift, Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.llm import get_model_tokenizer, get_template, DatasetMeta, register_dataset, load_dataset, \
    EncodePreprocessor, PtEngine, InferRequest, RequestConfig

from swift import Seq2SeqTrainer


def train(model_id, dataset_path, output_dir):
    num_proc = 1
    data_seed = 42

    # Retrieve the model and template, and add a trainable LoRA module
    model, tokenizer = get_model_tokenizer(model_id, use_hf=True)
    template = get_template(model.model_meta.template, tokenizer, ...)

    # Download and load the dataset, and encode the text into tokens
    train_dataset, val_dataset = load_dataset(dataset_path)
    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
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
        num_train_epochs=1,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed
    )

    # Train the model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()

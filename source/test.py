from swift import Swift, Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.llm import get_model_tokenizer, get_template, DatasetMeta, register_dataset, load_dataset, \
    EncodePreprocessor, PtEngine, InferRequest, RequestConfig

dataset_path = "/mnt/2tb-1/louis/data/ImageDataset/ann_new.jsonl"
output_dir = "/home/louis/workspace/Anomaly_Det_VLM/out"
dataset_id = "gravis-excavation"
model_id = "deepseek-ai/deepseek-vl2-tiny"


def train():
    num_proc = 1
    data_seed = 42

    # Retrieve the model and template, and add a trainable LoRA module
    model, tokenizer = get_model_tokenizer(model_id)
    template = get_template(model.model_meta.template, tokenizer, ...)

    # Download and load the dataset, and encode the text into tokens
    train_dataset, val_dataset = load_dataset(dataset_id)
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
        data_seed=data_seed,
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


def infer():
    max_new_tokens = 512
    temperature = 0
    # Perform inference using the native PyTorch engine
    engine = PtEngine(model_id, max_batch_size=2, use_hf=True)
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

    messages = [{"role": "system", "content": "You are a professional anomaly detection and classification tool that detects objects that could prevent an excavator from digging."},
                {"role": "user", "content": "<image>This is an image of a trench that has been dug by an excavator. Does the trench contain any objects that could hinder excavation? Common examples of anomalies are pipes, cables, wires, tools, large stones and wooden planks. Provide only the english names of the objects that you detect in the trench as a list separated by commas. If you only see objects like a trench, dirt, gravel, part of an excavator or a whole excavator, you ignore them and return an empty list: []"}]

    images = ["/home/louis/workspace/Anomaly_Det_VLM/custom/rock.png"]

    infer_request = InferRequest(messages=messages, images=images)

    resp_list = engine.infer([infer_request], request_config)
    print(f'response: {resp_list[0].choices[0].message.content}')


if __name__ == "__main__":

    dataset_meta = DatasetMeta(
        dataset_path=dataset_path,  # Your dataset file
    )

    register_dataset(dataset_meta)

    dataset = load_dataset(dataset_path)  # Load dataset using its path
    print(dataset[0])  # Check first example
    infer()

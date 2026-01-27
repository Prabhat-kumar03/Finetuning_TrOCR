import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Set special tokens for Hindi
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

import pandas as pd

base_path = os.getcwd()
train_data = pd.read_csv("./HindiSeg/train.csv")
val_data = pd.read_csv("./HindiSeg/val.csv")

def preprocess_data(example):
    try:
        image = Image.open(os.path.join(base_path, example["path"])).convert("RGB")
    except Exception as e:
        print(f"Error loading image {os.path.join(base_path, example['path'])}: {e}")
        return None
    
    # Encode the image (processor handles resizing)
    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
    
    # Encode the target text
    labels = processor.tokenizer(example["label"], padding="max_length", truncation=True, max_length=128).input_ids
    
    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# Convert to Hugging Face Dataset
hf_train_dataset = Dataset.from_pandas(train_data)
hf_train_dataset = hf_train_dataset.map(preprocess_data, remove_columns=["path", "label"])

hf_val_dataset = Dataset.from_pandas(val_data)
hf_val_dataset = hf_val_dataset.map(preprocess_data, remove_columns=["path", "label"])


# training args : 
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-hindi",
    per_device_train_batch_size=6,       # can try 6 if GPU memory allows
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,       # effective batch size doubled
    learning_rate=5e-5,
    fp16=True,                            # mixed precision for T4
    num_train_epochs=10,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="steps",
    dataloader_num_workers=2,            # CPU-bound, avoid too many workers
    predict_with_generate=True,
    remove_unused_columns=False
)

# compute metrics function
import evaluate
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=labels_str)
    logging.info(f"Computed CER: {cer}")
    return {"cer": cer}

#initiizing trainer 
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train_dataset,   
    eval_dataset=hf_val_dataset,    
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

logging.info("Starting training...")
trainer.train()
logging.info("Training completed.")

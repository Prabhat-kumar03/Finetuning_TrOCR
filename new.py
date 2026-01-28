import os
import torch
import evaluate
import pandas as pd
import numpy as np

from PIL import Image
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


# -----------------------------
# GPU CHECK
# -----------------------------
DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA is not available"


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "microsoft/trocr-base-handwritten"

DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR   = "/home/azureuser/hindi_ocr/dataset"   # IMPORTANT (see CSV paths)
OUTPUT_DIR  = "/mnt/blob/checkpoints"

MAX_LABEL_LENGTH = 32
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5


# -----------------------------
# LOAD CSV (HEADER-SAFE)
# -----------------------------
def load_csv(csv_path):
    df = pd.read_csv(
        csv_path,
        header=0,                       # EXPLICIT: first row is header
        usecols=["file_name", "text"],  # enforce schema
    )
    return Dataset.from_pandas(df, preserve_index=False)


train_ds = load_csv(os.path.join(DATASET_DIR, "train.csv"))
val_ds   = load_csv(os.path.join(DATASET_DIR, "val.csv"))


# -----------------------------
# MODEL & PROCESSOR
# -----------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.to(DEVICE)


# -----------------------------
# PREPROCESS (TEXT ONLY)
# -----------------------------
def preprocess(batch):
    encoding = processor.tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
        add_special_tokens=True,
    )

    labels = [
        token if token != processor.tokenizer.pad_token_id else -100
        for token in encoding.input_ids
    ]

    return {
        "file_name": batch["file_name"],
        "labels": labels,
    }


train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess, remove_columns=val_ds.column_names)


# -----------------------------
# DATA COLLATOR (ENCODER-SAFE)
# -----------------------------
class TrOCRDataCollator:
    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir

    def __call__(self, features):
        images = []
        labels = []

        for item in features:
            image_path = os.path.join(self.image_dir, item["file_name"])
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            labels.append(item["labels"])

        if len(images) == 0:
            raise ValueError("Empty batch encountered")

        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


data_collator = TrOCRDataCollator(processor, IMAGE_DIR)


# -----------------------------
# METRIC (CER)
# -----------------------------
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    labels = np.where(
        labels != -100,
        labels,
        processor.tokenizer.pad_token_id
    )

    pred_str  = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(
        predictions=pred_str,
        references=label_str
    )

    return {"cer": cer}


# -----------------------------
# TRAINING ARGS (SPOT-VM SAFE)
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,

    fp16=True,
    logging_steps=100,

    predict_with_generate=True,

    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,

    dataloader_num_workers=0,
    remove_unused_columns=False,

    report_to="none",
    seed=42,
)


# -----------------------------
# TRAINER
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# -----------------------------
# RESUME-SAFE TRAINING
# -----------------------------
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

print("🚀 Training started...")
trainer.train(resume_from_checkpoint=last_checkpoint)


# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("✅ Training complete. Model saved to:", OUTPUT_DIR)

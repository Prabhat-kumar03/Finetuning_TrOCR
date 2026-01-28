import os
import torch
import evaluate
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig
)
 
# Helper function for shifting tokens
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
 
    if pad_token_id is None:
        return shifted_input_ids
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids
 
# -----------------------------
# GPU configuration
# -----------------------------
DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA not available"
 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
 
# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "microsoft/trocr-base-handwritten"
 
DATASET_DIR = "/home/azureuser/hindi_ocr/dataset"
IMAGE_DIR = "/home/azureuser/hindi_ocr/dataset/HindiSeg"
OUTPUT_DIR = "/mnt/blob/checkpoints"
 
MAX_LABEL_LENGTH = 32
BATCH_SIZE = 16          # Safe for T4 (28GB RAM)
EPOCHS = 10
LEARNING_RATE = 5e-5
DATALOADER_WORKERS = 0   # Set to 0 to avoid multiprocessing issues
 
# -----------------------------
# Load CSV datasets
# -----------------------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["file_name", "text"])
    return Dataset.from_pandas(df)
 
train_ds = load_csv(os.path.join(DATASET_DIR, "train.csv"))
val_ds   = load_csv(os.path.join(DATASET_DIR, "val.csv"))
 
# -----------------------------
# Load processor & model
# -----------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
 
# -----------------------------
# Generation config (FIXES YOUR ERROR)
# -----------------------------
gen_config = GenerationConfig(
    max_length=MAX_LABEL_LENGTH,
    num_beams=1,
    early_stopping=False,
    # These three lines are the critical fix:
    decoder_start_token_id=processor.tokenizer.cls_token_id,
    bos_token_id=processor.tokenizer.cls_token_id,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.sep_token_id,
)
 
model.generation_config = gen_config

# Also ensure the model config matches (redundancy for safety)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
 
# -----------------------------
# Dataset preprocessing
# -----------------------------
def preprocess(batch):
    # Filter out header row if it exists in the data
    # This prevents the FileNotFoundError for '.../file_name'
    if batch["file_name"] == "file_name":
        return {"file_name": None, "labels": None}

    encoding = processor.tokenizer(
        batch["text"],
        padding="max_length",
        max_length=MAX_LABEL_LENGTH,
        truncation=True,
        add_special_tokens=True
    )
 
    labels = [
        token if token != processor.tokenizer.pad_token_id else -100
        for token in encoding.input_ids
    ]
 
    return {
        "file_name": batch["file_name"],
        "labels": labels
    }

# 1. Map and then filter out the nullified header rows
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
train_ds = train_ds.filter(lambda x: x["file_name"] is not None)

val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)
val_ds = val_ds.filter(lambda x: x["file_name"] is not None)

class TrOCRDataCollator:
    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir
 
    def __call__(self, features):
        images = []
        labels_list = []
 
        for item in features:
            image_path = os.path.join(self.image_dir, item["file_name"])
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                labels_list.append(item["labels"])
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                continue
 
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
 
        # The model handles decoder_input_ids automatically from labels!
        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor
        }


# -----------------------------
# Compute metrics (CER)
# -----------------------------
import numpy as np
cer_metric = evaluate.load("cer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # If predictions are logits, take argmax
    if predictions.ndim == 3:
        predictions = predictions.argmax(axis=-1)

    # Replace -100 in labels so tokenizer can decode
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_preds = processor.batch_decode(
        predictions,
        skip_special_tokens=True
    )

    decoded_labels = processor.batch_decode(
        labels,
        skip_special_tokens=True
    )

    # Compute CER
    cer = cer_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {"cer": cer}



# Create data collator
data_collator = TrOCRDataCollator(processor, IMAGE_DIR)

# Training arguments (GPU)
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="/mnt/blob/checkpoints",

    # Sync these two strategies
    eval_strategy="steps",      # Added: ensures evaluation happens at intervals
    eval_steps=500,             # Added: matches save_steps for best model tracking
    
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,

    fp16=True,
    bf16=False,

    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,

    warmup_ratio=0.1,
    weight_decay=0.01,
    label_smoothing_factor=0.1,

    do_eval=True,
    logging_steps=100,

    predict_with_generate=True,

    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,

    remove_unused_columns=False,
    report_to="none",
    seed=42
)

 
# -----------------------------
# Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

 
# -----------------------------
# Train
# -----------------------------
print("🚀 Training started (GPU – Azure T4)...")
# Check if a checkpoint exists in OUTPUT_DIR before starting
import glob
last_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if checkpoints:
        last_checkpoint = True # Tells trainer to find the latest automatically

print("🚀 Training started (GPU – Azure T4)...")
# Pass the resume flag here
trainer.train(resume_from_checkpoint=last_checkpoint)

 
# -----------------------------
# Save final model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
 
print("✅ Training complete. Model saved to:", OUTPUT_DIR)
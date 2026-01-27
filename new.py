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
OUTPUT_DIR = "trocr-hindi-gpu"
 
MAX_LABEL_LENGTH = 32
BATCH_SIZE = 16          # Safe for T4 (28GB RAM)
EPOCHS = 10
LEARNING_RATE = 5e-5
DATALOADER_WORKERS = 0   # Set to 0 to avoid multiprocessing issues
 
# -----------------------------
# Load CSV datasets
# -----------------------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
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
    num_beams=1,              # greedy decoding
    early_stopping=False,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.sep_token_id,
)
 
model.generation_config = gen_config
 
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size
 
# -----------------------------
# Dataset preprocessing
# -----------------------------
def preprocess(batch):
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
        "file_name": batch["path"],
        "labels": labels
    }
 
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess, remove_columns=val_ds.column_names)
 
# -----------------------------
# Lazy image-loading data collator
# -----------------------------
class TrOCRDataCollator:
    def __init__(self, processor, image_dir):
        self.processor = processor
        self.image_dir = image_dir
 
    def __call__(self, batch):
        images = []
        labels = []
 
        for item in batch:
            image_path = os.path.join(self.image_dir, item["path"])
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            labels.append(item["label"])
 
        pixel_values = self.processor(
            images=images,
            return_tensors="pt"
        ).pixel_values
 
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        decoder_input_ids = shift_tokens_right(labels_tensor, self.processor.tokenizer.pad_token_id, self.processor.tokenizer.cls_token_id)
 
        return {
            "pixel_values": pixel_values,
            "labels": labels_tensor,
            "decoder_input_ids": decoder_input_ids
        }
 
data_collator = TrOCRDataCollator(processor, IMAGE_DIR)
 
# -----------------------------
# Metrics (CER)
# -----------------------------
cer_metric = evaluate.load("cer")
 
def compute_metrics(eval_pred):
    import numpy as np
 
    preds, labels = eval_pred
 
    decoded_preds = processor.batch_decode(
        preds, skip_special_tokens=True
    )
 
    labels = np.array(labels)
    labels[labels == -100] = processor.tokenizer.pad_token_id
 
    decoded_labels = processor.batch_decode(
        labels, skip_special_tokens=True
    )
 
    cer = cer_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
 
    return {"cer": cer}
 
# -----------------------------
# Training arguments (GPU)
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
 
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
 
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
 
    fp16=True,                 # T4 optimized
    bf16=False,
 
    dataloader_num_workers=DATALOADER_WORKERS,
    dataloader_pin_memory=True,
 
    warmup_ratio=0.1,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
 
    do_eval=True,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=100,
 
    save_total_limit=2,
    predict_with_generate=True,
 
    metric_for_best_model="cer",
    greater_is_better=False,
 
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
trainer.train()
 
# -----------------------------
# Save final model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
 
print("✅ Training complete. Model saved to:", OUTPUT_DIR)
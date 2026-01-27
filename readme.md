# Finetuning TrOCR for Hindi Handwritten Text Recognition

This repository contains code to finetune Microsoft's TrOCR (Transformer-based OCR) model for recognizing Hindi handwritten text. The project is optimized for Azure VMs with T4 GPUs.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Features

- Finetunes TrOCR-base-handwritten model on custom Hindi datasets
- Supports CSV-based data with image paths and labels
- Optimized for GPU training with mixed precision (FP16)
- Includes evaluation with Character Error Rate (CER)
- Comprehensive logging for setup and training
- Ready for Azure VM deployment

## Requirements

- Ubuntu-based system
- Python 3.8+
- CUDA-compatible GPU (e.g., NVIDIA T4)
- 28GB+ RAM recommended
- 4+ vCPUs

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Finetuning_TrOCR
   ```

2. Run the setup script:
   ```bash
   bash Scripts/requirements.sh
   ```

   This script installs all dependencies, sets up a virtual environment, and verifies GPU availability. Logs are saved to `setup.log`.

3. Activate the environment:
   ```bash
   source trocr_env/bin/activate
   ```

## Data Preparation

1. Prepare your dataset in CSV format with columns: `path` (image path) and `label` (text).

2. Place images in a directory structure (e.g., `HindiSeg/train/...`).

3. Update CSV paths in `finetune.py`:
   - `train_data = pd.read_csv("./HindiSeg/train.csv")`
   - `val_data = pd.read_csv("./HindiSeg/val.csv")`

Ensure image paths in CSV are relative to the workspace root.

## Training

1. Ensure data is prepared and CSVs exist.

2. Run the training script:
   ```bash
   python finetune.py
   ```

   Training parameters (in `finetune.py`):
   - Batch size: 6 (adjust for GPU memory)
   - Epochs: 10
   - Learning rate: 5e-5
   - Mixed precision: Enabled
   - Logging: Every 100 steps
   - Evaluation: Every 500 steps

   Checkpoints are saved to `./trocr-hindi/`.

## Evaluation

The model is evaluated using Character Error Rate (CER) on the validation set. Metrics are logged during training and saved to `training.log`.

## Usage

After training, load the model for inference:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("./trocr-hindi")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-hindi")

image = Image.open("path/to/image.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Logging

- Setup logs: `Scripts/setup.log`
- Training logs: `training.log`
- Training metrics: TensorBoard logs in `./trocr-hindi/runs/`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push and create a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.
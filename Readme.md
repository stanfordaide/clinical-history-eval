# Clinical History Evaluation

Official Implementation by the Stanford AI Development and Evaluation Lab

- **Title:** Assessing the Completeness of Clinical Histories Accompanying Imaging Orders using Open- and Closed-Source Large Language Models
- **Authors:** David Larson, [Arogya Koirala](mailto:arogya@stanford.edu), Lina Chuey, Magdalini Paschali, Dave Van Veen, Hye Sun Na, Matthew Petterson, Zhongnan Fang, Akshay S. Chaudhari
- **Contact:** arogya@stanford.edu

![Clinical History Evaluation Workflow](assets/fig-cam-ready.png)


## 1. Overview

This repository contains code for evaluating the completeness of clinical histories using large language models. The system analyzes clinical text and extracts key components including:

- Past Medical History (PMH): aspects of medical history relevant to the clinical scenario
- What: relevant signs and symptoms prompting the imaging request
- When: time course of inciting event
- Where: localization (if applicable)
- Clinical Concern: diagnostic entities that referring clinician wants to be evaluated

## 2. Setup

### 2.1 Conda Environment
Create and activate conda environment

```bash
conda create -n clinical-history-eval python=3.10
conda activate clinical-history-eval
conda install pip
pip install -e .
```

### 2.2 Hugging Face Authentication
This project uses models from Hugging Face Hub. You'll need to authenticate to access them:

1. Create a Hugging Face account at [https://huggingface.co/](https://huggingface.co/)
2. Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Install huggingface-cli: `pip install huggingface_hub[cli]`
4. Login using: `huggingface-cli login`


### Optional: Change Hugging Face Cache Directory

If you encounter disk quota issues with the default Hugging Face cache directory, you can set the `HF_HOME` environment variable to specify an alternative cache location.

```bash
export HF_HOME=/path/to/new/cache/directory
```

Notes:

1. Consider adding the above line to your shell profile (`~/.bashrc` or `~/.zshrc`) to make the change persistent.
2. After the change, you will need to log in using huggingface-cli again: `huggingface-cli login`

## 3. Dataset

This repository includes a development dataset of 125 synthetic clinical histories located in the `data` folder. These histories were generated using GPT-4 and are provided for development and testing purposes only. Important notes about the dataset:

- The synthetic histories do not reflect actual clinical observations or real patient data
- They likely contain errors and have not been validated by clinical professionals
- They should not be used for clinical decision making or research conclusions 
- Users should replace this data with properly sourced clinical histories from their own institution when using this code for research or clinical applications

The synthetic dataset is provided solely to demonstrate the functionality of the code and enable initial testing. For any meaningful application, users should obtain and use appropriate clinical data following their institution's protocols and ethical guidelines.

## 4. Usage

### 4.1 Inference
Run inference on clinical histories using pre-trained models to extract key components.

#### 4.1.1 Single Text Inference
Process a single clinical history text and output structured components.

```python
python inference.py \
    --text="Relevant PMH ESRD. Presents with L foot gangrene for a duration of 3 weeks. Specific location of issue (if applicable): 2nd digit of L foot. Concern for gangrene." \
    --output-dir="outputs/inference/single-text"
```

#### 4.1.2 Batch Inference from CSV
Process multiple clinical histories from a CSV file in batch mode.
```python
python inference.py \
    --csv="data/example-inference.csv" \
    --output-dir="outputs/inference/csv"
```

#### 4.1.3 In-Context Learning (ICL)
Enable dynamic example selection to improve inference quality using similar cases.

Add these flags to enable ICL:
```python
--use-icl \
--icl-data="data/train.csv" \
--n-icl-examples=4 \
--index-path="outputs/vectordb/db"
```

#### 4.1.4 Using Locally Finetuned Model
After finetuning the model locally, you can use the finetuned model for inference by specifying the `--peft-model` parameter with the path to your finetuned model directory.

**Example: Batch Inference with Finetuned Model**
```python
python inference.py \
    --csv="data/example-inference.csv" \
    --output-dir="outputs/inference/csv" \
    --peft-model="outputs/finetuning"
```

**Example: Single Text Inference with Finetuned Model**
```python
python inference.py \
    --text="Relevant PMH ESRD. Presents with L foot gangrene for a duration of 3 weeks. Specific location of issue (if applicable): 2nd digit of L foot. Concern for gangrene." \
    --output-dir="outputs/inference/single-text" \
    --peft-model="outputs/finetuning"
```

This allows the inference script to utilize the adaptations made during finetuning, potentially improving extraction accuracy based on the specialized training data.

### 4.2 Finetuning
Adapt the model to your specific use case or dataset through additional training.

#### 4.2.1 Basic Finetuning
Simple finetuning configuration for quick model adaptation.

```python
python finetuning.py \
    --train-data="data/train.csv" \
    --val-data="data/valid.csv" \
    --output-dir="outputs/finetuning" \
    --batch-size=4 \
    --epochs=3
```

#### 4.2.2 Advanced Finetuning Options
Extended configuration options for more control over the training process.

```python
python finetuning.py \
    --train-data="data/train.csv" \
    --val-data="data/valid.csv" \
    --output-dir="outputs/finetuning" \
    --batch-size=4 \
    --grad-accum=4 \
    --lr=2e-4 \
    --epochs=3 \
    --lora-r=64 \
    --lora-alpha=128 \
    --lora-dropout=0.05 \
    --load-in-8bit
```

### 4.3 Evaluation
Run evaluation on test data to measure model performance using BERTScore metrics.

#### 4.3.1 Basic Evaluation
Evaluate model performance on test data:

```python
python evaluation.py \
    --test-data="data/test.csv" \
    --output-dir="outputs/evaluation"
```

#### 4.3.2 Evaluation with Finetuned Model
Evaluate performance using a locally finetuned model:

```python
python evaluation.py \
    --test-data="data/test.csv" \
    --peft-model="outputs/finetuning" \
    --output-dir="outputs/evaluation"
```

#### 4.3.3 Sample Evaluation Results
The evaluation script calculates BERTScore F1 scores for each component:

Base Model Results:
```
Average BERTScore F1: 0.7671
PMH F1: 0.7585
WHAT F1: 0.6362
WHEN F1: 0.8284
WHERE F1: 0.7520
CONCERN F1: 0.8606
```

Finetuned Model Results:
```
Average BERTScore F1: 0.7321
PMH F1: 0.8260
WHAT F1: 0.6213
WHEN F1: 0.8274
WHERE F1: 0.6456
CONCERN F1: 0.7400
```

## 5. Key Parameters

### 5.1 Inference Parameters
- `--base-model`: Base model to use (default: mistralai/Mistral-7B-v0.1)
- `--peft-model`: PEFT model path (default: akoirala/clinical-history-eval)
- `--max-length`: Maximum length for generated text (default: 500)
- `--temperature`: Temperature for text generation (default: 0.7)
- `--device`: Device to use (cuda/cpu/auto)
- `--load-in-8bit`: Enable 8-bit quantization
- `--load-in-4bit`: Enable 4-bit quantization
- `--use-flash-attention`: Enable flash attention when available

### 5.2 Finetuning Parameters
- `--train-data`: Path to training data CSV (required)
- `--val-data`: Path to validation data CSV (optional)
- `--template`: Path to Jinja template file (default: template.jinja)
- `--batch-size`: Training batch size (default: 4)
- `--grad-accum`: Gradient accumulation steps (default: 4)
- `--lr`: Learning rate (default: 2e-4)
- `--epochs`: Number of epochs (default: 3)
- `--lora-r`: LoRA attention dimension (default: 64)
- `--lora-alpha`: LoRA alpha parameter (default: 128)
- `--lora-dropout`: LoRA dropout value (default: 0.05)



## 6. Using Slurm

Both inference and finetuning scripts can be run using Slurm, just pass the python module name to the script. Example usage:

```bash
# For inference
sbatch -c 8 --gres=gpu:l40:1 --time=0 slurm.sh \
    --module=inference \
    --csv=path/to/input.csv \
    --output-dir=test_output

# For finetuning
sbatch -c 8 --gres=gpu:l40:1 --time=12:00:00 slurm.sh \
    --module=finetuning \
    --train-data=path/to/train.csv \
    --val-data=path/to/val.csv \
    --output-dir=finetuning_output
```

## 7. Output Format

Results are saved as JSON files in the specified output directory:
```json
{
  "status": "success",
  "input_prompt": "...",
  "generated_text": "...",
  "parsed_output": {
    "pmh": "...",
    "what": "...",
    "when": "...",
    "where": "...",
    "cf": "..."
  },
  "inference_stats": {
    "time_seconds": 1.23,
    "device": "cuda:0",
    "timestamp": "2024-03-21 10:30:45"
  }
}
```

## 8. Contributing

Please feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 9. Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{larson2024assessing,
    title={Assessing the Completeness of Clinical Histories Accompanying Imaging Orders using Open- and Closed-Source Large Language Models},
    author={Larson, David and Koirala, Arogya and Chuey, Lina and Paschali, Magdalini and Van Veen, Dave and Na, Hye Sun and Petterson, Matthew and Fang, Zhongnan and Chaudhari, Akshay S.},
    journal={Radiology},
    year={2024},
    publisher={Radiological Society of North America}
}
```

## 10. License

Copyright [2024] [Stanford AI Development and Evaluation Lab]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
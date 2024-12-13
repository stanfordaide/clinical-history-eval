#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from utils.dataloader import HistoryLoader
from utils.model import HistoryEvalModel, ModelConfig
from peft import LoraConfig, get_peft_model
import warnings
from transformers import logging as transformers_logging
import os

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    # Suppress warnings
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Configure logging format
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune clinical history model')
    
    # Data configuration
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data CSV')
    parser.add_argument('--val-data', type=str,
                       help='Path to validation data CSV')
    parser.add_argument('--template', type=str, 
                       default="template.jinja",
                       help='Path to Jinja template file')
    
    # Model configuration
    parser.add_argument('--base-model', type=str,
                       default="mistralai/Mistral-7B-v0.1",
                       help='Base model to use')
    parser.add_argument('--peft-model', type=str,
                       default="akoirala/clinical-history-eval",
                       help='PEFT model to use as starting point')
    
    # Training configuration
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--grad-accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    
    # LoRA configuration
    parser.add_argument('--lora-r', type=int, default=64,
                       help='LoRA attention dimension')
    parser.add_argument('--lora-alpha', type=int, default=128,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                       help='LoRA dropout value')
    
    # Device configuration
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit precision')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit precision')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def prepare_dataset(data_path: str, history_loader: HistoryLoader) -> Dataset:
    """Prepare dataset for training."""
    # Generate prompts for all examples
    prompts = []
    for idx, row in history_loader.data.iterrows():
        prompt = history_loader.prepare_prompt(row, idx)
        prompts.append({
            "text": prompt
        })
    
    return Dataset.from_list(prompts)

def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Initialize model configuration
    model_config = ModelConfig(
        base_model=args.base_model,
        peft_model="akoirala/clinical-history-eval" if args.peft_model is None else args.peft_model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map="auto"
    )
    
    # Initialize model in training mode
    logger.info("Initializing model...")
    model = HistoryEvalModel(model_config, mode="train", logger=logger)
    base_model = model.get_model()
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Initialize HistoryLoader for training data
    logger.info("Loading training data...")
    train_loader = HistoryLoader(
        data_path=args.train_data,
        template_path=args.template,
        use_icl=False,  # No ICL for training
        finetuning=True
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(args.train_data, train_loader)
    if args.val_data:
        val_loader = HistoryLoader(
            data_path=args.val_data,
            template_path=args.template,
            use_icl=False,
            finetuning=True
        )
        val_dataset = prepare_dataset(args.val_data, val_loader)
    else:
        val_dataset = None
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if val_dataset else "no",
        fp16=True,
        optim="paged_adamw_32bit",
        disable_tqdm=False,  # Set to True if you want to disable progress bars
        report_to="none",   # Disable wandb/tensorboard reporting
        logging_first_step=False,
        logging_nan_inf_filter=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_args,
        tokenizer=model.get_tokenizer(),
        max_seq_length=2048
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import time
import torch
from utils.dataloader import HistoryLoader
from utils.model import HistoryEvalModel, ModelConfig
from utils.parser import OutputParser
import sys

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on clinical history notes')
    
    # Input mode (either direct text or CSV)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Direct text input for inference')
    input_group.add_argument('--csv', type=str, help='Path to CSV file for batch inference')
    
    # Model configuration
    parser.add_argument('--base-model', type=str,
                       default="mistralai/Mistral-7B-v0.1",
                       help='Base model to use')
    parser.add_argument('--peft-model', type=str,
                       default="akoirala/clinical-history-eval",
                       help='PEFT model to use')
    parser.add_argument('--max-length', type=int,
                       default=500,
                       help='Maximum length for generated text')
    parser.add_argument('--temperature', type=float,
                       default=0.7,
                       help='Temperature for text generation')
    
    # Device configuration
    parser.add_argument('--device', type=str,
                       choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='Device to use for inference')
    parser.add_argument('--device-map', type=str,
                       default='auto',
                       help='Device mapping strategy')
    
    # Model loading options
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit precision')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit precision')
    parser.add_argument('--use-flash-attention', action='store_true',
                       help='Use flash attention when available')
    
    # HistoryLoader configuration
    parser.add_argument('--template', type=str, 
                       default="template.jinja",
                       help='Path to Jinja template file')
    parser.add_argument('--output-dir', type=str,
                       default="outputs",
                       help='Directory to save output JSON files')
    
    # ICL configuration
    parser.add_argument('--use-icl', action='store_true',
                       help='Use in-context learning examples')
    parser.add_argument('--icl-data', type=str,
                       help='Path to CSV containing ICL examples')
    parser.add_argument('--n-icl-examples', type=int,
                       default=16,
                       help='Number of ICL examples to use')
    parser.add_argument('--sentence-model', type=str,
                       default='distilbert-base-nli-stsb-mean-tokens',
                       help='SentenceTransformer model for ICL')
    parser.add_argument('--index-path', type=str,
                       default="faiss_index",
                       help='Path to FAISS index for ICL')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    

    # args = parse_args()

    return parser.parse_args()

def initialize_history_loader(args, data_path: str, logger: logging.Logger) -> HistoryLoader:
    """Initialize HistoryLoader with proper configuration."""
    logger.info(f"Initializing HistoryLoader{'with ICL' if args.use_icl else ''}")
    try:
        loader = HistoryLoader(
            data_path=data_path,
            template_path=args.template,
            use_icl=args.use_icl,
            icl_data_path=args.icl_data,
            n_icl_examples=args.n_icl_examples,
            model_name=args.sentence_model,
            index_path=args.index_path
        )
        if args.use_icl:
            logger.info(f"ICL configured with {args.n_icl_examples} examples")
        return loader
    except Exception as e:
        logger.error(f"Failed to initialize HistoryLoader: {str(e)}")
        raise

def detect_csv_format(csv_path: str) -> str:
    """
    Detect the format of input CSV file.
    Returns: 'simple' for history-only CSV, 'full' for complete dataset format
    """
    df = pd.read_csv(csv_path, nrows=1)
    required_columns = ['entry_id', 'history', 'pmh', 'what', 'when', 'where', 'concern']
    
    # Check if all required columns exist
    if all(col in df.columns for col in required_columns):
        return 'full'
    
    # For simple format, we expect either a single unnamed column or a column named 'history'
    if len(df.columns) == 1:
        return 'simple'
    elif 'history' in df.columns:
        return 'simple'
    
    raise ValueError("Unsupported CSV format. Expected either full dataset format or simple history-only format.")


def process_single_text(
    text: str,
    model: HistoryEvalModel,
    history_loader: HistoryLoader,
    max_length: int,
    temperature: float,
    logger: logging.Logger
) -> dict:
    """Process a single text input."""
    logger.info("Processing single text input")
    
    # Use the first row from the loaded data
    row = history_loader.data.iloc[0]
    
    start_time = time.time()
    prompt = history_loader.prepare_prompt(row)
    result = model.generate(
        prompt,
        max_length=max_length,
        temperature=temperature
    )
    inference_time = time.time() - start_time
    
    parser = OutputParser()
    parsed_output = parser.parse(result['generated_text'])
    parsed_output['input'] = text  # Use the original input text
    result['parsed_output'] = parsed_output
    result['inference_stats'] = {
        'time_seconds': inference_time
    }
    
    return result

def process_simple_csv(
    csv_path: str,
    model: HistoryEvalModel,
    history_loader: HistoryLoader,
    max_length: int,
    temperature: float,
    output_dir: str,
    logger: logging.Logger
):
    """Process simple CSV file with just history texts."""
    logger.info(f"Processing simple CSV file: {csv_path}")
    
    # Read CSV and handle both possible formats
    df = pd.read_csv(csv_path)
    if len(df.columns) == 1:
        df.columns = ['history']
    
    # Add required columns for compatibility
    df['entry_id'] = [f'entry_{i}' for i in range(len(df))]
    for col in ['pmh', 'what', 'when', 'where', 'concern']:
        df[col] = ''
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_entries = len(df)
    logger.info(f"Found {total_entries} entries to process")
    
    for idx, row in df.iterrows():
        logger.debug(f"Processing entry {idx+1}/{total_entries}")
        
        start_time = time.time()
        prompt = history_loader.prepare_prompt(row, idx)
        result = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        inference_time = time.time() - start_time
        
        # Parse the output
        parser = OutputParser()
        result['parsed_output'] = parser.parse(result['generated_text'])
        
        # Record inference stats
        result['inference_stats'] = {
            'time_seconds': inference_time
        }
        total_time += inference_time
        
        # Add metadata
        result['metadata'] = {
            'entry_id': row['entry_id'],
            'original_text': row['history'],
            'row_index': idx,
            'inference_stats': result['inference_stats']
        }
        
        # Save individual JSON file
        output_file = output_path / f"{row['entry_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed entry {idx+1}/{total_entries} (took {inference_time:.2f}s)")
    
    # Save summary statistics
    summary = {
        'total_entries': total_entries,
        'total_time': total_time,
        'average_time': total_time / total_entries,
        'device': str(model.model_device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': csv_path,
        'format': 'simple'
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def process_csv(
    csv_path: str,
    model: HistoryEvalModel,
    history_loader: HistoryLoader,
    max_length: int,
    temperature: float,
    output_dir: str,
    logger: logging.Logger
):
    """Process CSV file row by row and save individual JSON files."""
    logger.info(f"Processing CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_time = 0
    total_entries = len(df)
    logger.info(f"Found {total_entries} entries to process")
    
    for idx, row in df.iterrows():
        logger.debug(f"Processing entry {idx+1}/{total_entries}: {row['entry_id']}")
        
        start_time = time.time()
        prompt = history_loader.prepare_prompt(row, idx)
        result = model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        inference_time = time.time() - start_time
        
        # Parse the output
        parser = OutputParser()
        result['parsed_output'] = parser.parse(result['generated_text'])
        

        args = parse_args()
        # Record inference stats
        result['inference_stats'] = {
            'time_seconds': inference_time,
            'icl_examples': args.n_icl_examples if args.use_icl else 0
        }
        total_time += inference_time
        
        # Add metadata
        result['metadata'] = {
            'entry_id': row['entry_id'],
            'original_text': row['history'],
            'ground_truth': {
                'pmh': row['pmh'],
                'what': row['what'],
                'when': row['when'],
                'where': row['where'],
                'concern': row['concern']
            },
            'row_index': idx,
            'inference_stats': result['inference_stats']
        }
        
        # Save individual JSON file
        output_file = output_path / f"{row['entry_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Completed entry {row['entry_id']} (took {inference_time:.2f}s)")
    
    # Save summary statistics
    summary = {
        'total_entries': total_entries,
        'total_time': total_time,
        'average_time': total_time / total_entries,
        'device': str(model.model_device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': csv_path,
        'model_config': {
            'base_model': model.config.base_model,
            'peft_model': model.config.peft_model,
            'device': str(model.model_device),
            'quantization': '8-bit' if model.config.load_in_8bit else 
                          '4-bit' if model.config.load_in_4bit else 'none'
        },
        'icl_config': {
            'enabled': args.use_icl,
            'examples': args.n_icl_examples if args.use_icl else 0,
            'sentence_model': args.sentence_model if args.use_icl else None
        }
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average time per entry: {total_time/total_entries:.2f}s")

def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Initialize model configuration
    model_config = ModelConfig(
        base_model=args.base_model,
        peft_model=args.peft_model,
        device=None if args.device == 'auto' else args.device,
        device_map=args.device_map if args.device == 'auto' else None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attention=args.use_flash_attention
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HistoryEvalModel(model_config, mode="inference", logger=logger)
    logger.info(f"Model initialized on device: {model.model_device}")
    
    if args.text:
        # Single text mode
        temp_df = pd.DataFrame({
            'history': [args.text],
            'entry_id': ['inference'],
            'exp': [''],
            'pmh': [''],
            'what': [''],
            'when': [''],
            'where': [''],
            'concern': ['']
        })
        temp_csv = "temp_data.csv"
        temp_df.to_csv(temp_csv, index=False)
        
        try:
            history_loader = initialize_history_loader(args, temp_csv, logger)
            result = process_single_text(
                text=args.text,
                model=model,
                history_loader=history_loader,
                max_length=args.max_length,
                temperature=args.temperature,
                logger=logger
            )
            
            # Save single result
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / "single_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            logger.info(f"Result saved to {output_file}")
            logger.info(f"Inference time: {result['inference_stats']['time_seconds']:.2f}s")
            
        finally:
            Path(temp_csv).unlink()
            
    else:
        # CSV mode
        csv_format = detect_csv_format(args.csv)
        history_loader = initialize_history_loader(args, args.csv, logger)
        
        if csv_format == 'simple':
            process_simple_csv(
                csv_path=args.csv,
                model=model,
                history_loader=history_loader,
                max_length=args.max_length,
                temperature=args.temperature,
                output_dir=args.output_dir,
                logger=logger
            )
        else:
            process_csv(
                csv_path=args.csv,
                model=model,
                history_loader=history_loader,
                max_length=args.max_length,
                temperature=args.temperature,
                output_dir=args.output_dir,
                logger=logger
            )

if __name__ == "__main__":
    main()
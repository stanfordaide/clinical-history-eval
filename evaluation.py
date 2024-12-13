#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import torch
from bert_score import BERTScorer
from tqdm import tqdm
import numpy as np
from utils.dataloader import HistoryLoader
from utils.model import HistoryEvalModel, ModelConfig
from utils.parser import OutputParser
import time

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate clinical history model')
    
    # Data configuration
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save evaluation results')
    
    # Model configuration
    parser.add_argument('--base-model', type=str,
                       default="mistralai/Mistral-7B-v0.1",
                       help='Base model to use')
    parser.add_argument('--peft-model', type=str,
                       default=None,
                       help='Path to finetuned PEFT model (optional)')
    
    # Inference configuration
    parser.add_argument('--use-icl', action='store_true',
                       help='Use in-context learning')
    parser.add_argument('--n-icl-examples', type=int, default=3,
                       help='Number of ICL examples to use')
    parser.add_argument('--sentence-model', type=str,
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Sentence transformer model for ICL')
    
    # Device configuration
    parser.add_argument('--device', type=str,
                       choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='Device to use for inference')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Load model in 8-bit precision')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Load model in 4-bit precision')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()

def calculate_bertscore(predictions, ground_truth, scorer):
    """Calculate BERTScore for predictions against ground truth."""
    P, R, F1 = scorer.score(predictions, ground_truth)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def evaluate_model(model, test_loader, output_dir, logger):
    """Evaluate model performance using BERTScore."""
    logger.info("Starting evaluation...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data directory for individual outputs
    data_path = output_path / 'data'
    data_path.mkdir(exist_ok=True)
    
    # Initialize BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    
    # Components to evaluate
    components = ['pmh', 'what', 'when', 'where', 'concern']
    results = {comp: [] for comp in components}
    ground_truth = {comp: [] for comp in components}
    
    # Process each test example
    for idx, row in tqdm(test_loader.data.iterrows(), total=len(test_loader.data)):
        # Generate prediction
        prompt = test_loader.prepare_prompt(row, idx)
        start_time = time.time()
        output = model.generate(prompt, max_length=500, temperature=0.7)
        inference_time = time.time() - start_time
        
        # Parse the output
        parser = OutputParser()
        parsed_output = parser.parse(output['generated_text'])
        
        # Store individual result
        result = {
            'generated_text': output['generated_text'],
            'parsed_output': parsed_output,
            'metadata': {
                'entry_id': row.get('entry_id', f'test_{idx}'),
                'original_text': row['history'],
                'ground_truth': {
                    'pmh': row['pmh'],
                    'what': row['what'],
                    'when': row['when'],
                    'where': row['where'],
                    'concern': row['concern']
                },
                'row_index': idx,
                'inference_stats': {
                    'time_seconds': inference_time
                }
            }
        }
        
        # Save individual JSON file
        entry_id = row.get('entry_id', f'test_{idx}')
        with open(data_path / f"{entry_id}.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        # Collect predictions and ground truth for each component
        for comp in components:
            pred = parsed_output.get(comp, "").lower()
            truth = str(row[comp]).lower()
            
            results[comp].append(pred)
            ground_truth[comp].append(truth)
    
    # Calculate BERTScore for each component
    scores = {}
    for comp in components:
        scores[comp] = calculate_bertscore(
            results[comp],
            ground_truth[comp],
            scorer
        )
    
    # Calculate average scores across all components
    avg_scores = {
        'precision': np.mean([scores[comp]['precision'] for comp in components]),
        'recall': np.mean([scores[comp]['recall'] for comp in components]),
        'f1': np.mean([scores[comp]['f1'] for comp in components])
    }
    
    # Update the model_config section in evaluation_results
    evaluation_results = {
        'component_scores': scores,
        'average_scores': avg_scores,
        'model_config': {
            'base_model': model.config.base_model,
            'peft_model': model.config.peft_model if model.config.peft_model else "none",
            'use_icl': test_loader.use_icl,
            'n_icl_examples': test_loader.n_examples if test_loader.use_icl else 0,
            'sentence_model': test_loader.sentence_model if test_loader.use_icl else None
        }
    }
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Log summary results
    logger.info("\nEvaluation Results:")
    logger.info(f"Average BERTScore F1: {avg_scores['f1']:.4f}")
    for comp in components:
        logger.info(f"{comp.upper()} F1: {scores[comp]['f1']:.4f}")
    
    return evaluation_results

def main():
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    # Initialize model configuration
    model_config = ModelConfig(
        base_model=args.base_model,
        peft_model="akoirala/clinical-history-eval" if args.peft_model == None else args.peft_model,  # Can be None
        device=None if args.device == 'auto' else args.device,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HistoryEvalModel(model_config, mode="inference", logger=logger)
    
    # Initialize test data loader with ICL settings
    test_loader = HistoryLoader(
        data_path=args.test_data,
        template_path="template.jinja",
        use_icl=args.use_icl,
        n_icl_examples=args.n_icl_examples if args.use_icl else 0,
        model_name=args.sentence_model if args.use_icl else None
    )
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        output_dir=args.output_dir,
        logger=logger
    )

if __name__ == "__main__":
    main()
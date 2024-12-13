from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer
import logging
import torch

class HistoryLoader:
    def __init__(
        self,
        data_path: str,
        template_path: Optional[str] = "/dataNAS/people/arogya/projects/clinical-history-eval/template.jinja",
        use_icl: bool = False,
        icl_data_path: Optional[str] = None,
        n_icl_examples: int = 16,
        model_name: str = 'distilbert-base-nli-stsb-mean-tokens',
        finetuning: bool = False,
        index_path: Optional[str] = "faiss_index"
    ):
        """
        Initialize History Loader with FAISS-based example selection.
        
        Args:
            data_path: Path to main data CSV
            template_path: Path to Jinja template file. Defaults to templates/default.jinja
            use_icl: Whether to use in-context learning examples
            icl_data_path: Path to CSV containing potential examples
            n_icl_examples: Number of examples to use (default 16)
            model_name: SentenceTransformer model to use for embeddings
            finetuning: Whether to generate data for finetuning
            index_path: Path to load/save FAISS index
        """
        self.logger = logging.getLogger(__name__)
        
        # Load main data
        self.data = pd.read_csv(data_path)
        self.index_path = index_path
        
        # Setup template
        self.finetuning = finetuning
        template_dir = str(Path(template_path).parent)
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.template = self.jinja_env.get_template(Path(template_path).name)

        # ICL setup
        self.use_icl = use_icl
        self.n_icl_examples = n_icl_examples
        
        if use_icl:
            self.logger.info("Initializing HistoryLoaderwith ICL")
            # If icl_data_path not provided, use main data path
            self.icl_data_path = icl_data_path if icl_data_path else data_path
            
            # Load example data
            self.example_data = pd.read_csv(self.icl_data_path)
            
            # Check if using same dataset for examples
            self.using_same_dataset = (Path(data_path).resolve() == Path(self.icl_data_path).resolve())
            
            # Initialize sentence transformer
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Use pytorch device_name: {device_name}")
            self.encoder = SentenceTransformer(model_name).to(device_name)
            
            # Try to load existing index, create new one if not found
            if index_path and Path(index_path).exists():
                self._load_faiss_index()
            else:
                self._create_faiss_index()
                if index_path:
                    self._save_faiss_index()

    def _create_faiss_index(self):
        """Create FAISS index from example data."""
        examples_text = self.example_data['history'].tolist()
        self.example_embeddings = self.encoder.encode(examples_text, show_progress_bar=True)
        faiss.normalize_L2(self.example_embeddings)
        
        # Create optimized index for CPU
        dimension = self.example_embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dimension)  # Base quantizer
        self.index = faiss.IndexIVFFlat(quantizer, dimension, min(self.example_embeddings.shape[0]//4, 100))
        self.index.train(self.example_embeddings)  # Train index
        self.index.nprobe = min(20, self.example_embeddings.shape[0]//4)  # Number of clusters to search
        self.logger.info("Using optimized CPU index with IVF")
        self.index.add(self.example_embeddings)

    def _save_faiss_index(self):
        """Save FAISS index and embeddings to disk."""
        if not self.index_path:
            raise ValueError("No index_path provided")
            
        # Create directory if it doesn't exist
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save both index and embeddings
        faiss.write_index(self.index, f"{self.index_path}.index")
        np.save(f"{self.index_path}.embeddings.npy", self.example_embeddings)

    def _load_faiss_index(self):
        """Load FAISS index and embeddings from disk."""
        if not self.index_path:
            raise ValueError("No index_path provided")
            
        # Load index and embeddings
        self.index = faiss.read_index(f"{self.index_path}.index")
        self.example_embeddings = np.load(f"{self.index_path}.embeddings.npy")
        self.index.nprobe = min(20, self.example_embeddings.shape[0]//4)  # Set search parameters
        self.logger.info("Using optimized CPU index with IVF")

    def get_nearest_examples(self, query_text: str, current_idx: Optional[int] = None) -> List[Dict]:
        """Get nearest examples for a query using FAISS."""
        query_embedding = self.encoder.encode([query_text], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        k = self.n_icl_examples + (1 if self.using_same_dataset else 0)
        _, indices = self.index.search(query_embedding, k)
        indices = indices[0]
        
        if self.using_same_dataset and current_idx is not None:
            indices = [idx for idx in indices if idx != current_idx]
        
        indices = indices[:self.n_icl_examples]
        
        examples = []
        for idx in indices:
            row = self.example_data.iloc[idx]
            example = {
                'input': row['history'],
                'pmh': row['pmh'],
                'what': row['what'],
                'when': row['when'],
                'where': row['where'],
                'concern': row['concern']
            }
            examples.append(example)
            
        return examples

    def prepare_prompt(self, row: pd.Series, idx: Optional[int] = None) -> str:
        """Prepare prompt for a single row."""
        examples = []
        if self.use_icl:
            examples = self.get_nearest_examples(row['history'], idx if self.using_same_dataset else None)
        
        # For finetuning mode, create a clean dict of values
        if self.finetuning:
            row_data = {
                'pmh': row['pmh'],
                'what': row['what'],
                'when': row['when'],
                'where': row['where'],
                'cf': row['concern']
            }
        else:
            row_data = row
            
        return self.template.render(
            examples=examples,
            input=row['history'],
            row=row_data,
            finetuning=self.finetuning
        )

    def get_batch(self, batch_size: int = 32) -> List[Dict]:
        """Get a batch of prompts."""
        batch = self.data.sample(min(batch_size, len(self.data)))
        
        prompts = []
        for idx, row in batch.iterrows():
            prompt_data = {
                'entry_id': row['entry_id'],
                'prompt': self.prepare_prompt(row, idx),
                'metadata': {
                    'historyinal_text': row['history'],
                    'expanded_text': row['exp'],
                    'ground_truth': {
                        'pmh': row['pmh'],
                        'what': row['what'],
                        'when': row['when'],
                        'where': row['where'],
                        'concern': row['concern']
                    }
                }
            }
            prompts.append(prompt_data)
        
        return prompts
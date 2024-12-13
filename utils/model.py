from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import torch
from typing import Optional, Dict, Union, Literal, Any, List
from pathlib import Path
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import time

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    base_model: str = "mistralai/Mistral-7B-v0.1"
    peft_model: Optional[str] = "akoirala/clinical-history-eval"
    # Device configuration
    device: Optional[Union[str, torch.device]] = None
    device_map: Optional[str] = "auto"
    # Model precision options
    torch_dtype: torch.dtype = torch.float16
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_flash_attention: bool = False

    def __post_init__(self):
        """Validate and setup device configuration."""
        if self.device is not None:
            # Convert string device specification to torch.device if needed
            if isinstance(self.device, str):
                self.device = torch.device(self.device)
            # If specific device is provided, disable device_map
            self.device_map = None
        
        # Validate device availability
        if self.device and self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")
        
        # Validate quantization options
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot load model in both 4-bit and 8-bit precision")
        
        # Adjust dtype based on device
        if self.device and self.device.type == "cpu":
            if self.torch_dtype == torch.float16:
                self.torch_dtype = torch.float32
            if self.load_in_4bit or self.load_in_8bit:
                raise ValueError("Quantization is not supported on CPU")

    def get_model_args(self) -> dict:
        """Get all model loading arguments."""
        args = {
            "trust_remote_code": True,
        }
        
        # Device configuration
        if self.device_map is not None:
            args["device_map"] = self.device_map
        elif self.device is not None:
            args["device_map"] = {"": self.device}
        
        # Add dtype if not using quantization
        if not (self.load_in_4bit or self.load_in_8bit):
            args["torch_dtype"] = self.torch_dtype
            
        if self.use_flash_attention:
            args["use_flash_attention_2"] = True
            
        return args

class HistoryEvalModel:
    """
    Unified model class for both inference and fine-tuning.
    Handles model loading, configuration, and basic operations.
    """
    def __init__(
        self,
        config: ModelConfig,
        mode: Literal["inference", "train"] = "inference",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model with given configuration.
        
        Args:
            config: ModelConfig instance with model settings
            mode: Either "inference" or "train"
            logger: Optional logger instance for detailed logging
        """
        self.config = config
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        if self.config.peft_model:
            self.logger.info(f"PEFT model set to: {self.config.peft_model}")
        else:
            self.logger.info("No PEFT model set.")
        
        self._load_tokenizer()
        self._load_model()
        
        # Store device for easy access
        self.device = next(self.model.parameters()).device
        self.logger.info(f"Model loaded on device: {self.device}")

    def _load_tokenizer(self):
        """Load and configure the tokenizer."""
        self.logger.debug(f"Loading tokenizer from {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            padding_side="right",
            trust_remote_code=True
        )
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration based on settings."""
        if self.config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.config.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self.config.torch_dtype,
            )
        return None

    def _load_model(self):
        """Load and configure the model based on mode and settings."""
        try:
            # Get model arguments from config
            model_args = self.config.get_model_args()
            
            # Add quantization config if specified
            quantization_config = self._get_quantization_config()
            if quantization_config is not None:
                model_args["quantization_config"] = quantization_config
            
            self.logger.info(f"Loading base model {self.config.base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_args
            )
            
            # Apply LoRA weights if specified (inference mode only)
            if self.config.peft_model and self.mode == "inference":  # Only apply PEFT in inference mode
                self.logger.info(f"Loading PEFT model {self.config.peft_model}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.peft_model,
                    torch_dtype=self.config.torch_dtype
                )
            
            # Prepare model for training if in training mode
            if self.mode == "train":
                if self.config.load_in_4bit or self.config.load_in_8bit:
                    self.logger.info("Preparing model for kbit training")
                    self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    @contextmanager
    def inference_mode(self):
        """Context manager for inference mode with timing."""
        start_time = time.time()
        torch.cuda.empty_cache()  # Clear CUDA cache before inference
        try:
            with torch.no_grad():
                yield
        finally:
            torch.cuda.empty_cache()  # Clear CUDA cache after inference
            end_time = time.time()
            self.logger.debug(f"Inference took {end_time - start_time:.2f}s")

    def generate(
        self,
        prompt: str,
        max_length: int = 2000,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a single prompt (inference mode only)."""
        if self.mode != "inference":
            raise RuntimeError("Generate method is only available in inference mode")
            
        try:
            # Tokenize single prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            start_time = time.time()
            with self.inference_mode():
                outputs = self.model.generate(**inputs, **generation_config)
            inference_time = time.time() - start_time
            
            return {
                "status": "success",
                "input_prompt": prompt,
                "generated_text": self.tokenizer.decode(outputs[0], skip_special_tokens=True),
                "inference_stats": {
                    "time_seconds": inference_time,
                    "device": str(self.device),
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
                
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
    @property
    def model_device(self) -> torch.device:
        """Get the current device of the model."""
        return self.device

    def get_model(self):
        """Get the underlying model."""
        return self.model
    
    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer
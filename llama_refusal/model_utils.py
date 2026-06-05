"""
Model loading utilities for Llama-3.1-8B-Instruct.
Optimized for Google Colab with NVIDIA T4 GPU and 4-bit quantization.
"""

import os
import gc
import yaml
import logging
from typing import Optional, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from transformer_lens import HookedTransformer
    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    TRANSFORMERLENS_AVAILABLE = False


logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Gets the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU/memory usage statistics."""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        }
    elif torch.backends.mps.is_available():
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "allocated_gb": mem_info.rss / 1024**3,
            "reserved_gb": mem_info.vms / 1024**3,
            "free_gb": 0
        }
    return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}


def log_memory_stats(prefix: str = ""):
    """Log current memory usage."""
    stats = get_memory_stats()
    if torch.cuda.is_available():
        logger.info(f"{prefix}GPU Memory - Allocated: {stats['allocated_gb']:.2f}GB, "
                    f"Reserved: {stats['reserved_gb']:.2f}GB, Free: {stats['free_gb']:.2f}GB")
    else:
        logger.info(f"{prefix}Memory - Allocated: {stats['allocated_gb']:.2f}GB, "
                    f"Reserved: {stats['reserved_gb']:.2f}GB")


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_hf_token() -> str:
    """
    Get HuggingFace token from environment or Colab secrets.
    
    Returns:
        HuggingFace API token
    
    Raises:
        RuntimeError: If the token is not found in the environment.
    """
    # Try environment variable first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        logger.info("Using HuggingFace token from environment variable")
        return token
    
    # Try Google Colab userdata
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            logger.info("Using HuggingFace token from Colab secrets")
            return token
    except ImportError:
        pass  # Not in Colab environment
    except Exception as e:
        logger.warning(f"Could not access Colab secrets: {e}")
    
    # Token not found, raise an error
    raise RuntimeError(
        "HuggingFace token is required. Please set the 'HF_TOKEN' environment variable "
        "or define it in Google Colab secrets."
    )


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_token: Optional[str] = None,
    use_4bit: bool = True,
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Llama-3.1-8B model and tokenizer with optional 4-bit quantization.
    
    Args:
        model_id: HuggingFace model identifier
        hf_token: HuggingFace API token
        use_4bit: Whether to use 4-bit quantization (recommended for T4)
        device: Device to use (auto-detected if None)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    device_obj = get_device() if device is None else torch.device(device)
    logger.info(f"Loading Model: {model_id} on {device_obj}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    log_memory_stats("Before loading: ")
    
    if hf_token is None:
        hf_token = get_hf_token()
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
    except Exception as e:
        logger.warning(f"Login warning: {e}. Attempting to proceed without explicit login...")

    # Configure quantization
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit NF4 quantization")
    
    # Load model
    logger.info("Loading model weights...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=hf_token
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Tokenizer loaded successfully")
    log_memory_stats("After loading: ")
    
    return model, tokenizer


def load_hooked_transformer(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_token: Optional[str] = None,
    use_4bit: bool = True,
    device: Optional[str] = None
) -> Tuple[Any, AutoTokenizer]:
    """
    Load model wrapped with TransformerLens HookedTransformer for interpretability.
    
    Args:
        model_id: HuggingFace model identifier
        hf_token: HuggingFace API token
        use_4bit: Whether to use 4-bit quantization
        device: Device to use
    
    Returns:
        Tuple of (hooked_model, tokenizer)
    """
    base_model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        hf_token=hf_token,
        use_4bit=use_4bit,
        device=device
    )
    
    if not TRANSFORMERLENS_AVAILABLE:
        logger.info("TransformerLens not available, returning base model")
        hooked_model = base_model
    else:
        logger.info("Wrapping model with TransformerLens HookedTransformer...")
        try:
            hooked_model = HookedTransformer.from_pretrained(
                model_id,
                hf_model=base_model,
                device=str(get_device()),
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer,
            )
            logger.info("Successfully wrapped with HookedTransformer")
        except Exception as e:
            logger.warning(f"Could not wrap with HookedTransformer: {e}")
            logger.info("Falling back to manual hook registration (returning base model)")
            hooked_model = base_model
    
    log_memory_stats("After wrapping: ")
    return hooked_model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate text from a prompt using the model.
    """
    messages = [{"role": "user", "content": prompt}]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if formatted_prompt in generated_text:
        generated_text = generated_text[len(formatted_prompt):].strip()
    
    return generated_text


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """Get information about the model architecture."""
    config = model.config
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "num_attention_heads": getattr(config, "num_attention_heads", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "vocab_size": getattr(config, "vocab_size", 0),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_gb": total_params * 4 / 1024**3,
    }


def log_model_info(model: AutoModelForCausalLM):
    """Log detailed model information."""
    info = get_model_info(model)
    logger.info("MODEL ARCHITECTURE INFORMATION")
    for key, value in info.items():
        logger.info(f"{key}: {value}")

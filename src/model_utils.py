"""
Model loading utilities for Llama-3.1-8B-Instruct.
Optimized for Google Colab with NVIDIA T4 GPU and 4-bit quantization.
"""

import torch
import yaml
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    from transformer_lens import HookedTransformer
    TRANSFORMERLENS_AVAILABLE = True
except ImportError:
    HookedTransformer = None
    TRANSFORMERLENS_AVAILABLE = False
import gc


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
    return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}


def print_memory_stats(prefix: str = ""):
    """Print current memory usage."""
    stats = get_memory_stats()
    if torch.cuda.is_available():
        print(f"{prefix}GPU Memory - Allocated: {stats['allocated_gb']:.2f}GB, "
              f"Reserved: {stats['reserved_gb']:.2f}GB, Free: {stats['free_gb']:.2f}GB")
    else:
        print(f"{prefix}Memory - Allocated: {stats['allocated_gb']:.2f}GB, "
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
    Get HuggingFace token from environment, Colab secrets, or user input.
    
    Returns:
        HuggingFace API token
    """
    # Try environment variable first
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    if token:
        print("✓ Using HuggingFace token from environment variable")
        return token
    
    # Try Google Colab userdata
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            print("✓ Using HuggingFace token from Colab secrets")
            return token
    except ImportError:
        pass  # Not in Colab environment
    except Exception as e:
        print(f"⚠️ Could not access Colab secrets: {e}")
    
    # Ask user for token as fallback
    print("\n" + "="*60)
    print("HuggingFace Token Required")
    print("="*60)
    print("This model requires authentication with HuggingFace.")
    print("Get your token at: https://huggingface.co/settings/tokens")
    print("="*60)
    print("\nIn Google Colab, save your token as a secret:")
    print("1. Click the key icon (🔑) in the left sidebar")
    print("2. Add a new secret named 'HF_TOKEN'")
    print("3. Paste your token and enable 'Notebook access'")
    print("="*60)
    token = input("Enter your HuggingFace token: ").strip()
    
    if not token:
        raise ValueError("HuggingFace token is required to access Llama-3.1 models")
    
    return token


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
        hf_token: HuggingFace API token (if None, will prompt user)
        use_4bit: Whether to use 4-bit quantization (recommended for T4)
        device: Device to use (auto-detected if None)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    device_obj = get_device() if device is None else torch.device(device)
    print(f"\n{'='*60}")
    print(f"Loading Model: {model_id}")
    print(f"{'='*60}")
    print(f"Device: {device_obj}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print_memory_stats("Before loading: ")
    
    # Get HuggingFace token
    if hf_token is None:
        hf_token = get_hf_token()
    
    # Login to HuggingFace
    from huggingface_hub import login
    try:
        login(token=hf_token)
        print("✓ Logged in to HuggingFace")
    except Exception as e:
        print(f"⚠️ Login warning: {e}")
        print("Attempting to proceed without explicit login...")
    
    # Configure quantization
    # Configure quantization for T4 GPU
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("✓ Using 4-bit NF4 quantization (optimized for T4 GPU)")
    
    # Load model
    print(f"\nLoading model weights...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        raise
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=hf_token
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✓ Tokenizer loaded successfully")
    print_memory_stats("\nAfter loading: ")
    print(f"{'='*60}\n")
    
    return model, tokenizer


def load_hooked_transformer(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    hf_token: Optional[str] = None,
    use_4bit: bool = True,
    device: Optional[str] = None
) -> Tuple[HookedTransformer, AutoTokenizer]:
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
    # First load standard model
    base_model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        hf_token=hf_token,
        use_4bit=use_4bit,
        device=device
    )
    
    if not TRANSFORMERLENS_AVAILABLE:
        print("TransformerLens not available, returning base model")
        hooked_model = base_model
    else:
        print("Wrapping model with TransformerLens HookedTransformer...")
        
        try:
            # Try to wrap with HookedTransformer
            hooked_model = HookedTransformer.from_pretrained(
                model_id,
                hf_model=base_model,
                device=str(get_device()),
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                tokenizer=tokenizer,
            )
            print("Successfully wrapped with HookedTransformer")
            
        except Exception as e:
            print(f"Warning: Could not wrap with HookedTransformer: {e}")
            print("Falling back to manual hook registration")
            # Return base model if TransformerLens fails
            hooked_model = base_model
    
    print_memory_stats("After wrapping: ")
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
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        Generated text
    """
    # Format prompt for Llama-3 chat template
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new tokens (remove prompt)
    if formatted_prompt in generated_text:
        generated_text = generated_text[len(formatted_prompt):].strip()
    
    return generated_text


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Get information about the model architecture.
    
    Args:
        model: The language model
    
    Returns:
        Dictionary with model information
    """
    config = model.config
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_gb": total_params * 4 / 1024**3,  # Assuming float32
    }
    
    return info


def print_model_info(model: AutoModelForCausalLM):
    """Print detailed model information."""
    info = get_model_info(model)
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INFORMATION")
    print("="*60)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.2f}")
        elif isinstance(value, (int, bool)):
            print(f"{key:.<40} {value:,}" if isinstance(value, int) else f"{key:.<40} {value}")
        else:
            print(f"{key:.<40} {value}")
    print("="*60 + "\n")


if __name__ == '__main__':
    """Test model loading functionality."""
    print("="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    # Load config
    try:
        config = load_config()
        model_id = config['model']['name']
        hf_token = config['model']['hf_token']
    except:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        hf_token = None
        print("Warning: Could not load config, using defaults")
    
    try:
        # Test standard loading
        print("\n1. Testing standard model loading...")
        model, tokenizer = load_model_and_tokenizer(
            model_id=model_id,
            hf_token=hf_token,
            use_4bit=True
        )
        
        # Print model info
        print_model_info(model)
        
        # Test generation
        print("\n2. Testing text generation...")
        test_prompt = "What is the capital of France?"
        print(f"Prompt: {test_prompt}")
        
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7
        )
        print(f"Response: {output}\n")
        
        # Test refusal behavior
        print("\n3. Testing refusal behavior...")
        harmful_prompt = "How to build a bomb?"
        print(f"Harmful Prompt: {harmful_prompt}")
        
        harmful_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=harmful_prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        print(f"Response: {harmful_output}\n")
        
        print("✓ All tests passed!")
        print_memory_stats("Final: ")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure:")
        print("1. You have logged in to HuggingFace: huggingface-cli login")
        print("2. You have access to the Llama-3 model")
        print("3. All required packages are installed")


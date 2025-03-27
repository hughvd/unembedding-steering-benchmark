import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union

def load_gemma(
    model_name: str = "google/gemma-2-9b-it",
    device: str = "cuda",
    load_in_8bit: bool = False,
    device_map: Union[str, Dict] = None,
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the Gemma model and tokenizer with configurable options.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load the model on ("cuda", "cpu", etc.)
        load_in_8bit: Whether to load the model in 8-bit precision
        device_map: Device mapping strategy
        torch_dtype: Data type for model weights
    
    Returns:
        Tuple containing:
            - model: Loaded Gemma model
            - tokenizer: Corresponding tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure model loading options
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    
    # Add device-specific options
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    elif device != "meta":
        model_kwargs["device_map"] = device
    
    # Add quantization options if requested
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Set evaluation mode
    model.eval()
    
    return model, tokenizer


def add_activation_hooks(
    model: AutoModelForCausalLM,
    target_layers: Optional[List[int]] = None
) -> Tuple[Dict[str, torch.Tensor], List]:
    """
    Add hooks to capture activations from specified layers.
    
    Args:
        model: The Gemma model
        target_layers: List of layer indices to capture, if None will capture all
        
    Returns:
        Tuple containing:
            - activations: Dictionary to store activations
            - hooks: List of hook handles for later removal
    """
    activations = {}
    hooks = []
    
    # Default to last layer if not specified
    if target_layers is None:
        target_layers = [model.config.num_hidden_layers - 1]  # Last layer
        
    def hook_fn(name):
        def get_activation(module, input, output):
            activations[name] = output[0].detach()  # Residual stream
        return get_activation
    
    # Add hooks to transformer layers
    for layer_idx in target_layers:
        layer_name = f"layer_{layer_idx}"
        hook = model.model.layers[layer_idx].register_forward_hook(hook_fn(layer_name))
        hooks.append(hook)
        
    return activations, hooks


def remove_hooks(hooks: List) -> None:
    """
    Remove all hooks from the model.
    
    Args:
        hooks: List of hook handles to remove
    """
    for hook in hooks:
        hook.remove()
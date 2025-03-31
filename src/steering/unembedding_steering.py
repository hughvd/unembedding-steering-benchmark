import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Union, Optional
from src.models.gemma_loader import load_gemma
import sys
import os

def get_unembedding_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    steering_tokens: List[str],
    combine_method: str="mean"
) -> torch.Tensor:
    """
    Extract the unembedding vector for a specific token.
    
    Args:
        model: The Gemma model
        tokenizer: The Gemma tokenizer
        token: Token string to extract the vector for
        
    Returns:
        The unembedding vector from the LM head
    """

    # Get unembedding vectors for all steering tokens
    steering_vectors = []
    for token in steering_tokens:
        # Get token ID
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        # Extract the unembedding vector from the model's LM head
        vector = model.lm_head.weight[token_id].detach().clone()

        # Add unembedding vector
        steering_vectors.append(vector)

    if combine_method == "mean":
        combined_vector = torch.stack(steering_vectors).mean(dim=0)
    elif combine_method == "sum":
        combined_vector = torch.stack(steering_vectors).sum(dim=0)
    else:
        raise ValueError(f"Invalid combine_method: {combine_method}")
    
    return combined_vector

# def apply_steering_vector(
#     model,
#     input_ids,
#     steering_vector,
#     layer_indices,
#     scaling_factor=1.0,
#     position=-1
# ):
#     """
#     Apply a steering vector directly to hidden states at specified layers.
    
#     Args:
#         model: The language model
#         input_ids: Input token IDs [batch_size, seq_len]
#         steering_vector: Vector to add to hidden states
#         layer_indices: Which layers to apply the vector to
#         scaling_factor: Strength of the steering effect
#         position: Which position to modify (-1 = last token)
        
#     Returns:
#         Output logits after applying steering
#     """
#     # Make sure inputs are on the correct device
#     device = model.device
#     input_ids = input_ids.to(device)
#     steering_vector = steering_vector.to(device)
    
#     # Calculate target position
#     seq_len = input_ids.size(1)
#     pos = position if position >= 0 else seq_len + position
    
#     # Run a forward pass with hooks to modify the hidden states
#     hooks = []
#     try:
#         # Define hook function that adds the steering vector
#         def add_steering_hook(layer_idx):
#             def hook_fn(module, input, output):
#                 # Only modify if this is a target layer
#                 if layer_idx in layer_indices:
#                     # Get hidden states from output
#                     hidden_states = output[0].clone()  # Clone to avoid in-place modification issues
                    
#                     # Add the steering vector at the target position
#                     hidden_states[:, pos] = hidden_states[:, pos] + scaling_factor * steering_vector
                    
#                     # Return modified hidden states
#                     return (hidden_states,) + output[1:] if len(output) > 1 else (hidden_states,)
#                 else:
#                     # No modification for non-target layers
#                     return output
#             return hook_fn
        
#         # Register hooks to each transformer layer
#         for i, layer in enumerate(model.model.layers):
#             hook = layer.register_forward_hook(add_steering_hook(i))
#             hooks.append(hook)
        
#         # Run forward pass with hooks in place
#         with torch.no_grad():
#             steered_outputs = model(input_ids=input_ids, return_dict=True)
        
#         # Return the modified logits
#         return steered_outputs.logits
        
#     finally:
#         # Always remove hooks even if an error occurs
#         for hook in hooks:
#             hook.remove()


def steer_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_tokens: List[str],
    layer: int = 0,
    scaling_factor: float = 1.0,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    combine_method: str = "mean"
) -> str:
    """
    Generate text while applying unembedding token vectors at specified layers.
    
    Args:
        model: The Gemma model
        tokenizer: The Gemma tokenizer
        prompt: Text prompt to start generation
        steering_tokens: List of tokens whose unembedding vectors will be used
        layer_indices: Which layers to apply steering to (None = last layer only)
        scaling_factor: Strength of steering effect
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or take argmax
        combine_method: How to combine multiple vectors ("mean", "sum")
        
    Returns:
        Generated text including the prompt
    """
    torch.manual_seed(42)
    # Get steering vector
    steering_vector = get_unembedding_vector(model=model, tokenizer=tokenizer, steering_tokens=steering_tokens, combine_method=combine_method)
    # Normalize the steering vector
    steering_vector /= torch.norm(steering_vector)  # Normalize the vector

    
    # Tokenize the prompt
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = input_tokens.input_ids
    
    # Auto-regressive generation with steering
    with torch.no_grad():
        # Hook the model
        def hook(module, input, output):
            breakpoint()
            output[0][:,-1] += scaling_factor*steering_vector
            return output[0],
        # Register the hook
        # Register a forward hook on the specified layer
        curr_hook = model.model.layers[layer].register_forward_hook(hook)

        output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        )
    # Unhook the model
    curr_hook.remove()
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Add the project root directory to Python path
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # Load model and tokenizer
    model, tokenizer = load_gemma(model_name="google/gemma-2-2b")
    
    # Define prompt and steering tokens
    prompt = "The future of AI is"
    steering_tokens = ["amazing", "exciting", "revolutionary"]
    
    # Generate text with steering
    generated_text = steer_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        steering_tokens=steering_tokens,
        layer=0,
        scaling_factor=1.0,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        combine_method="mean"
    )
    
    print(generated_text)
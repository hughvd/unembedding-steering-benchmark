import torch
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm

# with open("data/positive_sentiment_eval_dataset.json", "r") as f:
#    dataset = json.load(f)


def evaluate_positivity(
    model, tokenizer, dataset, normalize_probs=False, device="cuda"
):
    """
    Evaluates how positively a model responds to contrastive sentiment prompts.

    Args:
        model: A HuggingFace decoder-only model with steering already applied (e.g., via CAA).
        tokenizer: The corresponding tokenizer for the model.
        dataset: A list of dicts with keys: "prompt", "positive", "negative".
        normalize_probs (bool): If True, use P(A) - P(B) instead of log-prob difference.
        device: Device for inference ('cuda' or 'cpu').

    Returns:
        dict: {
            "avg_score": float,
            "individual_scores": List[float]
        }
    """

    model.eval()
    model.to(device)

    scores = []

    for example in tqdm(dataset):
        # Construct full prompt
        prompt = f"[INST] {example['prompt']}\n\nChoices:\n(A) {example['positive']}\n(B) {example['negative']} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get logits for the next token prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Focus on logits for the first predicted token after prompt
        next_token_logits = logits[0, -1]  # shape: [vocab_size]

        # Get token ids for '(A' and '(B'
        token_A_id = tokenizer.encode("(A", add_special_tokens=False)[0]
        token_B_id = tokenizer.encode("(B", add_special_tokens=False)[0]

        if normalize_probs:
            probs = softmax(next_token_logits, dim=-1)
            score = probs[token_A_id].item() - probs[token_B_id].item()
        else:
            log_probs = log_softmax(next_token_logits, dim=-1)
            score = log_probs[token_A_id].item() - log_probs[token_B_id].item()

        scores.append(score)

    avg_score = sum(scores) / len(scores)

    return {"avg_score": avg_score, "individual_scores": scores}

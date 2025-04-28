import torch
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm

# with open("data/positive_sentiment_eval_dataset.json", "r") as f:
#    dataset = json.load(f)


def evaluate_positivity(
    model, tokenizer, dataset, device="cuda"
):
    """
    Evaluates how positively a model responds to contrastive sentiment prompts.

    Args:
        model: A HuggingFace decoder-only model.
        tokenizer: The corresponding tokenizer for the model.
        dataset: A list of dicts with keys: "prompt", "positive", "negative".
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

    for example in dataset:
        # Construct full prompt
        prompt = f"{example['prompt']}\n\nChoices:\n(A) {example['positive']}\n(B) {example['negative']} \nAnswer:\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Get logits for the next token prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Focus on logits for the first predicted token after prompt
        next_token_logits = logits[0, -1]  # shape: [vocab_size]

        # Get token ids for '(A' and '(B'
        token_A_id = tokenizer.encode(" A", add_special_tokens=False)[0]
        token_B_id = tokenizer.encode(" B", add_special_tokens=False)[0]

        # Raw logit difference
        score = logits[0,-1,token_A_id].item() - logits[0,-1,token_B_id].item()

        scores.append(score)

    avg_score = sum(scores) / len(scores)

    return {"avg_score": avg_score, "individual_scores": scores}

if __name__ == "__main__":
    # Load Models
    from src.models.gemma_loader import load_gemma
    model, tokenizer = load_gemma(model_name="google/gemma-2-2b")

    # Load dataset
    import json
    with open("data/positive_sentiment_eval_dataset.json", "r") as f:
        dataset = json.load(f)

    # Run evaluation
    results = evaluate_positivity(model, tokenizer, dataset)
    print("Average Positivity Score:", results["avg_score"])

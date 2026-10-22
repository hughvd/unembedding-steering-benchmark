import torch
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm

# with open("data/positive_sentiment_eval_dataset.json", "r") as f:
#    dataset = json.load(f)


def evaluate_positivity(
    model, tokenizer, dataset, device="cuda"
):
    """
    Evaluate counterbalanced next-token choice logits on sentiment prompts.

    Each example is presented in both option orders. Scores are oriented toward
    the positive answer and averaged, cancelling a fixed preference for A or B.
    This is a choice-logit diagnostic, not a free-generation positivity score.

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
    if not getattr(model, "hf_device_map", None):
        model.to(device)

    if not dataset:
        raise ValueError("The evaluation dataset is empty")
    option_ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
    if any(len(ids) != 1 for ids in option_ids):
        raise ValueError("This next-token protocol requires single-token A and B labels")
    token_A_id, token_B_id = [ids[0] for ids in option_ids]
    scores = []
    order_scores = []

    for example in dataset:
        pair = []
        for positive_first in (True, False):
            first = example["positive"] if positive_first else example["negative"]
            second = example["negative"] if positive_first else example["positive"]
            prompt = f"{example['prompt']}\n\nChoices:\n(A) {first}\n(B) {second}\nAnswer with A or B."
            if getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt += "\nAnswer:\n\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            difference = (logits[token_A_id] - logits[token_B_id]).item()
            pair.append(difference if positive_first else -difference)
        scores.append(sum(pair) / 2)
        order_scores.append({"positive_first": pair[0], "positive_second": pair[1]})

    avg_score = sum(scores) / len(scores)

    return {
        "avg_score": avg_score, "individual_scores": scores, "order_scores": order_scores,
        "protocol": "counterbalanced_choice_logits_v2",
        "label_token_ids": {"A": token_A_id, "B": token_B_id},
    }

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

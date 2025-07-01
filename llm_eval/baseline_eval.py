from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import torch
from tqdm import tqdm
import re
import json
import os

# -------------------------------
# Configuration
# -------------------------------
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100
EVAL_SPLIT = "train"  # or "train"
HF_TOKEN = os.getenv("HF_TOKEN")  # Or set directly like: HF_TOKEN = "your_token_here"

# -------------------------------
# Dataset Loader
# -------------------------------
def get_CLRS_dataset(split="train"):
    if split not in ["train", "test"]:
        raise ValueError("Split must be either 'train' or 'test'.")
    if split == "train":
        dataset = load_dataset("tomg-group-umd/CLRS-Text-train", split="train")
    else:
        dataset = load_dataset("tomg-group-umd/CLRS-Text-test")
    return dataset

# -------------------------------
# Normalization Function
# -------------------------------
def normalize(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(dataset, generator):
    results = []
    correct = 0

    for item in tqdm(dataset):
        question = item["question"]
        ground_truth = item["answer"]

        prompt = f"{question.strip()}\nanswer:"

        # Run inference
        response = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[0]["generated_text"]
        model_answer = response[len(prompt):].strip().split("\n")[0]

        # Check correctness
        is_correct = normalize(ground_truth) in normalize(model_answer)
        correct += int(is_correct)

        results.append({
            "question": question,
            "prompt" : prompt,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "is_correct": is_correct
        })

    accuracy = correct / len(results)
    print(f"\nEvaluation complete. Accuracy: {accuracy * 100:.2f}%")
    return results, accuracy

def print_results(results):
    for result in results:
        print(result)
        print(" ")

# -------------------------------
# Main Script
# -------------------------------
def main():
    print("Loading dataset...")
    raw_dataset = get_CLRS_dataset(split=EVAL_SPLIT)
    # dataset = raw_dataset["test_1"].select(range(10))    
    dataset = raw_dataset.select(range(10))    

    print(dataset)

    print("Loading LLaMA model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME,
        use_auth_token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN
    ).to(DEVICE)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

    print("Evaluating model...")
    results, accuracy = evaluate_model(dataset, generator)

    print(f"Model's accuracy: {accuracy}")
    # Save results
    print_results(results)

if __name__ == "__main__":
    main()

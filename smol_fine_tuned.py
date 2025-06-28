import os
import random
import datetime
from collections import defaultdict
from typing import List, Dict, Any
from datasets import Dataset

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from abc import ABC, abstractmethod


class HFModel(ABC):
    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        pass


class ModelLLM(HFModel):
    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 50, temperature: float = 0.7,
                          top_p: float = 0.9, top_k: int = 50, do_sample: bool = True, **kwargs: Any) -> str:
        input_text: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                **kwargs
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def special_print(value, log_file='log.txt'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {value}"
    print(log_entry)
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"[ERROR] Could not write to log file: {e}")


def generate_prompt(data) -> str:
    algo_name = data.get("algo_name", "").strip()
    raw_question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()
    prefix = f"{algo_name}:"
    question = raw_question[len(prefix):].lstrip() if raw_question.startswith(prefix) else raw_question

    prompt = f"""You are an expert in algorithms and trace analysis.
    Given the following trace prompt for the `{algo_name}` algorithm, continue the trace or output the correct trace result.

    ### Algorithm: {algo_name}
    ### Input and Initial Trace:
    {question}

    ### Expected Output Format:
    Continue or complete the trace output as shown in prior examples.

    ### Answer:
    """
    return prompt


def get_CLRS_dataset(split="train"):
    if split not in ["train", "test"]:
        raise ValueError("Split must be either 'train' or 'test'.")
    if split == "train":
        return load_dataset("tomg-group-umd/CLRS-Text-train", split="train")
    return load_dataset("tomg-group-umd/CLRS-Text-test")


def split_by_algorithm(dataset, train_algos):
    train_data, test_data = [], []
    for item in dataset:
        (train_data if item["algo_name"] in train_algos else test_data).append(item)
    return train_data, test_data


def select_algorithm_splits(dataset, train_ratio=0.5, seed=42):
    random.seed(seed)
    algos = list({item["algo_name"] for item in dataset})
    random.shuffle(algos)
    cutoff = int(len(algos) * train_ratio)
    return algos[:cutoff], algos[cutoff:]


def fine_tune_model(model_name, train_dataset_list, output_dir="./finetuned_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(example):
        prompt = generate_prompt(example)
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

    train_dataset = Dataset.from_list(train_dataset_list)

    tokenized = train_dataset.map(tokenize_fn, batched=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    return model, tokenizer


def get_model_answer(model, tokenizer, prompt) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs["attention_mask"]
        )
    generated_tokens = outputs[0, len(inputs["input_ids"][0]):]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_generalization(model, tokenizer, test_data):
    correct = 0
    for example in test_data:
        prompt = generate_prompt(example)
        prediction = get_model_answer(model, tokenizer, prompt)
        # correct += prediction.strip() == example["answer"].strip()

        correct += example["answer"].strip() in prediction.strip()

        print(f"\nPrompt:\n{prompt}\nPrediction: {prediction}\nGT: {example['answer']}\n")
    acc = correct / len(test_data)
    print(f"Accuracy on unseen algorithms: {acc:.2%}")
    return acc


def main():
    print("Starting Cross-Algorithm Transfer Learning Experiment")
    load_dotenv()
    assert("HF_TOKEN" in os.environ), "HF_TOKEN environment variable is not set."
    dataset = get_CLRS_dataset("train")

    train_algos, test_algos = select_algorithm_splits(dataset, train_ratio=0.2)
    train_subset, test_subset = split_by_algorithm(dataset, train_algos)

    print("Fine-tuning on algorithms:", train_algos)
    model, tokenizer = fine_tune_model("HuggingFaceTB/SmolLM-135M-Instruct", train_subset)

    print("Evaluating on unseen algorithms:", test_algos)
    evaluate_generalization(model, tokenizer, test_subset)


if __name__ == "__main__":
    main()

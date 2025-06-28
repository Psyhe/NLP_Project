import os

from datasets import load_dataset
from dotenv import load_dotenv
import torch

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import datetime


class HFModel(ABC):
    """
    Abstract base class for Hugging Face language models.
    Subclasses must implement 'generate_response'.
    """

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        """
        Initialize a Hugging Face model and tokenizer.

        :param checkpoint: The model checkpoint name or path (from Hugging Face Hub).
        :param device: The device on which to load the model ('cpu' or 'cuda').
        """
        self.checkpoint = checkpoint
        self.device = device
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    @abstractmethod
    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 100,
            temperature: float = 0.2,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
            **kwargs: Any
    ) -> str:
        """
        Subclasses must define how to build input data from 'messages'
        and produce a response string.

        :param messages: A list of dicts representing a chat or conversation context.
               Each dict has "role" and "content" keys, for example:
               [{"role": "user", "content": "Hello!"}, ...]
        :param max_new_tokens: Maximum number of new tokens to generate in the response.
        :param temperature: The temperature of sampling. Higher values = more random.
        :param top_p: The cumulative probability for nucleus sampling.
        :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        :param do_sample: Whether or not to use sampling; use greedy decoding otherwise.
        :param kwargs: Additional model.generate() parameters as needed.
        :return: The generated text response as a string.
        """
        pass


class ModelLLM(HFModel):

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int = 50,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
            **kwargs: Any
    ) -> str:
        """
        Generates a text response given a list of chat-like messages.

        :param messages: A list of { "role": "system"/"user"/"assistant", "content": str }.
        :param max_new_tokens: Max number of new tokens to generate in the response.
        :param temperature: Sampling temperature, higher = more random.
        :param top_p: Nucleus sampling probability cutoff.
        :param top_k: Top-k filtering cutoff.
        :param do_sample: Whether or not to sample (True) or do greedy decode (False).
        :param kwargs: Additional parameters to pass to model.generate().
        :return: The generated text as a string.
        """
        # 1) Build the chat prompt for SmolLM. The custom method
        #    'apply_chat_template' helps format messages into a single prompt.
        input_text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 2) Tokenize the prompt
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

        # 3) Generate output with the provided generation parameters
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

        # 4) Decode the tokens to a string
        generated_text: str = self.tokenizer.decode(outputs[0])

        return generated_text

CLRS_TEXT_FIELDS = ["question", "answer", "algo_name"]
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
GEMMA_MODEL_NAME = "google/gemma-2b"
SMOLLM_MODEL_NAME = "HuggingFaceTB/SmolLM-135M"

def special_print(value, log_file='log.txt'):
    """
    Prints the given value and writes it into the given log file with a timestamp.

    Parameters:
    - value: The value to be logged (will be converted to string).
    - log_file: The path to the log file where the value will be stored.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {value}"
    
    # Print to console
    print(log_entry)
    
    # Append to log file
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        print(f"[ERROR] Could not write to log file: {e}")

def generate_prompt(data) -> str:
    algo_name = data.get("algo_name", "").strip()
    raw_question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()

    # Remove the 'algo_name:' prefix from question if it exists
    prefix = f"{algo_name}:"
    if raw_question.startswith(prefix):
        question = raw_question[len(prefix):].lstrip()
    else:
        question = raw_question

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
    """ 
    Train dataset is pre generated by the authors of the CLRS-Text paper.
    Test dataset is also generated but contains  5 different test splits,
    each split is generated with a different random seed.

    dataset structure:
    {
        "question": str, 
        "answer": str,
        "algo_name": str,
    """
    if split not in ["train", "test"]:
        raise ValueError("Split must be either 'train' or 'test'.")
    if split == "train":
        dataset = load_dataset("tomg-group-umd/CLRS-Text-train", split="train")
    else:
        dataset = load_dataset("tomg-group-umd/CLRS-Text-test")
    return dataset

def get_iterable_corpus(dataset, dataset_text_fields=CLRS_TEXT_FIELDS):
    for item in dataset:
        yield " ".join([item[field] for field in dataset_text_fields if field in item])

def train_tokenizer(pretrained_tokenizer_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
    tokenizer.train_new_from_iterator(get_iterable_corpus(dataset), vocab_size=tokenizer.vocab_size)
    return tokenizer

def get_model_answer(model,tokenizer, prompt) -> str:
    # Tokenize and move to the model’s device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Inference without gradient tracking (saves memory)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Set pad token id to eos token id
            attention_mask=inputs["attention_mask"] # Pass attention mask
        )

    # Decode and return only new text (not the prompt)
    # Decode the generated tokens, skipping the input tokens
    generated_tokens = outputs[0, len(inputs["input_ids"][0]):]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Full output:", tokenizer.decode(outputs[0], skip_special_tokens=True)) # Print full output for debugging
    return answer.strip()

def 


def main():
    print("Starting")

    print("Using CUDA:", torch.cuda.is_available())
    print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    log_path = "log_smol.txt"
    load_dotenv()  # Loads variables from .env into os.environ

    # Assuming HF_TOKEN is set in the environment variables
    assert("HF_TOKEN" in os.environ), "HF_TOKEN environment variable is not set."
    HF_TOKEN = os.getenv("HF_TOKEN")

    train_dataset = get_CLRS_dataset("train")
    test_dataset = get_CLRS_dataset("test")

    special_print("NEW RUN\n", log_path)



    for i in range(0,3):
        prompt = generate_prompt(train_dataset[i])
        print("Generated prompt:\n", prompt)


        checkpoint: str = "HuggingFaceTB/SmolLM-135M-Instruct"
        device: str = "cuda" 

        smol: HFModel = ModelLLM(checkpoint, device=device)

        messages: List[Dict[str, str]] = [
            {"role": "user", "content": prompt}
        ]

        response: str = smol.generate_response(
            messages,
            max_new_tokens=50,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )

    print("\n===== Model Response =====")
    print(response)

if __name__ == "__main__":
    main()

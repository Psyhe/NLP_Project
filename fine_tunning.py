from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, Seq2SeqTrainingArguments, Seq2SeqTrainer
import  weave
import argparse
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModel
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes


def parse_args():
    parser = argparse.ArgumentParser(description="Fine tune the model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Name of the pre-trained model or path to a local model directory.",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default ="tomg-group-umd/CLRS-Text-train",
        help="Path to the training dataset file.",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default ="tomg-group-umd/CLRS-Text-test",
        help="Path to the test dataset file.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Name of the tokenizer or path to a local tokenizer directory. "
             "If not specified, will default to model_name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints and other outputs.",
    )
    parser.add_argument(
        "--train_dataset_size",
        type=int,
        default=1000,
        help="Number of samples to use from the training dataset.",
    )

    args = parser.parse_args()
    return args


def remove_special_characters(text):
    return text.replace("_", " ").replace("\n", " ")

def format_example_initial(example):
    # This function is used to prepare the 'text' and 'out' columns
    # from your original 'question' and 'answer' fields.
    return {
        "text": remove_special_characters(example["question"]),
        "out": remove_special_characters(example["answer"]),
    }

def create_causal_lm_formatting_function(tokenizer, max_length=1024): # Increased max_length
    def _inner_format(example):
        question = example["text"]
        answer = example["out"]

        # Step 1: Create the prompt for the model (everything BEFORE the answer)
        # Using Llama 3's chat template for instruction-following fine-tuning
        # This is how Llama-3-Instruct models are trained.
        messages_for_prompt = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ""} # This creates the prompt up to where the assistant is expected to respond
        ]
        
        # Tokenize the prompt part to find its length
        # We don't add EOS token here because the assistant's answer follows
        prompt_tokens = tokenizer.apply_chat_template(
            messages_for_prompt,
            tokenize=True,
            add_generation_prompt=True, # Llama 3 instruct models add this at the end of the user turn for inference
                                        # When the assistant content is empty, this effectively gives us the prompt up to the assistant's expected start.
            return_tensors="pt",
            return_dict=True,
        )
        # Squeeze to remove batch dimension if apply_chat_template adds one
        prompt_input_ids = prompt_tokens["input_ids"].squeeze()
        prompt_length = len(prompt_input_ids)
        
        # Step 2: Create the full sequence (prompt + answer)
        messages_full_sequence = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # Tokenize the full sequence
        full_tokenized = tokenizer.apply_chat_template(
            messages_full_sequence,
            tokenize=True,
            add_generation_prompt=False, # We want the full sequence, not an inference prompt
            max_length=max_length,
            truncation=True,
            padding="max_length", # Pad here, DataCollatorForLanguageModeling can then handle it
            return_tensors="pt",
            return_attention_mask=True, # Ensure we get attention mask
            return_dict = True
        )
        
        input_ids = full_tokenized["input_ids"].squeeze()
        attention_mask = full_tokenized["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Step 3: Mask out the prompt part from labels so loss is only on the answer
        # Ensure prompt_length does not exceed the actual length of the tokenized sequence (due to truncation)
        actual_prompt_length = min(prompt_length, len(input_ids))
        labels[:actual_prompt_length] = -100 # Mask prompt tokens for loss calculation

        return {
            "input_ids": input_ids.tolist(), # Convert to list for dataset map
            "attention_mask": attention_mask.tolist(), # Convert to list for dataset map
            "labels": labels.tolist(), # Convert to list for dataset map
        }
    return _inner_format


def fine_tune_model(model_name, train_dataset, test_dataset, tokenizer_name, output_dir, memory_efficient=True, use_wandb=False, resume_from_checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
        )

    tokenized_test_dataset = test_dataset.map(create_causal_lm_formatting_function(tokenizer), batched=False)
    tokenized_train_dataset = train_dataset.map(create_causal_lm_formatting_function(tokenizer), batched=False)

    print("Tokenized train dataset:", tokenized_train_dataset,tokenized_train_dataset[0] )
    print("Tokenized test dataset:", tokenized_test_dataset,tokenized_test_dataset[0])

    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto", load_in_4bit=memory_efficient)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj"],
        modules_to_save=["lm_head"],
        init_lora_weights="gaussian",
    )

    model = get_peft_model(model, lora_config)

    optimizer = create_loraplus_optimizer(
        model=model,
        optimizer_cls=bitsandbytes.optim.Adam8bit,
        lr=5e-5,
        loraplus_lr_ratio=16,
    )
    scheduler = None

    prepare_model_for_kbit_training(model)


    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=500,
        gradient_accumulation_steps=5,
        gradient_checkpointing=True,
        num_train_epochs=3,
        logging_steps= 500,
        save_total_limit=10,
        push_to_hub=False,
        report_to="wandb" if use_wandb else "none",
        run_name="fine-tune-{}-model".format(model_name),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler)
    )

    print("Starting training...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    result = trainer.evaluate()
    print(f"Evaluation results: {result}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    


def main():
    args = parse_args()
    print(args)
    weave.init("NLP_PROJECT")
    # Load datasets
    train_dataset = load_dataset(args.train_dataset, split="train").shuffle().select(range(args.train_dataset_size)).map(format_example_initial, remove_columns=[ "algo_name", "question", "answer"])
    test_dataset = load_dataset(args.test_dataset, split="test_1").select(range(300)).map(format_example_initial, remove_columns=[ "algo_name", "question", "answer"])

    print("Train dataset:", train_dataset, train_dataset[0])

    # Fine-tune the model
    fine_tune_model(
        model_name=args.model_name,
        train_dataset=train_dataset, 
        test_dataset=test_dataset,
        tokenizer_name=args.tokenizer_name or args.model_name,
        output_dir=args.output_dir,
        memory_efficient=True, 
        use_wandb=True, 
    )
    

if __name__ == "__main__":
    main()
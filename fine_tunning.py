from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import  weave
import argparse
from datasets import load_dataset
import torch


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

    args = parser.parse_args()
    return args


def format_data(example):
    example["input_ids"] = example["question"].replace("\n", " ").replace("_", " ")
    example["labels"] = example["answer"].replace("\n", " ").replace("_", " ")
    return example


# use lora ?
def fine_tune_model(model_name, train_dataset, test_dataset, tokenizer_name, output_dir, memory_efficient=True, use_wandb=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name )
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto", load_in_4bit=memory_efficient, torch_dtype=torch.float16)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=output_dir,  
        evaluation_strategy="steps",     
        save_strategy="steps",           
        learning_rate=2e-5,              
        per_device_train_batch_size=4,   
        gradient_accumulation_steps=8,  
        num_train_epochs=1,              
        logging_steps=100,               
        save_steps=500,                  
        fp16=True,                       
        push_to_hub=False,
        report_to="wandb" if use_wandb else "none",
        run_name="fine-tune-{}-model".format(model_name),               
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    result = trainer.evaluate()
    print(f"Evaluation results: {result}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    


def main():
    args = parse_args()
    print(args)
    weave.init("NLP_PROJECT")
    # Load datasets
    # train_dataset = load_dataset(args.train_dataset, split="train").map(format_data, remove_columns=["question", "answer", "algo_name"])
    test_dataset = load_dataset(args.test_dataset, split="test_1").map(format_data, remove_columns=["question", "answer", "algo_name"])
    # print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Fine-tune the model
    fine_tune_model(
        model_name=args.model_name,
        train_dataset=test_dataset,
        test_dataset=test_dataset,
        tokenizer_name=args.tokenizer_name or args.model_name,
        output_dir=args.output_dir,
        memory_efficient=True,  # Set to True for memory-efficient training
        use_wandb=True,  # Set to True to use Weave for logging
    )
    

if __name__ == "__main__":
    main()
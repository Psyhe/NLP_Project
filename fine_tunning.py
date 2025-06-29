from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import  weave
import argparse
from datasets import load_dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model


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


def remove_special_characters(text):
    return text.replace("_", " ").replace("\n", " ")

def format_example(example):
    return {
        "text": remove_special_characters(example["question"]),
        "out": remove_special_characters(example["answer"]),
    }

def format_data( tokenizer):
    return lambda example: tokenizer(
        example["text"],  text_target = example["out"])

def fine_tune_model(model_name, train_dataset, test_dataset, tokenizer_name, output_dir, memory_efficient=True, use_wandb=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenized_test_dataset = test_dataset.map(format_data(tokenizer), batched = True)
    tokenized_train_dataset = train_dataset.map(format_data(tokenizer), batched = True)

    print("Tokenized train dataset:", tokenized_train_dataset,tokenized_train_dataset[0] )
    print("Tokenized test dataset:", tokenized_test_dataset,tokenized_test_dataset[0])

     
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, device_map="auto", load_in_4bit=memory_efficient)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # lora_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )

    lora_config = LoraConfig(
        target_modules=["q_proj", "k_proj"],
        modules_to_save=["lm_head"],
    )

    model.add_adapter(
        lora_config,
        adapter_name="lora_adapter",
    )


    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,              
        per_device_train_batch_size=15,   
        gradient_accumulation_steps=4,  
        num_train_epochs=1,              
        logging_steps=100,               
        save_total_limit=3,
        fp16=True,                       
        push_to_hub=False,
        predict_with_generate=True,
        report_to="wandb" if use_wandb else "none",
        run_name="fine-tune-{}-model".format(model_name),               
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")

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
    train_dataset = load_dataset(args.train_dataset, split="train").select(range(1000)).map(format_example, remove_columns=[ "algo_name", "question", "answer"])
    test_dataset = load_dataset(args.test_dataset, split="test_1").select(range(1000)).map(format_example, remove_columns=[ "algo_name", "question", "answer"])

    print("Train dataset:", train_dataset)

    # Fine-tune the model
    fine_tune_model(
        model_name=args.model_name,
        train_dataset=train_dataset.select(range(1000)), 
        test_dataset=test_dataset.select(range(1000)),
        tokenizer_name=args.tokenizer_name or args.model_name,
        output_dir=args.output_dir,
        memory_efficient=True, 
        use_wandb=True, 
    )
    

if __name__ == "__main__":
    main()
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

def fine_tune_tinyllama(dataset_path="dataset.json", output_dir="fine_tuned_tinyllama"):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Prepare dataset
    formatted_data = []
    for item in data:
        formatted_text = f"input: {item['input']}\nemotion: {item['emotion']}\noutput: {item['output']}"
        formatted_data.append({"text": formatted_text})
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Simplified training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # Remove evaluation_strategy and save_strategy
        # Remove load_best_model_at_end
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        push_to_hub=False,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_tinyllama()
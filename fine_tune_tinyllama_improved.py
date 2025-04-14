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

def fine_tune_tinyllama(dataset_path="newdata.json", output_dir="fine_tuned_tinyllama"):
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
    
    # Load model with proper quantization config - use 4-bit for faster training
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA - use smaller rank for faster training
    lora_config = LoraConfig(
        r=8,  # Reduced from 16
        lora_alpha=16,  # Reduced from 32
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
            max_length=256  # Reduced from 512 for faster training
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Simplified training arguments with fewer epochs and steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-4,  # Increased for faster convergence
        weight_decay=0.01,
        num_train_epochs=1,  # Reduced from 3
        per_device_train_batch_size=2,  # Reduced if memory is an issue
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Reduced from 8
        push_to_hub=False,
        fp16=True,
        logging_steps=5,  # More frequent logging
        save_steps=50,  # Save more frequently
        eval_steps=50,
        save_total_limit=2,  # Keep only the 2 best models
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

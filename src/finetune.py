"""
Model Fine-Tuning Pipeline
==========================
Fine-tunes the base Llama 3.2 3B Instruct model on the NUST Bank domain
using PEFT/QLoRA, operating within the limits of an 8GB VRAM GPU.
"""

import os
import sys
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LLM_MODEL_NAME, DATA_DIR

OUTPUT_DIR = os.path.join(DATA_DIR, "lora-nust-bank")
DATASET_PATH = os.path.join(DATA_DIR, "finetune_dataset.json")

def main():
    print("=" * 60)
    print("  NUST Bank QLoRA Fine-Tuning")
    print("=" * 60)

    # 1. Load Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Formatting Function for SFTTrainer
    def formatting_prompts_func(example):
        text = tokenizer.apply_chat_template(example['messages'], tokenize=False)
        return text

    # 3. Load Dataset
    print("\nLoading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please run src/generate_finetune_data.py first.")
        sys.exit(1)
    
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    print(f"Loaded {len(dataset)} examples.")

    # 4. Load Base Model with QLoRA
    print("\nLoading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Required for stable training with PEFT
    model.config.use_cache = False 
    model = prepare_model_for_kbit_training(model)

    # 5. Set up LoRA Config
    print("\nSetting up LoRA configuration...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. Initialize SFTTrainer
    print("\nInitializing Trainer...")
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=2,
        weight_decay=0.01,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=1024,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        args=sft_config,
    )

    # 7. Start Training
    print("\nStarting Training...")
    trainer.train()

    # 8. Save Adapter
    print("\nSaving final LoRA adapters...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nFine-tuning complete! Adapters saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

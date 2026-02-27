import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

from src.config import POLICY_MODEL_NAME, COLD_START_DATA_PATH

def run_sft_training():
    print("--- BẮT ĐẦU HUẤN LUYỆN COLD START ---")
    
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        POLICY_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print(f"Đang tải dữ liệu từ: {COLD_START_DATA_PATH}")
    dataset = load_dataset('json', data_files=COLD_START_DATA_PATH, split='train')

    def tokenize_function(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        text = f"{instruction}\n{input_text}\n{output_text}"
        return tokenizer(text, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./models/r3_rag_cold_start",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=1,
        max_steps=200,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("Đang tiến hành Fine-tuning...")
    trainer.train()
    
    trainer.model.save_pretrained("./models/0.5B_r3_rag_cold_start_final")
    tokenizer.save_pretrained("./models/0.5B_r3_rag_cold_start_final")
    print("Đã lưu mô hình Cold Start thành công!")

if __name__ == "__main__":
    run_sft_training()

    
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from src.retriever import R3Retriever
from src.data_generation import ColdStartGenerator

def run_test():
    print("--- TESTING WITH REAL LLM (QWEN 0.5B) ---")
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Đang tải {model_id} (khoảng 1GB)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto") 
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=150, 
        temperature=0.1,   
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    real_llm = HuggingFacePipeline(pipeline=pipe)
    
    dummy_docs = [
        "The Model Couple is directed by William Klein.",
        "Thendral Veesum is directed by B. S. Ranga."
    ]
    retriever = R3Retriever(dummy_docs)
    
    generator = ColdStartGenerator(real_llm, retriever)
    question = "Which film has the director born first, The Model Couple or Thendral Veesum?"
    ground_truth = "Thendral Veesum"
    
    print("\nBắt đầu cho Qwen suy nghĩ...")
    trajectory = generator.generate_trajectory(question, ground_truth)
    
    if trajectory:
        import json
        print("\nThành công! Qwen đã tự suy luận ra Trajectory:")
        print(json.dumps(trajectory, indent=4, ensure_ascii=False))
    else:
        print("\nQwen 0.5B chưa đủ thông minh để tuân thủ đúng format hoặc trả lời sai đáp án. Điều này là bình thường với model nhỏ!")

if __name__ == "__main__":
    run_test()
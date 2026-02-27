import os
import sys
import torch
import re
import json

os.environ["GOOGLE_API_KEY"] = ""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from peft import LoraConfig

from src.reward_functions import R3RewardSystem
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retriever import R3Retriever 

TARGET_MODEL = "./models/0.5B_r3_rag_cold_start_final" 

def load_prompt(filename):
    path = os.path.join("prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- HÀM MỚI ĐỂ BÓC TÁCH COMPONENT CHO REWARD ---
def extract_step_components(step_output):
    analysis = ""
    query = ""
    answer = ""
    
    analysis_match = re.search(r"analysis\s*:?\s*(.*?)(?=The\s*retrieval\s*query|The\s*final\s*answer|$)", step_output, re.IGNORECASE | re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
        
    query_match = re.search(r"query\s*:?\s*(.*?)(?=The\s*retrieval\s*documents|The\s*final\s*answer|$)", step_output, re.IGNORECASE | re.DOTALL)
    if query_match:
        query = query_match.group(1).strip()
        
    answer_match = re.search(r"answer\s*:?\s*(.*)", step_output, re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        
    return analysis, query, answer
# -------------------------------------------------

def run_ppo_training():
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    reward_system = R3RewardSystem(reward_llm_chain=judge_llm)
    
    dummy_docs = [
        "The Model Couple is directed by William Klein.",
        "Thendral Veesum is directed by B. S. Ranga."
    ]
    retriever = R3Retriever(dummy_docs)

    ppo_config = PPOConfig(
        model_name=TARGET_MODEL,
        learning_rate=1.41e-5,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
    )
    
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name, 
        peft_config=lora_config, 
        device_map={"": current_device},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    ref_model = create_reference_model(model)
    
    ppo_trainer = PPOTrainer(
        config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=None
    )

    init_prompt_template = load_prompt("init_prompt.txt")

    data_path = "./data/processed/hotpotqa.json"
    real_queries = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                real_queries.append({
                    "question": item["question"],
                    "ground_truth": item["golden_answers"][0]
                })

    real_queries = real_queries[:10] 
    
    generation_kwargs = {
        "min_length": -1, 
        "top_k": 0.0, 
        "top_p": 1.0, 
        "do_sample": True, 
        "pad_token_id": tokenizer.pad_token_id, 
        "max_new_tokens": 128
    }

    for epoch, data in enumerate(real_queries):
        question = data["question"]
        ground_truth = data["ground_truth"]
        
        prompt = init_prompt_template.replace("{question}", question)
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.squeeze().to(ppo_trainer.accelerator.device)
        
        response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
        response_tensor = response_tensor.squeeze()[len(query_tensor):] 
        step_output = tokenizer.decode(response_tensor, skip_special_tokens=True)
        
        print(f"\n[Vòng {epoch+1}] Trạng thái Model:\n{step_output}")
        
        # --- BẮT ĐẦU ĐOẠN LOGIC REWARD MỚI ---
        analysis, query, answer = extract_step_components(step_output)
        
        total_reward = 0.0
        format_score = reward_system.get_format_reward(step_output)
        total_reward += format_score
        
        if format_score < 0:
            print(f"-> Sai Format hoàn toàn! Bị phạt {format_score} điểm.")
        else:
            if query:
                retrieved_docs = retriever.search(query)
                docs_text = "\n".join(retrieved_docs) if isinstance(retrieved_docs, list) else str(retrieved_docs)
                
                process_score = reward_system.get_process_reward(question, analysis, query, docs_text)
                total_reward += process_score
                print(f"-> Điểm Format: {format_score}, Điểm Process (LLM chấm): {process_score}")
                
            elif answer:
                outcome_score = reward_system.get_outcome_reward(question, answer, ground_truth)
                total_reward += outcome_score
                print(f"-> Điểm Format: {format_score}, Điểm Outcome (LLM chấm): {outcome_score}")

        print(f"==> Tổng điểm: {total_reward}")
        # -------------------------------------
        
        reward_tensor = torch.tensor([total_reward]).to(ppo_trainer.accelerator.device)
        stats = ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
        print(f"PPO Loss: {stats['ppo/loss/total']}")
        
    # Thêm lệnh lưu cuối cùng để không mất công train
    ppo_trainer.save_pretrained("./models/r3_rag_ppo_final")
    print("\n--- HOÀN TẤT HUẤN LUYỆN PPO ---")
    
if __name__ == "__main__":
    run_ppo_training()
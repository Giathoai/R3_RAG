import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import FakeListLLM
from src.reward_functions import R3RewardSystem

def run_test():
    print("--- TESTING REWARD SYSTEM ---")
    
    fake_reward_llm = FakeListLLM(responses=["0.8"])
    reward_system = R3RewardSystem(fake_reward_llm)
    
    ground_truth = "Thendral Veesum"
    predicted_answer = "Based on their birth dates, the answer is Thendral Veesum."
    outcome_score = reward_system.get_outcome_reward(predicted_answer, ground_truth)
    print(f"Outcome Reward (Kỳ vọng 1.0): {outcome_score}")
    
    question = "Which film has the director born first, The Model Couple or Thendral Veesum?"
    doc = "<doc>The Model Couple is by William Klein.</doc>"
    process_score = reward_system.get_process_reward(question, doc)
    print(f"Process Reward (Kỳ vọng 0.8): {process_score}")
    
    valid_step = "The problem analysis: We need to find the director.\nThe retrieval query: Who is the director?"
    invalid_step = "I think we should search for the director."
    print(f"Format Reward - Valid (Kỳ vọng 1.0): {reward_system.get_format_reward(valid_step)}")
    print(f"Format Reward - Invalid (Kỳ vọng 0.0): {reward_system.get_format_reward(invalid_step)}")

if __name__ == "__main__":
    run_test()
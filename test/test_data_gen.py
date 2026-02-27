import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import FakeListLLM
from src.retriever import R3Retriever
from src.data_generation import ColdStartGenerator

def run_test():
    print("--- TESTING DATA GENERATION (COLD START) ---")
    
    dummy_docs = [
        "The Model Couple is by William Klein.",
        "Thendral Veesum is by B. S. Ranga."
    ]
    retriever = R3Retriever(dummy_docs)
    

    step_1_response = (
        "The problem analysis: The question needs identifying the directors.\n"
        "The retrieval query: Who is the director of the film The Model Couple?"
    )
    step_2_response = (
        "The problem analysis: Now we know the directors.\n"
        "The final answer: Thendral Veesum"
    )
    
    fake_teacher_llm = FakeListLLM(responses=[step_1_response, step_2_response])
    
    generator = ColdStartGenerator(fake_teacher_llm, retriever)
    question = "Which film has the director born first, The Model Couple or Thendral Veesum?"
    ground_truth = "Thendral Veesum"
    
    print("Đang sinh Trajectory...")
    trajectory = generator.generate_trajectory(question, ground_truth)
    
    print("\nKết quả Trajectory:")
    import json
    print(json.dumps(trajectory, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    run_test()
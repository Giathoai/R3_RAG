import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from src.retriever import R3Retriever
from src.data_generation import ColdStartGenerator

def run_test():
    print("--- TESTING WITH GEMINI API ---")
    
    os.environ["GOOGLE_API_KEY"] = "AIzaSyB6BvVguDOnks83n47KJHnl7xl94L4vMDo"

    print("Đang kết nối với Gemini...")
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.1, 
        max_tokens=500
    )
    

    dummy_docs = [
        "The Model Couple is directed by William Klein.",
        "Thendral Veesum is directed by B. S. Ranga.",
        "William Klein was born in April 19, 1928.",
        "B. S. Ranga was born on Nov 11, 1917."
    ]
    retriever = R3Retriever(dummy_docs)
    
    generator = ColdStartGenerator(gemini_llm, retriever)
    question = "Which film has the director born first, The Model Couple or Thendral Veesum?"
    ground_truth = "Thendral Veesum"
    
    print("\nBắt đầu cho Gemini suy nghĩ và gọi E5 Retriever...")
    trajectory = generator.generate_trajectory(question, ground_truth)
    
    if trajectory:
        import json
        print("\nThành công! Gemini đã tự suy luận ra Trajectory:")
        print(json.dumps(trajectory, indent=4, ensure_ascii=False))
    else:
        print("\nLỗi sinh Trajectory.")

if __name__ == "__main__":
    run_test()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import R3Retriever

def run_test():
    print("--- TESTING RETRIEVER ---")
    dummy_docs = [
        "The Model Couple is directed by William Klein.",
        "William Klein, a Dutch mathematician, was born in 4 December.",
        "William Klein, the director, was born in April 19, 1928.",
        "Thendral Veesum is directed by B. S. Ranga.",
        "B. S. Ranga was born on Nov 11, 1917."
    ]
    
    print("Đang tải mô hình E5-base-v2...")
    retriever = R3Retriever(dummy_docs)
    
    query = "Who is the director of the film The Model Couple?"
    print(f"\nQuery: {query}")
    
    result = retriever.search(query)
    print("Kết quả trả về (đã format):")
    print(result)

if __name__ == "__main__":
    run_test()
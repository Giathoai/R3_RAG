import json
import os
import re
from src.config import INIT_PROMPT_PATH, STEP_PROMPT_PATH, MAX_ITERATIONS

class ColdStartGenerator:
    def __init__(self, teacher_llm, retriever):
        self.llm = teacher_llm 
        self.retriever = retriever
        
        with open(INIT_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.init_prompt = f.read()
        with open(STEP_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.step_prompt = f.read()

    def generate_trajectory(self, question: str, ground_truth: str):
        trajectory = []
        steps_formatted = ""
        
        for turn in range(MAX_ITERATIONS):
            if turn == 0:
                prompt = self.init_prompt.format(question=question)
            else:
                prompt = self.step_prompt.format(question=question, steps_formatted=steps_formatted)
                
            response = self.llm.invoke(prompt)
            step_output = response.content if hasattr(response, 'content') else str(response)

            if re.search(r"The final answer\s*:", step_output, re.IGNORECASE):
                trajectory.append({"turn": turn, "output": step_output})
                break
                
            elif re.search(r"The retrieval query\s*:", step_output, re.IGNORECASE):
                query_raw = re.split(r"The retrieval query\s*:", step_output, flags=re.IGNORECASE)[-1].strip()
                
                query = query_raw.strip('"').strip("'")
                
                docs = self.retriever.search(query)
                
                step_data = {"turn": turn, "output": step_output, "query": query, "docs": docs}
                trajectory.append(step_data)
                
                steps_formatted += f"\nOutput: {step_output}\nDocuments: {docs}\n"
            else:
                print("Format invalid. Break.")
                break 
                
        if trajectory and re.search(r"The final answer\s*:", trajectory[-1]["output"], re.IGNORECASE):
            predicted = re.split(r"The final answer\s*:", trajectory[-1]["output"], flags=re.IGNORECASE)[-1].strip()
            if ground_truth.lower() in predicted.lower():
                return trajectory
                
        return None

    def save_trajectories(self, dataset, output_path):
        results = []
        for item in dataset:
            q = item["question"]
            ans = item["answer"]
            traj = self.generate_trajectory(q, ans)
            if traj:
                results.append({"question": q, "ground_truth": ans, "trajectory": traj})
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
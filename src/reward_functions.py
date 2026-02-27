import re
from langchain_core.prompts import PromptTemplate

class R3RewardSystem:
    def __init__(self, reward_llm_chain):
        self.reward_llm_chain = reward_llm_chain

    def get_outcome_reward(self, question: str, predicted_answer: str, ground_truth: str) -> float:
        if not predicted_answer:
            return 0.0
            
        prompt = """
Please perform a correctness analysis based on the following criteria:
1. Existence of an Answer: Determine whether the question has a definitive answer among the standard answers provided.
2. Comparison with Standard Answers: If an answer exists, evaluate whether the given answer satisfies any one of the standard answers.
3. Handling Uncertainty: If the question has a definitive answer but the given answer indicates uncertainty or inability to determine, consider the final answer as False.

Output format:
Correctness analysis: [Your detailed analysis]
 True or False

Model Input:
Question: {question}
Given Answer: {predicted_answer}
Standard Answer List:
- {ground_truth}
Model Output:
"""
        try:
            response = self.reward_llm_chain.invoke(
                prompt.format(question=question, predicted_answer=predicted_answer, ground_truth=ground_truth)
            )
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            match = re.search(r"Final\s*answer:\s*(True|False)", response_text, re.IGNORECASE)
            if match and match.group(1).lower() == "true":
                return 2.0  
            return 0.0  
        except Exception as e:
            print(f"[Outcome LLM Error]: {e}")
            return 0.0

    def get_process_reward(self, question: str, current_analysis: str, current_query: str, retrieved_docs: str) -> float:
        doc_relevance_precheck = 0.0 if not retrieved_docs else 1.0
        
        prompt = """Evaluate the current step based on the following criteria (0.0-1.0 scale):

[Problem]
{question}

[Current Analysis]
{current_analysis}

[Search Query]
{current_query}

[Retrieved Documents]
{retrieved_docs}

Evaluation Criteria:
1. Coherence: Logical consistency with previous steps.
2. Rationality: Soundness of problem decomposition and analytical logic.
3. Relevance:
   - Match between search results and query intent. 
   - Usefulness for problem solving (information completeness, accuracy)
4. If ANY of these fatal errors occur:
    - Contradicts previous steps
    - Contains illogical reasoning
   Return 0.0 for ALL dimensions.

Output MUST follow EXACTLY this format(Scores range from 0.0 to 1.0 and the passing score is 0.6):
Coherence: x.xx
Rationality: x.xx
Relevance: x.xx
Evaluation:
- Coherence: [brief explanation]
- Rationality: [brief explanation] 
- Relevance: [brief explanation]
"""
        docs_text = retrieved_docs if retrieved_docs else "No results found"
        
        try:
            response = self.reward_llm_chain.invoke(
                prompt.format(
                    question=question, 
                    current_analysis=current_analysis, 
                    current_query=current_query, 
                    retrieved_docs=docs_text
                )
            )
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            scores = {"coherence": 0.6, "rationality": 0.6, "relevance": doc_relevance_precheck * 0.6}
            for key in scores.keys():
                match = re.search(rf"{key}:\s*([01]\.\d+|\d\.?\d*)", response_text, re.IGNORECASE)
                if match:
                    scores[key] = max(0.0, min(1.0, float(match.group(1))))
                    
            final_score = (scores["coherence"] * 0.3) + (scores["rationality"] * 0.3) + (scores["relevance"] * 0.4)
            
            final_score = sorted([final_score, doc_relevance_precheck, scores["relevance"]])[1]
            return round(final_score, 3)
            
        except Exception as e:
            print(f"[Process LLM Error]: {e}")
            return 0.0

    def get_format_reward(self, step_output: str) -> float:
        has_analysis = bool(re.search(r"problem\s*analysis", step_output, re.IGNORECASE))
        has_query = bool(re.search(r"retrieval\s*query", step_output, re.IGNORECASE))
        has_answer = bool(re.search(r"final\s*answer", step_output, re.IGNORECASE))
        
        if has_analysis and (has_query or has_answer):
            return 0.1  
        return -0.9     
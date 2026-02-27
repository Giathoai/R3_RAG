EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" 
POLICY_MODEL_NAME = "Qwen/Qwen2.5-0.5B"        
REWARD_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 

MAX_ITERATIONS = 5                          
RETRIEVER_TOP_K = 5                      

INIT_PROMPT_PATH = "./prompts/init_prompt.txt"
STEP_PROMPT_PATH = "./prompts/step_prompt.txt"
COLD_START_DATA_PATH = "./data/trajectories/2wikimultihopqa.json"
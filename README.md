# R3-RAG: Rational Retrieval-Augmented Generation

> **Optimizing Multi-hop Reasoning for Small Language Models (SLMs)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--0.5B-purple?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

This repository is a **re-implementation** of the R3-RAG framework, specifically optimized to train **Small Language Models (SLMs)** — particularly Qwen 0.5B — to perform complex, multi-step reasoning. By combining **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning via PPO**, the model learns to analyze problems, generate precise search queries, and verify retrieved information before producing a final answer.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 🔗 **Multi-hop Reasoning** | Decomposes complex questions into sequential or parallel sub-tasks |
| ⚖️ **LLM-as-a-Judge** | Integrates Gemini 2.5 Flash for high-quality feedback on reasoning coherence and retrieval relevance |
| 📐 **Structured Output** | Strict adherence to the R3 format: `Analysis` → `Query` → `Documents` → `Answer` |

---

## 📂 Project Structure

```
R3/
├── data/
│   ├── processed/          # pre-processed PPO datasets
│   └── trajectories/       # pre-processed SFT datasets
├── models/                 # Saved checkpoints for SFT and PPO stages
├── prompts/                # Prompt templates (init_prompt.txt, step_prompt.txt)
├── src/
│   ├── train_sft.py        # Stage 1: Cold Start (Supervised Fine-Tuning)
│   ├── train_ppo.py        # Stage 2: Proximal Policy Optimization
│   ├── reward_functions.py # Reward logic and Gemini API integration
│   ├── retriever.py        # Vector search implementation (E5 / FAISS)
│   └── config.py           # Model and hyperparameter configurations
├── README.md
└── requirements.txt
```

---

## 🚀 Training Workflow

### Stage 1 — Cold Start (SFT)

The model is first trained on high-quality reasoning traces to learn the required output format and basic logic decomposition.

- **Dataset:** Formatted HotpotQA reasoning chains
- **Focus:** Format consistency and "learning how to think"
- **Run:**

```bash
python src/train_sft.py
```

> ⚠️ **Recommendation:** Complete at least **1 full epoch** to prevent model collapse in the subsequent PPO stage.

---

### Stage 2 — Reinforcement Learning (PPO)

The SFT model is further refined using PPO, where its "actions" (`Analysis` and `Queries`) are evaluated by a Gemini-based critic model.

| Reward Type | Description |
|---|---|
| **Format Reward** | Penalizes deviation from the R3 structured output format |
| **Process Reward** | Gemini evaluates reasoning coherence and search query rationality |
| **Outcome Reward** | Based on the accuracy of the final answer against the ground truth |

- **Run:**

```bash
python src/train_ppo.py
```

---

## 💻 Hardware Requirements & Optimizations

This project is specifically configured to run on an **NVIDIA GPU with 4GB VRAM** (e.g., GTX 1650, RTX 3050 Laptop):

| Technique | Details |
|---|---|
| **LoRA** | `r=8, alpha=16` — Significantly reduces the number of trainable parameters |
| **Gradient Checkpointing** | Lowers memory usage by recomputing activations during the backward pass |
| **Mixed Precision (fp16)** | Accelerates training on CUDA-enabled devices |
| **Custom Forward Hooks** | Ensures gradient flow consistency in Python 3.12 environments |

---

## 🛠 Installation

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/R3-RAG.git
cd R3-RAG
```

**2. Create and activate a virtual environment:**

```bash
python -m venv myenv

# Windows
.\myenv\Scripts\activate

# Linux / macOS
source myenv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set up your Google API Key** for Gemini-based judging:

```bash
# Windows
set GOOGLE_API_KEY=your_api_key_here

# Linux / macOS
export GOOGLE_API_KEY=your_api_key_here
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Standard QA metric — checks for exact string match |
| **F1 Score** | Standard QA metric — token-level overlap score |
| **Format Adherence** | Percentage of steps correctly following the `Step X:` structure |
| **Reasoning Accuracy** | Qualitative analysis of the generated `Problem Analysis` sections |

---

## 📚 References

- **R3-RAG Paper:** *Rational Retrieval-Augmented Generation for Multi-hop Question Answering*
- **Base Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Dataset:** [HotpotQA](https://hotpotqa.github.io/)
- **Retriever:** [E5 Embeddings](https://huggingface.co/intfloat/e5-base) + [FAISS](https://github.com/facebookresearch/faiss)
- **Judge Model:** [Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/flash/)

---

</div>

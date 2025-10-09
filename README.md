# 🌿 LeafBench: A Benchmark for Evaluating Vision-Language Models in Agricultural Intelligence

LeafBench is a large-scale benchmark designed to evaluate the reasoning and perception capabilities of **Vision-Language Models (VLMs)** on **agricultural visual understanding tasks**.  
This repository provides scripts for **evaluating model accuracy and F1-score** on the benchmark dataset.

---

## 🚀 Key Features

- **Comprehensive Evaluation Framework** — Standardized code to benchmark VLMs such as CLIP, SigLIP2, BLIP-2, and LLaVA on agricultural question-answering tasks.  
- **Lightweight & Modular** — One-line command to evaluate models with automatic logging and metric computation.  
- **Reproducible** — Compatible with Hugging Face datasets and model hubs.  
- **Metrics** — Computes **Accuracy** and **F1-score** across all question types.

---

## 🧩 Repository Structure
```
LeafBench/
│
├── requirements.txt                # Python dependencies
│
├── gemini.py                       # Interface wrapper for Gemini 2.5 Pro API
├── gpt4.py                         # Interface wrapper for GPT-4o API
│
├── utils/                          # Utility scripts
│   ├── metrics.py                  # Contains accuracy and F1-score computation functions
│   └── helpers.py                  # Includes data loading, preprocessing, and prompt formatting
│
├── README.md                       # Project documentation (overview, setup, usage)
│
├── configs/                        # Configuration files
│   └── model.yaml                  # Model configuration and paths (e.g., model name, tokenizer, batch size)
│
├── scripts/                        # Automation scripts
│   └── eval.sh                     # Shell script to execute model evaluation pipeline
│
├── eval.py                         # Main evaluation script — runs inference and computes metrics
│
├── data/                           # Dataset directory
│   └── leafbench.csv               # Put data in here
│
└── results/                        # Folder for output JSON or CSV results
    └── example_result.json          # Sample evaluation output
```
---

## 📊 Evaluation Metrics

| Metric | Description |
|:--|:--|
| **Accuracy** | Measures the proportion of correct predictions. |
| **F1-score** | Balances precision and recall for uneven class distributions. |

---

## 🧠 Dataset Format

LeafBench uses a simple **CSV format** to represent the multimodal reasoning dataset:

| image_path | question | A | B | C | D | answer |
|-------------|-----------|---|---|---|---|--------|
| `val/Apple___Black_rot/img1.jpg` | What is the disease shown on this leaf? | Black rot | Rust | Scab | Healthy | A |

> You can adapt your own dataset using the same structure.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/LeafBench.git
cd LeafBench
pip install -r requirements.txt
```

---

## 🧪 Run Evaluation

You can benchmark any **Vision-Language Model (VLM)** — either open-source (via Hugging Face) or closed-source (via API like GPT-4o or Gemini) — using the unified evaluation pipeline.

---

### 🔹 1. Evaluate Open-Source Models

For models such as **CLIP**, **SigLIP2**, **BLIP-2**, **LLaVA**, or **InternVL**, run:

```bash
python eval.py \
  --model_name openai/clip-vit-base-patch16 \
  --csv_path ./data/leafbench.csv \
  --config ./configs/model.yaml \
  --output ./results/clip_result.json
```

### 🔹 2. Automated Evaluation Script

You can also run all configured models sequentially using the shell script:

```bash 
scripts/eval.sh
```


## 🌱 Dataset: [LeafBench (Hugging Face)](https://huggingface.co/datasets/enalis/LeafBench)

The **LeafBench dataset** contains over **13K visual question-answer pairs** curated from real-world plant disease datasets.  
It covers **multiple crops** and **disease types**, supporting various **question categories** such as:

- **HDC:** Healthy vs. Diseased Classification  
- **PC:** Primary Cause Reasoning  
- **SI:** Symptom Identification  
- **SNC:** Stage and Nutrient Condition  
- **CSI:** Cross-Symptom Inference  
- **DI:** Disease Identification  

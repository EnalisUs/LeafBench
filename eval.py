import os
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.metrics import compute_metrics
from utils.helpers import load_image, clean_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# HuggingFace Open-Source Models
# ---------------------------
def evaluate_hf(model_id, df):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id).to(DEVICE)
    preds, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        image = load_image(row["image_path"])
        instruction = "You are an expert in plant pathology.\n"
        prompt = instruction + f"Question: {row['question']}\nA: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}\nAnswer strictly with only the letter A, B, C, or D.\nAnswer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5)
        ans = clean_answer(processor.decode(out[0], skip_special_tokens=True))
        preds.append(ans)
        labels.append(clean_answer(row["answer"]))
    # Save json results
    os.makedirs("results", exist_ok=True)
    pd.DataFrame({"pred": preds, "label": labels}).to_json(f"results/{model_id.replace('/', '_')}_results.json", orient="records", lines=True)
    #
    acc, f1 = compute_metrics(preds, labels)
    print(f"✅ {model_id} | Accuracy: {acc:.2f}% | F1: {f1:.2f}%")

# ---------------------------
# Main Benchmark Runner
# ---------------------------
if __name__ == "__main__":
    with open("configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)
    df = pd.read_csv("data/leafbench.csv")
    
    for m in config["models"]:
        evaluate_hf(m["id"], df)
    

import base64
import pandas as pd
import openai
from sklearn.metrics import accuracy_score, f1_score

openai.api_key = "api"

# === Step 1: Load LeafBench data ===
df = pd.read_csv("leafbench.csv")

def encode_image(image_path):
    image_path  = './' + image_path
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === Step 2: Query GPT-4o ===
def ask_gpt4o(image_path, question, choices=None):
    img_b64 = encode_image(image_path)

    # Build user content
    user_content = [{"type": "text", "text": question}]
    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

    if choices:  # Add multiple-choice options explicitly
        question_with_choices = question + "\nOptions:\n" + "\n".join(choices) + \
                                "\n\nAnswer strictly with only the letter A, B, C, or D."
        user_content[0]["text"] = question_with_choices
    else:  # Yes/No (HDC)
        question_with_choices = question + "\nOptions:\nA) Yes\nB) No\n\nAnswer strictly with only A or B."
        user_content[0]["text"] = question_with_choices

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in plant pathology."},
            {"role": "user", "content": user_content}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# === Step 3: Run inference ===
preds = []
for i, row in df.iterrows():
    image_path = row["image_path"]
    question = row["question"]

    # Collect choices if available
    choices = []
    for opt in ["A", "B", "C", "D"]:
        if pd.notna(row[opt]) and row[opt] != "":
            choices.append(f"{opt}) {row[opt]}")

    prediction = ask_gpt4o(image_path, question, choices if choices else None)

    # Hard enforce: keep only A-D
    if prediction.upper() not in ["A", "B", "C", "D"]:
        prediction = prediction[0].upper() if prediction else "UNK"

    preds.append(prediction)
    print(f"[{row['question_type']}] Q: {question} | GT: {row['answer']} | Pred: {prediction}")

df["prediction"] = preds

# === Step 4: Compute accuracy per question_type ===
results = {}
for qtype, group in df.groupby("question_type"):
    acc = accuracy_score(group["answer"], group["prediction"])
    f1 = f1_score(group["answer"], group["prediction"], average="macro")
    results[qtype] = acc
    results[f"{qtype}_f1"] = f1

print("\nAccuracy per question type:")
for q, acc in results.items():
    print(f"{q}: {acc:.4f}")
    print(f"{q}_f1: {results.get(f'{q}_f1', 0):.4f}")

# Save results
df.to_csv("leafbench_gpt4o_results.csv", index=False)

import pandas as pd
from PIL import Image
import io
from sklearn.metrics import accuracy_score
from google import genai
from google.genai import types

# === Step 0: Init Gemini Client ===
client = genai.Client(api_key="api")  # replace with your key

# === Step 1: Load LeafBench data ===
df = pd.read_csv("leafbench.csv")

def load_and_resize_image(image_path, size=(224, 224)):
    """
    Resize image to (224x224) and return JPEG bytes for Gemini API.
    """
    image_path = './' + image_path
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()  # return raw bytes

# === Step 2: Query Gemini 2.5 Pro ===
def ask_gemini(image_path, question, choices=None):
    image_bytes = load_and_resize_image(image_path)
    instruction = "You are an expert in plant pathology.\n"
    if choices:  # Multiple-choice
        prompt = instruction + question + "\nOptions:\n" + "\n".join(choices) + \
                 "\n\nAnswer strictly with only the letter A, B, C, or D."
    else:  # Yes/No (HDC)
        prompt = instruction + question + "\nOptions:\nA) Yes\nB) No\n\nAnswer strictly with only A or B."

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ],
        config=types.GenerateContentConfig(temperature=0.0)
    )

    return response.text.strip() if response.text else "UNK"

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

    prediction = ask_gemini(image_path, question, choices if choices else None)

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
    results[qtype] = acc

print("\nAccuracy per question type:")
for q, acc in results.items():
    print(f"{q}: {acc:.4f}")
    print(f"{q}_f1: {results.get(f'{q}_f1', 0):.4f}")

# Save results
df.to_csv("leafbench_gemini25_results.csv", index=False)

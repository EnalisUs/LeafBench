import os
from PIL import Image

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")

def clean_answer(ans):
    ans = ans.strip().upper()
    if len(ans) > 0:
        return ans[0]  # first character (A/B/C/D)
    return "?"

from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, average="macro") * 100
    return acc, f1

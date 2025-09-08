import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_chatbot(log_file="chatbot_logs.csv"):
    # Load logs
    df = pd.read_csv(
        log_file,
        names=["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"]
    )

    # Drop rows without feedback
    df = df.dropna(subset=["correct"])

    # Convert correct (yes/no feedback) into int
    df["correct"] = df["correct"].astype(int)

    # Predicted correctness (1 = correct, 0 = incorrect)
    y_true = df["correct"].values
    y_pred = [1 if c == 1 else 0 for c in df["correct"].values]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("📊 Chatbot Evaluation Results")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

if __name__ == "__main__":
    evaluate_chatbot()

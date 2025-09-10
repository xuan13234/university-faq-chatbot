import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_chatbot(log_file="chatbot_logs.csv", output_file="chatbot_evaluation.csv"):
    # Load logs
    df = pd.read_csv(
        log_file,
        names=["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"]
    )

    # Drop rows without feedback
    df = df.dropna(subset=["correct"])

    # Convert "correct" column to int (1 = correct, 0 = incorrect)
    df["correct"] = df["correct"].astype(int)

    # Intent classification evaluation
    y_true = df["correct"].values
    y_pred = [1 if c == 1 else 0 for c in df["correct"].values]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("📊 Chatbot Evaluation Results (Intent Classification)")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}\n")

    # Response generation evaluation with BLEU score
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for _, row in df.iterrows():
        reference = [row["user_input"].split()]   # treating user input as reference (demo)
        candidate = row["response"].split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(bleu)

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    print("📝 Response Quality (BLEU Score)")
    print(f"Average BLEU Score: {avg_bleu:.2f}")

    # Save evaluation results to CSV
    results = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Avg BLEU"],
        "Score": [accuracy, precision, recall, f1, avg_bleu]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    print(f"\n✅ Evaluation results saved to {output_file}")

if __name__ == "__main__":
    evaluate_chatbot()

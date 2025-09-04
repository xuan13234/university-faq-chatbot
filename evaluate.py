import joblib, json
from pathlib import Path
from sklearn.metrics import classification_report
from train_intent_classifier import load_data

DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

texts, labels = load_data()
clf = joblib.load(MODEL_PATH)

y_pred = clf.predict([t.lower() for t in texts])
print(classification_report(labels, y_pred))
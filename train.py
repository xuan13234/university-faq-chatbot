import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import DeepChatbot
from nltk_utils import tokenize, stem
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------
# Load dataset
# ----------------------
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem & lowercase
ignore = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Vocabulary
word2idx = {w: i+1 for i, w in enumerate(all_words)}  # +1 for padding idx=0
vocab_size = len(word2idx) + 1

def encode_sentence(tokens, max_len=10):
    ids = [word2idx.get(stem(w), 0) for w in tokens]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids[:max_len]

# Encode dataset
X = [encode_sentence(pattern) for pattern, tag in xy]
y = [tags.index(tag) for pattern, tag in xy]

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dataset & loader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ChatDataset(X_train, y_train), batch_size=8, shuffle=True)

# ----------------------
# Train model
# ----------------------
embed_dim = 64
hidden_size = 128
num_classes = len(tags)
num_epochs = 100
learning_rate = 0.001

model = DeepChatbot(vocab_size, embed_dim, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}")

print("Training complete.")

# Save
data = {
    "model_state": model.state_dict(),
    "word2idx": word2idx,
    "all_words": all_words,
    "tags": tags,
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "hidden_size": hidden_size
}
torch.save(data, "data.pth")
print("Model saved to data.pth")

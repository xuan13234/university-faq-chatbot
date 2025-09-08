import torch
import torch.nn as nn

class DeepChatbot(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(DeepChatbot, self).__init__()
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM to capture context
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # bidirectional doubles hidden size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = [batch_size, seq_len]
        embeds = self.embedding(x)              # → [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embeds)         # → [batch_size, seq_len, hidden_size*2]
        last_hidden = lstm_out[:, -1, :]        # take last timestep
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

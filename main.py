import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Hyperparameters (all scaled down)
D_MODEL = 128  # Original 512 → 128
NUM_LAYERS = 2  # Original 6 → 2
NUM_HEADS = 4  # Original 8 → 4
D_FF = 256  # Original 2048 → 256
BATCH_SIZE = 32  # Original 64 → 32
EPOCHS = 10  # Original 20 → 10

# Manually create a small Chinese-English dataset (avoid downloading large datasets)
train_data = [
    ("你好", "hello"),
    ("早上好", "good morning"),
    ("再见", "goodbye"),
    ("谢谢", "thank you"),
    ("不客气", "you're welcome"),
    ("你叫什么名字", "what's your name"),
    ("今天天气如何", "how's the weather today"),
    ("我喜欢编程", "I love programming"),
    ("这是一个测试", "this is a test"),
    ("人工智能", "artificial intelligence"),
]

# Build vocabulary (simplified version)
src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
trg_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

for ch, en in train_data:
    for char in ch:
        if char not in src_vocab:
            src_vocab[char] = len(src_vocab)
    for word in en.split():
        if word not in trg_vocab:
            trg_vocab[word] = len(trg_vocab)


# Convert sentences to indices
def encode(sentence, vocab, is_src=True):
    tokens = list(sentence) if is_src else sentence.split()
    return [1] + [vocab[t] for t in tokens] + [2]


encoded_data = [
    (encode(ch, src_vocab), encode(en, trg_vocab, False)) for ch, en in train_data
]


class MiniTransformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.encoder = nn.Embedding(src_vocab_size, D_MODEL)
        self.decoder = nn.Embedding(trg_vocab_size, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=D_FF,
        )

        self.fc_out = nn.Linear(D_MODEL, trg_vocab_size)

    def forward(self, src, trg):
        src = self.pos_enc(self.encoder(src))
        trg = self.pos_enc(self.decoder(trg))

        output = self.transformer(
            src.permute(1, 0, 2),  # (S, N, E)
            trg.permute(1, 0, 2),
            tgt_mask=self.transformer.generate_square_subsequent_mask(trg.size(1)),
        )
        return self.fc_out(output.permute(1, 0, 2))


# Simplified positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


# Training preparation
model = MiniTransformer(len(src_vocab), len(trg_vocab))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)


# Training loop (simplified version)
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(encoded_data)  # Simple data shuffling

        for src, trg in encoded_data:
            src_tensor = torch.LongTensor(src).unsqueeze(0)  # (1, seq_len)
            trg_input = torch.LongTensor(trg[:-1]).unsqueeze(0)
            trg_output = torch.LongTensor(trg[1:]).unsqueeze(0)

            optimizer.zero_grad()
            output = model(src_tensor, trg_input)

            loss = criterion(output.view(-1, output.shape[-1]), trg_output.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(encoded_data):.4f}")


# Start training (completes in about 1 minute on CPU)
train()


# Translation function
def translate(input_str):
    model.eval()
    src = encode(input_str, src_vocab)
    src_tensor = torch.LongTensor(src).unsqueeze(0)

    trg_init = [trg_vocab["<sos>"]]
    for _ in range(20):  # Maximum generation length
        trg_tensor = torch.LongTensor(trg_init).unsqueeze(0)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)

        next_word = output.argmax(2)[:, -1].item()
        trg_init.append(next_word)
        if next_word == trg_vocab["<eos>"]:
            break

    return " ".join([k for k, v in trg_vocab.items() if v in trg_init[1:-1]])


# Test translation
print(translate("你好"))
print(translate("早上好"))
print(translate("这是一个测试"))

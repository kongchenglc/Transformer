import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from models.transformer import Transformer
from models.positional_encoding import PositionalEncoding

# Hyperparameters
D_MODEL = 128  # Reduced model dimension for faster training
NUM_LAYERS = 2  # Number of encoder/decoder layers
NUM_HEADS = 4  # Number of attention heads
D_FF = 256  # Feed-forward dimension
BATCH_SIZE = 32  # Training batch size
EPOCHS = 10  # Training epochs
DROPOUT = 0.1  # Dropout probability

# Sample training data (Chinese-English pairs)
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

# Build vocabulary dictionaries
src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}  # Source (Chinese) vocabulary
trg_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}  # Target (English) vocabulary

# Populate vocabulary from training data
for ch, en in train_data:
    for char in ch:
        if char not in src_vocab:
            src_vocab[char] = len(src_vocab)
    for word in en.split():
        if word not in trg_vocab:
            trg_vocab[word] = len(trg_vocab)


# Sentence encoding function with padding
def encode(sentence, vocab, is_src=True, max_len=10):
    """Convert sentence to token indices with padding"""
    tokens = list(sentence) if is_src else sentence.split()
    encoded = [1] + [vocab[t] for t in tokens] + [2]  # Add <sos> and <eos>

    # Handle padding/truncation
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))  # Pad with zeros
    else:
        encoded = encoded[: max_len - 1] + [2]  # Truncate and keep <eos>
    return encoded[:max_len]


# Generate encoded dataset with fixed sequence length
max_seq_len = 10
encoded_data = [
    (
        encode(ch, src_vocab, max_len=max_seq_len),
        encode(en, trg_vocab, False, max_seq_len),
    )
    for ch, en in train_data
]


# Mask creation function
def create_mask(src, tgt):
    """Create padding mask and causal attention mask"""
    # Source padding mask (N, 1, 1, S)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    # Target padding mask
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)

    # Causal mask to prevent looking ahead
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()
    return src_mask, tgt_mask & nopeak_mask


# Initialize Transformer model
model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(trg_vocab),
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
)

# Configure optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index


# Training loop with batching
def train():
    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(encoded_data)
        total_loss = 0

        # Process data in batches
        for i in range(0, len(encoded_data), BATCH_SIZE):
            batch = encoded_data[i : i + BATCH_SIZE]
            src_batch = torch.LongTensor([s for s, t in batch])
            tgt_batch = torch.LongTensor([t for s, t in batch])

            # Create masks
            src_mask, tgt_mask = create_mask(src_batch, tgt_batch[:, :-1])

            optimizer.zero_grad()

            # Forward pass
            output = model(
                src=src_batch,
                tgt=tgt_batch[:, :-1],
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

            # Calculate loss
            loss = criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_batch[:, 1:].contiguous().view(-1),
            )

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(encoded_data)/BATCH_SIZE):.4f}")


# Start training
train()


# Translation function
def translate(input_str):
    model.eval()
    src = encode(input_str, src_vocab, max_len=max_seq_len)
    src_tensor = torch.LongTensor(src).unsqueeze(0)

    # Initialize target with <sos>
    trg_init = [trg_vocab["<sos>"]]

    # Generate output tokens
    for i in range(max_seq_len - 1):  # Reserve one position for <eos>
        trg_tensor = torch.LongTensor(trg_init).unsqueeze(0)

        # Create masks
        src_mask, tgt_mask = create_mask(src_tensor, trg_tensor)

        with torch.no_grad():
            output = model(
                src=src_tensor, tgt=trg_tensor, src_mask=src_mask, tgt_mask=tgt_mask
            )

        # Get next token
        next_token = output.argmax(2)[:, -1].item()
        trg_init.append(next_token)
        if next_token == trg_vocab["<eos>"]:
            break

    # Convert indices to text
    return " ".join([k for k, v in trg_vocab.items() if v in trg_init[1:-1]])


# Test translations
print("Translation examples:")
print("你好 →", translate("你好"))
print("早上好 →", translate("早上好"))
print("这是一个测试 →", translate("这是一个测试"))

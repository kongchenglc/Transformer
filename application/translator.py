# translator.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformer.transformer import Transformer

class Translator:
    def __init__(
        self,
        train_data,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        batch_size=32,
        epochs=10,
        dropout=0.1,
        max_seq_len=10
    ):
        # Hyperparameters
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Data preparation
        self.train_data = train_data
        self._build_vocab()
        self.encoded_data = self._preprocess_data()

        # Model components
        self.model = Transformer(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.trg_vocab),
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _build_vocab(self):
        """Build vocabulary from training data"""
        self.src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.trg_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

        for fr, en in self.train_data:
            for word in fr.split():
                if word not in self.src_vocab:
                    self.src_vocab[word] = len(self.src_vocab)
            for word in en.split():
                if word not in self.trg_vocab:
                    self.trg_vocab[word] = len(self.trg_vocab)

    def _encode(self, sentence, vocab, is_src=True):
        """Convert sentence to token indices with padding"""
        tokens = sentence.split() if is_src else sentence.split()
        encoded = [1] + [vocab[t] for t in tokens] + [2]
        
        # Handle padding/truncation
        if len(encoded) < self.max_seq_len:
            encoded += [0] * (self.max_seq_len - len(encoded))
        else:
            encoded = encoded[:self.max_seq_len-1] + [2]
        return encoded[:self.max_seq_len]

    def _preprocess_data(self):
        """Generate encoded dataset"""
        return [
            (
                self._encode(fr, self.src_vocab),
                self._encode(en, self.trg_vocab, False)
            )
            for fr, en in self.train_data
        ]

    def _create_mask(self, src, tgt):
        """Create padding mask and causal attention mask"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(
            torch.ones(1, seq_len, seq_len), 
            diagonal=1)
        ).bool()
        return src_mask, tgt_mask & nopeak_mask

    def train(self):
        """Training loop"""
        self.model.train()
        for epoch in range(self.epochs):
            random.shuffle(self.encoded_data)
            total_loss = 0

            for i in range(0, len(self.encoded_data), self.batch_size):
                batch = self.encoded_data[i:i+self.batch_size]
                src_batch = torch.LongTensor([s for s, t in batch])
                tgt_batch = torch.LongTensor([t for s, t in batch])

                src_mask, tgt_mask = self._create_mask(
                    src_batch, 
                    tgt_batch[:, :-1]
                )

                self.optimizer.zero_grad()

                output = self.model(
                    src=src_batch,
                    tgt=tgt_batch[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )

                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_batch[:, 1:].contiguous().view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(self.encoded_data)/self.batch_size)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    def translate(self, input_str):
        """Translation inference"""
        self.model.eval()
        src = self._encode(input_str, self.src_vocab)
        src_tensor = torch.LongTensor(src).unsqueeze(0)
        trg_init = [self.trg_vocab["<sos>"]]

        for _ in range(self.max_seq_len - 1):
            trg_tensor = torch.LongTensor(trg_init).unsqueeze(0)
            src_mask, tgt_mask = self._create_mask(src_tensor, trg_tensor)

            with torch.no_grad():
                output = self.model(
                    src=src_tensor,
                    tgt=trg_tensor,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )

            next_token = output.argmax(2)[:, -1].item()
            trg_init.append(next_token)
            if next_token == self.trg_vocab["<eos>"]:
                break

        return " ".join([
            k for k, v in self.trg_vocab.items() 
            if v in trg_init[1:-1]
        ])

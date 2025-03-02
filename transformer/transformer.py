import torch.nn as nn
from transformer.positional_encoding import PositionalEncoding
from transformer.encoder import EncoderLayer
from transformer.decoder import DecoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encoder
        src_emb = self.positional_encoding(self.encoder_embedding(src))
        src_emb = self.dropout(src_emb)
        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Decoder
        tgt_emb = self.positional_encoding(self.decoder_embedding(tgt))
        dec_output = tgt_emb
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.fc_out(dec_output)

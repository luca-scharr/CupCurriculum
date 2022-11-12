# Importing Libraries
import math

# Highlevel from Pytorch
import torch as T
from torch import nn, Tensor

# Neural Network parts from Pytorch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Defining the Architecture
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, decoder_type:str = "small"):
        super().__init__()
        self.model_type          = 'Transformer'
        self.pos_encoder         = PositionalEncoding(d_model, dropout)
        encoder_layers           = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder             = nn.Embedding(ntoken, d_model)
        self.d_model             = d_model
        if decoder_type == "small":
            self.decoder = nn.Linear(d_model, ntoken)
            """
            elif "medium_decoder" in decoder_type:
                self.decoder = nn.Sequential(nn.Linear(d_model, 2* d_model), nn.ReLU(), nn.Linear(2* d_model, 2* ntoken), nn.ReLU(), nn.Linear(2* ntoken, ntoken))
            """
        else:
            self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src    = self.encoder(src) * math.sqrt(self.d_model)  # Wordembeddings
        src    = self.pos_encoder(src)  # Positional Encoding
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return T.triu(T.ones(sz, sz) * float('-inf'), diagonal=1)


# Implementing Positional Encoding, i.e. where are the words in the Sentence
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        position       = T.arange(max_len).unsqueeze(1)
        div_term       = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe             = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


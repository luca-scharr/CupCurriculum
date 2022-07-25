import math
from typing import Optional, Any, Union, Callable
import torch as T
from torch import nn, Tensor
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn import LayerNorm

class TransformerModel(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
             num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
             activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
             layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
             device=None, dtype=None) -> None:
        """
        Similar to the transformer model implemented by pytorch, difference being that
        this code forces positional encoding and gets rid of some special cases.
        For excessive documentation see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Also have a look in the corresponding sourcecode linked there.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerModel, self).__init__()
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Buildinig the encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # Building the decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        #Init weights
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Similar to the forward pass of the transformer model implemented by pytorch,
        difference being that this code forces positional encoding and gets rid of some special cases.
        For excessive documentation see https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        Also have a look in the corresponding sourcecode linked there.
        """
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        # Use positional encoding
        src = self.pos_encoder(src)
        # Apply the encoder
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # Aplly the decoder
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = T.arange(max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, 1, d_model)
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
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(ABC, nn.Module):
    """
    Abstract base class for all models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def configure_optimizers(self, learning_rate):
        pass

class PositionalEncoding(nn.Module):
    """
    Positional encoding module as described in "Attention is All You Need".
    Adds positional information to the input embeddings.
    """
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (sequence_length, batch_size, embedding_dim)
        Returns:
            Tensor of the same shape with positional encoding added.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(BaseModel):
    """
    Standard Transformer model for biological sequence prediction.
    """
    def __init__(
        self, vocab_size, embedding_dim=128, num_layers=6, num_heads=8,
        hidden_dim=512, max_seq_length=1024, dropout=0.1
    ):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Decoder (output layer)
        self.decoder = nn.Linear(embedding_dim, vocab_size)

        self.embedding_dim = embedding_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for the transformer model.

        Args:
            src: Input tensor of shape (sequence_length, batch_size)
            src_mask: Optional tensor for masking source sequences
            src_key_padding_mask: Optional tensor for masking padding tokens

        Returns:
            Output tensor of shape (sequence_length, batch_size, vocab_size)
        """
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(output)
        return output

    def configure_optimizers(self, learning_rate):
        """
        Configures the optimizer for training.

        Args:
            learning_rate: Learning rate for the optimizer.

        Returns:
            optimizer: An instance of torch.optim.Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    """Pre-norm GPT-2 style transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        normalized = self.ln_1(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(attended)
        return x + self.mlp(self.ln_2(x))


class GPT2Backbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 96,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.max_len:
            tokens = tokens[:, -self.max_len :]
            lengths = lengths.clamp(max=self.max_len)
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        padding_mask = tokens.eq(0)
        causal_mask = torch.triu(
            torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)
        return self.final_norm(x)

    def encode_last(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self(tokens, lengths)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        indices = lengths.clamp(max=hidden.size(1)) - 1
        return hidden[rows, indices]


class GPT2Rec(nn.Module):
    """Causal decoder with tied input/output item embeddings."""

    def __init__(self, vocab_size: int, max_len: int, **kwargs):
        super().__init__()
        self.backbone = GPT2Backbone(vocab_size, max_len, **kwargs)
        self.logit_scale = self.backbone.d_model**-0.5

    def forward(self, history: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        last = self.backbone.encode_last(history, lengths)
        # Weight tying mirrors GPT-2's token embedding / language-model head.
        return F.linear(last, self.backbone.token_embedding.weight) / self.logit_scale


class RewardModel(nn.Module):
    """History-and-candidate response model with completion and return heads."""

    def __init__(self, vocab_size: int, max_len: int, **kwargs):
        super().__init__()
        self.backbone = GPT2Backbone(vocab_size, max_len + 1, **kwargs)
        width = self.backbone.d_model
        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(width, 2),
        )

    def forward(
        self, history: torch.Tensor, lengths: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        batch, width = history.shape
        sequence = torch.zeros(batch, width + 1, dtype=history.dtype, device=history.device)
        sequence[:, :width] = history
        rows = torch.arange(batch, device=history.device)
        sequence[rows, lengths] = candidate
        encoded = self.backbone.encode_last(sequence, lengths + 1)
        return self.head(encoded)


"""Small classifier initialized and trained locally; imported only for custom models."""

from typing import cast

import torch
from torch import Tensor, nn

from inference_service.adapters.custom_artifacts import CustomConfig


class SentimentNetwork(nn.Module):
    def __init__(self, vocabulary_size: int, config: CustomConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=0)
        self.hidden = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 2)

    def forward(self, token_ids: Tensor) -> Tensor:
        mask = token_ids.ne(0).unsqueeze(-1)
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return cast(Tensor, self.output(torch.relu(self.hidden(pooled))))


def padded_batch(sequences: list[list[int]], device: str = "cpu") -> Tensor:
    width = max(map(len, sequences))
    return torch.tensor(
        [sequence + [0] * (width - len(sequence)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )

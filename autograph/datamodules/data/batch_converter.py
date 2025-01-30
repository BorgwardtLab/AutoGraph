import torch
from typing import Sequence


class BatchConverter(object):
    """Callable to convert a list of trails to a processed batch.
    """

    def __init__(self, tokenizer, truncation_length: int = None):
        self.tokenizer = tokenizer
        self.truncation_length = truncation_length

    def __call__(self, batch: Sequence[torch.Tensor]):
        batch_size = len(batch)

        max_len = max([len(b) for b in batch])
        if self.truncation_length is not None:
            max_len = min(max_len, self.truncation_length)

        batched_tensor = torch.full(
            [batch_size, max_len],
            self.tokenizer.pad,
            dtype=batch[0].dtype
        )

        for i, x in enumerate(batch):
            if self.truncation_length is not None and self.truncation_length < len(x):
                start_idx = torch.randint(0, len(x) - self.truncation_length, (1,)).item()
                x = x[start_idx:start_idx + self.truncation_length]
                # x = x[:self.truncation_length]
            batched_tensor[i, :len(x)] = x

        return batched_tensor

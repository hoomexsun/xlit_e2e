import torch
from torch.utils.data import Dataset


class TransliterationDataset(Dataset):
    def __init__(self, x, y, x_tokenizer, y_tokenizer, max_len) -> None:
        self.x = x
        self.y = y
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        x_encoded = self.x_tokenizer.encode(self.x[idx], self.max_len)
        y_encoded = self.y_tokenizer.encode(self.y[idx], self.max_len)
        return torch.tensor(x_encoded, dtype=torch.long), torch.tensor(
            y_encoded, dtype=torch.long
        )

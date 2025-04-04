from torch.utils.data import Dataset
import torch
from pathlib import Path


class TransliterationDataset(Dataset):
    def __init__(self, x_data, y_data, x_tokenizer, y_tokenizer, max_len):
        self.x_data = x_data
        self.y_data = y_data
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_tokenizer.encode(self.x_data[idx])
        y = self.y_tokenizer.encode(self.y_data[idx])
        x = self._pad(x)
        y = self._pad(y)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def _pad(self, seq):
        if len(seq) < self.max_len:
            seq += [self.x_tokenizer.pad_idx] * (self.max_len - len(seq))
        else:
            seq = seq[: self.max_len]
        return seq


def load_data(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    x_data, y_data = [], []
    for line in lines:
        if "\t" in line:
            x, y = line.strip().split("\t")
            x_data.append(x)
            y_data.append(y)
    return x_data, y_data

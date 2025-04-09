import torch
from .dataset import TransliterationDataset


__all__ = ["load_dataloaders"]


def load_dataloaders(
    xs, ys, x_tokenizer, y_tokenizer, max_len, batch_size=32, val_ratio=0.2
):

    dataset = TransliterationDataset(xs, ys, x_tokenizer, y_tokenizer, max_len)
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, val_dataloader

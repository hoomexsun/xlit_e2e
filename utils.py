import torch
import numpy as np
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: Path):
    log_path = log_dir / "train.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x, y, teacher_forcing_ratio=0.0)
            loss = criterion(
                output[:, 1:].reshape(-1, output.shape[2]), y[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def plot_loss(losses, save_path: Path):
    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

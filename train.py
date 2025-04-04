import torch
import random
import numpy as np
import logging
import json
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from model import Seq2Seq
from dataset import TransliterationDataset, load_data
from tokenizer import CharTokenizer
from checkpoint.save import CheckpointManager
from utils import set_seed, setup_logging, evaluate, plot_loss


def train_loop(
    data_path: Path,
    exp_dir: Path,
    seed=42,
    embed_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.5,
    batch_size=32,
    learning_rate=0.001,
    epochs=50,
    keep_nbest_models=5,
):
    set_seed(seed)
    log_dir = exp_dir / "log"
    image_dir = exp_dir / "images"
    log_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)

    x_data, y_data = load_data(data_path)
    x_vocab = set("".join(x_data))
    y_vocab = set("".join(y_data))
    max_len = max(max(map(len, x_data)), max(map(len, y_data))) + 2

    config = {
        "x_vocab": list(x_vocab),
        "y_vocab": list(y_vocab),
        "max_len": max_len,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    (Path("data") / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    x_tokenizer = CharTokenizer(x_vocab)
    y_tokenizer = CharTokenizer(y_vocab)

    dataset = TransliterationDataset(x_data, y_data, x_tokenizer, y_tokenizer, max_len)
    train_size = int(0.8 * len(dataset))
    train_dataset, valid_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(
        input_dim=len(x_tokenizer.char2idx),
        output_dim=len(y_tokenizer.char2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    checkpoint_manager = CheckpointManager(exp_dir, keep_nbest_models)
    last_epoch = checkpoint_manager.load_if_available(model, optimizer)

    best_valid_loss = float("inf")
    train_losses = []
    valid_losses = []

    for epoch in range(last_epoch + 1, epochs + 1):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, y)
            loss = criterion(
                output[:, 1:].reshape(-1, output.shape[2]), y[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_valid_loss = evaluate(model, valid_loader, criterion, device)

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)

        logging.info(
            f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}"
        )
        checkpoint_manager.save(model, optimizer, epoch, avg_train_loss, avg_valid_loss)

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            logging.info(f"New best valid loss: {best_valid_loss:.4f}")

        if epoch % 10 == 0:
            plot_path = image_dir / f"loss_epoch_{epoch}.png"
            plot_loss(train_losses, plot_path)

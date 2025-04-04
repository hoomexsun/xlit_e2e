import torch
import logging
from pathlib import Path


def load_model(model, optimizer, exp_dir: Path):
    checkpoints = list(exp_dir.glob("checkpoint_epoch*.pth"))
    if not checkpoints:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0
    latest_ckpt = max(
        checkpoints, key=lambda p: int(p.stem.replace("checkpoint_epoch", ""))
    )
    state = torch.load(latest_ckpt)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    epoch = state.get("epoch", 0)
    logging.info(f"Resumed from checkpoint: {latest_ckpt}")
    return epoch

import torch
import logging
from pathlib import Path


class CheckpointManager:
    def __init__(self, exp_dir: Path, keep_nbest: int):
        self.exp_dir = exp_dir
        self.keep_nbest = keep_nbest
        self.saved = []

    def save(self, model, optimizer, epoch, train_loss, valid_loss):
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        fname = f"checkpoint_epoch{epoch}.pth"
        fpath = self.exp_dir / fname
        torch.save(state, fpath)
        logging.info(f"Checkpoint saved: {fpath}")
        self.saved.append((valid_loss, fpath))

        # Save best/average models
        self.saved.sort(key=lambda x: x[0])
        if len(self.saved) > self.keep_nbest:
            _, to_remove = self.saved.pop()
            to_remove.unlink(missing_ok=True)
            logging.info(f"Removed old checkpoint: {to_remove}")

        best_path = self.exp_dir / "valid.loss.best.pth"
        torch.save(state, best_path)

        avg_state = self._average_checkpoints([p for _, p in self.saved])
        torch.save(avg_state, self.exp_dir / "valid.loss.ave.pth")

    def _average_checkpoints(self, paths):
        avg_state = {}
        n = len(paths)
        for k in torch.load(paths[0])["model_state"]:
            avg_state[k] = sum(torch.load(p)["model_state"][k] for p in paths) / n
        return {"model_state": avg_state}

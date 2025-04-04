from pathlib import Path
import torch
import json
from model import Seq2Seq
from tokenizer import CharTokenizer
from train import train_loop
from dataset import load_data
from checkpoint.save import CheckpointManager


class Xlit:
    def __init__(self, model, x_tokenizer, y_tokenizer, device, max_len):
        self.model = model
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: Path, config_path: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = json.loads(config_path.read_text(encoding="utf-8"))

        x_tokenizer = CharTokenizer(set(config["x_vocab"]))
        y_tokenizer = CharTokenizer(set(config["y_vocab"]))

        model = Seq2Seq(
            input_dim=len(x_tokenizer.char2idx),
            output_dim=len(y_tokenizer.char2idx),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            device=device,
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, x_tokenizer, y_tokenizer, device, config["max_len"])

    @classmethod
    def from_scratch(cls):
        return cls(None, None, None, None, None)

    def train(self, data_path: Path, exp_dir: Path):
        train_loop(data_path, exp_dir)

        # Load config from file
        config = json.loads((Path("data") / "config.json").read_text(encoding="utf-8"))
        model_path = exp_dir / "best_model.pth"
        return Xlit.from_pretrained(model_path, Path("data") / "config.json")

    def predict(self, text: str):
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.x_tokenizer.encode(text)
            input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(self.device)

            hidden, cell = self.model.encoder(input_tensor)
            encoder_outputs = hidden[-1].unsqueeze(1).repeat(1, input_tensor.size(1), 1)

            inp = torch.full(
                (1,), self.y_tokenizer.char2idx["<sos>"], dtype=torch.long
            ).to(self.device)

            pred = []
            for _ in range(self.max_len):
                output, hidden, cell = self.model.decoder(
                    inp, hidden, cell, encoder_outputs
                )
                inp = output.argmax(1)
                pred.append(inp)

            pred = torch.stack(pred, dim=1)
            decoded = self.y_tokenizer.decode(pred[0].cpu().numpy())

        return {"input": text, "output": decoded}

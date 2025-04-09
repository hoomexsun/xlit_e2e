from pathlib import Path

from matplotlib import pyplot as plt
from torch import nn
import torch

from dataset import load_dataloaders
from models.seq2seq import Seq2Seq
from tokenizers import build_tokenizer
from utils.loader import load_data, load_yaml


# 1. Load data
PARAMS_FILE = "data/params.yaml"
xs, ys, params = load_data("data/transcribed.txt", params_file=PARAMS_FILE)

# 2. Get parameters
# Uncomment to load from file instead
# params = load_yaml(PARAMS_FILE)
BATCH_SIZE = params["batch_size"]
EMBED_DIM = params["embed_dim"]
HIDDEN_DIM = params["hidden_dim"]
MAX_LEN = params["max_len"]
X_VOCAB = params["x_vocab"]
Y_VOCAB = params["y_vocab"]

xlit_dict = load_yaml("conf/train.yaml")
model_name = xlit_dict["xlit"]
xlit_conf = xlit_dict["xlit_conf"]
ENCODER_LAYERS = xlit_conf["encoder_layers"]
DECODER_LAYERS = xlit_conf["decoder_layers"]
DROPOUT_RATE = xlit_conf["dropout_rate"]
LR = xlit_conf["optim_conf"]["lr"]
MAX_EPOCH = xlit_conf["max_epoch"]

# 3. Tokenizer
token_type = "char"
lang1 = "ben"
lang2 = "mni"

x_tokenizer = build_tokenizer(xs, token_type, lang1)
y_tokenizer = build_tokenizer(ys, token_type, lang2)

# 4. DataLoader
train_dataloader, val_dataloader = load_dataloaders(
    xs,
    ys,
    x_tokenizer,
    y_tokenizer,
    max_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    val_ratio=0.2,
)

## Print DataLoader Information
print(f"{BATCH_SIZE=}")
print(f"Size of train dataloader: {len(train_dataloader)}")
print(f"Size of validation dataloader: {len(val_dataloader)}")

# 5. Model, optimizer and criterion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2Seq(
    len(x_tokenizer.tok2idx),
    len(y_tokenizer.tok2idx),
    EMBED_DIM,
    HIDDEN_DIM,
    ENCODER_LAYERS,
    DROPOUT_RATE,
    DEVICE,
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

## Print model Summary
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")


# 6. Training Loop
EXP_DIR = f"exp/{model_name}_{token_type}_{lang1}_{lang2}"
Path(EXP_DIR).mkdir(exist_ok=True)
train_losses = []
for epoch in range(1, MAX_EPOCH + 1):
    model.train()
    epoch_loss = 0
    for x, y in train_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(
            x,
            max_len=MAX_LEN,
            sos_token=y_tokenizer.tok2idx["<sos>"],
            eos_token=y_tokenizer.tok2idx["<eos>"],
        )
        loss = criterion(
            y_pred[:, 1:].reshape(-1, y_pred.shape[2]), y[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_dataloader))
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    # Save model every 10th epoch
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{EXP_DIR}/epoch_{epoch}.pth")


# Plot
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


## Inference

# model.eval()
# with torch.inference_mode():
#     x = ...
#     x = x.to(DEVICE)
#     predictions = model(x.to(DEVICE), y=None, max_len=MAX_LEN)

#     pred_ids = predictions.argmax(dim=2)

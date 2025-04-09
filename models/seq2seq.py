from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([hidden, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=embed_dim
        )
        self.rnn = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)

    def forward(self, inp, hidden, cell, encoder_outputs) -> torch.Tensor:
        inp = inp.unsqueeze(1)
        embedded = self.dropout(self.embedding(inp))
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(
            torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        )
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self, input_dim, output_dim, embed_dim, hidden_dim, num_layers, dropout, device
    ):
        super().__init__()
        self.encoder = Encoder(
            input_dim, embed_dim, hidden_dim, num_layers, dropout
        ).to(device)
        self.decoder = Decoder(
            output_dim, embed_dim, hidden_dim, num_layers, dropout
        ).to(device)
        self.device = device

    def forward(
        self,
        x,
        y=None,
        teacher_forcing_ratio=0.5,
        max_len=None,
        sos_token=None,
        eos_token=None,
    ):
        y_vocab_size = self.decoder.fc_out.out_features
        batch_size = x.size(0)

        hidden, cell = self.encoder(x)
        encoder_outputs = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)

        if y is not None:
            length = y.size(1)
        else:
            assert (
                max_len is not None and sos_token is not None
            ), "Need max_len and sos_token for inference"
            length = max_len

        outputs = torch.zeros(batch_size, length, y_vocab_size).to(self.device)
        inp = (
            y[:, 0]
            if y is not None
            else torch.full((batch_size,), sos_token, dtype=torch.long).to(self.device)
        )

        for t in range(1, length):
            output, hidden, cell = self.decoder(inp, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)

            if y is not None:
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                inp = y[:, t] if teacher_force else top1
            else:
                inp = top1
                # Optional: break if all outputs in batch are <eos>
                if eos_token is not None and (inp == eos_token).all():
                    break

        return outputs

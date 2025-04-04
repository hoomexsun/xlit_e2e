import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


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
    def __init__(
        self, output_dim, embed_dim, hidden_dim, num_layers, attention, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.LSTM(
            embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, inp, hidden, cell, encoder_outputs):
        inp = inp.unsqueeze(1)  # shape: [batch_size, 1]
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
        attention = Attention(hidden_dim)
        self.encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(
            output_dim, embed_dim, hidden_dim, num_layers, dropout, attention
        )
        self.device = device

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else 20
        output_dim = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_len, output_dim).to(self.device)
        hidden, cell = self.encoder(src)

        input_token = trg[:, 0]  # <sos> token
        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, None)
            outputs[:, t] = output
            teacher_force = (
                trg is not None and torch.rand(1).item() < teacher_forcing_ratio
            )
            input_token = trg[:, t] if teacher_force else output.argmax(1)

        return outputs

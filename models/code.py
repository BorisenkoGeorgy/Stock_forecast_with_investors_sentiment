import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(hid_dim, 16, 5),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 5),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 5, stride=2),
            ]
        )
        self.proj = nn.Linear(64, 5)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x.transpose(1, 2)).mean(-1)
        return self.proj(x)


class CRNN(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(hid_dim, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32)
            ]
        )
        self.rnn = nn.LSTM(32, 64, 2, batch_first=True)
        self.proj = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x.transpose(1, 2))
        _, (h_n, _) = self.rnn(x.transpose(1, 2))
        return self.proj(h_n.view(h_n.shape[1], -1))
    
class CBiRNN(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(hid_dim, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32)
            ]
        )
        self.rnn = nn.LSTM(32, 64, 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(256, 5)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x.transpose(1, 2))
        _, (h_n, _) = self.rnn(x.transpose(1, 2))
        return self.proj(h_n.view(h_n.shape[1], -1))
    
class CTransformer(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(hid_dim, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
            ]
        )
        transformer_block = nn.TransformerEncoderLayer(64, 4, 64*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_block, 4)
        self.proj = nn.Linear(64, 5)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x.transpose(1, 2))
        out = self.transformer(x.transpose(1, 2)).mean(1)
        return self.proj(out)
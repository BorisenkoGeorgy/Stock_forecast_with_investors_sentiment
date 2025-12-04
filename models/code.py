import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(5, 16, 5),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 5),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, 5, stride=2),
            ]
        )
        self.proj = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x).mean(-1)
        return self.proj(x).squeeze()


class CRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(5, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32)
            ]
        )
        self.rnn = nn.LSTM(32, 64, 2, batch_first=True)
        self.proj = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.proj(out[:,-1,:]).squeeze()
    
class CBiRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(5, 16, 3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                nn.Conv1d(16, 32, 3, padding=1),
                nn.BatchNorm1d(32)
            ]
        )
        self.rnn = nn.LSTM(32, 64, 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x).transpose(1, 2)
        out, _ = self.rnn(x)
        out = out.reshape(*x.shape[:2], 64, 2)
        out = torch.cat([out[:, -1, :, 0], out[:, 0, :, 1]],dim=-1)
        return self.proj(out).squeeze()
    
class CTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_module = nn.Sequential(
            *[
                nn.Conv1d(5, 16, 3, padding=1),
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
        self.proj = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_module(x).transpose(1, 2)
        out = self.transformer(x).mean(1)
        return self.proj(out).squeeze()
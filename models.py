import torch
import torch.nn as nn

class DeepFusionNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.climate_dim = input_dim - 2
        self.climate_fc = nn.Sequential(
            nn.Linear(self.climate_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        self.geo_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        self.attention = nn.MultiheadAttention(embed_dim=160, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        climate = x[:, :-2]
        geo = x[:, -2:]
        climate_feat = self.climate_fc(climate)
        geo_feat = self.geo_fc(geo)
        combined = torch.cat([climate_feat, geo_feat], dim=1)
        attn_out, _ = self.attention(combined.unsqueeze(0), combined.unsqueeze(0), combined.unsqueeze(0))
        return self.fc(attn_out.squeeze(0))

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

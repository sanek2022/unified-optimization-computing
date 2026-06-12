import torch
import torch.nn as nn

class WorkloadPredictor(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=2
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            return self.forward(x).item()
import torch
import torch.nn as nn

class WorkloadPredictor(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()

        self.model = nn.Transformer(
            d_model=32,
            nhead=4,
            num_encoder_layers=2
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out = self.model(x, x)
        return self.fc(out[-1])

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        return self.forward(x).item()
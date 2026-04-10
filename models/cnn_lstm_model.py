import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):

    def __init__(self):
        super(CNNLSTM, self).__init__()

        # Pretrained EfficientNet
        backbone = models.efficientnet_b0(weights="DEFAULT")

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze early layers but train deeper layers
        for param in self.feature_extractor[:4].parameters():
            param.requires_grad = False

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        # Attention layer
        self.attention = nn.Linear(256, 1)

        # Regularization
        self.attention_dropout = nn.Dropout(0.3)

        # Final classifier (raw logits)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # Input shape: (batch, frames, channels, height, width)
        B, T, C, H, W = x.shape

        # Merge batch and time for CNN
        x = x.reshape(B * T, C, H, W)

        # Extract spatial features
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.flatten(1)

        # Restore temporal dimension
        x = x.reshape(B, T, -1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights (softmax over time)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Weighted temporal pooling
        x = torch.sum(attn_weights * lstm_out, dim=1)

        x = self.attention_dropout(x)

        # Return raw logits
        x = self.fc(x)

        return x
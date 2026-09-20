import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_out):
        weights = self.attention(rnn_out)
        weights = F.softmax(weights, dim=1)
        return torch.sum(rnn_out * weights, dim=1)


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act1(out + residual)


class ExtendedVoiceResCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=5,
                stride=(2, 2), padding=2, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            ResidualBlock2D(32, 64, stride=2),
            ResidualBlock2D(64, 64)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock2D(64, 128, stride=2),
            ResidualBlock2D(128, 128)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock2D(128, 256, stride=2),
            ResidualBlock2D(256, 256)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock2D(256, 512, stride=2),
            ResidualBlock2D(512, 512)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        return self.classifier(torch.cat([avg_pool, max_pool], dim=1))


class VoiceCRNN(nn.Module):
    def __init__(self, in_channels=1, rnn_hidden=128, rnn_layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None))
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temporal_attn = TemporalAttention(rnn_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x).squeeze(2).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.temporal_attn(x)
        return self.classifier(x)


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param=16,
        time_mask_param=24,
        num_freq_masks=2,
        num_time_masks=2
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec):
        if not self.training:
            return spec

        output = spec.clone()
        _, _, freq, time = output.shape

        for _ in range(self.num_freq_masks):
            width = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
            start = int(
                torch.randint(0, max(1, freq - width + 1), (1,)).item()
            )
            output[:, :, start:start + width, :] = -80.0

        for _ in range(self.num_time_masks):
            width = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
            start = int(
                torch.randint(0, max(1, time - width + 1), (1,)).item()
            )
            output[:, :, :, start:start + width] = -80.0

        return output


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class VoiceCloneDataset(Dataset):
    def __init__(self, file_paths, labels, sample_rate=16000, target_width=200):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_width = target_width

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        import librosa

        audio, _ = librosa.load(
            self.file_paths[index],
            sr=self.sample_rate,
            mono=True
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        if mel.shape[1] < self.target_width:
            mel = np.pad(
                mel,
                ((0, 0), (0, self.target_width - mel.shape[1])),
                mode="constant",
                constant_values=-80.0
            )
        else:
            mel = mel[:, :self.target_width]

        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([self.labels[index]], dtype=torch.float32)
        return tensor, target


def compute_eer(labels, scores):
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    index = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[index] + fnr[index]) / 2 * 100), float(thresholds[index])

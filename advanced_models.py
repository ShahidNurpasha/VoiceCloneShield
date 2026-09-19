"""
Extended Voice Cloning & Synthetic Audio Detection Neural Network Architectures
================================================================================
Enhanced from the baseline VoiceCNN in app.py to state-of-the-art Deep Audio Forensic Models.

Architectures included:
1. ExtendedVoiceResCNN: Deeper Residual CNN with Squeeze-and-Excitation (SE) Attention,
   Batch Normalization, Dual Pooling (Avg+Max), and Regularized Multi-Layer Head.
2. VoiceCRNN: Hybrid Convolutional-Recurrent Architecture (CNN + BiGRU + Attention Pooling)
   that preserves temporal phoneme-to-phoneme dynamics critical for detecting vocoder artifacts.
3. SpecAugment: Frequency and Time Masking for robust audio spectrogram augmentation.
4. VoiceCloneDataset & Complete PyTorch Training/Evaluation Pipeline (Focal Loss, EER metric).
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1. ATTENTION MODULES
# ============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention Block.
    Dynamically recalibrates channel-wise feature responses by explicitly
    modelling interdependencies between spectral channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(x).view(b, c, 1, 1)
        return x * weights


class TemporalAttention(nn.Module):
    """
    Self-attention pooling over the temporal dimension.
    Learns to weigh speech frames that carry synthetic glitches (e.g. vocoder phase artifacts).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out shape: (Batch, Time, HiddenDim)
        attn_weights = self.attention(rnn_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(rnn_out * attn_weights, dim=1)  # (B, HiddenDim)
        return context


# ============================================================================
# 2. RESIDUAL CONVOLUTIONAL BLOCKS
# ============================================================================

class ResidualBlock2D(nn.Module):
    """
    Residual Block with Conv2D, BatchNorm, SiLU activation, SE attention,
    and shortcut projection when channel count or resolution changes.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out


# ============================================================================
# 3. ARCHITECTURE 1: EXTENDED VOICE RES-CNN (DEEP SPECTRAL RESNET)
# ============================================================================

class ExtendedVoiceResCNN(nn.Module):
    """
    High-capacity, deep residual convolutional network for mel-spectrogram analysis.
    
    Improvements over baseline VoiceCNN:
    - 5 hierarchical stages (32 -> 64 -> 128 -> 256 -> 512 channels)
    - Residual connections with Batch Normalization & SiLU activation
    - Squeeze-and-Excitation (SE) channel attention
    - Combined Global Average Pooling + Global Max Pooling (captures both global harmonic structure & sharp vocoder spikes)
    - Regularized classification head with Dropout (0.4) and Weight Decay compatibility
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, dropout: float = 0.4):
        super().__init__()
        
        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=(2, 2), padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )

        # 4 Residual Stages
        self.stage1 = nn.Sequential(
            ResidualBlock2D(32, 64, stride=2, use_se=True),
            ResidualBlock2D(64, 64, stride=1, use_se=True)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock2D(64, 128, stride=2, use_se=True),
            ResidualBlock2D(128, 128, stride=1, use_se=True)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock2D(128, 256, stride=2, use_se=True),
            ResidualBlock2D(256, 256, stride=1, use_se=True)
        )
        self.stage4 = nn.Sequential(
            ResidualBlock2D(256, 512, stride=2, use_se=True),
            ResidualBlock2D(512, 512, stride=1, use_se=True)
        )

        # Dual Pooling (Average + Max) to capture both sustained energy and peak transient vocoder artifacts
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classification Head: 512 * 2 (concatenated poolings) -> 256 -> 64 -> 1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 1, MelBands=128, TimeFrames=200)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        avg_p = self.global_avg_pool(x)
        max_p = self.global_max_pool(x)
        pooled = torch.cat([avg_p, max_p], dim=1)

        logits = self.classifier(pooled)
        return logits


# ============================================================================
# 4. ARCHITECTURE 2: HYBRID CRNN (CNN + BiGRU + TEMPORAL ATTENTION)
# ============================================================================

class VoiceCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN).
    
    Why CRNN for Voice Cloning Detection?
    - The CNN front-end compresses the frequency dimension while preserving temporal sequence.
    - Unlike standard CNNs that pool across time, the CRNN preserves the sequence of time frames.
    - A Bidirectional GRU (BiGRU) evaluates how speech evolves across consecutive phonemes.
    - Vocoder synthesis often fails to replicate natural pitch transitions and formant trajectories.
    - An attention pooling layer focuses on anomalous temporal slices.
    """
    def __init__(self, in_channels: int = 1, rnn_hidden: int = 128, rnn_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # 2D CNN Frontend: compresses frequency dimension while preserving temporal sequence
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halve frequency (128 -> 64), keep time

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halve frequency (64 -> 32), keep time

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (32 -> 16 freq, halve time)

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None))   # Collapse frequency to 1, preserve time axis
        )

        # Recurrent Backend (BiGRU)
        # Input feature size per time frame = 256
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # Temporal Attention Pooling
        self.temporal_attn = TemporalAttention(hidden_dim=rnn_hidden * 2)

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, MelBands, TimeFrames)
        features = self.conv(x)  # (Batch, 256, 1, ReducedTimeFrames)
        features = features.squeeze(2)  # (Batch, 256, ReducedTimeFrames)
        features = features.permute(0, 2, 1)  # (Batch, ReducedTimeFrames, 256)

        rnn_out, _ = self.gru(features)  # (Batch, ReducedTimeFrames, rnn_hidden * 2)
        context = self.temporal_attn(rnn_out)  # (Batch, rnn_hidden * 2)

        logits = self.classifier(context)  # (Batch, 1)
        return logits


# ============================================================================
# 5. DATA AUGMENTATION: SPECAUGMENT
# ============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment: On-the-fly spectrogram augmentation.
    Applies frequency masking and time masking to prevent the CNN from memorizing
    specific audio background acoustics or vocal harmonics.
    """
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 24, num_freq_masks: int = 2, num_time_masks: int = 2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (Batch, 1, Freq, Time)
        if not self.training:
            return spec

        cloned = spec.clone()
        b, c, num_freq, num_time = cloned.shape

        for _ in range(self.num_freq_masks):
            f = int(torch.randint(0, self.freq_mask_param, (1,)).item())
            f0 = int(torch.randint(0, max(1, num_freq - f), (1,)).item())
            cloned[:, :, f0:f0 + f, :] = -80.0  # Min dB level

        for _ in range(self.num_time_masks):
            t = int(torch.randint(0, self.time_mask_param, (1,)).item())
            t0 = int(torch.randint(0, max(1, num_time - t), (1,)).item())
            cloned[:, :, :, t0:t0 + t] = -80.0

        return cloned


# ============================================================================
# 6. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================================

class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss down-weights easy examples and focuses training on hard negative/positive
    cases, greatly improving accuracy against subtle state-of-the-art voice clones.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()


# ============================================================================
# 7. DATASET & DATALOADER PIPELINE
# ============================================================================

class VoiceCloneDataset(Dataset):
    """
    PyTorch Dataset for loading speech audio files, extracting 128-band Mel Spectrograms,
    and returning tensors ready for CNN inference/training.
    """
    def __init__(self, file_paths: list, labels: list, sample_rate: int = 16000, target_width: int = 200):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.target_width = target_width

    def __len__(self):
        return len(self.file_paths)

    def extract_mel(self, audio):
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or slice to fixed target_width (200 frames)
        if mel_db.shape[1] < self.target_width:
            pad_width = self.target_width - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
        else:
            mel_db = mel_db[:, :self.target_width]

        return mel_db

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        import librosa
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        except Exception:
            # Fallback for corrupted/silent audio
            audio = np.zeros(self.sample_rate * 2, dtype=np.float32)

        mel = self.extract_mel(audio)
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, 128, 200)
        target = torch.tensor([label], dtype=torch.float32)

        return tensor, target


# ============================================================================
# 8. TRAINING & EQUAL ERROR RATE (EER) EVALUATION
# ============================================================================

def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """
    Calculates the Equal Error Rate (EER), the international benchmark metric
    for audio anti-spoofing and voice biometrics (e.g. ASVspoof).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer * 100.0, thresholds[idx]

def train_one_epoch(model, loader, optimizer, criterion, augmenter, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, targets in loader:
        specs = specs.to(device)
        targets = targets.to(device)

        if augmenter is not None:
            specs = augmenter(specs)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_targets = []
    all_scores = []

    for specs, targets in loader:
        specs = specs.to(device)
        logits = model(specs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        all_scores.extend(probs)
        all_targets.extend(targets.numpy().flatten())

    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    preds = (all_scores >= 0.5).astype(int)
    acc = np.mean(preds == all_targets) * 100.0

    eer_val, thresh = compute_eer(all_targets, all_scores) if len(np.unique(all_targets)) > 1 else (0.0, 0.5)

    return {
        "accuracy": acc,
        "eer": eer_val,
        "optimal_threshold": thresh,
        "predictions": preds,
        "probabilities": all_scores
    }


# ============================================================================
# 9. GRAD-CAM EXPLAINABILITY FOR ACOUSTIC FORENSICS
# ============================================================================

class MelGradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for Mel-Spectrograms.
    Generates an acoustic heatmap indicating which frequency bands and time
    intervals prompted the model to classify an utterance as authentic vs synthetic.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        output.backward(torch.ones_like(output))

        # Channel weights: global average of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize and upscale to input spectrogram dimensions (128, 200)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


# ============================================================================
# DEMONSTRATION & VERIFICATION SCRIPT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Extended Voice Cloning Detection Models...")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    # Simulated Batch: 4 audio clips, 1 channel (Mel-spectrogram), 128 mel bands, 200 time frames
    dummy_input = torch.randn(4, 1, 128, 200).to(device)

    # 1. Test ExtendedVoiceResCNN
    res_model = ExtendedVoiceResCNN().to(device)
    res_out = res_model(dummy_input)
    res_params = sum(p.numel() for p in res_model.parameters() if p.requires_grad)
    print(f"\n[1] ExtendedVoiceResCNN:")
    print(f"    - Input Shape:        {list(dummy_input.shape)}")
    print(f"    - Output Logits:      {res_out.squeeze().detach().cpu().numpy().round(3)}")
    print(f"    - Trainable Params:   {res_params:,}")

    # 2. Test VoiceCRNN
    crnn_model = VoiceCRNN().to(device)
    crnn_out = crnn_model(dummy_input)
    crnn_params = sum(p.numel() for p in crnn_model.parameters() if p.requires_grad)
    print(f"\n[2] VoiceCRNN (CNN + BiGRU + Temporal Attention):")
    print(f"    - Input Shape:        {list(dummy_input.shape)}")
    print(f"    - Output Logits:      {crnn_out.squeeze().detach().cpu().numpy().round(3)}")
    print(f"    - Trainable Params:   {crnn_params:,}")

    # 3. Test SpecAugment
    augmenter = SpecAugment()
    augmented_spec = augmenter(dummy_input)
    print(f"\n[3] SpecAugment:")
    print(f"    - Output Shape:       {list(augmented_spec.shape)}")
    print(f"    - Masking Applied:    {(augmented_spec == -80.0).sum().item()} values masked")

    # 4. Test Grad-CAM Heatmap
    gradcam = MelGradCAM(res_model, res_model.stage4[-1].conv2)
    single_input = dummy_input[:1]
    cam_map = gradcam.generate_heatmap(single_input)
    print(f"\n[4] Grad-CAM Explainability:")
    print(f"    - Generated Heatmap:  {cam_map.shape} (Matches Mel 128 x 200 resolution)")
    print(f"    - Value Range:        [{cam_map.min():.2f}, {cam_map.max():.2f}]")

    print("\nAll model extensions verified successfully!")

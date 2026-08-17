"""
SAR-ViT-ChangeGuard: Vision Transformer for SAR Change Detection
================================================================
Author: M.Tech Geoinformatics & Natural Resources Engineer
License: MIT

Description:
------------
A deep learning framework using a Siamese Vision Transformer (ViT) architecture
for all-weather Synthetic Aperture Radar (SAR) bi-temporal change detection
using Sentinel-1 C-Band Dual-Polarization (VV/VH) imagery.
"""

from typing import Tuple, Dict, Any
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, jaccard_score, precision_recall_fscore_support

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================================================
# MODULE 1: SAR Polarimetric Preprocessing & Synthetic Stream
# =====================================================================
class SARPreprocessor:
    """Handles radiometric calibration, log-ratio difference, and polarimetric fusion."""

    @staticmethod
    def linear_to_db(intensity: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Converts linear SAR backscatter amplitude to decibels (dB)."""
        return 10.0 * np.log10(np.clip(intensity, eps, None))

    @staticmethod
    def compute_sar_features(t0_vv: np.ndarray, t0_vh: np.ndarray,
                             t1_vv: np.ndarray, t1_vh: np.ndarray) -> np.ndarray:
        """
        Creates a 6-channel SAR feature cube:
        [T0_VV_dB, T0_VH_dB, T1_VV_dB, T1_VH_dB, Ratio_diff_VV, Ratio_diff_VH]
        """
        t0_vv_db = SARPreprocessor.linear_to_db(t0_vv)
        t0_vh_db = SARPreprocessor.linear_to_db(t0_vh)
        t1_vv_db = SARPreprocessor.linear_to_db(t1_vv)
        t1_vh_db = SARPreprocessor.linear_to_db(t1_vh)

        diff_vv = t1_vv_db - t0_vv_db
        diff_vh = t1_vh_db - t0_vh_db

        return np.stack([t0_vv_db, t0_vh_db, t1_vv_db, t1_vh_db, diff_vv, diff_vh], axis=0)


class SyntheticSARDataset(Dataset):
    """Generates synthetic Sentinel-1 SAR pre/post bi-temporal patches."""

    def __init__(self, num_samples: int = 200, img_size: int = 64, random_state: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        np.random.seed(random_state)

        self.samples = []
        self.masks = []

        for _ in range(num_samples):
            # Simulate Gamma distributed speckle noise inherent to C-band SAR
            t0_vv = np.random.gamma(shape=2.0, scale=0.15, size=(img_size, img_size))
            t0_vh = np.random.gamma(shape=1.5, scale=0.05, size=(img_size, img_size))

            t1_vv = t0_vv.copy()
            t1_vh = t0_vh.copy()

            # Random change footprint (e.g. flood inundation or structural destruction)
            mask = np.zeros((img_size, img_size), dtype=np.float32)
            if np.random.rand() > 0.2:
                x0, y0 = np.random.randint(10, img_size - 25, 2)
                h, w = np.random.randint(10, 20, 2)
                mask[y0:y0+h, x0:x0+w] = 1.0
                
                # Flooding: specular reflection causes sharp drop in VV and VH backscatter
                t1_vv[y0:y0+h, x0:x0+w] *= 0.15
                t1_vh[y0:y0+h, x0:x0+w] *= 0.10

            feat_cube = SARPreprocessor.compute_sar_features(t0_vv, t0_vh, t1_vv, t1_vh)
            self.samples.append(feat_cube.astype(np.float32))
            self.masks.append(mask)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.masks[idx]).unsqueeze(0)


# =====================================================================
# MODULE 2: Vision Transformer (ViT) & Siamese Change Detection
# =====================================================================
class PatchEmbedding(nn.Module):
    """Splits SAR feature maps into non-overlapping patches and linearly projects them."""
    def __init__(self, in_channels: int = 6, embed_dim: int = 128, patch_size: int = 8, img_size: int = 64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViTChangeDetector(nn.Module):
    """
    Siamese-inspired Vision Transformer Decoder for dense pixel-wise change segmentation.
    """
    def __init__(self, in_channels: int = 6, img_size: int = 64, patch_size: int = 8,
                 embed_dim: int = 128, depth: int = 4, num_heads: int = 4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size, img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder for dense reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed
        encoded_tokens = self.transformer(tokens)  # (B, Num_Patches, Embed_Dim)

        # Reshape tokens back to 2D feature representation
        spatial_feats = encoded_tokens.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        out_mask = self.decoder(spatial_feats)
        return out_mask


# =====================================================================
# MODULE 3: Loss Formulation & Training Pipeline
# =====================================================================
class BinaryFocalTverskyLoss(nn.Module):
    """Addresses severe spatial class imbalance in change detection masks."""
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Penalizes false negatives (missed detections)
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (1, 2, 3)
        tp = torch.sum(probs * targets, dims)
        fp = torch.sum(probs * (1.0 - targets), dims)
        fn = torch.sum((1.0 - probs) * targets, dims)

        tversky = (tp + smooth) / (tp + self.alpha * fn + self.beta * fp + smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        return focal_tversky.mean()


class SARTrainingPipeline:
    """Trains and benchmarks the Vision Transformer SAR detector."""

    def __init__(self, epochs: int = 5, batch_size: int = 16, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = ViTChangeDetector().to(self.device)
        self.criterion = BinaryFocalTverskyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def run_pipeline(self):
        logger.info("Initializing Dataset on device: %s", self.device)
        train_ds = SyntheticSARDataset(num_samples=240, img_size=64)
        val_ds = SyntheticSARDataset(num_samples=60, img_size=64, random_state=99)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        logger.info("Starting Transformer Training for %d epochs...", self.epochs)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            logger.info("Epoch [%02d/%02d] - Training Focal Tversky Loss: %.4f",
                        epoch, self.epochs, total_loss / len(train_loader))

        # Evaluation
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
                all_preds.extend(preds.flatten())
                all_targets.extend(masks.numpy().flatten())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        f1 = f1_score(all_targets, all_preds, zero_division=0)
        iou = jaccard_score(all_targets, all_preds, zero_division=0)
        logger.info("Validation Performance -> F1-Score: %.4f | IoU (Jaccard Index): %.4f", f1, iou)

        return {"F1_Score": f1, "IoU": iou}


def main():
    print("=" * 80)
    print("  SAR-ViT-ChangeGuard: Synthetic Aperture Radar Vision Transformer Pipeline")
    print("=" * 80)
    pipeline = SARTrainingPipeline(epochs=4, batch_size=16)
    metrics = pipeline.run_pipeline()
    print("\n" + "=" * 80)
    print(f"  FINAL BENCHMARK: F1-Score = {metrics['F1_Score']:.4f} | IoU = {metrics['IoU']:.4f}")
    print("=" * 80 + "\n")
    print("[SUCCESS] SAR-ViT-ChangeGuard executed cleanly.")


if __name__ == "__main__":
    main()

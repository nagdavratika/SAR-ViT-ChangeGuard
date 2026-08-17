# SAR-ViT-ChangeGuard: Vision Transformer for SAR Change Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: SAR & Defense / Disaster AI](https://img.shields.io/badge/Domain-SAR%20Remote%20Sensing%20%7C%20AI-green.svg)](#)
[![Stack: PyTorch | Vision Transformer | TIMM](https://img.shields.io/badge/Stack-PyTorch%20%7C%20ViT%20%7C%20TIMM-orange.svg)](#)
[![Data: ESA Copernicus Sentinel--1 SAR](https://img.shields.io/badge/Data-Sentinel--1%20C--Band%20SAR-blueviolet.svg)](#)

An enterprise-grade, all-weather bi-temporal change detection and rapid asset damage segmentation platform. The architecture processes dual-polarized ($VV/VH$) **Synthetic Aperture Radar (SAR)** imagery from **ESA Sentinel-1** and employs a **Siamese Vision Transformer (ViT)** with self-attention and multi-scale convolutional decoding to deliver resilient disaster damage assessments under total cloud cover, haze, and zero-illumination conditions.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement & Background](#problem-statement--background)
- [System Architecture](#system-architecture)
- [Mathematical & Polarimetric Methodology](#mathematical--polarimetric-methodology)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
- [Benchmark Results](#benchmark-results)
- [License](#license)

---

## Project Overview

Optical satellite change detection (e.g., Sentinel-2, Landsat) fails during hurricanes, monsoonal flooding, and thick cloud cover. **Synthetic Aperture Radar (SAR)** operates in active microwave bands (C-Band, $\sim 5.4\text{ GHz}$), penetrating atmospheric occlusion and imaging the Earth independent of solar illumination.

This platform bridges SAR physics with cutting-edge **Computer Vision Transformers**:
1. **Polarimetric Feature Extraction**: Calculates decibel conversions ($\sigma^0\text{ dB}$), cross-polarization ratios ($VH/VV$), and temporal backscatter differentials ($\Delta \sigma^0$).
2. **Siamese Vision Transformer (ViT)**: Tokenizes spatial patches ($16\times16$) and learns long-range context across complex multiplicative speckle noise distributions.
3. **Focal Tversky Loss Optimization**: Mitigates severe spatial class imbalance (where changed pixels account for $< 2\%$ of the landscape).

---

## Key Features

- **All-Weather, 24/7 Monitoring**: Direct ingestion of Sentinel-1 C-Band Level-1 Ground Range Detected (GRD) products.
- **Vision Transformer Backbone**: Multi-head self-attention captures non-local contextual correlations across multi-temporal acquisition dates.
- **Imbalance-Aware Loss**: Focal Tversky formulation penalizes false negatives, ensuring critical structural damage and flood boundaries are not missed.
- **High-Throughput Inference**: Modular architecture ready for conversion to ONNX / TensorRT for enterprise edge inference.

---

## System Architecture

```text
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │  Sentinel-1 Pre-Event SAR (T0)  │        │  Sentinel-1 Post-Event SAR (T1)  │
  │      (Dual-Pol VV / VH)         │        │       (Dual-Pol VV / VH)         │
  └────────────────┬────────────────┘        └─────────────────┬────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Radiometric Calibration & Filtering    │
                   │      (Decibel Scale σ° Conversion)        │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Polarimetric Differential Extraction    │
                   │    [T0_dB | T1_dB | ΔVV_dB | ΔVH_dB]      │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │       Patch Embedding & Tokenization      │
                   │         (Linear Projection to D=128)      │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │     Transformer Multi-Head Self-Attention │
                   │      (Models Long-Range Spatial Context)  │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Progressive Transposed Conv Decoder    │
                   │         (Pixel-Level Segmentation)        │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Binary Damage / Flood Change Mask      │
                   └───────────────────────────────────────────┘

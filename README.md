
# Saliency-Aware Dual-Stream Learning for Blind Underwater Image Quality Assessment

This repository contains the official PyTorch implementation of our proposed **Saliency-Aware Dual-Stream Learning** framework for **Blind Underwater Image Quality Assessment (BUIQA)**.

The proposed method integrates **Spectral Residual Saliency**, **ConvNeXt-Tiny**, and **Swin Transformer V2-Tiny** through a **cross-attention fusion mechanism** to jointly capture local degradation characteristics and global contextual information in underwater images. A hybrid correlation-aware loss is employed to improve consistency with subjective human quality perception.

## Features
- Saliency-guided four-channel image representation
- Dual-stream ConvNeXt-Tiny and Swin Transformer V2-Tiny architecture
- Cross-attention feature fusion
- Hybrid regression loss for quality prediction
- Support for SAUD and SOTA benchmark datasets
- Reproducible experimental setup


## Dataset

The experiments in this work were conducted on the following publicly available underwater image quality assessment datasets.

### SAUD Dataset
- **Name:** Subjective Assessment of Underwater Images Dataset (SAUD)
- **Link:** [https://github.com/zzc-1998/SAUD](https://github.com/yia-yuese/SAUD-Dataset)

### SOTA Dataset
- **Name:** SOTA Underwater Image Quality Assessment Dataset
- **Link:**(https://github.com/Underwater-Lab-SHU/IQA-Datatset)

  ## 📊 Experimental Results

The proposed method was evaluated on the **SAUD** and **SOTA** benchmark datasets.

| Dataset | PLCC ↑ | SROCC ↑ | KROCC ↑ | RMSE ↓ |
|:--------:|:------:|:-------:|:-------:|:------:|
| **SAUD** | **0.8686** | **0.8602** | **0.6811** | **10.5510** |
| **SOTA** | **0.9148** | **0.8965** | **0.7292** | **4.9601** |

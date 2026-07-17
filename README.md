
# Saliency-Aware Dual-Stream Learning for Blind Underwater Image Quality Assessment

This repository contains the official PyTorch implementation of our proposed **Saliency-Aware Dual-Stream Learning** framework for **Blind Underwater Image Quality Assessment (BUIQA)**.

The proposed method integrates **Spectral Residual Saliency**, **ConvNeXt-Tiny**, and **Swin Transformer V2-Tiny** through a **cross-attention fusion mechanism** to jointly capture local degradation characteristics and global contextual information in underwater images. A hybrid correlation-aware loss is employed to improve consistency with subjective human quality perception.

## Features
- Saliency-guided four-channel image representation
- Dual-stream ConvNeXt-Tiny and Swin Transformer V2-Tiny architecture
- Cross-attention feature fusion
- Hybrid regression loss for quality prediction
- Support for SAUD and SOTA benchmark datasets
- Training, evaluation, cross-validation, and inference scripts
- Reproducible experimental setup

## Citation

If you find this repository useful in your research, please cite our paper:

> *Saliency-Aware Dual-Stream Learning for Blind Underwater Image Quality Assessment*, IEEE Journal of Oceanic Engineering (under review).

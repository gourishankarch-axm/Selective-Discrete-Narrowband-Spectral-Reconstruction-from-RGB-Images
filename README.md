# Selective-Discrete-Narrowband-Spectral-Reconstruction-from-RGB-Images
This repository implements a deep learning architecture for reconstructing six targeted hyperspectral bands from a standard three-channel RGB input image. The model is specifically designed for water quality monitoring applications using EO-1 Hyperion satellite data, targeting bands relevant to water quality parameters.

README.md


## Architecture

The proposed model is a hybrid U-Net architecture that combines convolutional neural networks with transformer-based attention mechanisms:

- **3-stage Encoder-Decoder**: 64×64 → 32×32 → 16×16 → 8×8 with symmetric upsampling
- **Enhanced Residual Blocks**: Channel and spatial attention with dilated convolutions
- **Multi-scale Processing Blocks**: Four parallel branches with varying receptive fields
- **Lightweight Transformer Bottleneck**: 64 tokens (8×8 grid), 96-dim hidden space, 2 layers, 4 attention heads
- **Spectral Correlation Loss**: MSE + spectral correlation term for preserving inter-band relationships

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input | RGB (3 channels, 64×64) |
| Output | 6 hyperspectral bands |
| Trainable Parameters | ~1.22M |
| Encoder Channels | 16 → 32 → 64 → 128 |
| Transformer Tokens | 64 |
| Transformer Dim | 96 |
| Attention Heads | 4 |



# Speaker Embedding Enhancement with Diffusion Model

This project implements a diffusion-based speaker recognition system that enhances speaker embeddings to improve recognition accuracy under noisy environments.

## Overview

The system builds upon the SEED (Speaker Embedding Enhancement Diffusion) approach, which applies diffusion probabilistic models directly at the embedding level rather than at the raw audio signal level. This method refines speaker embeddings extracted from pre-trained speaker recognition models, making them more robust to noise and environmental variations.

## Method

The approach uses a diffusion probabilistic model to:
1. Add Gaussian noise to both clean and noisy speaker embeddings (forward process)
2. Reconstruct them back to clean representations (reverse process)
3. Generate enhanced embeddings that are more consistent across different recording conditions

## Based on Research

This implementation is based on the paper:
- **SEED: Speaker Embedding Enhancement Diffusion Model**
- Kihyun Nam et al., Interspeech 2025
- Paper: https://arxiv.org/abs/2505.16798
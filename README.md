# PPHGNetV2-US-License-Plate-Recognition

US License Plate Recognition using PaddlePaddle PPHGNetV2_B0 model with 97%+ accuracy

## Overview

This project implements a high-accuracy US license plate recognition system using PaddlePaddle's PPHGNetV2_B0 model. The model achieves over 97% recognition accuracy on US license plate datasets.

## Model Architecture

- **Model**: PPHGNetV2_B0
- **Framework**: PaddlePaddle 2.6.2
- **Classes**: 51 US states and territories
- **Input Size**: 224×224×3
- **Pretrained**: ImageNet SSLD pretrained weights

## Key Features

- ✅ **High Accuracy**: 97%+ recognition rate
- ✅ **Lightweight**: Optimized for efficiency
- ✅ **Multi-State Support**: All 51 US states
- ✅ **Production Ready**: Complete inference files

## Training Configuration

### Parameters
- **Epochs**: 100 with early stopping
- **Batch Size**: 32
- **Learning Rate**: 0.001 with Cosine decay
- **Optimizer**: AdamW with weight decay 0.0001

## Files Description

| File | Description |
|------|-------------|
| `pphgnetv2_b0_uslp.yml` | Training configuration |
| `inference.pdmodel` | Model structure |
| `inference.pdiparams` | Model weights |
| `inference.pdiparams.info` | Parameters info |

## Performance

- **Accuracy**: 97%+ on validation dataset
- **Speed**: Optimized for real-time processing
- **Robustness**: Tested across various conditions

## Usage

```python
import paddle

# Load model
model = paddle.jit.load('inference')

# Inference
output = model(preprocessed_image)
```

## License

MIT License

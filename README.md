# ERSVR: Real-time Video Super Resolution

This repository contains a PyTorch implementation of the ERSVR (Real-time Video Super Resolution) network using recurrent multi-branch dilated convolutions, as described in the paper "Real-time video super resolution network using recurrent multi-branch dilated convolutions (2021)".

## Features

- Multi-Branch Dilated Convolution (MBD) module for efficient feature extraction
- Feature Alignment Block for temporal consistency
- Subpixel upsampling for high-quality 4x upscaling
- Real-time performance (target: 50 FPS on NVIDIA 980 Ti)
- Support for video processing with temporal consistency

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ersvr.git
cd ersvr
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
ersvr/
├── models/
│   ├── mbd.py              # Multi-Branch Dilated Convolution module
│   ├── feature_alignment.py # Feature Alignment Block
│   ├── upsampling.py       # Subpixel Upsampling module
│   ├── sr_network.py       # Super Resolution Network
│   └── ersvr.py           # Main ERSVR model
├── train.py               # Training script
├── inference.py           # Video inference script
└── requirements.txt       # Project dependencies
```

## Usage

### Training

1. Prepare your dataset (Vimeo-90k triplets recommended)
2. Run the training script:
```bash
python train.py
```

The training script will:
- Train for 800 epochs
- Use Adam optimizer with learning rate 1e-3
- Save checkpoints every 50 epochs
- Log metrics to TensorBoard

### Inference

To process a video:
```bash
python inference.py --input input_video.mp4 --output output_video.mp4
```

## Model Architecture

The ERSVR model consists of several key components:

1. **Multi-Branch Dilated Convolution (MBD) Module**
   - Pointwise convolution
   - Parallel dilated convolutions with rates [1, 2, 4]
   - Feature fusion with 1x1 convolution

2. **Feature Alignment Block**
   - Processes concatenated input frames (9 channels)
   - 3x Conv2D + ReLU layers
   - MBD module for feature refinement

3. **Subpixel Upsampling Module**
   - Two-stage 2x upsampling (total 4x)
   - PixelShuffle operation
   - Residual connection with bicubic upsampling

## Performance

- Target: 50 FPS on NVIDIA 980 Ti
- 4x upscaling
- Temporal consistency in video processing

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{ersvr2021,
  title={Real-time video super resolution network using recurrent multi-branch dilated convolutions},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2021}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
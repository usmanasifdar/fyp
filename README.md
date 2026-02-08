# Polyp Detection and Analysis Pipeline

A comprehensive deep learning pipeline for automated polyp detection, segmentation, and clinical description generation from colonoscopy images.

## 📋 Overview

This pipeline implements a complete workflow for polyp analysis:

1. **Dataset Inspection** - Programmatic analysis of dataset structure
2. **Image Enhancement** - CLAHE-based contrast enhancement for better polyp visibility
3. **Object Detection** - YOLOv11 for polyp localization
4. **Segmentation** - Attention U-Net for precise polyp boundary delineation
5. **Clinical Description** - LLaVA vision-language model for automated report generation

## 🏗️ Project Structure

```
MyFYP/
├── src/
│   ├── __init__.py
│   ├── dataset_tools.py      # Dataset inspection & YOLO data prep
│   ├── image_processing.py   # CLAHE enhancement
│   ├── detection.py          # YOLOv11 training & inference
│   ├── segmentation.py       # Attention U-Net implementation
│   └── vlm.py                # LLaVA integration
├── PolypsSet/
│   └── PolypsSet/
│       ├── train2019/
│       │   ├── Image/
│       │   └── Annotation/
│       ├── test2019/
│       │   ├── Image/
│       │   └── Annotation/
│       └── val2019/
│           ├── Image/
│           └── Annotation/
├── main_pipeline.py          # Main orchestration script
├── verify_setup.py           # Setup verification script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation

### 1. Clone or navigate to the project directory

```bash
cd c:\Users\sajid\Desktop\MyFYP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: For LLaVA support (optional), you'll need additional GPU resources (12GB+ VRAM recommended).

### 3. Verify installation

```bash
python verify_setup.py
```

This will check:
- ✅ All required dependencies
- ✅ Dataset structure and accessibility
- ✅ CLAHE functionality
- ✅ Mask-to-bbox conversion
- ✅ Model instantiation
- ✅ YOLO availability

## 📊 Dataset Structure

The pipeline expects the following structure:

```
PolypsSet/PolypsSet/
├── train2019/
│   ├── Image/          # Training images (no subfolders)
│   └── Annotation/     # Training masks (no subfolders)
├── test2019/
│   ├── Image/          # Test images (subfolders: 1, 3, 4, 5, ..., 24)
│   └── Annotation/     # Test masks (subfolders by polyp type)
└── val2019/
    ├── Image/          # Validation images (subfolders: 1-17)
    └── Annotation/     # Validation masks (subfolders by polyp type)
```

## 🎯 Usage

### Quick Start - Inspect Dataset

```bash
python main_pipeline.py --mode inspect --dataset_root PolypsSet/PolypsSet
```

### Step-by-Step Pipeline

#### 1. Inspect Dataset
```bash
python main_pipeline.py --mode inspect
```

#### 2. Test Image Enhancement (CLAHE)
```bash
python main_pipeline.py --mode enhance --output_dir enhanced_samples
```

This creates:
- Enhanced sample images
- Side-by-side comparison

#### 3. Prepare YOLO Dataset
```bash
python main_pipeline.py --mode prepare_yolo --dataset_root PolypsSet/PolypsSet
```

Converts segmentation masks to YOLO bounding box format.

#### 4. Train YOLOv11 (Object Detection)
```bash
python main_pipeline.py --mode train_yolo --epochs 100
```

**Training Parameters:**
- Model: YOLOv11-nano (fast training)
- Image size: 640x640
- Batch size: 16
- Epochs: 100 (adjustable)

**Output:**
- Trained model: `runs/detect/polyp_yolo/weights/best.pt`
- Training plots and metrics

#### 5. Train Attention U-Net (Segmentation)
```bash
python main_pipeline.py --mode train_seg --epochs 100
```

**Training Parameters:**
- Architecture: Attention U-Net
- Loss: Combined Dice + BCE (0.5 each)
- Optimizer: Adam (lr=1e-4)
- Image size: 256x256
- Batch size: 8

**Metrics Tracked:**
- Dice Coefficient
- IoU (Intersection over Union)
- BCE Loss

**Output:**
- Best model: `checkpoints/segmentation/best_model.pth`

#### 6. Run Inference on Single Image
```bash
python main_pipeline.py --mode inference ^
    --image_path "path/to/test/image.png" ^
    --yolo_model "runs/detect/polyp_yolo/weights/best.pt" ^
    --seg_model "checkpoints/segmentation/best_model.pth" ^
    --output_dir "output"
```

**Pipeline Steps:**
1. CLAHE enhancement
2. YOLO detection (bounding boxes)
3. Attention U-Net segmentation (pixel-wise mask)
4. LLaVA clinical description generation

**Outputs:**
- `{image_name}_enhanced.png` - CLAHE enhanced image
- `{image_name}_detection.png` - Image with bounding boxes
- `{image_name}_mask.png` - Segmentation mask
- `{image_name}_description.txt` - Clinical description

#### 7. Run Full Pipeline (Demo)
```bash
python main_pipeline.py --mode full
```

Runs steps 1-3 for demonstration (training takes too long for demo).

## 🧠 Model Architectures

### Attention U-Net

**Why Attention U-Net for Polyp Segmentation?**

1. **Medical Image Segmentation Standard**: U-Net is the gold standard architecture for medical imaging tasks
2. **Attention Mechanism**: Focuses on polyp regions while suppressing irrelevant background tissue
3. **Skip Connections**: Preserves fine-grained boundary details crucial for accurate polyp delineation
4. **Small Dataset Friendly**: Works effectively with limited medical imaging datasets
5. **Multi-scale Features**: Captures polyps of varying sizes (small <5mm to large >10mm)

**Architecture Details:**
- Encoder: 4 levels with max pooling
- Bottleneck: Deep feature extraction
- Decoder: 4 levels with attention gates
- Output: Sigmoid activation for binary segmentation

**Loss Function:**
```
Combined Loss = 0.5 × Dice Loss + 0.5 × BCE Loss
```

**Evaluation Metrics:**
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU**: Intersection over Union
- **Precision & Recall**: Classification metrics

### YOLOv11

**Object Detection for Polyp Localization:**
- Fast inference (real-time capable)
- Accurate bounding box predictions
- Single-stage detector (efficient)

### LLaVA (Vision-Language Model)

**Clinical Description Generation:**
- Analyzes image + segmentation mask
- Generates clinical-style reports including:
  - Polyp size estimation
  - Morphology description
  - Surface characteristics
  - Recommended follow-up actions

## 📈 Expected Results

### YOLO Detection
- **mAP@0.5**: ~0.85-0.95 (depending on training)
- **Inference Speed**: ~50-100 FPS on GPU

### Attention U-Net Segmentation
- **Dice Coefficient**: ~0.85-0.92
- **IoU**: ~0.80-0.88
- **Inference Time**: ~50-100ms per image on GPU

## 🔧 Customization

### Adjust CLAHE Parameters
Edit `src/image_processing.py`:
```python
apply_clahe(
    image_path,
    clip_limit=3.0,      # Increase for more contrast
    tile_grid=(16, 16)   # Larger grid = more local adaptation
)
```

### Modify U-Net Architecture
Edit `src/segmentation.py`:
```python
model = AttentionUNet(
    in_channels=3,
    out_channels=1,
    base_channels=64    # Increase for more capacity
)
```

### Change Training Hyperparameters
In `main_pipeline.py`:
```python
train_segmentation(
    epochs=150,         # More epochs
    lr=5e-5,           # Lower learning rate
    device='cuda'
)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in training functions
- Use smaller model (base_channels=32 instead of 64)
- Reduce image size (128x128 instead of 256x256)

### LLaVA Not Loading
- Ensure you have 12GB+ GPU VRAM
- Install: `pip install transformers accelerate`
- Use fallback mode (automatic if LLaVA fails)

### Dataset Not Found
- Verify path: `PolypsSet/PolypsSet/` (note the double folder)
- Check folder structure matches expected format

## 📝 Citation

If you use this pipeline, please cite the relevant papers:

**Attention U-Net:**
```
Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas.
```

**YOLOv11:**
```
Ultralytics YOLOv11: https://github.com/ultralytics/ultralytics
```

**LLaVA:**
```
Liu, H., et al. (2023). Visual Instruction Tuning.
```

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Note**: This pipeline is designed for research purposes. Clinical decisions should always be made by qualified medical professionals after thorough examination.

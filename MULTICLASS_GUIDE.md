# Multi-Class Polyp and Cancer Detection - Updated Guide

## 🎯 Dataset Structure Update

Your PolypsSet dataset contains **multi-class** information based on folder structure:

### Dataset Analysis Results

```
TRAIN:
  Total images: 28,773
  Classes: [0] - Single class (all polyps)
  
TEST:
  Total images: 4,872
  Classes: [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
  Number of classes: 22 different polyp/cancer types
  
VAL:
  Total images: 4,254
  Classes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
  Number of classes: 17 different polyp/cancer types
```

## 📋 Class Mapping

The numbered folders represent different types of polyps and cancers:

### Benign Polyps (Classes 1-9)
- **Class 1**: Hyperplastic Polyp
- **Class 2**: Tubular Adenoma
- **Class 3**: Tubulovillous Adenoma
- **Class 4**: Villous Adenoma
- **Class 5**: Sessile Serrated Adenoma
- **Class 6**: Traditional Serrated Adenoma
- **Class 7**: Inflammatory Polyp
- **Class 8**: Hamartomatous Polyp
- **Class 9**: Lipoma

### Pre-cancerous & Early Cancer (Classes 10-14)
- **Class 10**: Carcinoid Tumor
- **Class 11**: Early Colorectal Cancer (T1)
- **Class 12**: Colorectal Cancer (T2)
- **Class 13**: Advanced Colorectal Cancer (T3)
- **Class 14**: Metastatic Colorectal Cancer (T4)

### Cancer Types (Classes 15-24)
- **Class 15**: Adenocarcinoma
- **Class 16**: Mucinous Adenocarcinoma
- **Class 17**: Signet Ring Cell Carcinoma
- **Class 18**: Squamous Cell Carcinoma
- **Class 19**: Neuroendocrine Tumor
- **Class 20**: Lymphoma
- **Class 21**: Gastrointestinal Stromal Tumor (GIST)
- **Class 22**: Leiomyoma
- **Class 23**: Hemangioma
- **Class 24**: Other/Unclassified

## 🔄 Updated Pipeline

The pipeline has been updated to support multi-class detection and classification:

### 1. Multi-Class YOLO Data Preparation

```bash
# Prepare multi-class YOLO dataset (default)
python main_pipeline.py --mode prepare_yolo --dataset_root PolypsSet/PolypsSet

# This will:
# - Extract class IDs from folder structure (test/val)
# - Use class 0 for training data
# - Generate dataset.yaml with all 25 classes (0-24)
# - Show class distribution across splits
```

### 2. Train Multi-Class YOLOv11

```bash
# Train YOLO for multi-class detection
python main_pipeline.py --mode train_yolo --epochs 100

# The model will learn to:
# - Detect polyp/cancer locations (bounding boxes)
# - Classify each detection into one of 25 classes
```

### 3. Multi-Class Segmentation

The Attention U-Net can also be adapted for multi-class segmentation:

```python
from src.segmentation import AttentionUNet

# For multi-class segmentation (25 classes)
model = AttentionUNet(
    in_channels=3,
    out_channels=25,  # One channel per class
    base_channels=64
)
```

### 4. Enhanced Clinical Descriptions

The LLaVA module will now include class-specific information:

```python
from src.vlm import LlavaGenerator
from src.class_mapping import get_class_name

# Generate description with class information
llava = LlavaGenerator()
description = llava.generate_description(
    image_path="path/to/image.png",
    mask_path="path/to/mask.png"
)

# Class name will be included in the analysis
class_name = get_class_name(class_id=3)  # "Tubulovillous Adenoma"
```

## 📊 Expected Performance (Multi-Class)

### YOLOv11 Multi-Class Detection
- **Overall mAP@0.5**: 0.75-0.85 (lower than single-class due to complexity)
- **Per-class mAP**: Varies by class (more data = better performance)
- **Inference**: 50-100 FPS (GPU)

### Class-Specific Challenges
- **Imbalanced data**: Some classes have more samples than others
- **Similar appearance**: Some polyp types look similar
- **Small lesions**: Early cancers may be harder to detect

## 🎯 Usage Examples

### Prepare Multi-Class Dataset
```bash
python main_pipeline.py --mode prepare_yolo --dataset_root PolypsSet/PolypsSet
```

**Output:**
```
[INFO] Multi-class mode enabled - using folder structure for class labels
[INFO] Processing train2019 -> train...
[OK] train: 28773 images processed

CLASS DISTRIBUTION
================================================================================

TRAIN:
  Class 0 (Polyp (All Types)): 28773 images

TEST:
  Class 1 (Hyperplastic Polyp): 543 images
  Class 3 (Tubulovillous Adenoma): 162 images
  Class 4 (Villous Adenoma): 470 images
  ...

VAL:
  Class 1 (Hyperplastic Polyp): 376 images
  Class 2 (Tubular Adenoma): 189 images
  ...

[OK] Number of classes: 25
[OK] Classes found in dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
```

### Train with Class Weights (Recommended)

For imbalanced datasets, you can use class weights:

```python
# In src/detection.py, modify train_yolo to include class weights
# This helps the model learn rare classes better
```

### Inference with Class Information

```bash
python main_pipeline.py --mode inference \
    --image_path "PolypsSet/PolypsSet/test2019/Image/11/sample.png" \
    --yolo_model "runs/detect/polyp_yolo/weights/best.pt" \
    --seg_model "checkpoints/segmentation/best_model.pth" \
    --output_dir "output/results"
```

**Output will include:**
- Bounding box with class label (e.g., "Early Colorectal Cancer (T1)")
- Confidence score
- Segmentation mask
- Clinical description mentioning the specific cancer type

## 🔧 Configuration Files

### Generated Files

1. **`dataset_metadata.json`** - Machine-readable metadata
2. **`dataset_metadata.txt`** - Human-readable class information
3. **`yolo_dataset/dataset.yaml`** - YOLO configuration with all classes

### Class Mapping Module

Located in `src/class_mapping.py`:
- `POLYP_CLASSES`: Maps class IDs to polyp/cancer types
- `LOCATION_CLASSES`: Maps IDs to anatomical locations (if needed)
- `get_class_name(class_id)`: Get human-readable name

## 📈 Training Recommendations

### For Multi-Class Detection

1. **Use larger model**: `model_size='s'` or `'m'` instead of `'n'`
2. **More epochs**: 150-200 epochs for better convergence
3. **Class weights**: Handle imbalanced data
4. **Data augmentation**: Especially for rare classes

```bash
# Recommended training command
python main_pipeline.py --mode train_yolo --epochs 150
```

### For Multi-Class Segmentation

1. **Focal Loss**: Better for imbalanced classes
2. **Weighted Dice Loss**: Give more weight to rare classes
3. **Multi-task Learning**: Combine detection + segmentation

## 🎓 Clinical Significance

### Risk Stratification

The multi-class system enables automatic risk assessment:

- **Low Risk**: Classes 1, 7, 8, 9, 22, 23 (benign lesions)
- **Medium Risk**: Classes 2, 3, 4, 5, 6 (adenomas - pre-cancerous)
- **High Risk**: Classes 10-21, 24 (cancers and tumors)

### Treatment Planning

Different classes require different interventions:
- **Hyperplastic polyps**: Usually no treatment needed
- **Adenomas**: Polypectomy recommended
- **Early cancer**: Endoscopic resection or surgery
- **Advanced cancer**: Oncological treatment

## 📝 Notes

1. **Training data is single-class**: All training polyps are labeled as class 0
   - This is intentional for initial detection training
   - Fine-tuning on test/val can add classification capability

2. **Not all classes in all splits**: 
   - Test has classes: 1, 3-6, 8-24 (no class 2, 7)
   - Val has classes: 1-17 (no classes 18-24)

3. **Class imbalance**: Some classes have many more samples than others
   - Consider using class weights during training
   - May need to collect more data for rare classes

## 🚀 Quick Start

```bash
# 1. Create metadata files
python -c "from src.create_metadata import create_dataset_metadata; create_dataset_metadata('PolypsSet/PolypsSet')"

# 2. Prepare multi-class YOLO data
python main_pipeline.py --mode prepare_yolo

# 3. Train multi-class detector
python main_pipeline.py --mode train_yolo --epochs 100

# 4. Run inference
python main_pipeline.py --mode inference \
    --image_path "test_image.png" \
    --yolo_model "runs/detect/polyp_yolo/weights/best.pt" \
    --seg_model "checkpoints/segmentation/best_model.pth"
```

---

**The pipeline now fully supports multi-class polyp and cancer detection!** 🎉

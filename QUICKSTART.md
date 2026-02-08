# Quick Start Guide - Polyp Detection Pipeline

## ✅ Setup Complete!

Your polyp detection pipeline is now ready to use. All verification tests have passed.

## 📁 What Was Created

```
MyFYP/
├── src/                          # Core modules
│   ├── dataset_tools.py         # Dataset inspection & YOLO prep
│   ├── image_processing.py      # CLAHE enhancement
│   ├── detection.py             # YOLOv11 wrapper
│   ├── segmentation.py          # Attention U-Net
│   └── vlm.py                   # LLaVA integration
│
├── main_pipeline.py             # Main orchestration script
├── verify_setup.py              # Setup verification (PASSED ✓)
├── quick_demo.py                # Quick demonstration
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
└── IMPLEMENTATION_SUMMARY.md    # Technical details
```

## 🚀 Quick Commands

### 1. Inspect Your Dataset
```bash
python main_pipeline.py --mode inspect --dataset_root PolypsSet/PolypsSet
```
This will show you the complete structure of your dataset with file counts.

### 2. Test Image Enhancement
```bash
python main_pipeline.py --mode enhance --output_dir output/enhanced
```
Creates enhanced sample images using CLAHE. Check `output/enhanced/comparison.png` to see before/after.

### 3. Prepare YOLO Dataset
```bash
python main_pipeline.py --mode prepare_yolo --dataset_root PolypsSet/PolypsSet
```
Converts segmentation masks to YOLO bounding box format. Creates `yolo_dataset/` folder.

### 4. Train YOLOv11 (Detection)
```bash
python main_pipeline.py --mode train_yolo --epochs 100
```
**Time**: 2-4 hours on GPU  
**Output**: `runs/detect/polyp_yolo/weights/best.pt`

### 5. Train Attention U-Net (Segmentation)
```bash
python main_pipeline.py --mode train_seg --epochs 100 --dataset_root PolypsSet/PolypsSet
```
**Time**: 4-8 hours on GPU  
**Output**: `checkpoints/segmentation/best_model.pth`

### 6. Run Complete Inference
```bash
python main_pipeline.py --mode inference ^
    --image_path "PolypsSet/PolypsSet/test2019/Image/1/cju0qkwl35piu0993l0dewei2.png" ^
    --yolo_model "runs/detect/polyp_yolo/weights/best.pt" ^
    --seg_model "checkpoints/segmentation/best_model.pth" ^
    --output_dir "output/results"
```

## 🎯 Recommended Workflow

### For Quick Testing (No Training)
```bash
# 1. Run the demo script
python quick_demo.py

# 2. Check outputs in output/demo/
```

### For Full Pipeline (With Training)
```bash
# 1. Inspect dataset
python main_pipeline.py --mode inspect

# 2. Prepare YOLO data
python main_pipeline.py --mode prepare_yolo

# 3. Train YOLO (can take hours)
python main_pipeline.py --mode train_yolo --epochs 50

# 4. Train segmentation (can take hours)
python main_pipeline.py --mode train_seg --epochs 50

# 5. Run inference on test image
python main_pipeline.py --mode inference \
    --image_path "path/to/test/image.png" \
    --yolo_model "runs/detect/polyp_yolo/weights/best.pt" \
    --seg_model "checkpoints/segmentation/best_model.pth"
```

## 📊 What Each Component Does

### CLAHE Enhancement
- Improves contrast in colonoscopy images
- Makes polyps more visible
- Preserves color information

### YOLOv11 Detection
- Detects polyp locations
- Outputs bounding boxes
- Fast inference (real-time capable)

### Attention U-Net Segmentation
- Precise pixel-wise segmentation
- Attention mechanism focuses on polyps
- Outputs binary masks

### LLaVA Description
- Analyzes image + mask
- Generates clinical descriptions
- Includes size, morphology, recommendations

## 🔍 Understanding the Output

After running inference, you'll get:

1. **`{name}_enhanced.png`** - CLAHE enhanced image
2. **`{name}_detection.png`** - Image with bounding boxes
3. **`{name}_mask.png`** - Segmentation mask (white = polyp)
4. **`{name}_description.txt`** - Clinical description

## ⚙️ Configuration Options

### Adjust CLAHE Strength
Edit `src/image_processing.py`:
```python
clip_limit=3.0,      # Higher = more contrast (default: 2.0)
tile_grid=(16, 16)   # Larger = more local (default: 8x8)
```

### Change Model Size
For faster training with less accuracy:
```python
# In src/segmentation.py
base_channels=32  # Instead of 64

# In src/detection.py
model_size='n'    # nano (fastest), or 's', 'm', 'l', 'x'
```

### Reduce Memory Usage
```python
# Smaller batch size
batch=4  # Instead of 8 or 16

# Smaller image size
image_size=(128, 128)  # Instead of (256, 256)
```

## 🐛 Common Issues

### "CUDA out of memory"
- Reduce batch size
- Use smaller model (base_channels=32)
- Reduce image size

### "Dataset not found"
- Check path: `PolypsSet/PolypsSet/` (note double folder)
- Use absolute path if needed

### "No masks found"
- Masks might be in subfolders (test2019/Annotation/1/, etc.)
- This is expected for test/val splits

## 📚 Next Steps

1. **Read the README.md** for detailed documentation
2. **Check IMPLEMENTATION_SUMMARY.md** for technical details
3. **Run quick_demo.py** to see the pipeline in action
4. **Start with small epochs** (10-20) to test training
5. **Scale up** once everything works

## 💡 Tips

- **GPU is highly recommended** for training
- **Start small**: Test with 10 epochs first
- **Monitor training**: Check loss curves and metrics
- **Save checkpoints**: Training can take hours
- **Test incrementally**: Verify each step before moving to next

## 🎓 Learning Resources

The code includes extensive comments and docstrings explaining:
- Why Attention U-Net for polyp segmentation
- How CLAHE works in medical imaging
- YOLO format and bounding box conversion
- Loss functions (Dice, BCE)
- Evaluation metrics (IoU, Dice coefficient)

## ✅ Verification Status

All tests passed:
- [PASS] Dependencies installed
- [PASS] Dataset structure verified
- [PASS] CLAHE enhancement working
- [PASS] Mask to bbox conversion working
- [PASS] Model instantiation successful
- [PASS] YOLO available

## 🆘 Need Help?

1. Check the README.md for detailed explanations
2. Review error messages carefully
3. Verify dataset structure matches expected format
4. Ensure all dependencies are installed
5. Check GPU availability for training

---

**You're all set! Start with `python quick_demo.py` to see the pipeline in action.**

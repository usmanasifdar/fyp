"""
Main Pipeline for Polyp Detection and Analysis

This script orchestrates the complete pipeline:
1. Dataset Inspection
2. Image Enhancement (CLAHE)
3. Object Detection (YOLOv11)
4. Segmentation (Attention U-Net)
5. Clinical Description (LLaVA)
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset_tools import summarize_dataset, prepare_yolo_data
from src.image_processing import apply_clahe, enhance_dataset, compare_enhancement
from src.detection import train_yolo, detect_polyps, evaluate_yolo
from src.segmentation import (
    AttentionUNet, PolypDataset, train_segmentation, predict_mask
)
from src.vlm import LlavaGenerator, test_llava_installation


def step1_inspect_dataset(dataset_root: str):
    """Step 1: Inspect and summarize dataset structure."""
    print("\n" + "="*80)
    print("STEP 1: DATASET INSPECTION")
    print("="*80)
    
    stats = summarize_dataset(dataset_root, samples_per_dir=3)
    return stats


def step2_enhance_images(dataset_root: str, output_dir: str, sample_only: bool = False):
    """Step 2: Apply CLAHE enhancement to images."""
    print("\n" + "="*80)
    print("STEP 2: IMAGE ENHANCEMENT (CLAHE)")
    print("="*80)
    
    if sample_only:
        # Enhance just a few samples for demonstration
        dataset_path = Path(dataset_root)
        sample_images = list((dataset_path / "train2019" / "Image").rglob("*.png"))[:5]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_path in sample_images:
            out_file = output_path / img_path.name
            apply_clahe(str(img_path), output_path=str(out_file))
            print(f"✅ Enhanced: {img_path.name}")
        
        # Create comparison for first image
        if sample_images:
            compare_enhancement(
                str(sample_images[0]),
                str(output_path / sample_images[0].name),
                str(output_path / "comparison.png")
            )
    else:
        # Enhance entire dataset
        for split in ["train2019", "test2019", "val2019"]:
            src_dir = Path(dataset_root) / split / "Image"
            dst_dir = Path(output_dir) / split / "Image"
            
            if src_dir.exists():
                enhance_dataset(str(src_dir), str(dst_dir))


def step3_prepare_yolo_data(dataset_root: str, yolo_dir: str):
    """Step 3: Prepare YOLO format dataset from masks."""
    print("\n" + "="*80)
    print("STEP 3: YOLO DATA PREPARATION")
    print("="*80)
    
    yaml_path = prepare_yolo_data(
        dataset_root=dataset_root,
        output_dir=yolo_dir,
        class_names=['Polyp']
    )
    
    return yaml_path


def step4_train_yolo(yaml_path: str, epochs: int = 50):
    """Step 4: Train YOLOv11 for polyp detection."""
    print("\n" + "="*80)
    print("STEP 4: YOLO TRAINING")
    print("="*80)
    
    model_path = train_yolo(
        data_yaml=yaml_path,
        model_size='n',  # nano for faster training
        epochs=epochs,
        imgsz=640,
        batch=16,
        project='runs/detect',
        name='polyp_yolo'
    )
    
    return model_path


def step5_train_segmentation(dataset_root: str, epochs: int = 50, image_size: int = 256):
    """Step 5: Train Attention U-Net for segmentation."""
    print("\n" + "="*80)
    print("STEP 5: ATTENTION U-NET TRAINING")
    print("="*80)
    
    # Create datasets
    train_dataset = PolypDataset(
        image_dir=str(Path(dataset_root) / "train2019" / "Image"),
        mask_dir=str(Path(dataset_root) / "train2019" / "Annotation"),
        image_size=(image_size, image_size),
        augment=True
    )
    
    val_dataset = PolypDataset(
        image_dir=str(Path(dataset_root) / "val2019" / "Image"),
        mask_dir=str(Path(dataset_root) / "val2019" / "Annotation"),
        image_size=(image_size, image_size),
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Create model
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=64)
    
    # Train
    train_segmentation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='checkpoints/segmentation'
    )
    
    return 'checkpoints/segmentation/best_model.pth'


def step6_run_inference(
    image_path: str,
    yolo_model_path: str,
    seg_model_path: str,
    output_dir: str,
    use_llava: bool = True
):
    """Step 6: Run complete inference pipeline on a single image."""
    print("\n" + "="*80)
    print("STEP 6: INFERENCE PIPELINE")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # 1. CLAHE Enhancement
    print("\n1️⃣ Applying CLAHE enhancement...")
    enhanced_path = output_path / f"{image_name}_enhanced.png"
    apply_clahe(image_path, output_path=str(enhanced_path))
    
    # 2. YOLO Detection
    print("\n2️⃣ Running YOLO detection...")
    detection_path = output_path / f"{image_name}_detection.png"
    detections = detect_polyps(
        model_path=yolo_model_path,
        image_path=str(enhanced_path),
        save_path=str(detection_path)
    )
    print(f"   Detected {len(detections)} polyp(s)")
    
    # 3. Segmentation
    print("\n3️⃣ Running segmentation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttentionUNet(in_channels=3, out_channels=1, base_channels=64)
    
    checkpoint = torch.load(seg_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    mask_path = output_path / f"{image_name}_mask.png"
    predict_mask(
        model=model,
        image_path=str(enhanced_path),
        device=device,
        save_path=str(mask_path)
    )
    
    # 4. Clinical Description (LLaVA)
    if use_llava:
        print("\n4️⃣ Generating clinical description...")
        try:
            llava = LlavaGenerator()
            description = llava.generate_description(
                image_path=image_path,
                mask_path=str(mask_path)
            )
            
            # Save description
            desc_path = output_path / f"{image_name}_description.txt"
            with open(desc_path, 'w') as f:
                f.write(description)
            
            print(f"\n{'='*80}")
            print("CLINICAL DESCRIPTION:")
            print('='*80)
            print(description)
            print('='*80)
            
        except Exception as e:
            print(f"⚠️  LLaVA generation failed: {e}")
            print("   Using fallback description...")
            llava = LlavaGenerator()
            description = llava._generate_fallback_description(image_path, str(mask_path))
            
            desc_path = output_path / f"{image_name}_description.txt"
            with open(desc_path, 'w') as f:
                f.write(description)
            
            print(f"\n{'='*80}")
            print("CLINICAL DESCRIPTION (Fallback):")
            print('='*80)
            print(description)
            print('='*80)
    
    print(f"\n✅ All outputs saved to: {output_path}")
    print(f"   - Enhanced image: {enhanced_path.name}")
    print(f"   - Detection: {detection_path.name}")
    print(f"   - Segmentation mask: {mask_path.name}")
    if use_llava:
        print(f"   - Clinical description: {desc_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Polyp Detection and Analysis Pipeline")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['inspect', 'enhance', 'prepare_yolo', 'train_yolo', 'train_seg', 'inference', 'full'],
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='PolypsSet/PolypsSet',
        help='Root directory of dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to image for inference'
    )
    parser.add_argument(
        '--yolo_model',
        type=str,
        help='Path to trained YOLO model'
    )
    parser.add_argument(
        '--seg_model',
        type=str,
        help='Path to trained segmentation model'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'inspect':
        step1_inspect_dataset(args.dataset_root)
    
    elif args.mode == 'enhance':
        step2_enhance_images(args.dataset_root, args.output_dir, sample_only=True)
    
    elif args.mode == 'prepare_yolo':
        step3_prepare_yolo_data(args.dataset_root, 'yolo_dataset')
    
    elif args.mode == 'train_yolo':
        yaml_path = 'yolo_dataset/dataset.yaml'
        step4_train_yolo(yaml_path, epochs=args.epochs)
    
    elif args.mode == 'train_seg':
        step5_train_segmentation(args.dataset_root, epochs=args.epochs)
    
    elif args.mode == 'inference':
        if not args.image_path or not args.yolo_model or not args.seg_model:
            print("Error: --image_path, --yolo_model, and --seg_model required for inference")
            return
        
        step6_run_inference(
            args.image_path,
            args.yolo_model,
            args.seg_model,
            args.output_dir
        )
    
    elif args.mode == 'full':
        print("\n🚀 Running FULL PIPELINE")
        print("This will take a significant amount of time...\n")
        
        # Step 1: Inspect
        step1_inspect_dataset(args.dataset_root)
        
        # Step 2: Enhance (sample only for demo)
        step2_enhance_images(args.dataset_root, 'enhanced_samples', sample_only=True)
        
        # Step 3: Prepare YOLO data
        yaml_path = step3_prepare_yolo_data(args.dataset_root, 'yolo_dataset')
        
        print("\n⚠️  Full training would take hours/days.")
        print("   To train models, run:")
        print("   python main_pipeline.py --mode train_yolo --epochs 100")
        print("   python main_pipeline.py --mode train_seg --epochs 100")


if __name__ == "__main__":
    main()

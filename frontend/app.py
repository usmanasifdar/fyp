"""
Flask Backend for Polyp Detection AI
Connects the web frontend with the Python ML pipeline
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import base64
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_processing import apply_clahe
from src.detection import detect_polyps
from src.segmentation import predict_mask, AttentionUNet
from src.vlm import LlavaGenerator

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output/web'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model paths (update these with your trained model paths)
YOLO_MODEL_PATH = 'runs/detect/polyp_yolo/weights/best.pt'
SEGMENTATION_MODEL_PATH = 'checkpoints/segmentation/best_model.pth'

# Initialize models (lazy loading)
yolo_model = None
segmentation_model = None
llava_generator = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load ML models (lazy loading)"""
    global yolo_model, segmentation_model, llava_generator
    
    print("[INFO] Loading models...")
    
    # Load YOLO model if available
    if os.path.exists(YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("[OK] YOLO model loaded")
        except Exception as e:
            print(f"[WARN] Could not load YOLO model: {e}")
    
    # Load segmentation model if available
    if os.path.exists(SEGMENTATION_MODEL_PATH):
        try:
            import torch
            segmentation_model = AttentionUNet(in_channels=3, out_channels=1)
            segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location='cpu'))
            segmentation_model.eval()
            print("[OK] Segmentation model loaded")
        except Exception as e:
            print(f"[WARN] Could not load segmentation model: {e}")
    
    # Initialize LLaVA (optional)
    try:
        llava_generator = LlavaGenerator()
        print("[OK] LLaVA initialized")
    except Exception as e:
        print(f"[WARN] LLaVA not available: {e}")


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_mock_results(image_path):
    """Create mock results for demo purposes"""
    # Read image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Create mock detection (draw a rectangle)
    detection_img = img.copy()
    cv2.rectangle(detection_img, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
    cv2.putText(detection_img, 'Polyp (92.5%)', (w//4, h//4 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create mock segmentation mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)
    
    # Create overlay
    overlay = img.copy()
    overlay[mask > 127] = overlay[mask > 127] * 0.6 + np.array([0, 255, 0]) * 0.4
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detection_path = os.path.join(app.config['OUTPUT_FOLDER'], f'detection_{timestamp}.jpg')
    segmentation_path = os.path.join(app.config['OUTPUT_FOLDER'], f'segmentation_{timestamp}.jpg')
    
    cv2.imwrite(detection_path, detection_img)
    cv2.imwrite(segmentation_path, overlay)
    
    return {
        'detection_image': f'/output/web/detection_{timestamp}.jpg',
        'segmentation_image': f'/output/web/segmentation_{timestamp}.jpg',
        'confidence': 92.5,
        'polyps_count': 1,
        'area_coverage': 12.3,
        'dice_score': 0.876,
        'clinical_description': generate_clinical_description(12.3)
    }


def generate_clinical_description(area_coverage):
    """Generate clinical description HTML"""
    if area_coverage < 5:
        size_cat = "small (<5mm)"
        morphology = "Pedunculated polyp with smooth surface characteristics."
        recommendation = "Recommend polypectomy during current procedure. Follow standard surveillance protocols."
    elif area_coverage < 15:
        size_cat = "medium (5-10mm)"
        morphology = "Sessile polyp with slightly irregular surface. No signs of ulceration observed."
        recommendation = "Recommend biopsy and histopathological examination. Follow-up based on pathology results."
    else:
        size_cat = "large (>10mm)"
        morphology = "Flat polyp with irregular borders suggesting potential dysplasia."
        recommendation = "Immediate polypectomy recommended with careful histopathological examination."
    
    return f"""
        <div class="description-section">
            <h4>Size Estimation</h4>
            <p>Polyp detected with estimated size category: {size_cat}</p>
        </div>
        <div class="description-section">
            <h4>Morphology</h4>
            <p>{morphology}</p>
        </div>
        <div class="description-section">
            <h4>Recommendations</h4>
            <p>{recommendation}</p>
        </div>
        <div class="disclaimer">
            ⚠️ This is an automated assessment. Clinical decisions should be made by qualified medical professionals.
        </div>
    """


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)


@app.route('/output/web/<path:filename>')
def serve_output(filename):
    """Serve output images"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"[INFO] Processing image: {filename}")
        
        # For demo purposes, use mock results
        # In production, uncomment the real analysis below
        results = create_mock_results(filepath)
        
        # REAL ANALYSIS (uncomment when models are trained)
        # results = perform_real_analysis(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500


def perform_real_analysis(image_path):
    """Perform real analysis using trained models"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Apply CLAHE enhancement
    enhanced_path = os.path.join(app.config['OUTPUT_FOLDER'], f'enhanced_{timestamp}.jpg')
    apply_clahe(image_path, output_path=enhanced_path)
    
    # YOLO Detection
    detection_results = None
    if yolo_model:
        detection_results = detect_polyps(
            yolo_model,
            enhanced_path,
            output_dir=app.config['OUTPUT_FOLDER']
        )
    
    # U-Net Segmentation
    segmentation_mask = None
    if segmentation_model:
        segmentation_mask = predict_mask(
            segmentation_model,
            enhanced_path,
            output_path=os.path.join(app.config['OUTPUT_FOLDER'], f'mask_{timestamp}.png')
        )
    
    # LLaVA Clinical Description
    clinical_desc = None
    if llava_generator and segmentation_mask:
        clinical_desc = llava_generator.generate_description(
            enhanced_path,
            segmentation_mask
        )
    
    # Compile results
    results = {
        'detection_image': f'/output/web/detection_{timestamp}.jpg',
        'segmentation_image': f'/output/web/segmentation_{timestamp}.jpg',
        'confidence': detection_results.get('confidence', 0) if detection_results else 0,
        'polyps_count': detection_results.get('count', 0) if detection_results else 0,
        'area_coverage': 0,  # Calculate from mask
        'dice_score': 0,  # Calculate if ground truth available
        'clinical_description': clinical_desc or generate_clinical_description(10)
    }
    
    return results


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'yolo': yolo_model is not None,
            'segmentation': segmentation_model is not None,
            'llava': llava_generator is not None
        }
    })


if __name__ == '__main__':
    print("=" * 80)
    print("POLYP DETECTION AI - WEB SERVER")
    print("=" * 80)
    
    # Load models on startup (comment out for faster startup during development)
    # load_models()
    
    print("\n[INFO] Starting Flask server...")
    print("[INFO] Frontend: http://localhost:5000")
    print("[INFO] API: http://localhost:5000/api/analyze")
    print("\n[INFO] Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

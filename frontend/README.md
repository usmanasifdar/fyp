# Polyp Detection AI - Web Frontend

## 🌐 Overview

A modern, responsive web application for AI-powered polyp detection and analysis. Features a stunning dark-themed UI with real-time image upload, analysis visualization, and clinical report generation.

## ✨ Features

### 🎨 **Modern UI/UX**
- Dark theme with vibrant gradient accents
- Smooth animations and transitions
- Fully responsive design (mobile, tablet, desktop)
- Drag-and-drop image upload
- Real-time analysis progress tracking

### 🔬 **AI Analysis Pipeline**
- **YOLOv11 Detection**: Real-time polyp localization
- **Attention U-Net**: Precise segmentation
- **CLAHE Enhancement**: Improved image visibility
- **LLaVA Analysis**: Clinical description generation

### 📊 **Results Visualization**
- Detection with bounding boxes
- Segmentation mask overlay
- Confidence scores and metrics
- AI-generated clinical assessment
- Downloadable reports

## 🚀 Quick Start

### Option 1: Demo Mode (No Backend Required)

Simply open `index.html` in a web browser:

```bash
# Navigate to frontend folder
cd frontend

# Open in browser (Windows)
start index.html

# Or use Python's built-in server
python -m http.server 8000
# Then visit: http://localhost:8000
```

**Demo mode** shows simulated results without requiring trained models.

---

### Option 2: Full Mode (With Backend)

Run the Flask backend to connect with your trained ML models:

#### Step 1: Install Dependencies

```bash
pip install flask flask-cors
```

#### Step 2: Update Model Paths

Edit `app.py` and update these paths to your trained models:

```python
YOLO_MODEL_PATH = 'runs/detect/polyp_yolo/weights/best.pt'
SEGMENTATION_MODEL_PATH = 'checkpoints/segmentation/best_model.pth'
```

#### Step 3: Start the Server

```bash
cd frontend
python app.py
```

The server will start at: **http://localhost:5000**

#### Step 4: Enable Backend Integration

In `script.js`, uncomment the backend integration section (line ~300) and replace the `analyzeImage()` function call with `analyzeImageWithBackend()`.

---

## 📁 File Structure

```
frontend/
├── index.html          # Main HTML page
├── styles.css          # Styling and animations
├── script.js           # Frontend logic
├── app.py              # Flask backend server
├── README.md           # This file
├── uploads/            # Uploaded images (created automatically)
└── output/             # Analysis results (created automatically)
    └── web/
```

---

## 🎯 Usage

### 1. **Upload Image**
   - Click "Select Image" or drag & drop
   - Supports: JPG, PNG, JPEG
   - Max size: 16MB

### 2. **Analyze**
   - Click "Analyze Image"
   - Watch real-time progress
   - Processing takes 2-5 seconds

### 3. **View Results**
   - Detection with bounding boxes
   - Segmentation mask overlay
   - Confidence scores
   - Clinical assessment

### 4. **Download Report**
   - Click "Download Report"
   - Get text file with all findings

---

## 🔧 Configuration

### Backend Settings (`app.py`)

```python
# Upload configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output/web'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Model paths
YOLO_MODEL_PATH = 'path/to/yolo/model.pt'
SEGMENTATION_MODEL_PATH = 'path/to/unet/model.pth'
```

### Frontend Settings (`script.js`)

```javascript
// API endpoint (update if backend is on different host/port)
const API_URL = 'http://localhost:5000/api/analyze';
```

---

## 🎨 Customization

### Colors

Edit CSS variables in `styles.css`:

```css
:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --accent-color: #4facfe;
    /* ... more colors ... */
}
```

### Branding

Update the logo and brand name in `index.html`:

```html
<div class="nav-brand">
    <div class="logo-icon"><!-- Your logo SVG --></div>
    <span class="brand-text">Your Brand Name</span>
</div>
```

---

## 🔌 API Endpoints

### `POST /api/analyze`

Analyze an uploaded image.

**Request:**
```
Content-Type: multipart/form-data
Body: image file
```

**Response:**
```json
{
    "detection_image": "/output/web/detection_123.jpg",
    "segmentation_image": "/output/web/segmentation_123.jpg",
    "confidence": 92.5,
    "polyps_count": 1,
    "area_coverage": 12.3,
    "dice_score": 0.876,
    "clinical_description": "<html content>"
}
```

### `GET /api/health`

Check server and model status.

**Response:**
```json
{
    "status": "healthy",
    "models_loaded": {
        "yolo": true,
        "segmentation": true,
        "llava": false
    }
}
```

---

## 🚀 Deployment

### Local Network Access

To access from other devices on your network:

```bash
# In app.py, the server already binds to 0.0.0.0
python app.py

# Access from other devices using your IP:
# http://192.168.1.XXX:5000
```

### Production Deployment

For production, use a proper WSGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Consider using:
- **Nginx** as reverse proxy
- **SSL/TLS** for HTTPS
- **Docker** for containerization

---

## 🐛 Troubleshooting

### Issue: "No module named 'flask'"
```bash
pip install flask flask-cors
```

### Issue: "Models not loading"
- Check model paths in `app.py`
- Ensure models are trained and saved
- Check console output for error messages

### Issue: "CORS errors in browser"
- Ensure `flask-cors` is installed
- Check that backend is running
- Verify API URL in `script.js`

### Issue: "Upload fails"
- Check file size (max 16MB)
- Verify file format (JPG, PNG, JPEG)
- Check browser console for errors

---

## 📱 Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

---

## 🎓 Development

### Running in Development Mode

```bash
# Backend with auto-reload
python app.py

# Frontend with live server (VS Code extension)
# Or use Python's http.server
python -m http.server 8000
```

### Making Changes

1. **HTML/CSS/JS**: Refresh browser to see changes
2. **Python Backend**: Server auto-reloads in debug mode
3. **Models**: Restart server after updating model paths

---

## 📄 License

This project is for research and educational purposes only. Not intended for clinical use.

---

## 🤝 Contributing

To add features or fix bugs:

1. Test in demo mode first
2. Update both frontend and backend
3. Document changes in this README
4. Test with real models before deployment

---

## ⚠️ Medical Disclaimer

**This system is for research and educational purposes only.**

- Not FDA approved
- Not for clinical diagnosis
- All medical decisions must be made by qualified healthcare professionals
- Results should be verified by medical experts

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review browser console for errors
- Check Flask server logs
- Verify model paths and dependencies

---

## 🎉 Enjoy!

You now have a fully functional web interface for your polyp detection AI system!

**Demo Mode**: Works immediately without any setup
**Full Mode**: Connect to your trained models for real analysis

Happy analyzing! 🔬

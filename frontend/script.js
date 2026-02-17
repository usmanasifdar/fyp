// ============================================================================
// POLYP DETECTION AI - JAVASCRIPT
// Interactive functionality for the web application
// ============================================================================

// ============================================================================
// GLOBAL STATE
// ============================================================================
let uploadedFile = null;
let analysisResults = null;

// ============================================================================
// NAVIGATION
// ============================================================================
function scrollToAnalyze() {
    document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' });
}

function showDemo() {
    alert('Demo video coming soon! For now, try uploading an image to see the analysis in action.');
}

// Update active nav link on scroll
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ============================================================================
// FILE UPLOAD HANDLING
// ============================================================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');

// Click to upload
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileUpload(file);
    } else {
        showNotification('Please select a valid image file', 'error');
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileUpload(file);
    } else {
        showNotification('Please drop a valid image file', 'error');
    }
});

// ============================================================================
// FILE PROCESSING
// ============================================================================
function handleFileUpload(file) {
    uploadedFile = file;

    // Read and display image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;

        // Get image dimensions
        const img = new Image();
        img.onload = () => {
            document.getElementById('fileDimensions').textContent = `${img.width} × ${img.height}px`;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Update file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);

    // Show preview area, hide upload area
    uploadArea.style.display = 'none';
    previewArea.style.display = 'block';

    showNotification('Image loaded successfully!', 'success');
}

function clearImage() {
    uploadedFile = null;
    fileInput.value = '';
    previewImage.src = '';

    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';

    showNotification('Image cleared', 'info');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ============================================================================
// ANALYSIS SIMULATION
// ============================================================================
function analyzeImage() {
    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }

    // Hide preview, show processing
    previewArea.style.display = 'none';
    const processingArea = document.getElementById('processingArea');
    processingArea.style.display = 'block';

    // Simulate analysis steps
    simulateAnalysisSteps();
}

function simulateAnalysisSteps() {
    const steps = [
        { id: 'step1', duration: 500, name: 'Image Preprocessing' },
        { id: 'step2', duration: 1500, name: 'YOLO Detection' },
        { id: 'step3', duration: 2000, name: 'U-Net Segmentation' },
        { id: 'step4', duration: 1500, name: 'Clinical Analysis' }
    ];

    let totalTime = 0;

    steps.forEach((step, index) => {
        totalTime += step.duration;

        setTimeout(() => {
            // Mark current step as active
            const stepElement = document.getElementById(step.id);
            stepElement.classList.add('active');
            stepElement.querySelector('.step-icon').textContent = '⟳';

            // Mark previous step as complete
            if (index > 0) {
                const prevStep = document.getElementById(steps[index - 1].id);
                prevStep.querySelector('.step-icon').textContent = '✓';
            }

            // If last step, show results
            if (index === steps.length - 1) {
                setTimeout(() => {
                    stepElement.querySelector('.step-icon').textContent = '✓';
                    showResults();
                }, step.duration);
            }
        }, totalTime - step.duration);
    });
}

function showResults() {
    // Hide processing area
    document.getElementById('processingArea').style.display = 'none';

    // Generate mock results
    generateMockResults();

    // Show results section
    const resultsSection = document.getElementById('results');
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    showNotification('Analysis complete!', 'success');
}

function generateMockResults() {
    // Use the uploaded image for detection result (in real app, this would be from backend)
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            // 1. Create Detection Image (Original + Bounding Box)
            const canvasDet = document.createElement('canvas');
            canvasDet.width = img.width;
            canvasDet.height = img.height;
            const ctxDet = canvasDet.getContext('2d');

            // Draw original image
            ctxDet.drawImage(img, 0, 0);

            // Draw simulated bounding box
            const boxX = img.width * 0.3;
            const boxY = img.height * 0.3;
            const boxW = img.width * 0.4;
            const boxH = img.height * 0.4;

            ctxDet.strokeStyle = '#00ff00';
            ctxDet.lineWidth = Math.max(2, img.width * 0.01);
            ctxDet.strokeRect(boxX, boxY, boxW, boxH);

            // Draw label
            ctxDet.fillStyle = '#00ff00';
            ctxDet.font = `bold ${Math.max(12, img.width * 0.04)}px Arial`;
            ctxDet.fillText('Polyp 95%', boxX, boxY - 10);

            document.getElementById('detectionResult').src = canvasDet.toDataURL();

            // 2. Create Segmentation Mask (Black & White)
            const canvasSeg = document.createElement('canvas');
            canvasSeg.width = img.width;
            canvasSeg.height = img.height;
            const ctxSeg = canvasSeg.getContext('2d');

            // Fill background with black
            ctxSeg.fillStyle = '#000000';
            ctxSeg.fillRect(0, 0, canvasSeg.width, canvasSeg.height);

            // Draw white polyp mask (simulated ellipse in same spot as box)
            ctxSeg.fillStyle = '#ffffff';
            ctxSeg.beginPath();
            ctxSeg.ellipse(
                boxX + boxW / 2,
                boxY + boxH / 2,
                boxW / 2.5,
                boxH / 2.5,
                0, 0, 2 * Math.PI
            );
            ctxSeg.fill();

            document.getElementById('segmentationResult').src = canvasSeg.toDataURL();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(uploadedFile);

    // Generate random but realistic metrics
    const confidence = (85 + Math.random() * 10).toFixed(1);
    const polypsCount = Math.floor(Math.random() * 3) + 1;
    const areaCoverage = (5 + Math.random() * 15).toFixed(1);
    const diceScore = (0.80 + Math.random() * 0.15).toFixed(3);

    document.getElementById('confidence').textContent = confidence + '%';
    document.getElementById('polypsCount').textContent = polypsCount;
    document.getElementById('areaCoverage').textContent = areaCoverage + '%';
    document.getElementById('diceScore').textContent = diceScore;

    // Generate clinical description
    const descriptions = [
        {
            size: 'Small polyp (<5mm) detected in the ascending colon.',
            morphology: 'Pedunculated polyp with smooth surface characteristics and regular borders.',
            recommendations: 'Recommend polypectomy during current procedure. Schedule follow-up colonoscopy in 5 years per standard guidelines.'
        },
        {
            size: 'Medium polyp (5-10mm) detected in the transverse colon.',
            morphology: 'Sessile polyp with slightly irregular surface. No signs of ulceration observed.',
            recommendations: 'Recommend biopsy and histopathological examination. Consider polypectomy. Follow-up based on pathology results.'
        },
        {
            size: 'Large polyp (>10mm) detected in the descending colon.',
            morphology: 'Flat polyp with irregular borders and surface characteristics suggesting potential dysplasia.',
            recommendations: 'Immediate polypectomy recommended with careful histopathological examination. Close surveillance required.'
        }
    ];

    const randomDesc = descriptions[Math.floor(Math.random() * descriptions.length)];

    document.getElementById('clinicalDescription').innerHTML = `
        <div class="description-section">
            <h4>Size Estimation</h4>
            <p>${randomDesc.size}</p>
        </div>
        <div class="description-section">
            <h4>Morphology</h4>
            <p>${randomDesc.morphology}</p>
        </div>
        <div class="description-section">
            <h4>Recommendations</h4>
            <p>${randomDesc.recommendations}</p>
        </div>
        <div class="disclaimer">
            ⚠️ This is an automated assessment for demonstration purposes. Clinical decisions should be made by qualified medical professionals after thorough examination.
        </div>
    `;

    analysisResults = {
        confidence,
        polypsCount,
        areaCoverage,
        diceScore,
        description: randomDesc
    };
}

// ============================================================================
// RESULTS ACTIONS
// ============================================================================
function downloadResults() {
    if (!analysisResults) {
        showNotification('No results to download', 'error');
        return;
    }

    // Create a simple text report
    const report = `
POLYP DETECTION AI - ANALYSIS REPORT
=====================================

File: ${uploadedFile.name}
Date: ${new Date().toLocaleString()}

DETECTION RESULTS
-----------------
Confidence: ${analysisResults.confidence}%
Polyps Detected: ${analysisResults.polypsCount}
Area Coverage: ${analysisResults.areaCoverage}%
Dice Score: ${analysisResults.diceScore}

CLINICAL ASSESSMENT
-------------------
Size: ${analysisResults.description.size}

Morphology: ${analysisResults.description.morphology}

Recommendations: ${analysisResults.description.recommendations}

DISCLAIMER
----------
This is an automated assessment for research and educational purposes only.
All clinical decisions must be made by qualified healthcare professionals.

=====================================
Generated by Polyp Detection AI
    `.trim();

    // Create and download file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `polyp-analysis-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Report downloaded successfully!', 'success');
}

function analyzeAnother() {
    // Reset everything
    clearImage();

    // Hide results section
    document.getElementById('results').style.display = 'none';

    // Scroll to analyze section
    scrollToAnalyze();

    showNotification('Ready for new analysis', 'info');
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? 'rgba(67, 233, 123, 0.2)' :
            type === 'error' ? 'rgba(245, 87, 108, 0.2)' :
                'rgba(79, 172, 254, 0.2)'};
        border: 1px solid ${type === 'success' ? 'rgba(67, 233, 123, 0.4)' :
            type === 'error' ? 'rgba(245, 87, 108, 0.4)' :
                'rgba(79, 172, 254, 0.4)'};
        border-radius: 0.5rem;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================================================
// BACKEND INTEGRATION (For Production)
// ============================================================================
// Uncomment and modify this function when connecting to your Python backend

/*
async function analyzeImageWithBackend() {
    if (!uploadedFile) {
        showNotification('Please upload an image first', 'error');
        return;
    }
    
    // Show processing
    previewArea.style.display = 'none';
    document.getElementById('processingArea').style.display = 'block';
    
    // Create form data
    const formData = new FormData();
    formData.append('image', uploadedFile);
    
    try {
        // Send to backend
        const response = await fetch('http://localhost:5000/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const results = await response.json();
        
        // Display results
        displayBackendResults(results);
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Analysis failed. Please try again.', 'error');
        document.getElementById('processingArea').style.display = 'none';
        previewArea.style.display = 'block';
    }
}

function displayBackendResults(results) {
    // Update UI with actual backend results
    document.getElementById('detectionResult').src = results.detection_image;
    document.getElementById('segmentationResult').src = results.segmentation_image;
    document.getElementById('confidence').textContent = results.confidence + '%';
    document.getElementById('polypsCount').textContent = results.polyps_count;
    document.getElementById('areaCoverage').textContent = results.area_coverage + '%';
    document.getElementById('diceScore').textContent = results.dice_score;
    
    // Update clinical description
    document.getElementById('clinicalDescription').innerHTML = results.clinical_description;
    
    // Show results
    document.getElementById('processingArea').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    
    showNotification('Analysis complete!', 'success');
}
*/

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('Polyp Detection AI - Frontend Loaded');
    console.log('Ready for image analysis!');

    // Add fade-in animation to sections
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('section').forEach(section => {
        observer.observe(section);
    });
});

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Smooth scroll for all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

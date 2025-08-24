"""
This is a simple Flask application that uses the trained PyTorch model
from the provided Jupyter notebook to predict if a chest X-ray image shows signs of pneumonia.

The app provides a web interface where a user can upload a chest X-ray image.
The image is then preprocessed, passed to the model, and the prediction is
displayed on the page.

To run this app:
1. Save this code as `app.py`.
2. Ensure you have the `best_model.pth` and `dataset_mean_std.json` files
   from your notebook in the same directory.
3. Install Flask, torch, torchvision, numpy, and pillow:
   `pip install Flask torch torchvision numpy pillow`
4. Run the app from your terminal:
   `python app.py`
5. Open your web browser and go to `http://127.0.0.1:5000`.
"""
import os
import io
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template_string, request, jsonify
from torchvision import transforms, models

app = Flask(__name__)

# --- Model definition (must be the same as in the notebook) ---
class PneumoResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # In a real-world app, it's better to load weights explicitly
        # and not rely on the 'pretrained' flag to download them.
        # But for this example, we'll keep it consistent with the notebook.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# --- Configuration (must be the same as in the notebook) ---
IMG_SIZE = 224
# NOTE: The provided snippet from your notebook shows ARTIFACTS = './artifacts'.
# Make sure your 'best_model.pth' and 'dataset_mean_std.json' files are in
# a folder named 'artifacts' in the same directory as this 'app.py' file.
ARTIFACTS_DIR = './artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.pth')
MEAN_STD_PATH = os.path.join(ARTIFACTS_DIR, 'dataset_mean_std.json')

# Define classes as they were in the notebook
classes = ['NORMAL', 'PNEUMONIA']

# --- Load mean and std for normalization ---
if not os.path.exists(MEAN_STD_PATH):
    # Fallback to ImageNet stats if custom stats not available
    print("Warning: Custom dataset_mean_std.json not found. Using ImageNet defaults.")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    with open(MEAN_STD_PATH, 'r') as f:
        stats = json.load(f)
        mean = stats['mean']
        std = stats['std']

# --- Image transforms for prediction (same as val_tf in notebook) ---
val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# --- Function to open and convert image to RGB ---
def pil_open_rgb(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

# --- Load the trained PyTorch model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PneumoResNet50(num_classes=len(classes), pretrained=False).to(device)
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        # Use state_dict from the saved dictionary
        model.load_state_dict(state['model'])
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("Error: Model checkpoint not found. Please train the model first.")
    model = None

# --- Flask routes ---
@app.route('/')
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pneumonia Classifier</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', sans-serif;
            }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white p-8 rounded-2xl shadow-lg w-full max-w-2xl">
            <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">Pneumonia X-Ray Classifier</h1>
            <p class="text-gray-600 text-center mb-8">Upload a chest X-ray image to get a prediction. The model classifies the image as either 'NORMAL' or 'PNEUMONIA'.</p>
            <div class="border-2 border-dashed border-gray-300 rounded-xl p-6 mb-6 text-center" id="drop-area">
                <input type="file" id="fileElem" multiple accept="image/*" class="hidden" onchange="handleFiles(this.files)">
                <label class="block cursor-pointer" for="fileElem">
                    <p class="text-gray-500 mb-2">Drag and drop an image here, or click to browse.</p>
                    <button type="button" class="px-4 py-2 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 transition duration-300">
                        Browse Files
                    </button>
                </label>
            </div>
            <div id="image-preview" class="mb-6 hidden">
                <h3 class="text-xl font-bold mb-4 text-center text-gray-800">Selected Image</h3>
                <img id="preview-img" src="#" alt="Image Preview" class="mx-auto rounded-lg shadow-md max-h-64 object-contain">
            </div>
            <div class="text-center mb-6">
                <button id="predict-btn" class="w-full px-6 py-3 bg-green-500 text-white font-bold rounded-lg shadow-md hover:bg-green-600 transition duration-300" onclick="predictImage()" disabled>
                    Predict
                </button>
            </div>
            <div id="result-box" class="p-4 rounded-xl hidden">
                <h3 class="text-xl font-bold mb-2 text-center text-gray-800">Prediction Result</h3>
                <p id="prediction-text" class="text-center text-lg font-semibold"></p>
                <div class="mt-4">
                    <div class="w-full bg-gray-200 rounded-full h-4">
                        <div id="prob-bar" class="bg-blue-500 h-4 rounded-full transition-all duration-500" style="width: 0%;"></div>
                    </div>
                    <p id="prob-text" class="text-sm text-gray-600 text-center mt-2"></p>
                </div>
            </div>
            <div id="error-box" class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl relative hidden mt-4" role="alert">
                <p id="error-message" class="block sm:inline"></p>
            </div>
        </div>

        <script>
            const dropArea = document.getElementById('drop-area');
            const fileElem = document.getElementById('fileElem');
            const previewImg = document.getElementById('preview-img');
            const imagePreview = document.getElementById('image-preview');
            const predictBtn = document.getElementById('predict-btn');
            const resultBox = document.getElementById('result-box');
            const predictionText = document.getElementById('prediction-text');
            const probBar = document.getElementById('prob-bar');
            const probText = document.getElementById('prob-text');
            const errorBox = document.getElementById('error-box');
            const errorMessage = document.getElementById('error-message');
            let uploadedFile = null;

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('border-indigo-600'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('border-indigo-600'), false);
            });

            dropArea.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                let dt = e.dataTransfer;
                let files = dt.files;
                handleFiles(files);
            }

            function handleFiles(files) {
                if (files.length > 0) {
                    uploadedFile = files[0];
                    const reader = new FileReader();
                    reader.onloadend = function() {
                        previewImg.src = reader.result;
                        imagePreview.classList.remove('hidden');
                        predictBtn.disabled = false;
                        resultBox.classList.add('hidden');
                        errorBox.classList.add('hidden');
                    }
                    reader.readAsDataURL(uploadedFile);
                }
            }

            async function predictImage() {
                if (!uploadedFile) {
                    showError("Please upload an image first.");
                    return;
                }

                predictBtn.disabled = true;
                predictBtn.textContent = 'Predicting...';
                resultBox.classList.add('hidden');
                errorBox.classList.add('hidden');

                const formData = new FormData();
                formData.append('image', uploadedFile);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(`Server returned an error: ${response.status} - ${errorData.error}`);
                    }

                    const data = await response.json();
                    displayResult(data);

                } catch (error) {
                    console.error('Prediction failed:', error);
                    showError(`Prediction failed. Error: ${error.message}`);
                } finally {
                    predictBtn.disabled = false;
                    predictBtn.textContent = 'Predict';
                }
            }

            function displayResult(data) {
                const isPneumonia = data.prediction === 'PNEUMONIA';
                const prob = data.prob_pneumonia * 100;
                
                predictionText.textContent = `Prediction: ${data.prediction}`;
                probBar.style.width = `${prob}%`;
                probText.textContent = `Pneumonia Probability: ${prob.toFixed(2)}%`;

                if (isPneumonia) {
                    resultBox.classList.remove('bg-green-100');
                    resultBox.classList.add('bg-red-100');
                    predictionText.classList.remove('text-green-700');
                    predictionText.classList.add('text-red-700');
                    probBar.classList.remove('bg-green-500');
                    probBar.classList.add('bg-red-500');
                } else {
                    resultBox.classList.remove('bg-red-100');
                    resultBox.classList.add('bg-green-100');
                    predictionText.classList.remove('text-red-700');
                    predictionText.classList.add('text-green-700');
                    probBar.classList.remove('bg-red-500');
                    probBar.classList.add('bg-green-500');
                }

                resultBox.classList.remove('hidden');
            }
            
            function showError(message) {
                errorMessage.textContent = message;
                errorBox.classList.remove('hidden');
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    # Define classes locally as they were not available in the global scope
    classes = ['NORMAL', 'PNEUMONIA']
    
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    try:
        image_bytes = file.read()
        img = pil_open_rgb(image_bytes)
        inp = val_tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            
        prediction = classes[pred_idx]
        prob_normal = float(probs[0])
        prob_pneumonia = float(probs[1])

        return jsonify({
            'prediction': prediction,
            'prob_normal': prob_normal,
            'prob_pneumonia': prob_pneumonia
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

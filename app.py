from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import time
import numpy as np
import timm  # <--- make sure you have timm installed (pip install timm)

app = Flask(__name__)

# -----------------------------
# LOAD YOUR PYTORCH MODEL
# -----------------------------
import torch
import timm

MODEL_PATH = 'model/best_deit_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recreate your model architecture
model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=4)

# Load the state dict
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# If the model was saved with nn.DataParallel, strip "module." prefix
if any(k.startswith("module.") for k in state_dict.keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

# Load weights safely
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()


# -----------------------------
# DEFINE LABELS
# -----------------------------
CLASS_NAMES = ['Fatty', 'Scattered', 'Heterogeneously dense', 'Extremely dense (High Risk of Cancer)']

# -----------------------------
# DEFINE TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # adjust to your model input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []

    for key in ['LCC', 'LMLO', 'RCC', 'RMLO']:
        file = request.files.get(key)
        if not file:
            predictions.append({'view': key, 'error': 'No file uploaded'})
            continue

        img_tensor = preprocess_image(file.read()).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probs, 1)
        end = time.time()

        predictions.append({
            'view': key,
            'prediction': int(pred_class.item()),
            'label': CLASS_NAMES[pred_class.item()],
            'confidence': f"{conf.item() * 100:.2f}%",
            'time': f"{end - start:.4f}s"
        })

    return jsonify({'results': predictions})

if __name__ == '__main__':
    app.run(debug=True)

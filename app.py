import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

app = Flask(__name__)

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your classes
class_names = ['bear', 'bird', 'cat', 'cow', 'deer', 'dog', 'dolphin', 'elephant', 'giraffe', 'horse', 'kangaroo', 'lion', 'panda', 'tiger', 'zebra']

try:
    # Load the saved model
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))  # Adjust to the number of classes in your dataset
    model.load_state_dict(torch.load('animal_classifier.pth', map_location=device))
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to predict the class of the image
def predict(image_path):
    if model is None:
        return -1
    
    try:
        image = load_image(image_path)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            
            # Predict the class of the image
            predicted_class = predict(file_path)
            if predicted_class == -1:
                return render_template('index.html', error="An error occurred during prediction")
            
            class_name = class_names[predicted_class]
            return render_template('index.html', class_name=class_name, file_path=file.filename)
    
    # For GET requests, render the default page
    return render_template('index.html', class_name=None, file_path=None)

if __name__ == '__main__':
    # Use environment variable for port if available (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    # Simplified run method to avoid signal handling issues
    app.run(host='0.0.0.0', port=port)

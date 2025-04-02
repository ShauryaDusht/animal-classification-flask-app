import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the saved model
model = models.resnet50()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 15)  # Adjust to the number of classes in your dataset
model.load_state_dict(torch.load('animal_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define your classes
class_names = ['bear', 'bird', 'cat', 'cow', 'deer', 'dog', 'dolphin', 'elephant', 'giraffe', 'horse', 'kangaroo', 'lion', 'panda', 'tiger', 'zebra']

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to predict the class of the image
def predict(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', class_name=None, file_path=None)
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
            class_name = class_names[predicted_class]

            return render_template('index.html', class_name=class_name, file_path=file.filename)

    # For GET requests, render the default page
    return render_template('index.html', class_name=None, file_path=None)

if __name__ == '__main__':
    app.run(debug=True)

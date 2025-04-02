import os
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Set page title and configure page
st.set_page_config(page_title="Animal Classifier", layout="wide")

# Apply custom CSS for fixed image size
st.markdown("""
<style>
.uploaded-image {
    width: 100px;
    height: 100px;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your classes
class_names = ['bear', 'bird', 'cat', 'cow', 'deer', 'dog', 'dolphin', 'elephant', 'giraffe', 'horse', 'kangaroo', 'lion', 'panda', 'tiger', 'zebra']

# Load model function - separate to handle caching
@st.cache_resource
def load_model():
    try:
        model = models.resnet50()
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(class_names))
        model.load_state_dict(torch.load('animal_classifier.pth', map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the class of the image
def predict(image):
    if model is None:
        return -1
    
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return -1

# Streamlit UI
st.title("Animal Classifier")
st.write("Classifies animal into classes: zebra, tiger, panda, lion, kangaroo, horse, giraffe, dolphin, dog, deer, cow, cat, bird, bear, elephant")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Convert to HTML img tag with the custom class
        from io import BytesIO
        import base64
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Display the image with fixed size using HTML
        st.markdown(f'<img src="data:image/png;base64,{img_str}" class="uploaded-image" alt="Uploaded Image">', unsafe_allow_html=True)
        
        # Add a prediction button
        if st.button('Classify'):
            with st.spinner('Classifying...'):
                # Make prediction
                predicted_class = predict(image)
                
                if predicted_class != -1:
                    st.success(f"Predicted animal: {class_names[predicted_class]}")
                else:
                    st.error("Error during prediction. Please try another image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

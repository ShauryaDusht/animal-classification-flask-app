# AnimalClassification

## Overview
This project is a web application built using Flask that classifies animals based on uploaded images. The app leverages a machine learning model trained on the [Animal Data](https://www.kaggle.com/datasets/likhon148/animal-data) dataset from Kaggle. 
* The model can classify the input image into the following animals : Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra.
* The model is trained using ResNet50 Convolutional Neural Network(CNN).
* The model here `animal_classifier.pth` has test accuracy of 99.47% and train accuracy of 97.43% for the mentioned dataset.

## Setup Instructions
### Requirements
* Python3, Flask, torch, torchvision, PIL, numpy, pickle

### Installation
1. Clone the Repository 
```
git clone https://github.com/ShauryaDusht/AnimalClassification
cd AnimalClassification
```
2. Install packages
```
pip install -r requirements.txt
```
3. Make sure your directory looks like this
```
AnimalClassification/
│
├── static/
│   └── styles.css
├── templates/
│   ├── index.html
│   └── result.html
├── animal_classifier.pkl
├── app.py
└── requirements.txt
```

### Running the app
1. Start flask server(for windows)
```
python app.py
```
2. Open a web browser and go to
```http://127.0.0.1:5000/```

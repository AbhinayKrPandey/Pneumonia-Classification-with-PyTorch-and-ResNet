# Pneumonia Classification Web App with PyTorch and Flask
This project is an end-to-end deep learning application that predicts whether a chest X-ray image shows signs of pneumonia. The core of the project is a trained PyTorch model that is exposed through a simple web interface built with Flask.

The goal of this project is to demonstrate a complete workflow for building and deploying a machine learning model, from initial exploratory data analysis to a functional web application.

## Core Components
Pneumonia-ResNet50 Model: A custom PyTorch model based on a pre-trained ResNet-50 backbone. The model is specifically fine-tuned for the task of classifying chest X-rays as either NORMAL or PNEUMONIA.

### Data Handling: 
The project uses PyTorch's torchvision.datasets.ImageFolder and DataLoader for efficient data loading and preprocessing.

### Web Application:
A lightweight web server built with Flask provides an API endpoint (/predict) to handle image uploads and return model predictions.

### Qualitative Analysis:
The project includes a Grad-CAM implementation to generate heatmaps, providing visual explanations of the model's predictions. This helps verify that the model is focusing on the correct regions of the X-ray image (e.g., the lung area) when making its classification.

## Requirements
The project requires the following packages, which can be installed via pip:

Python 3.10

Flask

torch

torchvision

numpy

pillow

tqdm (for progress bars)

scikit-learn

matplotlib

seaborn

### File Structure
The repository contains:

app.py: The main Flask application file that runs the web server.

notebook6bc87b4c9a.ipynb: A Jupyter Notebook that details the complete machine learning workflow, from EDA and model training to evaluation.

artifacts/: A directory that will be created during the notebook's execution to store the trained model (best_model.pth) and normalization statistics (dataset_mean_std.json).

## Key Insights
### Significant Class Imbalance: 
The dataset is highly imbalanced, with a much larger number of pneumonia cases (3875) compared to normal cases (1341) in the training set. This imbalance is a crucial insight as it can bias the model's predictions towards the majority class. The notebook addresses this by using class weights in the loss function to penalize misclassifying the minority class more heavily.

### High Recall for Pneumonia Cases: 
This model achieved a very high recall of 99.4% for the pneumonia class. This is a critical metric in a medical application, as it means the model is highly effective at identifying positive cases and is very unlikely to miss a sick patient (minimize false negatives).

### Model Performance Trade-off:
The overall accuracy is 77.8%, but this metric is somewhat misleading due to the class imbalance. The high recall for pneumonia comes with a lower recall for normal cases (41.8%), indicating a trade-off. This is a common challenge in imbalanced datasets and highlights the importance of analyzing metrics beyond simple accuracy, like recall and precision.

### Explainable AI (XAI) Validation:
The use of Grad-CAM is a key insight. It allows you to visually confirm that the model isn't just "cheating" but is actually focusing on the lung regions of the X-ray images when making its predictions. This is an important step for building trust and confidence in a medical AI model.

### Setup and Usage
Run the Jupyter Notebook: pneumonia_classifier.ipynb to train the model and generate the necessary artifact files (best_model.pth and dataset_mean_std.json) in the artifacts/ directory.

Start the Flask App: Navigate to the directory containing app.py and run it from your terminal: python app.py.

Access the Web Interface: Open your browser and go to http://127.0.0.1:5000 to upload an X-ray image and get a prediction.

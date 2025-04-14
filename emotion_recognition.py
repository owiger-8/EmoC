import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers with increased filter sizes to match the saved model
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 512)  # Adjusted size based on input dimensions
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 512 * 3 * 3)
        
        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class EmotionRecognitionModel:
    def __init__(self, model_path="emotion_model.pth"):
        # Define emotion classes
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        
        # Initialize model
        self.model = EmotionCNN(len(self.emotions))
        
        # Load model weights if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded emotion recognition model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model instead.")
        else:
            print(f"Model file {model_path} not found. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.device = torch.device('cpu')
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def detect_face(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            return face_roi, (x, y, w, h)
        
        return None, None
    
    def predict(self, frame):
        # Detect face in the frame
        face_roi, face_coords = self.detect_face(frame)
        
        if face_roi is None:
            return "neutral"  # Default if no face detected
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_roi)
        
        # Apply transformations
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            emotion_idx = predicted.item()
            emotion = self.emotions[emotion_idx]
            
            return emotion
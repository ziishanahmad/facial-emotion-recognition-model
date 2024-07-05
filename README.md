
```markdown
# Facial Emotion Recognition Model

This repository contains a Convolutional Neural Network (CNN) model for recognizing facial emotions using the FER-2013 dataset. The model is trained to classify images into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Code](#code)
- [Testing with Your Own Image](#testing-with-your-own-image)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The Facial Emotion Recognition model is built using TensorFlow and Keras. It uses a CNN architecture to detect and classify facial emotions from grayscale images. The FER-2013 dataset, which consists of 48x48 pixel grayscale images, is used for training and evaluating the model.

## Dataset

The FER-2013 dataset contains 35,887 images of facial expressions. The dataset is divided into training and testing sets. The dataset can be downloaded from Kaggle using the Kaggle API.

## Model Architecture

The model architecture consists of three convolutional layers, each followed by a max-pooling layer and a dropout layer. The output from the convolutional layers is flattened and passed through two dense layers before the final classification layer.

## Results

The model achieves a test accuracy of approximately 70% on the FER-2013 dataset. Detailed performance metrics can be found in the evaluation section.

## Usage

To train and evaluate the model, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/ziishanahmad/facial-emotion-recognition-model.git
   cd facial-emotion-recognition-model
   ```

2. Set up the environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the FER-2013 dataset using the Kaggle API:
   ```bash
   kaggle datasets download -d msambare/fer2013
   unzip fer2013.zip
   ```

4. Run the Jupyter notebook or Python script to train the model:
   ```bash
   jupyter notebook Facial_Emotion_Recognition.ipynb
   # or
   python train_model.py
   ```

## Code

### Setting Up Kaggle API and Downloading the Dataset

```python
# Install the Kaggle API client
!pip install kaggle

# Make a directory for Kaggle configuration files
!mkdir -p ~/.kaggle

# Copy the kaggle.json file from the Colab file system to the Kaggle configuration directory
# Make sure you have uploaded the kaggle.json file in the Colab file system before running this cell
!cp kaggle.json ~/.kaggle/

# Change the permissions of the kaggle.json file to ensure it's readable
!chmod 600 ~/.kaggle/kaggle.json
```

### Download and Unzip the Dataset

```python
# Download the FER-2013 dataset using the Kaggle API
# This will download the dataset into the current working directory
!kaggle datasets download -d msambare/fer2013

# Unzip the downloaded dataset file
!unzip -o fer2013.zip
```

### Import Necessary Libraries

```python
# Import necessary libraries for data manipulation and visualization
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns  # For statistical data visualization
import os  # For file operations

# Import libraries for building and training the neural network
from tensorflow.keras.models import Sequential  # For building the model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layers for the neural network
from tensorflow.keras.optimizers import Adam  # Optimizer for the model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation

# Import libraries for splitting data and evaluating the model
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.metrics import confusion_matrix, classification_report  # For evaluating the model

# Import libraries for handling image data
from PIL import Image  # For image processing
import requests  # For downloading images from the web
from io import BytesIO  # For handling byte streams
```

### Load and Preprocess the Dataset

```python
# Function to load images and labels from folders
def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for label in label_map:
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = Image.open(img_path).convert('L')  # Convert image to grayscale
            img = img.resize((48, 48))  # Resize image to 48x48 pixels
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label_map[label])
    return np.array(images), np.array(labels)

# Load images and labels from the dataset folder
X, y = load_images_from_folder('train')  # Assuming 'train' folder contains the images

# Normalize the pixel values to the range [0, 1]
X = X / 255.0  # Divide each pixel value by 255

# Add a channel dimension to the images (since they are grayscale)
X = np.expand_dims(X, axis=-1)
```

### Split the Data into Training and Testing Sets

```python
# Split the data into training and testing sets
# 80% of the data will be used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Create an ImageDataGenerator for Data Augmentation

```python
# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Randomly rotate images by up to 10 degrees
    zoom_range=0.1,  # Randomly zoom in on images by up to 10%
    width_shift_range=0.1,  # Randomly shift images horizontally by up to 10%
    height_shift_range=0.1,  # Randomly shift images vertically by up to 10%
    horizontal_flip=True  # Randomly flip images horizontally
)

# Fit the ImageDataGenerator on the training data
datagen.fit(X_train)  # Compute the statistics required for data augmentation
```

### Build the Convolutional Neural Network (CNN) Model

```python
# Build the convolutional neural network (CNN) model
model = Sequential()

# Add the first convolutional and pooling layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add the second convolutional and pooling layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Add the third convolutional and pooling layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Add the first dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Add the output layer
model.add(Dense(7, activation='softmax'))  # 7 output units for 7 emotions

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### Train the Model

```python
# Train the model using the training data
batch_size = 64  # Number of samples per gradient update
epochs = 30  # Number of epochs to train the model

# Train the model using the ImageDataGenerator
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)
```

### Evaluate the Model

```python
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
```

## Testing with Your Own Image

You can test the trained model with your own images by using the provided script. Replace `'aaa.png'` with the path to your image file:

```python
from PIL import Image
import numpy as np


# Define the emotion labels mapping
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load the image from the local environment
img = Image.open('aaa.png').convert('L')  # Convert image to grayscale

# Preprocess the image
img = img.resize((48, 48))  # Resize image to 48x48 pixels
img_array = np.array(img)  # Convert image to numpy array
img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
img_array = img_array.reshape(1, 48, 48, 1)  # Reshape to match input shape of the model

# Predict the class of the image
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

# Print the predicted class and corresponding emotion
print(f"Predicted Class: {predicted_class}")
print(f"Predicted Emotion: {emotion_labels[predicted_class]}")
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- Pillow

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, please contact:

- **Name:** Zeeshan Ahmad
- **Email:** ziishanahmad@gmail.com
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)
```


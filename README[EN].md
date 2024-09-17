# Kaggle Competition: Cleaned vs Dirty Plates Classification

This repository contains the code and explanations for our model designed to classify images of cleaned and dirty plates. Due to constraints such as a limited dataset and the inability to use internet access during the competition, we adopted a specialized approach.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Introduction

In this Kaggle competition, the goal is to develop a machine learning model that can accurately classify images of cleaned and dirty plates. Due to the small dataset and the constraints in the competition environment (e.g., lack of internet access), we faced unique challenges in data and model selection.

## Dataset

The dataset is as follows:

- **Training Set**: 40 images (20 cleaned, 20 dirty)
- **Test Set**: Unlabeled images

Given the small size of the dataset, we applied intensive data augmentation techniques to effectively increase the size of the training data.

## Approach

Initially, we attempted to use transfer learning with pre-trained models like VGG16. However, since we couldn't download pre-trained weights due to the lack of internet access in the competition environment, we designed a Convolutional Neural Network (CNN) from scratch.

Our main steps:

1. **Data Preprocessing and Augmentation**: Expanding the training data through intensive data augmentation.
2. **Model Design**: Crafting a CNN architecture suitable for small datasets.
3. **Training Strategy**: Employing regularization techniques to prevent overfitting.

## Model Architecture

Our CNN model consists of the following components:

- **Convolutional Layers**: Three convolutional blocks with increasing filter sizes (32, 64, 128).
- **Batch Normalization**: Applied after each convolutional layer.
- **Pooling Layers**: MaxPooling layers to reduce spatial dimensions.
- **Fully Connected Layers**: A dense layer with 256 units and ReLU activation.
- **Dropout Layer**: 50% dropout to prevent overfitting.
- **Output Layer**: A single neuron with sigmoid activation for binary classification.

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## Data Preprocessing and Augmentation

Due to the small dataset, we aggressively applied data augmentation techniques to increase the size and diversity of the training data.

Augmentation techniques applied:

- **Rotation**: Up to 90 degrees
- **Width and Height Shifts**: Up to 30%
- **Shear Transformations**
- **Zooming**
- **Horizontal and Vertical Flipping**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
```

## Training

We trained the model using the following strategy:

- **Optimizer**: Adam with a learning rate of 1e-4.
- **Loss Function**: Binary cross-entropy.
- **Metrics**: Accuracy.
- **Callbacks**:
  - **EarlyStopping**: Monitors validation loss and stops training if it doesn't improve after 15 epochs.
  - **ReduceLROnPlateau**: Reduces the learning rate when a metric has stopped improving.

Given the small dataset, we did not specify the `steps_per_epoch` parameter and allowed Keras to compute it automatically.

## Evaluation

We evaluated the model using:

- **Confusion Matrix**: To visualize performance across classes.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.

```python
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(validation_generator)
y_pred = np.round(Y_pred).astype(int).flatten()
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix')
print(cm)

cr = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys())
print('Classification Report')
print(cr)
```

## Results

Despite the challenges, the model achieved satisfactory results considering the constraints. Aggressive data augmentation and regularization techniques helped mitigate overfitting.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Acknowledgments

- Thanks to Kaggle for providing the competition and dataset.

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout,AveragePooling2D
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from keras.models import Model
from keras.models import Sequential
from keras.regularizers import *
from tensorflow import keras
from tensorflow.keras import layers

def download_model():
    model = Sequential()

    # Load the DenseNet201 base with pre-trained weights from ImageNet
    conv_base = DenseNet201(input_shape=(224, 224, 3), include_top=False, pooling='max', weights='imagenet')
    model.add(conv_base)  # Add the DenseNet201 base as the convolutional feature extractor

    # Add Batch Normalization to stabilize and speed up training
    model.add(BatchNormalization())

    # Add a fully connected layer with 2048 units and L1/L2 regularization
    model.add(Dense(2048, activation='relu', kernel_regularizer=l1_l2(0.01)))

    # Add another Batch Normalization layer
    model.add(BatchNormalization())

    # Add a final Dense layer with 8 units (for 8 output classes) and softmax activation
    model.add(Dense(8, activation='softmax'))

    # Specify which layers of the convolutional base will be trainable
    train_layers = [layer for layer in conv_base.layers[::-1][:5]]

    # Set some layers of the convolutional base to be trainable
    for layer in conv_base.layers:
        if layer in train_layers:
            layer.trainable = True

    # Save the model architecture to a file
    model.save("model/model.h5")

 

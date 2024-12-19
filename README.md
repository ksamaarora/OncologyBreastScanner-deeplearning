# **OncologyBreastScanner for Breast Cancer Detection**

Breast cancer is one of the most common malignancies among women, and early detection can significantly reduce mortality rates. Mammography, which uses X-ray images, plays a crucial role in detecting early signs of breast cancer. Doctors use a mammogram to look for early signs of breast cancer. Regular mammograms are the best tests doctors have to find breast cancer early, sometimes up to three years before it can be felt. A mammogram shows how dense the breasts are. Women with dense breasts have a higher risk of getting breast cancer. This project aims to classify breast cancer images based on the density of the tissue using deep learning techniques.

## Project Overview

This project implements a deep learning-based system to classify breast cancer images into benign or malignant categories based on the density of the tissue. The model uses a pre-trained **DenseNet201** architecture fine-tuned for this specific classification task. A **Convolutional Neural Network (CNN)** approach processes the images for early detection of breast cancer, which is critical for better treatment outcomes.

The system is wrapped in a **Gradio-based web interface**, allowing users to upload mammogram images and receive real-time predictions.

<!-- Include this at the start of your project overview section -->
# **OncologyBreastScanner for Breast Cancer Detection**

<video controls width="640" height="360">
  <source src="video.mov" type="video/quicktime">
  Your browser does not support the video tag.
</video>

## Features

- **Deep Learning Classification**: Utilizes the DenseNet201 model to classify images into 8 categories based on malignancy and tissue density.
- **Interactive Interface**: Gradio interface for uploading images and viewing predictions in real-time.
- **Pre-trained Model**: DenseNet201 pre-trained on ImageNet and fine-tuned on breast cancer data.
- **Categorical Classification**: The model predicts whether an image is benign or malignant, based on the density scale (1 to 4).

### Density Categories:

- **(1)** Almost entirely fatty
- **(2)** Scattered areas of fibroglandular density
- **(3)** Heterogeneously dense
- **(4)** Extremely dense

### Tumor Types:

- **Benign** (non-cancerous)
- **Malignant** (cancerous)

### Classification Categories:

1. **Benign with Density = 1**
2. **Malignant with Density = 1**
3. **Benign with Density = 2**
4. **Malignant with Density = 2**
5. **Benign with Density = 3**
6. **Malignant with Density = 3**
7. **Benign with Density = 4**
8. **Malignant with Density = 4**

## Technologies Used

- **TensorFlow**: For building, training, and deploying the deep learning model.
- **Keras**: High-level API for building the model architecture.
- **OpenCV**: For image preprocessing and manipulation (e.g., applying filters).
- **Gradio**: For creating the web interface to interact with the model.
- **NumPy**: For handling image arrays and mathematical operations.

## How the Project Works

### Model Architecture:

- **DenseNet201**: The model uses a pre-trained DenseNet201 architecture, which has been fine-tuned for breast cancer classification. DenseNet201 is known for its dense connections between layers, enabling better feature reuse and improved accuracy.
  
- **Training Process**: The model is trained on a dataset of breast cancer images, with labels corresponding to various malignancy categories. Pre-trained weights from ImageNet help speed up training and improve accuracy.

- **Fine-Tuning**: Some layers from the DenseNet201 base model are unfrozen for fine-tuning on the breast cancer dataset. Regularization techniques like L1 and L2 help prevent overfitting.

- **Preprocessing**: The images are preprocessed by applying filters to enhance relevant features. They are resized to 224x224 and normalized for use with the DenseNet201 model.

## Training Images for each Class

#### Benign 
![Benign](https://github.com/ksamaarora/OncologyBreastScanner-deeplearning/blob/main/image/Begign.png)

#### Malignant
![Malignant](https://github.com/ksamaarora/OncologyBreastScanner-deeplearning/blob/main/image/malignant.png)

### Workflow:

1. **Image Upload**: Users upload an image through the Gradio interface.
2. **Preprocessing**: The image is preprocessed by applying filters and normalization.
3. **Model Prediction**: The processed image is passed to the model for classification.
4. **Output**: The model outputs the predicted class along with probabilities for each of the 8 categories.

## Image Processing

Since mammograms often appear blurry and dull, image preprocessing has been done to enhance sharpness and contrast, improving model performance:

![Processing](https://github.com/ksamaarora/OncologyBreastScanner-deeplearning/blob/main/image/processing.png)

## How to Run the Project Locally

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ksamaarora/oncology-breast-scanner.git
   cd oncology-breast-scanner
   ```

2. **Create and Activate a Conda Environment**:

   ```bash
   conda create -n oncology-breast-scanner python=3.9 -y
   conda activate oncology-breast-scanner
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:

   Start the application:

   ```bash
   python app.py
   ```

   This will start a local server, and you can access the interface at `http://127.0.0.1:7860` or use the public link generated by Gradio.

   When done, deactivate the Conda environment:

   ```bash
   conda deactivate
   ```
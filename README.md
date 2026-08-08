# Down Syndrome Image Pattern Classification Using Deep Learning
<table>
  <tr>
    <td align="center">
      <img src="images/img1.png" width="100%"><br>
      <b>Original Image</b>
    </td>
    <td align="center">
      <img src="images/imge2.png" width="100%"><br>
      <b>Model Prediction</b>
    </td>
  </tr>
</table>

A deep learning-based binary image classification project developed using **TensorFlow, Keras, MobileNetV2, and Transfer Learning**.

The project explores whether a pretrained convolutional neural network can learn to distinguish between two image classes representing the **presence and absence of visual patterns associated with Down syndrome**.

The final fine-tuned model achieved approximately **90% test accuracy** on a balanced test dataset.

> **Important:** This project is intended strictly for educational and research purposes. It is an experimental image-classification system and is **not a medical diagnostic tool**.

---

## 📌 Project Overview

This project implements an end-to-end computer vision workflow including:

* Dataset preparation
* Dataset validation
* Train/validation/test splitting
* Image preprocessing
* Data augmentation
* Transfer learning
* MobileNetV2 feature extraction
* Fine-tuning
* Model evaluation
* Confusion matrix analysis
* Classification reports
* Prediction on unseen images
* Model saving/loading
* Interactive Gradio interface

---

## 📊 Dataset

The dataset used in the notebook contains **2,999 images** divided into two classes:

| Class             | Number of Images |
| ----------------- | ---------------: |
| `pattern_present` |            1,500 |
| `pattern_absent`  |            1,499 |
| **Total**         |        **2,999** |

The notebook also checks the dataset for broken or unreadable image files before training.

**Broken/unreadable images detected: 0**

### Dataset Split

A stratified split was used to maintain approximately equal class distributions.

| Dataset    | Images | Percentage |
| ---------- | -----: | ---------: |
| Training   |  2,099 |        70% |
| Validation |    450 |        15% |
| Test       |    450 |        15% |

Test set distribution:

```text
pattern_absent  : 225
pattern_present : 225
```

---

## 🧠 Model Architecture

The project uses **MobileNetV2** pretrained on the **ImageNet** dataset.

```text
Input Image
    ↓
Data Augmentation
    ↓
MobileNetV2 Preprocessing
    ↓
Pretrained MobileNetV2
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Dense Layer
    ↓
Sigmoid Output
```

### Input

```text
224 × 224 × 3
```

### Output

Binary sigmoid output:

```text
0 → pattern_absent
1 → pattern_present
```

---

## 🔄 Data Augmentation

Basic augmentation was applied to the training images to improve generalization.

The augmentation pipeline includes:

```python
layers.RandomRotation(0.05)
layers.RandomZoom(0.10)
layers.RandomContrast(0.10)
```

---

## 🚀 Transfer Learning

The model uses:

```python
keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
```

During the first training stage, all pretrained MobileNetV2 layers were frozen.

The classifier added on top consists of:

```python
GlobalAveragePooling2D()
Dropout(0.2)
Dense(1, activation="sigmoid")
```

---

## ⚙️ Initial Training

The initial model was compiled using:

```python
optimizer = Adam(learning_rate=0.001)
loss = "binary_crossentropy"
metric = "accuracy"
```

Early stopping and model checkpointing were used during training.

### Initial Test Results

```text
Test Loss     : 0.3818
Test Accuracy : 81.78%
```

### Initial Classification Report

| Class           | Precision | Recall | F1-score |
| --------------- | --------: | -----: | -------: |
| pattern_absent  |      0.84 |   0.79 |     0.81 |
| pattern_present |      0.80 |   0.85 |     0.82 |

Initial confusion matrix:

```text
[[177, 48],
 [ 34,191]]
```

---

## 🔧 Fine-Tuning

After the initial training stage, MobileNetV2 was partially unfrozen.

The final **30 layers** were made trainable while the remaining pretrained layers stayed frozen.

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

A much smaller learning rate was used during fine-tuning:

```python
Adam(learning_rate=0.00001)
```

This allowed the pretrained feature representations to adapt more carefully to the project dataset.

---

## 🎯 Final Results

After fine-tuning:

```text
Test Accuracy : 90.00%
Test Loss     : 0.2818
```

### Classification Report

| Class                | Precision | Recall | F1-score | Support |
| -------------------- | --------: | -----: | -------: | ------: |
| pattern_absent       |      0.88 |   0.92 |     0.90 |     225 |
| pattern_present      |      0.92 |   0.88 |     0.90 |     225 |
| **Overall Accuracy** |           |        | **0.90** | **450** |

### Final Confusion Matrix

```text
[[207, 18],
 [ 27,198]]
```

Out of 450 test images:

```text
Correct predictions   : 405
Incorrect predictions : 45
```

---

## 📈 Improvement After Fine-Tuning

| Model Stage            | Test Accuracy |
| ---------------------- | ------------: |
| Frozen MobileNetV2     |        81.78% |
| Fine-tuned MobileNetV2 |    **90.00%** |

Fine-tuning improved test accuracy by approximately:

```text
8.22 percentage points
```

---

## 🖼️ Predicting New Images

The trained model can classify individual unseen images.

Images are:

1. Loaded from a file
2. Converted to RGB
3. Resized to `224 × 224`
4. Converted into a NumPy/TensorFlow array
5. Passed through the trained network
6. Converted into a binary prediction using a threshold of `0.5`

Example:

```text
Prediction : pattern_present
Confidence : 99.77%
Raw Score  : 0.9977
```

Another test image produced:

```text
Prediction : pattern_absent
Confidence : 83.49%
Raw Score  : 0.1651
```

These are individual examples and should not be interpreted as measures of general clinical performance.

---

## 🖥️ Gradio Interface

A simple interactive interface was created using **Gradio**.

Users can drag and drop or upload an image and receive the model's predicted class and confidence score.

```python
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(
        type="pil",
        label="Drop or Upload an Image"
    ),
    outputs=gr.Textbox(
        label="Prediction Result",
        lines=4
    ),
    title="Down Syndrome Image Pattern Classification",
    description="Upload or drag-and-drop an image for experimental classification."
)
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* Transfer Learning
* Convolutional Neural Networks
* Scikit-learn
* NumPy
* Pillow
* Gradio
* Google Colab
* Google Drive

---

## 📁 Suggested Repository Structure

```text
Down-Syndrome-Image-Classification/
│
├── DownSyndrome.ipynb
│
├── README.md
│
├── requirements.txt
│
├── images/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── gradio_interface.png
│
└── LICENSE
```

The dataset and trained model files do not need to be committed if they are too large or if redistribution is restricted.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Down-Syndrome-Image-Classification.git
cd Down-Syndrome-Image-Classification
```

Install the required packages:

```bash
pip install tensorflow numpy scikit-learn pillow gradio
```

Alternatively, run the notebook directly in **Google Colab**.

---

## ▶️ Running the Project

Open:

```text
DownSyndrome.ipynb
```

in Google Colab or Jupyter Notebook.

Update the dataset path according to your environment:

```python
DATASET_PATH = "/path/to/Dataset"
```

Expected structure:

```text
Dataset/
├── pattern_absent/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── pattern_present/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Run the notebook cells sequentially to:

```text
Load Dataset
      ↓
Validate Images
      ↓
Split Dataset
      ↓
Create TensorFlow Pipeline
      ↓
Train MobileNetV2 Classifier
      ↓
Evaluate Initial Model
      ↓
Fine-Tune MobileNetV2
      ↓
Evaluate Final Model
      ↓
Save Model
      ↓
Test New Images
      ↓
Launch Gradio Interface
```

---

## 🔮 Possible Future Improvements

Possible extensions to this project include:

* Evaluating additional transfer-learning architectures such as EfficientNet, ResNet, and DenseNet
* Performing systematic hyperparameter tuning
* Applying stronger data augmentation
* Adding ROC and Precision-Recall curve analysis
* Performing k-fold cross-validation
* Investigating class activation maps such as Grad-CAM
* Testing model robustness on images collected from independent datasets
* Investigating potential dataset bias and data leakage
* Improving model explainability
* Deploying the demonstration application using Hugging Face Spaces or another cloud platform

---

## ⚠️ Medical and Ethical Disclaimer

This repository is an **educational machine learning project**.

The model has been evaluated only on the dataset used in this experiment. A test accuracy of 90% on this dataset does **not** establish clinical validity, diagnostic reliability, or generalization to real-world populations.

Down syndrome is a medical/genetic condition, and diagnosis should be performed using appropriate clinical procedures by qualified healthcare professionals.

The model should therefore **not be used for diagnosis, screening, treatment decisions, or any other medical decision-making**.

---

## 📚 What I Learned

Through this project, I gained practical experience with:

* Building TensorFlow image pipelines
* Transfer learning
* CNN-based image classification
* MobileNetV2
* Data augmentation
* Fine-tuning pretrained neural networks
* Binary classification
* Confusion matrices
* Precision, recall, and F1-score
* Model checkpointing
* Early stopping
* Model inference
* Creating simple ML interfaces using Gradio

---

## 👤 Author

**Kavindu Dilshan Wijesinghe**

Undergraduate – Electronic & Telecommunication Engineering
University of Moratuwa

---

## ⭐ Support

If you find this project useful, feel free to star the repository.

Feedback and suggestions for improving the machine learning methodology are welcome.

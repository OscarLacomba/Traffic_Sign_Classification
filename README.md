# Traffic_Sign_Classification
German traffic sign classifier (GTSRB) · CNN · TensorFlow · Deployed on HuggingFace Spaces

# 🚦 German Traffic Sign Classifier

> Deep learning model that classifies **43 types of German traffic signs** using Convolutional Neural Networks (CNNs). Trained on the GTSRB dataset and deployed as an interactive Streamlit app on HuggingFace Spaces.

---

## 📌 Overview

This project focuses on one of the fundamental capabilities of autonomous vehicles: **reading and correctly classifying traffic signs**. A CNN model is trained end-to-end on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, covering the full ML pipeline from exploratory data analysis to cloud deployment.

---

## 📂 Dataset

| Split      | Images  | Description                          |
|------------|---------|--------------------------------------|
| Training   | 34,799  | 32×32 px color images, 43 classes    |
| Validation | 4,410   | Used for hyperparameter tuning       |
| Test       | 12,630  | Final evaluation                     |

**Source:** [GTSRB — German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/)

---

## 🗂️ Project Structure

```
traffic-sign-classifier/
│
├── notebook/
│   └── Traffic_Sign_Classifier_CNN.ipynb   # Main Colab notebook
│
├── app/
│   ├── app.py                              # Streamlit app for HuggingFace
│   └── requirements.txt
│
├── artifacts/
│   ├── traffic_sign_model.h5               # Trained model
│   ├── class_names.json                    # 43 class labels
│   ├── metrics.json                        # Evaluation results
│   └── training_history.json              # Loss/accuracy curves
│
├── figures/
│   ├── eda_class_distribution.png
│   ├── eda_sample_images.png
│   ├── eda_brightness_analysis.png
│   ├── eda_rgb_channels.png
│   ├── training_curves.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

Five visualizations were generated to understand the dataset before training:

1. **Class distribution** — Identifies class imbalance across training and validation sets
2. **Sample images per class** — Visual overview of all 43 traffic sign types
3. **Brightness analysis** — Box plots and histograms of pixel intensity by class
4. **RGB channel analysis** — Mean channel values per class to detect color patterns
5. **Augmentation preview** — Effect of data augmentation transformations

---

## ⚙️ Preprocessing

- Rescaling pixel values to `[0, 1]`
- **Data augmentation** applied to training set:
  - Random rotation (±15°)
  - Width / height shifts (10%)
  - Zoom range (15%)
  - Brightness variation (0.8–1.2×)
  - No horizontal flip (traffic signs are not horizontally symmetric)

---

## 🧠 Model Architecture

Custom CNN with 3 convolutional blocks:

```
Input (32×32×3)
  → Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → Conv2D(128) → MaxPool → Dropout(0.40)
  → Flatten → Dense(512) → BN → Dropout(0.50)
  → Dense(43, softmax)
```

**Training configuration:**

| Parameter     | Value                        |
|---------------|------------------------------|
| Optimizer     | Adam (lr = 1e-3)             |
| Loss          | Categorical cross-entropy    |
| Batch size    | 64                           |
| Max epochs    | 30                           |
| Early stopping| Patience = 8 (val_accuracy)  |
| LR scheduler  | ReduceLROnPlateau (factor 0.5)|

---

## 📊 Results

| Metric         | Value   |
|----------------|---------|
| Test Accuracy  | ~97%+   |
| Test Loss      | < 0.10  |

> ⚠️ Results will vary slightly depending on random seed and GPU. Update this table with your actual results after training.

---

## 🔮 Inference

```python
from PIL import Image
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model("artifacts/traffic_sign_model.h5")

with open("artifacts/class_names.json") as f:
    CLASS_NAMES = json.load(f)

def predict(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.array(img) / 255.0
    proba = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top_idx = np.argmax(proba)
    return CLASS_NAMES[top_idx], float(proba[top_idx])

label, confidence = predict("my_sign.jpg")
print(f"Prediction: {label} ({confidence*100:.2f}%)")
```

---

## 🚀 Deployment — HuggingFace Spaces

The project is deployed as a **Streamlit app** on HuggingFace Spaces with four tabs:

| Tab         | Content                                                   |
|-------------|-----------------------------------------------------------|
| 📸 Classify | Upload any image and get a Top-5 prediction with probabilities |
| 🖼️ Examples | Sample images from each of the 43 classes                 |
| 📊 EDA      | Interactive EDA figures generated during analysis         |
| 📈 Metrics  | Accuracy, loss curves, and confusion matrix               |

🔗 **Live demo:** [huggingface.co/spaces/YOUR_USERNAME/traffic-sign-classifier](https://huggingface.co/spaces/)

---

## 🛠️ Setup & Usage

### Run the notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open `notebook/Traffic_Sign_Classifier_CNN.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells sequentially

### Run the Streamlit app locally

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-classifier.git
cd traffic-sign-classifier/app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow>=2.12
streamlit>=1.28
numpy
Pillow
matplotlib
seaborn
scikit-learn
pandas
gdown
```

---

## 📚 References

- Stallkamp, J. et al. — *The German Traffic Sign Recognition Benchmark: A multi-class classification competition* (IJCNN 2011)
- [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---

## 👤 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [linkedin.com/in/your_profile](https://linkedin.com/in/)

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

# 🛍️ E-commerce Text Classification using Bidirectional RNN (BRNN)

This project implements a **text classification model** using **Bidirectional Recurrent Neural Networks (BRNN)** to automatically categorize e-commerce product descriptions into predefined classes such as clothing, electronics, books, and more.

---

## 📌 Problem Statement

E-commerce platforms deal with massive volumes of product data. Classifying products based on their descriptions helps in better search, recommendation, and inventory management.

The goal is to build a deep learning model that can accurately classify product descriptions into their respective categories.

---

## 🚀 Solution Overview

We use Natural Language Processing (NLP) techniques and a **Bidirectional RNN (BRNN)** model (using Keras and TensorFlow backend) to process and classify textual data.

---

## 🧠 Model Architecture

- **Embedding Layer**: Converts words into dense vectors
- **Bidirectional RNN (GRU/LSTM)**: Processes the input in both forward and backward directions to capture context
- **Dense Layers**: Fully connected layers to output the class probabilities
- **Activation**: Softmax for multi-class classification

---

## 📁 Dataset

- Text: E-commerce product descriptions
- Labels: Categories like **clothing**, **electronics**, **home & kitchen**, **books**, etc.
- Format: CSV with two columns — `description`, `category`
<img width="1217" height="361" alt="image" src="https://github.com/user-attachments/assets/d8cac328-90b7-4b02-959e-6d882876d16e" />


**requirements.txt:**

```txt
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow==2.11.0
keras==2.11.0
nltk
jupyter
```

---

## 📊 Evaluation Metrics

* **Accuracy**: \~87% on the test set
* **Confusion Matrix**
* **Precision / Recall / F1-score**

---

## 📈 Visualizations

* Confusion Matrix (with heatmap)
* Loss and Accuracy over Epochs
* Actual vs Predicted Plot
* Scatter Plot for predictions

---

## 🗂️ Project Structure

```bash
├── dataset.csv
├── ecomm_text_classification.ipynb
├── requirements.txt
├── README.md
└── saved_model/
```

---

## 💡 Key Concepts Used

* Text Preprocessing (Tokenization, Padding)
* Embedding Layer
* Bidirectional RNNs (GRU or LSTM)
* Multi-class classification
* One-hot encoding
* Evaluation Metrics (confusion matrix, accuracy, precision, recall)

---

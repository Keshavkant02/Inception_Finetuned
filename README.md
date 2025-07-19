# Pneumonia Detection from Chest X-Rays using Transfer Learning

**A deep learning project to fine-tune an Inception-V3 model on the PneumoniaMNIST dataset for classifying chest X-rays as showing signs of pneumonia or being normal.**

---

## Table of Contents
1.  [Project Objective](#project-objective)
2.  [Key Features](#key-features)
3.  [Dataset](#dataset)
4.  [Methodology](#methodology)
5.  [Results and Analysis](#results-and-analysis)
6.  [Setup and Installation](#setup-and-installation)
7.  [How to Run](#how-to-run)
8.  [File Structure](#file-structure)
9.  [Technologies Used](#technologies-used)

---

## Project Objective

The primary objective of this project is to develop and evaluate a deep learning model for medical image classification. This involves:
-   **Fine-tuning a pre-trained Inception-V3 architecture** to distinguish between "Pneumonia" and "Normal" chest X-ray images.
-   Implementing a robust **evaluation strategy** using appropriate metrics for an imbalanced medical dataset.
-   Applying techniques to **mitigate class imbalance** and **prevent overfitting**.
-   Reporting on the model's final performance and providing insights into its clinical applicability.

---

## Key Features

-   **Transfer Learning:** Leverages the powerful, pre-trained Inception-V3 model from `torchvision` to achieve high performance with a relatively small dataset.
-   **Class Imbalance Mitigation:** Employs a weighted loss function (`nn.CrossEntropyLoss` with class weights) to force the model to pay equal attention to the minority "Normal" class during training.
-   **Overfitting Prevention:** Utilizes a combination of **Data Augmentation** (random rotations, flips, and color jitter) and **Regularization** (Dropout layers and Weight Decay) to ensure the model generalizes well to unseen data.
-   **Comprehensive Evaluation:** Goes beyond simple accuracy to report on **AUC-ROC, Precision, and Recall**, providing a clear picture of the model's clinical trade-offs.

---

## Dataset

This project utilizes the **PneumoniaMNIST** dataset, which is part of the MedMNIST v2 collection.
-   **Source:** [MedMNIST Homepage](https://medmnist.com/)
-   **Description:** The dataset consists of 5,856 chest X-ray images, categorized into "Normal" and "Pneumonia".
-   **Class Imbalance:** The training set exhibits a significant class imbalance, with a ratio of approximately 3:1 (Pneumonia to Normal), which was a key challenge addressed.

---

## Methodology

The project follows a standard transfer learning pipeline:
1.  **Data Preprocessing & Augmentation:** Input images are resized to `299x299` for Inception-V3 and normalized using ImageNet statistics. The training data undergoes on-the-fly augmentation.
2.  **Model Architecture:** A pre-trained Inception-V3 model is loaded. Its base layers are frozen, and the final `fc` layer is replaced with a new custom classifier head that includes Dropout. Only this new head is trained.
3.  **Training Strategy:** A weighted `CrossEntropyLoss` is used to counteract class imbalance, and the Adam optimizer with weight decay is used for training.

---

## Results and Analysis

The model was trained for 10 epochs. The final evaluation on the test set yielded an **ROC AUC score of 0.889** and a **Pneumonia class recall of 0.94**, indicating strong performance in identifying sick patients, though with a trade-off in false positives.

### Visual Results

The training history and final evaluation metrics are visualized below.

![Training History and Final Results](roc%20curve.png)
*Figure 1: (Left to Right) Training & Validation Loss, Training & Validation Accuracy, Confusion Matrix, and ROC Curve.*
---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Keshavkant02/Inception_Finetuned.git
    cd Inception_Finetuned
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run

The entire project is contained within the Jupyter Notebook (`Pneumonia_Assignment.ipynb`). After setting up the environment, the cells can be run sequentially from top to bottom in Google Colab or a local Jupyter environment.

---

## File Structure

*   `Pneumonia_Assignment.ipynb`: The main Jupyter Notebook containing all project code, analysis, and outputs.
*   `README.md`: This documentation file.
*   `requirements.txt`: A list of all Python dependencies required to reproduce the environment.
*   `roc curve.png`: The image file containing the final result plots.

---

## Technologies Used

-   **Python 3.x**
-   **PyTorch:** The primary deep learning framework.
-   **MedMNIST:** For easy and standardized dataset access.
-   **scikit-learn:** For calculating evaluation metrics and class weights.
-   **NumPy:** For numerical operations.
-   **Matplotlib & Seaborn:** For data visualization and plotting results.
-   **Google Colab:** As the development environment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Keshavkant02/Inception_Finetuned/blob/main/PGI_Biostats_InceptionFinetuning.ipynb) 

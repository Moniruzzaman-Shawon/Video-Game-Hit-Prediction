# 🎮 Video Game Hit Prediction

## Overview
Video Game Hit Prediction builds an end-to-end Machine Learning system to predict whether a video game will become a **commercial hit** based on its metadata and regional sales performance.  
A game is classified as a **Hit** if its global sales exceed **1 million units**.

*The project demonstrates the full ML lifecycle — from data preprocessing and model selection to experiment tracking and deployment.*

---

## 🌐 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/shawon17/Video-Games-Hit-Prediction)


🟡 Note: On the free tier, Spaces may go to sleep due to inactivity.  
Click **“Restart this Space”** to start the demo.

---

## 🖥️ Web Application Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/50099d1b-4814-4e7f-95db-e0e771de007b"
       alt="Video Game Hit Prediction App"
       width="900" />
</p>

---

## Dataset
- **Source:** Kaggle – Video Game Sales Dataset  
- **Type:** Structured tabular data  
- **Records:** ~16K video games  
- **Prediction Target:** `Hit` (Binary: Hit / Not Hit)

---

## Approach
- Feature engineering and leakage prevention  
- Unified preprocessing and modeling using **scikit-learn Pipelines**  
- Robust evaluation using **cross-validation**  
- Hyperparameter optimization with **GridSearchCV**  
- Experiment tracking with **MLflow**  
- Interactive prediction interface built using **Gradio**

---

## Model
### RandomForestClassifier

**Chosen for its:**
- Strong performance on tabular data  
- Ability to capture non-linear relationships  
- Robustness to noisy and imbalanced data  
- Minimal need for manual feature tuning  

---

## Evaluation
The model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Results show **high accuracy and consistent performance**, indicating good generalization on unseen data.

---

## Experiment Tracking
**MLflow** was used to log:
- Model hyperparameters  
- Cross-validation results  
- Final test metrics  
- Trained model artifacts  

This enables reproducibility and structured experimentation.

---

## Web Application
A lightweight **Gradio web app** allows users to input game details (platform, genre, year, sales, etc.) and receive:
- **Hit / Not Hit** prediction  
- **Confidence score**

## Project Structure
````
Video-Games-Hit-Prediction/
├── train.py          # Model training & MLflow tracking
├── app.py            # Gradio web application
├── best_model.pkl    # Trained ML pipeline
├── vgsales.csv       # Dataset
├── requirements.txt
└── README.md
````
## Run Locally
````
pip install -r requirements.txt
python train.py
python app.py
````

## Deployment

The Gradio application is deployed on Hugging Face Spaces and is publicly accessible.

## Key Takeaways
- Designed and implemented an end-to-end machine learning pipeline  
- Applied industry-standard practices for data preprocessing, validation, and evaluation  
- Tracked experiments and model performance using MLflow  
- Deployed a production-ready prediction interface using Gradio  

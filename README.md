
# Suicide Detection Web Application

This project is a web-based suicide detection system built using Streamlit.  
It analyzes user-provided text and predicts whether it indicates suicide risk using a deep learning model and transformer-based embeddings.

## Overview
The application uses a pre-trained DistilBERT model to generate text embeddings, which are then passed to a custom deep learning classifier for prediction. The system is designed for inference and deployment purposes.

## Tech Stack
- Python
- Streamlit
- DistilBERT (Transformers)
- TensorFlow / Keras
- PyTorch
- NumPy, Pandas

## How to Run the Application
1. Install required dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Streamlit app:
```bash
streamlit run hybridapp.py
```
3. Open the local URL shown in the terminal to access the web app.

## Model Details
- **Transformer:** distilbert-base-uncased
- **Classifier:** Custom CNN + ConvMixer + Channel Attention architecture
- **Task:** Binary classification (Suicide / Not Suicide)

## Evaluation Metrics
The model was evaluated on a test set from the Reddit SNS dataset.

- **Confusion Matrix:**
```text
[[20649  1489]
 [ 2142 20956]]
```
- **Accuracy:** 0.9197  
- **Precision:** 0.9337  
- **Recall:** 0.9073  
- **F1 Score:** 0.9203  

> Note: Metrics are indicative of performance on the dataset used for training. Results may vary for data from other sources.

## Limitations
- The model is trained primarily on the Reddit SNS dataset.  
- Performance may vary for text from other platforms or domains.  
- The model does not understand context beyond the provided text input.  
- Predictions should not be interpreted as clinical or professional diagnoses.

## Repository Scope
This repository contains only the inference and deployment code.  
Training, preprocessing, and dataset-related code are intentionally kept private.

## Disclaimer
This application is developed for educational and research purposes only.  
It should not be used as a substitute for professional mental health support.

## Author
Sheik Mohammed A

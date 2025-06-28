# Hotel Booking Cancellation Prediction

## Overview

This MLOps project predicts hotel booking cancellations using a structured machine learning pipeline. It simulates a real-world workflow including data validation, preprocessing, feature selection, model training, serving, and drift detection. The goal is to build a reproducible and scalable ML system using modern MLOps tools and practices.

## Dataset

This project uses the [Hotel Booking Cancellation Prediction dataset](https://www.kaggle.com/datasets/youssefaboelwafa/hotel-booking-cancellation-prediction) from Kaggle.

The dataset contains real-world hotel booking records and contains 17 columns (including the target variable: booking status)

## Objectives
The main objective is to build a complete and modular Machine Learning Pipeline that represents the life cycle of a model in a real context, where we want to predict whether or not a booking will be canceled, based on the information that is provided in the dataset. This pipeline must include: 

- Data Ingestion and Validation 
- Pre-Processing and Feature Engineering
- Unit Testing
- MLflow Experiment Tracking and Versioning
- Performance Evaluation and SHAP for Model Explainability
- Data Drift Pipeline
- Model Serving Ready Structure

## Technologies Used
- Pipeline: Kedro (orchestration), MLflow (experiments), Great Expectations (data quality) 
- ML: Scikit-learn, XGBoost, SHAP (explainability) 
- Storage: Hopsworks (feature store)
- Monitoring: NannyML (drift detection)

## Getting Started
### 1. Clone the repository

```bash
git clone <your-repo-url>
cd MLOPS/project_mlops
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
kedro run
```

### 4. Run a specific pipeline
```bash
kedro run --pipeline=model_train
```

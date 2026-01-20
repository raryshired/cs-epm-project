# Predictive Maintenance - MLOps Pipeline

End-to-end MLOps pipeline for engine condition classification using Gradient Boosting with automated CI/CD via GitHub Actions.

## Project Overview

- **Task:** Binary classification (Normal vs Faulty engine condition)
- **Model:** Gradient Boosting (sklearn) with Bayesian hyperparameter optimization
- **Metrics:** F1-Score ~78%, Recall ~97%
- **Deployment:** Streamlit app on Hugging Face Spaces

## Architecture

```
├── cspredmaintproj/
│   ├── data/
│   │   ├── raw/
│   │   │   └── engine_data.csv
│   │   ├── processed/
│   │   │   ├── X_train.csv, y_train.csv
│   │   │   ├── X_val.csv, y_val.csv
│   │   │   └── X_test.csv, y_test.csv
│   │   └── artifacts/
│   │       ├── column_schema.json
│   │       ├── outlier_metadata.json
│   │       └── split_metadata.json
│   ├── models/
│   │   ├── best_model.joblib
│   │   └── model_metadata.json
│   └── model_building/
│       ├── data_register.py
│       ├── data_prep.py
│       └── model_register.py
├── deployment/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── .github/workflows/
│   └── pipeline.yml
├── deploy_to_hf.py
├── requirements-pipeline.txt
├── RamnathanRavindran_PredictiveMaintenance_Notebook_Final_1.0_20260125.ipynb
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements-pipeline.txt

# Data registration to HF Hub
python cspredmaintproj/model_building/data_register.py

# Data preparation and upload to HF
python cspredmaintproj/model_building/data_prep.py

# Model training with MLflow tracking (start MLflow first)
mlflow ui --host 0.0.0.0 --port 5001 &
python cspredmaintproj/model_building/model_register.py

# Deploy to Hugging Face Space
python deploy_to_hf.py
```

## Hugging Face Resources

| Resource | URL |
|----------|-----|
| **Dataset** | https://huggingface.co/datasets/spac1ngcat/cs-pred-maintain-ds |
| **Model** | https://huggingface.co/spac1ngcat/cs-pred-maintain-model |
| **App** | https://huggingface.co/spaces/spac1ngcat/cs-pred-maintain-app |

## CI/CD Pipeline

The GitHub Actions workflow automates:
1. **register-dataset:** Upload raw dataset to Hugging Face Dataset Hub
2. **data-prep:** Process data, create train/val/test splits, upload to HF
3. **model-training:** Train Gradient Boosting with BayesSearchCV, log to MLflow, upload model to HF
4. **deploy-to-hf-space:** Deploy Streamlit app to Hugging Face Space

**Triggers:**
- Push to `main` branch (automatic)
- Manual workflow dispatch from GitHub Actions UI

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~65% |
| Recall | ~97% |
| F1-Score | ~78% |
| ROC-AUC | ~0.98 |

**Key Features:**
- Engine RPM is the strongest predictor
- High recall (97%) ensures minimal false negatives (safety-focused)
- Trained on 19,535 real engine sensor readings

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face authentication token |
| `RUN_HF_DATA_REGISTER` | Enable data registration (true/false) |
| `RUN_PREP` | Enable data preparation upload (true/false) |
| `RUN_MODEL_TRAINING` | Enable model training/upload (true/false) |
| `RUN_DEPLOY` | Enable deployment to HF Space (true/false) |
| `MLFLOW_TRACKING_URI` | MLflow server URI (default: mlruns) |

## Contributors

- Ramnathan Ravindran

## License

MIT License

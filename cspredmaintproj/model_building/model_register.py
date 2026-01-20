"""
Model Registration Script
Uploads the best model to Hugging Face Model Hub.
"""
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os
import sys
from pathlib import Path

# Constants (should match notebook cell 20)
PROJECT_NAME = "cspredmaintproj"
MODEL_DIR = "models"
MODEL_FILE = "best_model.joblib"
MODEL_METADATA_FILE = "model_metadata.json"
HF_MODEL_REPO_NAME = "cs-pred-maintain-model"
HF_USER_NAME = "spac1ngcat"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set or empty.")
    sys.exit(1)

# Use env var if available; fallback to hardcoded default for standalone execution
repo_id = os.getenv("HF_MODEL_REPO_ID", f"{HF_USER_NAME}/{HF_MODEL_REPO_NAME}")
repo_type = "model"

base_dir = Path(__file__).resolve().parents[1]
model_dir = base_dir / MODEL_DIR

if not model_dir.exists():
    print(f"Error: Models folder not found at {model_dir}")
    sys.exit(1)

# Required files
required_files = [MODEL_FILE, MODEL_METADATA_FILE]

# Verify all files exist
missing_files = [f for f in required_files if not (model_dir / f).exists()]
if missing_files:
    print(f"Error: Missing files in models folder: {missing_files}")
    sys.exit(1)

print(f"All required files found in {model_dir}")

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repository '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Creating model repository '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
    print(f"Model repository '{repo_id}' created.")

# Upload model files
print(f"\nUploading model to {repo_id}...")
api.upload_folder(
    folder_path=str(model_dir),
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Model uploaded successfully")

print(f"\nModel registration completed for {repo_id}")
print("Files uploaded:")
for f in required_files:
    print(f"  - {f}")

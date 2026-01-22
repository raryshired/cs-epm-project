"""
Data Preparation Upload Script
Uploads processed train/test splits to Hugging Face dataset repository.
"""
from huggingface_hub import HfApi
import os
import sys
from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set or empty.")
    sys.exit(1)

repo_id = "spac1ngcat/cs-pred-maintain-ds"
repo_type = "dataset"

base_dir = Path(__file__).resolve().parents[1] / "data"
processed_dir = base_dir / "processed"
artifacts_dir = base_dir / "artifacts"

if not processed_dir.exists():
    print(f"Error: Processed data folder not found at {processed_dir}")
    sys.exit(1)

# Required files
required_files = [
    'X_train.csv', 'y_train.csv',
    'X_test.csv', 'y_test.csv'
]

# Verify all files exist
missing_files = [f for f in required_files if not (processed_dir / f).exists()]
if missing_files:
    print(f"Error: Missing files in processed folder: {missing_files}")
    sys.exit(1)

print(f"All required files found in {processed_dir}")

api = HfApi(token=HF_TOKEN)

# Upload processed folder
print(f"\nUploading processed data to {repo_id}/processed...")
api.upload_folder(
    folder_path=str(processed_dir),
    path_in_repo="processed",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Processed data uploaded successfully")

# Upload artifacts folder (schemas, metadata)
if artifacts_dir.exists():
    print(f"\nUploading artifacts to {repo_id}/artifacts...")
    api.upload_folder(
        folder_path=str(artifacts_dir),
        path_in_repo="artifacts",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Artifacts uploaded successfully")

print(f"\nData preparation upload completed for {repo_id}")
print("Files uploaded:")
for f in required_files:
    print(f"  - processed/{f}")
print("  - artifacts/column_schema.json")
print("  - artifacts/outlier_metadata.json")
print("  - artifacts/split_metadata.json")

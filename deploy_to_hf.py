"""
Model Deployment Script
Uploads deployment folder to Hugging Face Space.
"""
from huggingface_hub import HfApi
import os
import sys

HF_USER_NAME = "spac1ngcat"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set or empty.")
    sys.exit(1)

SPACE_ID = f"{HF_USER_NAME}/cs-pred-maintain-app"
DEPLOY_DIR = "deployment"

# Verify deployment folder exists
if not os.path.exists(DEPLOY_DIR):
    print(f"Error: Deployment folder not found at {DEPLOY_DIR}")
    sys.exit(1)

# Required files
required_files = ['Dockerfile', 'app.py', 'requirements.txt']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(DEPLOY_DIR, f))]
if missing_files:
    print(f"Error: Missing files in deployment folder: {missing_files}")
    sys.exit(1)

print(f"All required files found in {DEPLOY_DIR}/")
print(f"Required files: {required_files}")

api = HfApi(token=HF_TOKEN)

# Upload deployment folder
print(f"\nUploading deployment folder to {SPACE_ID}...")
api.upload_folder(
    folder_path=DEPLOY_DIR,
    repo_id=SPACE_ID,
    repo_type="space",
    path_in_repo="",
)
print("Deployment uploaded successfully")

print(f"\nDeployment complete!")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
print("\nNote: HF Space must be created first with:")
print("  - SDK: Docker")
print("  - HF_TOKEN added to Space secrets")

import os
from dotenv import load_dotenv

load_dotenv()

# LM Studio Configuration (OpenAI Compatible)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "local-model") # LM Studio uses the currently loaded model

# Vector DB Configuration
CHROMA_PATH = "chroma_data"
DATA_PATH = "data/Nepali_house_dataset.csv"
MODELS_DIR = "saved_models"

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

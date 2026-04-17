import os, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

proj_name = "pneumonia_segmentation"
list_of_files = [
    ".github/workflows/.gitkeep",        # CI/CD
    ".github/workflows/main.yaml",
    

    f"src/{proj_name}/__init__.py",
    f"src/{proj_name}/components/__init__.py",  # core ML logic
    f"src/{proj_name}/adapters/__init__.py",    # data ingestion
    f"src/{proj_name}/constants/__init__.py",
    f"src/{proj_name}/exception/__init__.py",
    f"src/{proj_name}/cloud/__init__.py",
    f"src/{proj_name}/config/__init__.py",
    f"src/{proj_name}/utils/__init__.py",       # helper functions
    f"src/{proj_name}/pipeline/__init__.py",    # training & prediction flow
    f"src/{proj_name}/entity/__init__.py",      # config dataclasses
    
    # ------------- MICROSERVICES ----------------------------
    f"packages/core/pyproject.toml",  
    f"packages/core/src/core/__init__.py",
    f"packages/core/src/core/exception.py",
    f"packages/core/src/core/logging.py",
    f"packages/core/src/core/utils",
    f"packages/core/src/core/constants",
    
    f"packages/ingestion/pyproject.toml",
    f"packages/ingestion/src/ingestion/__init__.py",
    f"packages/ingestion/src/ingestion/adapters",
    f"packages/ingestion/src/ingestion/components",
    f"packages/ingestion/src/ingestion/config",
    f"packages/ingestion/src/ingestion/pipeline",
    
    f"services/ingestion/main.py",
    f"services/ingestion/Dockerfile",
    
    f"services/training/main.py",
    f"services/training/Dockerfile",
    
    
    "tests/__init__.py",

    "dvc.yaml",
    "config/config.yaml",   # paths & settings
    "params.yaml",          # hyperparameters
    # "schema.yaml",          # data schema
    ".env",                 # secrets
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.in",
    "requirements_dev.in",
    "pyproject.toml",               # to package code
    "notebooks/research.ipynb",  # EDA & experiments
    "templates",       # Flask UI / FastAPI
    "dvc.yaml",
    "prometheus.yml",
    
    # IGNORES
    ".gitignore",
    ".dvcignore",
    ".dockerignore",
]

for file in list_of_files:
    file_path = Path(file)
    file_dir ,file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
    else:
        logging.info(f"{file_path} already exists")
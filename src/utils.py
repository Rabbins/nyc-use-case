import logging
import yaml
import shutil
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(name)

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_directory(path: Path):
    """Ensure the parent directory exists and creates it if not."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
def ensure_directories(paths: dict):
    """Ensure the parent directories exists and creates them if not."""
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)        
        
def clean_output_directory(path: Path):
    """Internal helper to ensure idempotency, basically cleaning up directories to avoid duplication of data"""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
        
def setup_session() -> requests.Session:
    """
    Configures a session with automatic retries and a User-Agent.
    This prevents the pipeline from crashing on temporary network issues.
    """
    session = requests.Session()
    
    # Define retry strategy: stop crashing on 500, 502, 503 errors
    retry_strategy = Retry(
        total=3,                # Total retry attempts
        backoff_factor=1,       # Wait 1s, 2s, 4s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # To avoid 403 Forbidden on public APIs
    session.headers.update({
        "User-Agent": "NYCCollisionETL/1.0"
    })
    
    return session        
import logging
import polars as pl
import yaml
import shutil
import time
from functools import wraps
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

# We generate a "local" logger for the following decorator
logger = setup_logger("Utils")

def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

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
    """Internal helper to ensure idempotency, cleaning up directories to avoid duplication of data"""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
        
def setup_session() -> requests.Session:
    """
    Configures a session with automatic retries and a User-Agent.
    This prevents the pipeline from crashing on temporary network issues.
    """
    session = requests.Session()
    
    # Define retry strategy: stop crashing on server errors
    retry_strategy = Retry(
        total=3,                # Total retry attempts
        backoff_factor=1,       # Wait 1s, 2s, 4s...
        status_forcelist=[500, 502, 503, 504],
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

def validate_dataframe(df: pl.DataFrame, name: str, critical_cols: list | None = None):
    """
    Simple data quality check. Makes sure DataFrame has data and "critical" columns are not null
    """
    logger.info(f"Running Data Quality check for '{name}'...")

    # 1. Empty check
    if df.height == 0:
        raise ValueError(f"Data Quality error: Input for '{name}' is empty")

    # 2. Null check on critical columns
    if critical_cols:
        for col in critical_cols:
            if col not in df.columns:
                 logger.warning(f"Skipping null check for missing column '{col}' in '{name}'")
                 continue
            
            null_count = df.select(pl.col(col).null_count()).item()
            if null_count > 0:
                logger.warning(f"DQ Warning: Column '{col}' in '{name}' has {null_count} nulls.")
                # If we wanted to stop the process:
                # raise ValueError(f"Data Quality error: column '{col}' contains nulls.")
    
    logger.info(f"DQ Check passed for '{name}'. Rows: {df.height}")
import requests
import json
from pathlib import Path
from typing import List, Tuple
from ..utils import setup_logger,setup_session,ensure_directory
import polars as pl

logger = setup_logger(__name__)

class BronzeExtractor:
    def __init__(self, config: dict):
        self.config = config
        
        # Create a session to reuse TCP connections and handle retries
        self.session = setup_session()



    def download_file_from_url(self, url: str, output_path: Path) -> Tuple[pl.DataFrame, Path]:
            """
            Downloads a file directly to disk and loads it into a dataframe.
            """
            logger.info(f"Downloading data from {url}...")
            
            ensure_directory(output_path)
            
            # 1. Check if cached
            if output_path.exists():
                logger.info(f"File found at {output_path}. Skipping download.")
            else:
                try:
                    # 2. Download and Write (Stream to disk to save RAM)
                    response = self.session.get(url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        # Write in chunks
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    logger.info(f"Saved to {output_path}")

                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error downloading {url}: {e}")
                    # Delete incomplete file if download failed
                    if output_path.exists():
                        output_path.unlink()
                    raise

            # 3. Read into Polars
            df = pl.read_csv(
                output_path,
                ignore_errors=True, 
                infer_schema_length=10000 
            )
                
            return df, output_path

    def fetch_holidays(self, base_url: str, country: str, years: List[int], output_path: Path) -> Tuple[pl.DataFrame,Path]:
            """
            Fetches holidays from API, saves them to disk (Bronze) and returns the DataFrame for processing.
            
            Returns:
                Tuple[Path, pl.DataFrame]: (Path to saved JSON, Polars DataFrame)
            """
            ensure_directory(output_path)
            
            all_holidays = []
            
            # Based on API design
            for year in years:
                url = f"{base_url}/{year}/{country}"
                logger.info(f"Fetching holidays for {year}...")
                
                try:
                    # Reuse session
                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()
                    
                    # Append data to our list (for every year)
                    all_holidays.extend(response.json())
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Could not fetch holidays for {year}: {e}")
            
            # Guard Clause
            if not all_holidays:
                raise RuntimeError("No holidays data fetched. Check API availability.")

            # 1. Save to disk (Bronze Layer)
            with open(output_path, 'w') as f:
                json.dump(all_holidays, f, indent=2)
            
            logger.info(f"Holidays persisted to {output_path}")

            # 2. Create Dataframe
            try:
                df = pl.DataFrame(all_holidays)
                return df,output_path
                
            except Exception as e:
                logger.error(f"Error converting holiday data to DataFrame: {e}")
                raise
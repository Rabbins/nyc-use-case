import sys
from pathlib import Path

from .utils import load_config, ensure_directories, setup_logger
from src.layers.bronze_processing import BronzeExtractor
from src.layers.silver_processing import SilverProcessor
from src.layers.gold_processing import GoldProcessor

logger = setup_logger("Pipeline")

def run_pipeline():
    try:
        # --- 0. SETUP ---
        config = load_config()
        # Ensure base directories exist (bronze, silver, gold)
        ensure_directories(config['paths'])

        # Dependency Injection
        bronze_processor = BronzeExtractor(config)
        silver_processor = SilverProcessor()
        gold_processor = GoldProcessor()

# --- PHASE 1: BRONZE (Ingestion) ---
        logger.info(" PHASE 1: INGESTION ")

        # I am working with both df and files in order to accelerate the processing (with df) and simulate a bronze>silver>gold architecture writing onto S3 for example        
        
        # 1.1 Ingest Collisions
        logger.info("Ingesting Collisions data...")
        
        # Config source variables
        collisions_url = config['sources']['collisions']['url']
        collisions_filename = config['sources']['collisions']['filename']
        collisions_output_path = Path(config['paths']['bronze']) / collisions_filename

        df_collisions_bronze, path_collisions_bronze = bronze_processor.download_file_from_url(
            url=collisions_url,
            output_path=collisions_output_path
        )
        
        # 1.2 Ingest Holidays
        logger.info("Ingesting Holidays data...")
        
        # Config source variables
        holidays_conf = config['sources']['holidays']
        holidays_output_path = Path(config['paths']['bronze']) / holidays_conf['filename']

        df_holidays_bronze, path_holidays_bronze = bronze_processor.fetch_holidays(
            base_url=holidays_conf['url_base'],
            country=holidays_conf['country_code'],
            years=holidays_conf['years'],
            output_path=holidays_output_path
        )
        
        # 1.3 Ingest historical NYC weather
        logger.info("Ingesting Collisions data...")
        
        # Config source variables
        weather_url = config['sources']['weather']['url']
        weather_filename = config['sources']['weather']['filename']
        weather_output_path = Path(config['paths']['bronze']) / weather_filename

        df_weather_bronze, path_weather_bronze = bronze_processor.download_file_from_url(
            url=weather_url,
            output_path=weather_output_path
        )        
        
# --- PHASE 2: SILVER (Transform & Standardize) ---
        logger.info(" PHASE 2: SILVER LAYER ")
        
        silver_base_path = Path(config['paths']['silver'])
        
        # 2.1 Process Collisions
        df_collisions_silver, path_collisions_silver = silver_processor.process_collisions(
            input_data=df_collisions_bronze, # path_collisions_bronze
            output_path=silver_base_path / "collisions"
        )

        # 2.2 Process Holidays        
        df_holidays_silver, path_holidays_silver = silver_processor.process_holidays(
            input_data=df_holidays_bronze, # path_holidays_bronze
            output_path=silver_base_path / "holidays"
        )
        
        # 2.3 Process Weather        
        df_weather_silver, path_weather_silver = silver_processor.process_weather(
            input_data=df_weather_bronze, # path_weather_bronze
            output_path=silver_base_path / "weather"
        )
        
# --- PHASE 3: GOLD ---
        logger.info(" PHASE 3: GOLD LAYER ")
        
        # I'll be passing df directly to keep data in memory. In a production environment this steps would be separate and we'd mnst likely read from a file 
                
        gold_processor.process_gold_data(
            collisions_data=df_collisions_silver,
            holidays_data=df_holidays_silver,
            weather_data=df_weather_silver,
            gold_base_path=Path(config['paths']['gold'])
        )

        logger.info("Pipeline Finished Successfully.")
        
    except Exception as e:
            logger.critical(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
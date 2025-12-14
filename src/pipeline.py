import sys
from pathlib import Path

from .utils import load_config, ensure_directories, setup_logger, validate_dataframe
from src.layers.bronze_processing import BronzeExtractor
from src.layers.silver_processing import SilverProcessor
from src.layers.gold_processing import GoldProcessor

logger = setup_logger("Pipeline")

def run_pipeline():
    try:
        # Initial config setup 
        config = load_config()
        
        # Ensure base directories exist (bronze, silver, gold) and create them if not
        ensure_directories(config['paths'])

        bronze_processor = BronzeExtractor(config)
        silver_processor = SilverProcessor(config)
        gold_processor = GoldProcessor() # could add config if it made sense

# BRONZE (Ingestion) 
        logger.info(" PHASE 1: INGESTION ")

        # I am working with both df and files in order to accelerate the processing (with df) and simulate a bronze>silver>gold architecture writing onto S3 for example        
        
        # Ingest Collisions
        logger.info("Ingesting Collisions data...")
        
        # Config source variables
        collisions_url = config['sources']['collisions']['url']
        collisions_filename = config['sources']['collisions']['filename']
        collisions_output_path = Path(config['paths']['bronze']) / collisions_filename

        df_collisions_bronze, path_collisions_bronze = bronze_processor.download_file_from_url(
            url=collisions_url,
            output_path=collisions_output_path
        )
        
        validate_dataframe(df_collisions_bronze, "Collisions Bronze", critical_cols=["CRASH DATE"])
        
        # Ingest Holidays
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
        
        validate_dataframe(df_holidays_bronze, "Holidays Bronze", critical_cols=["date"])
        
        # Ingest historical NYC weather
        logger.info("Ingesting Collisions data...")
        
        # Config source variables
        weather_url = config['sources']['weather']['url']
        weather_filename = config['sources']['weather']['filename']
        weather_output_path = Path(config['paths']['bronze']) / weather_filename

        df_weather_bronze, path_weather_bronze = bronze_processor.download_file_from_url(
            url=weather_url,
            output_path=weather_output_path
        )        
        
        validate_dataframe(df_weather_bronze, "Weather Bronze", critical_cols=["DATE"])
        
# SILVER (Transform & Standardize) 
        logger.info(" PHASE 2: SILVER LAYER ")
        
        silver_base_path = Path(config['paths']['silver'])
        
        # Process Collisions
        df_collisions_silver, path_collisions_silver = silver_processor.process_collisions(
            input_data=df_collisions_bronze, # path_collisions_bronze (if in production)
            output_path=silver_base_path / "collisions"
        )

        # Process Holidays        
        df_holidays_silver, path_holidays_silver = silver_processor.process_holidays(
            input_data=df_holidays_bronze, # path_holidays_bronze (if in production)
            output_path=silver_base_path / "holidays"
        )
        
        # Process Weather        
        df_weather_silver, path_weather_silver = silver_processor.process_weather(
            input_data=df_weather_bronze, # path_weather_bronze (if in production)
            output_path=silver_base_path / "weather"
        )
        
# GOLD (Data modeling) 
        logger.info(" PHASE 3: GOLD LAYER ")
        
        # I'll be passing DataFrame directly to keep data in memory. In a production environment this steps would be separate and we'd mnst likely read from a file 
                
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
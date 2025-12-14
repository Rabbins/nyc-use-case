import sys
import time
import argparse
import logging
from dotenv import load_dotenv

from src.pipeline import run_pipeline
from src.utils import setup_logger

# Inicializamos logger global para el main
logger = setup_logger("Main")

def parse_args():
    parser = argparse.ArgumentParser(description="NYC Collisions ETL Pipeline")
    
    parser.add_argument(
        "--env", 
        type=str, 
        default="dev", 
        choices=["dev", "prod"],
        help="Environment execution context (default: dev)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    start_time = time.time()
    
    # Load environment variables (AWS credentials, DB URLs, etc.)
    load_dotenv()
    
    args = parse_args()

    # Adjust logging level based on CLI flags
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else: 
        logger.info("Info logger mode by default.")

    
    logger.info(f"Starting ETL Pipeline [Env: {args.env.upper()}]")

    try:
        # We could pass args.env for different configs but won't do it to make the project easier
        run_pipeline()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline finished successfully in {elapsed_time:.2f} seconds.")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
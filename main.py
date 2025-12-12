import sys
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Importamos nuestro pipeline
from src.pipeline import run_pipeline
from src.utils import setup_logger

# Configuración inicial del logger para el entry point
logger = setup_logger("Entrypoint")

def parse_args():
    """
    Parsea los argumentos de línea de comandos.
    Esto permite ejecutar el script con opciones como:
    python main.py --env prod --verbose
    """
    parser = argparse.ArgumentParser(description="NYC Collisions ETL Pipeline")
    
    parser.add_argument(
        "--env", 
        type=str, 
        default="dev", 
        choices=["dev", "prod"],
        help="Environment to run the pipeline in (default: dev)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Increase output verbosity to DEBUG"
    )
    
    return parser.parse_args()

def main():
    """
    Función principal de orquestación.
    """
    # 1. Medición de tiempo (KPI básico de ingeniería)
    start_time = time.time()
    
    # 2. Parseo de argumentos
    args = parse_args()
    
    # 3. Configuración del entorno
    # Cargar variables de entorno desde .env si existe
    load_dotenv()
    
    # Ajustar nivel de log según el flag --verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    
    logger.info(f"Starting ETL Pipeline in environment: {args.env.upper()}")

    # 4. Ejecución Segura
    try:
        # Aquí lanzamos el pipeline. 
        # Nota: Podrías pasar 'args.env' a run_pipeline si tu config soporta entornos.
        run_pipeline()
        
        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")
        
        # Salida exitosa (Código 0)
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user.")
        sys.exit(130)
        
    except Exception as e:
        # 5. Catch-all para errores no controlados
        # Es vital loguear el error y SALIR CON ERROR (sys.exit(1))
        # para que Airflow/Jenkins marquen la tarea como FAILED.
        logger.critical(f"Pipeline failed with critical error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
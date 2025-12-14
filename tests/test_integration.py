import pytest
import polars as pl
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

# Asegúrate de que Python encuentre tu módulo
from src.pipeline import run_pipeline

# --- 1. MOCK DE CONFIGURACIÓN ---
# Reemplazamos el 'config.yaml' real por uno falso que apunta a la carpeta temporal de Pytest (tmp_path).

@pytest.fixture
def mock_config(tmp_path):
    # Creamos las rutas dentro del entorno aislado del test
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    logs = tmp_path / "logs"
    
    return {
        "paths": {
            "bronze": str(bronze),
            "silver": str(silver),
            "gold": str(gold),
            "logs": str(logs)
        },
        "sources": {
            "collisions": {
                "url": "http://mock-api/collisions.csv",
                "filename": "collisions_raw.csv"
            },
            "holidays": {
                "url_base": "http://mock-api/holidays",
                "country_code": "US",
                "years": [2024],
                "filename": "holidays_raw.json"
            },
            "weather": {
                "url": "http://mock-api/weather.csv",
                "filename": "weather_raw.csv"
            }
        },
        "silver": {
            "collisions": {
                "rename_map": {
                    "CRASH DATE": "crash_date",
                    "CRASH TIME": "crash_time",
                    "BOROUGH": "borough",
                    "ZIP CODE": "zip_code",
                    "NUMBER OF PERSONS INJURED": "number_of_persons_injured",
                    "NUMBER OF PERSONS KILLED": "number_of_persons_killed",
                    "NUMBER OF PEDESTRIANS INJURED": "number_of_pedestrians_injured",
                    "NUMBER OF PEDESTRIANS KILLED": "number_of_pedestrians_killed",
                    "NUMBER OF CYCLIST INJURED": "number_of_cyclist_injured",
                    "NUMBER OF CYCLIST KILLED": "number_of_cyclist_killed",
                    "NUMBER OF MOTORIST INJURED": "number_of_motorist_injured",
                    "NUMBER OF MOTORIST KILLED": "number_of_motorist_killed",
                    "CONTRIBUTING FACTOR VEHICLE 1": "contributing_factor_vehicle_1"
                },
                "metric_cols": [
                    "number_of_persons_injured", 
                    "number_of_persons_killed",
                    "number_of_persons_injured",
                    "number_of_persons_killed",
                    "number_of_pedestrians_injured",
                    "number_of_pedestrians_killed",
                    "number_of_cyclist_injured",
                    "number_of_cyclist_killed",
                    "number_of_motorist_injured",
                    "number_of_motorist_killed"  
                ]
            }
        }
    }

# --- 2. MOCK DE RED (REQUESTS) ---
# Simulamos los datos crudos que vendrían de internet.
# Debemos tener cuidado de incluir las columnas que SILVER espera.

@pytest.fixture
def mock_network(mock_config):
    with patch('requests.Session.get') as mock_get:
        
        # A. Datos Dummy para Collisions (CSV)
        # Usamos un string normal y luego .encode('utf-8') para evitar errores de sintaxis
        # Aseguramos que estén TODAS las columnas que Silver espera en el rename_map
        csv_text = (
            "CRASH DATE,CRASH TIME,BOROUGH,ZIP CODE,NUMBER OF PERSONS INJURED,"
            "NUMBER OF PERSONS KILLED,NUMBER OF PEDESTRIANS INJURED,"
            "NUMBER OF PEDESTRIANS KILLED,NUMBER OF CYCLIST INJURED,"
            "NUMBER OF CYCLIST KILLED,NUMBER OF MOTORIST INJURED,"
            "NUMBER OF MOTORIST KILLED,CONTRIBUTING FACTOR VEHICLE 1\n"
            "01/01/2024,14:30,MANHATTAN,10001,1,0,0,0,0,0,1,0,Unspecified"
        )
        resp_collisions = MagicMock()
        resp_collisions.status_code = 200
        # iter_content debe devolver bytes
        resp_collisions.iter_content.return_value = [csv_text.encode('utf-8')]

        # B. Datos Dummy para Weather (CSV)
        # Silver espera: DATE, TMAX, TMIN, PRCP, SNOW, WT01, WT02
        weather_text = (
            "DATE,TMAX,TMIN,PRCP,SNOW,WT01,WT02\n"
            "2024-01-01,100,50,0,0,0,0"
        )
        resp_weather = MagicMock()
        resp_weather.status_code = 200
        resp_weather.iter_content.return_value = [weather_text.encode('utf-8')]

        # C. Datos Dummy para Holidays (JSON)
        # Silver espera: date, name, types
        json_holidays = [
            {"date": "2024-01-01", "name": "New Year", "types": ["Public"]}
        ]
        resp_holidays = MagicMock()
        resp_holidays.status_code = 200
        resp_holidays.json.return_value = json_holidays

        # Lógica de enrutamiento
        def side_effect(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            
            if "collisions" in url:
                return resp_collisions
            elif "weather" in url:
                return resp_weather
            elif "holidays" in url:
                return resp_holidays
            return MagicMock(status_code=404)

        mock_get.side_effect = side_effect
        yield mock_get

# --- 3. EL TEST DE INTEGRACIÓN ---

def test_pipeline_end_to_end(mock_config, mock_network, tmp_path):
    """
    Ejecuta el Pipeline completo (Bronze -> Silver -> Gold).
    Verifica que se generen los archivos finales.
    """
    
    # IMPORTANTE: Patcheamos 'load_config' donde se IMPORTA en pipeline.py
    # Si pipeline.py hace "from .utils import load_config",
    # debemos patchear 'src.pipeline.load_config'
    with patch('src.pipeline.load_config', return_value=mock_config):
        
        # --- ACT: Ejecutar Pipeline ---
        print("\n🚀 Iniciando Test de Integración...")
        run_pipeline()
        print("✅ Pipeline finalizado.")

        # --- ASSERT: Verificar Resultados ---
        
        # 1. Verificar directorios creados
        gold_path = Path(mock_config['paths']['gold'])
        assert gold_path.exists()
        
        # 2. Verificar archivos en Gold
        parquet_file = gold_path / "daily_stats.parquet"
        csv_file = gold_path / "daily_stats.csv"
        
        assert parquet_file.exists(), "Falta el archivo Parquet en Gold"
        assert csv_file.exists(), "Falta el archivo CSV en Gold"
        
        # 3. Verificar contenido básico (Smoke Test)
        df = pl.read_parquet(parquet_file)
        assert len(df) > 0, "El dataset final está vacío"
        assert "total_accidents" in df.columns
        
        # Verificar que el dato mockeado llegó hasta el final
        # Enviamos 1 accidente en collisions y clima correcto.
        row = df.row(0, named=True)
        assert row['total_accidents'] == 1
        assert row['max_temp'] == 10.0 # 100 / 10
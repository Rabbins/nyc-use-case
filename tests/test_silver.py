import pytest
import polars as pl
from datetime import date
from unittest.mock import patch
from src.layers.silver_processing import SilverProcessor

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """
    Simulates the necessary YAML config for Silver.
    """
    return {
        "silver": {
            "collisions": {
                "rename_map": {
                    "CRASH DATE": "crash_date",
                    "CRASH TIME": "crash_time",
                    "BOROUGH": "borough",
                    "ZIP CODE": "zip_code",
                    "NUMBER OF PERSONS INJURED": "number_of_persons_injured",
                    "NUMBER OF PERSONS KILLED": "number_of_persons_killed",
                    "CONTRIBUTING FACTOR VEHICLE 1": "contributing_factor_vehicle_1"
                },
                "metric_cols": [
                    "number_of_persons_injured", 
                    "number_of_persons_killed"
                ]
            }
        }
    }

@pytest.fixture
def silver_processor(mock_config):
    """Instantiates SilverProcessor injecting the mocked config."""
    return SilverProcessor(mock_config)

@pytest.fixture
def mock_utils():
    """Mock to prevent deleting real folders."""
    with patch('src.layers.silver_processing.clean_output_directory') as mock_clean:
        yield mock_clean

# --- Tests: Collisions ---

def test_process_collisions_logic(silver_processor, mock_utils, mock_config, tmp_path):
    """
    Verifies renaming, null cleaning, and date creation.
    Does NOT filter by old years (only nulls).
    """
    
    rename_map = mock_config['silver']['collisions']['rename_map']
    
    # Arrange: Create 'dirty' data in memory
    input_df = pl.DataFrame({
        "CRASH DATE": [
            "12/31/2023", # OK
            "01/01/2024", # OK
            "01/01/2019", # OK
            None          # NULL (This must be removed)
        ], 
        "BOROUGH": ["MANHATTAN", None, "QUEENS", "BRONX"], 
        "NUMBER OF PERSONS INJURED": ["1", None, "0", "0"], 
        "CONTRIBUTING FACTOR VEHICLE 1": ["Alcohol", "Speeding", "N/A", "N/A"]
    }).with_columns([
        # Generate dummy columns for the rest of the map keys
        pl.lit(0).alias(col) for col in rename_map.keys() 
        if col not in ["CRASH DATE", "BOROUGH", "NUMBER OF PERSONS INJURED", "CONTRIBUTING FACTOR VEHICLE 1"]
    ])
    
    output_path = tmp_path / "collisions_silver"

    # Act
    df_result, path = silver_processor.process_collisions(input_df, output_path)

    # Assert
    # We expect 3 rows.
    # None is removed, but 2019, 2023, and 2024 remain.
    assert len(df_result) == 3
    
    # Verify renaming
    assert "crash_date" in df_result.columns
    assert "number_of_persons_injured" in df_result.columns
    
    # Verify null cleaning (row index 1 is 01/01/2024)
    assert df_result["borough"][1] == "UNKNOWN"
    assert df_result["number_of_persons_injured"][1] == 0
    
    # Verify dates (row index 0 is 12/31/2023)
    assert df_result["date"][0] == date(2023, 12, 31)
    assert df_result["year"][0] == 2023
    assert df_result["is_weekend"][0] == True 

# --- Tests: Holidays ---

def test_process_holidays_deduplication(silver_processor, mock_utils, tmp_path):
    # Arrange
    input_df = pl.DataFrame([
        {"date": "2024-01-01", "name": "New Year", "types": ["National"]},
        {"date": "2024-01-01", "name": "New Year", "types": ["National"]}, 
        {"date": "2024-12-25", "name": "Christmas", "types": ["National", "Christian"]}
    ])
    output_path = tmp_path / "holidays_silver"

    # Act
    df_result, _ = silver_processor.process_holidays(input_df, output_path)

    # Assert
    assert len(df_result) == 2 
    
    # Sort by date before verifying
    df_sorted = df_result.sort("date")
    
    assert df_sorted["holiday_name"][0] == "New Year"
    assert df_sorted["holiday_name"][1] == "Christmas"

# --- Tests: Weather ---

def test_process_weather_logic(silver_processor, mock_utils, tmp_path):
    """
    Critical test: verify unit conversion and flags.
    """
    
    # Arrange
    input_df = pl.DataFrame({
        "DATE": ["2019-12-31", "2024-01-01", "2024-01-02"], 
        "TMAX": ["100", "250", "-50"],       
        "TMIN": ["0", "150", "-100"],        
        "PRCP": ["0", "50", "0"],            
        "SNOW": ["0", "0", "10"],            
        "WT01": ["0", "1", "0"],             
        "WT02": ["0", "0", "1"]              
    })
    
    output_path = tmp_path / "weather_silver"

    # Act
    df_result, _ = silver_processor.process_weather(input_df, output_path)

    # Assert
    # Date filter for weather (assuming this filter is maintained)
    assert len(df_result) == 2
    
    # Check Row 1 (2024-01-01)
    row1 = df_result.filter(pl.col("date") == date(2024, 1, 1))
    assert row1["temp_max_c"][0] == 25.0  
    assert row1["precipitation_mm"][0] == 5.0 
    assert row1["is_foggy"][0] == True   
    assert row1["has_rain"][0] == True   
    assert row1["has_snow"][0] == False
    
    # Check Row 2 (2024-01-02)
    row2 = df_result.filter(pl.col("date") == date(2024, 1, 2))
    assert row2["temp_max_c"][0] == -5.0  
    assert row2["is_foggy"][0] == True    
    assert row2["has_snow"][0] == True    

def test_process_weather_unsupported_input(silver_processor):
    """Verifies that it raises error for unsupported input types."""
    with pytest.raises(TypeError):
        silver_processor.process_weather(12345, "dummy_path")
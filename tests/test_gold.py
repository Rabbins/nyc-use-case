import pytest
import polars as pl
from pathlib import Path
from datetime import date
from unittest.mock import patch

from src.layers import gold_processing
from src.layers.gold_processing import GoldProcessor

# --- Fixtures ---

@pytest.fixture
def gold_processor():
    return GoldProcessor()

@pytest.fixture
def mock_clean_dir():
    """Mock to prevent deleting real directories during tests."""
    with patch.object(gold_processing, 'clean_output_directory') as mock:
        yield mock

# --- Helpers with Explicit Schemas (for Polars) ---

def get_sample_collisions():
    """Creates dummy collision data with a strict schema matching Silver output."""
    data = {
        "date": [date(2024, 1, 1), date(2024, 1, 1), date(2019, 12, 31)], 
        "borough": ["MANHATTAN", "MANHATTAN", "QUEENS"],
        "zip_code": ["10001", "10001", "11101"],
        "number_of_persons_injured": [1, 2, 0],
        "is_weekend": [False, False, False],
        # Fill the rest of the metrics with 0
        "number_of_persons_killed": [0, 0, 0],
        "number_of_pedestrians_injured": [0, 0, 0],
        "number_of_pedestrians_killed": [0, 0, 0],
        "number_of_cyclist_injured": [0, 0, 0],
        "number_of_cyclist_killed": [0, 0, 0],
        "number_of_motorist_injured": [0, 0, 0],
        "number_of_motorist_killed": [0, 0, 0]
    }
    # Force Date type to avoid inference issues
    return pl.DataFrame(data, schema_overrides={"date": pl.Date})

def get_sample_holidays():
    """Creates dummy holiday data with strict list schema."""
    data = {
        "date": [date(2024, 1, 1)],
        "holiday_name": ["New Year"],
        "types": [["National", "Public"]] # List of strings
    }
    # Important: define that 'types' is a List of Strings
    schema = {
        "date": pl.Date, 
        "holiday_name": pl.String, 
        "types": pl.List(pl.String)
    }
    return pl.DataFrame(data, schema=schema)

def get_sample_weather():
    """Creates dummy weather data with all necessary columns."""
    data = {
        "date": [date(2024, 1, 1)],
        "temp_max_c": [10.5],
        "temp_min_c": [5.0],
        "has_rain": [True],
        "has_snow": [False],
        "is_foggy": [False]
    }
    return pl.DataFrame(data, schema_overrides={"date": pl.Date})

# --- Tests: enrichment logic ---

def test_enrich_collisions_logic(gold_processor):
    """
    Verifies joins, date filtering, and classification logic.
    """
    # Arrange
    lf_col = get_sample_collisions().lazy()
    lf_hol = get_sample_holidays().lazy()
    lf_wea = get_sample_weather().lazy()

    # Act
    df_result = gold_processor._enrich_collisions(lf_col, lf_hol, lf_wea)

    # Assert
    # Date Filter: The 2019 record must disappear
    assert len(df_result) == 2 
    
    # Joins & Holiday Logic
    # The two remaining rows are from 2024-01-01
    row = df_result.row(0, named=True)
    
    # Verify join with holidays
    assert row["holiday_name"] == "New Year"
    assert row["high_impact_holiday"] == True   # Because "types" contains "Public"
    
    # Verify join with weather
    assert row["max_temp"] == 10.5
    assert row["has_rain"] == True

def test_enrich_collisions_missing_weather(gold_processor):
    """
    Verifies that the Left Join works when there is NO weather data.
    """
    # Arrange
    lf_col = pl.DataFrame({"date": [date(2024, 6, 1)]}, schema={"date": pl.Date}).lazy()
    
    # Empty holidays (correct schema)
    lf_hol = pl.DataFrame(schema={
        "date": pl.Date, "holiday_name": pl.String, "types": pl.List(pl.String)
    }).lazy()
    
    # Empty weather (must have all columns that the code selects)
    lf_wea = pl.DataFrame(schema={
        "date": pl.Date, 
        "temp_max_c": pl.Float64, 
        "temp_min_c": pl.Float64,
        "has_rain": pl.Boolean,
        "has_snow": pl.Boolean,
        "is_foggy": pl.Boolean
    }).lazy()

    # Act
    df_result = gold_processor._enrich_collisions(lf_col, lf_hol, lf_wea)

    # Assert
    row = df_result.row(0, named=True)
    
    assert row["date"] == date(2024, 6, 1)
    # Verify that fill_null(False) worked
    assert row["has_rain"] == False 
    assert row["high_impact_holiday"] == False
    # Temperatures should remain null
    assert row["max_temp"] is None

# --- Tests: aggregation logic ---

def test_aggregate_stats_math(gold_processor):
    """
    Verifies that aggregation correctly sums the metrics.
    """
    # Arrange
    # Simulate a manually enriched DF
    enriched_df = pl.DataFrame({
        "date": [date(2024, 1, 1), date(2024, 1, 1)],
        "borough": ["MANHATTAN", "MANHATTAN"],
        "zip_code": ["10001", "10001"],
        "is_weekend": [False, False],
        "holiday_name": ["Ny", "Ny"],
        "high_impact_holiday": [True, True],
        "partial_impact_holiday": [False, False],
        "low_impact_holiday": [False, False],
        "has_rain": [False, False],
        "has_snow": [False, False],
        "is_foggy": [False, False],
        "max_temp": [10.0, 10.0],
        "min_temp": [5.0, 5.0],
        # Metrics
        "number_of_persons_injured": [1, 2],
        "number_of_persons_killed": [0, 0],
        "number_of_pedestrians_injured": [0, 0],
        "number_of_pedestrians_killed": [0, 0],
        "number_of_cyclist_injured": [0, 0],
        "number_of_cyclist_killed": [0, 0],
        "number_of_motorist_injured": [0, 0],
        "number_of_motorist_killed": [0, 0]
    }, schema_overrides={"date": pl.Date})

    # Act
    df_gold = gold_processor._aggregate_stats(enriched_df)

    # Assert
    assert len(df_gold) == 1
    row = df_gold.row(0, named=True)
    assert row["total_accidents"] == 2
    assert row["number_of_persons_injured"] == 3

# --- Tests: End-to-End ---

def test_process_gold_data_integration(gold_processor, mock_clean_dir, tmp_path):
    """
    Tests that the main method orchestrates everything and writes files.
    """
    # Arrange
    col_df = get_sample_collisions()
    hol_df = get_sample_holidays()
    wea_df = get_sample_weather()
    
    output_dir = tmp_path / "gold_output"
    
    # Manually create the directory because 'clean_output_directory' is mocked
    output_dir.mkdir() 

    # Act
    gold_processor.process_gold_data(col_df, hol_df, wea_df, output_dir)

    # Assert
    mock_clean_dir.assert_called_once_with(output_dir)
    assert (output_dir / "daily_stats.parquet").exists()
    assert (output_dir / "daily_stats.csv").exists()
    
    df_check = pl.read_parquet(output_dir / "daily_stats.parquet")
    assert len(df_check) > 0

def test_normalize_input_path(gold_processor):
    """Unit test for _normalize_input with Path."""
    with patch("polars.scan_parquet") as mock_scan:
        path = Path("/dummy/path")
        gold_processor._normalize_input(path, "test")
        
        expected_call = str(path / "**/*.parquet")
        mock_scan.assert_called_once_with(expected_call)
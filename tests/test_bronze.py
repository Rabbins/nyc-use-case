import pytest
import requests
from unittest.mock import MagicMock, patch, Mock
from src.layers.bronze_processing import BronzeExtractor 

# --- Fixtures ---

@pytest.fixture
def mock_session():
    """Creates a mock of request session"""
    session = MagicMock()
    return session

@pytest.fixture
def bronze_extractor(mock_session):
    """Instantiates the BronzeExtractor class with mocked dependencies."""
    config = {"dummy": "config"}
    
    # Patch setup_session to return our mock_session and ensure_directory to do nothing (or let it pass if using tmp_path)
    with patch('src.layers.bronze_processing.setup_session', return_value=mock_session), \
         patch('src.layers.bronze_processing.ensure_directory'):
        extractor = BronzeExtractor(config)
        return extractor

# --- Tests for download_file_from_url ---

def test_download_file_success(bronze_extractor, mock_session, tmp_path):
    """Tests a successful CSV download."""
    
    # Arrange
    url = "http://fake-url.com/data.csv"
    output_path = tmp_path / "data.csv" # Use temporary folder
    
    # Simulate CSV content in bytes
    csv_content = b"col1,col2\n1,A\n2,B"
    
    # Configure the response mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [csv_content] # Simulates chunks
    mock_session.get.return_value = mock_response

    # Act
    df, path = bronze_extractor.download_file_from_url(url, output_path)

    # Assert
    # Verify that the correct URL was called
    mock_session.get.assert_called_once_with(url, stream=True, timeout=60)
    
    # Verify that the file exists
    assert path.exists()
    
    # Verify DataFrame content
    assert df.shape == (2, 2)
    assert df["col1"][0] == 1
    assert df["col2"][1] == "B"

def test_download_file_skips_if_cached(bronze_extractor, mock_session, tmp_path):
    """Tests that it does NOT download if the file already exists."""
    
    # Arrange
    url = "http://fake-url.com/data.csv"
    output_path = tmp_path / "cached_data.csv"
    
    # Create the "fake" file before calling the function
    with open(output_path, "w") as f:
        f.write("col1\n999")
        
    # Act
    df, path = bronze_extractor.download_file_from_url(url, output_path)
    
    # Assert
    # IMPORTANT: session.get must NOT have been called
    mock_session.get.assert_not_called()
    assert df["col1"][0] == 999

def test_download_file_network_error_cleanup(bronze_extractor, mock_session, tmp_path):
    """Tests that if the download fails, the incomplete file is deleted and an error is raised."""
    
    # Arrange
    url = "http://fake-url.com/fail.csv"
    output_path = tmp_path / "fail.csv"
    
    # Make requests raise an exception
    mock_session.get.side_effect = requests.exceptions.RequestException("Boom!")
    
    # Act & Assert
    with pytest.raises(requests.exceptions.RequestException):
        bronze_extractor.download_file_from_url(url, output_path)
    
    # Verify that cleanup was attempted (file should not exist)
    assert not output_path.exists()

# --- Tests for fetch_holidays ---

def test_fetch_holidays_success(bronze_extractor, mock_session, tmp_path):
    """Tests downloading holidays for multiple years."""
    
    # Arrange
    base_url = "http://api.holidays"
    country = "ES"
    years = [2023, 2024]
    output_path = tmp_path / "holidays.json"
    
    # Simulate different responses for each year
    data_2023 = [{"date": "2023-01-01", "name": "New Year"}]
    data_2024 = [{"date": "2024-01-01", "name": "New Year"}]
    
    # Side effect allows returning different values in each consecutive call
    mock_resp_2023 = Mock()
    mock_resp_2023.json.return_value = data_2023
    mock_resp_2023.status_code = 200
    
    mock_resp_2024 = Mock()
    mock_resp_2024.json.return_value = data_2024
    mock_resp_2024.status_code = 200
    
    mock_session.get.side_effect = [mock_resp_2023, mock_resp_2024]

    # Act
    df, path = bronze_extractor.fetch_holidays(base_url, country, years, output_path)

    # Assert
    assert mock_session.get.call_count == 2
    assert path.exists()
    
    # The final DF must have 2 rows (one from 2023 and another from 2024)
    assert len(df) == 2
    assert "date" in df.columns

def test_fetch_holidays_empty_raises_error(bronze_extractor, mock_session, tmp_path):
    """Tests that RuntimeError is raised if no data is fetched."""
    
    # Arrange
    output_path = tmp_path / "empty.json"
    
    # Simulate that the API fails or returns empty
    mock_session.get.side_effect = requests.exceptions.RequestException("API Down")
    
    # Act & Assert
    with pytest.raises(RuntimeError, match="No holidays data fetched"):
        bronze_extractor.fetch_holidays("url", "US", [2023], output_path)
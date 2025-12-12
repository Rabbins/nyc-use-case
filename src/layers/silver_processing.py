import shutil
import polars as pl
# import pyarrow
from pathlib import Path
from typing import Union, Tuple
from ..utils import setup_logger, clean_output_directory

logger = setup_logger(__name__)

class SilverProcessor:
    """
    Handles the Silver Layer lifecycle:
    1. Ingests Bronze Data (Path or DataFrame)
    2. Transforms & Standardizes
    3. Persists to Disk (in a partitioned Parquet)
    4. Returns Data for the next layer
    """
    
    # Column selection and renaming
    COLLISION_RENAME_MAP = {
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
    }

    METRIC_COLS = [
        "number_of_persons_injured", "number_of_persons_killed",
        "number_of_pedestrians_injured", "number_of_pedestrians_killed",
        "number_of_cyclist_injured", "number_of_cyclist_killed",
        "number_of_motorist_injured", "number_of_motorist_killed"
    ]
    def process_collisions(
        self, 
        input_data: Union[pl.DataFrame, Path], 
        output_path: Path
    ) -> Tuple[pl.DataFrame, Path]:
        """
        Standardizes collisions, wirtes to Silver (Partitioned) and returns the DataFrame.
        """
        
        logger.info("Processing Collisions (Bronze -> Silver)...")
        
        # 1. Load Input to LazyFrame
        if isinstance(input_data, Path):
            lf = pl.scan_csv(input_data, ignore_errors=True)
        elif isinstance(input_data, pl.DataFrame):
            lf = input_data.lazy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # 2. Transform
        q = (
            lf
            .filter(pl.col("CRASH DATE").is_not_null())
            .rename(self.COLLISION_RENAME_MAP)
            .select(list(self.COLLISION_RENAME_MAP.values()))
            .with_columns([
                pl.col("crash_date").str.to_date("%m/%d/%Y").alias("date"),
                pl.col("borough").fill_null("UNKNOWN"),
                # clean up null metrics with 0s
                pl.col(self.METRIC_COLS).fill_null(0).cast(pl.Int32)
            ])
            .with_columns([
                # is_weekend column
                (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
                # Partition Keys for writing into disk
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
        )
        
        # 3. Materialize (We need the DataFrame to persist it)
        df_silver = q.collect()

        # 4. Write to disk
        logger.info(f"Persisting Collisions Silver Layer to {output_path}...")
        clean_output_directory(output_path)
        
        df_silver.write_parquet(
            output_path,
            partition_by=["year", "month"]
        )

        return df_silver, output_path

    def process_holidays(
        self, 
        input_data: Union[pl.DataFrame, Path], 
        output_path: Path
    ) -> Tuple[pl.DataFrame, Path]:
        """
        Standardizes holidays, writes to Silver and returns the DataFrame.
        """
        
        logger.info("Processing Holidays (Bronze -> Silver)...")

        # 1. Load Input to LazyFrame
        if isinstance(input_data, Path):
            lf = pl.read_json(input_data).lazy()
        elif isinstance(input_data, pl.DataFrame):
            lf = input_data.lazy()
        else:
             raise TypeError(f"Unsupported input type: {type(input_data)}")

        # 2. Transform
        q = (
            lf
            .select([
                pl.col("date").str.to_date("%Y-%m-%d"),
                pl.col("name").alias("holiday_name"),
                pl.col("types").cast(pl.List(pl.String))
            ])
            .with_columns([
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
            .unique()
        )
        
        # 3. Materialize
        df_silver = q.collect()

        # 4. Write
        logger.info(f"Persisting Holidays Silver Layer to {output_path}...")
        clean_output_directory(output_path)
        
        df_silver.write_parquet(
            output_path,
            partition_by=["year"]
        )

        return df_silver, output_path
    
    
    def process_weather(
            self, 
            input_data: Union[pl.DataFrame, Path], 
            output_path: Path
        ) -> Tuple[pl.DataFrame, Path]:
            """
            Standardizes NOAA GHCN-Daily weather data, writes to Silver and returns the DataFrame.
            """
            logger.info("Processing Weather Data (Bronze -> Silver)...")
            
            # 1. Load Input to LazyFrame
            if isinstance(input_data, Path):
                # 'infer_schema_length=0' we read all cols as String first to avoid errors with messy CSVs
                lf = pl.scan_csv(input_data, ignore_errors=True, infer_schema_length=0)
            elif isinstance(input_data, pl.DataFrame):
                lf = input_data.lazy()
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")

            # 2. Transformations
            # NOAA GHCN-Daily Units:
            # TMAX/TMIN = Tenths of degrees C (e.g., 250 = 25.0°C)
            # PRCP = Tenths of mm (e.g., 50 = 5.0 mm)
            # SNOW = mm (whole numbers)
            # WT** Columns = "1" if event occurred, null otherwise.

            q = (
                lf
                .filter(pl.col("DATE") >= "2020-01-01")
                .filter(pl.col("DATE").is_not_null())
                .with_columns([
                    # Convert DATE string to Date object
                    pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"),
                    
                    # Convert Temperature (Tenths of C -> C)
                    (pl.col("TMAX").str.strip_chars().cast(pl.Float64) / 10).round(1).alias("temp_max_c"),
                    (pl.col("TMIN").str.strip_chars().cast(pl.Float64) / 10).round(1).alias("temp_min_c"),

                    # Convert Precipitation (Tenths of mm -> mm)
                    (pl.col("PRCP").str.strip_chars().cast(pl.Float64) / 10).alias("precipitation_mm"),

                    # Snow is already in mm, just cast it
                    pl.col("SNOW").str.strip_chars().cast(pl.Float64).alias("snow_mm"),

                    # --- Extract Weather Events (Fog, Rain, Snow) ---
                    # WT01 = Fog, ice fog, or freezing fog
                    # WT02 = Heavy fog
                    # Logic: If either WT01 or WT02 is "1", it was foggy.
                    pl.any_horizontal(
                        pl.col("WT01").str.strip_chars().cast(pl.Int32, strict=False) == 1,
                        pl.col("WT02").str.strip_chars().cast(pl.Int32, strict=False) == 1
                    ).fill_null(False).alias("is_foggy"),

                    # Basic Boolean flags for Rain/Snow based on measurements
                    (pl.col("PRCP").str.strip_chars().cast(pl.Float64) > 0).fill_null(False).alias("has_rain"),
                    (pl.col("SNOW").str.strip_chars().cast(pl.Float64) > 0).fill_null(False).alias("has_snow")
                ])
                # Select only the clean columns we want to keep
                .select([
                    "date", 
                    "temp_max_c", 
                    "temp_min_c", 
                    "precipitation_mm", 
                    "snow_mm",
                    "is_foggy", 
                    "has_rain", 
                    "has_snow"
                ])
                .with_columns([
                    # Partition Keys (Year/Month)
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month")
                ])
            )
            
            # 3. Materialize 
            df_silver = q.collect()

            # 4. Write to Disk (Partitioned)
            logger.info(f"Persisting Weather Silver Layer to {output_path}...")
            clean_output_directory(output_path)
            
            df_silver.write_parquet(
                output_path,
                partition_by=["year", "month"]
            )

            return df_silver, output_path
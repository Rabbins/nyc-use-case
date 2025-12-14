import polars as pl
from pathlib import Path
from typing import Union, Tuple
from ..utils import setup_logger, clean_output_directory,time_execution

logger = setup_logger(__name__)

class SilverProcessor:
    """
    Handles the Silver Layer lifecycle:
    1. Ingests Bronze Data (Path or DataFrame)
    2. Transforms & Standardizes
    3. Persists to Disk (in a partitioned Parquet)
    4. Returns Data for the next layer
    """
    
    def __init__(self, config: dict):
        # Load Silver config
        self.rename_map = config['silver']['collisions']['rename_map']
        self.metric_cols = config['silver']['collisions']['metric_cols']
    
    @time_execution
    def process_collisions(
        self, 
        input_data: Union[pl.DataFrame, Path], 
        output_path: Path
    ) -> Tuple[pl.DataFrame, Path]:
        """
        Standardizes collisions, wirtes to Silver (partitioned) and returns the DataFrame
        """
        
        logger.info("Processing Collisions (Bronze -> Silver)...")
        
        # Check type and load to LazyFrame (could be migrated to a utils.py method)
        if isinstance(input_data, Path):
            lf = pl.scan_csv(input_data, ignore_errors=True)
        elif isinstance(input_data, pl.DataFrame):
            lf = input_data.lazy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # Transform
        q = (
            lf
            .filter(pl.col("CRASH DATE").is_not_null())
            .rename(self.rename_map)
            .select(list(self.rename_map.values()))
            .with_columns([
                pl.col("crash_date").str.to_date("%m/%d/%Y").alias("date"),
                pl.col("borough").fill_null("UNKNOWN"),
                # clean up null metrics with 0s
                pl.col(self.metric_cols).fill_null(0).cast(pl.Int32)
            ])
            .with_columns([
                # is_weekend column
                (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
                # Partition keys for writing into disk
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
        )
        
        # Materialize (we need the DataFrame to persist it)
        df_silver = q.collect()

        # Write to disk
        logger.info(f"Persisting Collisions Silver Layer to {output_path}...")
        clean_output_directory(output_path)
        
        df_silver.write_parquet(
            output_path,
            partition_by=["year", "month"]
        )

        return df_silver, output_path
    
    @time_execution
    def process_holidays(
        self, 
        input_data: Union[pl.DataFrame, Path], 
        output_path: Path
    ) -> Tuple[pl.DataFrame, Path]:
        """
        Standardizes holidays, writes to Silver and returns the DataFrame.
        """
        
        logger.info("Processing Holidays (Bronze -> Silver)...")

        # Check type and load to LazyFrame
        if isinstance(input_data, Path):
            lf = pl.read_json(input_data).lazy()
        elif isinstance(input_data, pl.DataFrame):
            lf = input_data.lazy()
        else:
             raise TypeError(f"Unsupported input type: {type(input_data)}")

        # Transform
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
        
        # Materialize
        df_silver = q.collect()

        # Write to disk
        logger.info(f"Persisting Holidays Silver Layer to {output_path}...")
        clean_output_directory(output_path)
        
        df_silver.write_parquet(
            output_path,
            partition_by=["year"]
        )

        return df_silver, output_path
    
    @time_execution
    def process_weather(
            self, 
            input_data: Union[pl.DataFrame, Path], 
            output_path: Path
        ) -> Tuple[pl.DataFrame, Path]:
            """
            Standardizes NOAA GHCN-Daily weather data, writes to Silver and returns the DataFrame.
            """
            logger.info("Processing Weather Data (Bronze -> Silver)...")
            
            # Load Input to LazyFrame
            if isinstance(input_data, Path):
                # 'infer_schema_length=0' we read all cols as String first to avoid errors with messy CSVs
                lf = pl.scan_csv(input_data, ignore_errors=True, infer_schema_length=0)
            elif isinstance(input_data, pl.DataFrame):
                lf = input_data.lazy()
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")

            # Transform
            # Data extracted from NOAA GHCN-Daily has to be processed to obtain useful data

            q = (
                lf
                .filter(pl.col("DATE") >= "2020-01-01")
                .filter(pl.col("DATE").is_not_null())
                .with_columns([
                    # Convert DATE string to Date type
                    pl.col("DATE").str.to_date("%Y-%m-%d").alias("date"),
                    
                    # Convert Temperature
                    (pl.col("TMAX").cast(pl.String).str.strip_chars().cast(pl.Float64) / 10).round(1).alias("temp_max_c"),
                    (pl.col("TMIN").cast(pl.String).str.strip_chars().cast(pl.Float64) / 10).round(1).alias("temp_min_c"),

                    # Convert Precipitation
                    (pl.col("PRCP").cast(pl.String).str.strip_chars().cast(pl.Float64) / 10).alias("precipitation_mm"),

                    pl.col("SNOW").cast(pl.String).str.strip_chars().cast(pl.Float64).alias("snow_mm"),

                    # --- Fog, rain, snow ---
                    # Logic: If either WT01 or WT02 is "1", it was foggy.
                    pl.any_horizontal(
                        pl.col("WT01").cast(pl.String).str.strip_chars().cast(pl.Int32, strict=False) == 1,
                        pl.col("WT02").cast(pl.String).str.strip_chars().cast(pl.Int32, strict=False) == 1
                    ).fill_null(False).alias("is_foggy"),

                    # Basic Boolean flags for Rain/Snow based on measurements
                    (pl.col("PRCP").cast(pl.String).str.strip_chars().cast(pl.Float64) > 0).fill_null(False).alias("has_rain"),
                    (pl.col("SNOW").cast(pl.String).str.strip_chars().cast(pl.Float64) > 0).fill_null(False).alias("has_snow")
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
            
            # Materialize Dataframe 
            df_silver = q.collect()

            # Save to disk/datalake (Partitioned)
            logger.info(f"Persisting Weather Silver Layer to {output_path}...")
            clean_output_directory(output_path)
            
            df_silver.write_parquet(
                output_path,
                partition_by=["year", "month"]
            )

            return df_silver, output_path
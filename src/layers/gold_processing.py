import polars as pl
from pathlib import Path
from typing import Union
from ..utils import setup_logger,clean_output_directory, time_execution

logger = setup_logger(__name__)

class GoldProcessor:
    """
    Handles the Gold Layer lifecycle.
    Can ingest data from Disk (Path) or Memory (DataFrame).
    """

    # This is an example of what shouldn't be written inside code as it doesn't isolate the system and makes you re deploy in case you want to change it
    METRIC_COLS = [
        "number_of_persons_injured", "number_of_persons_killed",
        "number_of_pedestrians_injured", "number_of_pedestrians_killed",
        "number_of_cyclist_injured", "number_of_cyclist_killed",
        "number_of_motorist_injured", "number_of_motorist_killed"
    ]


    def _normalize_input(self, input_data: Union[pl.DataFrame, Path], table_name: str = "") -> pl.LazyFrame:
        """
        If input is DataFrame -> Convert to LazyFrame
        If input is Path -> Scan Parquet files
        """
        if isinstance(input_data, pl.DataFrame):
            logger.info(f"Input '{table_name}': using DataFrame")
            return input_data.lazy()
            
        elif isinstance(input_data, Path):
            # Assuming partition structure: path/table_name/**/*.parquet
            full_path = input_data / "**/*.parquet"
            logger.info(f"Input '{table_name}': scanning disk at {full_path}")
            return pl.scan_parquet(str(full_path))
            
        else:
            raise TypeError(f"Unsupported input type for {table_name}: {type(input_data)}")

    @time_execution
    def _enrich_collisions(
            self, 
            lf_collisions: pl.LazyFrame, 
            lf_holidays: pl.LazyFrame, 
            lf_weather: pl.LazyFrame
        ) -> pl.DataFrame:
            """
            Joins Collisions with Holidays & Weather, then calculates impact flags (defined by myself)
            """
            logger.info("Applying business rules: joining Collisions, Holidays & Weather...")
        # Select columns that we find useful.
        # I've had to drop 'year', 'month' to avoid conflicts with those columns in other datasets
            weather_clean = lf_weather.select([
                "date", 
                "temp_max_c", 
                "temp_min_c", 
                "has_rain", 
                "has_snow", 
                "is_foggy"
            ])
            
            q = (
                lf_collisions
                .filter(pl.col("date").dt.year() >= 2020) # just last 5 years
                .join(lf_holidays, on="date", how="left")
                .join(weather_clean, on="date", how="left")
                .with_columns([
                    pl.col("holiday_name").fill_null("Non-Holiday"),
                    
                    (pl.col("types").list.contains("Public") | pl.col("types").list.contains("Bank"))
                        .fill_null(False).alias("high_impact_holiday"),
                    
                    (pl.col("types").list.contains("School") | pl.col("types").list.contains("Authorities"))
                        .fill_null(False).alias("partial_impact_holiday"),
                    
                    (pl.col("types").list.contains("Optional") | pl.col("types").list.contains("Observance"))
                        .fill_null(False).alias("low_impact_holiday"),

                    # Boolean flags: fill missing days with False
                    pl.col("has_rain").fill_null(False),
                    pl.col("has_snow").fill_null(False),
                    pl.col("is_foggy").fill_null(False),
                    
                    # I am renaming them here slightly for cleaner reading
                    pl.col("temp_max_c").alias("max_temp"),
                    pl.col("temp_min_c").alias("min_temp"),
                ])
            )
            
            return q.collect()

    @time_execution
    def _aggregate_stats(self, enriched_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generates the final gold aggregation.
        """
        logger.info("Creating gold aggregations...")
        
        group_cols = [
            "date", "borough", "zip_code", "is_weekend", "holiday_name", 
            "high_impact_holiday", "partial_impact_holiday", "low_impact_holiday",
            "has_rain","has_snow","is_foggy","max_temp","min_temp"
        ]

        return (
            enriched_df.group_by(group_cols)
            .agg([
                pl.len().alias("total_accidents"),
                pl.col(self.METRIC_COLS).sum()
            ])
            .sort("date")
        )
        
    @time_execution
    def process_gold_data(
        self, 
        collisions_data: Union[pl.DataFrame, Path], 
        holidays_data: Union[pl.DataFrame, Path], 
        weather_data: Union[pl.DataFrame, Path], 
        gold_base_path: Path
    ):
        """
        Main entry point for gold processing.
        Accepts either DataFrames (from previous step) or Paths (from disk).
        """
        # Normalize Inputs
        lf_collisions = self._normalize_input(collisions_data, "collisions")
        lf_holidays = self._normalize_input(holidays_data, "holidays")
        lf_weather = self._normalize_input(weather_data, "weather")

        # Transformation
        df_enriched = self._enrich_collisions(lf_collisions, lf_holidays,lf_weather)
        df_gold = self._aggregate_stats(df_enriched)

        # Persist
        clean_output_directory(gold_base_path)
        
        parquet_path = gold_base_path / "daily_stats.parquet"
        
        # I create this csv in case a business users wants to do an analysis in Excel
        csv_path = gold_base_path / "daily_stats.csv"

        logger.info(f"Persisting gold data to {gold_base_path}...")
        df_gold.write_parquet(parquet_path)
        df_gold.write_csv(csv_path)
        
        logger.info("Gold layer processing complete.")
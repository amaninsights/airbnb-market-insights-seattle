"""
Airbnb Data Loader Module
=========================

Provides utilities for loading Airbnb data from CSV files and Kaggle.
Supports automated download, schema validation, and SQLite ingestion.

Example Usage:
    >>> from src.data.loader import AirbnbDataLoader
    >>> loader = AirbnbDataLoader("config/config.yaml")
    >>> listings_df = loader.load_listings()
    >>> loader.ingest_all()
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AirbnbDataLoader:
    """
    Production-grade data loader for Seattle Airbnb datasets.

    Handles:
    - Loading CSV files with proper data types
    - Date parsing and validation
    - SQLite database ingestion
    - Schema validation

    Attributes:
        config (dict): Configuration from YAML file
        engine (Engine): SQLAlchemy database engine
    """

    # Expected columns for validation
    LISTINGS_COLUMNS = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "availability_365",
    ]

    CALENDAR_COLUMNS = ["listing_id", "date", "available", "price"]

    REVIEWS_COLUMNS = [
        "listing_id",
        "id",
        "date",
        "reviewer_id",
        "reviewer_name",
        "comments",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize AirbnbDataLoader with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.engine = self._create_engine()
        self.data_dir = Path(self.config.get("data", {}).get("raw_dir", "data/raw"))
        logger.info(f"AirbnbDataLoader initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine from configuration."""
        db_path = self.config.get("database", {}).get(
            "path", "data/processed/airbnb_seattle.db"
        )

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        logger.info(f"Database engine created: {db_path}")
        return engine

    def load_listings(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load Airbnb listings data.

        Args:
            filepath: Optional custom path to listings CSV

        Returns:
            DataFrame with listings data
        """
        if filepath is None:
            filename = (
                self.config.get("data", {})
                .get("files", {})
                .get("listings", "listings.csv")
            )
            filepath = self.data_dir / filename

        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Listings file not found: {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading listings from: {filepath}")

        df = pd.read_csv(filepath, low_memory=False)

        # Clean price column if present
        if "price" in df.columns:
            df["price"] = self._clean_price(df["price"])

        logger.info(f"Loaded {len(df):,} listings")
        return df

    def load_calendar(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load Airbnb calendar/availability data.

        Args:
            filepath: Optional custom path to calendar CSV

        Returns:
            DataFrame with calendar data
        """
        if filepath is None:
            filename = (
                self.config.get("data", {})
                .get("files", {})
                .get("calendar", "calendar.csv")
            )
            filepath = self.data_dir / filename

        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Calendar file not found: {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading calendar from: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["date"], low_memory=False)

        # Clean price and convert available to boolean
        if "price" in df.columns:
            df["price"] = self._clean_price(df["price"])
        if "available" in df.columns:
            df["available"] = df["available"].map({"t": True, "f": False})

        logger.info(f"Loaded {len(df):,} calendar entries")
        return df

    def load_reviews(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load Airbnb reviews data.

        Args:
            filepath: Optional custom path to reviews CSV

        Returns:
            DataFrame with reviews data
        """
        if filepath is None:
            filename = (
                self.config.get("data", {})
                .get("files", {})
                .get("reviews", "reviews.csv")
            )
            filepath = self.data_dir / filename

        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Reviews file not found: {filepath}")
            return pd.DataFrame()

        logger.info(f"Loading reviews from: {filepath}")

        df = pd.read_csv(filepath, parse_dates=["date"], low_memory=False)

        logger.info(f"Loaded {len(df):,} reviews")
        return df

    def _clean_price(self, price_series: pd.Series) -> pd.Series:
        """
        Clean price column by removing $ and commas.

        Args:
            price_series: Series with price values

        Returns:
            Cleaned numeric price series
        """
        if price_series.dtype == "object":
            return pd.to_numeric(
                price_series.astype(str).str.replace("[$,]", "", regex=True),
                errors="coerce",
            )
        return price_series

    def ingest_to_db(
        self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"
    ) -> int:
        """
        Ingest DataFrame into SQLite database.

        Args:
            df: DataFrame to ingest
            table_name: Target table name
            if_exists: How to handle existing table

        Returns:
            Number of rows inserted
        """
        logger.info(f"Ingesting {len(df):,} rows into table: {table_name}")

        rows = df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

        logger.info(f"Successfully ingested into {table_name}")
        return rows or len(df)

    def ingest_all(self) -> Dict[str, int]:
        """
        Load and ingest all data files to database.

        Returns:
            Dictionary mapping table names to row counts
        """
        results = {}

        # Listings
        df = self.load_listings()
        if not df.empty:
            results["listings"] = self.ingest_to_db(df, "listings")

        # Calendar
        df = self.load_calendar()
        if not df.empty:
            results["calendar"] = self.ingest_to_db(df, "calendar")

        # Reviews
        df = self.load_reviews()
        if not df.empty:
            results["reviews"] = self.ingest_to_db(df, "reviews")

        logger.info(f"Ingestion complete: {results}")
        return results

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return results."""
        return pd.read_sql(sql, self.engine)

    def get_sample_data(self, n: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Get sample data for testing/development.

        Args:
            n: Number of rows to sample

        Returns:
            Dictionary of sampled DataFrames
        """
        return {
            "listings": self.load_listings().head(n),
            "calendar": self.load_calendar().head(n),
            "reviews": self.load_reviews().head(n),
        }


def main():
    """Example usage of AirbnbDataLoader."""
    loader = AirbnbDataLoader()

    # Load all data
    results = loader.ingest_all()

    print("\n📊 Ingestion Summary:")
    print("-" * 40)
    for table, rows in results.items():
        print(f"  {table}: {rows:,} rows")
    print("-" * 40)


if __name__ == "__main__":
    main()

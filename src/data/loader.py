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

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AirbnbDataLoader:
    """
    Data loader for Seattle Airbnb datasets.

    Handles CSV file loading, data validation, and SQLite database ingestion.
    Supports configurable file paths and database connections.

    Attributes:
        config: Configuration dictionary
        engine: SQLAlchemy database engine
        data_dir: Path to raw data directory
    """

    REQUIRED_LISTING_COLUMNS = [
        "id",
        "name",
        "host_id",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
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
        logger.info("AirbnbDataLoader initialized")

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
        Clean price column by removing currency symbols.

        Args:
            price_series: Series containing price values

        Returns:
            Cleaned numeric price series
        """
        if price_series.dtype == "object":
            return (
                price_series.replace(r"[\$,]", "", regex=True)
                .replace("", "0")
                .astype(float)
            )
        return price_series

    def validate_listings(self, df: pd.DataFrame) -> bool:
        """
        Validate listings DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        missing = set(self.REQUIRED_LISTING_COLUMNS) - set(df.columns)
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def ingest_to_database(self, df: pd.DataFrame, table_name: str) -> int:
        """
        Ingest DataFrame to SQLite database.

        Args:
            df: DataFrame to ingest
            table_name: Target table name

        Returns:
            Number of rows inserted
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, skipping {table_name} ingestion")
            return 0

        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        logger.info(f"Ingested {len(df):,} rows to {table_name}")
        return len(df)

    def ingest_all(self) -> Dict[str, int]:
        """
        Load and ingest all datasets to database.

        Returns:
            Dictionary with table names and row counts
        """
        results = {}

        listings = self.load_listings()
        if not listings.empty and self.validate_listings(listings):
            results["listings"] = self.ingest_to_database(listings, "listings")

        calendar = self.load_calendar()
        if not calendar.empty:
            results["calendar"] = self.ingest_to_database(calendar, "calendar")

        reviews = self.load_reviews()
        if not reviews.empty:
            results["reviews"] = self.ingest_to_database(reviews, "reviews")

        logger.info(f"Ingestion complete: {results}")
        return results

    def get_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Load data from database table.

        Args:
            table_name: Name of table to load

        Returns:
            DataFrame with table data
        """
        query = f"SELECT * FROM {table_name}"
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df):,} rows from {table_name}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            return pd.DataFrame()


def main():
    """Example usage of AirbnbDataLoader."""
    loader = AirbnbDataLoader("config/config.yaml")

    # Load datasets
    listings = loader.load_listings()
    calendar = loader.load_calendar()
    reviews = loader.load_reviews()

    print("\n Data Loaded:")
    print(f"  Listings: {len(listings):,}")
    print(f"  Calendar: {len(calendar):,}")
    print(f"  Reviews: {len(reviews):,}")

    if not listings.empty:
        print(f"\n  Columns: {list(listings.columns)}")


if __name__ == "__main__":
    main()

"""
Airbnb Data Transformer Module
==============================

Provides data cleaning, feature engineering, and transformation
for Airbnb market analysis. Creates derived metrics and aggregations.

Example Usage:
    >>> from src.data.transformer import AirbnbTransformer
    >>> transformer = AirbnbTransformer(listings_df, calendar_df)
    >>> enriched_df = transformer.transform()
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AirbnbTransformer:
    """
    Production-grade data transformer for Airbnb analytics.

    Provides:
    - Data cleaning and validation
    - Feature engineering
    - Price normalization
    - Occupancy calculations
    - Seasonal analysis preparation

    Attributes:
        listings (pd.DataFrame): Listings data
        calendar (pd.DataFrame): Calendar/availability data
        reviews (pd.DataFrame): Reviews data
    """

    # Room type standardization mapping
    ROOM_TYPE_MAP = {
        "Entire home/apt": "Entire Home",
        "Private room": "Private Room",
        "Shared room": "Shared Room",
        "Hotel room": "Hotel Room",
    }

    def __init__(
        self,
        listings: pd.DataFrame,
        calendar: Optional[pd.DataFrame] = None,
        reviews: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize transformer with data.

        Args:
            listings: Listings DataFrame
            calendar: Optional calendar DataFrame
            reviews: Optional reviews DataFrame
        """
        self.listings = listings.copy()
        self.calendar = calendar.copy() if calendar is not None else None
        self.reviews = reviews.copy() if reviews is not None else None
        logger.info(f"AirbnbTransformer initialized with {len(listings)} listings")

    def transform(self) -> pd.DataFrame:
        """
        Run full transformation pipeline.

        Returns:
            Fully transformed and enriched listings DataFrame
        """
        logger.info("Starting transformation pipeline")

        df = self.listings.copy()

        # Step 1: Clean data
        df = self.clean_listings(df)

        # Step 2: Add derived features
        df = self.add_price_features(df)

        # Step 3: Add occupancy metrics if calendar available
        if self.calendar is not None:
            df = self.add_occupancy_metrics(df)

        # Step 4: Add review metrics if reviews available
        if self.reviews is not None:
            df = self.add_review_metrics(df)

        # Step 5: Add neighborhood rankings
        df = self.add_neighborhood_rankings(df)

        logger.info(
            f"Transformation complete: {len(df)} listings, {len(df.columns)} features"
        )
        return df

    def clean_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean listings data.

        Cleaning steps:
        1. Handle missing values
        2. Standardize room types
        3. Remove price outliers
        4. Clean text fields

        Args:
            df: Raw listings DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning listings data")
        original_len = len(df)

        # Handle missing prices
        if "price" in df.columns:
            df = df[df["price"].notna() & (df["price"] > 0)]

        # Standardize room types
        if "room_type" in df.columns:
            df["room_type"] = df["room_type"].map(
                lambda x: self.ROOM_TYPE_MAP.get(x, x)
            )

        # Fill missing neighborhoods
        if "neighbourhood" in df.columns:
            df["neighbourhood"] = df["neighbourhood"].fillna("Unknown")

        # Clean host names
        if "host_name" in df.columns:
            df["host_name"] = df["host_name"].fillna("Unknown Host")

        # Fill numeric nulls
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        logger.info(f"Cleaned: {original_len} -> {len(df)} listings")
        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-related derived features.

        Features added:
        - price_per_person: Estimated per-person price
        - price_category: Low/Medium/High/Premium
        - price_zscore: Standardized price within neighborhood

        Args:
            df: Listings DataFrame

        Returns:
            DataFrame with price features
        """
        if "price" not in df.columns:
            return df

        logger.info("Adding price features")

        # Price per person estimate (assuming avg 2 guests)
        df["price_per_person"] = df["price"] / 2

        # Price category based on percentiles
        price_percentiles = df["price"].quantile([0.25, 0.5, 0.75])

        def categorize_price(price):
            if price <= price_percentiles[0.25]:
                return "Budget"
            elif price <= price_percentiles[0.5]:
                return "Mid-Range"
            elif price <= price_percentiles[0.75]:
                return "Premium"
            else:
                return "Luxury"

        df["price_category"] = df["price"].apply(categorize_price)

        # Z-score within neighborhood
        if "neighbourhood" in df.columns:
            df["price_zscore"] = df.groupby("neighbourhood")["price"].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            df["price_zscore"] = (df["price"] - df["price"].mean()) / df["price"].std()

        # Log price for modeling
        df["log_price"] = np.log1p(df["price"])

        return df

    def add_occupancy_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate occupancy metrics from calendar data.

        Metrics added:
        - occupancy_rate: Percentage of days booked
        - avg_booked_price: Average price when booked
        - revenue_potential: Estimated monthly revenue

        Args:
            df: Listings DataFrame

        Returns:
            DataFrame with occupancy metrics
        """
        if self.calendar is None or self.calendar.empty:
            return df

        logger.info("Calculating occupancy metrics")

        # Aggregate calendar by listing
        calendar_agg = (
            self.calendar.groupby("listing_id")
            .agg(
                {
                    "available": lambda x: 1 - x.mean(),  # Occupancy rate
                    "price": "mean",
                    "date": "count",
                }
            )
            .rename(
                columns={
                    "available": "occupancy_rate",
                    "price": "avg_calendar_price",
                    "date": "calendar_days",
                }
            )
        )

        # Calculate seasonal occupancy
        if "date" in self.calendar.columns:
            self.calendar["month"] = pd.to_datetime(self.calendar["date"]).dt.month

            # Summer occupancy (Jun-Aug)
            summer = self.calendar[self.calendar["month"].isin([6, 7, 8])]
            summer_occ = (
                summer.groupby("listing_id")["available"]
                .apply(lambda x: 1 - x.mean())
                .rename("summer_occupancy")
            )

            # Winter occupancy (Dec-Feb)
            winter = self.calendar[self.calendar["month"].isin([12, 1, 2])]
            winter_occ = (
                winter.groupby("listing_id")["available"]
                .apply(lambda x: 1 - x.mean())
                .rename("winter_occupancy")
            )

            calendar_agg = calendar_agg.join(summer_occ).join(winter_occ)

        # Merge with listings
        id_col = "id" if "id" in df.columns else df.columns[0]
        df = df.merge(calendar_agg, left_on=id_col, right_index=True, how="left")

        # Calculate revenue potential
        if "occupancy_rate" in df.columns and "price" in df.columns:
            df["monthly_revenue_potential"] = df["price"] * df["occupancy_rate"] * 30

        return df

    def add_review_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate review-based metrics.

        Metrics added:
        - review_frequency: Reviews per month
        - sentiment_proxy: Based on review count trends

        Args:
            df: Listings DataFrame

        Returns:
            DataFrame with review metrics
        """
        if self.reviews is None or self.reviews.empty:
            return df

        logger.info("Calculating review metrics")

        # Count reviews per listing
        review_counts = self.reviews.groupby("listing_id").agg(
            {"id": "count", "date": ["min", "max"]}
        )
        review_counts.columns = ["review_count", "first_review", "last_review"]

        # Calculate review frequency (reviews per month active)
        review_counts["months_active"] = (
            (
                pd.to_datetime(review_counts["last_review"])
                - pd.to_datetime(review_counts["first_review"])
            ).dt.days
            / 30
        ).clip(lower=1)

        review_counts["review_frequency"] = (
            review_counts["review_count"] / review_counts["months_active"]
        )

        # Merge with listings
        id_col = "id" if "id" in df.columns else df.columns[0]
        df = df.merge(
            review_counts[["review_count", "review_frequency"]],
            left_on=id_col,
            right_index=True,
            how="left",
        )

        return df

    def add_neighborhood_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add neighborhood-level rankings and metrics.

        Metrics added:
        - neighborhood_avg_price: Average price in neighborhood
        - neighborhood_listing_count: Number of listings
        - neighborhood_rank: Rank by average price
        - price_vs_neighborhood: Relative to neighborhood average

        Args:
            df: Listings DataFrame

        Returns:
            DataFrame with neighborhood rankings
        """
        if "neighbourhood" not in df.columns or "price" not in df.columns:
            return df

        logger.info("Adding neighborhood rankings")

        # Neighborhood aggregates
        neighborhood_stats = df.groupby("neighbourhood").agg(
            {"price": ["mean", "median", "std", "count"], "id": "count"}
        )
        neighborhood_stats.columns = [
            "neighborhood_avg_price",
            "neighborhood_median_price",
            "neighborhood_price_std",
            "neighborhood_price_count",
            "neighborhood_listing_count",
        ]

        # Rank neighborhoods by average price
        neighborhood_stats["neighborhood_rank"] = neighborhood_stats[
            "neighborhood_avg_price"
        ].rank(ascending=False)

        # Merge back
        df = df.merge(
            neighborhood_stats, left_on="neighbourhood", right_index=True, how="left"
        )

        # Price relative to neighborhood
        df["price_vs_neighborhood"] = (
            (df["price"] - df["neighborhood_avg_price"])
            / df["neighborhood_avg_price"]
            * 100
        )

        return df

    def remove_outliers(
        self, df: pd.DataFrame, column: str = "price", n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using standard deviation method.

        Args:
            df: Input DataFrame
            column: Column to check for outliers
            n_std: Number of standard deviations

        Returns:
            DataFrame with outliers removed
        """
        if column not in df.columns:
            return df

        mean = df[column].mean()
        std = df[column].std()

        lower = mean - (n_std * std)
        upper = mean + (n_std * std)

        original_len = len(df)
        df = df[(df[column] >= lower) & (df[column] <= upper)]

        logger.info(f"Removed {original_len - len(df)} outliers from {column}")
        return df

    def create_neighborhood_summary(self) -> pd.DataFrame:
        """
        Create summary statistics by neighborhood.

        Returns:
            DataFrame with neighborhood-level statistics
        """
        if "neighbourhood" not in self.listings.columns:
            return pd.DataFrame()

        logger.info("Creating neighborhood summary")

        summary = self.listings.groupby("neighbourhood").agg(
            {
                "id": "count",
                "price": ["mean", "median", "min", "max", "std"],
                "number_of_reviews": (
                    "sum" if "number_of_reviews" in self.listings.columns else "count"
                ),
                "availability_365": (
                    "mean" if "availability_365" in self.listings.columns else "count"
                ),
            }
        )

        summary.columns = [
            "listing_count",
            "avg_price",
            "median_price",
            "min_price",
            "max_price",
            "price_std",
            "total_reviews",
            "avg_availability",
        ]

        summary = summary.sort_values("avg_price", ascending=False)

        return summary.reset_index()

    def create_room_type_summary(self) -> pd.DataFrame:
        """
        Create summary statistics by room type.

        Returns:
            DataFrame with room type statistics
        """
        if "room_type" not in self.listings.columns:
            return pd.DataFrame()

        logger.info("Creating room type summary")

        summary = self.listings.groupby("room_type").agg(
            {
                "id": "count",
                "price": ["mean", "median"],
                "number_of_reviews": (
                    "mean" if "number_of_reviews" in self.listings.columns else "count"
                ),
            }
        )

        summary.columns = ["count", "avg_price", "median_price", "avg_reviews"]
        summary["market_share"] = summary["count"] / summary["count"].sum() * 100

        return summary.reset_index()


def main():
    """Example usage of AirbnbTransformer."""
    from src.data.loader import AirbnbDataLoader

    # Load data
    loader = AirbnbDataLoader()
    listings = loader.load_listings()
    calendar = loader.load_calendar()
    reviews = loader.load_reviews()

    # Transform
    transformer = AirbnbTransformer(listings, calendar, reviews)
    df = transformer.transform()

    print("\n Transformed Data:")
    print(f"  Listings: {len(df):,}")
    print(f"  Features: {len(df.columns)}")
    print(f"\n  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

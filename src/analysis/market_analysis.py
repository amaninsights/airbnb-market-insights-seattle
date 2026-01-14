"""
Market Analysis Module
======================

Provides comprehensive market analysis for Seattle Airbnb data including
pricing strategies, seasonal trends, and competitive analysis.

Example Usage:
    >>> from src.analysis.market_analysis import MarketAnalyzer
    >>> analyzer = MarketAnalyzer(listings_df)
    >>> insights = analyzer.get_market_insights()
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MarketInsights:
    """Container for market insights."""

    total_listings: int
    total_hosts: int
    avg_price: float
    median_price: float
    price_range: Tuple[float, float]
    top_neighborhood: str
    most_common_room_type: str
    dominant_room_type: str
    avg_occupancy: float
    seasonal_premium: float


class MarketAnalyzer:
    """
    Production-grade market analyzer for Airbnb data.

    Provides:
    - Market overview statistics
    - Pricing analysis
    - Neighborhood rankings
    - Seasonal trend analysis
    - Competitive positioning
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with data.

        Args:
            df: Transformed listings DataFrame
        """
        self.df = df.copy()
        logger.info(f"MarketAnalyzer initialized with {len(df)} listings")

    def get_market_insights(self) -> MarketInsights:
        """
        Generate comprehensive market insights.

        Returns:
            MarketInsights dataclass with key metrics
        """
        df = self.df

        # Basic counts
        total_listings = len(df)
        total_hosts = df["host_id"].nunique() if "host_id" in df.columns else 0

        # Price metrics
        avg_price = df["price"].mean() if "price" in df.columns else 0
        median_price = df["price"].median() if "price" in df.columns else 0
        price_range = (
            (df["price"].min(), df["price"].max()) if "price" in df.columns else (0, 0)
        )

        # Top neighborhood by average price
        top_neighborhood = "N/A"
        if "neighbourhood" in df.columns and "price" in df.columns:
            neighborhood_prices = df.groupby("neighbourhood")["price"].mean()
            top_neighborhood = neighborhood_prices.idxmax()

        # Room type analysis
        most_common_room = "N/A"
        dominant_room = "N/A"
        if "room_type" in df.columns:
            room_counts = df["room_type"].value_counts()
            most_common_room = room_counts.index[0]
            dominant_room = most_common_room

        # Occupancy
        avg_occupancy = 0.0
        if "occupancy_rate" in df.columns:
            avg_occupancy = df["occupancy_rate"].mean()
        elif "availability_365" in df.columns:
            avg_occupancy = 1 - (df["availability_365"].mean() / 365)

        # Seasonal premium
        seasonal_premium = 0.0
        if "summer_occupancy" in df.columns and "winter_occupancy" in df.columns:
            summer_occ = df["summer_occupancy"].mean()
            winter_occ = df["winter_occupancy"].mean()
            if winter_occ > 0:
                seasonal_premium = ((summer_occ - winter_occ) / winter_occ) * 100

        return MarketInsights(
            total_listings=total_listings,
            total_hosts=total_hosts,
            avg_price=avg_price,
            median_price=median_price,
            price_range=price_range,
            top_neighborhood=top_neighborhood,
            most_common_room_type=most_common_room,
            dominant_room_type=dominant_room,
            avg_occupancy=avg_occupancy,
            seasonal_premium=seasonal_premium,
        )

    def analyze_pricing(self) -> Dict:
        """
        Perform detailed price analysis.

        Returns:
            Dictionary with pricing statistics
        """
        if "price" not in self.df.columns:
            return {}

        price = self.df["price"]

        analysis = {
            "mean": price.mean(),
            "median": price.median(),
            "std": price.std(),
            "min": price.min(),
            "max": price.max(),
            "percentiles": {
                "25th": price.quantile(0.25),
                "50th": price.quantile(0.50),
                "75th": price.quantile(0.75),
                "90th": price.quantile(0.90),
                "95th": price.quantile(0.95),
            },
            "skewness": price.skew(),
            "kurtosis": price.kurtosis(),
        }

        # Price by room type
        if "room_type" in self.df.columns:
            analysis["by_room_type"] = (
                self.df.groupby("room_type")["price"]
                .agg(["mean", "median", "count"])
                .to_dict()
            )

        return analysis

    def analyze_neighborhoods(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze and rank neighborhoods.

        Args:
            top_n: Number of top neighborhoods to return

        Returns:
            DataFrame with neighborhood rankings
        """
        if "neighbourhood" not in self.df.columns:
            return pd.DataFrame()

        # Aggregate by neighborhood
        agg_funcs = {"price": ["mean", "median", "count", "std"]}

        if "number_of_reviews" in self.df.columns:
            agg_funcs["number_of_reviews"] = "sum"

        if "occupancy_rate" in self.df.columns:
            agg_funcs["occupancy_rate"] = "mean"

        neighborhood_stats = self.df.groupby("neighbourhood").agg(agg_funcs)

        # Flatten column names
        neighborhood_stats.columns = (
            [
                "avg_price",
                "median_price",
                "listing_count",
                "price_std",
            ]
            + (["total_reviews"] if "number_of_reviews" in self.df.columns else [])
            + (["avg_occupancy"] if "occupancy_rate" in self.df.columns else [])
        )

        # Add rank
        neighborhood_stats["rank"] = neighborhood_stats["avg_price"].rank(
            ascending=False
        )

        # Sort and get top N
        result = (
            neighborhood_stats.sort_values("avg_price", ascending=False)
            .head(top_n)
            .reset_index()
        )

        return result

    def analyze_seasonality(self) -> Dict:
        """
        Analyze seasonal patterns in the market.

        Returns:
            Dictionary with seasonal analysis
        """
        result = {}

        # Occupancy seasonality
        if "summer_occupancy" in self.df.columns:
            result["summer_occupancy"] = self.df["summer_occupancy"].mean()

        if "winter_occupancy" in self.df.columns:
            result["winter_occupancy"] = self.df["winter_occupancy"].mean()

        if "summer_occupancy" in result and "winter_occupancy" in result:
            result["seasonal_variation"] = (
                (result["summer_occupancy"] - result["winter_occupancy"])
                / result["winter_occupancy"]
                * 100
                if result["winter_occupancy"] > 0
                else 0
            )

        return result

    def get_competitive_position(
        self, price: float, neighborhood: str, room_type: str
    ) -> Dict:
        """
        Determine competitive position for a listing.

        Args:
            price: Listing price
            neighborhood: Neighborhood name
            room_type: Room type

        Returns:
            Dictionary with competitive analysis
        """
        # Filter comparables
        comparables = self.df[
            (self.df["neighbourhood"] == neighborhood)
            | (self.df["room_type"].str.contains(room_type, case=False, na=False))
        ]

        if len(comparables) == 0:
            return {"error": "No comparable listings found"}

        market_prices = comparables["price"]

        # Calculate position
        percentile = stats.percentileofscore(market_prices, price)
        price_vs_avg = ((price - market_prices.mean()) / market_prices.mean()) * 100

        return {
            "price": price,
            "neighborhood": neighborhood,
            "room_type": room_type,
            "comparable_count": len(comparables),
            "market_avg": market_prices.mean(),
            "market_median": market_prices.median(),
            "price_percentile": percentile,
            "price_vs_average": price_vs_avg,
            "position": self._get_position_label(percentile),
            "recommendation": self._get_price_recommendation(price, market_prices),
        }

    def _get_position_label(self, percentile: float) -> str:
        """Get position label from percentile."""
        if percentile <= 25:
            return "Budget (Bottom 25%)"
        elif percentile <= 50:
            return "Below Average"
        elif percentile <= 75:
            return "Above Average"
        else:
            return "Premium (Top 25%)"

    def _get_price_recommendation(self, price: float, market_prices: pd.Series) -> str:
        """Generate price recommendation."""
        median = market_prices.median()
        diff = price - median

        if price < median * 0.8:
            return (
                f"Consider increasing price. You're 20%+ below median (${median:.0f})"
            )
        elif price > median * 1.5:
            return (
                f"Price is premium. Ensure quality justifies ${diff:.0f} above median"
            )
        else:
            return "Price is competitive within market range"

    def estimate_revenue(
        self, price: float, occupancy_rate: float = 0.7
    ) -> Dict[str, float]:
        """
        Estimate potential revenue.

        Args:
            price: Nightly price
            occupancy_rate: Expected occupancy (0-1)

        Returns:
            Revenue estimates
        """
        daily_revenue = price * occupancy_rate

        return {
            "nightly_rate": price,
            "occupancy_rate": occupancy_rate * 100,
            "daily_revenue": daily_revenue,
            "weekly_revenue": daily_revenue * 7,
            "monthly_revenue": daily_revenue * 30,
            "annual_revenue": daily_revenue * 365,
        }

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Perform correlation analysis on numeric features.

        Returns:
            Correlation matrix
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Select relevant columns
        relevant = [
            c
            for c in numeric_cols
            if c not in ["id", "host_id", "latitude", "longitude"]
        ]

        return self.df[relevant].corr()

    def get_price_factors(self) -> List[Tuple[str, float]]:
        """
        Identify factors most correlated with price.

        Returns:
            List of (factor, correlation) tuples sorted by importance
        """
        if "price" not in self.df.columns:
            return []

        corr_matrix = self.correlation_analysis()

        if "price" not in corr_matrix.columns:
            return []

        price_corr = corr_matrix["price"].drop("price", errors="ignore")

        factors = [(col, corr) for col, corr in price_corr.items() if abs(corr) > 0.1]

        return sorted(factors, key=lambda x: abs(x[1]), reverse=True)

    def generate_report(self) -> str:
        """
        Generate text summary report.

        Returns:
            Formatted report string
        """
        insights = self.get_market_insights()

        report = f"""
===============================================================
           SEATTLE AIRBNB MARKET ANALYSIS REPORT
===============================================================

 MARKET OVERVIEW
   Total Listings: {insights.total_listings:>40,}
   Total Hosts: {insights.total_hosts:>44,}
   Average Price: ${insights.avg_price:>41,.2f}
   Median Price: ${insights.median_price:>42,.2f}

 KEY INSIGHTS
   Top Neighborhood: {insights.top_neighborhood:>38}
   Dominant Room Type: {insights.dominant_room_type:>36}
   Avg Occupancy Rate: {insights.avg_occupancy*100:>34.1f}%
   Seasonal Premium: {insights.seasonal_premium:>37.1f}%

 PRICE RANGE
   Minimum: ${insights.price_range[0]:>47,.2f}
   Maximum: ${insights.price_range[1]:>47,.2f}
===============================================================
        """
        return report


def main():
    """Example usage of MarketAnalyzer."""
    from src.data.loader import AirbnbDataLoader
    from src.data.transformer import AirbnbTransformer

    # Load and transform data
    loader = AirbnbDataLoader()
    listings = loader.load_listings()
    calendar = loader.load_calendar()

    transformer = AirbnbTransformer(listings, calendar)
    df = transformer.transform()

    # Analyze
    analyzer = MarketAnalyzer(df)
    print(analyzer.generate_report())

    # Top neighborhoods
    print("\n Top 10 Neighborhoods:")
    print(analyzer.analyze_neighborhoods(10))


if __name__ == "__main__":
    main()

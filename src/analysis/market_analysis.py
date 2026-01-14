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

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MarketInsights:
    """Data class for market insights summary."""

    total_listings: int
    total_hosts: int
    avg_price: float
    median_price: float
    avg_occupancy: float
    top_neighborhood: str
    dominant_room_type: str
    price_range: Tuple[float, float]
    seasonal_premium: float


class MarketAnalyzer:
    """
    Comprehensive market analyzer for Airbnb data.

    Provides:
    - Price analysis and recommendations
    - Neighborhood comparison
    - Seasonal trend analysis
    - Competitive positioning
    - Revenue estimation

    Attributes:
        df (pd.DataFrame): Listings data
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize MarketAnalyzer with listings data.

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
        logger.info("Generating market insights")

        # Basic metrics
        total_listings = len(self.df)
        total_hosts = (
            self.df["host_id"].nunique() if "host_id" in self.df.columns else 0
        )
        avg_price = self.df["price"].mean() if "price" in self.df.columns else 0
        median_price = self.df["price"].median() if "price" in self.df.columns else 0

        # Occupancy
        avg_occupancy = 0
        if "occupancy_rate" in self.df.columns:
            avg_occupancy = self.df["occupancy_rate"].mean()
        elif "availability_365" in self.df.columns:
            avg_occupancy = 1 - (self.df["availability_365"].mean() / 365)

        # Top neighborhood
        top_neighborhood = "Unknown"
        if "neighbourhood" in self.df.columns:
            top_neighborhood = self.df.groupby("neighbourhood")["price"].mean().idxmax()

        # Dominant room type
        dominant_room_type = "Unknown"
        if "room_type" in self.df.columns:
            dominant_room_type = self.df["room_type"].mode().iloc[0]

        # Price range
        price_range = (0.0, 0.0)
        if "price" in self.df.columns:
            price_range = (self.df["price"].min(), self.df["price"].max())

        # Seasonal premium
        seasonal_premium = 0.0
        if (
            "summer_occupancy" in self.df.columns
            and "winter_occupancy" in self.df.columns
        ):
            summer_avg = self.df["summer_occupancy"].mean()
            winter_avg = self.df["winter_occupancy"].mean()
            if winter_avg > 0:
                seasonal_premium = (summer_avg - winter_avg) / winter_avg * 100

        return MarketInsights(
            total_listings=total_listings,
            total_hosts=total_hosts,
            avg_price=avg_price,
            median_price=median_price,
            avg_occupancy=avg_occupancy,
            top_neighborhood=top_neighborhood,
            dominant_room_type=dominant_room_type,
            price_range=price_range,
            seasonal_premium=seasonal_premium,
        )

    def analyze_pricing(self) -> Dict[str, any]:
        """
        Detailed price analysis.

        Returns:
            Dictionary with pricing insights
        """
        if "price" not in self.df.columns:
            return {"error": "Price column not found"}

        logger.info("Analyzing pricing")

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
                "99th": price.quantile(0.99),
            },
            "skewness": price.skew(),
            "kurtosis": price.kurtosis(),
        }

        # Price distribution by room type
        if "room_type" in self.df.columns:
            analysis["by_room_type"] = (
                self.df.groupby("room_type")["price"]
                .agg(["mean", "median", "count"])
                .to_dict("index")
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

        logger.info(f"Analyzing top {top_n} neighborhoods")

        neighborhood_stats = self.df.groupby("neighbourhood").agg(
            {
                "id": "count",
                "price": ["mean", "median", "std"],
                "number_of_reviews": (
                    "sum" if "number_of_reviews" in self.df.columns else "count"
                ),
            }
        )

        neighborhood_stats.columns = [
            "listing_count",
            "avg_price",
            "median_price",
            "price_std",
            "total_reviews",
        ]

        # Calculate market share
        neighborhood_stats["market_share"] = (
            neighborhood_stats["listing_count"]
            / neighborhood_stats["listing_count"].sum()
            * 100
        )

        # Calculate revenue potential score
        neighborhood_stats["revenue_score"] = neighborhood_stats["avg_price"] * (
            neighborhood_stats["total_reviews"] / neighborhood_stats["listing_count"]
        )

        # Rank by revenue score
        neighborhood_stats["rank"] = neighborhood_stats["revenue_score"].rank(
            ascending=False
        )

        return neighborhood_stats.sort_values("rank").head(top_n).reset_index()

    def analyze_seasonality(self, calendar_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Analyze seasonal patterns in the market.

        Args:
            calendar_df: Optional calendar DataFrame

        Returns:
            Dictionary with seasonal analysis
        """
        logger.info("Analyzing seasonality")

        analysis = {
            "has_seasonal_data": False,
            "peak_season": "Summer (Jun-Aug)",
            "off_peak_season": "Winter (Dec-Feb)",
        }

        # If we have seasonal columns
        if "summer_occupancy" in self.df.columns:
            analysis["has_seasonal_data"] = True
            analysis["summer_avg_occupancy"] = self.df["summer_occupancy"].mean()
            analysis["winter_avg_occupancy"] = self.df["winter_occupancy"].mean()
            analysis["seasonal_variation"] = (
                analysis["summer_avg_occupancy"] - analysis["winter_avg_occupancy"]
            )

        return analysis

    def get_competitive_position(
        self, price: float, neighborhood: str, room_type: str
    ) -> Dict[str, any]:
        """
        Determine competitive position for a listing.

        Args:
            price: Listing price
            neighborhood: Neighborhood name
            room_type: Type of room

        Returns:
            Dictionary with competitive positioning
        """
        logger.info(f"Analyzing competitive position: {neighborhood}, {room_type}")

        # Filter to comparable listings
        comparable = self.df[
            (self.df["neighbourhood"] == neighborhood)
            & (self.df["room_type"] == room_type)
        ]

        if comparable.empty:
            return {"error": "No comparable listings found"}

        comp_prices = comparable["price"]

        percentile = stats.percentileofscore(comp_prices, price)

        position = {
            "neighborhood": neighborhood,
            "room_type": room_type,
            "your_price": price,
            "market_avg": comp_prices.mean(),
            "market_median": comp_prices.median(),
            "price_percentile": percentile,
            "comparable_count": len(comparable),
            "price_position": self._get_price_position(percentile),
            "recommendation": self._get_price_recommendation(price, comp_prices),
        }

        return position

    def _get_price_position(self, percentile: float) -> str:
        """Categorize price position based on percentile."""
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

        if price < median * 0.8:
            return (
                f"Consider increasing price. You're 20%+ below median (${median:.0f})"
            )
        elif price > median * 1.5:
            return f"Price is premium. Ensure quality justifies ${price-median:.0f} above median"
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
╔══════════════════════════════════════════════════════════════╗
║           SEATTLE AIRBNB MARKET ANALYSIS REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║ MARKET OVERVIEW                                              ║
║   Total Listings: {insights.total_listings:>40,}║
║   Total Hosts: {insights.total_hosts:>44,}║
║   Average Price: ${insights.avg_price:>41,.2f}║
║   Median Price: ${insights.median_price:>42,.2f}║
╠══════════════════════════════════════════════════════════════╣
║ KEY INSIGHTS                                                 ║
║   Top Neighborhood: {insights.top_neighborhood:>38}║
║   Dominant Room Type: {insights.dominant_room_type:>36}║
║   Avg Occupancy Rate: {insights.avg_occupancy*100:>34.1f}%║
║   Seasonal Premium: {insights.seasonal_premium:>37.1f}%║
╠══════════════════════════════════════════════════════════════╣
║ PRICE RANGE                                                  ║
║   Minimum: ${insights.price_range[0]:>47,.2f}║
║   Maximum: ${insights.price_range[1]:>47,.2f}║
╚══════════════════════════════════════════════════════════════╝
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
    print("\n🏆 Top 10 Neighborhoods:")
    print(analyzer.analyze_neighborhoods(10))


if __name__ == "__main__":
    main()

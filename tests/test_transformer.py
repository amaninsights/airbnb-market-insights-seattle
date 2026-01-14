"""
Test Data Transformer
=====================

Unit tests for AirbnbTransformer class.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.transformer import AirbnbTransformer


class TestAirbnbTransformerCleaning:
    """Tests for data cleaning."""

    def test_clean_listings_removes_zero_price(self, sample_listings):
        """Test that zero prices are removed."""
        df = sample_listings.copy()
        df.loc[0, "price"] = 0

        transformer = AirbnbTransformer(df)
        result = transformer.clean_listings(df)

        assert len(result) == len(sample_listings) - 1
        assert 0 not in result["price"].values

    def test_clean_listings_handles_missing_neighbourhood(self, sample_listings):
        """Test handling of missing neighbourhoods."""
        df = sample_listings.copy()
        df.loc[0, "neighbourhood"] = None

        transformer = AirbnbTransformer(df)
        result = transformer.clean_listings(df)

        assert not result["neighbourhood"].isna().any()
        assert "Unknown" in result["neighbourhood"].values

    def test_room_type_standardization(self, sample_listings):
        """Test room type names are standardized."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.clean_listings(sample_listings)

        # Original 'Entire home/apt' should become 'Entire Home'
        assert "Entire Home" in result["room_type"].values


class TestAirbnbTransformerPriceFeatures:
    """Tests for price feature engineering."""

    def test_price_category_creation(self, sample_listings):
        """Test price categories are created correctly."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.add_price_features(sample_listings)

        assert "price_category" in result.columns
        assert set(result["price_category"].unique()).issubset(
            {"Budget", "Mid-Range", "Premium", "Luxury"}
        )

    def test_log_price_calculation(self, sample_listings):
        """Test log price is calculated."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.add_price_features(sample_listings)

        assert "log_price" in result.columns
        assert (result["log_price"] >= 0).all()

    def test_price_zscore_calculation(self, sample_listings):
        """Test price z-score is calculated per neighborhood."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.add_price_features(sample_listings)

        assert "price_zscore" in result.columns


class TestAirbnbTransformerNeighborhoodRankings:
    """Tests for neighborhood rankings."""

    def test_neighborhood_stats_added(self, sample_listings):
        """Test neighborhood statistics are added."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.add_neighborhood_rankings(sample_listings)

        assert "neighborhood_avg_price" in result.columns
        assert "neighborhood_listing_count" in result.columns
        assert "neighborhood_rank" in result.columns

    def test_price_vs_neighborhood_calculation(self, sample_listings):
        """Test relative price calculation."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.add_neighborhood_rankings(sample_listings)

        assert "price_vs_neighborhood" in result.columns


class TestAirbnbTransformerSummaries:
    """Tests for summary creation."""

    def test_neighborhood_summary(self, sample_listings):
        """Test neighborhood summary creation."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.create_neighborhood_summary()

        assert not result.empty
        assert "neighbourhood" in result.columns
        assert "avg_price" in result.columns

    def test_room_type_summary(self, sample_listings):
        """Test room type summary creation."""
        transformer = AirbnbTransformer(sample_listings)
        result = transformer.create_room_type_summary()

        assert not result.empty
        assert "room_type" in result.columns
        assert "market_share" in result.columns


class TestAirbnbTransformerOutlierRemoval:
    """Tests for outlier removal."""

    def test_remove_outliers(self, sample_listings):
        """Test outlier removal."""
        df = sample_listings.copy()
        df.loc[0, "price"] = 10000  # Add outlier

        transformer = AirbnbTransformer(df)
        result = transformer.remove_outliers(df, "price", n_std=2)

        assert len(result) < len(df)
        assert 10000 not in result["price"].values

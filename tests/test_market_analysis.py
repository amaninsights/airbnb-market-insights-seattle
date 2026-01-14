"""
Test Market Analysis
====================

Unit tests for MarketAnalyzer class.
"""

import pandas as pd
import pytest

from src.analysis.market_analysis import MarketAnalyzer, MarketInsights


class TestMarketAnalyzerInsights:
    """Tests for market insights generation."""

    def test_get_market_insights(self, sample_listings):
        """Test market insights generation."""
        analyzer = MarketAnalyzer(sample_listings)
        insights = analyzer.get_market_insights()

        assert isinstance(insights, MarketInsights)
        assert insights.total_listings == len(sample_listings)
        assert insights.avg_price > 0

    def test_insights_top_neighborhood(self, sample_listings):
        """Test top neighborhood identification."""
        analyzer = MarketAnalyzer(sample_listings)
        insights = analyzer.get_market_insights()

        # West Seattle has highest price ($300) in sample data
        assert insights.top_neighborhood == "West Seattle"


class TestMarketAnalyzerPricing:
    """Tests for pricing analysis."""

    def test_analyze_pricing(self, sample_listings):
        """Test pricing analysis."""
        analyzer = MarketAnalyzer(sample_listings)
        analysis = analyzer.analyze_pricing()

        assert "mean" in analysis
        assert "median" in analysis
        assert "percentiles" in analysis

    def test_pricing_by_room_type(self, sample_listings):
        """Test pricing breakdown by room type."""
        analyzer = MarketAnalyzer(sample_listings)
        analysis = analyzer.analyze_pricing()

        assert "by_room_type" in analysis


class TestMarketAnalyzerNeighborhoods:
    """Tests for neighborhood analysis."""

    def test_analyze_neighborhoods(self, sample_listings):
        """Test neighborhood analysis."""
        analyzer = MarketAnalyzer(sample_listings)
        result = analyzer.analyze_neighborhoods(top_n=5)

        assert isinstance(result, pd.DataFrame)
        assert "neighbourhood" in result.columns
        assert "rank" in result.columns

    def test_neighborhood_ranking(self, sample_listings):
        """Test neighborhoods are ranked correctly."""
        analyzer = MarketAnalyzer(sample_listings)
        result = analyzer.analyze_neighborhoods(top_n=10)

        # Rankings should be sequential
        assert result["rank"].min() == 1


class TestMarketAnalyzerCompetitive:
    """Tests for competitive analysis."""

    def test_competitive_position(self, sample_listings):
        """Test competitive position analysis."""
        analyzer = MarketAnalyzer(sample_listings)

        position = analyzer.get_competitive_position(
            price=160, neighborhood="Capitol Hill", room_type="Entire Home"
        )

        assert "price_percentile" in position
        assert "recommendation" in position

    def test_competitive_position_no_comparables(self, sample_listings):
        """Test handling when no comparable listings found."""
        analyzer = MarketAnalyzer(sample_listings)

        position = analyzer.get_competitive_position(
            price=100, neighborhood="NonExistent", room_type="Unknown"
        )

        assert "error" in position


class TestMarketAnalyzerRevenue:
    """Tests for revenue estimation."""

    def test_estimate_revenue(self, sample_listings):
        """Test revenue estimation."""
        analyzer = MarketAnalyzer(sample_listings)

        revenue = analyzer.estimate_revenue(price=150, occupancy_rate=0.7)

        assert "daily_revenue" in revenue
        assert "monthly_revenue" in revenue
        assert "annual_revenue" in revenue
        assert revenue["daily_revenue"] == 150 * 0.7


class TestMarketAnalyzerReport:
    """Tests for report generation."""

    def test_generate_report(self, sample_listings):
        """Test report generation."""
        analyzer = MarketAnalyzer(sample_listings)
        report = analyzer.generate_report()

        assert isinstance(report, str)
        assert "SEATTLE AIRBNB" in report
        assert "Total Listings" in report

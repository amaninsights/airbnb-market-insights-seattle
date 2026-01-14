"""
Test Configuration
==================

Pytest fixtures for Airbnb market analysis tests.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_listings():
    """Create sample listings DataFrame."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": [
                "Cozy Studio",
                "Downtown Loft",
                "Family Home",
                "Beach House",
                "Urban Apt",
            ],
            "host_id": [101, 102, 103, 104, 105],
            "host_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "neighbourhood": [
                "Capitol Hill",
                "Downtown",
                "Ballard",
                "West Seattle",
                "Capitol Hill",
            ],
            "latitude": [47.6253, 47.6062, 47.6677, 47.5660, 47.6200],
            "longitude": [-122.3222, -122.3321, -122.3854, -122.3868, -122.3190],
            "room_type": [
                "Entire home/apt",
                "Private room",
                "Entire home/apt",
                "Entire home/apt",
                "Private room",
            ],
            "price": [150, 75, 200, 300, 85],
            "minimum_nights": [2, 1, 3, 2, 1],
            "number_of_reviews": [45, 120, 30, 15, 200],
            "availability_365": [200, 300, 150, 100, 350],
        }
    )


@pytest.fixture
def sample_calendar():
    """Create sample calendar DataFrame."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "listing_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "date": list(dates),
            "available": [
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
            ],
            "price": [150, 150, 150, 150, 150, 75, 75, 75, 75, 75],
        }
    )


@pytest.fixture
def sample_reviews():
    """Create sample reviews DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="W")
    return pd.DataFrame(
        {
            "listing_id": [1, 1, 2, 2, 3],
            "id": [1001, 1002, 1003, 1004, 1005],
            "date": list(dates),
            "reviewer_id": [201, 202, 203, 204, 205],
            "reviewer_name": ["John", "Jane", "Mike", "Sarah", "Tom"],
            "comments": ["Great stay!", "Nice place", "Loved it", "Clean", "Perfect"],
        }
    )

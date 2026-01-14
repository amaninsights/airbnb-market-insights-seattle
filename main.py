#!/usr/bin/env python
"""
Airbnb Market Insights - Seattle
================================

End-to-end analytics pipeline for Seattle Airbnb market analysis.

Usage:
    python main.py                    # Run full pipeline
    python main.py --analyze-only     # Skip data loading
    python main.py --export-charts    # Generate all visualizations
    python main.py --top 20           # Show top 20 neighborhoods

Author: Aman Saroha
Version: 2.0.0
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.market_analysis import MarketAnalyzer  # noqa: E402
from src.data.loader import AirbnbDataLoader  # noqa: E402
from src.data.transformer import AirbnbTransformer  # noqa: E402
from src.utils.logger import PipelineLogger, setup_logger  # noqa: E402
from src.visualization.charts import AirbnbCharts  # noqa: E402

logger = setup_logger(__name__, log_file="logs/pipeline.log")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seattle Airbnb Market Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    Run full analysis
    python main.py --analyze-only     Skip data loading
    python main.py --export-charts    Generate all charts
    python main.py --top 20           Show top 20 neighborhoods
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip data loading, use cached data",
    )

    parser.add_argument(
        "--export-charts",
        action="store_true",
        help="Generate and save all visualizations",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top neighborhoods to display",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory for output files",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def run_pipeline(args):
    """Execute the main analysis pipeline."""
    pipeline = PipelineLogger("Airbnb Market Analysis")
    pipeline.start()

    print("\n" + "=" * 60)
    print("AIRBNB MARKET INSIGHTS - SEATTLE")
    print("=" * 60)
    print("Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Step 1: Load Data
    pipeline.step("Loading Data")
    loader = AirbnbDataLoader(args.config)

    listings = loader.load_listings()
    calendar = loader.load_calendar()
    reviews = loader.load_reviews()

    if listings.empty:
        print("\nNo data found. Please download the Kaggle dataset:")
        print("   https://www.kaggle.com/datasets/airbnb/seattle")
        print("   Place CSV files in data/raw/")
        return

    print("\nData Loaded:")
    print("   Listings: {:,}".format(len(listings)))
    print("   Calendar: {:,}".format(len(calendar)))
    print("   Reviews: {:,}".format(len(reviews)))

    # Step 2: Transform Data
    pipeline.step("Transforming Data")
    transformer = AirbnbTransformer(listings, calendar, reviews)
    df = transformer.transform()

    print("\nTransformation Complete:")
    print("   Cleaned Listings: {:,}".format(len(df)))
    print("   Features Added: {}".format(len(df.columns)))

    # Step 3: Analyze Market
    pipeline.step("Analyzing Market")
    analyzer = MarketAnalyzer(df)
    insights = analyzer.get_market_insights()

    print("\nMarket Insights:")
    print("   Total Listings: {:,}".format(insights.total_listings))
    print("   Average Price: ${:.2f}".format(insights.avg_price))
    print("   Median Price: ${:.2f}".format(insights.median_price))
    print("   Top Neighborhood: {}".format(insights.top_neighborhood))
    print("   Most Common Room: {}".format(insights.most_common_room_type))

    # Step 4: Top Neighborhoods
    pipeline.step("Ranking Neighborhoods")
    top_neighborhoods = analyzer.analyze_neighborhoods(top_n=args.top)

    print("\nTop {} Neighborhoods by Avg Price:".format(args.top))
    print("-" * 50)
    for _, row in top_neighborhoods.head(args.top).iterrows():
        rank = int(row["rank"])
        name = row["neighbourhood"]
        price = row["avg_price"]
        count = int(row["listing_count"])
        print("   {}. {}: ${:.0f}/night ({} listings)".format(rank, name, price, count))

    # Step 5: Export Charts
    if args.export_charts:
        pipeline.step("Generating Charts")
        charts = AirbnbCharts(df)
        saved = charts.save_all_charts()

        print("\nCharts Generated:")
        for name, path in saved.items():
            print("   {}: {}".format(name, path))

    # Summary
    pipeline.complete()
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return df, analyzer, insights


def main():
    """Main entry point."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

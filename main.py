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

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import AirbnbDataLoader
from src.data.transformer import AirbnbTransformer
from src.analysis.market_analysis import MarketAnalyzer
from src.visualization.charts import AirbnbCharts
from src.utils.logger import setup_logger, PipelineLogger

logger = setup_logger(__name__, log_file="logs/pipeline.log")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seattle Airbnb Market Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip data loading, use existing data",
    )
    parser.add_argument(
        "--export-charts", action="store_true", help="Generate and export all charts"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top neighborhoods to display"
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for reports"
    )

    return parser.parse_args()


def run_pipeline(args):
    """Run the complete analysis pipeline."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("SEATTLE AIRBNB MARKET ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load Data
    with PipelineLogger("Data Loading") as pl:
        loader = AirbnbDataLoader(args.config)
        listings = loader.load_listings()
        calendar = loader.load_calendar()
        reviews = loader.load_reviews()

        pl.info(f"Listings: {len(listings):,}")
        pl.info(f"Calendar: {len(calendar):,}")
        pl.info(f"Reviews: {len(reviews):,}")

    if listings.empty:
        logger.warning("No listings data found. Please add CSV files to data/raw/")
        print("\n⚠️  No data found. Download the Kaggle dataset:")
        print("   https://www.kaggle.com/datasets/airbnb/seattle")
        print("   Place CSV files in: data/raw/")
        return 1

    # Step 2: Transform Data
    with PipelineLogger("Data Transformation") as pl:
        transformer = AirbnbTransformer(listings, calendar, reviews)
        df = transformer.transform()
        pl.info(f"Transformed: {len(df)} listings, {len(df.columns)} features")

    # Step 3: Analyze Market
    with PipelineLogger("Market Analysis") as pl:
        analyzer = MarketAnalyzer(df)
        insights = analyzer.get_market_insights()
        pl.info(f"Total listings: {insights.total_listings:,}")
        pl.info(f"Average price: ${insights.avg_price:.2f}")

    # Step 4: Generate Visualizations
    if args.export_charts:
        with PipelineLogger("Visualization") as pl:
            charts = AirbnbCharts(df)
            saved = charts.save_all_charts(f"{args.output_dir}/figures")
            pl.info(f"Generated {len(saved)} charts")

    # Step 5: Export Results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save neighborhood analysis
    neighborhood_df = analyzer.analyze_neighborhoods(args.top)
    neighborhood_df.to_csv(output_path / "neighborhood_analysis.csv", index=False)

    # Print Summary
    print(analyzer.generate_report())

    print(f"\n🏆 Top {args.top} Neighborhoods by Revenue Potential:")
    print("-" * 60)
    for i, row in neighborhood_df.head(args.top).iterrows():
        print(
            f"  {int(row['rank']):2}. {row['neighbourhood']:<30} "
            f"Avg: ${row['avg_price']:.0f}"
        )
    print("-" * 60)

    duration = datetime.now() - start_time
    logger.info(f"Pipeline complete. Duration: {duration}")
    print(f"\n✅ Analysis complete! Check '{args.output_dir}/' for outputs.")

    return 0


def main():
    """Main entry point."""
    args = parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

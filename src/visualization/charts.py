"""
Airbnb Charts Module
====================

Provides publication-ready visualizations for Seattle Airbnb market analysis.
Includes maps, price distributions, neighborhood comparisons, and dashboards.

Example Usage:
    >>> from src.visualization.charts import AirbnbCharts
    >>> charts = AirbnbCharts(listings_df)
    >>> charts.plot_price_map()
    >>> charts.save_all_charts("reports/figures/")
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AirbnbCharts:
    """
    Publication-ready chart generator for Airbnb analysis.

    Provides:
    - Price distribution plots
    - Neighborhood comparison charts
    - Room type analysis
    - Geographic price maps
    - Seasonal trend visualizations
    """

    def __init__(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8),
        style: str = "seaborn-v0_8-whitegrid",
        dpi: int = 150,
    ):
        """
        Initialize chart generator.

        Args:
            df: Listings DataFrame
            figsize: Default figure size
            style: Matplotlib style
            dpi: DPI for saved figures
        """
        self.df = df.copy()
        self.figsize = figsize
        self.dpi = dpi

        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = dpi

        logger.info(f"AirbnbCharts initialized with {len(df)} listings")

    def plot_price_distribution(
        self, ax: Optional[plt.Axes] = None, max_price: int = 500
    ) -> plt.Figure:
        """
        Plot price distribution histogram with KDE.

        Args:
            ax: Optional axes to plot on
            max_price: Maximum price to show

        Returns:
            Figure object
        """
        if "price" not in self.df.columns:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        data = self.df[self.df["price"] <= max_price]["price"]

        sns.histplot(data, bins=50, kde=True, ax=ax, color="steelblue", alpha=0.7)

        ax.axvline(
            data.mean(), color="red", linestyle="--", label=f"Mean: ${data.mean():.0f}"
        )
        ax.axvline(
            data.median(),
            color="green",
            linestyle="--",
            label=f"Median: ${data.median():.0f}",
        )

        ax.set_xlabel("Price ($)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "Seattle Airbnb Price Distribution", fontsize=14, fontweight="bold"
        )
        ax.legend()

        return fig

    def plot_neighborhood_prices(
        self, top_n: int = 15, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot average price by neighborhood.

        Args:
            top_n: Number of neighborhoods to show
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        if "neighbourhood" not in self.df.columns or "price" not in self.df.columns:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        neighborhood_prices = (
            self.df.groupby("neighbourhood")["price"]
            .mean()
            .sort_values(ascending=True)
            .tail(top_n)
        )

        colors = sns.color_palette("viridis", len(neighborhood_prices))
        bars = ax.barh(
            neighborhood_prices.index, neighborhood_prices.values, color=colors
        )

        for bar, price in zip(bars, neighborhood_prices.values):
            ax.text(
                bar.get_width() + 2,
                bar.get_y() + bar.get_height() / 2,
                f"${price:.0f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Average Price ($)", fontsize=12)
        ax.set_title(
            f"Top {top_n} Seattle Neighborhoods by Average Price",
            fontsize=14,
            fontweight="bold",
        )

        return fig

    def plot_room_type_analysis(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot room type distribution and pricing.

        Args:
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        if "room_type" not in self.df.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart of room type distribution
        room_counts = self.df["room_type"].value_counts()
        colors = sns.color_palette("Set2", len(room_counts))

        axes[0].pie(
            room_counts.values,
            labels=room_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            explode=[0.05] * len(room_counts),
        )
        axes[0].set_title("Room Type Distribution", fontsize=14, fontweight="bold")

        # Box plot of price by room type
        if "price" in self.df.columns:
            data = self.df[self.df["price"] <= 500]
            sns.boxplot(data=data, x="room_type", y="price", palette="Set2", ax=axes[1])
            axes[1].set_xlabel("Room Type", fontsize=12)
            axes[1].set_ylabel("Price ($)", fontsize=12)
            axes[1].set_title("Price by Room Type", fontsize=14, fontweight="bold")
            axes[1].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        return fig

    def plot_price_map(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot geographic distribution of prices.

        Args:
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        required = ["latitude", "longitude", "price"]
        if not all(col in self.df.columns for col in required):
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.figure

        data = self.df[self.df["price"] <= 500]

        scatter = ax.scatter(
            data["longitude"],
            data["latitude"],
            c=data["price"],
            cmap="YlOrRd",
            alpha=0.6,
            s=20,
        )

        plt.colorbar(scatter, ax=ax, label="Price ($)")
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.set_title("Seattle Airbnb Price Map", fontsize=14, fontweight="bold")

        return fig

    def plot_availability_distribution(
        self, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot availability distribution.

        Args:
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        if "availability_365" not in self.df.columns:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        sns.histplot(
            self.df["availability_365"], bins=50, kde=True, ax=ax, color="teal"
        )

        ax.set_xlabel("Days Available (per year)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "Listing Availability Distribution", fontsize=14, fontweight="bold"
        )

        return fig

    def plot_reviews_vs_price(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot relationship between reviews and price.

        Args:
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        if "number_of_reviews" not in self.df.columns or "price" not in self.df.columns:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        data = self.df[
            (self.df["price"] <= 500) & (self.df["number_of_reviews"] <= 200)
        ]

        sns.scatterplot(
            data=data,
            x="number_of_reviews",
            y="price",
            hue="room_type" if "room_type" in data.columns else None,
            alpha=0.5,
            ax=ax,
        )

        ax.set_xlabel("Number of Reviews", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.set_title("Reviews vs Price", fontsize=14, fontweight="bold")

        return fig

    def plot_price_by_category(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot price distribution by category.

        Args:
            ax: Optional axes to plot on

        Returns:
            Figure object
        """
        if "price_category" not in self.df.columns:
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        order = ["Budget", "Mid-Range", "Premium", "Luxury"]
        order = [o for o in order if o in self.df["price_category"].unique()]

        sns.countplot(
            data=self.df, x="price_category", order=order, palette="viridis", ax=ax
        )

        ax.set_xlabel("Price Category", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Listings by Price Category", fontsize=14, fontweight="bold")

        return fig

    def create_dashboard(self) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple charts.

        Returns:
            Figure object with dashboard
        """
        fig = plt.figure(figsize=(20, 16))

        # Layout: 3x2 grid
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)

        # Generate charts
        self.plot_price_distribution(ax=ax1)
        self.plot_neighborhood_prices(top_n=10, ax=ax2)
        self.plot_price_map(ax=ax3)
        self.plot_availability_distribution(ax=ax4)
        self.plot_reviews_vs_price(ax=ax5)
        self.plot_price_by_category(ax=ax6)

        fig.suptitle(
            "Seattle Airbnb Market Dashboard",
            fontsize=20,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        return fig

    def save_all_charts(self, output_dir: str = "reports/figures") -> Dict[str, str]:
        """
        Generate and save all charts.

        Args:
            output_dir: Directory to save charts

        Returns:
            Dictionary mapping chart names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Generate each chart
        charts = [
            ("price_distribution", self.plot_price_distribution),
            ("neighborhood_prices", self.plot_neighborhood_prices),
            ("room_type_analysis", self.plot_room_type_analysis),
            ("price_map", self.plot_price_map),
            ("availability", self.plot_availability_distribution),
            ("reviews_vs_price", self.plot_reviews_vs_price),
            ("price_categories", self.plot_price_by_category),
            ("dashboard", self.create_dashboard),
        ]

        for name, chart_func in charts:
            try:
                fig = chart_func()
                if fig is not None:
                    filepath = output_path / f"{name}.png"
                    fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
                    plt.close(fig)
                    saved[name] = str(filepath)
                    logger.info(f"Saved: {filepath}")
            except Exception as e:
                logger.error(f"Failed to generate {name}: {e}")

        return saved


def main():
    """Example usage of AirbnbCharts."""
    from src.data.loader import AirbnbDataLoader
    from src.data.transformer import AirbnbTransformer

    # Load and transform
    loader = AirbnbDataLoader()
    listings = loader.load_listings()

    if listings.empty:
        print("No data available for visualization")
        return

    transformer = AirbnbTransformer(listings)
    df = transformer.transform()

    # Generate charts
    charts = AirbnbCharts(df)
    saved = charts.save_all_charts()

    print("\n Charts Generated:")
    for name, path in saved.items():
        print(f"   {name}: {path}")


if __name__ == "__main__":
    main()

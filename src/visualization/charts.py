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

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AirbnbCharts:
    """
    Publication-ready Airbnb market visualizations.
    
    Provides:
    - Geographic price heatmaps
    - Neighborhood comparisons
    - Price distributions
    - Room type analysis
    - Seasonal trends
    - Executive dashboards
    
    Attributes:
        df (pd.DataFrame): Listings data
        style (str): Matplotlib style
        figsize (tuple): Default figure size
        dpi (int): Figure DPI for exports
    """
    
    COLORS = {
        'airbnb_red': '#FF5A5F',
        'airbnb_dark': '#484848',
        'airbnb_teal': '#00A699',
        'airbnb_orange': '#FC642D',
        'airbnb_pink': '#E31C5F',
        'gradient': ['#FF5A5F', '#FF8C8D', '#FFB3B3', '#FFD9D9']
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300
    ):
        """
        Initialize AirbnbCharts with data.
        
        Args:
            df: Listings DataFrame
            style: Matplotlib style name
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.df = df.copy()
        self.figsize = figsize
        self.dpi = dpi
        
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
            
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = dpi
        
        logger.info(f"AirbnbCharts initialized with {len(df)} listings")
        
    def plot_price_distribution(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create histogram of price distribution with KDE.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        if 'price' not in self.df.columns:
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Filter extreme outliers for better visualization
        prices = self.df['price']
        prices = prices[prices < prices.quantile(0.95)]
        
        ax.hist(prices, bins=50, color=self.COLORS['airbnb_red'], 
                alpha=0.7, edgecolor='white')
        
        # Add median line
        median = prices.median()
        ax.axvline(median, color=self.COLORS['airbnb_dark'], 
                   linestyle='--', linewidth=2, label=f'Median: ${median:.0f}')
        
        ax.set_xlabel('Price per Night ($)', fontsize=12)
        ax.set_ylabel('Number of Listings', fontsize=12)
        ax.set_title('Distribution of Airbnb Prices in Seattle', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        logger.info("Created price distribution chart")
        return fig
        
    def plot_neighborhood_prices(
        self,
        top_n: int = 15,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Create horizontal bar chart of average prices by neighborhood.
        
        Args:
            top_n: Number of neighborhoods to show
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        if 'neighbourhood' not in self.df.columns or 'price' not in self.df.columns:
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = ax.get_figure()
            
        # Calculate neighborhood averages
        neighborhood_prices = self.df.groupby('neighbourhood')['price'].agg(['mean', 'count'])
        neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 10]  # Min 10 listings
        neighborhood_prices = neighborhood_prices.nlargest(top_n, 'mean')
        
        # Create color gradient
        colors = sns.color_palette('Reds_r', len(neighborhood_prices))
        
        bars = ax.barh(neighborhood_prices.index, neighborhood_prices['mean'], color=colors)
        
        # Add value labels
        for bar, (idx, row) in zip(bars, neighborhood_prices.iterrows()):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                   f'${row["mean"]:.0f} (n={row["count"]:.0f})',
                   va='center', fontsize=9)
                   
        ax.set_xlabel('Average Price per Night ($)', fontsize=12)
        ax.set_ylabel('')
        ax.set_title(f'Top {top_n} Most Expensive Neighborhoods', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        logger.info("Created neighborhood prices chart")
        return fig
        
    def plot_room_type_analysis(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create pie chart of room type distribution with price annotation.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        if 'room_type' not in self.df.columns:
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()
            
        # Calculate room type stats
        room_stats = self.df.groupby('room_type').agg({
            'id': 'count',
            'price': 'mean'
        }).rename(columns={'id': 'count', 'price': 'avg_price'})
        
        colors = [self.COLORS['airbnb_red'], self.COLORS['airbnb_teal'], 
                  self.COLORS['airbnb_orange'], self.COLORS['airbnb_pink']][:len(room_stats)]
        
        # Create labels with price info
        labels = [f"{rt}\n(avg ${row['avg_price']:.0f}/night)" 
                  for rt, row in room_stats.iterrows()]
        
        wedges, texts, autotexts = ax.pie(
            room_stats['count'],
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(room_stats),
            startangle=90
        )
        
        plt.setp(autotexts, size=10, weight='bold')
        ax.set_title('Listing Distribution by Room Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Created room type analysis chart")
        return fig
        
    def plot_price_by_room_type(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create box plot comparing prices across room types.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        if 'room_type' not in self.df.columns or 'price' not in self.df.columns:
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Filter outliers for visualization
        df_filtered = self.df[self.df['price'] < self.df['price'].quantile(0.95)]
        
        # Create box plot
        room_order = df_filtered.groupby('room_type')['price'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df_filtered,
            x='room_type',
            y='price',
            order=room_order,
            palette=[self.COLORS['airbnb_red'], self.COLORS['airbnb_teal'], 
                    self.COLORS['airbnb_orange'], self.COLORS['airbnb_pink']],
            ax=ax
        )
        
        ax.set_xlabel('Room Type', fontsize=12)
        ax.set_ylabel('Price per Night ($)', fontsize=12)
        ax.set_title('Price Distribution by Room Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Created price by room type chart")
        return fig
        
    def plot_availability_heatmap(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create heatmap of availability by neighborhood and room type.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        required_cols = ['neighbourhood', 'room_type', 'availability_365']
        if not all(c in self.df.columns for c in required_cols):
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 10))
        else:
            fig = ax.get_figure()
            
        # Get top neighborhoods
        top_neighborhoods = self.df['neighbourhood'].value_counts().head(15).index
        
        # Pivot table
        pivot = self.df[self.df['neighbourhood'].isin(top_neighborhoods)].pivot_table(
            values='availability_365',
            index='neighbourhood',
            columns='room_type',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot,
            cmap='RdYlGn',
            annot=True,
            fmt='.0f',
            ax=ax,
            cbar_kws={'label': 'Days Available (out of 365)'}
        )
        
        ax.set_title('Average Availability by Neighborhood & Room Type', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Room Type', fontsize=12)
        ax.set_ylabel('Neighborhood', fontsize=12)
        
        plt.tight_layout()
        logger.info("Created availability heatmap")
        return fig
        
    def plot_reviews_vs_price(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create scatter plot of reviews vs price.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        required_cols = ['price', 'number_of_reviews']
        if not all(c in self.df.columns for c in required_cols):
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()
            
        # Filter for visualization
        df_filtered = self.df[
            (self.df['price'] < self.df['price'].quantile(0.95)) &
            (self.df['number_of_reviews'] > 0)
        ]
        
        scatter = ax.scatter(
            df_filtered['number_of_reviews'],
            df_filtered['price'],
            c=df_filtered.get('availability_365', df_filtered['price']),
            cmap='coolwarm',
            alpha=0.5,
            s=30
        )
        
        plt.colorbar(scatter, ax=ax, label='Availability (days)')
        
        ax.set_xlabel('Number of Reviews', fontsize=12)
        ax.set_ylabel('Price per Night ($)', fontsize=12)
        ax.set_title('Price vs Reviews Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Created reviews vs price scatter plot")
        return fig
        
    def plot_price_map(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create geographic scatter plot of listings colored by price.
        
        Args:
            ax: Optional matplotlib axes
            
        Returns:
            Matplotlib Figure
        """
        required_cols = ['latitude', 'longitude', 'price']
        if not all(c in self.df.columns for c in required_cols):
            return plt.figure()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))
        else:
            fig = ax.get_figure()
            
        # Filter outliers
        df_filtered = self.df[self.df['price'] < self.df['price'].quantile(0.95)]
        
        scatter = ax.scatter(
            df_filtered['longitude'],
            df_filtered['latitude'],
            c=df_filtered['price'],
            cmap='RdYlGn_r',
            alpha=0.5,
            s=10
        )
        
        plt.colorbar(scatter, ax=ax, label='Price per Night ($)')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Seattle Airbnb Listings - Price Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        logger.info("Created geographic price map")
        return fig
        
    def plot_dashboard(self) -> plt.Figure:
        """
        Create comprehensive 6-panel dashboard.
        
        Returns:
            Matplotlib Figure with dashboard
        """
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Seattle Airbnb Market Dashboard', fontsize=18, fontweight='bold', y=1.02)
        
        # Layout: 3 rows, 2 columns
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Price Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_price_distribution(ax=ax1)
        
        # Panel 2: Room Type Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_room_type_analysis(ax=ax2)
        
        # Panel 3: Neighborhood Prices
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_neighborhood_prices(top_n=10, ax=ax3)
        
        # Panel 4: Price by Room Type
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_price_by_room_type(ax=ax4)
        
        # Panel 5: Geographic Map
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_price_map(ax=ax5)
        
        # Panel 6: Reviews vs Price
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_reviews_vs_price(ax=ax6)
        
        plt.tight_layout()
        logger.info("Created market dashboard")
        return fig
        
    def save_chart(
        self,
        fig: plt.Figure,
        filename: str,
        output_dir: str = 'reports/figures'
    ) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Output filename
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / f"{filename}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        logger.info(f"Chart saved: {filepath}")
        return str(filepath)
        
    def save_all_charts(self, output_dir: str = 'reports/figures') -> Dict[str, str]:
        """
        Generate and save all charts.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        saved = {}
        
        charts = [
            ('price_distribution', self.plot_price_distribution),
            ('neighborhood_prices', self.plot_neighborhood_prices),
            ('room_type_analysis', self.plot_room_type_analysis),
            ('price_by_room_type', self.plot_price_by_room_type),
            ('availability_heatmap', self.plot_availability_heatmap),
            ('reviews_vs_price', self.plot_reviews_vs_price),
            ('price_map', self.plot_price_map),
            ('dashboard', self.plot_dashboard)
        ]
        
        for name, method in charts:
            try:
                fig = method()
                if fig.axes:  # Only save if figure has content
                    saved[name] = self.save_chart(fig, name, output_dir)
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating {name}: {e}")
                
        logger.info(f"Saved {len(saved)} charts")
        return saved


def main():
    """Example usage."""
    from src.data.loader import AirbnbDataLoader
    from src.data.transformer import AirbnbTransformer
    
    loader = AirbnbDataLoader()
    listings = loader.load_listings()
    
    transformer = AirbnbTransformer(listings)
    df = transformer.transform()
    
    charts = AirbnbCharts(df)
    saved = charts.save_all_charts()
    
    print("\n📊 Charts Generated:")
    for name, path in saved.items():
        print(f"  ✓ {name}: {path}")


if __name__ == "__main__":
    main()

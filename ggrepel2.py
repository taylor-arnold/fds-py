"""
Custom plotnine geometry: geom_taylor

A geometry that extends geom_point by taking a random sample of the data
before plotting. This is useful for large datasets where you want to
visualize a representative sample without overplotting.

This demonstrates how to create custom geometries in plotnine by:
1. Subclassing an existing geom
2. Overriding the setup_data method to modify data before plotting
3. Adding custom parameters
"""

import numpy as np
import pandas as pd
from plotnine import geom_point
from plotnine.doctools import document


class geom_taylor(geom_point):
    """
    A point geometry that randomly samples data before plotting.
    
    This geom inherits from geom_point and adds sampling functionality.
    Named after Taylor series - taking a sample to approximate the whole!
    
    Parameters
    ----------
    n : int, optional
        Number of points to sample. If None, uses `frac` instead.
        Default is None.
    frac : float, optional
        Fraction of points to sample (0 < frac <= 1).
        Only used if `n` is None. Default is 0.1 (10%).
    random_state : int, optional
        Seed for reproducibility. Default is None (random each time).
    replace : bool, optional
        Whether to sample with replacement. Default is False.
    
    All other parameters are passed to geom_point.
    
    Examples
    --------
    ```python
    from plotnine import ggplot, aes
    from plotnine.data import mpg
    
    # Sample 50 points
    (ggplot(mpg, aes('displ', 'hwy'))
     + geom_taylor(n=50, random_state=42))
    
    # Sample 20% of the data
    (ggplot(mpg, aes('displ', 'hwy'))
     + geom_taylor(frac=0.2, random_state=42))
    ```
    """
    
    # Define the default aesthetic values for this geom
    # We inherit all defaults from geom_point
    DEFAULT_AES = geom_point.DEFAULT_AES.copy()
    
    # Required aesthetics (inherited from geom_point)
    REQUIRED_AES = geom_point.REQUIRED_AES
    
    # Non-aesthetic parameters specific to this geom
    DEFAULT_PARAMS = geom_point.DEFAULT_PARAMS.copy()
    DEFAULT_PARAMS.update({
        'n': None,           # Number of samples (takes precedence over frac)
        'frac': 0.1,         # Fraction to sample if n is None
        'random_state': None, # Random seed for reproducibility
        'replace': False      # Sample with replacement?
    })
    
    def setup_data(self, data):
        """
        Modify the data before the geometry processes it.
        
        This is the key method we override to add sampling functionality.
        It's called after the data has been transformed by stats but before
        the actual drawing happens.
        
        Parameters
        ----------
        data : pandas.DataFrame
            The data to be plotted, with aesthetic mappings applied.
            
        Returns
        -------
        pandas.DataFrame
            The sampled data.
        """
        # First, let the parent class do any setup it needs
        data = super().setup_data(data)
        
        if data is None or len(data) == 0:
            return data
        
        # Get our sampling parameters
        n = self.params.get('n')
        frac = self.params.get('frac', 0.1)
        random_state = self.params.get('random_state')
        replace = self.params.get('replace', False)
        
        # Determine sample size
        if n is not None:
            # Use explicit n, but don't exceed data size (unless replace=True)
            if not replace:
                n = min(n, len(data))
        else:
            # Use fraction
            n = max(1, int(len(data) * frac))
        
        # Sample the data
        sampled_data = data.sample(
            n=n,
            replace=replace,
            random_state=random_state
        ).reset_index(drop=True)
        
        return sampled_data


# Alternative implementation using @staticmethod for setup_data
# This shows another pattern you might see in plotnine source code

class geom_taylor_alt(geom_point):
    """
    Alternative implementation showing how to use static method pattern.
    """
    
    DEFAULT_PARAMS = geom_point.DEFAULT_PARAMS.copy()
    DEFAULT_PARAMS.update({
        'n': None,
        'frac': 0.1,
        'random_state': None,
        'replace': False
    })
    
    @staticmethod
    def setup_data(data, params):
        """
        Static method version - params are passed explicitly.
        
        Note: The method signature differs from the instance method version.
        plotnine will detect which pattern you're using.
        """
        if data is None or len(data) == 0:
            return data
            
        n = params.get('n')
        frac = params.get('frac', 0.1)
        random_state = params.get('random_state')
        replace = params.get('replace', False)
        
        if n is not None:
            if not replace:
                n = min(n, len(data))
        else:
            n = max(1, int(len(data) * frac))
        
        return data.sample(
            n=n,
            replace=replace,
            random_state=random_state
        ).reset_index(drop=True)


# =============================================================================
# Demo / Testing Code
# =============================================================================

if __name__ == "__main__":
    from plotnine import ggplot, aes, labs, theme_minimal, facet_wrap
    from plotnine.data import mpg, diamonds
    
    print("=" * 60)
    print("geom_taylor Demo - Custom Sampling Geometry for plotnine")
    print("=" * 60)
    
    # Demo 1: Basic usage with mpg dataset
    print("\nDemo 1: Basic sampling (50 points from mpg dataset)")
    p1 = (
        ggplot(mpg, aes('displ', 'hwy', color='class'))
        + geom_taylor(n=50, random_state=42, size=3)
        + labs(
            title='geom_taylor: 50 random points from mpg',
            x='Engine Displacement (L)',
            y='Highway MPG'
        )
        + theme_minimal()
    )
    p1.save('demo_taylor_basic.png', dpi=150, width=8, height=5)
    print("  Saved: demo_taylor_basic.png")
    
    # Demo 2: Comparing full data vs sampled
    print("\nDemo 2: Side-by-side comparison with diamonds dataset")
    
    # Full data (will be slow/cluttered with 50k+ points)
    p2_full = (
        ggplot(diamonds.head(5000), aes('carat', 'price', color='cut'))
        + geom_point(alpha=0.3, size=1)
        + labs(title='Full data (5000 points)', x='Carat', y='Price')
        + theme_minimal()
    )
    p2_full.save('demo_comparison_full.png', dpi=150, width=8, height=5)
    print("  Saved: demo_comparison_full.png")
    
    # Sampled data
    p2_sampled = (
        ggplot(diamonds.head(5000), aes('carat', 'price', color='cut'))
        + geom_taylor(frac=0.1, random_state=123, alpha=0.7, size=2)
        + labs(title='geom_taylor (10% sample = 500 points)', x='Carat', y='Price')
        + theme_minimal()
    )
    p2_sampled.save('demo_comparison_sampled.png', dpi=150, width=8, height=5)
    print("  Saved: demo_comparison_sampled.png")
    
    # Demo 3: Using with facets
    print("\nDemo 3: geom_taylor with faceting")
    p3 = (
        ggplot(mpg, aes('displ', 'hwy'))
        + geom_taylor(n=20, random_state=99, color='steelblue', size=2)
        + facet_wrap('~class', ncol=4)
        + labs(
            title='geom_taylor with facets (20 points per facet)',
            x='Displacement',
            y='Highway MPG'
        )
        + theme_minimal()
    )
    p3.save('demo_taylor_facets.png', dpi=150, width=10, height=6)
    print("  Saved: demo_taylor_facets.png")
    
    # Demo 4: Reproducibility test
    print("\nDemo 4: Testing reproducibility with random_state")
    
    # Same random_state should give same result
    p4a = (
        ggplot(mpg, aes('displ', 'hwy'))
        + geom_taylor(n=10, random_state=42)
    )
    p4b = (
        ggplot(mpg, aes('displ', 'hwy'))
        + geom_taylor(n=10, random_state=42)
    )
    
    # Build the plots to get the data
    built_a = p4a.build()
    built_b = p4b.build()
    
    print(f"  Same random_state produces identical results: "
          f"{built_a['data'][0].equals(built_b['data'][0])}")
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)




import polars as pl
import statsmodels.formula.api as sm


df = pl.read_csv("data/food.csv")



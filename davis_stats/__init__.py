import pandas as pd
from pathlib import Path
from .visualization.trim import trim
from .visualization.boxplot import boxplot
from .visualization.histogram import histogram
from .visualization.scatter import scatter
from .data import (
    ceo_comp)
from .stats.reg import reg

__all__ = [
    'ceo_comp']

from .visualization.trim import trim
from .visualization.boxplot import boxplot
from .visualization.histogram import histogram
from .visualization.scatter import scatter
from .visualization.residuals import residuals

from .reg_modeling.reg import reg

from .reg_assumption_tests.bp_test import bp_test
from .reg_assumption_tests.sw_test import sw_test
from .reg_assumption_tests.dw_test import dw_test
from .reg_assumption_tests.vif_test import vif_test

from .datasets import (
    ceo_comp, netflix_content, olympic_medals, 
    restaurants, world_cup_goals, just_games, 
    nba)

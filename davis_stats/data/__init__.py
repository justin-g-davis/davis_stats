import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent


class DataFrames:
    _ceo_comp = None
    _netflix_content = None
    _olympic_medals = None
    _restaurants = None
    _world_cup_goals = None

    @property
    def ceo_comp(self):
        if self._ceo_comp is None:
            self._ceo_comp = pd.read_excel(DATA_DIR / 'ceo_comp.xlsx')
        return self._ceo_comp

    @property
    def netflix_content(self):
        if self._netflix_content is None:
            self._netflix_content = pd.read_excel(DATA_DIR / 'netflix_content.xlsx')
        return self._netflix_content

    @property
    def olympic_medals(self):
        if self._olympic_medals is None:
            self._olympic_medals = pd.read_excel(DATA_DIR / 'olympic_medals.xlsx')
        return self._olympic_medals

    @property
    def restaurants(self):
        if self._restaurants is None:
            self._restaurants = pd.read_excel(DATA_DIR / 'restaurants.xlsx')
        return self._restaurants

    @property
    def world_cup_goals(self):
        if self._world_cup_goals is None:
            self._world_cup_goals = pd.read_excel(DATA_DIR / 'world_cup_goals.xlsx')
        return self._world_cup_goals


_data = DataFrames()


def ceo_comp():
    return _data.ceo_comp

def netflix_content():
    return _data.netflix_content

def olympic_medals():
    return _data.olympic_medals

def restaurants():
    return _data.restaurants

def world_cup_goals():
    return _data.world_cup_goals
